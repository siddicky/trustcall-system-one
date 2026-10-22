"""The diff and apply steps, on hand-built trees (no model, no transport)."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import datetime as dt
import enum
from decimal import Decimal
from typing import Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import PatchOp, compile_schema, s1_field
from trustcall_system_one._compile import FieldGroup, SchemaPlan, UnionGroup
from trustcall_system_one._diff import (
    MISSING,
    Change,
    DiffResult,
    GroupNode,
    LeafNode,
    UnionNode,
    apply_ops,
    build_doc,
    diff_group,
    is_change,
    jsonable,
    leaves_with_old,
    lookup,
    new_positions,
    ops_for_errors,
    pointer,
    same,
    sort_changes,
    token_subset,
)

# --------------------------------------------------------------------------- #
# Building trees from a plan
# --------------------------------------------------------------------------- #


def _key(item: Any) -> str:
    return item.alias if item.alias is not None else str(item.path[-1])


def group_node(
    group: FieldGroup,
    values: dict[str, Any],
    *,
    key: str = "",
    absent: set[str] = frozenset(),  # type: ignore[assignment]
    handled: set[str] = frozenset(),  # type: ignore[assignment]
    chosen: dict[str, str] | None = None,
) -> GroupNode:
    """A tree whose leaves answer as ``values`` says (by dotted path)."""
    node = GroupNode(key, group, present=group.pretty not in absent)
    for spec in group.specs:
        if spec.engine not in ("clairic", "clef", "const"):
            continue
        node.leaves.append(
            LeafNode(
                key=_key(spec),
                spec=spec,
                value=values.get(spec.pretty),
                handled=spec.pretty in handled,
                identity=spec.identity or str(spec.path[-1]) in group.identity,
                spans=((3, 5),) if spec.pretty in values else (),
            )
        )
    for child in group.children:
        node.children.append(
            group_node(
                child,
                values,
                key=_key(child),
                absent=absent,
                handled=handled,
                chosen=chosen,
            )
        )
    for union in group.unions:
        label = (chosen or {}).get(union.spec.pretty)
        member = next((m for m in union.members if m.label == label), None)
        node.unions.append(
            UnionNode(
                key=_key(union),
                union=union,
                chosen=member,
                member=(
                    group_node(member.group, values, key=_key(union), absent=absent)
                    if member is not None
                    else None
                ),
            )
        )
    return node


def run(
    model: type[BaseModel],
    values: dict[str, Any],
    record: dict[str, Any],
    **kwargs: Any,
) -> DiffResult:
    allow = bool(kwargs.pop("allow_null_removals", False))
    plan = compile_schema(model)
    return diff_group(
        group_node(plan.root, values, **kwargs), record, allow_null_removals=allow
    )


def summary(result: DiffResult) -> list[tuple[str, str, Any]]:
    return [(c.op, c.path, c.value) for c in result.changes]


def rejections(result: DiffResult) -> list[tuple[str | None, str, str]]:
    return [(r.op, r.path, r.reason) for r in result.rejected]


class Mood(enum.Enum):
    GOOD = "good"
    BAD = "bad"


class Flat(BaseModel):
    name: str
    city: Optional[str] = None
    age: Optional[int] = None
    score: Optional[float] = None
    price: Optional[Decimal] = None
    born: Optional[dt.date] = None
    seen: Optional[dt.datetime] = None
    mood: Optional[Mood] = None
    notes: list[str] = Field(default_factory=list)
    scores: list[int] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# add / replace / unchanged / not mentioned
# --------------------------------------------------------------------------- #


def test_an_answer_the_record_lacks_is_an_add() -> None:
    result = run(Flat, {"city": "Berlin"}, {"name": "Dana"})
    assert summary(result) == [("add", "/city", "Berlin")]
    (change,) = result.changes
    assert (change.old, change.field_path, change.fields) == (None, "city", ("city",))
    assert change.spans == ((3, 5),)


def test_a_null_in_the_record_counts_as_lacking() -> None:
    result = run(Flat, {"city": "Berlin"}, {"name": "Dana", "city": None})
    assert summary(result) == [("add", "/city", "Berlin")]


def test_a_different_answer_is_a_replace() -> None:
    result = run(Flat, {"city": "Berlin"}, {"name": "Dana", "city": "Austin"})
    assert summary(result) == [("replace", "/city", "Berlin")]
    assert result.changes[0].old == "Austin"


def test_an_equal_answer_makes_no_operation_and_no_rejection() -> None:
    result = run(Flat, {"city": "Austin"}, {"name": "Dana", "city": "Austin"})
    assert result.changes == []
    assert "/city" not in {r.path for r in result.rejected}


@pytest.mark.parametrize(
    ("field", "answer", "stored"),
    [
        ("city", "Austin", "  Austin "),  # stripped
        ("born", dt.date(2025, 3, 4), "2025-03-04"),  # ISO dates
        (
            "seen",
            dt.datetime(2025, 3, 4, 10, 0, tzinfo=dt.timezone.utc),
            "2025-03-04T10:00:00+00:00",
        ),
        (
            "seen",
            dt.datetime(2025, 3, 4, 10, 0, tzinfo=dt.timezone.utc),
            "2025-03-04T10:00:00Z",
        ),
        ("score", 0.1 + 0.2, 0.3),  # math.isclose
        ("score", 3.0, 3),
        ("age", 41, 41.0),
        ("price", Decimal("12.50"), "12.5"),
        ("price", Decimal("12.50"), 12.5),
        ("mood", Mood.GOOD, "good"),  # the enum's value
    ],
)
def test_equal_after_normalisation_is_no_change(
    field: str, answer: Any, stored: Any
) -> None:
    result = run(Flat, {field: answer}, {"name": "Dana", field: stored})
    assert result.changes == [], (field, answer, stored)


@pytest.mark.parametrize(
    ("field", "answer", "stored"),
    [
        ("born", dt.date(2025, 3, 5), "2025-03-04"),
        ("score", 0.31, 0.3),
        ("price", Decimal("12.51"), "12.5"),
        ("mood", Mood.BAD, "good"),
        ("age", 42, 41),
    ],
)
def test_a_real_difference_is_a_replace(field: str, answer: Any, stored: Any) -> None:
    result = run(Flat, {field: answer}, {"name": "Dana", field: stored})
    assert [(c.op, c.path) for c in result.changes] == [("replace", f"/{field}")]


def test_values_in_operations_are_json_compatible() -> None:
    result = run(
        Flat,
        {"born": dt.date(2025, 3, 4), "price": Decimal("1.50"), "mood": Mood.BAD},
        {"name": "Dana"},
    )
    assert {c.path: c.value for c in result.changes} == {
        "/born": "2025-03-04",
        "/price": "1.50",
        "/mood": "bad",
    }
    assert jsonable(dt.timedelta(hours=1)) == "PT1H"


def test_a_null_answer_is_not_mentioned_and_never_an_operation() -> None:
    result = run(Flat, {}, {"name": "Dana", "city": "Austin"})
    assert result.changes == []
    by_path = {r.path: r for r in result.rejected}
    assert by_path["/city"].op is None and by_path["/city"].reason == "not_mentioned"
    assert {r.reason for r in result.rejected} == {"not_mentioned"}
    assert "/name" in by_path and "/notes" in by_path


def test_a_null_the_ladder_already_explained_adds_no_second_rejection() -> None:
    result = run(Flat, {}, {"name": "Dana", "city": "Austin"}, handled={"city"})
    assert "/city" not in {r.path for r in result.rejected}


def test_a_field_missing_from_a_record_is_not_mentioned_too() -> None:
    result = run(Flat, {}, {"name": "Dana"})
    assert (
        "not_mentioned" in {r.reason for r in result.rejected}
    ) and not result.changes


# --------------------------------------------------------------------------- #
# Identity fields
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("answer", "stored"),
    [
        ("Dana", "Dana Whitfield"),
        ("Whitfield", "Dana Whitfield"),
        ("sarah", "Sarah Johnson"),
        ("Sarah", "Sarah Johnson"),
        ("Dana W", "Dana W Whitfield"),
    ],
)
def test_an_identity_value_is_never_replaced_by_a_token_subset(
    answer: str, stored: str
) -> None:
    result = run(Flat, {"name": answer}, {"name": stored})
    assert result.changes == []
    assert rejections(result)[0] == ("replace", "/name", "identity_subset")
    assert stored in result.rejected[0].detail and answer in result.rejected[0].detail


@pytest.mark.parametrize(
    ("answer", "stored"),
    [
        ("Dana Smith", "Dana Whitfield"),  # a different name
        ("Dana Whitfield-Smith", "Dana Whitfield"),  # a longer name
        ("Whitfield Dana", "Dana Whitfield"),  # the same tokens: not a subset
        ("Danielle", "Dana Whitfield"),
    ],
)
def test_other_identity_changes_go_on_to_the_gate(answer: str, stored: str) -> None:
    result = run(Flat, {"name": answer}, {"name": stored})
    assert summary(result) == [("replace", "/name", answer)]


def test_a_non_identity_field_may_be_replaced_by_a_subset() -> None:
    result = run(Flat, {"city": "Berlin"}, {"name": "Dana", "city": "Berlin Mitte"})
    assert summary(result) == [("replace", "/city", "Berlin")]


def test_s1_field_identity_and_compound_identity_are_honoured() -> None:
    class Tagged(BaseModel):
        label: str
        code: str = Field(json_schema_extra=s1_field(identity=True))

    plan = compile_schema(Tagged)
    node = group_node(plan.root, {"code": "AB", "label": "x"})
    result = diff_group(node, {"label": "x y", "code": "AB CD"})
    # code is the identity: AB is a subset of AB CD; label is not an identity
    assert rejections(result)[0][2] == "identity_subset"
    assert summary(result) == [("replace", "/label", "x")]


def test_token_subset_unit() -> None:
    assert token_subset("Dana", "Dana Whitfield")
    assert not token_subset("Dana Whitfield", "Dana Whitfield")
    assert not token_subset("", "Dana")
    assert token_subset("o'neil", "Mary O'Neil Smith")  # words, not substrings


# --------------------------------------------------------------------------- #
# Scalar lists: set semantics
# --------------------------------------------------------------------------- #


def test_only_elements_the_record_lacks_are_appended() -> None:
    result = run(
        Flat,
        {"notes": ["Prefers email", "Likes tea at four o'clock sharp"]},
        {"name": "D", "notes": ["Prefers email", "Met in Rome"]},
    )
    assert summary(result) == [("add", "/notes/-", "Likes tea at four o'clock sharp")]


def test_elements_of_the_record_missing_from_the_answer_are_left_alone() -> None:
    result = run(
        Flat,
        {"notes": ["Prefers email"]},
        {"name": "D", "notes": ["Met in Rome", "Prefers email"]},
    )
    assert result.changes == []
    assert "/notes" not in {r.path for r in result.rejected}


def test_a_list_the_record_lacks_is_added_whole_once() -> None:
    for record in ({"name": "D"}, {"name": "D", "notes": None}):
        result = run(Flat, {"notes": ["a b c", "d e f", "a b c"]}, record)
        assert summary(result) == [("add", "/notes", ["a b c", "d e f"])]


def test_a_repeat_inside_the_answer_is_appended_once() -> None:
    result = run(
        Flat,
        {"notes": ["new fact here", "new fact here"]},
        {"name": "D", "notes": ["old"]},
    )
    assert summary(result) == [("add", "/notes/-", "new fact here")]


def test_every_new_element_is_its_own_operation_in_answer_order() -> None:
    result = run(Flat, {"scores": [3, 1, 2]}, {"name": "D", "scores": [1]})
    assert summary(result) == [("add", "/scores/-", 3), ("add", "/scores/-", 2)]


def test_near_duplicates_are_marked_for_a_clef_check() -> None:
    result = run(
        Flat,
        {"notes": ["Prefers email to phone calls"]},
        {"name": "D", "notes": ["Prefers email over phone", "Unrelated thing"]},
    )
    (change,) = result.changes
    assert change.op == "add" and change.path == "/notes/-"
    assert [(old, round(r, 1)) for old, r in change.near] == [
        ("Prefers email over phone", 0.8)
    ]


def test_a_new_element_that_resembles_nothing_is_not_marked() -> None:
    result = run(
        Flat, {"notes": ["Moving to Berlin"]}, {"name": "D", "notes": ["Likes tea"]}
    )
    (change,) = result.changes
    assert change.near == []


def test_a_very_close_repeat_is_a_duplicate_without_asking() -> None:
    result = run(
        Flat,
        {"notes": ["prefers email over phone."]},
        {"name": "D", "notes": ["Prefers email over phone"]},
    )
    assert result.changes == []
    (dup,) = [r for r in result.rejected if r.reason == "duplicate"]
    assert (dup.op, dup.path) == ("add", "/notes/-")
    assert dup.p is not None and dup.p >= 0.9


def test_number_lists_compare_by_value() -> None:
    assert new_positions(compile_schema(Flat).spec(("scores",)), [1, 2], [1.0, 3]) == [
        1
    ]


# --------------------------------------------------------------------------- #
# Optional sub-models
# --------------------------------------------------------------------------- #


class Address(BaseModel):
    city: str
    zip: Optional[str] = None


class Author(BaseModel):
    name: str
    address: Optional[Address] = None


class Book(BaseModel):
    title: str
    author: Author


def test_an_optional_sub_model_absent_in_the_input_makes_no_operation() -> None:
    result = run(
        Book,
        {"title": "Dune"},
        {"title": "Dune", "author": {"name": "F", "address": {"city": "Reno"}}},
        absent={"author.address"},
    )
    assert result.changes == []
    assert (None, "/author/address", "not_mentioned") in rejections(result)


def test_a_sub_model_the_record_lacks_is_one_add_of_the_whole_document() -> None:
    result = run(
        Book,
        {"author.address.city": "Portland", "author.address.zip": "97201"},
        {"title": "Dune", "author": {"name": "F"}},
    )
    (change,) = result.changes
    assert (change.op, change.path) == ("add", "/author/address")
    assert change.value == {"city": "Portland", "zip": "97201"}
    assert change.fields == ("author.address.city", "author.address.zip")
    assert change.field_path == "author.address" and change.old is None


def test_a_null_sub_model_in_the_record_is_replaced_by_an_add() -> None:
    result = run(
        Book,
        {"author.address.city": "Portland"},
        {"title": "Dune", "author": {"name": "F", "address": None}},
    )
    assert summary(result) == [("add", "/author/address", {"city": "Portland"})]


def test_a_required_sub_model_the_record_lacks_is_added_whole_too() -> None:
    result = run(Book, {"author.name": "Frank"}, {"title": "Dune"})
    assert summary(result) == [("add", "/author", {"name": "Frank"})]


def test_a_sub_model_the_record_has_is_diffed_leaf_by_leaf() -> None:
    result = run(
        Book,
        {"author.address.city": "Portland", "author.address.zip": "97201"},
        {
            "title": "Dune",
            "author": {"name": "F", "address": {"city": "Reno", "zip": "97201"}},
        },
    )
    assert summary(result) == [("replace", "/author/address/city", "Portland")]


def test_a_sub_model_with_nothing_stated_makes_no_operation() -> None:
    result = run(Book, {}, {"title": "Dune", "author": {"name": "F"}})
    assert result.changes == []


def test_a_missing_sub_model_with_no_answers_is_not_mentioned() -> None:
    result = run(Book, {}, {"title": "Dune"})
    assert result.changes == []
    assert (None, "/author", "not_mentioned") in rejections(result)


def test_build_doc_keeps_non_null_values_by_document_key() -> None:
    plan = compile_schema(Book)
    node = group_node(plan.root, {"title": "T", "author.name": "N"})
    assert build_doc(node) == {"title": "T", "author": {"name": "N"}}
    empty = group_node(plan.root, {})
    assert build_doc(empty) is None


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #


class Aliased(BaseModel):
    full_name: str = Field(alias="fullName")
    home: Optional[Address] = Field(default=None, alias="homeAddress")


def test_paths_use_the_document_keys() -> None:
    result = run(
        Aliased,
        {"full_name": "Dana Smith", "home.city": "Rome"},
        {"fullName": "Dana Whitfield"},
    )
    assert [(c.op, c.path) for c in result.changes] == [
        ("replace", "/fullName"),
        ("add", "/homeAddress"),
    ]
    assert result.changes[0].field_path == "full_name"  # the dotted field path


# --------------------------------------------------------------------------- #
# Unions of models
# --------------------------------------------------------------------------- #


class Cat(BaseModel):
    kind: Literal["cat"] = "cat"
    name: str
    lives: Optional[int] = None


class Dog(BaseModel):
    kind: Literal["dog"] = "dog"
    name: str


class Owner(BaseModel):
    name: str
    pet: Union[Cat, Dog]


class LooseOwner(BaseModel):
    name: str
    pet: Optional[Union[Cat, Dog]] = None


def owner_values(**extra: Any) -> dict[str, Any]:
    return {"pet.name": "Rex", **extra}


def test_a_different_union_member_is_one_replace_of_the_whole_value() -> None:
    result = run(
        Owner,
        owner_values(),
        {"name": "Sam", "pet": {"kind": "cat", "name": "Tom", "lives": 9}},
        chosen={"pet": "Dog"},
    )
    (change,) = result.changes
    assert (change.op, change.path) == ("replace", "/pet")
    assert change.value == {"kind": "dog", "name": "Rex"}
    assert change.old == {"kind": "cat", "name": "Tom", "lives": 9}
    assert change.fields == ("pet.name",)


def test_the_same_union_member_is_diffed_leaf_by_leaf() -> None:
    result = run(
        Owner,
        owner_values(),
        {"name": "Sam", "pet": {"kind": "dog", "name": "Max"}},
        chosen={"pet": "Dog"},
    )
    assert summary(result) == [("replace", "/pet/name", "Rex")]


def test_a_union_the_record_lacks_is_one_add() -> None:
    result = run(LooseOwner, owner_values(), {"name": "Sam"}, chosen={"pet": "Dog"})
    assert summary(result) == [("add", "/pet", {"kind": "dog", "name": "Rex"})]


def test_a_record_union_that_fits_no_member_is_replaced() -> None:
    result = run(
        Owner,
        owner_values(),
        {"name": "Sam", "pet": {"nonsense": True}},
        chosen={"pet": "Dog"},
    )
    assert [(c.op, c.path) for c in result.changes] == [("replace", "/pet")]


def test_a_union_with_no_chosen_member_is_not_mentioned() -> None:
    result = run(Owner, {}, {"name": "Sam", "pet": {"kind": "dog", "name": "Max"}})
    assert result.changes == []
    assert (None, "/pet", "not_mentioned") in rejections(result)


# --------------------------------------------------------------------------- #
# Removals
# --------------------------------------------------------------------------- #


class Removable(BaseModel):
    name: str
    nickname: Optional[str] = None
    tier: str = "free"
    tags: list[str] = Field(default_factory=list)


def test_removals_need_the_opt_in() -> None:
    result = run(Removable, {}, {"name": "D", "nickname": "Dee", "tier": "pro"})
    assert not [c for c in result.changes if c.removes]


def test_with_the_opt_in_a_silent_field_proposes_a_removal() -> None:
    result = run(
        Removable,
        {},
        {"name": "D", "nickname": "Dee", "tier": "pro", "tags": ["a b"]},
        allow_null_removals=True,
    )
    assert [(c.op, c.path, c.value, c.removes) for c in result.changes] == [
        ("replace", "/nickname", None, True),  # nullable: set to null
        ("remove", "/tier", None, True),  # has a default: the key goes
        ("remove", "/tags", None, True),
    ]
    assert ("remove", "/name", "required_field") in rejections(result)
    assert (None, "/nickname", "not_mentioned") not in rejections(result)


def test_a_field_the_record_does_not_hold_has_nothing_to_remove() -> None:
    result = run(Removable, {}, {"name": "D", "tags": []}, allow_null_removals=True)
    assert result.changes == []


def test_an_explained_null_is_not_removed() -> None:
    result = run(
        Removable,
        {},
        {"name": "D", "nickname": "Dee"},
        allow_null_removals=True,
        handled={"nickname"},
    )
    assert result.changes == []


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def change(op: str, path: str, value: Any = None) -> Change:
    return Change(op, path, value)  # type: ignore[arg-type]


def test_replace_first_then_remove_then_add() -> None:
    ordered = sort_changes(
        [
            change("add", "/notes/-", "n"),
            change("remove", "/b"),
            change("replace", "/a", 1),
            change("add", "/c", 2),
            change("replace", "/d", 3),
        ]
    )
    assert [(c.op, c.path) for c in ordered] == [
        ("replace", "/a"),
        ("replace", "/d"),
        ("remove", "/b"),
        ("add", "/notes/-"),
        ("add", "/c"),
    ]


def test_removals_go_from_the_highest_array_index_down() -> None:
    ordered = sort_changes(
        [
            change("remove", "/items/1"),
            change("remove", "/items/10"),
            change("remove", "/items/2"),
            change("remove", "/aaa"),
        ]
    )
    assert [c.path for c in ordered] == ["/aaa", "/items/10", "/items/2", "/items/1"]


def test_a_set_to_null_counts_as_a_removal_in_the_order() -> None:
    ordered = sort_changes([change("add", "/x", 1), change("replace", "/y", None)])
    assert [c.path for c in ordered] == ["/y", "/x"]


def test_sorting_is_stable_and_deterministic() -> None:
    changes = [change("add", "/notes/-", str(i)) for i in range(5)]
    assert sort_changes(changes) == changes
    assert sort_changes(list(reversed(changes))) == list(reversed(changes))


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #


def test_apply_ops_works_on_a_copy() -> None:
    doc = {"a": 1, "notes": ["x"]}
    result = apply_ops(
        doc,
        [PatchOp("replace", "/a", 2), PatchOp("add", "/notes/-", "y")],
    )
    assert result.doc == {"a": 2, "notes": ["x", "y"]}
    assert doc == {"a": 1, "notes": ["x"]}
    assert result.failed == []
    assert result.applied == [
        PatchOp("replace", "/a", 2),
        PatchOp("add", "/notes/-", "y"),
    ]


def test_a_string_append_becomes_a_concatenating_replace() -> None:
    result = apply_ops({"note": "Likes tea."}, [PatchOp("add", "/note/-", " Moved.")])
    assert result.doc == {"note": "Likes tea. Moved."}
    assert result.applied == [
        PatchOp("replace", "/note", "Likes tea. Moved.", repaired_from="/note/-")
    ]


def test_a_string_append_of_a_non_string_is_not_repaired() -> None:
    result = apply_ops({"note": "x"}, [PatchOp("add", "/note/-", 5)])
    assert result.doc == {"note": "x"}
    assert [op.path for op, _ in result.failed] == ["/note/-"]


def test_missing_parents_are_added_outermost_first() -> None:
    result = apply_ops({}, [PatchOp("add", "/author/address/city", "Reno")])
    assert result.doc == {"author": {"address": {"city": "Reno"}}}
    assert result.applied == [
        PatchOp("add", "/author", {}, repaired_from="/author/address/city"),
        PatchOp("add", "/author/address", {}, repaired_from="/author/address/city"),
        PatchOp("add", "/author/address/city", "Reno"),
    ]


def test_a_missing_list_parent_is_created_as_a_list() -> None:
    result = apply_ops({"name": "D"}, [PatchOp("add", "/notes/-", "first")])
    assert result.doc == {"name": "D", "notes": ["first"]}
    assert result.applied[0] == PatchOp("add", "/notes", [], repaired_from="/notes/-")


def test_only_the_missing_part_of_the_parents_is_added() -> None:
    result = apply_ops(
        {"author": {"name": "F"}}, [PatchOp("add", "/author/address/city", "Reno")]
    )
    assert result.doc == {"author": {"name": "F", "address": {"city": "Reno"}}}
    assert [op.path for op in result.applied] == [
        "/author/address",
        "/author/address/city",
    ]


def test_missing_parents_are_not_invented_under_a_scalar() -> None:
    result = apply_ops(
        {"author": "Frank"}, [PatchOp("add", "/author/address/city", "x")]
    )
    assert result.doc == {"author": "Frank"} and len(result.failed) == 1


def test_a_replace_of_a_missing_key_fails_without_stopping_the_rest() -> None:
    result = apply_ops(
        {"a": 1},
        [PatchOp("replace", "/zzz", 1), PatchOp("replace", "/a", 2)],
    )
    assert result.doc == {"a": 2}
    assert [(op.path, bool(error)) for op, error in result.failed] == [("/zzz", True)]
    assert [op.path for op in result.applied] == ["/a"]


def test_a_remove_applies() -> None:
    result = apply_ops({"a": 1, "b": 2}, [PatchOp("remove", "/b")])
    assert result.doc == {"a": 1}


# --------------------------------------------------------------------------- #
# Validation errors back to operations
# --------------------------------------------------------------------------- #


def test_ops_for_errors_matches_on_one_line_of_the_path() -> None:
    ops = [
        PatchOp("replace", "/age", -1),
        PatchOp("add", "/notes/-", "x"),
        PatchOp("add", "/address/city", "Rome"),
        PatchOp("replace", "/name", "N"),
    ]
    assert ops_for_errors(ops, [("age",)]) == [ops[0]]
    assert ops_for_errors(ops, [("notes", 3)]) == [ops[1]]
    assert ops_for_errors(ops, [("address",)]) == [ops[2]]
    assert ops_for_errors(ops, [("address", "zip")]) == []
    assert ops_for_errors(ops, [("age",), ("name",)]) == [ops[0], ops[3]]


def test_a_model_wide_error_implicates_nothing() -> None:
    assert ops_for_errors([PatchOp("replace", "/age", 1)], [()]) == []


# --------------------------------------------------------------------------- #
# Small units
# --------------------------------------------------------------------------- #


def test_pointer_escapes() -> None:
    assert pointer(["a", 1, "b/c", "d~e"]) == "/a/1/b~1c/d~0e"


def test_lookup_distinguishes_missing_from_null() -> None:
    record = {"a": {"b": None}}
    assert lookup(record, ["a", "b"]) is None
    assert lookup(record, ["a", "c"]) is MISSING
    assert lookup(record, ["x", "y"]) is MISSING
    assert lookup(None, ["a"]) is MISSING
    assert not MISSING


def test_same_booleans_are_not_numbers() -> None:
    assert same(None, True, True) and not same(None, True, 1)
    assert not same("number", 1, True)
    assert same(None, None, None) and not same(None, None, 0)


def test_same_compares_numeric_strings_only_for_numeric_fields() -> None:
    assert same("number", "12.50", 12.5)
    assert not same("text", "12.50", 12.5)
    assert not same("number", "twelve", 12)


def test_same_moments_of_different_kinds_differ() -> None:
    assert same("date", "2025-03-04", "2025-03-04")
    assert not same("date", "2025-03-04", "2025-03-04T00:00:00")
    assert not same("datetime", "2025-03-04T10:00:00", "2025-03-04T10:00:00+00:00")


def test_is_change() -> None:
    plan = compile_schema(Flat)
    city, notes = plan.spec(("city",)), plan.spec(("notes",))
    assert not is_change(city, None, "x")
    assert is_change(city, "a", MISSING) and is_change(city, "a", None)
    assert not is_change(city, "a", "a ") and is_change(city, "b", "a")
    assert is_change(notes, ["x"], MISSING) and not is_change(notes, ["x"], ["x", "y"])
    assert is_change(notes, ["z"], ["x"])


def test_leaves_with_old_follows_the_record_like_the_diff() -> None:
    plan: SchemaPlan = compile_schema(Book)
    node = group_node(
        plan.root, {"title": "T", "author.name": "N", "author.address.city": "C"}
    )
    seen = {
        leaf.spec.pretty: old
        for leaf, old in leaves_with_old(
            node, {"title": "T0", "author": {"name": "N0"}}
        )
    }
    assert seen["title"] == "T0" and seen["author.name"] == "N0"
    assert seen["author.address.city"] is MISSING  # no address in the record
    # an absent group has no leaves at all
    absent = group_node(plan.root, {}, absent={"author.address"})
    assert "author.address.city" not in {
        leaf.spec.pretty for leaf, _ in leaves_with_old(absent, {})
    }


def test_leaves_with_old_does_not_borrow_values_from_another_union_member() -> None:
    plan = compile_schema(Owner)
    node = group_node(plan.root, owner_values(), chosen={"pet": "Dog"})
    union: UnionGroup = plan.unions[0]
    assert union.members
    seen = {
        leaf.spec.pretty: old
        for leaf, old in leaves_with_old(
            node, {"name": "S", "pet": {"kind": "cat", "name": "Rex", "lives": 9}}
        )
    }
    assert seen["pet.name"] is MISSING  # the cat's name is not the dog's
