"""Tests for nested sub-models, field groups and unions of models in the compiler."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import datetime as dt
import enum
from decimal import Decimal
from typing import Annotated, Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field, create_model
from typing_extensions import NotRequired, TypedDict

from trustcall_system_one import (
    FieldGroup,
    FieldSpec,
    Limits,
    SchemaPlan,
    compile_schema,
    s1_field,
)
from trustcall_system_one._flatten import prune_nulls, unflatten
from trustcall_system_one._names import Path
from trustcall_system_one._types import UNSTATED
from trustcall_system_one._wire import ChoiceQuestion, NoulQuestion


class Address(BaseModel):
    """An address."""

    city: str
    zip: Optional[str] = None


class Author(BaseModel):
    name: str
    address: Optional[Address] = None


class Book(BaseModel):
    title: str
    author: Author


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


def pretties(specs: Any) -> list[str]:
    return [spec.pretty for spec in specs]


def instr(question: Any) -> str:
    """The instructions of a Clef question (always a string here)."""
    text = question.instructions
    assert isinstance(text, str)
    return text


# --------------------------------------------------------------------------- #
# Expansion into dotted leaf paths
# --------------------------------------------------------------------------- #


def test_nested_models_expand_into_dotted_leaf_paths_depth_first() -> None:
    plan = compile_schema(Book)
    assert pretties(plan.specs) == [
        "title",
        "author",
        "author.name",
        "author.address",
        "author.address.city",
        "author.address.zip",
    ]
    assert [spec.engine for spec in plan.specs] == [
        "clairic",
        "submodel",
        "clairic",
        "submodel",
        "clairic",
        "clairic",
    ]
    assert plan.spec(("author", "address", "city")).path == (
        "author",
        "address",
        "city",
    )
    assert plan.spec(("author",)).models == (Author,)


def test_wire_names_come_from_the_full_path() -> None:
    plan = compile_schema(Book)
    assert list(plan.clairic_fields()) == [
        "title",
        "author.name",
        "author.address.city",
        "author.address.zip",
    ]
    assert plan.spec(("author", "address", "city")).wire_name == "author.address.city"
    assert plan.clairic_names.name_to_path["author.address.zip"] == (
        "author",
        "address",
        "zip",
    )
    # the optional sub-model's presence question has a name of its own
    assert list(plan.clef_questions()) == ["author.address.__presence"]


def test_same_leaf_name_in_two_sub_models_does_not_collide() -> None:
    class Publisher(BaseModel):
        name: str
        address: Address

    class Edition(BaseModel):
        author: Author
        publisher: Publisher

    plan = compile_schema(Edition)
    names = list(plan.clairic_fields())
    assert len(names) == len(set(names)) == 6
    assert "author.name" in names and "publisher.name" in names
    assert "author.address.city" in names and "publisher.address.city" in names


def test_nested_leaves_keep_the_parent_context_as_the_prefix() -> None:
    plan = compile_schema(Book)
    city = plan.spec(("author", "address", "city"))
    assert city.context == "Book → author → address"
    assert city.instructions == "Book → author → address: city"
    assert city.scope == "author → address"
    assert city.scoped == "city (author → address)"
    name = plan.spec(("author", "name"))
    assert name.instructions == "Book → author: name"
    top = plan.spec(("title",))
    assert top.context == "" and top.scope == "" and top.scoped == "title"
    assert top.instructions == "title"


def test_a_described_nested_field_uses_its_own_description() -> None:
    class Inner(BaseModel):
        city: str = Field(description="The city where the printing house stands")

    class Outer(BaseModel):
        press: Inner

    spec = compile_schema(Outer).spec(("press", "city"))
    assert spec.description == "The city where the printing house stands"
    assert (
        spec.instructions == "Outer → press: The city where the printing house stands"
    )


def test_nested_clef_fields_carry_the_context_too() -> None:
    class Inner(BaseModel):
        color: Color
        happy: bool

    class Outer(BaseModel):
        inner: Inner

    plan = compile_schema(Outer)
    color = plan.spec(("inner", "color"))
    assert color.engine == "clef"
    assert color.wire_name == color.clef_name == "inner.color"
    assert color.clef is not None
    assert instr(color.clef).startswith("Outer → inner: ")
    assert color.clef_context == "Outer → inner"
    assert plan.spec(("inner", "happy")).clef_name == "inner.happy"


# --------------------------------------------------------------------------- #
# Field groups
# --------------------------------------------------------------------------- #


def test_groups_form_a_tree_with_the_root_first() -> None:
    plan = compile_schema(Book)
    root, author, address = plan.groups
    assert plan.root is root
    assert [g.path for g in plan.groups] == [(), ("author",), ("author", "address")]
    assert root.model is Book and author.model is Author and address.model is Address
    assert pretties(root.specs) == ["title"]
    assert pretties(author.specs) == ["author.name"]
    assert pretties(address.specs) == ["author.address.city", "author.address.zip"]
    assert root.children == (author,) and author.children == (address,)
    assert address.children == ()
    assert list(root.walk()) == [root, author, address]
    assert address.pretty == "author.address" and root.pretty == ""


def test_group_flags_describe_the_sub_model_field() -> None:
    class Maybe(BaseModel):
        a: Optional[Address] = None  # nullable and defaulted
        b: Optional[Address]  # nullable, no default
        c: Address = Address(city="x")  # defaulted, not nullable
        d: Address  # required

    plan = compile_schema(Maybe)
    flags = {
        g.pretty: (g.optional, g.nullable, g.has_default)
        for g in plan.groups
        if g.pretty in "abcd" and g.pretty
    }
    assert flags == {
        "a": (True, True, True),
        "b": (True, True, False),
        "c": (True, False, True),
        "d": (False, False, False),
    }
    assert plan.root.optional is False


def test_only_an_optional_group_gets_a_presence_question() -> None:
    plan = compile_schema(Book)
    author = plan.groups[1]
    address = plan.groups[2]
    assert author.spec is not None and author.spec.clef is None  # required
    assert address.spec is not None
    question = address.spec.clef
    assert isinstance(question, NoulQuestion)
    assert question.instructions == "Book → author: Does `input` state address?"
    assert question.criteria is not None
    assert "address" in str(question.criteria.true)
    assert "does not mention" in str(question.criteria.false)
    assert address.spec.clef_name == address.spec.wire_name
    assert plan.root.spec is None


def test_group_description_comes_from_the_sub_model_field() -> None:
    class Outer(BaseModel):
        home: Optional[Address] = Field(None, description="Where the author lives")

    plan = compile_schema(Outer)
    group = plan.groups[1]
    assert group.description == "Where the author lives"
    assert group.spec is not None and group.spec.clef is not None
    assert "Where the author lives" in instr(group.spec.clef)
    assert plan.root.description == ""


def test_the_root_group_describes_the_model() -> None:
    assert compile_schema(Address).root.description == "An address."


def test_group_identity_is_computed_per_level() -> None:
    class Inner(BaseModel):
        code: str = Field(json_schema_extra=s1_field(identity=True))
        name: str

    class Outer(BaseModel):
        title: str
        label: str
        inner: Inner

    plan = compile_schema(Outer)
    assert plan.identity == ("title",)
    assert plan.root.identity == ("title",)
    assert plan.groups[1].identity == ("code",)
    marked = plan.spec(("inner", "code"))
    assert marked.identity is True
    # the default rule runs per group too
    assert compile_schema(Book).groups[1].identity == ("name",)
    assert compile_schema(Book).groups[2].identity == ("city",)


def test_plan_identity_ignores_nested_names() -> None:
    class Inner(BaseModel):
        name: str

    class Outer(BaseModel):
        inner: Inner
        note: Optional[str] = None

    assert compile_schema(Outer).identity == ()


def test_nested_collections_stay_deferred() -> None:
    class Series(BaseModel):
        books: list[Book] = []

    class Shelf(BaseModel):
        label: str
        series: Series
        tags: dict[str, int] = {}

    plan = compile_schema(Shelf)
    assert pretties(plan.deferred()) == ["series.books", "tags"]
    assert plan.spec(("series", "books")).models == (Book,)
    assert plan.groups[1].specs[0].engine == "collection"


# --------------------------------------------------------------------------- #
# Depth
# --------------------------------------------------------------------------- #


def test_max_depth_caps_nested_expansion_with_a_warning() -> None:
    plan = compile_schema(Book, limits=Limits(max_depth=2))
    assert plan.spec(("author", "name")).engine == "clairic"
    address = plan.spec(("author", "address"))
    assert address.engine == "skip" and address.warnings == ("max_depth",)
    assert address.clef is None
    assert pretties(plan.specs) == ["title", "author", "author.name", "author.address"]
    assert [g.pretty for g in plan.groups] == ["", "author"]
    assert any("author.address: max_depth" in w for w in plan.warnings)
    shallow = compile_schema(Book, limits=Limits(max_depth=1))
    assert shallow.spec(("author",)).engine == "skip"
    assert [g.pretty for g in shallow.groups] == [""]


def test_a_self_recursive_optional_model_unrolls_max_recursion_levels() -> None:
    class Node(BaseModel):
        name: str
        parent: Optional["Node"] = None

    Node.model_rebuild()
    plan = compile_schema(Node)  # max_recursion=2: the root and two parents
    names = [spec.pretty for spec in plan.specs if spec.engine == "clairic"]
    assert names == ["name", "parent.name", "parent.parent.name"]
    deepest = plan.spec(("parent", "parent", "parent"))
    assert deepest.engine == "skip" and deepest.warnings == ("max_recursion",)
    assert [g.pretty for g in plan.groups] == ["", "parent", "parent.parent"]
    assert all(g.optional for g in plan.groups[1:])
    assert len(plan.clef_questions()) == 2  # one presence question per parent
    # one warning for the recursion, not one per skipped instance
    assert [w for w in plan.warnings if "max_recursion" in w] == [
        "parent.parent.parent: max_recursion: Node reappears on its own ancestor "
        "chain more than limits.max_recursion=2 times; the field is skipped"
    ]
    assert len(plan.warnings) == 1


def test_max_recursion_and_max_depth_both_bound_the_unrolling() -> None:
    class Node(BaseModel):
        name: str
        parent: Optional["Node"] = None

    Node.model_rebuild()
    flat = compile_schema(Node, limits=Limits(max_recursion=0))
    assert [g.pretty for g in flat.groups] == [""]
    assert flat.spec(("parent",)).warnings == ("max_recursion",)
    deep = compile_schema(Node, limits=Limits(max_recursion=20))
    names = [spec.pretty for spec in deep.specs if spec.engine == "clairic"]
    assert len(names) == 8  # leaves at most 8 deep: max_depth wins
    assert deep.spec(tuple(["parent"] * 8)).warnings == ("max_depth",)
    small = compile_schema(Node, limits=Limits(max_depth=3, max_recursion=20))
    assert [g.pretty for g in small.groups] == ["", "parent", "parent.parent"]
    assert sum("max_depth" in w for w in deep.warnings) == 1


def test_recursive_collections_are_named_once_in_the_warnings() -> None:
    class Tree(BaseModel):
        label: str
        parent: Optional["Tree"] = None
        children: list["Tree"] = []

    Tree.model_rebuild()
    plan = compile_schema(Tree)
    assert [spec.pretty for spec in plan.deferred()] == [
        "parent.parent.children",
        "parent.children",
        "children",
    ]
    # the compile warnings name each (model, field) once
    assert len(plan.warnings) == 1 and "max_recursion" in plan.warnings[0]


def test_max_groups_is_a_compile_error_not_a_silent_truncation() -> None:
    class Tri(BaseModel):
        v: str
        a: Optional["Tri"] = None
        b: Optional["Tri"] = None
        c: Optional["Tri"] = None

    Tri.model_rebuild()
    with pytest.raises(ValueError, match="max_groups") as info:
        compile_schema(Tri, limits=Limits(max_recursion=8, max_groups=64))
    message = str(info.value)
    assert "Tri" in message and "max_recursion" in message
    assert "max_depth" in message and "s1_field(skip=True)" in message
    assert len(compile_schema(Tri).groups) == 13  # 1 + 3 + 9 with the default
    with pytest.raises(ValueError, match="max_groups"):
        compile_schema(Book, limits=Limits(max_groups=2))
    assert len(compile_schema(Book, limits=Limits(max_groups=3)).groups) == 3


def test_union_members_count_towards_max_groups() -> None:
    class Home(BaseModel):
        pet: Union[Cat, Dog]

    with pytest.raises(ValueError, match="max_groups"):
        compile_schema(Home, limits=Limits(max_groups=2))
    plan = compile_schema(Home, limits=Limits(max_groups=3))
    assert len(plan.groups) + len(plan.member_groups) == 3


def test_a_recursive_union_leaves_out_the_member_past_the_limit() -> None:
    class Num(BaseModel):
        kind: Literal["num"]
        n: float

    class BinOp(BaseModel):
        kind: Literal["op"]
        op: str
        left: Union[Num, "BinOp"]

    BinOp.model_rebuild()

    class Calc(BaseModel):
        expr: Union[Num, BinOp]

    plan = compile_schema(Calc)
    # BinOp may reappear on its own chain twice (three BinOps deep); past that
    # the member is simply not offered and the union is left with Num
    level = plan.unions[0]
    for _ in range(3):
        level = level.members[1].group.unions[0]
    deepest = level
    assert [m.label for m in level.members] == ["Num"]
    assert [m.label for m in deepest.members] == ["Num"]
    assert deepest.spec.warnings == ("max_recursion",)
    assert len(deepest.members) == 1 and deepest.spec.clef is None
    assert len([w for w in plan.warnings if "max_recursion" in w]) == 1
    # every union field up the chain still offers both members
    top = plan.unions[0]
    assert [m.label for m in top.members] == ["Num", "BinOp"]


def test_a_member_whose_required_recursive_field_was_skipped_is_left_out() -> None:
    class Leaf(BaseModel):
        kind: Literal["leaf"]
        size: int

    class Branch(BaseModel):
        kind: Literal["branch"]
        up: "Branch"  # required and recursive: it can never be filled in

    Branch.model_rebuild()

    class Tree(BaseModel):
        root: Union[Leaf, Branch]

    plan = compile_schema(Tree)
    (union,) = plan.unions
    assert [m.label for m in union.members] == ["Leaf"]
    assert union.spec.clef is None  # one option left: nothing to ask
    assert dict(union.spec.key_map or {}) == {"leaf": "Leaf"}
    assert any("Branch" in w and "left out of the union" in w for w in plan.warnings)
    # the wire names of the survivors are untouched
    assert list(plan.clairic_fields()) == ["root.Leaf.size"]
    assert plan.clef_questions() == {}


def test_a_union_with_no_viable_member_is_skipped() -> None:
    class A(BaseModel):
        kind: Literal["a"]
        up: "A"

    class B(BaseModel):
        kind: Literal["b"]
        up: "B"

    A.model_rebuild()
    B.model_rebuild()

    class Holder(BaseModel):
        x: Union[A, B]
        y: int

    plan = compile_schema(Holder)
    assert plan.spec(("x",)).engine == "skip"
    assert plan.spec(("x",)).warnings[-1] == "no_viable_member"
    assert plan.unions == ()
    assert any("no_viable_member" in w for w in plan.warnings)


def test_a_required_sub_model_made_unviable_removes_the_member_that_holds_it() -> None:
    class Loop(BaseModel):
        again: "Loop"

    Loop.model_rebuild()

    class WithLoop(BaseModel):
        kind: Literal["loop"]
        inner: Loop

    class Plain(BaseModel):
        kind: Literal["plain"]
        x: int

    class Holder(BaseModel):
        item: Union[WithLoop, Plain]

    plan = compile_schema(Holder)
    assert [m.label for m in plan.unions[0].members] == ["Plain"]


def test_an_optional_recursive_field_does_not_remove_a_member() -> None:
    class Rec(BaseModel):
        kind: Literal["rec"]
        up: Optional["Rec"] = None

    Rec.model_rebuild()

    class Plain(BaseModel):
        kind: Literal["plain"]

    class Holder(BaseModel):
        item: Union[Rec, Plain]

    plan = compile_schema(Holder)
    assert [m.label for m in plan.unions[0].members] == ["Rec", "Plain"]


def test_long_nested_names_are_shortened_with_the_context() -> None:
    class Leaf(BaseModel):
        a_remarkably_long_leaf_name_that_pushes_the_wire_name_over_the_limit: str

    class Mid(BaseModel):
        a_long_middle_section_name: Leaf

    class Top(BaseModel):
        some_long_top_level_section: Mid

    plan = compile_schema(Top)
    (spec,) = [s for s in plan.specs if s.engine == "clairic"]
    assert len(spec.wire_name) <= 64
    assert spec.wire_name in plan.clairic_names.shortened
    assert spec.instructions.startswith(
        "Top → some long top level section → a long middle section name: "
    )


# --------------------------------------------------------------------------- #
# Every flat type inside a nested model
# --------------------------------------------------------------------------- #

FLAT_TYPES: list[Any] = [
    str,
    int,
    float,
    Decimal,
    dt.date,
    dt.time,
    dt.datetime,
    dt.timedelta,
    Optional[str],
    Annotated[str, Field(min_length=2, max_length=5)],
    list[str],
    list[int],
    list[dt.date],
    tuple[str, ...],
    set[str],
    list[Literal["a", "b"]],
    list[Color],
    bool,
    Optional[bool],
    Literal["a", "b"],
    Optional[Literal["a", "b"]],
    Color,
    Literal["only"],
    Union[int, str],
    Union[Literal["a"], Literal["b"]],
    Optional[Union[int, str]],
    list[Union[int, str]],
    dict[str, Any],
]


def flat_spec(annotation: Any) -> FieldSpec:
    model = create_model("M", f=(annotation, Field(..., description="Thing")))
    return compile_schema(model).specs[0]


def nested_spec(annotation: Any, depth: int) -> FieldSpec:
    wrapped: list[Any] = [
        create_model("Leaf", f=(annotation, Field(..., description="Thing")))
    ]
    path: tuple[str, ...] = ("f",)
    for level in range(depth):
        fields: dict[str, Any] = {f"s{level}": (wrapped[-1], ...)}
        wrapped.append(create_model(f"L{level}", **fields))
        path = (f"s{level}", *path)
    return compile_schema(wrapped[-1]).spec(path)


def shape(spec: FieldSpec) -> dict[str, Any]:
    """Everything about a spec except names and the text that names the path."""
    clairic = (
        spec.clairic.model_dump(exclude_none=True, exclude={"instructions"})
        if spec.clairic
        else None
    )
    clef = None
    if spec.clef is not None:
        clef = spec.clef.model_dump(exclude_none=True, exclude={"instructions"})
    return {
        "engine": spec.engine,
        "clairic": clairic,
        "clef": clef,
        "key_map": dict(spec.key_map) if spec.key_map is not None else None,
        "item_kind": spec.item_kind,
        "element_choices": (
            dict(spec.element_choices) if spec.element_choices is not None else None
        ),
        "post": spec.post,
        "required": spec.required,
        "nullable": spec.nullable,
        "has_default": spec.has_default,
        "null_threshold": spec.null_threshold,
        "min_confidence": spec.min_confidence,
        "hints": spec.hints,
        "warnings": spec.warnings,
        "description": spec.description,
    }


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("annotation", FLAT_TYPES, ids=repr)
def test_every_flat_type_compiles_the_same_inside_a_nested_model(
    annotation: Any, depth: int
) -> None:
    flat = flat_spec(annotation)
    nested = nested_spec(annotation, depth)
    assert shape(nested) == shape(flat)
    assert len(nested.path) == depth + 1
    if flat.clairic is not None:
        assert nested.clairic is not None
        assert nested.instructions == f"{nested.context}: {flat.instructions}"
        assert nested.instructions.startswith("L")
    if flat.clef is not None:
        assert nested.clef is not None and nested.clef_context
        assert instr(nested.clef).endswith(instr(flat.clef)) or (
            flat.clairic is not None
        )


def test_nested_optional_and_defaulted_leaves_use_the_optional_threshold() -> None:
    class Inner(BaseModel):
        a: str
        b: Optional[str] = None
        c: int = 3

    class Outer(BaseModel):
        inner: Inner

    plan = compile_schema(Outer)
    assert plan.spec(("inner", "a")).null_threshold == 0.7
    assert plan.spec(("inner", "b")).null_threshold == 0.3
    assert plan.spec(("inner", "c")).null_threshold == 0.3
    assert plan.spec(("inner", "c")).has_default is True


def test_nested_const_and_unstated_work_per_level() -> None:
    class Inner(BaseModel):
        kind: Literal["inner"]
        mood: Optional[Literal["happy", "sad"]] = None
        shade: Literal["dark", "light"]

    class Outer(BaseModel):
        inner: Inner

    plan = compile_schema(Outer)
    kind = plan.spec(("inner", "kind"))
    assert kind.engine == "const" and kind.wire_name == "" and kind.clef is None
    mood = plan.spec(("inner", "mood"))
    assert mood.engine == "clef" and mood.key_map is not None
    assert list(mood.key_map) == ["happy", "sad", "unstated"]
    assert mood.key_map["unstated"] is UNSTATED
    shade = plan.spec(("inner", "shade"))
    assert shade.key_map is not None and "unstated" not in shade.key_map


# --------------------------------------------------------------------------- #
# Aliases at every level
# --------------------------------------------------------------------------- #


class Inner(BaseModel):
    val: int = Field(alias="v")
    other: Optional[str] = Field(None, validation_alias="o")


class Outer(BaseModel):
    inner: Inner = Field(alias="in")
    maybe: Optional[Inner] = Field(None, alias="mb")
    top: str = Field("x", alias="t")


def test_aliases_are_recorded_per_level() -> None:
    plan = compile_schema(Outer)
    assert plan.spec(("inner",)).alias == "in"
    assert plan.spec(("inner", "val")).alias == "v"
    assert plan.spec(("inner", "other")).alias == "o"
    assert plan.spec(("maybe",)).alias == "mb"
    assert plan.spec(("maybe", "val")).alias == "v"
    assert plan.spec(("top",)).alias == "t"
    assert plan.groups[1].alias == "in" and plan.groups[2].alias == "mb"
    assert plan.root.alias is None


def test_unflatten_keys_every_level_by_its_alias() -> None:
    plan = compile_schema(Outer)
    values: dict[Path, Any] = {
        ("top",): "y",
        ("inner", "val"): 3,
        ("inner", "other"): "z",
        ("maybe", "val"): 4,
        ("maybe", "other"): None,
    }
    doc = unflatten(values, plan=plan)
    assert doc == {
        "t": "y",
        "in": {"v": 3, "o": "z"},
        "mb": {"v": 4, "o": None},
    }
    assert Outer.model_validate(doc).inner.val == 3
    assert unflatten(values) != doc  # without the plan the field names stay


def test_prune_nulls_works_at_every_level() -> None:
    plan = compile_schema(Outer)
    doc = {
        "t": None,  # defaulted: dropped
        "in": {"v": 3, "o": None},  # nested defaulted None dropped
        "mb": None,  # a defaulted sub-model that is None: dropped
    }
    assert prune_nulls(doc, plan) == {"in": {"v": 3}}
    # field names are accepted and renamed to the validation keys at each level
    by_name = {"top": None, "inner": {"val": 3, "other": "k"}, "maybe": {"val": 1}}
    assert prune_nulls(by_name, plan) == {
        "in": {"v": 3, "o": "k"},
        "mb": {"v": 1},
    }
    # a None for a required nested field stays so validation can report it
    assert prune_nulls({"in": {"v": None}}, plan) == {"in": {"v": None}}
    # unknown keys are kept unchanged, nested or not
    assert prune_nulls({"in": {"v": 1, "zzz": None}, "q": None}, plan) == {
        "in": {"v": 1, "zzz": None},
        "q": None,
    }


def test_unflatten_and_prune_follow_the_chosen_union_member() -> None:
    class Cat(BaseModel):
        kind: Literal["cat"]
        lives: int = Field(alias="l")

    class Dog(BaseModel):
        kind: Literal["dog"]
        tricks: int = Field(alias="tr")

    class Home(BaseModel):
        pet: Union[Cat, Dog] = Field(alias="p")

    plan = compile_schema(Home)
    values: dict[Path, Any] = {("pet", "kind"): "dog", ("pet", "tricks"): 2}
    doc = unflatten(values, plan=plan, members={("pet",): "Dog"})
    assert doc == {"p": {"kind": "dog", "tr": 2}}
    assert prune_nulls(doc, plan, members={("pet",): "Dog"}) == doc
    assert Home.model_validate(doc).pet == Dog(kind="dog", tr=2)
    cat = unflatten(
        {("pet", "kind"): "cat", ("pet", "lives"): 9},
        plan=plan,
        members={("pet",): "Cat"},
    )
    assert cat == {"p": {"kind": "cat", "l": 9}}


def test_typed_dict_sub_models_expand_too() -> None:
    class Geo(TypedDict):
        lat: float
        note: NotRequired[str]

    class Place(TypedDict):
        name: str
        geo: Geo
        maybe: Optional[Geo]

    plan = compile_schema(Place)
    assert pretties(plan.specs) == [
        "name",
        "geo",
        "geo.lat",
        "geo.note",
        "maybe",
        "maybe.lat",
        "maybe.note",
    ]
    assert plan.groups[1].model.__name__ == "Geo"
    assert plan.groups[1].optional is False and plan.groups[2].optional is True
    assert plan.spec(("geo", "note")).has_default is True


# --------------------------------------------------------------------------- #
# Unions of models
# --------------------------------------------------------------------------- #


class Cat(BaseModel):
    """A house cat."""

    kind: Literal["cat"]
    lives: int
    name: str


class Dog(BaseModel):
    kind: Literal["dog"]
    tricks: int
    name: str


class Fish(BaseModel):
    """A fish."""

    water: str


class Bird(BaseModel):
    """A bird."""

    wingspan: float


def union_of(annotation: Any, default: Any = ...) -> SchemaPlan:
    model = create_model("Home", pet=(annotation, Field(default)))
    return compile_schema(model)


def choice_of(plan: SchemaPlan) -> ChoiceQuestion:
    question = plan.spec(("pet",)).clef
    assert isinstance(question, ChoiceQuestion)
    return question


def test_a_discriminated_union_chooses_among_the_tag_values() -> None:
    plan = union_of(Annotated[Union[Cat, Dog], Field(discriminator="kind")])
    spec = plan.spec(("pet",))
    assert spec.engine == "union_models"
    assert spec.models == (Cat, Dog)
    assert spec.wire_name == spec.clef_name == "pet"
    question = choice_of(plan)
    assert list(question.criteria) == ["cat", "dog"]
    assert question.criteria["cat"] == "A house cat."
    assert question.criteria["dog"] == "Dog: has the fields kind, tricks, name."
    assert dict(spec.key_map or {}) == {"cat": "Cat", "dog": "Dog"}
    (union,) = plan.unions
    assert union.discriminator == "kind"
    assert dict(union.option_tags) == {"cat": "cat", "dog": "dog"}
    assert [m.label for m in union.members] == ["Cat", "Dog"]
    assert [m.options for m in union.members] == [("cat",), ("dog",)]
    assert union.member_for("dog") is union.members[1]
    assert union.optional is False


def test_a_common_literal_tag_is_found_without_a_discriminator() -> None:
    plan = union_of(Union[Cat, Dog])
    (union,) = plan.unions
    assert union.discriminator == "kind"
    assert list(choice_of(plan).criteria) == ["cat", "dog"]


def test_an_undiscriminated_union_chooses_among_the_model_names() -> None:
    plan = union_of(Union[Fish, Bird])
    (union,) = plan.unions
    assert union.discriminator is None and not union.option_tags
    question = choice_of(plan)
    assert dict(question.criteria) == {"Fish": "A fish.", "Bird": "A bird."}
    assert dict(plan.spec(("pet",)).key_map or {}) == {"Fish": "Fish", "Bird": "Bird"}


def test_a_shared_literal_value_is_not_a_discriminator() -> None:
    class A(BaseModel):
        kind: Literal["x"]
        a: int

    class B(BaseModel):
        kind: Literal["x"]
        b: int

    (union,) = union_of(Union[A, B]).unions
    assert union.discriminator is None
    assert list(union.spec.key_map or {}) == ["A", "B"]


def test_a_non_literal_common_field_is_not_a_discriminator() -> None:
    plan = union_of(Union[Fish, Cat])
    (union,) = plan.unions
    assert union.discriminator is None


def test_an_enum_tag_gives_the_member_names_as_options() -> None:
    class Kind(enum.Enum):
        CAT = "cat"
        DOG = "dog"

    class C(BaseModel):
        kind: Literal[Kind.CAT]
        x: int

    class D(BaseModel):
        kind: Literal[Kind.DOG]
        y: int

    plan = union_of(Union[C, D])
    (union,) = plan.unions
    assert union.discriminator == "kind"
    assert len(choice_of(plan).criteria) == 2


def test_a_tag_with_several_values_gives_a_member_several_options() -> None:
    class Big(BaseModel):
        size: Literal["large", "huge"]
        weight: int

    class Small(BaseModel):
        size: Literal["small"]
        length: int

    plan = union_of(Union[Big, Small])
    (union,) = plan.unions
    assert union.discriminator == "size"
    assert list(choice_of(plan).criteria) == ["large", "huge", "small"]
    assert [m.options for m in union.members] == [("large", "huge"), ("small",)]
    assert dict(union.spec.key_map or {}) == {
        "large": "Big",
        "huge": "Big",
        "small": "Small",
    }
    assert dict(union.option_tags) == {
        "large": "large",
        "huge": "huge",
        "small": "small",
    }
    # a tag with several values is a Clef choice inside the member
    big_tag = union.members[0].group.specs[0]
    assert big_tag.engine == "clef"


def test_an_optional_union_gets_an_unstated_option() -> None:
    plan = union_of(Optional[Union[Cat, Dog]], None)
    question = choice_of(plan)
    assert list(question.criteria) == ["cat", "dog", "unstated"]
    spec = plan.spec(("pet",))
    assert spec.key_map is not None and spec.key_map["unstated"] is UNSTATED
    (union,) = plan.unions
    assert union.optional and union.nullable and union.has_default
    assert all("unstated" not in m.options for m in union.members)
    # a defaulted but non-nullable union is optional too
    defaulted = union_of(Union[Fish, Bird], Fish(water="sea"))
    assert "unstated" in choice_of(defaulted).criteria
    assert defaulted.unions[0].nullable is False


def test_unstated_avoids_a_collision_with_a_tag_value() -> None:
    class A(BaseModel):
        kind: Literal["unstated"]
        a: int

    class B(BaseModel):
        kind: Literal["b"]
        b: int

    plan = union_of(Optional[Union[A, B]], None)
    assert list(choice_of(plan).criteria) == ["unstated", "b", "unstated_2"]


def test_members_are_compiled_as_subtrees_that_are_not_in_the_flat_specs() -> None:
    class Home(BaseModel):
        owner: str
        pet: Union[Cat, Dog]

    plan = compile_schema(Home)
    assert pretties(plan.specs) == ["owner", "pet"]
    assert pretties(plan.member_specs) == [
        "pet.kind",
        "pet.lives",
        "pet.name",
        "pet.kind",
        "pet.tricks",
        "pet.name",
    ]
    (union,) = plan.unions
    cat, dog = union.members
    assert cat.group.path == dog.group.path == ("pet",)
    assert cat.group.member == "Cat" and dog.group.member == "Dog"
    assert cat.group.model is Cat and cat.group.optional is False
    assert pretties(cat.group.specs) == ["pet.kind", "pet.lives", "pet.name"]
    assert plan.root.unions == (union,)
    assert list(plan.all_groups()) == [plan.root, cat.group, dog.group]
    assert plan.groups == (plan.root,)
    # the const tag is filled in by code
    assert cat.group.specs[0].engine == "const"
    assert cat.group.specs[0].key_map == {"cat": "cat"}


def test_member_wire_names_are_unique_even_when_members_share_field_names() -> None:
    class Home(BaseModel):
        pet: Union[Cat, Dog]

    plan = compile_schema(Home)
    names = list(plan.clairic_fields())
    assert names == [
        "pet.Cat.lives",
        "pet.Cat.name",
        "pet.Dog.tricks",
        "pet.Dog.name",
    ]
    union = plan.unions[0]
    cat_name = union.members[0].group.specs[2]
    dog_name = union.members[1].group.specs[2]
    assert cat_name.path == dog_name.path == ("pet", "name")
    assert cat_name.wire_name != dog_name.wire_name
    assert cat_name.wire_path == ("pet", "Cat", "name")
    assert cat_name.instructions == "Home → pet (Cat: A house cat): name"
    assert dog_name.instructions == "Home → pet (Dog): name"
    assert plan.clairic_names.name_to_path["pet.Dog.name"] == ("pet", "Dog", "name")
    # a shared path is ambiguous: say which member, or use the exact wire path
    with pytest.raises(ValueError, match=r"several union members.*Cat, Dog"):
        plan.spec(("pet", "name"))
    assert plan.spec(("pet", "name"), member="Cat") is cat_name
    assert plan.spec(("pet", "name"), member="Dog") is dog_name
    assert plan.spec_by_wire_path(("pet", "Dog", "name")) is dog_name
    assert plan.spec(("pet", "lives")).member_labels == ("Cat",)
    with pytest.raises(KeyError):
        plan.spec(("pet", "name"), member="Fish")
    with pytest.raises(KeyError):
        plan.spec_by_wire_path(("pet", "Cat", "tricks"))
    assert list(plan.clef_questions()) == ["pet"]


def test_members_share_a_label_only_with_a_suffix() -> None:
    other = create_model("Fish", water=(str, ...), depth=(int, ...))
    plan = union_of(Union[Fish, other])
    (union,) = plan.unions
    assert [m.label for m in union.members] == ["Fish", "Fish_2"]
    assert list(choice_of(plan).criteria) == ["Fish", "Fish_2"]
    assert len(set(plan.clairic_fields())) == 3


def test_union_members_may_contain_sub_models_and_unions() -> None:
    class Wing(BaseModel):
        span: float
        tip: Optional[Address] = None

    class Eagle(BaseModel):
        kind: Literal["eagle"]
        wing: Wing

    class Owl(BaseModel):
        kind: Literal["owl"]
        prey: Union[Fish, Bird]

    plan = union_of(Union[Eagle, Owl])
    (union,) = plan.unions
    eagle, owl = union.members
    assert [c.pretty for c in eagle.group.children] == ["pet.wing"]
    assert eagle.group.children[0].children[0].optional is True
    assert [u.path for u in owl.group.unions] == [("pet", "prey")]
    inner = owl.group.unions[0]
    assert [m.label for m in inner.members] == ["Fish", "Bird"]
    # the nested union's names live under its member
    assert inner.spec.wire_name == "pet.Owl.prey"
    assert [m.group.specs[0].wire_name for m in inner.members] == [
        "pet.Owl.prey.Fish.water",
        "pet.Owl.prey.Bird.wingspan",
    ]
    assert "pet.Eagle.wing.tip.__presence" in plan.clef_questions()
    assert plan.unions == (union,)  # the nested union belongs to its member


def test_union_choice_text_and_thresholds() -> None:
    class Home(BaseModel):
        pet: Union[Cat, Dog] = Field(description="The family pet")

    plan = compile_schema(Home)
    spec = plan.spec(("pet",))
    assert spec.description == "The family pet"
    assert spec.clef is not None
    assert spec.clef.instructions == "Pick the option that best fits: The family pet"
    assert spec.min_confidence == 0.5
    assert spec.required is True


def test_union_criteria_override_and_instructions_option() -> None:
    class Home(BaseModel):
        pet: Union[Fish, Bird] = Field(
            json_schema_extra=s1_field(
                instructions="the kind of animal", criteria={"Fish": "Lives in water"}
            )
        )

    spec = compile_schema(Home).spec(("pet",))
    assert spec.clef is not None
    assert instr(spec.clef).endswith("the kind of animal")
    assert isinstance(spec.clef, ChoiceQuestion)
    assert spec.clef.criteria == {"Fish": "Lives in water", "Bird": "A bird."}


def test_a_union_past_the_depth_limit_is_skipped() -> None:
    class Home(BaseModel):
        pet: Union[Cat, Dog]

    plan = compile_schema(Home, limits=Limits(max_depth=1))
    spec = plan.spec(("pet",))
    assert spec.engine == "skip" and spec.warnings == ("max_depth",)
    assert plan.unions == ()


def test_a_union_of_models_and_primitives_is_unsupported() -> None:
    spec = union_of(Union[Cat, int]).spec(("pet",))
    assert spec.engine == "skip" and spec.warnings == ("unsupported_union",)


def test_a_union_of_typed_dicts_compiles() -> None:
    class A(TypedDict):
        x: int

    class B(TypedDict):
        y: str

    plan = union_of(Union[A, B])
    (union,) = plan.unions
    assert [m.label for m in union.members] == ["A", "B"]
    assert all(issubclass(m.model, BaseModel) for m in union.members)


def test_unions_inside_sub_models_are_groups_of_the_sub_model() -> None:
    class Kennel(BaseModel):
        pet: Optional[Union[Cat, Dog]] = None

    class Home(BaseModel):
        kennel: Kennel

    plan = compile_schema(Home)
    kennel = plan.groups[1]
    assert [u.path for u in kennel.unions] == [("kennel", "pet")]
    assert plan.unions == kennel.unions
    assert plan.spec(("kennel", "pet")).clef_name == "kennel.pet"


def test_collections_of_unions_stay_collections() -> None:
    class Home(BaseModel):
        pets: list[Union[Cat, Dog]] = []

    plan = compile_schema(Home)
    assert plan.spec(("pets",)).engine == "collection"
    assert plan.unions == ()


def test_group_types_are_exported() -> None:
    plan = compile_schema(Book)
    assert isinstance(plan.root, FieldGroup)


# --------------------------------------------------------------------------- #
# Review fixes: owners, member identity, option counts
# --------------------------------------------------------------------------- #


def test_owners_are_the_humanised_ancestor_names() -> None:
    plan = compile_schema(Book)
    assert plan.spec(("title",)).owners == ()
    assert plan.spec(("author", "name")).owners == ("author",)
    assert plan.spec(("author", "address", "city")).owners == ("author", "address")

    class Inner(BaseModel):
        x: int

    class Outer(BaseModel):
        home_office: Inner

    assert compile_schema(Outer).spec(("home_office", "x")).owners == ("home office",)


def test_member_leaves_name_the_member_in_their_context() -> None:
    class Short(BaseModel):
        """A short one."""

        a: int

    class Long(BaseModel):
        """A member whose docstring is far too long to repeat in every question."""

        a: int

    class Plain(BaseModel):
        a: int

    class Home(BaseModel):
        pet: Union[Short, Long, Plain]

    plan = compile_schema(Home)
    (union,) = plan.unions
    short, long, plain = (m.group.specs[0] for m in union.members)
    assert short.context == "Home → pet (Short: A short one)"
    assert long.context == "Home → pet (Long)"
    assert plain.context == "Home → pet (Plain)"
    assert plain.instructions == "Home → pet (Plain): a"
    assert plain.scope == "pet (Plain)" and plain.scoped == "a (pet (Plain))"
    assert short.member_labels == ("Short",)


def test_nested_members_keep_the_member_in_the_context() -> None:
    class Wing(BaseModel):
        span: float

    class Eagle(BaseModel):
        kind: Literal["eagle"]
        wing: Wing

    class Owl(BaseModel):
        kind: Literal["owl"]

    class Sky(BaseModel):
        bird: Union[Eagle, Owl]

    plan = compile_schema(Sky)
    span = plan.spec(("bird", "wing", "span"))
    assert span.context == "Sky → bird (Eagle) → wing"
    assert span.owners == ("bird", "wing")


def test_the_number_of_options_not_members_is_checked_against_255() -> None:
    def member(name: str, start: int, count: int) -> Any:
        values = tuple(f"{name}{i}" for i in range(start, start + count))
        tag: Any = Literal[values]
        return create_model(name, kind=(tag, ...), x=(int, ...))

    a = member("A", 0, 130)
    b = member("B", 0, 130)

    class Home(BaseModel):
        pet: Union[a, b]  # type: ignore[valid-type]

    with pytest.raises(ValueError, match="260 options"):
        compile_schema(Home)
    # an optional union needs one more option for `unstated`
    c = member("C", 0, 127)
    d = member("D", 0, 128)

    class Tight(BaseModel):
        pet: Union[c, d]  # type: ignore[valid-type]

    assert len(compile_schema(Tight).unions[0].spec.key_map or {}) == 255

    class TooTight(BaseModel):
        pet: Optional[Union[c, d]] = None  # type: ignore[valid-type]

    with pytest.raises(ValueError, match="256 options"):
        compile_schema(TooTight)
