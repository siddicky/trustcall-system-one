"""Tests for flatten, unflatten and prune_nulls."""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import AliasPath, BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import NotRequired, TypedDict

from trustcall_system_one import compile_schema
from trustcall_system_one._flatten import flatten, prune_nulls, unflatten
from trustcall_system_one._names import Path


def test_flatten_uses_str_and_int_segments() -> None:
    doc = {"a": 1, "b": {"c": [10, {"d": None}], "e": "x"}}
    assert flatten(doc) == {
        ("a",): 1,
        ("b", "c", 0): 10,
        ("b", "c", 1, "d"): None,
        ("b", "e"): "x",
    }


def test_flatten_keeps_empty_containers_as_leaves() -> None:
    doc: dict[str, Any] = {"a": {}, "b": [], "c": [[], {}]}
    assert flatten(doc) == {("a",): {}, ("b",): [], ("c", 0): [], ("c", 1): {}}
    assert unflatten(flatten(doc)) == doc


def test_flatten_of_an_empty_document_is_empty() -> None:
    assert flatten({}) == {}
    assert unflatten({}) == {}


def test_flatten_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="strings"):
        flatten({"a": {1: "x"}})


def test_tuples_are_leaves() -> None:
    assert flatten({"a": (1, 2)}) == {("a",): (1, 2)}


def test_unflatten_rebuilds_dicts_and_lists() -> None:
    items: dict[Path, Any] = {
        ("a",): 1,
        ("b", "c", 0): 10,
        ("b", "c", 1, "d"): None,
    }
    assert unflatten(items) == {"a": 1, "b": {"c": [10, {"d": None}]}}


def test_unflatten_orders_list_items_by_index_not_insertion() -> None:
    items: dict[Path, Any] = {("t", 2): "c", ("t", 0): "a", ("t", 1): "b"}
    assert unflatten(items) == {"t": ["a", "b", "c"]}


def test_unflatten_compacts_list_holes_and_reports_them() -> None:
    warnings: list[str] = []
    result = unflatten({("t", 0): "a", ("t", 3): "d", ("t", 1): "b"}, warnings)
    assert result == {"t": ["a", "b", "d"]}
    assert len(warnings) == 1
    assert "t" in warnings[0]
    assert "[2]" in warnings[0]


def test_unflatten_without_holes_reports_nothing() -> None:
    warnings: list[str] = []
    unflatten({("t", 0): "a", ("t", 1): "b"}, warnings)
    assert warnings == []


def test_unflatten_holes_without_a_collector_do_not_raise() -> None:
    assert unflatten({("t", 5): "x"}) == {"t": ["x"]}


def test_unflatten_reports_a_hole_at_the_start() -> None:
    warnings: list[str] = []
    assert unflatten({("t", 1): "b"}, warnings) == {"t": ["b"]}
    assert "[0]" in warnings[0]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ({("a",): 1, ("a", "b"): 2}, "conflicting"),
        ({("a", 0): 1, ("a", "b"): 2}, "mixes"),
        ({("a", -1): 1}, "negative"),
        ({(): 1}, "empty"),
        ({(0,): 1}, "object"),
    ],
)
def test_unflatten_rejects_inconsistent_paths(
    items: dict[Path, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        unflatten(items)


def test_unflatten_rejects_odd_segments() -> None:
    with pytest.raises(TypeError):
        unflatten({(1.5,): 1})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        unflatten({("a", True): 1})


# Round trip. Every JSON-like value is allowed, including empty containers
# (they are leaves). Excluded: NaN (never equal to itself), tuples (not JSON
# arrays) and non-string dict keys (rejected by flatten).
_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
)
_values = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=6), children, max_size=4),
    ),
    max_leaves=25,
)


@given(doc=st.dictionaries(st.text(max_size=6), _values, max_size=5))
def test_unflatten_inverts_flatten(doc: dict[str, Any]) -> None:
    warnings: list[str] = []
    assert unflatten(flatten(doc), warnings) == doc
    assert warnings == []


class Pet(BaseModel):
    name: str


class Card(BaseModel):
    name: str
    nickname: str | None = None
    age: int | None
    city: str = "Paris"
    pet: Pet | None = None


def test_prune_nulls_drops_only_defaulted_none_values() -> None:
    plan = compile_schema(Card)
    doc = {"name": None, "nickname": None, "age": None, "city": None, "pet": None}
    assert prune_nulls(doc, plan) == {"name": None, "age": None}


def test_prune_nulls_keeps_values_and_unknown_keys() -> None:
    plan = compile_schema(Card)
    doc = {"name": "A", "nickname": "Al", "city": None, "extra": None}
    assert prune_nulls(doc, plan) == {"name": "A", "nickname": "Al", "extra": None}


def test_pruned_document_validates_with_defaults() -> None:
    plan = compile_schema(Card)
    pruned = prune_nulls({"name": "A", "age": None, "city": None}, plan)
    card = Card.model_validate(pruned)
    assert card.city == "Paris"
    assert card.age is None


class Item(TypedDict):
    id: int
    note: NotRequired[str]


def test_prune_nulls_drops_none_for_not_required_typed_dict_keys() -> None:
    plan = compile_schema(Item)
    assert prune_nulls({"id": 1, "note": None}, plan) == {"id": 1}


# --------------------------------------------------------------------------- #
# Aliases: documents must be accepted by model_validate
# --------------------------------------------------------------------------- #


class Aliased(BaseModel):
    full_name: str = Field(alias="fullName")
    nick: str | None = Field(default=None, alias="nickName")
    age: int = Field(default=3, validation_alias="ageYears")
    plain: str = "p"


def test_unflatten_with_a_plan_uses_the_validation_keys() -> None:
    plan = compile_schema(Aliased)
    doc = unflatten({("full_name",): "Ada", ("age",): 40, ("plain",): "x"}, plan=plan)
    assert doc == {"fullName": "Ada", "ageYears": 40, "plain": "x"}
    assert Aliased.model_validate(doc) == Aliased(
        fullName="Ada", ageYears=40, plain="x"
    )


def test_unflatten_without_a_plan_keeps_field_names() -> None:
    assert unflatten({("full_name",): "Ada"}) == {"full_name": "Ada"}


def test_aliased_documents_validate_without_populate_by_name() -> None:
    assert Aliased.model_config.get("populate_by_name") in (None, False)
    plan = compile_schema(Aliased)
    with pytest.raises(ValidationError):
        Aliased.model_validate({"full_name": "Ada"})
    doc = prune_nulls(
        unflatten({("full_name",): "Ada", ("nick",): None, ("age",): None}), plan
    )
    assert doc == {"fullName": "Ada"}
    assert Aliased.model_validate(doc).age == 3


def test_prune_nulls_accepts_alias_keys_too() -> None:
    plan = compile_schema(Aliased)
    doc = prune_nulls({"fullName": "Ada", "nickName": None, "ageYears": None}, plan)
    assert doc == {"fullName": "Ada"}


def test_populate_by_name_models_still_get_alias_keys() -> None:
    class Both(BaseModel):
        model_config = ConfigDict(populate_by_name=True)

        full_name: str = Field(alias="fullName")

    plan = compile_schema(Both)
    doc = unflatten({("full_name",): "Ada"}, plan=plan)
    assert doc == {"fullName": "Ada"}
    assert Both.model_validate(doc).full_name == "Ada"


def test_alias_path_falls_back_to_the_field_name() -> None:
    class Pathy(BaseModel):
        model_config = ConfigDict(populate_by_name=True)

        a: str = Field(validation_alias=AliasPath("x", "y"))

    plan = compile_schema(Pathy)
    assert unflatten({("a",): "v"}, plan=plan) == {"a": "v"}
    assert plan.spec(("a",)).warnings == ("alias_path",)
