"""Tests for the One and Many wrappers and cardinality resolution."""

import dataclasses

import pytest
from pydantic import BaseModel

from trustcall_system_one import Many, One
from trustcall_system_one._cardinality import resolve_cardinality


class Person(BaseModel):
    name: str


class Ticket(BaseModel):
    title: str


def test_many_and_one_are_frozen_wrappers() -> None:
    many = Many(Person, identity="name", label="a person")
    one = One(Person, label="the customer")
    assert (many.schema, many.identity, many.label) == (Person, ("name",), "a person")
    assert (one.schema, one.label) == (Person, "the customer")
    with pytest.raises(dataclasses.FrozenInstanceError):
        many.label = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        one.label = "x"  # type: ignore[misc]
    assert hash(many) == hash(Many(Person, identity="name", label="a person"))
    assert Many(Person) == Many(Person)
    assert Many(Person) != Many(Ticket)


def test_options_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        Many(Person, "name")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        One(Person, "the customer")  # type: ignore[call-arg]


def test_identity_forms() -> None:
    assert Many(Person).identity is None
    assert Many(Person, identity=("first", "last")).identity == ("first", "last")
    assert Many(Person, identity=["first", "last"]).identity == ("first", "last")
    for bad in ("", [], [""], [1]):
        with pytest.raises(ValueError, match="identity"):
            Many(Person, identity=bad)  # type: ignore[arg-type]


def test_wrappers_work_as_schema_less_markers() -> None:
    marker = Many(identity="name")
    assert marker.schema is None and marker.identity == ("name",)
    assert One().schema is None


def test_construction_does_nothing_with_the_schema() -> None:
    Many("not a schema")  # type: ignore[arg-type]  # only invoke time cares


def test_explicit_wrappers_win() -> None:
    many = Many(Person)
    one = One(Person)
    assert resolve_cardinality(many, None) is many
    assert resolve_cardinality(one, None) is one
    assert resolve_cardinality(many, {"Person": {"name": "x"}}) is many
    assert resolve_cardinality(one, [Person(name="a")]) is one


def test_default_is_one() -> None:
    assert resolve_cardinality(Person, None) == One(Person)
    assert resolve_cardinality(Person) == One(Person)


def test_dict_form_existing_implies_one() -> None:
    assert resolve_cardinality(Person, {"Person": {"name": "x"}}) == One(Person)
    assert resolve_cardinality(Person, {"Ticket": {"title": "t"}}) == One(Person)


def test_list_form_existing_with_records_of_the_schema_implies_many() -> None:
    assert resolve_cardinality(Person, [Person(name="a")]) == Many(Person)
    assert resolve_cardinality(Person, [("id1", Person(name="a"))]) == Many(Person)
    assert resolve_cardinality(Person, [("id1", "Person", {"name": "a"})]) == Many(
        Person
    )
    assert resolve_cardinality(Person, [Ticket(title="t")]) == One(Person)
    assert resolve_cardinality(Person, []) == One(Person)


def test_one_with_several_list_form_records_raises() -> None:
    records = [Person(name="a"), Person(name="b")]
    with pytest.raises(ValueError, match="Many"):
        resolve_cardinality(One(Person), records)
    assert resolve_cardinality(One(Person), [Person(name="a")]) == One(Person)
    assert resolve_cardinality(
        One(Person), [Person(name="a"), Ticket(title="t")]
    ) == One(Person)
