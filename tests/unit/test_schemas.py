"""Tests for coercing schema forms into pydantic models."""

import sys
from collections.abc import Callable
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import NotRequired, Required, TypedDict

from trustcall_system_one._schemas import _parse_arg_docs, coerce_schema


class Person(BaseModel):
    name: str


def test_base_model_is_returned_unchanged() -> None:
    assert coerce_schema(Person) is Person


class Address(TypedDict):
    """A postal address."""

    street: str
    zip_code: NotRequired[str]
    unit: Annotated[int, Field(description="Apartment number", ge=1)]


class Partial(TypedDict, total=False):
    id: Required[int]
    note: str


def test_typed_dict_becomes_a_model_with_name_doc_and_fields() -> None:
    model = coerce_schema(Address)
    assert issubclass(model, BaseModel)
    assert model.__name__ == "Address"
    assert model.__doc__ == "A postal address."
    assert list(model.model_fields) == ["street", "zip_code", "unit"]
    assert model.model_fields["street"].is_required()
    assert not model.model_fields["zip_code"].is_required()
    assert model.model_fields["unit"].description == "Apartment number"
    instance: Any = model(street="x", unit=2)
    assert instance.zip_code is None
    with pytest.raises(ValidationError):
        model(street="x", unit=0)


def test_typed_dict_total_false_with_required_marker() -> None:
    model = coerce_schema(Partial)
    assert model.model_fields["id"].is_required()
    assert not model.model_fields["note"].is_required()


def test_typed_dict_conversion_is_cached() -> None:
    assert coerce_schema(Address) is coerce_schema(Address)


def greet(
    name: str,
    times: int = 1,
    shout: bool | None = None,
) -> str:
    """Greet someone.

    Args:
        name (str): Who to greet. Use their first
            name only.
        times: How many times.

    Returns:
        The greeting.
    """
    return f"{name}" * times


def test_callable_becomes_a_model_named_after_the_function() -> None:
    model = coerce_schema(greet)
    assert model.__name__ == "greet"
    assert model.__doc__ is not None
    assert model.__doc__.startswith("Greet someone.")
    fields = model.model_fields
    assert list(fields) == ["name", "times", "shout"]
    assert fields["name"].is_required()
    assert fields["times"].default == 1
    assert fields["shout"].default is None
    assert fields["name"].description == "Who to greet. Use their first name only."
    assert fields["times"].description == "How many times."
    assert fields["shout"].description is None


def test_callable_conversion_is_cached() -> None:
    assert coerce_schema(greet) is coerce_schema(greet)


def test_callable_without_annotation_is_rejected() -> None:
    def bad(x, y: int) -> None:
        pass

    with pytest.raises(TypeError, match="'x' needs a type annotation"):
        coerce_schema(bad)


def test_callable_with_varargs_is_rejected() -> None:
    def bad(*args: int) -> None:
        pass

    with pytest.raises(TypeError, match="args"):
        coerce_schema(bad)


def test_arg_docs_parser_handles_missing_and_trailing_sections() -> None:
    assert _parse_arg_docs("No sections here.") == {}
    doc = (
        "Do it.\n\nArgs:\n    a: One.\n    b (int): Two\n        more.\n\n"
        "Returns:\n    x: no\n"
    )
    assert _parse_arg_docs(doc) == {"a": "One.", "b": "Two more."}


def test_json_schema_builds_a_model_named_by_title() -> None:
    pytest.importorskip("dydantic")
    model = coerce_schema(
        {
            "title": "Invoice",
            "description": "An invoice.",
            "type": "object",
            "properties": {
                "number": {"type": "string", "description": "Invoice number"},
                "total": {"type": "number"},
            },
            "required": ["number"],
        }
    )
    assert model.__name__ == "Invoice"
    assert model.__doc__ == "An invoice."
    assert model.model_fields["number"].is_required()
    assert not model.model_fields["total"].is_required()
    instance: Any = model(number="1")
    assert instance.total is None


def test_json_schema_is_not_mutated_and_equal_schemas_share_a_model() -> None:
    pytest.importorskip("dydantic")
    schema: dict[str, Any] = {
        "title": "Thing",
        "type": "object",
        "properties": {"a": {"type": "integer"}},
    }
    snapshot = repr(schema)
    first = coerce_schema(schema)
    assert repr(schema) == snapshot
    assert coerce_schema(dict(schema)) is first


def test_json_schema_must_describe_an_object() -> None:
    pytest.importorskip("dydantic")
    with pytest.raises(ValueError, match="object"):
        coerce_schema({"title": "X", "type": "array", "items": {"type": "string"}})


def test_json_schema_without_dydantic_gives_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "dydantic", None)
    with pytest.raises(ImportError, match=r"trustcall-system-one\[jsonschema\]"):
        coerce_schema({"title": "Unique", "type": "object", "properties": {}})


@pytest.mark.parametrize("bad", [42, "a string", None, [1, 2], Person(name="x")])
def test_unsupported_inputs_raise_type_error(bad: Any) -> None:
    with pytest.raises(TypeError):
        coerce_schema(bad)


def test_plain_class_is_rejected_with_guidance() -> None:
    class Plain:
        x: int

    factory: Callable[..., Any] = Plain
    with pytest.raises(TypeError, match="BaseModel"):
        coerce_schema(factory)


class Geo(TypedDict):
    lat: float


class Trip(TypedDict):
    start: Geo
    stops: list[Geo]
    maybe: Geo | None
    by_name: dict[str, Geo]
    tagged: Annotated[Geo, Field(description="Tagged")]
    pair: NotRequired[tuple[Geo, ...]]


def test_nested_typed_dicts_become_nested_models() -> None:
    model = coerce_schema(Trip)
    start = model.model_fields["start"].annotation
    assert isinstance(start, type) and issubclass(start, BaseModel)
    assert start.__name__ == "Geo"
    assert model.model_fields["tagged"].description == "Tagged"
    for name in ("stops", "maybe", "by_name", "pair"):
        text = repr(model.model_fields[name].annotation)
        assert "TypedDict" not in text and "Geo" in text
    instance: Any = model.model_validate(
        {
            "start": {"lat": 1},
            "stops": [{"lat": 2}],
            "maybe": None,
            "by_name": {"a": {"lat": 3}},
            "tagged": {"lat": 4},
        }
    )
    assert instance.stops[0].lat == 2.0
    assert instance.by_name["a"].lat == 3.0
