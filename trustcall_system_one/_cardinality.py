"""Cardinality wrappers: one entity of a schema, or many."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from trustcall_system_one._schemas import SchemaLike, coerce_schema

__all__ = ["Many", "One", "resolve_cardinality"]


def _identity_fields(identity: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if identity is None:
        return None
    names = (identity,) if isinstance(identity, str) else tuple(identity)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("identity must be a field name or a sequence of field names")
    return names


@dataclass(frozen=True)
class Many:
    """Extract any number of entities of a schema.

    Also usable as an ``Annotated`` marker for a nested list, where the schema
    is the list's item type and is left out: ``Annotated[list[Pet],
    Many(identity="name")]``.

    Attributes:
        schema: The schema (a pydantic model, TypedDict, JSON Schema or
            function), or ``None`` when used as a marker.
        identity: Field name, or names for a compound identity, that tell two
            entities apart. ``None`` lets the engine pick one.
        label: How an entity is called in prompts; defaults to the schema title.
        kind: What kind of thing an entity is, as a short noun phrase ("person",
            "customer order"), for the questions that decide whether something
            found in the text is one ("Is `entities.e0.label` a person?"). Defaults
            to the leading noun phrase of the schema's description ("A person in an
            address book" is a "person"), else the ``label``, else the schema title.
    """

    schema: SchemaLike | None = None
    identity: str | Sequence[str] | None = field(default=None, kw_only=True)
    label: str | None = field(default=None, kw_only=True)
    kind: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Normalise ``identity`` to a tuple of field names."""
        object.__setattr__(self, "identity", _identity_fields(self.identity))


@dataclass(frozen=True)
class One:
    """Extract exactly one entity of a schema.

    Also usable as an ``Annotated`` marker for a nested model.

    Attributes:
        schema: The schema, or ``None`` when used as a marker.
        label: How the entity is called in prompts; defaults to
            ``"the <schema title>"``.
    """

    schema: SchemaLike | None = None
    label: str | None = field(default=None, kw_only=True)


def _schema_name(item: SchemaLike) -> str:
    return coerce_schema(item).__name__


def _record_name(entry: Any) -> str | None:
    """Return the schema name an ``existing`` entry belongs to, if it names one."""
    if isinstance(entry, BaseModel):
        return type(entry).__name__
    if isinstance(entry, tuple):
        if len(entry) == 3 and isinstance(entry[1], str):
            return entry[1]
        if len(entry) == 2 and isinstance(entry[1], BaseModel):
            return type(entry[1]).__name__
    return None


def _records_of(name: str, existing: Any) -> tuple[int, bool]:
    """Count the records for ``name`` and whether they came as a list."""
    if isinstance(existing, Mapping):
        value = existing.get(name)
        if value is None:
            return 0, False
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value), False
        return 1, False
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        count = sum(1 for entry in existing if _record_name(entry) == name)
        return count, True
    return 0, False


def resolve_cardinality(
    item: SchemaLike | One | Many, existing: Any = None
) -> One | Many:
    """Decide whether a schema is extracted once or many times.

    The first rule that applies wins: an explicit ``One`` or ``Many`` wrapper;
    list-form ``existing`` with records of the schema implies ``Many``;
    dict-form ``{schema name: record}`` implies ``One``; otherwise ``One``.

    Args:
        item: A schema or a wrapper around one.
        existing: The ``existing`` argument of ``invoke`` (any of its shapes), or
            ``None``.

    Returns:
        The wrapper to run.

    Raises:
        ValueError: If a ``One`` wrapper meets several list-form records of its
            schema.
    """
    if isinstance(item, Many):
        return item
    if isinstance(item, One):
        if item.schema is not None and existing is not None:
            count, as_list = _records_of(_schema_name(item.schema), existing)
            if as_list and count > 1:
                raise ValueError(
                    f"One({_schema_name(item.schema)}) was given {count} existing "
                    "records; use Many(...) to extract several"
                )
        return item
    if existing is not None:
        count, as_list = _records_of(_schema_name(item), existing)
        if as_list and count >= 1:
            return Many(item)
    return One(item)
