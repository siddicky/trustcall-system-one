"""The ``existing`` argument: five input shapes, one normal form.

``coerce_existing`` accepts trustcall's shapes:

* ``{schema_name: record}``: at most one record per schema, whose id is the schema
  name;
* ``list[SchemaInstance]``;
* ``list[(record_id, schema_name, record)]``;
* ``list[(record_id, BaseModel)]``: the schema name is the model's class name;
* a bare ``list[BaseModel]``: the ids are minted (``uuid4``).

A record may be a dict or a pydantic model (dumped by alias). Records of a schema
the extractor does not know are handled by ``existing_schema_policy`` and the name
``"__any__"`` stands for a record that may belong to any schema.

``normalise_record`` rewrites a record's keys, level by level, to the keys the
schema validates (an alias where the field has one), so that diffs, patches and
validation all talk about the same keys.
"""

import logging
import uuid
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from trustcall_system_one._compile import FieldGroup, SchemaPlan, UnionGroup
from trustcall_system_one._types import SchemaInstance

__all__ = [
    "ANY",
    "ExistingRecord",
    "ExistingSet",
    "coerce_existing",
    "fits",
    "normalise_record",
    "record_json",
    "target_for",
]

logger = logging.getLogger(__name__)

ANY = "__any__"
"""The schema name of a record that may belong to any schema."""

Policy = bool | Literal["ignore"]


@dataclass(frozen=True)
class ExistingRecord:
    """One existing record in normal form.

    Attributes:
        record_id: The caller's id for it (the schema name for the dict form).
        schema_name: The schema it belongs to, or ``"__any__"``.
        data: The record as a JSON-compatible dict, keys as the caller gave them
            (see ``normalise_record``).
        listed: Whether it came in list form (the dict form holds one record
            per schema and implies ``One``; list-form records of a schema imply
            ``Many``).
    """

    record_id: str
    schema_name: str
    data: dict[str, Any] = field(hash=False)
    listed: bool = False


@dataclass(frozen=True)
class ExistingSet:
    """The coerced ``existing`` argument.

    Attributes:
        records: Records of a known schema, or ``"__any__"``, in input order.
        opaque: Records of an unknown schema kept by
            ``existing_schema_policy=False``: context only, never patched.
        warnings: Notes for ``ExtractionResult.warnings``.
    """

    records: tuple[ExistingRecord, ...] = ()
    opaque: tuple[ExistingRecord, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        """Every record id (known, ``__any__`` and opaque), in input order."""
        return tuple(r.record_id for r in (*self.records, *self.opaque))

    def named(self, schema_name: str) -> list[ExistingRecord]:
        """The records that name this schema."""
        return [r for r in self.records if r.schema_name == schema_name]

    def without_unclaimed_any(
        self, schema_name: str, claims: Mapping[str, str]
    ) -> "ExistingSet":
        """The records as one schema sees them.

        An ``"__any__"`` record is seen only by the schema that claimed it (see
        ``Extractor._claim_any``); the other schemas see no such record.
        """
        kept = tuple(
            r
            for r in self.records
            if r.schema_name != ANY or claims.get(r.record_id) == schema_name
        )
        return ExistingSet(kept, self.opaque, self.warnings)

    def untouched(self, handled: Collection[tuple[str, str]]) -> tuple[str, ...]:
        """Ids of the records that were not dealt with, in input order.

        Args:
            handled: ``(schema name, record id)`` of the records that got a
                result (the schema name is ``"__any__"`` for such a record).
        """
        return tuple(
            r.record_id
            for r in (*self.records, *self.opaque)
            if (r.schema_name, r.record_id) not in handled
        )

    def for_schema(self, schema_name: str) -> list[ExistingRecord]:
        """The records a schema may update: its own, else the ``__any__`` ones."""
        named = self.named(schema_name)
        return named or self.named(ANY)


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #


def record_json(record: Any, *, where: str) -> dict[str, Any]:
    """Turn a dict or a pydantic model into a JSON-compatible dict.

    A model is dumped by alias; values that JSON cannot hold (dates, decimals,
    enums, sets) are converted the way pydantic's JSON mode does.

    Raises:
        ValueError: If ``record`` is neither, or cannot be converted.
    """
    try:
        if isinstance(record, BaseModel):
            data: Any = record.model_dump(mode="json", by_alias=True)
        elif isinstance(record, Mapping):
            data = to_jsonable_python(dict(record))
        else:
            raise ValueError(
                f"{where}: a record must be a dict or a pydantic model, got "
                f"{type(record).__name__}"
            )
    except (PydanticSerializationError, TypeError) as exc:
        raise ValueError(f"{where}: the record cannot be converted: {exc}") from exc
    if not all(isinstance(key, str) for key in data):
        raise ValueError(f"{where}: the record's keys must be strings")
    return dict(data)


def _record_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{where}: the record id must be a non-empty string, got {value!r}"
        )
    return value


def _schema_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: the schema name must be a string, got {value!r}")
    return value


class _Builder:
    """Collects records and applies ``existing_schema_policy`` to each."""

    def __init__(self, known: Collection[str], policy: Policy) -> None:
        self.known = frozenset(known)
        self.policy = policy
        self.records: list[ExistingRecord] = []
        self.opaque: list[ExistingRecord] = []
        self.warnings: list[str] = []

    def add(
        self, record_id: str, schema_name: str, data: dict[str, Any], *, listed: bool
    ) -> None:
        item = ExistingRecord(record_id, schema_name, data, listed)
        if schema_name in self.known or schema_name == ANY:
            self.records.append(item)
            return
        where = f"record {record_id!r} (schema {schema_name!r})"
        if self.policy is True:
            raise ValueError(
                f"{where} doesn't match any schema. Known schemas: "
                f"{sorted(self.known)} (or {ANY!r}); set existing_schema_policy "
                "to False to keep such records as context or to 'ignore' to "
                "drop them"
            )
        if self.policy is False:
            message = (
                f"{where} does not match any schema; it is kept as context only "
                "and is never patched"
            )
            logger.warning(message)
            self.warnings.append(message)
            self.opaque.append(item)
            return
        logger.debug("ignoring %s: unknown schema", where)

    def done(self) -> ExistingSet:
        seen: set[tuple[str, str]] = set()
        for r in (*self.records, *self.opaque):
            key = (r.schema_name, r.record_id)
            if key in seen:
                raise ValueError(
                    f"existing has the record id {r.record_id!r} twice for "
                    f"schema {r.schema_name!r}"
                )
            seen.add(key)
        return ExistingSet(
            tuple(self.records), tuple(self.opaque), tuple(self.warnings)
        )


def _from_mapping(existing: Mapping[Any, Any], build: _Builder) -> None:
    for key, record in existing.items():
        name = _schema_name(key, "existing")
        if isinstance(record, (list, tuple)):
            raise ValueError(
                f"existing[{name!r}] is a list: the dict form holds one record "
                "per schema; pass a list of (record_id, schema_name, record) "
                "tuples for several"
            )
        data = record_json(record, where=f"existing[{name!r}]")
        build.add(name, name, data, listed=False)


def _from_item(index: int, item: Any, build: _Builder) -> None:
    where = f"existing[{index}]"
    if isinstance(item, SchemaInstance) or (isinstance(item, tuple) and len(item) == 3):
        record_id, schema_name, record = item
        build.add(
            _record_id(record_id, where),
            _schema_name(schema_name, where),
            record_json(record, where=where),
            listed=True,
        )
    elif isinstance(item, tuple) and len(item) == 2:
        record_id, model = item
        if not isinstance(model, BaseModel):
            raise ValueError(
                f"{where}: a (record_id, record) pair needs a pydantic model, "
                f"got {type(model).__name__}; use (record_id, schema_name, "
                "record) for a dict"
            )
        build.add(
            _record_id(record_id, where),
            type(model).__name__,
            record_json(model, where=where),
            listed=True,
        )
    elif isinstance(item, BaseModel):
        build.add(
            str(uuid.uuid4()),
            type(item).__name__,
            record_json(item, where=where),
            listed=True,
        )
    else:
        raise ValueError(
            f"Invalid item at index {index} in existing. Provided: {item!r}. "
            "Expected a SchemaInstance, a (record_id, schema_name, record) "
            "tuple, a (record_id, BaseModel) pair or a BaseModel"
        )


def coerce_existing(
    existing: Any, known: Collection[str], policy: Policy = True
) -> ExistingSet:
    """Bring ``existing`` (any of its five shapes) to one normal form.

    Args:
        existing: The ``existing`` argument of ``invoke``; ``None`` or an empty
            container means no records.
        known: Names of the extractor's schemas.
        policy: ``existing_schema_policy``: ``True`` raises for a record of an
            unknown schema, ``False`` keeps it as an opaque record (context
            only, never patched, with a warning) and ``"ignore"`` drops it.

    Returns:
        The records, with ids (minted for a bare model).

    Raises:
        ValueError: For an unknown schema under ``policy=True``, an item or
            container of the wrong type, a record that is not a dict or model,
            or a repeated id within one schema.
    """
    build = _Builder(known, policy)
    if existing is None:
        return build.done()
    if isinstance(existing, Mapping):
        _from_mapping(existing, build)
    elif isinstance(existing, (list, tuple)):
        for index, item in enumerate(existing):
            _from_item(index, item, build)
    else:
        raise ValueError(
            f"Invalid type for existing. Provided: {type(existing).__name__}, "
            "expected a dict or a list. Supported shapes:\n"
            "1. {schema_name: record}\n"
            "2. list[SchemaInstance]\n"
            "3. list[(record_id, schema_name, record)]\n"
            "4. list[(record_id, BaseModel)]\n"
            "5. list[BaseModel]"
        )
    return build.done()


def fits(plan: SchemaPlan, record: ExistingRecord) -> bool:
    """Whether the record, with its keys normalised, validates as the schema."""
    try:
        plan.schema.model_validate(normalise_record(plan, record.data))
    except ValidationError:
        return False
    return True


def target_for(schema_name: str, existing: ExistingSet) -> ExistingRecord | None:
    """The record a ``One`` schema updates, if there is one.

    Args:
        schema_name: The schema.
        existing: The coerced ``existing``.

    Returns:
        The record, or ``None`` when ``existing`` has none for the schema (the
        schema is then extracted as an insert).

    Raises:
        ValueError: For several records of a ``One`` schema (list-form records of
            a schema without a wrapper imply ``Many`` and never get here).
    """
    records = existing.for_schema(schema_name)
    if len(records) > 1:
        raise ValueError(
            f"One({schema_name}) was given {len(records)} existing records; "
            "use Many(...) to extract several"
        )
    return records[0] if records else None


# --------------------------------------------------------------------------- #
# Key normalisation
# --------------------------------------------------------------------------- #


def _names(owner: Iterable[Any]) -> dict[str, str]:
    """Map every key a field answers to (its name, its alias) to the validation key."""
    mapping: dict[str, str] = {}
    for item in owner:
        name = str(item.path[-1])
        key = item.alias if item.alias is not None else name
        mapping[name] = key
        mapping[key] = key
    return mapping


def _normalise_group(group: FieldGroup, data: Mapping[str, Any]) -> dict[str, Any]:
    names = _names(group.specs)
    children = {
        (c.alias if c.alias is not None else str(c.path[-1])): c for c in group.children
    }
    unions = {
        (u.alias if u.alias is not None else str(u.path[-1])): u for u in group.unions
    }
    for child in group.children:
        names.update(_names([child]))
    for union in group.unions:
        names.update(_names([union]))
    out: dict[str, Any] = {}
    # exact validation keys first: they win over a field name that maps to them
    for pass_exact in (True, False):
        for key, value in data.items():
            target = names.get(key, key)
            if (key == target) != pass_exact or (not pass_exact and target in out):
                continue
            sub_group = children.get(target)
            sub_union = unions.get(target)
            if sub_group is not None and isinstance(value, Mapping):
                value = _normalise_group(sub_group, value)
            elif sub_union is not None and isinstance(value, Mapping):
                value = _normalise_union(sub_union, value)
            out[target] = value
    return out


def _normalise_union(union: UnionGroup, data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a union value as the member it validates as (else the best fit)."""
    best: dict[str, Any] | None = None
    best_hits = -1
    for member in union.members:
        candidate = _normalise_group(member.group, data)
        try:
            member.model.model_validate(candidate)
        except ValidationError:
            known = {
                (s.alias if s.alias is not None else str(s.path[-1]))
                for s in member.group.specs
            }
            hits = sum(key in known for key in candidate)
            if hits > best_hits:
                best, best_hits = candidate, hits
        else:
            return candidate
    return best if best is not None else dict(data)


def normalise_record(plan: SchemaPlan, data: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a record's keys to the keys the schema validates.

    A field is accepted under its name or its alias and written under the key
    ``model_validate`` expects (the alias, when it has one), at every level of
    sub-models and unions of models. Keys the schema does not know are kept
    as they are, and collections (``list[Model]``) are left alone until M9.

    Args:
        plan: The compiled schema.
        data: The record.

    Returns:
        A new dict.
    """
    return _normalise_group(plan.root, data)
