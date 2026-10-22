"""Nested collections: what a list or mapping field of a record turns into.

A ``list[Model]`` or ``dict[str, X]`` field is a collection of entities of its
own (``SchemaPlan.collection``). The collection engine finds and extracts them
exactly as it does at the top level, scoped to the parent entity; this module
turns what it produced into the parent's terms:

* an insert puts the items into the parent's document (a list in first-mention
  order; a mapping keyed by the extracted key) and lifts the items' answers and
  rejections to paths of the parent (``pets.0.name``, ``/pets/0/name``);
* an update turns the item records into operations inside the parent's patch
  list (``replace /items/1/quantity``, ``add /items/-``, ``remove /items/2``).

A mapping entry is an item whose document is ``{"key": k, "value": v}``; its
operations are rebased onto the entry (``/value/qty`` of the entry ``k`` is
``/<field>/k/qty``) and the key itself is never renamed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from trustcall_system_one._compile import (
    KEY_FIELD,
    VALUE_FIELD,
    CollectionPlan,
    FieldSpec,
)
from trustcall_system_one._diff import Change, pointer
from trustcall_system_one._names import Path
from trustcall_system_one._types import (
    FieldAnswer,
    RejectedOp,
    ResultRecord,
    SkippedEntity,
)
from trustcall_system_one._verify import Ctx, Nest

__all__ = [
    "COLLECTIONS",
    "NestedInsert",
    "NestedUpdate",
    "doc_key",
    "guard",
    "nest_for",
    "nested_update",
]

COLLECTIONS = ("collection", "mapping")
"""Engines of the fields that hold a collection of entities."""


def doc_key(spec: FieldSpec) -> str:
    """The key ``model_validate`` expects for a field."""
    return spec.alias if spec.alias is not None else str(spec.path[-1])


def guard(ctx: Ctx, spec: FieldSpec, plan: CollectionPlan) -> str:
    """Why a collection field is not extracted, or ``""``.

    ``limits.max_depth`` is the longest leaf path of the whole document and
    ``limits.max_recursion`` how many times an item model may reappear on its own
    chain of ancestors; past either the collection is left alone (and a warning
    says so, once), so a model that contains a list of itself stops unrolling.

    Args:
        ctx: The context of the entity that holds the field.
        spec: The collection or mapping field.
        plan: Its item plan.

    Returns:
        ``"max_depth"``, ``"max_recursion"`` or the empty string.
    """
    limits = ctx.run.limits
    if ctx.nest.depth + len(spec.path) + 1 > limits.max_depth:
        return "max_depth"
    chain = ctx.nest.chain or (ctx.plan.schema,)
    if chain.count(plan.item.schema) > limits.max_recursion:
        return "max_recursion"
    return ""


def guard_warning(ctx: Ctx, spec: FieldSpec, plan: CollectionPlan, why: str) -> str:
    """The warning for a collection that a limit keeps from being extracted."""
    limits = ctx.run.limits
    where = f"{ctx.plan.name}.{spec.pretty}"
    if why == "max_depth":
        return (
            f"{where}: max_depth: the collection is nested deeper than "
            f"limits.max_depth={limits.max_depth}; it is not extracted"
        )
    return (
        f"{where}: max_recursion: {plan.item.name} reappears on its own ancestor "
        f"chain more than limits.max_recursion={limits.max_recursion} times; the "
        "collection is not extracted"
    )


def nest_for(ctx: Ctx, spec: FieldSpec, plan: CollectionPlan) -> Nest:
    """Where the item plan of a collection field sits in the whole document."""
    chain = ctx.nest.chain or (ctx.plan.schema,)
    return Nest(ctx.nest.depth + len(spec.path), (*chain, plan.item.schema))


def _dotted(path: Path) -> str:
    return ".".join(str(part) for part in path)


def lift_answer(
    answer: FieldAnswer, prefix: Path, *, entry: bool
) -> FieldAnswer | None:
    """An item's answer as an answer of the parent record.

    The path gets the collection's path and the item's position (or key) in front
    (``("pets", 0, "name")``); for a mapping entry the ``value`` segment goes (the
    entry's value *is* the field below the key) and the key's own answer is left
    out.

    Returns:
        The answer, or ``None`` for a mapping entry's key.
    """
    path = answer.path
    if entry:
        if path[:1] == (KEY_FIELD,):
            return None
        if path[:1] == (VALUE_FIELD,):
            path = path[1:]
    full = (*prefix, *path)
    return replace(answer, path=full, pretty=_dotted(full))


def lift_pointer(path: str, prefix: str, *, entry: bool) -> str | None:
    """An item's JSON Pointer as a pointer into the parent's document.

    Returns:
        The pointer, or ``None`` when it addresses a mapping entry's key (a key is
        never renamed).
    """
    if not entry:
        return prefix + path
    if path == "/" + KEY_FIELD or path.startswith(f"/{KEY_FIELD}/"):
        return None
    if path == "/" + VALUE_FIELD:
        return prefix
    if path.startswith(f"/{VALUE_FIELD}/"):
        return prefix + path[len(VALUE_FIELD) + 1 :]
    return prefix + path


def _item_value(plan: CollectionPlan, record: ResultRecord) -> Any:
    """What an item contributes to its parent's document (a model or a plain dict)."""
    value = record.value
    if value is None:
        return None
    if plan.is_dict:
        return getattr(value, VALUE_FIELD, None)
    if plan.typed_dict:  # a NotRequired key the item lacks stays out of the dict
        return value.model_dump(by_alias=True, exclude_defaults=True)
    return value


def _jsonable(value: Any) -> Any:
    """A new item as a JSON document: aliases for keys, defaults left out."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_defaults=True)
    return to_jsonable_python(value, by_alias=True)


@dataclass
class NestedInsert:
    """The items an insert found for one collection field of an entity.

    Attributes:
        spec: The collection or mapping field.
        plan: Its item plan.
        base: The document keys down to the group that holds the field.
        entries: The valid items as ``(key, value)`` in first-mention order: the
            key is the extracted key of a mapping entry (``None`` in a list), the
            value what goes into the parent's document.
        answers: The items' answers lifted to the parent's paths.
        rejected: The items' rejections lifted to the parent's pointers, and one
            ``invalid_item`` for each item that failed validation.
        skipped: The entities that were found and not used.
        extracted: Whether any extraction request was made.
        attempts: The most extraction rounds an item needed.
    """

    spec: FieldSpec
    plan: CollectionPlan
    base: tuple[str, ...]
    entries: list[tuple[str | None, Any]] = field(default_factory=list)
    answers: dict[str, FieldAnswer] = field(default_factory=dict)
    rejected: list[RejectedOp] = field(default_factory=list)
    skipped: tuple[SkippedEntity, ...] = ()
    extracted: bool = False
    attempts: int = 0

    @property
    def pointer(self) -> str:
        """JSON Pointer of the field in the parent's document."""
        return pointer((*self.base, doc_key(self.spec)))

    def document(self) -> Any:
        """The field's value: a list, or a dict keyed by the entries' keys."""
        if self.plan.is_dict:
            return {key: value for key, value in self.entries if key is not None}
        return [value for _, value in self.entries]

    @classmethod
    def build(
        cls,
        spec: FieldSpec,
        plan: CollectionPlan,
        base: tuple[str, ...],
        records: Sequence[ResultRecord],
        skipped: Sequence[SkippedEntity],
        *,
        extracted: bool,
        attempts: int,
    ) -> "NestedInsert":
        """Collect the item records of an insert into the parent's terms."""
        out = cls(spec, plan, base, skipped=tuple(skipped))
        out.extracted, out.attempts = extracted, attempts
        for record in records:
            if not record.valid or record.value is None:
                label = record.entity.label if record.entity else "an item"
                out.rejected.append(
                    RejectedOp(
                        None,
                        out.pointer,
                        "invalid_item",
                        f"{label}: " + "; ".join(record.errors),
                    )
                )
                continue
            value = _item_value(plan, record)
            key: str | None = None
            if plan.is_dict:
                raw = getattr(record.value, KEY_FIELD, None)
                key = raw if isinstance(raw, str) and raw.strip() else None
                if key is None or any(key == k for k, _ in out.entries):
                    out.rejected.append(
                        RejectedOp(
                            None,
                            out.pointer,
                            "duplicate" if key else "invalid_item",
                            f"the entry {key!r} has no usable key or repeats one",
                        )
                    )
                    continue
            where = key if key is not None else len(out.entries)
            out.entries.append((key, value))
            out.add_record(record, where)
        return out

    def add_record(self, record: ResultRecord, where: str | int) -> None:
        """Lift an item's answers and rejections to the parent's paths."""
        prefix = (*self.spec.path, where)
        at = f"{self.pointer}{pointer([where])}"
        for answer in record.fields.values():
            lifted = lift_answer(answer, prefix, entry=self.plan.is_dict)
            if lifted is not None:
                self.answers[lifted.pretty] = lifted
        for rejected in record.rejected:
            path = lift_pointer(rejected.path, at, entry=self.plan.is_dict)
            if path is not None:
                self.rejected.append(replace(rejected, path=path))


@dataclass
class NestedUpdate:
    """The operations inside a parent's patch list that its nested items need.

    Attributes:
        proposed: Operations that still need the parent's near-duplicate check
            and change gate (``add`` of a new item).
        fixed: Operations already judged by the item's own update (its field
            changes, gated against the item's record and attributed to the item)
            or decided by the removal question; the parent applies them as they
            are.
        answers: The items' answers lifted to the parent's paths.
        rejected: Their rejections lifted to the parent's pointers.
        skipped: The entities that were found and not used.
        lists: Pointers of the list fields that were asked (for ``apply_ops``).
        extracted: Whether any extraction request was made.
        attempts: The most extraction rounds an item needed.
    """

    proposed: list[Change] = field(default_factory=list)
    fixed: list[Change] = field(default_factory=list)
    answers: dict[str, FieldAnswer] = field(default_factory=dict)
    rejected: list[RejectedOp] = field(default_factory=list)
    skipped: list[SkippedEntity] = field(default_factory=list)
    lists: set[str] = field(default_factory=set)
    extracted: bool = False
    attempts: int = 0

    @property
    def changes(self) -> list[Change]:
        """Every operation, the proposed ones first."""
        return [*self.proposed, *self.fixed]

    def merge(self, other: "NestedUpdate") -> None:
        """Add what another collection produced."""
        self.proposed.extend(other.proposed)
        self.fixed.extend(other.fixed)
        self.answers.update(other.answers)
        self.rejected.extend(other.rejected)
        self.skipped.extend(other.skipped)
        self.lists |= other.lists
        self.extracted = self.extracted or other.extracted
        self.attempts = max(self.attempts, other.attempts)


def _changed(op: str, path: str, value: Any, field_path: str) -> Change:
    kind: Any = op
    return Change(op=kind, path=path, value=value, field_path=field_path)


def nested_update(
    spec: FieldSpec,
    plan: CollectionPlan,
    base: tuple[str, ...],
    records: Sequence[ResultRecord],
    skipped: Sequence[SkippedEntity],
    *,
    existing: int,
    keys: Sequence[str],
    extracted: bool,
    attempts: int,
) -> NestedUpdate:
    """Turn the item records of an update into operations of the parent.

    Args:
        spec: The collection or mapping field.
        plan: Its item plan.
        base: The document keys down to the group that holds the field.
        records: What the collection engine produced: an ``update`` record for
            each item matched to an existing one (``json_doc_id`` its index, or
            its key in a mapping), an ``insert`` for each new item and a
            ``delete`` for each one to remove.
        skipped: The entities that were found and not used.
        existing: How many items the parent record holds (a new list item is
            appended after them).
        keys: The keys of the entries the record holds (mappings).
        extracted: Whether any extraction request was made.
        attempts: The most extraction rounds an item needed.

    Returns:
        The operations (the parent judges the ``proposed`` ones), the lifted
        answers and rejections and the skipped entities.
    """
    out = NestedUpdate(skipped=list(skipped), extracted=extracted, attempts=attempts)
    holder = pointer((*base, doc_key(spec)))
    if not plan.is_dict:
        out.lists.add(holder)
    field_path = spec.pretty
    appended = 0
    for record in records:
        if record.kind == "update":
            ident = record.json_doc_id or ""
            where: str | int = ident if plan.is_dict else int(ident)
            at = f"{holder}{pointer([where])}"
            prefix = (*spec.path, where)
            for answer in record.fields.values():
                lifted = lift_answer(answer, prefix, entry=plan.is_dict)
                if lifted is not None:
                    out.answers[lifted.pretty] = lifted
            for rejected in record.rejected:
                path = lift_pointer(rejected.path, at, entry=plan.is_dict)
                if path is not None:
                    out.rejected.append(replace(rejected, path=path))
            if not record.valid:
                out.rejected.append(
                    RejectedOp(
                        None,
                        at,
                        "invalid_item",
                        "the item is invalid after its update: "
                        + "; ".join(record.errors),
                    )
                )
                continue
            for op in record.patches:
                path = lift_pointer(op.path, at, entry=plan.is_dict)
                if path is None:
                    out.rejected.append(
                        RejectedOp(
                            op.op,
                            at,
                            "key_fixed",
                            "the key of a mapping entry is its identity and is "
                            "not renamed",
                        )
                    )
                    continue
                out.fixed.append(_changed(op.op, path, op.value, field_path))
        elif record.kind == "insert":
            if not record.valid or record.value is None:
                label = record.entity.label if record.entity else "an item"
                out.rejected.append(
                    RejectedOp(
                        None,
                        holder,
                        "invalid_item",
                        f"{label}: " + "; ".join(record.errors),
                    )
                )
                continue
            value = _jsonable(_item_value(plan, record))
            if plan.is_dict:
                raw = getattr(record.value, KEY_FIELD, None)
                key = raw.strip() if isinstance(raw, str) else ""
                if not key or key in keys:
                    out.rejected.append(
                        RejectedOp(
                            "add",
                            f"{holder}{pointer([key])}",
                            "duplicate" if key else "invalid_item",
                            f"the entry {key!r} is already in the record or has no key",
                        )
                    )
                    continue
                where = key
                at = f"{holder}{pointer([key])}"
            else:
                where = existing + appended
                at = f"{holder}/-"
            prefix = (*spec.path, where)
            names: list[str] = []
            for answer in record.fields.values():
                lifted = lift_answer(answer, prefix, entry=plan.is_dict)
                if lifted is not None:
                    out.answers[lifted.pretty] = lifted
                    names.append(lifted.pretty)
            spans = tuple(
                m.span
                for m in (record.entity.mentions if record.entity else ())
                if m.span
            )
            change = _changed("add", at, value, field_path)
            change.fields = tuple(names)
            change.spans = spans
            out.proposed.append(change)
            for rejected in record.rejected:
                shown = at if plan.is_dict else f"{holder}/{where}"
                path = lift_pointer(rejected.path, shown, entry=plan.is_dict)
                if path is not None:
                    out.rejected.append(replace(rejected, path=path))
            appended += 0 if plan.is_dict else 1
        elif record.kind == "delete":
            ident = record.json_doc_id or ""
            place = f"{holder}{pointer([ident])}"
            out.fixed.append(_changed("remove", place, None, field_path))
    return out
