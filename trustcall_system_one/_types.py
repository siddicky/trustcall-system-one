"""Result and accounting types shared across the package."""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import BaseModel

from trustcall_system_one._names import Path
from trustcall_system_one._wire import ClairicUsage, ClefUsage, normalize_model_id

if TYPE_CHECKING:
    from trustcall_system_one._config import CostReport

__all__ = [
    "UNSTATED",
    "CallKind",
    "CallRecord",
    "EntityInfo",
    "ExtractionResult",
    "FieldAnswer",
    "Mention",
    "PatchOp",
    "RejectedOp",
    "ResultRecord",
    "SchemaInstance",
    "SkippedEntity",
    "Usage",
    "json_pointer",
]


class _Unstated:
    """The type of ``UNSTATED``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSTATED"

    def __copy__(self) -> "_Unstated":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "_Unstated":
        return self

    def __reduce__(self) -> str:
        return "UNSTATED"


UNSTATED = _Unstated()
"""Sentinel for a Clef choice that says the input does not state the value.

It is never a field value: for a nullable field it stands for ``None`` and for a
field with a default it means "leave the default".
"""

CallKind = Literal[
    "presence",
    "discover",
    "count",
    "coref",
    "qualify",
    "match",
    "delete",
    "extract",
    "crosscheck",
    "clef_fields",
    "attribution",
    "reask",
    "null_check",
    "stated",
    "adjudicate",
    "list_dedup",
    "sentences",
    "list_kind",
    "novelty",
    "record_presence",
    "gate",
    "relative_edit",
    "repair",
    "raw",
]

_TOKEN_FIELDS = (
    "state_tokens",
    "image_tokens",
    "schema_tokens",
    "input_tokens",
    "output_tokens",
    "passes",
    "requests",
)


@dataclass
class Usage:
    """Token and request totals, with breakdowns by model and by call kind.

    The nested ``Usage`` values in ``by_model`` and ``by_kind`` are leaf totals
    and have empty breakdowns of their own.
    """

    state_tokens: int = 0
    image_tokens: int = 0
    schema_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    passes: int = 0
    requests: int = 0
    by_model: dict[str, "Usage"] = dataclasses.field(default_factory=dict)
    by_kind: dict[str, "Usage"] = dataclasses.field(default_factory=dict)

    def _add_totals(self, other: "Usage") -> None:
        for name in _TOKEN_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def add_call(self, model: str, kind: str, usage: ClairicUsage | ClefUsage) -> None:
        """Record one successful call.

        Args:
            model: Model id the call went to; bare names are stored as the full
                ``@cf/...`` id.
            kind: The ``CallMeta.kind`` of the call.
            usage: The ``usage`` object parsed from the response.
        """
        call = Usage(
            state_tokens=getattr(usage, "state_tokens", None) or 0,
            image_tokens=getattr(usage, "image_tokens", None) or 0,
            schema_tokens=getattr(usage, "schema_tokens", None) or 0,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            passes=getattr(usage, "passes", None) or 0,
            requests=1,
        )
        self._add_totals(call)
        self.by_model.setdefault(normalize_model_id(model), Usage())._add_totals(call)
        self.by_kind.setdefault(kind, Usage())._add_totals(call)

    def merge(self, other: "Usage") -> None:
        """Add ``other`` (totals and breakdowns) into this instance in place."""
        self._add_totals(other)
        for mine, theirs in (
            (self.by_model, other.by_model),
            (self.by_kind, other.by_kind),
        ):
            for key, value in theirs.items():
                mine.setdefault(key, Usage())._add_totals(value)

    def __add__(self, other: "Usage") -> "Usage":
        """Return the sum of two usages without mutating either."""
        if not isinstance(other, Usage):
            return NotImplemented
        total = Usage()
        total.merge(self)
        total.merge(other)
        return total


@dataclass(frozen=True)
class CallRecord:
    """One transport call, as logged on a result.

    ``error`` is the failure message for a call that raised (``None`` on
    success) and ``status_code`` the HTTP status when one was received.
    """

    model: str
    kind: CallKind
    schema: str = ""
    entity: str | None = None
    field: str | None = None
    scope: str = ""
    n_items: int = 0
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    duration_ms: float = 0.0
    retries: int = 0
    error: str | None = None
    status_code: int | None = None


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

Span = tuple[int, int]
"""``(start, end)`` offsets into the original input text."""


def json_pointer(path: Path) -> str:
    """Render a path as an RFC 6901 JSON Pointer (``("a", 0)`` -> ``"/a/0"``)."""
    return "".join(
        "/" + str(part).replace("~", "~0").replace("/", "~1") for part in path
    )


class SchemaInstance(NamedTuple):
    """An existing record with its id and the name of its schema.

    This is the element of the list form of ``existing`` (a plain
    ``(record_id, schema_name, record)`` tuple is accepted as well).

    Attributes:
        record_id: The id the caller knows the record by (it comes back as
            ``ResultRecord.json_doc_id``).
        schema_name: Name of the schema, or ``"__any__"`` for a record that may
            belong to any schema of the extractor.
        record: The record, as a dict or a pydantic model.
    """

    record_id: str
    schema_name: str
    record: "dict[str, Any] | BaseModel"


@dataclass(frozen=True)
class Mention:
    """A piece of the input that refers to an entity.

    Attributes:
        text: The mention as written.
        span: ``(start, end)`` in the original input, when known.
    """

    text: str
    span: Span | None = None


@dataclass(frozen=True)
class EntityInfo:
    """What the engine knows about the entity a record describes.

    Attributes:
        key: Identifier of the entity within its schema (``"e0"`` for a ``One``).
        label: How the entity was named in prompts (``"the Person"``).
        mentions: Every mention found for it (empty for a ``One`` schema).
        first_mention: The mention that orders entities.
        qualify_p: Probability from the presence or qualification check, when
            one ran.
        match_probs: Probability of each existing record as the match, by record
            id, and ``"__new__"`` for "none of them" (a ``Many`` schema with
            existing records; empty otherwise).
        parent: Key of the parent entity for nested items.
    """

    key: str
    label: str
    mentions: tuple[Mention, ...] = ()
    first_mention: Mention | None = None
    qualify_p: float | None = None
    match_probs: Mapping[str, float] = field(default_factory=dict, hash=False)
    parent: str | None = None


@dataclass(frozen=True)
class FieldAnswer:
    """Everything known about how one field got its value.

    Attributes:
        path: Position of the field in the document.
        pretty: ``path`` joined with dots.
        wire_name: The Clairic field or Clef question name that was sent.
        engine: ``"clairic"``, ``"clef"`` or ``"const"``.
        value: The final Python value (``None`` for a null, a rejected answer
            and a field left to its default).
        raw: The model's raw answer: the Clairic ``value``, or a dict with the
            Clef answer's fields (``noul``, or ``choice`` and its odds).
        p_null: Clairic's probability that the value is null.
        confidence: The model's confidence in the answer.
        guess: Clairic's ``guess``. Metadata only, never used as a value.
        span: Supporting text in the original input.
        spans: Supporting spans of a list answer, aligned with ``value``.
        source: ``"clairic"``, ``"clef"``, ``"const"``, or ``"default"`` when
            the value is null and pydantic's default applies.
        attribution_p: Probability from the attribution check: a float for a
            scalar, one entry per original element for a list (``None`` for an
            element that got no answer, never NaN).
        gate_p: Probability from the change gate (updates).
        crosscheck_agree: Whether the input-only cross-check agreed (updates).
        reask: Whether a re-ask contributed to the value.
        repaired: Whether the value comes from the validation repair loop (the
            first answer failed validation and a narrow re-ask replaced it).
        rejected_reason: Why the answer was thrown away, if it was.
        warnings: Short notes, such as ``span_outside_input``; for a field the
            repair loop worked on, its repair history (``repair[1]: ...``).
    """

    path: Path
    pretty: str
    wire_name: str
    engine: str
    value: Any = None
    raw: Any = None
    p_null: float | None = None
    confidence: float | None = None
    guess: Any = None
    span: Span | None = None
    spans: tuple[Span, ...] | None = None
    source: Literal["clairic", "clef", "const", "default"] = "clairic"
    attribution_p: float | tuple[float | None, ...] | None = None
    gate_p: float | None = None
    crosscheck_agree: bool | None = None
    reask: bool = False
    repaired: bool = False
    rejected_reason: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchOp:
    """One applied RFC 6902 operation (updates).

    Attributes:
        op: ``"add"``, ``"replace"`` or ``"remove"``.
        path: JSON Pointer.
        value: The value for ``add`` and ``replace``.
        repaired_from: The path as first computed when apply repaired it.
    """

    op: str
    path: str
    value: Any = None
    repaired_from: str | None = None


@dataclass(frozen=True)
class RejectedOp:
    """An operation or answer that was computed and then thrown away.

    Attributes:
        op: The operation, or ``None`` for an answer that never became one
            (every rejection of an insert).
        path: JSON Pointer of the field. For an element dropped from a list
            answer the last segment is the element's position in the model's
            own answer list (one index space for every cause: conversion
            failure, no matching option, attribution), not in the final value.
        reason: Short machine-readable reason, such as ``attribution``.
        detail: A longer explanation.
        p: The probability or confidence behind the decision, when there is one.
    """

    op: str | None
    path: str
    reason: str
    detail: str = ""
    p: float | None = None


@dataclass(frozen=True)
class ResultRecord:
    """One extracted, updated or deleted record.

    Attributes:
        schema_name: Name of the schema.
        json_doc_id: Id of the existing record (``None`` for inserts).
        kind: ``"insert"``, ``"update"`` or ``"delete"``. An update that applies
            no operation is still ``"update"``, with empty ``patches`` and
            ``after == before``. A delete (a ``Many`` schema with
            ``enable_deletes``) names the record in ``json_doc_id``, keeps it as
            ``before`` and has no ``value``, no ``after`` and no ``entity``; its
            ``delete_p`` is the probability that the input says to delete it.
        value: The validated model, or ``None`` when validation failed.
        before: The record before the change (updates and deletes).
        after: The document after the change; for an invalid insert, the
            candidate document that failed validation.
        patches: Operations that were applied, in the order they were applied.
        rejected: Answers and operations that were thrown away.
        fields: How each asked field got its value, keyed by dotted path.
        entity: What is known about the entity.
        valid: Whether ``value`` passed validation.
        errors: Pydantic error strings when it did not.
        delete_p: For a delete, the probability that the input says the record
            should be deleted; ``None`` otherwise.
    """

    schema_name: str
    json_doc_id: str | None
    kind: Literal["insert", "update", "delete"]
    value: BaseModel | None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    patches: tuple[PatchOp, ...] = ()
    rejected: tuple[RejectedOp, ...] = ()
    fields: Mapping[str, FieldAnswer] = field(default_factory=dict, hash=False)
    entity: EntityInfo | None = None
    valid: bool = True
    errors: tuple[str, ...] = ()
    delete_p: float | None = None


@dataclass(frozen=True)
class SkippedEntity:
    """An entity that was found and not turned into a record.

    Attributes:
        schema_name: Name of the schema.
        entity: The entity.
        reason: ``inserts_disabled``, ``updates_disabled``, ``ambiguous_match``,
            ``not_qualified``, ``conflict_update_delete``, ``validation_failed``
            or ``empty_after_verification``.
        detail: A longer explanation.
        candidates: Options with their match probability, the most likely first
            (``ambiguous_match``: existing record ids and ``"__new__"``;
            ``conflict_update_delete`` and ``inserts_disabled``: the record or
            records involved).
    """

    schema_name: str
    entity: EntityInfo
    reason: str
    detail: str = ""
    candidates: tuple[tuple[str, float], ...] = ()


def _entity_key(record: "ResultRecord") -> str:
    return record.entity.key if record.entity else ""


def _no_cost() -> "CostReport":
    from trustcall_system_one._config import CostReport

    return CostReport(usd=0.0, unpriced_models=(), complete=True)


@dataclass(frozen=True)
class ExtractionResult:
    """The outcome of one ``invoke``.

    Attributes:
        records: Every record, in order of first mention (deletes last).
        responses: The validated models of the valid records, in order.
        skipped: Entities that were found and not used.
        untouched: Ids of existing records that got neither a record nor a
            skipped entity: the input did not mention them (presence gate),
            they belong to a schema that did not run, or they are opaque
            records kept by ``existing_schema_policy=False``.
        deleted: Ids of existing records that were deleted.
        attempts: Extraction rounds that ran: 0 when nothing was extracted, else
            1 plus the repair rounds of the record that needed the most (one
            repair round, however many fields it re-asks, is one attempt).
        usage: Token and request totals by model and by call kind.
        cost: Priced part of ``usage``.
        calls: One ``CallRecord`` per transport call, failed ones included.
        warnings: Human-readable notes about anything that was degraded.
    """

    records: tuple[ResultRecord, ...] = ()
    responses: tuple[BaseModel, ...] = ()
    skipped: tuple[SkippedEntity, ...] = ()
    untouched: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    attempts: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: "CostReport" = field(default_factory=lambda: _no_cost())
    calls: tuple[CallRecord, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def response_metadata(self) -> list[dict[str, str | None]]:
        """One ``{"id", "json_doc_id"}`` entry per response, like trustcall's.

        ``id`` is ``"<schema name>:<entity key>"`` (there is no tool call id) and
        ``json_doc_id`` the id of the existing record, ``None`` for an insert.
        """
        return [
            {
                "id": f"{record.schema_name}:{_entity_key(record)}",
                "json_doc_id": record.json_doc_id,
            }
            for record in self.records
            if record.valid and record.value is not None
        ]
