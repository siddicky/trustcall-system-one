"""The vocabulary of the evals: operations, expectations, cases and their scores."""

import dataclasses
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from trustcall_system_one import Transport

__all__ = [
    "ANY",
    "Case",
    "CaseResult",
    "MustNot",
    "Op",
    "Rejected",
    "Val",
    "WorldLike",
    "normalise",
    "values_equal",
]


class _Any:
    """The type of ``ANY``."""

    def __repr__(self) -> str:
        return "ANY"


ANY = _Any()
"""A value that is not compared."""

OpKind = Literal["add", "replace", "remove", "insert", "delete"]


def normalise(value: Any) -> Any:
    """A comparable form of a value.

    Text is case-folded, stripped of trailing punctuation and its whitespace
    collapsed; numbers (and numeric strings) become floats; containers are
    normalised recursively.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = " ".join(value.split()).casefold().rstrip(".;,")
        try:
            number = float(text.replace(",", ""))
        except ValueError:
            return text
        return number if math.isfinite(number) else text
    if isinstance(value, Mapping):
        return {str(k): normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    return normalise(str(value))


def values_equal(first: Any, second: Any) -> bool:
    """Whether two values are the same after ``normalise``.

    Two strings never meet as numbers ("02139" is not "2139").
    """
    if isinstance(first, str) and isinstance(second, str):
        return " ".join(first.split()).casefold().rstrip(".;,") == " ".join(
            second.split()
        ).casefold().rstrip(".;,")
    return bool(normalise(first) == normalise(second))


def folded(text: Any) -> str:
    """Case-folded, whitespace-collapsed text of a value (for ``contains``)."""
    if isinstance(text, (list, tuple)):
        return " | ".join(folded(t) for t in text)
    return " ".join(str(text).split()).casefold()


@dataclass(frozen=True)
class Op:
    """An operation of a result: expected by a case, or produced by a run.

    ``kind`` is a JSON Patch verb of an update (``add``, ``replace``, ``remove``;
    ``path`` is the pointer inside the record) or a record-level ``insert`` /
    ``delete``.

    Attributes:
        kind: The operation.
        path: JSON Pointer inside the record; empty for ``insert`` and ``delete``.
        value: The value (expected: ``ANY`` does not compare).
        who: The record: the id of an existing record (updates and deletes), or a
            part of the entity's label (inserts). ``None`` matches any record.
        contains: Expected only: the value, as text, must contain this text
            (case-folded). For a verbatim sentence whose exact words a model
            may choose.
    """

    kind: OpKind
    path: str = ""
    value: Any = ANY
    who: str | None = None
    contains: str | None = None

    def matches(self, actual: "Op") -> bool:
        """Whether ``actual`` is the operation this one expects."""
        if actual.kind != self.kind or actual.path != self.path:
            return False
        if self.who is not None and not _same_record(self.who, actual.who, self.kind):
            return False
        if self.contains is not None:
            return folded(self.contains) in folded(actual.value)
        if self.value is ANY:
            return True
        return values_equal(self.value, actual.value)

    def label(self) -> str:
        """A short description for reports."""
        who = f"{self.who}: " if self.who is not None else ""
        shown = (
            f" ~{self.contains!r}"
            if self.contains is not None
            else ""
            if self.value is ANY
            else f" {json.dumps(self.value, default=str)}"
        )
        return f"{who}{self.kind} {self.path}{shown}".strip()


def _same_record(wanted: str, actual: str | None, kind: str) -> bool:
    if actual is None:
        return False
    if kind == "insert":
        return folded(wanted) in folded(actual)
    return wanted == actual


@dataclass(frozen=True)
class Val:
    """A value that a record must hold at the end.

    Attributes:
        who: The record (see ``Op.who``).
        path: JSON Pointer of the field in the final document (``/city``,
            ``/items/1/quantity``).
        value: The expected value; ``None`` means absent or null.
        contains: Instead of ``value``: the field (or any element of a list
            field) contains this text, case-folded.
    """

    who: str
    path: str
    value: Any = None
    contains: str | None = None


@dataclass(frozen=True)
class Rejected:
    """Something the run must have thrown away, and say so (a ``RejectedOp``).

    Attributes:
        reason: The rejection's reason (``duplicate``, ``attribution``, ...).
        path: Its JSON Pointer; ``None`` for any.
        who: The record (see ``Op.who``); ``None`` for any.
    """

    reason: str
    path: str | None = None
    who: str | None = None


@dataclass(frozen=True)
class MustNot:
    """Something that must not happen.

    Build with ``MustNot.op``, ``MustNot.value``, ``MustNot.insert`` or
    ``MustNot.delete``.

    Attributes:
        category: ``bleed`` (a value on the wrong record), ``wrong_delete``,
            ``wrong_insert`` or ``wrong_op``; reports count them separately.
        kind: Operation kind to forbid, or ``None`` for any.
        path: Forbidden path or path prefix (``/items/0`` forbids everything
            inside the first item); ``None`` for any.
        who: The record the forbidden thing concerns; ``None`` for any.
        value: The forbidden value (``ANY`` for any).
        contains: Forbidden text inside the value.
        final: For a ``value`` check: look at the final document, not at the
            operations.
    """

    category: str
    kind: str | None = None
    path: str | None = None
    who: str | None = None
    value: Any = ANY
    contains: str | None = None
    final: bool = False

    @staticmethod
    def op(
        kind: str | None = None,
        path: str | None = None,
        who: str | None = None,
        value: Any = ANY,
        contains: str | None = None,
    ) -> "MustNot":
        """No operation like this: ``kind`` on ``path`` (or below) of ``who``."""
        return MustNot("wrong_op", kind, path, who, value, contains)

    @staticmethod
    def value_at(
        who: str, path: str, value: Any = ANY, contains: str | None = None
    ) -> "MustNot":
        """The final record ``who`` never holds this value at ``path``."""
        return MustNot("bleed", None, path, who, value, contains, final=True)

    @staticmethod
    def insert(who: str | None = None) -> "MustNot":
        """No record is inserted (for the entity whose label has ``who`` in it)."""
        return MustNot("wrong_insert", "insert", "", who)

    @staticmethod
    def delete(who: str | None = None) -> "MustNot":
        """No record (or not this one) is deleted."""
        return MustNot("wrong_delete", "delete", "", who)

    def label(self) -> str:
        """A short description for reports."""
        if self.final:
            shown = self.contains if self.contains is not None else self.value
            return f"{self.who} {self.path} must not hold {shown!r}"
        parts = [
            self.kind or "any op",
            self.path or "",
            f"on {self.who}" if self.who else "",
        ]
        return "no " + " ".join(p for p in parts if p)


class WorldLike(Protocol):
    """Anything that answers a case offline: a ``FakeWorld`` or a wrapper of one."""

    def transport(self) -> Transport:
        """The transport the extractor runs on."""
        ...


@dataclass
class Case:
    """One small eval case.

    Attributes:
        id: A short unique name.
        text: The input.
        schemas: Builds the schema list given to ``create_extractor`` (so each run
            has fresh ``Many``/``One`` wrappers).
        existing: The ``existing`` argument of ``invoke``.
        options: Extra ``create_extractor`` keyword arguments (``enable_inserts``
            and the like). A configuration's overrides come on top.
        expect_ops: Operations the run must produce (recall) and nothing else
            (precision).
        allow_ops: Operations that may also happen: a legitimate extra note
            ("Emma took up rock climbing") neither fails the case nor lowers
            precision. Each entry absorbs ONE produced operation (the same note
            added twice is one allowed and one extra), and anything else not in
            ``expect_ops`` still counts. An allowed note is judged by ``must_not``
            too: another person's facts on a record are bleed.
        expect_values: Values the final records must hold.
        expect_rejected: Rejections the result must report (a near-duplicate that
            was recognised as such, not merely never proposed).
        must_not: Things that must not happen.
        tags: Groups for ``--tag`` (``notes``, ``bleed``, ``update``, ...).
        world: Builds the scripted ground truth that answers the case offline
            (a ``FakeWorld``, or anything with ``transport()``). Without one the
            case runs live only.
        note: What the case is about, and what live runs have shown.
    """

    id: str
    text: str
    schemas: Callable[[], list[Any]]
    existing: Any = None
    options: Mapping[str, Any] = field(default_factory=dict)
    expect_ops: Sequence[Op] = ()
    allow_ops: Sequence[Op] = ()
    expect_values: Sequence[Val] = ()
    expect_rejected: Sequence[Rejected] = ()
    must_not: Sequence[MustNot] = ()
    tags: Sequence[str] = ()
    world: Callable[[], WorldLike] | None = None
    note: str = ""


@dataclass
class CaseResult:
    """The score of one case under one configuration.

    Attributes:
        case: The case id.
        matched_expected: Expected operations that were produced.
        n_expected: Expected operations.
        matched_actual: Produced operations that were expected.
        n_actual: Produced operations.
        values_correct: Expected values that hold.
        n_values: Expected values.
        violations: The ``must_not`` checks that failed, as ``(category, label)``.
        valid: Records that validated.
        n_records: Records with a document (insert and update).
        requests: Model requests made.
        input_tokens: Input tokens over all requests.
        wall_s: Wall-clock seconds.
        error: The exception that stopped the case, if any.
        warnings: The result's warnings.
        ops: The produced operations, for the report.
        missing: Expected operations and values that did not happen.
        extra: Produced operations that were not expected.
        forwarded: Requests that went to the live API during a replay.
        n_notes: Expected notes: operations that add to a ``notes`` list and
            values of a ``notes`` list.
        notes_found: How many of those were produced.
        allowed: Produced operations that were allowed extras (not counted in
            precision).
        n_rejections: Expected rejections.
        rejections_found: How many were reported.
        unknown_outcomes: Calls that may have been run and charged although no
            answer came back: ``UncertainOutcomeError`` (a timeout, a dropped
            connection, a 5xx) and calls cancelled after they may have been sent.
        reported_cost_usd: The cost in USD that responses reported (Experiential
            Labs reports one per call); it covers the calls that answered.
        output_tokens: Output tokens over all requests.
        by_provider: Requests and tokens per provider that answered (``requests``,
            ``input_tokens``, ``output_tokens``), from the call log (failed calls
            included), so that a combination's cost can be estimated from prices.
        probabilities: The attribution and change-gate probabilities behind the
            result, keyed ``"<record> <field> attribution"`` / ``"... gate"``
            (a list field has one attribution per element). These are the margins
            a threshold sweep moves along.
    """

    case: str
    matched_expected: int = 0
    n_expected: int = 0
    matched_actual: int = 0
    n_actual: int = 0
    values_correct: int = 0
    n_values: int = 0
    violations: list[tuple[str, str]] = field(default_factory=list)
    valid: int = 0
    n_records: int = 0
    requests: int = 0
    input_tokens: int = 0
    wall_s: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    forwarded: int = 0
    probabilities: dict[str, Any] = field(default_factory=dict)
    n_notes: int = 0
    notes_found: int = 0
    allowed: int = 0
    n_rejections: int = 0
    rejections_found: int = 0
    unknown_outcomes: int = 0
    reported_cost_usd: float = 0.0
    output_tokens: int = 0
    by_provider: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        """Share of produced operations that were expected (1 when none)."""
        return self.matched_actual / self.n_actual if self.n_actual else 1.0

    @property
    def recall(self) -> float:
        """Share of expected operations that were produced (1 when none)."""
        return self.matched_expected / self.n_expected if self.n_expected else 1.0

    @property
    def passed(self) -> bool:
        """Everything expected happened, nothing else did, nothing forbidden did."""
        return (
            self.error is None
            and self.matched_expected == self.n_expected
            and self.matched_actual == self.n_actual
            and self.values_correct == self.n_values
            and self.rejections_found == self.n_rejections
            and not self.violations
        )

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready form."""
        out = dataclasses.asdict(self)
        out.update(precision=self.precision, recall=self.recall, passed=self.passed)
        return out


_POINTER_ESCAPE = re.compile(r"~[01]")


def split_pointer(pointer: str) -> list[str]:
    """The parts of a JSON Pointer, unescaped."""
    if not pointer:
        return []
    return [
        _POINTER_ESCAPE.sub(lambda m: "~" if m.group() == "~0" else "/", part)
        for part in pointer.lstrip("/").split("/")
    ]
