"""The change gate and the list near-duplicate check (updates).

Every operation the diff computed is judged by Clef before it is applied: ONE
request per operation (call kind ``gate``, model ``gate_model``), each with its
own structured state::

    {"input": <text>,
     "entity": {"label": ..., "mentions": [...]},
     "record": <the existing record>,
     "proposed_change": {"op": ..., "path": ..., "field": ..., "from": ..., "to": ...}}

The question is "Does ``input`` state the information in ``proposed_change.to``
about ``entity.label``, as new or changed compared to ``record``?"; an addition
passes at ``gates.add_threshold``, a replacement at ``gates.replace_threshold``
and a removal at ``gates.remove_threshold``. The gate is a backstop: the
deterministic rules of the diff and the verification ladder run first.

An element appended to a list that resembles an element of the record (text
ratio 0.6-0.9) is first checked with one Clef question of call kind
``list_dedup``: "Is the new element the same fact as the old one?".
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trustcall_system_one._config import GateConfig
from trustcall_system_one._diff import Change, apply_ops
from trustcall_system_one._run import gather
from trustcall_system_one._verify import Ctx
from trustcall_system_one._wire import NoulAnswer, NoulCriteria, NoulQuestion

__all__ = [
    "DedupVerdict",
    "GateVerdict",
    "dedup_elements",
    "gate_changes",
    "note_novelty",
    "numeric_replace",
    "record_after",
    "relative_edits",
    "threshold",
]

DUPLICATE_AT = 0.6
"""Probability from which Clef calls a new list element a repeat of an old one."""

GATE_QUESTION = (
    "Does `input` state the information in `proposed_change.to` about "
    "`entity.label`, as new or changed compared to `record`?"
)
GATE_TRUE = (
    "The input states the information in `proposed_change.to` about "
    "`entity.label`, and `record` does not already hold it."
)
GATE_FALSE = (
    "The input does not state it about `entity.label`: it only gives a shorter or "
    "partial form, another value, something about somebody else, or it repeats "
    "what `record` already holds."
)
REMOVE_QUESTION = (
    "Does `input` state that {field} of `entity.label` was removed or no longer "
    "applies?"
)
REMOVE_TRUE = (
    "The input says that {field} of `entity.label` no longer applies, or was removed."
)
REMOVE_FALSE = (
    "The input does not say that: it is silent about {field}, or it gives a new "
    "value for it."
)


NOVELTY_QUESTION = (
    'Does the sentence "{text}" say anything about `entity.label` that `record` '
    "does not already hold?"
)
NOVELTY_TRUE = "The sentence adds information that `record` does not hold."
NOVELTY_FALSE = "Everything the sentence says is already held by `record`."

RELATIVE_QUESTION = (
    "Does `input` state a change relative to the current value (e.g. 'two more', "
    "'one fewer', 'double it') rather than a new value?"
)
RELATIVE_TRUE = (
    "The input says how the current value changes (more, fewer, double), not what "
    "the new value is."
)
RELATIVE_FALSE = "The input gives the new value itself."


@dataclass(frozen=True)
class GateVerdict:
    """The change gate's answer for one operation.

    Attributes:
        passed: Whether the operation may be applied.
        p: The probability Clef gave (``None`` when no answer came back).
        needed: The threshold it had to reach.
    """

    passed: bool
    p: float | None
    needed: float


@dataclass(frozen=True)
class DedupVerdict:
    """The near-duplicate check's answer for one appended list element.

    Attributes:
        duplicate: Whether the element repeats ``other``.
        p: The probability Clef gave (``None`` when no answer came back).
        other: The record element it was compared with.
    """

    duplicate: bool
    p: float | None
    other: str


def threshold(gates: GateConfig, change: Change) -> float:
    """The probability an operation needs: add, replace or remove/set-null."""
    if change.removes:
        return gates.remove_threshold
    return gates.add_threshold if change.op == "add" else gates.replace_threshold


def _gate_question(change: Change) -> NoulQuestion:
    if change.removes:
        what = f"`{change.field_path}`"
        return NoulQuestion(
            type="noul",
            instructions=REMOVE_QUESTION.format(field=what),
            criteria=NoulCriteria(
                true=REMOVE_TRUE.format(field=what),
                false=REMOVE_FALSE.format(field=what),
            ),
        )
    return NoulQuestion(
        type="noul",
        instructions=GATE_QUESTION,
        criteria=NoulCriteria(true=GATE_TRUE, false=GATE_FALSE),
    )


def gate_state(ctx: Ctx, record: dict[str, Any], change: Change) -> dict[str, Any]:
    """The structured Clef state for one proposed change.

    Args:
        ctx: The entity context.
        record: The existing record.
        change: The operation under review.

    Returns:
        ``{"input", "entity", "record", "proposed_change"}``; when the whole
        state is over the Clef budget the input is narrowed to windows around
        the operation's spans (the record and the change are never cut).
    """
    proposed = {
        "op": change.op,
        "path": change.path,
        "field": change.field_path,
        "from": change.old,
        "to": change.value,
    }
    return ctx.fitted(
        spans=list(change.spans) or None, record=record, proposed_change=proposed
    )


async def gate_changes(
    ctx: Ctx,
    record: dict[str, Any],
    changes: Sequence[Change],
    *,
    records: Mapping[int, dict[str, Any]] | None = None,
) -> list[GateVerdict]:
    """Ask Clef, once per operation, whether the input supports it.

    The requests are independent (each has its own state) and run together, on
    ``gate_model``.

    Args:
        ctx: The entity context.
        record: The existing record.
        changes: The operations to judge.
        records: The record to judge an operation against instead, by ``id`` of the
            change (a note is judged against the record after the update's other
            operations).

    Returns:
        One verdict per change, in order. An operation passes when Clef's
        probability reaches the threshold of its kind; a missing answer is
        never a pass.
    """
    gates = ctx.run.gates

    async def one(change: Change) -> GateVerdict:
        needed = threshold(gates, change)
        answers = await ctx.run.call_clef(
            {"gate": _gate_question(change)},
            gate_state(ctx, (records or {}).get(id(change), record), change),
            meta=ctx.meta("gate", field=change.field_path),
            model=ctx.run.gate_model,
        )
        answer = answers.get("gate")
        p = answer.noul if isinstance(answer, NoulAnswer) else None
        return GateVerdict(p is not None and p >= needed, p, needed)

    return list(await gather(*(one(change) for change in changes)))


async def dedup_elements(
    ctx: Ctx, record: dict[str, Any], changes: Sequence[Change]
) -> dict[int, DedupVerdict]:
    """Ask Clef whether each near-duplicate list element repeats the old one.

    One question per element that has ``Change.near`` candidates (the most
    similar one is compared), all in one request of kind ``list_dedup``.

    Args:
        ctx: The entity context.
        record: The existing record.
        changes: The appended-element operations.

    Returns:
        The verdict by position in ``changes``, for the elements that were
        asked. An element is a duplicate at ``p >= 0.6``.
    """
    questions: dict[str, NoulQuestion] = {}
    others: dict[int, str] = {}
    for index, change in enumerate(changes):
        if not change.near:
            continue
        other = change.near[0][0]
        others[index] = other
        questions[f"dedup.{index}"] = NoulQuestion(
            type="noul",
            instructions=(
                f'Is "{change.value}" the same fact as "{other}", as far as '
                "`entity.label` goes?"
            ),
            criteria=NoulCriteria(
                true=f'"{change.value}" says the same thing as "{other}".',
                false=f'"{change.value}" adds something that "{other}" does not say.',
            ),
        )
    if not questions:
        return {}
    spans = [span for change in changes if change.near for span in change.spans]
    answers = await ctx.run.call_clef(
        questions,
        ctx.fitted(spans=spans or None, record=record),
        meta=ctx.meta(
            "list_dedup",
            field=",".join(sorted({changes[i].field_path for i in others})),
        ),
    )
    verdicts: dict[int, DedupVerdict] = {}
    for index, other in others.items():
        answer = answers.get(f"dedup.{index}")
        p = answer.noul if isinstance(answer, NoulAnswer) else None
        verdicts[index] = DedupVerdict(p is not None and p >= DUPLICATE_AT, p, other)
    return verdicts


def numeric_replace(change: Change) -> bool:
    """Whether an operation replaces one number with another.

    Such an operation is a candidate for a relative edit ("two more" read as the
    stated 2).
    """
    return (
        change.op == "replace"
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in (change.old, change.value)
        )
        and change.old != change.value
    )


async def relative_edits(
    ctx: Ctx, record: dict[str, Any], changes: Sequence[Change]
) -> list[float | None]:
    """Ask Clef, per numeric replacement, whether the input states a RELATIVE change.

    "Please add two more of the blue mugs" says how much to add to the line's
    quantity (3), not that the quantity is 2; the library does no arithmetic on a
    record's values, so writing the 2 over the 3 loses data. One request per
    operation (call kind ``relative_edit``, model ``gate_model``) with the gate's
    state.

    Args:
        ctx: The entity context.
        record: The existing record.
        changes: The operations to judge (``numeric_replace`` ones).

    Returns:
        One probability per change, in order; ``None`` for a missing answer.
    """

    async def one(change: Change) -> float | None:
        answers = await ctx.run.call_clef(
            {"relative": _relative_question()},
            gate_state(ctx, record, change),
            meta=ctx.meta("relative_edit", field=change.field_path),
            model=ctx.run.gate_model,
        )
        answer = answers.get("relative")
        return answer.noul if isinstance(answer, NoulAnswer) else None

    return list(await gather(*(one(change) for change in changes)))


def _relative_question() -> NoulQuestion:
    return NoulQuestion(
        type="noul",
        instructions=RELATIVE_QUESTION,
        criteria=NoulCriteria(true=RELATIVE_TRUE, false=RELATIVE_FALSE),
    )


def record_after(record: dict[str, Any], changes: Sequence[Change]) -> dict[str, Any]:
    """The record once the given operations are applied (those that apply).

    The notes gate and the novelty check compare a sentence with this, not with
    the record as it was: "Bob Roe's new phone number is 555-0199" is not news
    about a record whose phone is changed to 555-0199 in the same update.

    Args:
        record: The existing record.
        changes: The operations to apply, in order. One that cannot be applied is
            left out.

    Returns:
        A patched copy.
    """
    doc: dict[str, Any] = apply_ops(record, [change.as_op() for change in changes]).doc
    return doc


async def note_novelty(
    ctx: Ctx, record: dict[str, Any], notes: Sequence[Change]
) -> list[float | None]:
    """Ask Clef, per appended sentence, whether it adds anything to the record.

    One question per sentence, each in its own request when
    ``verify.isolate_decisions`` is on (call kind ``novelty``). The record is the
    one after the update's other operations (``record_after``).

    Args:
        ctx: The entity context.
        record: The record the sentences are compared with.
        notes: The appended sentences (``add`` operations).

    Returns:
        One probability per note, in order; ``None`` for a missing answer.
    """
    questions: dict[str, NoulQuestion] = {
        f"novelty.{i}": NoulQuestion(
            type="noul",
            instructions=NOVELTY_QUESTION.format(text=change.value),
            criteria=NoulCriteria(true=NOVELTY_TRUE, false=NOVELTY_FALSE),
        )
        for i, change in enumerate(notes)
    }
    if not questions:
        return []
    spans = [span for change in notes for span in change.spans]
    answers = await ctx.run.call_clef(
        questions,
        ctx.fitted(spans=spans or None, record=record),
        meta=ctx.meta("novelty", field=",".join(sorted({c.field_path for c in notes}))),
        decisive=True,
    )
    out: list[float | None] = []
    for i in range(len(notes)):
        answer = answers.get(f"novelty.{i}")
        out.append(answer.noul if isinstance(answer, NoulAnswer) else None)
    return out
