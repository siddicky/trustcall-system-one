"""The verification ladder: answer-level checks, re-asks, null checks, attribution.

Every function here works on ``FieldState`` objects, one per Clairic field. A
step is parallelised across fields: a step's requests are fired together (Clef
questions that share a state share a request) and the next step starts when the
whole step is done.

For an insert the ladder is, in order:

1. low-confidence re-ask, with a Clef adjudication when the two answers differ;
2. borderline-null check, which can re-ask a null or null a value;
3. attribution of every value (and every list element) to the entity.

Clef-engine answers are not verified further; they are decisions already.
"""

import dataclasses
import enum
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from trustcall_system_one._chunk import (
    chunk_size,
    identity_state,
    merge_parts,
    part_of,
    split_ranges,
)
from trustcall_system_one._coerce import coerce, coerce_union
from trustcall_system_one._compile import (
    FieldSpec,
    Role,
    SchemaPlan,
    render_instructions,
)
from trustcall_system_one._errors import StateTruncatedError
from trustcall_system_one._render import (
    RenderedState,
    mention_windows,
    narrow_text,
    render_clairic,
)
from trustcall_system_one._run import Answer, Question, Run, gather
from trustcall_system_one._types import CallKind
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClairicAnswer,
    ClairicField,
    ClairicResponse,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
)
from trustcall_system_one.transport import CallMeta

__all__ = [
    "Asked",
    "Ctx",
    "FieldState",
    "Resolved",
    "accept",
    "agree",
    "ask_clairic",
    "display",
    "resolve",
    "resolve_clef",
    "run_ladder",
]

Span = tuple[int, int]

REASK_SUFFIX = " Answer only from the text near these mentions."
_SHORT_CODES = frozenset({"abstain", "unstated", "type_uncertain", "missing_answer"})
_AGREE_RATIO = 0.9
_NONE_OF_THESE = "none_of_these"
_NOT_STATED = "The input states no value for this."


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class Ctx:
    """What the passes for one entity share.

    Attributes:
        run: The invoke's context.
        plan: The compiled schema.
        text: The user's input, verbatim.
        label: How prompts name the entity (``"the Person"``).
        key: The entity's key in results.
        mentions: How the input refers to the entity (empty for a ``One``).
    """

    run: Run
    plan: SchemaPlan
    text: str
    label: str
    key: str = "e0"
    mentions: tuple[str, ...] = ()

    def meta(self, kind: CallKind, *, field: str | None = None) -> CallMeta:
        """Routing context for a call. The entity is its key, not its label."""
        return CallMeta(kind=kind, schema=self.plan.name, entity=self.key, field=field)

    def desc(self, spec: FieldSpec) -> str:
        """The field's description, put through ``verify.instruction_template``."""
        return self.run.verify.instruction_template.format(
            label=self.label, desc=spec.description
        )

    def clairic_field(
        self,
        spec: FieldSpec,
        *,
        suffix: str = "",
        null_threshold: float | None = None,
    ) -> ClairicField:
        """Return a copy of the spec's Clairic field about this entity.

        The instructions are rebuilt from the field's parts: the parent-context
        prefix is kept as it is, the description is put through
        ``verify.instruction_template`` and the type and constraint hints follow.
        The plan itself is never changed.
        """
        assert spec.clairic is not None
        update: dict[str, Any] = {
            "instructions": render_instructions(
                "clairic",
                self.desc(spec),
                prefix=spec.context,
                hints=spec.hints,
                limit=self.run.limits.max_instruction_chars,
            )
            + suffix
        }
        if null_threshold is not None:
            update["null_threshold"] = null_threshold
        return spec.clairic.model_copy(update=update)

    def clef_question(self, spec: FieldSpec) -> Question:
        """Return a copy of the spec's Clef question about this entity.

        Rebuilt from the field's parts (so a question the compiler clipped is not
        a problem) with the templated description.
        """
        assert spec.clef is not None
        role: Role = "union_type" if spec.clairic is not None else spec.clef.type
        text = render_instructions(
            role,
            self.desc(spec),
            prefix=spec.clef_context,
            limit=self.run.limits.max_instruction_chars,
        )
        return spec.clef.model_copy(update={"instructions": text})

    def rendered(
        self,
        spans: list[Span] | None = None,
        *,
        windows: list[Span] | None = None,
    ) -> RenderedState:
        """Render the Clairic state.

        Args:
            spans: Spans in the input to keep windows of context around.
            windows: Explicit ranges of the input to keep (wins over ``spans``).

        Returns:
            The full input when neither is given.
        """
        if windows is None and spans:
            windows = mention_windows(
                len(self.text), spans, self.run.verify.window_chars
            )
        return render_clairic(
            self.text,
            label=self.label,
            mentions=self.mentions,
            windows=windows or None,
        )

    def entity_state(
        self,
        narrow_to: list[Span] | None = None,
        *,
        anchors: list[Span] | None = None,
    ) -> dict[str, Any]:
        """Clef state ``{input, entity}``.

        Args:
            narrow_to: Cut ``input`` to windows around these spans (an isolated
                check).
            anchors: Spans to keep if the full input is over the Clef budget.
        """
        text = self.text
        if narrow_to:
            windows = mention_windows(
                len(text), narrow_to, self.run.verify.window_chars
            )
            text = narrow_text(text, windows)
        return self.run.clef_state(
            text,
            entity={"label": self.label, "mentions": list(self.mentions)},
            spans=None if narrow_to else anchors,
        )


# --------------------------------------------------------------------------- #
# Asking Clairic, in chunks when the state is too long
# --------------------------------------------------------------------------- #

Asked = dict[str, tuple[ClairicAnswer | None, RenderedState | None]]
"""Per field: the answer and the state to map its spans with."""


async def ask_clairic(
    ctx: Ctx,
    fields: dict[str, ClairicField],
    *,
    meta: CallMeta,
    windows: list[Span] | None = None,
) -> Asked:
    """Ask Clairic for ``fields`` on the input (or on ``windows`` of it).

    When Clairic reports ``truncated``, the same fields are asked again on
    consecutive overlapping chunks that together cover everything that was
    asked, concurrently, and the answers are merged per field (see
    ``_chunk.merge_parts``). The chunk size comes from how much of the input the
    truncated response was seen to read. If a chunk is itself truncated the
    input is split smaller, once; if that is still truncated a warning is
    recorded, or ``StateTruncatedError`` raised in strict mode. A warning is
    always recorded when the input is chunked.

    Args:
        ctx: The entity context.
        fields: Clairic fields keyed by wire name.
        meta: Routing context (the call kind).
        windows: Ranges of the input to ask about; ``None`` for all of it.

    Returns:
        Each field's answer and the state to map its spans with. For a chunked
        ask the spans are already in original-input offsets.
    """
    run = ctx.run
    rendered = ctx.rendered(windows=windows)
    response = await run.call_clairic(fields, rendered, meta=meta)
    anchored = run.anchor(rendered, response)
    if not response.truncated:
        return {name: (response.answers.get(name), anchored) for name in fields}

    ranges = windows or [(0, len(ctx.text))]
    overlap = run.verify.window_chars
    covered = sum(end - start for start, end in ranges)
    size = chunk_size(_chars_read(rendered, response, anchored), covered, overlap)
    responses: list[tuple[Span, ClairicResponse, RenderedState | None]] = []
    attempts = 0
    still_truncated = True
    while attempts < 2:
        attempts += 1
        chunks = split_ranges(ranges, size, overlap)
        if len(chunks) == 1 and chunks[0] == ranges[0] and len(ranges) == 1:
            break  # nothing to split: it is one range no longer than a chunk
        responses = await _ask_chunks(ctx, fields, chunks, meta)
        truncated = [item for item in responses if item[1].truncated]
        if not truncated:
            still_truncated = False
            break
        read = [
            _chars_read(ctx.rendered(windows=[chunk]), resp, anch)
            for chunk, resp, anch in truncated
        ]
        smaller = min(
            (r for r in read if r is not None),
            default=None,
        )
        shrunk = chunk_size(smaller, size, overlap)
        if attempts == 2 or shrunk >= size:
            break
        size = shrunk
    if responses:
        run.warn(
            f"clairic_chunked: the input of {ctx.plan.name} did not fit one "
            f"Clairic request and was split into {len(responses)} overlapping "
            "chunks whose answers were merged"
        )
    if still_truncated:
        message = (
            f"Clairic still reported a truncated state for {ctx.plan.name} after "
            "the input was split into chunks, so part of the input was not read"
        )
        if run.strict:
            raise StateTruncatedError(message)
        run.warn(f"clairic_truncated: {message}")
    if not responses:  # could not split: use what the full request returned
        return {name: (response.answers.get(name), anchored) for name in fields}
    merged: Asked = {}
    identity = identity_state(ctx.text)
    for name in fields:
        parts = [
            part_of(resp.answers[name], anch, chunk[0])
            for chunk, resp, anch in responses
            if name in resp.answers
        ]
        merged[name] = (merge_parts(parts), identity)
    return merged


def _chars_read(
    sent: RenderedState, response: ClairicResponse, anchored: RenderedState | None
) -> int | None:
    """How many input characters Clairic was seen to read, if it said."""
    if response.state_text is None or response.state_text == sent.text:
        return None
    if anchored is not None:
        return anchored.input_chars
    return len(response.state_text)


async def _ask_chunks(
    ctx: Ctx, fields: dict[str, ClairicField], chunks: list[Span], meta: CallMeta
) -> list[tuple[Span, ClairicResponse, RenderedState | None]]:
    async def one(
        chunk: Span,
    ) -> tuple[Span, ClairicResponse, RenderedState | None]:
        rendered = ctx.rendered(windows=[chunk])
        response = await ctx.run.call_clairic(fields, rendered, meta=meta)
        return chunk, response, ctx.run.anchor(rendered, response)

    return await gather(*(one(chunk) for chunk in chunks))


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def display(value: Any) -> str:
    """Render a value for a question: ISO dates, plain decimals, ``; `` lists."""
    if isinstance(value, (date, time, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return "; ".join(display(item) for item in value)
    return str(value)


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def agree(first: Any, second: Any) -> bool:
    """Whether two answers for one field say the same thing.

    Text agrees when it is equal after case and whitespace folding or has a
    ``difflib.SequenceMatcher`` ratio of at least 0.9; numbers compare by value;
    lists agree when they have the same size and every element has a distinct
    agreeing partner; anything else must be equal.
    """
    if first is None or second is None:
        return first is None and second is None
    if isinstance(first, str) and isinstance(second, str):
        a, b = _normalise(first), _normalise(second)
        return a == b or SequenceMatcher(None, a, b).ratio() >= _AGREE_RATIO
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return False
        remaining = list(second)
        for item in first:
            for index, other in enumerate(remaining):
                if agree(item, other):
                    del remaining[index]
                    break
            else:
                return False
        return True
    if _is_number(first) and _is_number(second):
        return Decimal(str(first)) == Decimal(str(second))
    return bool(first == second)


@dataclass(frozen=True)
class Resolved:
    """A model answer turned into a value, with the metadata that came with it.

    ``ok`` is false when the answer could not be used (``reason`` is a short
    code, ``detail`` the explanation). Spans are in the ORIGINAL input. For a
    list, ``origin`` gives the position in the model's own answer list of each
    element of ``value`` and ``lost`` the ``(position, reason, detail)`` of every
    element dropped while converting it (the one index space of rejected
    elements).
    """

    value: Any = None
    ok: bool = True
    reason: str | None = None
    detail: str = ""
    raw: Any = None
    span: Span | None = None
    spans: tuple[Span, ...] | None = None
    p_null: float | None = None
    confidence: float | None = None
    guess: Any = None
    notes: tuple[str, ...] = ()
    origin: tuple[int, ...] | None = None
    lost: tuple[tuple[int, str, str], ...] = ()


@dataclass
class FieldState:
    """One Clairic field as it moves through the ladder.

    Attributes:
        spec: The field.
        resolved: The current answer.
        type_answer: The Clef type answer of a union field.
        reject: Why the answer was thrown away (``None`` while it stands).
        reject_detail: Explanation of ``reject``.
        reject_p: The probability behind ``reject``.
        attribution_p: Attribution probability, or one per element for a list.
        reask: Whether a re-ask contributed to the value.
        dropped: List elements thrown away by the attribution check, as
            ``(position in the model's answer list, reason, detail, p)``.
        notes: Short notes for ``FieldAnswer.warnings``.
    """

    spec: FieldSpec
    resolved: Resolved
    type_answer: ChoiceAnswer | None = None
    reject: str | None = None
    reject_detail: str = ""
    reject_p: float | None = None
    attribution_p: float | tuple[float, ...] | None = None
    reask: bool = False
    dropped: list[tuple[int, str, str, float | None]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def value(self) -> Any:
        """The standing value: ``None`` when null or rejected."""
        return None if self.reject else self.resolved.value

    @property
    def spans(self) -> list[Span]:
        """Known spans of the standing value, in the original input."""
        resolved = self.resolved
        if resolved.spans:
            return list(resolved.spans)
        return [resolved.span] if resolved.span else []

    def has_value(self) -> bool:
        """Whether a non-null value stands."""
        return self.reject is None and self.resolved.value is not None

    def set_reject(self, reason: str, detail: str = "", p: float | None = None) -> None:
        """Throw the current answer away."""
        self.reject, self.reject_detail, self.reject_p = reason, detail, p


# --------------------------------------------------------------------------- #
# Resolving answers
# --------------------------------------------------------------------------- #


def _map_span(
    rendered: RenderedState | None, span: Span | None, notes: list[str]
) -> Span | None:
    if span is None:
        return None
    if rendered is None:
        if "span_unknown" not in notes:
            notes.append("span_unknown")
        return None
    mapped = rendered.to_original(span)
    if mapped is None and "span_outside_input" not in notes:
        notes.append("span_outside_input")
    return mapped


def _code(reason: str | None) -> str:
    """Reduce a coercion message to a short reason code."""
    if reason is None:
        return "invalid"
    return reason if reason in _SHORT_CODES else "unparseable"


async def resolve(
    ctx: Ctx,
    spec: FieldSpec,
    answer: ClairicAnswer | None,
    rendered: RenderedState | None,
    type_answer: ChoiceAnswer | None,
) -> Resolved:
    """Turn a Clairic answer into a value.

    Applies ``coerce`` (or ``coerce_union`` with the Clef type answer), maps spans
    back to the original input, and runs the Clef pass over the elements of a
    ``list[Literal | Enum]``.

    Args:
        ctx: The entity context.
        spec: The field.
        answer: The Clairic answer, or ``None`` when the response lacked one.
        rendered: The state the answer was asked on, as anchored on its
            ``state_text`` (``None`` when the input could not be located).
        type_answer: For a union of primitives, the Clef type answer.

    Returns:
        The resolved answer. Answer-level thresholds are not applied here (see
        ``accept``).
    """
    if answer is None:
        return Resolved(ok=False, reason="missing_answer", detail="no answer returned")
    gates = ctx.run.gates
    raw: Any = answer.value
    if spec.post == "union":
        if type_answer is None:
            return Resolved(
                ok=False,
                reason="missing_answer",
                detail="no type answer for the union",
                raw=raw,
            )
        coerced = coerce_union(spec, answer, type_answer, gates)
        raw = {"text": answer.value, "type": type_answer.choice}
    else:
        coerced = coerce(spec, answer, gates)
    notes: list[str] = []
    span = _map_span(rendered, coerced.span, notes)
    spans: tuple[Span, ...] | None = None
    if coerced.spans is not None:
        mapped = [_map_span(rendered, pair, notes) for pair in coerced.spans]
        if all(pair is not None for pair in mapped):
            spans = tuple(pair for pair in mapped if pair is not None)
    if not coerced.ok:
        return Resolved(
            ok=False,
            reason=_code(coerced.reason),
            detail=coerced.reason or "",
            raw=raw,
            p_null=coerced.p_null,
            confidence=coerced.confidence,
            guess=coerced.guess,
            notes=tuple(notes),
        )
    if coerced.reason:
        notes.append(coerced.reason)
    value = coerced.value
    origin = coerced.kept
    lost = [(index, "unparseable", message) for index, message in coerced.failed]
    if spec.item_kind == "choice" and isinstance(value, list) and value:
        before = origin if origin is not None else tuple(range(len(value)))
        value, spans, element_notes, positions, drops = await _choose_elements(
            ctx, spec, value, spans
        )
        notes.extend(element_notes)
        lost.extend((before[pos], reason, detail) for pos, reason, detail in drops)
        origin = tuple(before[pos] for pos in positions) if value is not None else None
    return Resolved(
        value=value,
        raw=raw,
        span=span,
        spans=spans,
        p_null=coerced.p_null,
        confidence=coerced.confidence,
        guess=coerced.guess,
        notes=tuple(notes),
        origin=origin,
        lost=tuple(lost),
    )


async def _choose_elements(
    ctx: Ctx,
    spec: FieldSpec,
    elements: list[Any],
    spans: tuple[Span, ...] | None,
) -> tuple[
    list[Any] | None,
    tuple[Span, ...] | None,
    list[str],
    list[int],
    list[tuple[int, str, str]],
]:
    """Map each text element of a ``list[Literal | Enum]`` to an allowed value.

    One Clef ``choice`` per element offers every allowed value plus a
    ``none_of_these`` option; an element that picks it, or whose pick is too
    uncertain, is dropped.

    Returns:
        The kept values (``None`` if none is left), their spans, notes, the
        position in ``elements`` of each kept value and ``(position, reason,
        detail)`` for every element that was dropped.
    """
    assert spec.element_choices is not None
    options = {option: str(value) for option, value in spec.element_choices.items()}
    none_id = _NONE_OF_THESE
    while none_id in options:
        none_id += "_2"
    criteria: dict[str, Any] = {
        **options,
        none_id: "None of the other options fits this item.",
    }
    questions: dict[str, Question] = {}
    for index, element in enumerate(elements):
        questions[f"{spec.wire_name}.el{index}"] = ChoiceQuestion(
            type="choice",
            instructions=(
                f'Which option is "{element}" in the input, for {ctx.desc(spec)}?'
            ),
            criteria=criteria,
        )
    answers = await ctx.run.call_clef(
        questions,
        ctx.run.clef_state(ctx.text, spans=list(spans) if spans else None),
        meta=ctx.meta("clef_fields", field=spec.wire_name),
    )
    gates = ctx.run.gates
    kept: list[Any] = []
    kept_spans: list[Span] = []
    positions: list[int] = []
    drops: list[tuple[int, str, str]] = []
    notes: list[str] = []
    for index, element in enumerate(elements):
        answer = answers.get(f"{spec.wire_name}.el{index}")
        if not isinstance(answer, ChoiceAnswer):
            notes.append(f"element_dropped: {element!r} got no answer")
            drops.append((index, "missing_answer", f"{element!r} got no answer"))
            continue
        chosen = answer.probabilities.get(answer.choice, 0.0)
        if answer.choice == none_id or answer.choice not in spec.element_choices:
            notes.append(f"element_dropped: {element!r} matches none of the options")
            drops.append((index, "none_of_these", f"{element!r} matches no option"))
            continue
        if chosen < gates.min_choice_prob or (
            answer.confidence < gates.min_choice_confidence
        ):
            notes.append(f"element_dropped: {element!r} is too uncertain")
            drops.append((index, "low_choice_prob", f"{element!r} is too uncertain"))
            continue
        value = spec.element_choices[answer.choice]
        if spec.post.startswith(("set", "frozenset")) and value in kept:
            continue
        kept.append(value)
        positions.append(index)
        if spans is not None:
            kept_spans.append(spans[index])
    if elements and not kept:
        return None, None, [*notes, "all elements dropped"], [], drops
    spans_out = tuple(kept_spans) if spans is not None else None
    return kept, spans_out, notes, positions, drops


def resolve_clef(spec: FieldSpec, answer: Answer | None, ctx: Ctx) -> Resolved:
    """Turn a Clef answer into a value and apply the choice thresholds.

    Args:
        spec: A Clef-engine field.
        answer: The Clef answer, or ``None`` when the response lacked one.
        ctx: The entity context (for the thresholds).

    Returns:
        The resolved answer: not ok on an abstaining noul, an unknown option,
        a chosen option below ``min_choice_prob`` or an answer whose confidence
        is below the field's ``min_confidence``. A choice that says
        ``unstated`` is ``ok`` with a ``None`` value.
    """
    if answer is None:
        return Resolved(ok=False, reason="missing_answer", detail="no answer returned")
    gates = ctx.run.gates
    raw = answer.model_dump(exclude={"type"})
    coerced = coerce(spec, answer, gates)
    if not coerced.ok:
        return Resolved(
            ok=False,
            reason=_code(coerced.reason),
            detail=coerced.reason or "",
            raw=raw,
            confidence=coerced.confidence,
        )
    if coerced.reason == "unstated":
        return Resolved(raw=raw, confidence=coerced.confidence, notes=("unstated",))
    if (
        isinstance(answer, ChoiceAnswer)
        and coerced.score is not None
        and coerced.score < gates.min_choice_prob
    ):
        return Resolved(
            ok=False,
            reason="low_choice_prob",
            detail=f"chosen option has p={coerced.score:.2f} < {gates.min_choice_prob}",
            raw=raw,
            confidence=coerced.confidence,
        )
    floor = spec.min_confidence or 0.0
    if coerced.confidence is not None and coerced.confidence < floor:
        return Resolved(
            ok=False,
            reason="low_confidence",
            detail=f"confidence {coerced.confidence:.2f} < {floor}",
            raw=raw,
            confidence=coerced.confidence,
        )
    return Resolved(value=coerced.value, raw=raw, confidence=coerced.confidence)


def accept(
    spec: FieldSpec, resolved: Resolved, null_threshold: float | None = None
) -> tuple[str, str, float | None] | None:
    """Apply the answer-level thresholds to a resolved Clairic answer.

    Args:
        spec: The field (it holds ``null_threshold`` and ``min_confidence``).
        resolved: The answer.
        null_threshold: Replaces the spec's ``null_threshold``.

    Returns:
        ``None`` when the answer stands, else ``(reason, detail, p)``. A null
        answer always stands.
    """
    if not resolved.ok:
        return (resolved.reason or "invalid", resolved.detail, None)
    if resolved.value is None:
        return None
    limit = null_threshold if null_threshold is not None else spec.null_threshold
    if resolved.p_null is not None and limit is not None and resolved.p_null > limit:
        return (
            "p_null",
            f"p_null {resolved.p_null:.2f} is above the threshold {limit}",
            resolved.p_null,
        )
    floor = spec.min_confidence or 0.0
    if resolved.confidence is not None and resolved.confidence < floor:
        return (
            "low_confidence",
            f"confidence {resolved.confidence:.2f} < {floor}",
            resolved.confidence,
        )
    return None


# --------------------------------------------------------------------------- #
# Re-asking
# --------------------------------------------------------------------------- #


async def _reask(
    ctx: Ctx,
    st: FieldState,
    *,
    spans: list[Span] | None,
    null_threshold: float | None = None,
) -> Resolved:
    """Ask one field again, alone, on windows around ``spans`` (or all the input)."""
    spec = st.spec
    windows = (
        mention_windows(len(ctx.text), spans, ctx.run.verify.window_chars)
        if spans
        else None
    )
    asked = await ask_clairic(
        ctx,
        {
            spec.wire_name: ctx.clairic_field(
                spec, suffix=REASK_SUFFIX, null_threshold=null_threshold
            )
        },
        meta=ctx.meta("reask", field=spec.wire_name),
        windows=windows,
    )
    answer, anchored = asked[spec.wire_name]
    return await resolve(ctx, spec, answer, anchored, st.type_answer)


def _effective(
    spec: FieldSpec, resolved: Resolved, null_threshold: float | None = None
) -> Resolved | None:
    """Return ``resolved`` when it passes the thresholds, else ``None``."""
    return None if accept(spec, resolved, null_threshold) else resolved


async def _adjudicate(
    ctx: Ctx, st: FieldState, first: Any, second: Any
) -> tuple[str | None, float | None]:
    """Ask Clef which of two disagreeing values the input states.

    Returns:
        ``("a" | "b", p)`` when one wins with ``p >= verify.adjudicate_min``,
        else ``(None, p)``.
    """
    spec = st.spec
    question = ChoiceQuestion(
        type="choice",
        instructions=(
            f"Which value does `input` state for {spec.description} of `entity.label`?"
        ),
        criteria={
            "a": display(first),
            "b": _NOT_STATED if second is None else display(second),
            "neither": "The input states neither",
        },
    )
    answers = await ctx.run.call_clef(
        {"adjudicate": question},
        ctx.entity_state(anchors=st.spans or None),
        meta=ctx.meta("adjudicate", field=spec.wire_name),
    )
    answer = answers.get("adjudicate")
    if not isinstance(answer, ChoiceAnswer):
        return None, None
    p = answer.probabilities.get(answer.choice)
    if (
        answer.choice in ("a", "b")
        and p is not None
        and p >= ctx.run.verify.adjudicate_min
    ):
        return answer.choice, p
    return None, p


def _adopt(st: FieldState, resolved: Resolved) -> None:
    st.resolved = resolved
    st.reject = None
    st.reject_detail = ""
    st.reject_p = None


async def _low_confidence(ctx: Ctx, st: FieldState) -> None:
    """Re-ask one answer; keep the better of two agreeing, adjudicate otherwise."""
    spec = st.spec
    first = st.resolved
    second = await _reask(ctx, st, spans=st.spans or None)
    st.reask = True
    checked = _effective(spec, second)
    if (
        checked is not None
        and checked.value is not None
        and agree(first.value, checked.value)
    ):
        if (checked.confidence or 0.0) > (first.confidence or 0.0):
            _adopt(st, checked)
        return
    other = checked.value if checked is not None else None
    verdict, p = await _adjudicate(ctx, st, first.value, other)
    if verdict == "a":
        return
    if verdict == "b":
        if checked is not None and checked.value is not None:
            _adopt(st, checked)
        else:
            st.notes.append("adjudicated_to_null")
            _adopt(st, dataclasses.replace(first, value=None))
        return
    st.set_reject(
        "adjudication_failed",
        "the re-ask disagreed and the adjudication was not decisive",
        p,
    )


# --------------------------------------------------------------------------- #
# Borderline null
# --------------------------------------------------------------------------- #


def _in_band(ctx: Ctx, st: FieldState) -> bool:
    p_null = st.resolved.p_null
    low, high = ctx.run.verify.null_band
    return (
        st.reject in (None, "p_null")
        and st.resolved.ok
        and p_null is not None
        and low <= p_null <= high
    )


async def _borderline(ctx: Ctx, targets: list[FieldState]) -> None:
    """Ask Clef whether the input states each borderline field, then act."""
    if not targets:
        return
    questions: dict[str, Question] = {
        f"nullcheck.{st.spec.wire_name}": NoulQuestion(
            type="noul",
            instructions=f"Does `input` state {ctx.desc(st.spec)}?",
            criteria=NoulCriteria(
                true="The input states it, directly or by clear implication.",
                false="The input does not state it.",
            ),
        )
        for st in targets
    }
    answers = await ctx.run.call_clef(
        questions, ctx.run.clef_state(ctx.text), meta=ctx.meta("null_check")
    )
    await gather(*(_borderline_one(ctx, st, answers) for st in targets))


async def _borderline_one(ctx: Ctx, st: FieldState, answers: dict[str, Answer]) -> None:
    verify = ctx.run.verify
    answer = answers.get(f"nullcheck.{st.spec.wire_name}")
    if not isinstance(answer, NoulAnswer):
        return
    p = answer.noul
    if st.value is None and p >= verify.null_confirm_yes:
        second = await _reask(ctx, st, spans=None, null_threshold=0.95)
        st.reask = True
        if (
            accept(st.spec, second, null_threshold=0.95) is None
            and second.value is not None
        ):
            _adopt(st, second)
    elif st.value is not None and p < verify.null_confirm_no:
        st.set_reject("null_check", "the input does not state it", p)


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #

_ATTR_TRUE = "The input states this about the entity `entity.label`."
_ATTR_FALSE = (
    "The input does not state this about `entity.label`: it says something "
    "else, concerns another entity, or does not say it."
)


def _attribution_question(
    spec: FieldSpec, value: Any, *, element: bool
) -> NoulQuestion:
    text = display(value)
    if element:
        instructions = f'Does `input` state this about `entity.label`: "{text}"?'
    else:
        instructions = (
            f'Does `input` state that {spec.description} of `entity.label` is "{text}"?'
        )
    return NoulQuestion(
        type="noul",
        instructions=instructions,
        criteria=NoulCriteria(true=_ATTR_TRUE, false=_ATTR_FALSE),
    )


def _is_list(st: FieldState) -> bool:
    return isinstance(st.resolved.value, list)


def _attr_id(st: FieldState, index: int | None) -> str:
    base = f"attr.{st.spec.wire_name}"
    return base if index is None else f"{base}.{index}"


def _attribution_questions(
    states: list[FieldState],
) -> dict[str, Question]:
    questions: dict[str, Question] = {}
    for st in states:
        if _is_list(st):
            for index, element in enumerate(st.resolved.value):
                questions[_attr_id(st, index)] = _attribution_question(
                    st.spec, element, element=True
                )
        else:
            questions[_attr_id(st, None)] = _attribution_question(
                st.spec, st.resolved.value, element=False
            )
    return questions


def _p(answers: dict[str, Answer], key: str) -> float | None:
    answer = answers.get(key)
    return answer.noul if isinstance(answer, NoulAnswer) else None


def _origin(st: FieldState, position: int) -> int:
    """Position, in the model's own answer list, of the element at ``position``."""
    origin = st.resolved.origin
    return (
        origin[position] if origin is not None and position < len(origin) else position
    )


def _drop(
    st: FieldState, position: int, value: Any, p: float | None, *, missing: bool
) -> None:
    """Record that a list element was dropped for lack of attribution."""
    reason = "attribution_missing" if missing else "attribution"
    st.dropped.append((_origin(st, position), reason, display(value), p))


async def _attribution(ctx: Ctx, targets: list[FieldState]) -> None:
    """Check that each value, and each list element, is stated about the entity."""
    if not targets:
        return
    questions = _attribution_questions(targets)
    if not questions:
        return
    anchors = [span for st in targets for span in st.spans]
    answers = await ctx.run.call_clef(
        questions,
        ctx.entity_state(anchors=anchors or None),
        meta=ctx.meta("attribution"),
    )
    await gather(*(_judge(ctx, st, answers) for st in targets))


async def _judge(ctx: Ctx, st: FieldState, answers: dict[str, Answer]) -> None:
    verify = ctx.run.verify
    if not _is_list(st):
        p = _p(answers, _attr_id(st, None))
        st.attribution_p = p
        if p is None:  # no answer is not a pass: check again, in isolation
            await _recheck_scalar(ctx, st)
            return
        if p >= verify.attribution_keep:
            return
        if p < verify.attribution_drop:
            st.set_reject(
                "attribution", "the input does not say this about the entity", p
            )
            return
        await _recheck_scalar(ctx, st)
        return
    elements = list(st.resolved.value)
    spans = list(st.resolved.spans) if st.resolved.spans else None
    probabilities: list[float] = []
    keep: list[bool] = []
    pending: list[int] = []
    for index in range(len(elements)):
        answered = _p(answers, _attr_id(st, index))
        p = math.nan if answered is None else answered
        probabilities.append(p)
        if answered is not None and p >= verify.attribution_keep:
            keep.append(True)
        elif answered is not None and p < verify.attribution_drop:
            keep.append(False)
            _drop(st, index, elements[index], p, missing=False)
        else:  # in the middle band, or unanswered (which is not a pass)
            keep.append(False)
            pending.append(index)
    st.attribution_p = tuple(probabilities)
    if pending:
        await _recheck_elements(ctx, st, elements, spans, pending, keep, probabilities)
    survivors = [i for i, flag in enumerate(keep) if flag]
    if not survivors:
        reasons = {reason for _, reason, _, _ in st.dropped}
        st.set_reject(
            "attribution_missing"
            if reasons == {"attribution_missing"}
            else "attribution",
            "no element is stated about the entity",
            None,
        )
        return
    st.resolved = dataclasses.replace(
        st.resolved,
        value=[elements[i] for i in survivors],
        spans=tuple(spans[i] for i in survivors) if spans else None,
    )
    if len(survivors) < len(elements):
        st.notes.append(
            f"dropped {len(elements) - len(survivors)} element(s): attribution"
        )


async def _recheck_scalar(ctx: Ctx, st: FieldState) -> None:
    """Middle band: re-ask in isolation, then check the attribution once more."""
    verify = ctx.run.verify
    second = await _reask(ctx, st, spans=st.spans or None)
    st.reask = True
    checked = _effective(st.spec, second)
    if checked is None or checked.value is None:
        st.set_reject("attribution", "the re-ask found no value to attribute", None)
        return
    spans = [checked.span] if checked.span else []
    answers = await ctx.run.call_clef(
        {
            _attr_id(st, None): _attribution_question(
                st.spec, checked.value, element=False
            )
        },
        ctx.entity_state(spans or None),
        meta=ctx.meta("attribution", field=st.spec.wire_name),
    )
    p = _p(answers, _attr_id(st, None))
    st.attribution_p = p
    if p is None:
        st.set_reject(
            "attribution_missing",
            "no attribution answer came back, even after a re-ask",
            None,
        )
    elif p >= verify.attribution_keep:
        _adopt(st, checked)
    else:
        st.set_reject(
            "attribution", "still not stated about the entity after a re-ask", p
        )


async def _recheck_elements(
    ctx: Ctx,
    st: FieldState,
    elements: list[Any],
    spans: list[Span] | None,
    pending: list[int],
    keep: list[bool],
    probabilities: list[float],
) -> None:
    """Middle band for list elements: one isolated re-ask, one more check each."""
    verify = ctx.run.verify
    around = [spans[i] for i in pending] if spans else None
    second = await _reask(ctx, st, spans=around)
    st.reask = True
    checked = _effective(st.spec, second)
    candidates: dict[int, tuple[Any, Span | None]] = {}
    if checked is not None and isinstance(checked.value, list):
        found = list(checked.value)
        found_spans = list(checked.spans) if checked.spans else None
        for index in pending:
            for j, other in enumerate(found):
                if agree(elements[index], other):
                    candidates[index] = (other, found_spans[j] if found_spans else None)
                    del found[j]
                    if found_spans:
                        del found_spans[j]
                    break
    for index in pending:
        if index not in candidates:
            was = probabilities[index]
            _drop(
                st,
                index,
                elements[index],
                None if math.isnan(was) else was,
                missing=math.isnan(was),
            )
    if not candidates:
        return
    questions: dict[str, Question] = {
        _attr_id(st, index): _attribution_question(st.spec, value, element=True)
        for index, (value, _) in candidates.items()
    }
    narrowed = [span for _, span in candidates.values() if span is not None]
    answers = await ctx.run.call_clef(
        questions,
        ctx.entity_state(narrowed or None),
        meta=ctx.meta("attribution", field=st.spec.wire_name),
    )
    for index, (value, _) in candidates.items():
        p = _p(answers, _attr_id(st, index))
        if p is not None:
            probabilities[index] = p
        if p is not None and p >= verify.attribution_keep:
            keep[index] = True
            elements[index] = value
        else:
            _drop(st, index, value, p, missing=p is None)
    st.attribution_p = tuple(probabilities)


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #


async def run_ladder(ctx: Ctx, states: list[FieldState]) -> None:
    """Run the verification ladder for an insert over every Clairic field.

    Args:
        ctx: The entity context (``ctx.run.verify`` switches the steps).
        states: One state per Clairic field, already resolved and thresholded.
            They are updated in place.
    """
    verify = ctx.run.verify
    if verify.reask_low_confidence:
        await gather(
            *(
                _low_confidence(ctx, st)
                for st in states
                if st.has_value()
                and st.resolved.confidence is not None
                and st.resolved.confidence < verify.low_confidence
            )
        )
    if verify.borderline_null:
        await _borderline(ctx, [st for st in states if _in_band(ctx, st)])
    if verify.attribution:
        await _attribution(
            ctx,
            [
                st
                for st in states
                if st.has_value() and not (_is_list(st) and not st.resolved.value)
            ],
        )
