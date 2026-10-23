"""The escalation tier: a stronger model for the cases the default setup fails.

Opt-in (``EscalationConfig``) and narrow. Four kinds, each a *proposal* that goes
through the verification ladder and the change gates like an answer of the default
setup; the model never writes, deletes or overrides a confident Clef answer:

* ``relative_dates``: a date or datetime that is invalid or null while its mention
  window holds a relative phrase. The model quotes the phrase and gives the date; code
  checks that the phrase is in the input, that the date parses and that it is what the
  phrase can mean (``_relative.plausible``).
* ``list_rows``: the rows of a nested ``list[Model]``. The model quotes each row's
  identity and its field values; every quote must occur verbatim in the input
  (whitespace folded) and is coerced like a Clairic answer. A row becomes a mention
  (discovery) or fills the null fields of a row the engine already has; nothing is
  ever deleted on its word.
* ``attribution_tiebreak``: an attribution still between ``attribution_drop`` and
  ``attribution_keep`` after the second look. ``{answer: yes|no}``: yes counts as
  ``prob_cap`` (0.8), no as ``1 - prob_cap``, like the hard picks of the Decisions
  API, so it can keep a value but never authorise a removal.
* ``dotted_discovery``: a top-level ``Many`` with a dotted identity that found no
  entity while presence says there are some. The model quotes each entity's
  descriptive label and its identity value; both must occur verbatim.

Every call is a ``CallRecord`` of kind ``escalate`` (``CallRecord.field`` starts with
the sub-kind) and says so in a warning.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any

from trustcall_system_one._coerce import coerce
from trustcall_system_one._relative import find_phrases, plausible
from trustcall_system_one._render import mention_windows
from trustcall_system_one._run import Run, gather
from trustcall_system_one._types import Mention
from trustcall_system_one._wire import ClairicAnswer
from trustcall_system_one.transport import CallMeta

if TYPE_CHECKING:
    from trustcall_system_one._compile import FieldSpec, SchemaPlan
    from trustcall_system_one._types import EntityInfo
    from trustcall_system_one._verify import Ctx, FieldState

__all__ = [
    "RowProposal",
    "escalate_discovery",
    "escalate_states",
    "locate",
    "route_discovery",
    "tiebreak",
]

Span = tuple[int, int]
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CLOCK = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
_MAX_INPUT_CHARS_PER_TOKEN = 3
_SCALAR_TYPES = frozenset({"text", "number", "date", "datetime", "time", "duration"})
_PREAMBLE = (
    "You help a data-extraction library with one narrow task. Answer only from what "
    "the text says; never guess, and never invent text: every quote you give must be "
    "copied exactly, character for character, from the text."
)


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #


def locate(
    source: str,
    quote: str,
    *,
    near: Span | None = None,
    taken: Sequence[Span] = (),
) -> Span | None:
    """Where ``quote`` occurs in ``source``, verbatim apart from white space.

    Args:
        source: The input.
        quote: What the model quoted.
        near: Prefer the occurrence closest to this span.
        taken: Occurrences already used (not returned again).

    Returns:
        The ``(start, end)`` of the occurrence, or ``None`` when the input does not
        contain the quote (the model made it up).
    """
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    found = [m.span() for m in re.finditer(pattern, source)]
    found = [span for span in found if span not in set(taken)]
    if not found:
        return None
    if near is None:
        return found[0]
    centre = (near[0] + near[1]) / 2
    return min(found, key=lambda s: abs((s[0] + s[1]) / 2 - centre))


def _windows(ctx: "Ctx") -> list[Span]:
    """The ranges of the input that concern the entity (all of it for a ``One``)."""
    spans = [m.span for m in (ctx.info.mentions if ctx.info else ()) if m.span]
    if not spans:
        return [(0, len(ctx.text))]
    return mention_windows(len(ctx.text), spans, ctx.run.verify.window_chars)


def _joined(text: str, windows: Sequence[Span]) -> str:
    return "\n[...]\n".join(text[a:b] for a, b in windows)


def _too_long(run: Run, text: str) -> bool:
    budget = getattr(run.escalation.transport, "state_token_budget", None)
    limit = int(budget) if isinstance(budget, int) else 100_000
    return len(text) > limit * _MAX_INPUT_CHARS_PER_TOKEN


def _meta(ctx_like: Any, sub: str, detail: str | None = None) -> CallMeta:
    """``CallMeta`` of an escalation call: the sub-kind leads ``field``."""
    return CallMeta(
        kind="escalate",
        schema=ctx_like.plan.name,
        entity=getattr(ctx_like, "key", None),
        field=f"{sub}:{detail}" if detail else sub,
        scope=getattr(ctx_like, "scope", ""),
    )


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    """A strict JSON schema: every property required, nothing else allowed."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_NULLABLE_TEXT = {"type": ["string", "null"]}


def _install(
    st: "FieldState",
    value: Any,
    sub: str,
    *,
    span: Span | None,
    raw: Any,
) -> None:
    """Put a verified proposal in a field's place: the ladder judges it next.

    There is no Clairic confidence or ``p_null`` to threshold, and no input-only
    Clairic pass could confirm it (that pass is the model that failed), so the state
    is marked ``escalated``: the attribution check (Clef) is what has to agree.
    """
    from trustcall_system_one._verify import Resolved

    st.resolved = Resolved(
        value=value,
        raw=raw,
        span=span,
        notes=(f"escalated:{sub}",),
    )
    st.reject = None
    st.reject_detail = ""
    st.reject_p = None
    st.escalated = sub
    st.notes.append(f"escalated:{sub}")


# --------------------------------------------------------------------------- #
# relative_dates
# --------------------------------------------------------------------------- #

_DATE_SCHEMA = _object(
    {
        "phrase": _NULLABLE_TEXT,
        "iso_date": _NULLABLE_TEXT,
        "time": _NULLABLE_TEXT,
    }
)
_RETRYABLE_REJECTS = frozenset({None, "unparseable", "invalid", "missing_answer"})


def _reference(run: Run) -> date | None:
    if not run.reference_date:
        return None
    try:
        return date.fromisoformat(run.reference_date[:10])
    except ValueError:
        return None


def _date_kind(spec: "FieldSpec") -> str | None:
    clairic = spec.clairic
    if clairic is None or clairic.type not in ("date", "datetime"):
        return None
    return clairic.type


def _failed(st: "FieldState") -> bool:
    """Whether a state has no usable value because it is null or did not parse."""
    if st.has_value() or st.escalated:
        return False
    return st.reject in _RETRYABLE_REJECTS


async def _relative_date(ctx: "Ctx", st: "FieldState", today: date) -> None:
    run, spec = ctx.run, st.spec
    kind = _date_kind(spec)
    # routed: asked whatever Clairic read, and its date is replaced when they differ
    routed = run.escalation.routes("relative_dates")
    if kind is None or st.escalated or not (routed or _failed(st)):
        return
    if not run.escalation_budget("relative_dates"):
        return
    windows = _windows(ctx)
    phrases = find_phrases(ctx.text, windows)
    if not phrases:  # nothing relative near the entity: nothing to ask
        return
    text = _joined(ctx.text, windows)
    if _too_long(run, text):
        run.warn(
            f"escalation_skipped: relative_dates: the input of {ctx.plan.name} is "
            "longer than the escalation model's input budget"
        )
        return
    wants_time = kind == "datetime"
    instructions = (
        f"{_PREAMBLE} Today is {today:%A}, {today.isoformat()}. A relative phrase "
        "such as 'yesterday', 'three days ago', 'in two weeks' or 'last Tuesday' "
        "names a date counted from today: 'last <weekday>' is the most recent such "
        "weekday before today, 'next <weekday>' the first one after today."
    )
    question = (
        f"Which relative date phrase of the text gives {ctx.desc(spec)}? Quote the "
        "phrase exactly as the text writes it in `phrase` and give the date it names "
        "in `iso_date` as YYYY-MM-DD"
        + (
            " and its clock time in `time` as HH:MM (24-hour), null if none"
            if wants_time
            else ""
        )
        + ". If the text states no such date by a relative phrase, answer null for "
        "every key."
    )
    payload = {
        "instructions": instructions,
        "input": f"Text:\n{text}\n\nAbout: {ctx.label}\n\n{question}",
        "schema": _DATE_SCHEMA,
        "schema_name": "relative_date",
    }
    answer = await run.call_escalate(
        "relative_dates", payload, _meta(ctx, "relative_dates", spec.pretty)
    )
    if answer is None:
        return
    verdict = _check_date(answer, ctx.text, windows, phrases, today, wants_time)
    where = f"{ctx.plan.name}.{spec.pretty}"
    if isinstance(verdict, str):
        run.warn(f"escalation:relative_dates: {where}: rejected ({verdict})")
        return
    value, span, quote = verdict
    if routed and st.has_value() and st.resolved.value == value:
        run.warn(f"escalation:relative_dates: {where}: routed, Clairic agrees")
        return
    run.warn(
        f"escalation:relative_dates: {where}: "
        f"{'routed, ' if routed else ''}proposed {value.isoformat()}"
    )
    _install(
        st, value, "relative_dates", span=None, raw={"phrase": quote, "span": span}
    )


def _check_date(
    answer: dict[str, Any],
    source: str,
    windows: Sequence[Span],
    phrases: Sequence[Any],
    today: date,
    wants_time: bool,
) -> "tuple[date | datetime, Span, str] | str":
    """Verify what the model said: a reason it is refused, or the date and its quote."""
    quote, iso = answer.get("phrase"), answer.get("iso_date")
    if quote is None and iso is None:
        return "the model found no relative date"
    if not isinstance(quote, str) or not isinstance(iso, str):
        return "phrase and iso_date must both be given"
    span = locate(source, quote)
    if span is None:
        return "the phrase is not in the input"
    if not any(w[0] <= span[0] and span[1] <= w[1] for w in windows):
        return "the phrase is outside the entity's mention window"
    inside = [p for p in phrases if span[0] <= p.start and p.end <= span[1]]
    if not inside:
        return "the quote holds no relative phrase"
    if not _ISO_DATE.fullmatch(iso.strip()):
        return "iso_date is not YYYY-MM-DD"
    try:
        found = date.fromisoformat(iso.strip())
    except ValueError:
        return "iso_date is not a real date"
    if not plausible(inside[0].text, found, today):
        return "the date cannot be what the phrase says"
    if not wants_time:
        return found, span, quote
    clock = answer.get("time")
    matched = _CLOCK.fullmatch(clock.strip()) if isinstance(clock, str) else None
    if matched is None:
        return "a datetime needs a clock time and none was given"
    stamp = datetime.combine(found, time(int(matched.group(1)), int(matched.group(2))))
    return stamp, span, quote


# --------------------------------------------------------------------------- #
# attribution_tiebreak
# --------------------------------------------------------------------------- #

_YESNO_SCHEMA = _object({"answer": {"type": "string", "enum": ["yes", "no"]}})


async def tiebreak(
    ctx: "Ctx",
    st: "FieldState",
    instructions: Any,
    *,
    span: Span | None,
    detail: str | None = None,
) -> float | None:
    """Settle an attribution that two looks left between the drop and keep bars.

    Args:
        ctx: The entity context.
        st: The field whose value is in question.
        instructions: The attribution question as Clef was asked it (it names
            ``input`` and ``entity.label``).
        span: Where the value was found, to narrow the text to.
        detail: Distinguishes the elements of a list in the call log.

    Returns:
        ``prob_cap`` for yes, ``1 - prob_cap`` for no, or ``None`` when the
        escalation did not answer (off, a limit, a failure): the caller then does what
        it did without the tier. A state the escalation itself proposed is never asked
        (it would confirm itself).
    """
    run = ctx.run
    if st.escalated or not run.escalation_budget("attribution_tiebreak"):
        return None
    anchors = [span] if span else []
    windows = (
        mention_windows(
            len(ctx.text),
            [
                *anchors,
                *[m.span for m in (ctx.info.mentions if ctx.info else ()) if m.span],
            ],
            run.verify.window_chars,
        )
        if anchors
        else [(0, len(ctx.text))]
    )
    text = _joined(ctx.text, windows)
    if _too_long(run, text):
        return None
    claim = (
        str(instructions)
        .replace("`entity.label`", f'"{ctx.label}"')
        .replace("`input`", "the text")
    )
    payload = {
        "instructions": f"{_PREAMBLE} Answer yes only when the text says it or it "
        "follows from the text; otherwise answer no.",
        "input": f"Text:\n{text}\n\nQuestion: {claim}",
        "schema": _YESNO_SCHEMA,
        "schema_name": "yes_or_no",
    }
    answer = await run.call_escalate(
        "attribution_tiebreak",
        payload,
        _meta(ctx, "attribution_tiebreak", f"{st.spec.wire_name}{detail or ''}"),
    )
    if answer is None or answer.get("answer") not in ("yes", "no"):
        return None
    cap = run.escalation.prob_cap
    p = cap if answer["answer"] == "yes" else 1.0 - cap
    run.warn(
        f"escalation:attribution_tiebreak: {ctx.plan.name}.{st.spec.pretty}: "
        f"{answer['answer']} (counted as {p:.2f})"
    )
    return p


# --------------------------------------------------------------------------- #
# list_rows and dotted_discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RowProposal:
    """One row of a nested list as the model quoted it, after verification.

    Attributes:
        identity: The quote that names the row (verbatim in the input).
        span: Where it is in the input.
        values: Field path (``FieldSpec.pretty``) to ``(coerced value, span of the
            quote)``, only for the quotes that occur in the input and coerce.
    """

    identity: str
    span: Span
    values: "dict[str, tuple[Any, Span | None]]"


def _row_specs(plan: "SchemaPlan") -> "list[FieldSpec]":
    return [
        spec
        for spec in plan.all_specs()
        if spec.engine == "clairic"
        and spec.clairic is not None
        and spec.clairic.type in _SCALAR_TYPES
        and spec.post != "union"
    ]


def _verify_rows(
    answer: dict[str, Any], source: str, specs: "Sequence[FieldSpec]"
) -> list[RowProposal]:
    """Keep the rows and values whose quotes occur in the input and coerce."""
    rows = answer.get("rows")
    out: list[RowProposal] = []
    taken: list[Span] = []
    if not isinstance(rows, list):
        return out
    by_name = {spec.pretty: spec for spec in specs}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("identity"), str):
            continue
        quote = row["identity"]
        span = locate(source, quote, taken=taken)
        if span is None:  # an identity that is not in the input is made up
            continue
        taken.append(span)
        values: dict[str, tuple[Any, Span | None]] = {}
        fields = row.get("fields")
        for name, text in fields.items() if isinstance(fields, dict) else ():
            spec = by_name.get(name)
            if spec is None or not isinstance(text, str) or not text.strip():
                continue
            where = locate(source, text, near=span)
            if where is None:
                continue
            assert spec.clairic is not None
            coerced = coerce(spec, ClairicAnswer(type=spec.clairic.type, value=text))
            if coerced.ok and coerced.value is not None:
                values[name] = (coerced.value, where)
        out.append(RowProposal(quote, span, values))
    return out


async def _ask_rows(
    run: Run,
    plan: "SchemaPlan",
    parent: "EntityInfo | None",
    source: str,
    meta_like: Any,
    noun: str,
) -> list[RowProposal]:
    specs = _row_specs(plan)
    if not specs or _too_long(run, source):
        return []
    names = ", ".join(
        f"{spec.pretty} ({spec.description or spec.pretty})" for spec in specs
    )
    fields_schema = _object({spec.pretty: _NULLABLE_TEXT for spec in specs})
    of = f" of {parent.label}" if parent is not None else ""
    payload = {
        "instructions": f"{_PREAMBLE} The text lists the rows{of} of a table or list "
        f"(each row is one {noun}). Report every row, once each.",
        "input": (
            f"Text:\n{source}\n\nFor each {noun}{of}, give `identity`: the exact words "
            "that label the row (its position code, number, name or model), and "
            "`fields`: for each field below the exact words of the text that give its "
            "value in that row, or null when the row does not state it. Fields: "
            f"{names}."
        ),
        "schema": _object(
            {
                "rows": {
                    "type": "array",
                    "items": _object(
                        {"identity": {"type": "string"}, "fields": fields_schema}
                    ),
                }
            }
        ),
        "schema_name": "rows",
    }
    answer = await run.call_escalate(
        "list_rows", payload, _meta(meta_like, "list_rows")
    )
    if answer is None:
        return []
    rows = _verify_rows(answer, source, specs)
    asked = answer.get("rows")
    dropped = (len(asked) if isinstance(asked, list) else 0) - len(rows)
    run.warn(
        f"escalation:list_rows: {plan.name}: {len(rows)} row(s) verified in the "
        f"input" + (f", {dropped} dropped (not in the input)" if dropped > 0 else "")
    )
    return rows


async def rows_for(ctx: "Ctx") -> list[RowProposal]:
    """The verified rows of this entity's list (one call per list and invoke)."""
    key = ("rows", ctx.scope, ctx.plan.name, ctx.lineage)
    return await ctx.run.once(
        key,
        lambda: _ask_rows(
            ctx.run, ctx.plan, ctx.parent, ctx.text, ctx, ctx.plan.title.lower()
        ),
    )


def _row_of(ctx: "Ctx", rows: Sequence[RowProposal]) -> RowProposal | None:
    """The proposed row that is this entity: its identity lies on a mention."""
    spans = [m.span for m in (ctx.info.mentions if ctx.info else ()) if m.span]
    for row in rows:
        for a, b in spans:
            if a < row.span[1] and row.span[0] < b:
                return row
    texts = {" ".join(m.text.split()) for m in (ctx.info.mentions if ctx.info else ())}
    return next((r for r in rows if " ".join(r.identity.split()) in texts), None)


def _incomplete(ctx: "Ctx", states: "Sequence[FieldState]") -> bool:
    """Whether a nested row came back empty, or without a field it cannot do without.

    On an update, whether nothing at all was proposed for a row the input mentions.
    """
    clairic = [st for st in states if st.spec.engine == "clairic"]
    if not clairic:
        return False
    if ctx.existing is not None:
        return not any(st.has_value() for st in clairic)
    if not any(st.has_value() for st in clairic):
        return True
    return any(st.spec.required and not st.has_value() for st in clairic)


async def _fill_row(ctx: "Ctx", states: "Sequence[FieldState]") -> None:
    run = ctx.run
    if ctx.parent is None or not run.escalation.runs("list_rows"):
        return
    routed = run.escalation.routes("list_rows")
    if not routed and not _incomplete(ctx, states):
        return
    row = _row_of(ctx, await rows_for(ctx))
    if row is None:
        return
    filled = []
    for st in states:
        got = row.values.get(st.spec.pretty)
        if got is None or st.escalated:
            continue
        if routed:
            # the tier's reading of the row stands in for Clairic's where they differ
            if st.has_value() and st.resolved.value == got[0]:
                continue
        elif st.has_value() or st.reject not in (None, *_FILLABLE):
            continue
        value, span = got
        _install(st, value, "list_rows", span=span, raw=row.identity)
        filled.append(st.spec.pretty)
    if filled:
        run.warn(
            f"escalation:list_rows: {ctx.plan.name}: "
            f"{'routed, ' if routed else ''}filled {', '.join(filled)} from "
            "the proposed row; the verification ladder judges them"
        )


_FILLABLE = frozenset(
    {"unparseable", "invalid", "missing_answer", "p_null", "low_confidence"}
)


async def escalate_states(ctx: "Ctx", states: "Sequence[FieldState]") -> None:
    """The per-field and per-row escalations of one entity, before its ladder.

    Args:
        ctx: The entity context.
        states: Every Clairic field state of the entity (they are changed in place).
    """
    run = ctx.run
    if not run.escalation.enabled:
        return
    today = _reference(run)
    if today is not None and run.escalation.runs("relative_dates"):
        await gather(*(_relative_date(ctx, st, today) for st in states))
    await _fill_row(ctx, states)


async def escalate_discovery(
    run: Run,
    *,
    plan: "SchemaPlan",
    source: str,
    parent: "EntityInfo | None",
    scope_key: str,
    noun: str,
    identifier: str,
    dotted: bool,
    found: int,
    count: int | None,
    present: bool,
    meta_like: Any,
    lineage: tuple[str, ...] = (),
) -> list[Mention]:
    """Mentions the model found where discovery found none (or too few).

    Args:
        run: The invoke's context.
        plan: The item plan of the collection.
        source: The input.
        parent: The entity whose list this is, or ``None`` at the top level.
        scope_key: ``CallMeta.scope`` of the collection.
        noun: What one entity is called.
        identifier: The identity as words (``msn``).
        dotted: Whether the identity is a dotted path through sub-models.
        found: How many mentions discovery found.
        count: Clairic's count, when it gave one.
        present: Whether presence says the kind is there (or was not asked).
        meta_like: Supplies ``plan``, ``scope`` and ``key`` for the call log.
        lineage: The ancestors of a nested collection (they keep two lists of the same
            item model apart).

    Returns:
        Mentions (the quoted label or identity, with spans) for the collection's own
        grouping, qualification and extraction; empty when nothing applies.
    """
    if not run.escalation.enabled:
        return []
    if parent is None:
        if not (dotted and present and found == 0):
            return []
        if not run.escalation.runs("dotted_discovery"):
            return []
        key = ("dotted", plan.name)
        return await run.once(
            key,
            lambda: _ask_entities(run, plan, source, noun, identifier, meta_like),
        )
    if not run.escalation.runs("list_rows"):
        return []
    short = (found == 0 and present) or (count is not None and count > found)
    if not short:
        return []
    rows = await run.once(
        ("rows", scope_key, plan.name, lineage),
        lambda: _ask_rows(run, plan, parent, source, meta_like, noun),
    )
    return [Mention(row.identity, row.span) for row in rows]


async def route_discovery(
    run: Run,
    *,
    plan: "SchemaPlan",
    source: str,
    parent: "EntityInfo | None",
    scope_key: str,
    noun: str,
    identifier: str,
    dotted: bool,
    meta_like: Any,
    lineage: tuple[str, ...] = (),
) -> list[Mention]:
    """Mentions asked of the tier BEFORE Clairic's discovery (``mode="route"``).

    A nested list (its rows) or a top-level ``Many`` with a dotted identity (its
    entities). Empty when the mode, the kind or the budget says no, when the call
    failed or when none of its quotes is in the input: the normal discovery then runs
    as it would without the tier.
    """
    if parent is None:
        if not (dotted and run.escalation.routes("dotted_discovery")):
            return []
        return await run.once(
            ("dotted", plan.name),
            lambda: _ask_entities(run, plan, source, noun, identifier, meta_like),
        )
    if not run.escalation.routes("list_rows"):
        return []
    rows = await run.once(
        ("rows", scope_key, plan.name, lineage),
        lambda: _ask_rows(run, plan, parent, source, meta_like, noun),
    )
    return [Mention(row.identity, row.span) for row in rows]


async def _ask_entities(
    run: Run,
    plan: "SchemaPlan",
    source: str,
    noun: str,
    identifier: str,
    meta_like: Any,
) -> list[Mention]:
    if _too_long(run, source):
        return []
    payload = {
        "instructions": f"{_PREAMBLE} The text mentions {noun}s; each is identified "
        f"by its {identifier or 'identity'}, which the text may give in a table cell "
        "or a heading far from the words that describe it.",
        "input": (
            f"Text:\n{source}\n\nList every {noun} the text describes, once each. For "
            f"each give `label`: the exact words the text uses to name or describe it "
            f"(its model, type, name or heading), and `identity`: its "
            f"{identifier or 'identity'} "
            "exactly as the text writes it."
        ),
        "schema": _object(
            {
                "entities": {
                    "type": "array",
                    "items": _object(
                        {"label": {"type": "string"}, "identity": {"type": "string"}}
                    ),
                }
            }
        ),
        "schema_name": "entities",
    }
    answer = await run.call_escalate(
        "dotted_discovery", payload, _meta(meta_like, "dotted_discovery")
    )
    if answer is None:
        return []
    entities = answer.get("entities")
    out: list[Mention] = []
    taken: list[Span] = []
    for item in entities if isinstance(entities, list) else ():
        if not isinstance(item, dict):
            continue
        label, ident = item.get("label"), item.get("identity")
        if not (isinstance(label, str) and isinstance(ident, str)):
            continue
        span = locate(source, label, taken=taken)
        if span is None or locate(source, ident) is None:  # both must be in the input
            continue
        taken.append(span)
        out.append(Mention(" ".join(label.split()), span))
    asked = len(entities) if isinstance(entities, list) else 0
    run.warn(
        f"escalation:dotted_discovery: {plan.name}: {len(out)} of {asked} "
        "entit(ies) had both quotes in the input"
    )
    return sorted(out, key=lambda m: m.span or (0, 0))
