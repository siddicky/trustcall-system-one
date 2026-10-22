"""The single-entity insert pass: presence, extraction, typed follow-ups, assembly.

``insert_one`` extracts one entity of a flat schema from the user's text:

1. presence gate (Clef ``noul``), skipped when the schema is forced;
2. Clairic and Clef extraction, in parallel;
3. typed follow-ups (elements of ``list[Literal | Enum]``, primitive unions) and
   answer-level thresholds;
4. the verification ladder (``_verify.run_ladder``);
5. assembly: build the document, validate it, and describe how every field got its
   value.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from trustcall_system_one._compile import FieldSpec, SchemaPlan
from trustcall_system_one._errors import ExtractionValidationError, StateTruncatedError
from trustcall_system_one._flatten import prune_nulls, unflatten
from trustcall_system_one._names import Path
from trustcall_system_one._render import RenderedState
from trustcall_system_one._run import Answer, Run, both, gather
from trustcall_system_one._types import (
    EntityInfo,
    FieldAnswer,
    RejectedOp,
    ResultRecord,
    json_pointer,
)
from trustcall_system_one._verify import (
    Ctx,
    FieldState,
    Resolved,
    accept,
    resolve,
    resolve_clef,
    run_ladder,
)
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ClairicAnswer,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
)

__all__ = ["InsertOutcome", "insert_one"]

Span = tuple[int, int]

_MILESTONE = {
    "submodel": "M4",
    "union_models": "M4",
    "collection": "M9",
    "mapping": "M9",
}
_HEAD_WINDOWS = 4  # a truncated state with no spans is retried on this many windows


@dataclass(frozen=True)
class InsertOutcome:
    """What ``insert_one`` produced.

    Attributes:
        record: The record, or ``None`` when the presence gate found no entity.
        extracted: Whether any extraction request was made.
    """

    record: ResultRecord | None
    extracted: bool


def _note_unsupported(run: Run, plan: SchemaPlan) -> None:
    """Warn about fields this milestone leaves to their pydantic default."""
    for spec in plan.specs:
        milestone = _MILESTONE.get(spec.engine)
        if milestone is not None:
            run.warn(
                f"{plan.name}.{spec.pretty}: {spec.engine} fields are not "
                f"extracted until {milestone}; left unset so the model default "
                "applies"
            )
    for message in plan.warnings:
        run.warn(f"{plan.name}: {message}")


async def _presence(ctx: Ctx) -> float:
    """Ask Clef whether the input contains an entity of the schema at all."""
    plan = ctx.plan
    title = plan.title
    article = "an" if title[:1].lower() in "aeiou" else "a"
    thing = f"{article} {title}"
    described = " ".join(plan.description.split()).rstrip(" .!?:;")
    question = NoulQuestion(
        type="noul",
        instructions=(
            f"Does `input` contain {thing}"
            + (f": {described}" if described else "")
            + "?"
        ),
        criteria=NoulCriteria(
            true=f"The input describes {thing}.",
            false=f"The input does not describe {thing}.",
        ),
    )
    answers = await ctx.run.call_clef(
        {"presence": question},
        ctx.run.clef_state(ctx.text),
        meta=ctx.meta("presence"),
    )
    answer = answers["presence"]
    assert isinstance(answer, NoulAnswer)
    return answer.noul


def _spans_in(
    answers: dict[str, ClairicAnswer], anchored: RenderedState | None
) -> list[Span]:
    """Collect the answers' spans, mapped to the original input."""
    if anchored is None:
        return []
    found: list[Span] = []
    for answer in answers.values():
        pairs = [answer.span] if answer.span else []
        pairs += answer.spans or []
        for pair in pairs:
            if len(pair) == 2:
                mapped = anchored.to_original((pair[0], pair[1]))
                if mapped is not None:
                    found.append(mapped)
    return found


async def _clairic_pass(
    ctx: Ctx, specs: list[FieldSpec]
) -> dict[str, tuple[ClairicAnswer | None, RenderedState | None]]:
    """One Clairic request for ``specs``; a truncated state is retried narrowed."""
    run = ctx.run
    fields = {spec.wire_name: ctx.clairic_field(spec) for spec in specs}
    single = specs[0].wire_name if len(specs) == 1 else None
    meta = ctx.meta("extract", field=single)
    rendered = ctx.rendered()
    response = await run.call_clairic(fields, rendered, meta=meta)
    anchored = run.anchor(rendered, response)
    if response.truncated:
        spans = _spans_in(response.answers, anchored)
        window = run.verify.window_chars
        windows = None
        if not spans:  # nothing to anchor on: Clairic reads the head, so keep it
            windows = [(0, min(len(ctx.text), _HEAD_WINDOWS * window))]
        narrowed = ctx.rendered(spans or None, windows=windows)
        response = await run.call_clairic(fields, narrowed, meta=meta)
        anchored = run.anchor(narrowed, response)
        if response.truncated:
            message = (
                f"Clairic reported a truncated state for {ctx.plan.name} even "
                "after narrowing it to windows around the mentions"
            )
            if run.strict:
                raise StateTruncatedError(message)
            run.warn(f"clairic_truncated: {message}")
    return {name: (response.answers.get(name), anchored) for name in fields}


async def _extract(
    ctx: Ctx, specs: list[FieldSpec]
) -> dict[str, tuple[ClairicAnswer | None, RenderedState | None]]:
    if not specs:
        return {}
    groups = [[spec] for spec in specs] if ctx.run.passes.split == "field" else [specs]
    parts = await gather(*(_clairic_pass(ctx, group) for group in groups))
    merged: dict[str, tuple[ClairicAnswer | None, RenderedState | None]] = {}
    for part in parts:
        merged.update(part)
    return merged


async def _clef_pass(ctx: Ctx, specs: list[FieldSpec]) -> dict[str, Answer]:
    """One Clef request set with the questions of every Clef-backed field."""
    questions = {spec.clef_name: ctx.clef_question(spec) for spec in specs}
    if not questions:
        return {}
    return await ctx.run.call_clef(
        questions, ctx.run.clef_state(ctx.text), meta=ctx.meta("clef_fields")
    )


async def _resolve_clairic(
    ctx: Ctx,
    spec: FieldSpec,
    extracted: tuple[ClairicAnswer | None, RenderedState | None],
    clef_answers: dict[str, Answer],
) -> FieldState:
    answer, anchored = extracted
    type_answer = None
    if spec.post == "union":
        candidate = clef_answers.get(spec.clef_name)
        type_answer = candidate if isinstance(candidate, ChoiceAnswer) else None
    resolved = await resolve(ctx, spec, answer, anchored, type_answer)
    state = FieldState(spec, resolved, type_answer)
    rejection = accept(spec, resolved)
    if rejection is not None:
        state.set_reject(*rejection)
    return state


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _source(spec: FieldSpec, value: Any) -> Any:
    if value is None and spec.has_default:
        return "default"
    return spec.engine


def _unique(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _clairic_answer(state: FieldState) -> tuple[FieldAnswer, list[RejectedOp]]:
    spec, resolved = state.spec, state.resolved
    pointer = json_pointer(spec.path)
    value = state.value
    rejected: list[RejectedOp] = []
    if state.reject:
        rejected.append(
            RejectedOp(None, pointer, state.reject, state.reject_detail, state.reject_p)
        )
    for position, reason, detail, p in state.dropped:
        rejected.append(RejectedOp(None, f"{pointer}/{position}", reason, detail, p))
    answer = FieldAnswer(
        path=spec.path,
        pretty=spec.pretty,
        wire_name=spec.wire_name,
        engine=spec.engine,
        value=value,
        raw=resolved.raw,
        p_null=resolved.p_null,
        confidence=resolved.confidence,
        guess=resolved.guess,
        span=resolved.span,
        spans=resolved.spans,
        source=_source(spec, value),
        attribution_p=state.attribution_p,
        reask=state.reask,
        rejected_reason=state.reject,
        warnings=_unique([*resolved.notes, *state.notes]),
    )
    return answer, rejected


def _clef_answer(
    spec: FieldSpec, resolved: Resolved
) -> tuple[FieldAnswer, list[RejectedOp]]:
    rejected: list[RejectedOp] = []
    reason = None if resolved.ok else (resolved.reason or "invalid")
    if reason is not None:
        rejected.append(
            RejectedOp(None, json_pointer(spec.path), reason, resolved.detail, None)
        )
    value = resolved.value if resolved.ok else None
    answer = FieldAnswer(
        path=spec.path,
        pretty=spec.pretty,
        wire_name=spec.wire_name,
        engine=spec.engine,
        value=value,
        raw=resolved.raw,
        confidence=resolved.confidence,
        source=_source(spec, value),
        rejected_reason=reason,
        warnings=_unique(list(resolved.notes)),
    )
    return answer, rejected


def _const_answer(spec: FieldSpec) -> FieldAnswer:
    assert spec.key_map is not None
    value = next(iter(spec.key_map.values()))
    return FieldAnswer(
        path=spec.path,
        pretty=spec.pretty,
        wire_name="",
        engine="const",
        value=value,
        raw=value,
        source="const",
    )


def _error_strings(exc: ValidationError) -> list[str]:
    errors = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        errors.append(f"{where}: {error['msg']}" if where else error["msg"])
    return errors


def _assemble(
    ctx: Ctx,
    answers: dict[str, FieldAnswer],
    rejected: list[RejectedOp],
    qualify_p: float | None,
) -> ResultRecord:
    plan = ctx.plan
    values: dict[Path, Any] = {
        spec.path: answers[spec.pretty].value
        for spec in plan.specs
        if spec.pretty in answers
    }
    document = prune_nulls(unflatten(values, plan=plan), plan)
    entity = EntityInfo(key=ctx.key, label=ctx.label, qualify_p=qualify_p)
    model: BaseModel | None
    try:
        model = plan.schema.model_validate(document)
    except ValidationError as exc:
        errors = _error_strings(exc)
        record = ResultRecord(
            schema_name=plan.name,
            json_doc_id=None,
            kind="insert",
            value=None,
            after=document,
            rejected=tuple(rejected),
            fields=answers,
            entity=entity,
            valid=False,
            errors=tuple(errors),
        )
        if ctx.run.strict:
            raise ExtractionValidationError(
                f"{plan.name} failed validation: " + "; ".join(errors),
                errors=errors,
                record=record,
                calls=tuple(ctx.run.calls),
            ) from exc
        return record
    # Report the values as the model holds them (enums, sets, ...).
    final = {
        key: dataclasses.replace(answer, value=getattr(model, str(answer.path[0])))
        if answer.value is not None and hasattr(model, str(answer.path[0]))
        else answer
        for key, answer in answers.items()
    }
    return ResultRecord(
        schema_name=plan.name,
        json_doc_id=None,
        kind="insert",
        value=model,
        after=model.model_dump(),
        rejected=tuple(rejected),
        fields=final,
        entity=entity,
        valid=True,
    )


async def insert_one(
    run: Run,
    plan: SchemaPlan,
    text: str,
    *,
    label: str,
    skip_presence: bool = False,
) -> InsertOutcome:
    """Extract one entity of a flat schema from ``text``.

    Args:
        run: The invoke's context.
        plan: The compiled schema. Nested models, unions of models, collections
            and mappings are left unset (with a warning); a required field that
            depends on one makes the record invalid.
        text: The user's input.
        label: How prompts name the entity (``"the Person"``).
        skip_presence: Skip the presence gate (the schema was forced).

    Returns:
        The outcome: no record when the gate finds no entity, else an insert
        record that is valid or not.

    Raises:
        ExtractionValidationError: In strict mode, if validation fails.
        StateTruncatedError: In strict mode, if Clairic keeps truncating.
        TransportError: If a request failed (it is in ``run.calls`` first).
    """
    ctx = Ctx(run=run, plan=plan, text=text, label=label)
    _note_unsupported(run, plan)
    qualify_p: float | None = None
    if not skip_presence:
        qualify_p = await _presence(ctx)
        if qualify_p < run.presence.threshold_absent:
            return InsertOutcome(None, extracted=False)

    clairic_specs = [spec for spec in plan.specs if spec.engine == "clairic"]
    clef_specs = [spec for spec in plan.specs if spec.clef is not None]
    extracted, clef_answers = await both(
        _extract(ctx, clairic_specs), _clef_pass(ctx, clef_specs)
    )

    states = await gather(
        *(
            _resolve_clairic(ctx, spec, extracted[spec.wire_name], clef_answers)
            for spec in clairic_specs
        )
    )
    await run_ladder(ctx, list(states))

    by_name = {state.spec.pretty: state for state in states}
    answers: dict[str, FieldAnswer] = {}
    rejected: list[RejectedOp] = []
    for spec in plan.specs:
        if spec.engine == "clairic":
            answer, rejects = _clairic_answer(by_name[spec.pretty])
        elif spec.engine == "clef":
            answer, rejects = _clef_answer(
                spec, resolve_clef(spec, clef_answers.get(spec.clef_name), ctx)
            )
        elif spec.engine == "const":
            answer, rejects = _const_answer(spec), []
        else:
            continue
        answers[spec.pretty] = answer
        rejected.extend(rejects)
    return InsertOutcome(_assemble(ctx, answers, rejected, qualify_p), extracted=True)
