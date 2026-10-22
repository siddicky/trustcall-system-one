"""The single-entity insert pass: presence, groups, typed follow-ups, assembly.

``insert_one`` extracts one entity of a schema from the user's text:

1. presence gate (Clef ``noul``), skipped when the schema is forced;
2. the root group: Clairic extraction, the Clef questions of its Clef-typed
   fields and of its unions of models, in parallel, while its sub-models run
   concurrently (an optional sub-model is gated by its own presence question and
   skipped when the input does not state it; a union's chosen member is
   extracted once the choice is known);
3. typed follow-ups (elements of ``list[Literal | Enum]``, primitive unions) and
   answer-level thresholds;
4. the verification ladder (``_verify.run_ladder``) over every leaf of every
   group at once;
5. assembly: decide which optional sub-models and union members stand, build the
   nested document, validate it, and describe how every field got its value.
"""

import asyncio
import contextlib
import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from trustcall_system_one._compile import (
    FieldGroup,
    FieldSpec,
    SchemaPlan,
    UnionGroup,
    UnionMember,
)
from trustcall_system_one._errors import ExtractionValidationError
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
    ask_clairic,
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
Extracted = dict[str, tuple[ClairicAnswer | None, RenderedState | None]]

_MILESTONE = {"collection": "M9", "mapping": "M9"}


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
    for spec in plan.deferred():
        run.warn(
            f"{plan.name}.{spec.pretty}: {spec.engine} fields are not "
            f"extracted until {_MILESTONE[spec.engine]}; left unset so the model "
            "default applies"
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


# --------------------------------------------------------------------------- #
# Walking the groups
# --------------------------------------------------------------------------- #


@dataclass
class _Leaf:
    """One field of a group and how it was answered.

    Exactly one of ``state`` (a Clairic field), ``resolved`` (a Clef field) or
    ``filled`` (a value set by code: a const or a union's tag) is used.
    """

    spec: FieldSpec
    state: FieldState | None = None
    resolved: Resolved | None = None
    filled: bool = False
    value: Any = None

    def final(self) -> Any:
        """The value that stands, ``None`` when null or rejected."""
        if self.state is not None:
            return self.state.value
        if self.resolved is not None:
            return self.resolved.value if self.resolved.ok else None
        return self.value


@dataclass
class _GroupResult:
    """A group as it was asked: its leaves, sub-models and unions.

    ``presence`` is the presence probability of an optional group (``None`` for
    a group that is always asked); ``present`` is false when that probability
    kept the group from being asked and, after the ladder, when a borderline
    group came back all null.
    """

    group: FieldGroup
    present: bool = True
    presence: float | None = None
    borderline: bool = False
    leaves: list[_Leaf] = field(default_factory=list)
    children: list["_GroupResult"] = field(default_factory=list)
    unions: list["_UnionResult"] = field(default_factory=list)


@dataclass
class _UnionResult:
    """A union of models: the choice and the member subtrees that were asked."""

    union: UnionGroup
    choice: Answer | None
    resolved: Resolved
    chosen: UnionMember | None = None
    members: dict[str, _GroupResult] = field(default_factory=dict)
    fallback: bool = False


@dataclass
class _Unit:
    """The groups that are asked together when ``passes.split == "entity"``."""

    shared: "asyncio.Future[Extracted] | None" = None


def _group_meta_field(group: FieldGroup) -> str | None:
    return group.pretty or None


async def _clairic_pass(
    ctx: Ctx, specs: list[FieldSpec], field_: str | None
) -> Extracted:
    """One Clairic request for ``specs`` (chunked if the input is too long)."""
    fields = {spec.wire_name: ctx.clairic_field(spec) for spec in specs}
    single = specs[0].wire_name if len(specs) == 1 else field_
    return await ask_clairic(
        ctx,
        fields,
        meta=ctx.meta("extract", field=single),
        descriptions={spec.wire_name: spec.scoped for spec in specs},
    )


async def _extract(
    ctx: Ctx, specs: list[FieldSpec], unit: _Unit, group: FieldGroup
) -> Extracted:
    """Ask for the Clairic fields of one group, as ``passes.split`` says."""
    if not specs:
        return {}
    if unit.shared is not None:
        shared = await unit.shared
        return {spec.wire_name: shared[spec.wire_name] for spec in specs}
    where = _group_meta_field(group)
    if ctx.run.passes.split == "field":
        parts = await gather(*(_clairic_pass(ctx, [spec], where) for spec in specs))
    else:
        parts = [await _clairic_pass(ctx, specs, where)]
    merged: Extracted = {}
    for part in parts:
        merged.update(part)
    return merged


async def _clef_pass(
    ctx: Ctx, specs: list[FieldSpec], group: FieldGroup
) -> dict[str, Answer]:
    """One Clef request set with the questions of every Clef-backed field."""
    questions = {spec.clef_name: ctx.clef_question(spec) for spec in specs}
    if not questions:
        return {}
    return await ctx.run.call_clef(
        questions,
        ctx.run.clef_state(ctx.text),
        meta=ctx.meta("clef_fields", field=_group_meta_field(group)),
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


async def _ask_presence(ctx: Ctx, group: FieldGroup) -> float:
    """Ask Clef whether the input states an optional sub-model."""
    spec = group.spec
    assert spec is not None and spec.clef is not None
    answers = await ctx.run.call_clef(
        {spec.clef_name: ctx.clef_question(spec)},
        ctx.run.clef_state(ctx.text),
        meta=ctx.meta("presence", field=group.pretty),
    )
    answer = answers.get(spec.clef_name)
    if not isinstance(answer, NoulAnswer):
        # no answer is not "absent": ask for the group, the ladder will judge it
        return ctx.run.presence.threshold_present
    return answer.noul


async def _run_child(ctx: Ctx, group: FieldGroup, unit: _Unit) -> _GroupResult:
    """Run a sub-model group, after its presence question when it is optional."""
    if not group.optional:
        return await _run_group(ctx, group, unit, None)
    presence = ctx.run.presence
    p = await _ask_presence(ctx, group)
    if p < presence.threshold_absent:
        return _GroupResult(group, present=False, presence=p)
    result = await _run_group(ctx, group, unit, None)
    result.presence = p
    result.borderline = p < presence.threshold_present
    return result


async def _run_group(
    ctx: Ctx, group: FieldGroup, unit: _Unit, fixed: Mapping[Path, Any] | None
) -> _GroupResult:
    """Ask for one group: its fields, its sub-models and its unions.

    The sub-models run concurrently with the group's own requests; the members
    of a union are asked once the union's choice is known.
    """
    fixed = fixed or {}
    clairic_specs = [
        spec
        for spec in group.specs
        if spec.engine == "clairic" and spec.path not in fixed
    ]
    clef_specs = [
        spec for spec in group.specs if spec.clef is not None and spec.path not in fixed
    ]
    union_specs = [union.spec for union in group.unions]

    async def own() -> tuple[list[_Leaf], list[_UnionResult]]:
        extracted, clef_answers = await both(
            _extract(ctx, clairic_specs, unit, group),
            _clef_pass(ctx, [*clef_specs, *union_specs], group),
        )
        leaves: list[_Leaf] = []
        pending: list[tuple[FieldSpec, Any]] = []
        for spec in group.specs:
            if spec.path in fixed:
                leaves.append(_Leaf(spec, filled=True, value=fixed[spec.path]))
            elif spec.engine == "const":
                leaves.append(_Leaf(spec, filled=True, value=_const_value(spec)))
            elif spec.engine == "clairic":
                pending.append((spec, len(leaves)))
                leaves.append(_Leaf(spec))
            elif spec.engine == "clef":
                resolved = resolve_clef(spec, clef_answers.get(spec.clef_name), ctx)
                leaves.append(_Leaf(spec, resolved=resolved))
        states = await gather(
            *(
                _resolve_clairic(ctx, spec, extracted[spec.wire_name], clef_answers)
                for spec, _ in pending
            )
        )
        for (_, index), state in zip(pending, states, strict=True):
            leaves[index].state = state
        unions = await gather(
            *(
                _run_union(ctx, union, clef_answers.get(union.spec.clef_name))
                for union in group.unions
            )
        )
        return leaves, unions

    outcomes: list[Any] = await gather(
        own(), *(_run_child(ctx, child, unit) for child in group.children)
    )
    leaves, unions = outcomes[0]
    return _GroupResult(
        group, leaves=leaves, children=list(outcomes[1:]), unions=unions
    )


def _const_value(spec: FieldSpec) -> Any:
    assert spec.key_map is not None
    return next(iter(spec.key_map.values()))


async def _run_unit(
    ctx: Ctx, group: FieldGroup, fixed: Mapping[Path, Any] | None = None
) -> _GroupResult:
    """Run a group and the sub-models below it as one unit of passes."""
    unit = _Unit()
    if ctx.run.passes.split == "entity":
        specs = [
            spec
            for member in group.walk()
            for spec in member.specs
            if spec.engine == "clairic" and spec.path not in (fixed or {})
        ]
        if specs:
            unit.shared = asyncio.ensure_future(_clairic_pass(ctx, specs, None))
    try:
        return await _run_group(ctx, group, unit, fixed)
    finally:
        if unit.shared is not None:
            unit.shared.cancel()
            await asyncio.gather(unit.shared, return_exceptions=True)


async def _run_union(
    ctx: Ctx, union: UnionGroup, answer: Answer | None
) -> _UnionResult:
    """Resolve a union's choice and ask for the chosen member (or all of them)."""
    resolved = resolve_clef(union.spec, answer, ctx)
    result = _UnionResult(union, answer, resolved)
    run = ctx.run
    where = union.spec.pretty
    if resolved.ok and resolved.value is not None:
        member = next((m for m in union.members if m.label == resolved.value), None)
        assert member is not None and isinstance(answer, ChoiceAnswer)
        result.chosen = member
        result.members[member.label] = await _run_unit(
            ctx, member.group, _tag_fill(union, member, answer.choice)
        )
        return result
    if resolved.ok:  # the choice says the input does not state the union
        return result
    if union.optional:
        run.warn(
            f"{ctx.plan.name}.{where}: the choice between "
            f"{', '.join(m.label for m in union.members)} was not confident "
            f"({resolved.reason}); the field is treated as unstated"
        )
        return result
    run.warn(
        f"{ctx.plan.name}.{where}: the choice between "
        f"{', '.join(m.label for m in union.members)} was not confident "
        f"({resolved.reason}); every member was extracted and the one with the "
        "most verified fields was kept"
    )
    result.fallback = True
    results = await gather(*(_run_unit(ctx, m.group) for m in union.members))
    result.members = {
        member.label: item for member, item in zip(union.members, results, strict=True)
    }
    return result


def _tag_fill(
    union: UnionGroup, member: UnionMember, option: str
) -> dict[Path, Any] | None:
    """The tag value to fill in for a member chosen through ``option``.

    A single-valued tag is a const and needs no help; a tag with several values
    is a Clef choice that is not asked, since the union's choice already named
    the value.
    """
    if union.discriminator is None or option not in union.option_tags:
        return None
    for spec in member.group.specs:
        if spec.path[-1] == union.discriminator and spec.engine != "const":
            return {spec.path: union.option_tags[option]}
    return None


# --------------------------------------------------------------------------- #
# After the ladder: which groups and members stand
# --------------------------------------------------------------------------- #


def _leaf_states(result: _GroupResult) -> Iterator[FieldState]:
    """Every Clairic state below a group, including every union member asked."""
    for leaf in result.leaves:
        if leaf.state is not None:
            yield leaf.state
    for child in result.children:
        yield from _leaf_states(child)
    for union in result.unions:
        for member in union.members.values():
            yield from _leaf_states(member)


def _count(result: _GroupResult) -> int:
    """How many non-null, verified fields stand below a group."""
    if not result.present:
        return 0
    total = sum(
        1 for leaf in result.leaves if not leaf.filled and leaf.final() is not None
    )
    total += sum(_count(child) for child in result.children)
    for union in result.unions:
        if union.chosen is not None:
            total += _count(union.members[union.chosen.label])
    return total


def _settle(result: _GroupResult) -> None:
    """Decide the borderline sub-models and the union fallbacks, bottom up."""
    for child in result.children:
        _settle(child)
        if child.present and child.borderline and _count(child) == 0:
            child.present = False
    for union in result.unions:
        for member in union.members.values():
            _settle(member)
        if union.fallback:
            best = max(
                union.union.members,
                key=lambda m: (
                    _count(union.members[m.label]),
                    -union.union.members.index(m),
                ),
            )
            if _count(union.members[best.label]) > 0:
                union.chosen = best


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
    for position, reason, detail in resolved.lost:
        rejected.append(RejectedOp(None, f"{pointer}/{position}", reason, detail, None))
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


def _filled_answer(spec: FieldSpec, value: Any) -> FieldAnswer:
    """The answer of a field whose value code supplied (a const or a union's tag)."""
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


@dataclass
class _Assembly:
    """What walking the results produces."""

    values: dict[Path, Any] = field(default_factory=dict)
    answers: dict[str, FieldAnswer] = field(default_factory=dict)
    rejected: list[RejectedOp] = field(default_factory=list)
    members: dict[Path, str] = field(default_factory=dict)
    groups: list[tuple[FieldAnswer, bool]] = field(default_factory=list)


def _group_answer(
    spec: FieldSpec,
    raw: Any,
    confidence: float | None,
    *,
    present: bool,
    reason: str | None,
) -> FieldAnswer:
    """The answer for a sub-model's presence or a union's choice."""
    return FieldAnswer(
        path=spec.path,
        pretty=spec.pretty,
        wire_name=spec.wire_name,
        engine="clef",
        value=None,
        raw=raw,
        confidence=confidence,
        source="clef" if present else "default",
        rejected_reason=reason,
    )


def _emit_leaf(leaf: _Leaf, out: _Assembly) -> None:
    spec = leaf.spec
    if leaf.state is not None:
        answer, rejects = _clairic_answer(leaf.state)
    elif leaf.resolved is not None:
        answer, rejects = _clef_answer(spec, leaf.resolved)
    else:
        answer, rejects = _filled_answer(spec, leaf.value), []
    out.values[spec.path] = answer.value
    out.answers[spec.pretty] = answer
    out.rejected.extend(rejects)


def _absent(group: FieldGroup | UnionGroup, out: _Assembly) -> None:
    """Record a sub-model or union that is not there.

    A nullable one gets ``None``; ``prune_nulls`` turns that into the default
    when the field has one. A non-nullable one is left out, so the default
    applies.
    """
    if group.nullable:
        out.values[group.path] = None


def _emit(result: _GroupResult, out: _Assembly) -> None:
    group = result.group
    if group.spec is not None and group.spec.clef is not None:
        p = result.presence
        out.groups.append(
            (
                _group_answer(
                    group.spec,
                    {"noul": p},
                    None,
                    present=result.present,
                    reason=None if result.present else "absent",
                ),
                result.present,
            )
        )
    if not result.present:
        _absent(group, out)
        return
    for leaf in result.leaves:
        _emit_leaf(leaf, out)
    for child in result.children:
        _emit(child, out)
    for union in result.unions:
        _emit_union(union, out)


def _emit_union(result: _UnionResult, out: _Assembly) -> None:
    union = result.union
    spec = union.spec
    raw = (
        result.choice.model_dump(exclude={"type"})
        if result.choice is not None
        else None
    )
    confidence = result.resolved.confidence
    chosen = result.chosen
    reason = None
    if not result.resolved.ok:
        reason = result.resolved.reason or "invalid"
        out.rejected.append(
            RejectedOp(
                None, json_pointer(spec.path), reason, result.resolved.detail, None
            )
        )
        if result.fallback and chosen is not None:
            reason = "low_confidence_fallback"
    out.groups.append(
        (
            _group_answer(
                spec, raw, confidence, present=chosen is not None, reason=reason
            ),
            chosen is not None,
        )
    )
    if chosen is None:
        _absent(union, out)
        return
    out.members[union.path] = chosen.label
    _emit(result.members[chosen.label], out)


def _lookup(model: BaseModel, path: Path) -> Any:
    """Follow a path of field names through a validated model."""
    value: Any = model
    for part in path:
        value = getattr(value, str(part))
    return value


def _assemble(
    ctx: Ctx,
    root: _GroupResult,
    qualify_p: float | None,
) -> ResultRecord:
    plan = ctx.plan
    out = _Assembly()
    _emit(root, out)
    answers = out.answers
    rejected = out.rejected
    document = prune_nulls(
        unflatten(out.values, plan=plan, members=out.members),
        plan,
        members=out.members,
    )
    entity = EntityInfo(key=ctx.key, label=ctx.label, qualify_p=qualify_p)
    model: BaseModel | None
    group_answers = {answer.pretty: answer for answer, _ in out.groups}
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
            fields={**answers, **group_answers},
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
    # Report the values as the model holds them (enums, sets, nested models, ...).
    final: dict[str, FieldAnswer] = {}
    for key, answer in answers.items():
        if answer.value is not None:
            with contextlib.suppress(AttributeError):
                answer = dataclasses.replace(answer, value=_lookup(model, answer.path))
        final[key] = answer
    for key, (answer, present) in ((a.pretty, (a, p)) for a, p in out.groups):
        if present:
            with contextlib.suppress(AttributeError):
                answer = dataclasses.replace(answer, value=_lookup(model, answer.path))
        final[key] = answer
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
    """Extract one entity of a schema from ``text``.

    Nested sub-models and unions of models are expanded into groups (see
    ``compile_schema``): an optional sub-model is gated by its own presence
    question, a union is resolved by one Clef choice and only the chosen
    member's fields are asked. Collections and mappings are left unset (with a
    warning); a required field that depends on one makes the record invalid.

    Args:
        run: The invoke's context.
        plan: The compiled schema.
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

    root = await _run_unit(ctx, plan.root)
    await run_ladder(ctx, list(_leaf_states(root)))
    _settle(root)
    return InsertOutcome(_assemble(ctx, root, qualify_p), extracted=True)
