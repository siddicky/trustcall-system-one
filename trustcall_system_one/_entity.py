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

``update_one`` is the update path for a single existing record: the same groups
are asked with the record shown after the input (and a second, input-only
Clairic pass per group), the answers go through the cross-check and the ladder,
``_diff`` turns what differs from the record into operations, each is judged by
the change gate (``_gates``), applied, and the patched document is validated.
"""

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from trustcall_system_one._compile import (
    FieldGroup,
    FieldSpec,
    SchemaPlan,
    UnionGroup,
    UnionMember,
)
from trustcall_system_one._diff import (
    Change,
    GroupNode,
    LeafNode,
    UnionNode,
    apply_ops,
    diff_group,
    is_change,
    leaves_with_old,
    new_positions,
    ops_for_errors,
    sort_changes,
)
from trustcall_system_one._errors import ExtractionValidationError
from trustcall_system_one._existing import ExistingRecord, normalise_record
from trustcall_system_one._gates import dedup_elements, gate_changes
from trustcall_system_one._names import Path
from trustcall_system_one._render import RenderedState
from trustcall_system_one._run import Answer, Run, gather
from trustcall_system_one._types import (
    EntityInfo,
    FieldAnswer,
    PatchOp,
    RejectedOp,
    ResultRecord,
    SkippedEntity,
    json_pointer,
)
from trustcall_system_one._verify import (
    Ctx,
    ExistingBlock,
    FieldState,
    Resolved,
    accept,
    ask_clairic,
    crosscheck,
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

__all__ = ["InsertOutcome", "UpdateOutcome", "insert_one", "update_one"]

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


def _note_unsupported(run: Run, plan: SchemaPlan, *, updating: bool = False) -> None:
    """Warn about fields this milestone leaves alone.

    An insert leaves them to their pydantic default, an update leaves them as the
    record has them. A field a recursive model declares at every level is named
    once.
    """
    outcome = (
        "left as the record has it"
        if updating
        else "left unset so the model default applies"
    )
    seen: set[tuple[type[BaseModel], str]] = set()
    for group in plan.all_groups():
        for spec in group.specs:
            key = (group.model, str(spec.path[-1]))
            if spec.engine in _MILESTONE and key not in seen:
                seen.add(key)
                run.warn(
                    f"{plan.name}.{spec.pretty}: {spec.engine} fields are not "
                    f"extracted until {_MILESTONE[spec.engine]}; {outcome}"
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
    """One Clairic request for ``specs`` (chunked if the input is too long).

    For an update with the cross-check on, the same fields are asked a second
    time, concurrently, without the existing record (call kind ``crosscheck``);
    those answers are kept in ``ctx.crossed``.
    """
    fields = {spec.wire_name: ctx.clairic_field(spec) for spec in specs}
    single = specs[0].wire_name if len(specs) == 1 else field_
    descriptions = {spec.wire_name: spec.scoped for spec in specs}
    primary = ask_clairic(
        ctx,
        fields,
        meta=ctx.meta("extract", field=single),
        descriptions=descriptions,
    )
    if not ctx.crosscheck:
        return await primary
    plain = ctx.input_only()
    outcomes: list[Any] = await gather(
        primary,
        ask_clairic(
            plain,
            fields,
            meta=plain.meta("crosscheck", field=single),
            descriptions=descriptions,
        ),
    )
    ctx.crossed.update(outcomes[1])
    extracted: Extracted = outcomes[0]
    return extracted


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
    echoed = ctx.echoes.get(spec.wire_name)
    notes: list[str] = []
    if echoed:  # chunks whose answers were read from the existing record
        if resolved.ok and resolved.value is None:
            resolved = dataclasses.replace(
                resolved, ok=False, reason="echo_span", detail="; ".join(echoed)
            )
        else:
            notes = [f"echo_span: {note}" for note in echoed]
    state = FieldState(spec, resolved, type_answer, notes=notes)
    rejection = accept(spec, resolved)
    if rejection is not None:
        state.set_reject(*rejection)
    return state


async def _ask_presences(
    ctx: Ctx, groups: list[FieldGroup], parent: FieldGroup
) -> dict[Path, float]:
    """Ask Clef, in one request, whether the input states each optional sub-model.

    The siblings share a state, so their questions share a request; ``meta.field``
    lists their paths. An answer that is missing lands in the middle of the
    borderline band: the group is asked, and judged by what it comes back with.
    """
    questions = {}
    for group in groups:
        assert group.spec is not None and group.spec.clef is not None
        questions[group.spec.clef_name] = ctx.clef_question(group.spec)
    answers = await ctx.run.call_clef(
        questions,
        ctx.run.clef_state(ctx.text),
        meta=ctx.meta("presence", field=",".join(g.pretty for g in groups)),
    )
    presence = ctx.run.presence
    middle = (presence.threshold_absent + presence.threshold_present) / 2
    found: dict[Path, float] = {}
    for group in groups:
        assert group.spec is not None
        answer = answers.get(group.spec.clef_name)
        found[group.path] = answer.noul if isinstance(answer, NoulAnswer) else middle
    return found


async def _run_child(
    ctx: Ctx,
    group: FieldGroup,
    unit: _Unit,
    presences: "asyncio.Future[dict[Path, float]] | None",
) -> _GroupResult:
    """Run a sub-model group, after its presence question when it is optional."""
    if not group.optional:
        return await _run_group(ctx, group, unit, None)
    assert presences is not None
    presence = ctx.run.presence
    p = (await presences)[group.path]
    if p < presence.threshold_absent:
        return _GroupResult(group, present=False, presence=p)
    result = await _run_group(ctx, group, unit, None)
    result.presence = p
    result.borderline = p < presence.threshold_present
    return result


async def _cancel(*futures: "asyncio.Future[Any] | None") -> None:
    """Cancel whatever is still running and wait for it to stop."""
    live = [f for f in futures if f is not None]
    for future in live:
        future.cancel()
    await asyncio.gather(*live, return_exceptions=True)


async def _run_group(
    ctx: Ctx, group: FieldGroup, unit: _Unit, fixed: Mapping[Path, Any] | None
) -> _GroupResult:
    """Ask for one group: its fields, its sub-models and its unions.

    The sub-models run concurrently with the group's own requests, and a union's
    chosen member is asked as soon as the group's Clef answers (the choice among
    them) are in, without waiting for the Clairic answers.
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
    union_specs = [union.spec for union in group.unions if union.spec.clef is not None]
    optional = [child for child in group.children if child.optional]
    ensure = asyncio.ensure_future
    extract_f = ensure(_extract(ctx, clairic_specs, unit, group))
    clef_f = ensure(_clef_pass(ctx, [*clef_specs, *union_specs], group))
    presence_f = ensure(_ask_presences(ctx, optional, group)) if optional else None

    async def leaves_flow() -> list[_Leaf]:
        # awaited one after the other and never cancelled here: the union flow
        # waits on the same futures, and the outer cleanup stops them
        extracted = await extract_f
        clef_answers = await clef_f
        leaves: list[_Leaf] = []
        pending: list[tuple[FieldSpec, int]] = []
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
        return leaves

    async def unions_flow() -> list[_UnionResult]:
        clef_answers = await clef_f
        return await gather(
            *(
                _run_union(ctx, union, clef_answers.get(union.spec.clef_name))
                for union in group.unions
            )
        )

    try:
        outcomes: list[Any] = await gather(
            leaves_flow(),
            unions_flow(),
            *(_run_child(ctx, child, unit, presence_f) for child in group.children),
        )
    finally:
        await _cancel(extract_f, clef_f, presence_f)
    return _GroupResult(
        group, leaves=outcomes[0], children=list(outcomes[2:]), unions=outcomes[1]
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
    run = ctx.run
    where = union.spec.pretty
    if union.spec.clef is None:  # a single option: nothing to ask
        only = union.members[0]
        result = _UnionResult(union, None, Resolved(value=only.label))
        result.chosen = only
        result.members[only.label] = await _run_unit(
            ctx, only.group, _tag_fill(union, only, only.options[0])
        )
        return result
    resolved = resolve_clef(union.spec, answer, ctx)
    result = _UnionResult(union, answer, resolved)
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


def _default_of(model: type[BaseModel], spec: FieldSpec) -> Any:
    """The field's default, or ``PydanticUndefined`` when it has none."""
    info = model.model_fields.get(str(spec.path[-1]))
    if info is None or info.is_required():
        return PydanticUndefined
    return info.get_default(call_default_factory=True)


def _evidence(leaf: _Leaf, model: type[BaseModel]) -> bool:
    """Whether a leaf says the input is about this group.

    A verified, non-null Clairic value does. A Clef answer does only when it is
    not the answer the field would have anyway: a field with a default whose
    answer differs from the default, or a nullable field that was answered
    rather than left ``unstated``. A required Clef field (a ``bool`` or
    ``Literal`` that always resolves) never does, and neither does a constant.
    """
    if leaf.state is not None:
        return leaf.state.value is not None
    resolved = leaf.resolved
    if resolved is None or not resolved.ok or resolved.value is None or leaf.filled:
        return False
    spec = leaf.spec
    if "unstated" in resolved.notes:
        return False
    if spec.has_default:
        default = _default_of(model, spec)
        if default is not PydanticUndefined and resolved.value != default:
            return True
    return spec.nullable


def _count(result: _GroupResult) -> int:
    """How many pieces of evidence stand below a group (see ``_evidence``)."""
    if not result.present:
        return 0
    total = sum(_evidence(leaf, result.group.model) for leaf in result.leaves)
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


def _clairic_answer(
    state: FieldState, where: str | None = None
) -> tuple[FieldAnswer, list[RejectedOp]]:
    spec, resolved = state.spec, state.resolved
    pointer = where if where is not None else json_pointer(spec.path)
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
        crosscheck_agree=state.crosscheck_agree,
        reask=state.reask,
        rejected_reason=state.reject,
        warnings=_unique([*resolved.notes, *state.notes]),
    )
    return answer, rejected


def _clef_answer(
    spec: FieldSpec, resolved: Resolved, where: str | None = None
) -> tuple[FieldAnswer, list[RejectedOp]]:
    rejected: list[RejectedOp] = []
    reason = None if resolved.ok else (resolved.reason or "invalid")
    if reason is not None:
        rejected.append(
            RejectedOp(
                None,
                where if where is not None else json_pointer(spec.path),
                reason,
                resolved.detail,
                None,
            )
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


def _error_strings(
    exc: ValidationError, prefix: str = "", skip: set[tuple[Any, ...]] | None = None
) -> list[str]:
    """Format validation errors as ``"path: message"``.

    Args:
        exc: The validation error.
        prefix: Dotted path of the document the error was raised for.
        skip: Locations of ``missing`` errors to leave out (fields that were
            reported already by the validation of their own sub-document).
    """
    errors = []
    for error in exc.errors():
        if skip and error["type"] == "missing" and tuple(error["loc"]) in skip:
            continue
        where = ".".join(
            [*([prefix] if prefix else []), *(str(part) for part in error["loc"])]
        )
        errors.append(f"{where}: {error['msg']}" if where else error["msg"])
    return errors


@dataclass
class _Note:
    """The answer for a sub-model's presence or a union's choice."""

    spec: FieldSpec
    raw: Any
    confidence: float | None
    present: bool
    reason: str | None


@dataclass
class _Assembly:
    """What walking the results produces."""

    answers: dict[str, FieldAnswer] = field(default_factory=dict)
    rejected: list[RejectedOp] = field(default_factory=list)
    notes: dict[str, _Note] = field(default_factory=dict)
    dropped: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fatal: set[tuple[Any, ...]] = field(default_factory=set)

    def mark(self) -> tuple[int, frozenset[tuple[Any, ...]]]:
        """Where the errors stand, to undo what a dropped fragment added."""
        return len(self.errors), frozenset(self.fatal)

    def rollback(self, mark: tuple[int, frozenset[tuple[Any, ...]]]) -> None:
        """Forget the errors a fragment raised once it is dropped."""
        del self.errors[mark[0] :]
        self.fatal = set(mark[1])


def _key(spec: FieldSpec | FieldGroup | UnionGroup) -> str:
    """The key ``model_validate`` expects for a field."""
    path = spec.path
    return spec.alias if spec.alias is not None else str(path[-1])


def _absent(doc: dict[str, Any], owner: FieldGroup | UnionGroup) -> None:
    """Record a sub-model or union that is not there.

    A nullable one gets ``None`` unless it has a default (which then applies, as
    for any other null); a non-nullable one is left out, so its default applies.
    """
    if owner.nullable and not owner.has_default:
        doc[_key(owner)] = None


def _emit_leaf(leaf: _Leaf, doc: dict[str, Any], out: _Assembly) -> None:
    spec = leaf.spec
    if leaf.state is not None:
        answer, rejects = _clairic_answer(leaf.state)
    elif leaf.resolved is not None:
        answer, rejects = _clef_answer(spec, leaf.resolved)
    else:
        answer, rejects = _filled_answer(spec, leaf.value), []
    out.answers[spec.pretty] = answer
    out.rejected.extend(rejects)
    if answer.value is not None or not spec.has_default:
        doc[_key(spec)] = answer.value


def _validated(
    ctx: Ctx,
    model: type[BaseModel],
    document: dict[str, Any],
    path: Path,
    out: _Assembly,
    *,
    what: str,
    optional: bool,
) -> BaseModel | None:
    """Validate a sub-document on its own, so the instance is what the parent gets.

    A sub-model that is optional and invalid is dropped (it is absent, with a
    warning and a rejected op); a required one reports its errors under its own
    path and the record is invalid.
    """
    try:
        return model.model_validate(document)
    except ValidationError as exc:
        label = ".".join(str(part) for part in path)
        errors = _error_strings(exc, label)
        if optional:
            out.dropped.append(path)
            out.rejected.append(
                RejectedOp(
                    None,
                    json_pointer(path),
                    "invalid_optional_group",
                    "; ".join(errors),
                    None,
                )
            )
            ctx.run.warn(
                f"{ctx.plan.name}.{label}: the {what} failed validation "
                f"({'; '.join(errors)}); it is treated as absent"
            )
        else:
            out.errors.extend(errors)
        return None


def _fragment(
    ctx: Ctx, result: _GroupResult, out: _Assembly, trail: tuple[str, ...]
) -> dict[str, Any] | None:
    """Build the document of one group, keyed by validation keys.

    Returns ``None`` for a sub-model that is absent. Optional sub-models and the
    chosen member of a union become validated instances, so validation of the
    parent cannot pick another model for them; required sub-models stay
    documents and are validated with the root. Fields whose value is ``None``
    and that have a default are left out, so the default applies.
    """
    group = result.group
    if group.spec is not None and group.spec.clef is not None:
        out.notes[group.pretty] = _Note(
            group.spec,
            {"noul": result.presence},
            None,
            result.present,
            None if result.present else "absent",
        )
    if not result.present:
        return None
    doc: dict[str, Any] = {}
    for leaf in result.leaves:
        _emit_leaf(leaf, doc, out)
    for child in result.children:
        mark = out.mark()
        sub = _fragment(ctx, child, out, (*trail, _key(child.group)))
        if sub is None:
            _absent(doc, child.group)
        elif child.group.optional:
            instance = _validated(
                ctx,
                child.group.model,
                sub,
                child.group.path,
                out,
                what="optional sub-model",
                optional=True,
            )
            if instance is None:
                out.rollback(mark)  # what was wrong inside it goes with it
                _absent(doc, child.group)
                _unsay(out, child.group.pretty)
            else:
                doc[_key(child.group)] = sub if child.group.typed_dict else instance
        else:
            doc[_key(child.group)] = sub
    for union in result.unions:
        _place_union(ctx, union, doc, out, trail)
    return doc


def _unsay(out: _Assembly, pretty_path: str) -> None:
    """Mark a dropped sub-model's notes, and those of everything below it.

    The presence note of the dropped group and the presence and union-choice
    notes of the groups and unions inside it (found by path prefix) say that
    nothing of them stands: ``present`` is false and the reason is
    ``invalid_optional_group``.
    """
    for key, note in out.notes.items():
        if key == pretty_path or key.startswith(f"{pretty_path}."):
            note.present = False
            note.reason = "invalid_optional_group"


def _union_note(result: _UnionResult, out: _Assembly, where: str) -> UnionMember | None:
    """Record a union's choice (and why it was rejected, if it was).

    Returns:
        The member that was chosen, if any.
    """
    spec = result.union.spec
    chosen = result.chosen
    reason = None
    if not result.resolved.ok:
        reason = result.resolved.reason or "invalid"
        out.rejected.append(
            RejectedOp(None, where, reason, result.resolved.detail, None)
        )
        if result.fallback and chosen is not None:
            reason = "low_confidence_fallback"
    if spec.clef is not None:
        raw = (
            result.choice.model_dump(exclude={"type"})
            if result.choice is not None
            else None
        )
        out.notes[spec.pretty] = _Note(
            spec, raw, result.resolved.confidence, chosen is not None, reason
        )
    return chosen


def _place_union(
    ctx: Ctx,
    result: _UnionResult,
    doc: dict[str, Any],
    out: _Assembly,
    trail: tuple[str, ...],
) -> None:
    union = result.union
    spec = union.spec
    chosen = _union_note(result, out, json_pointer(spec.path))
    if chosen is None:
        _absent(doc, union)
        return
    mark = out.mark()
    member = _fragment(ctx, result.members[chosen.label], out, (*trail, _key(union)))
    assert member is not None
    instance = _validated(
        ctx,
        chosen.model,
        member,
        union.path,
        out,
        what=f"{chosen.label} member",
        optional=union.optional,
    )
    if instance is not None:
        # a TypedDict member is a plain dict to its parent, not an instance
        doc[_key(union)] = (
            member if result.members[chosen.label].group.typed_dict else instance
        )
        return
    if union.optional:
        out.rollback(mark)
        _absent(doc, union)
        _unsay(out, spec.pretty)
    else:
        out.fatal.add((*trail, _key(union)))


def _lookup(model: BaseModel, path: Path) -> Any:
    """Follow a path of field names through a validated model."""
    value: Any = model
    for part in path:
        value = getattr(value, str(part))
    return value


def _plain(value: Any) -> Any:
    """Turn the model instances in a document back into dicts, keyed by alias."""
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _note_answer(note: _Note, value: Any) -> FieldAnswer:
    """The answer of a sub-model's presence or a union's choice.

    ``engine`` is the container's own (``submodel`` or ``union_models``), so a
    consumer that walks leaf answers can skip it; the question's answer is in
    ``raw``.
    """
    spec = note.spec
    return FieldAnswer(
        path=spec.path,
        pretty=spec.pretty,
        wire_name=spec.wire_name,
        engine=spec.engine,
        value=value,
        raw=note.raw,
        confidence=note.confidence,
        source="clef" if note.present else "default",
        rejected_reason=note.reason,
    )


def _assemble(ctx: Ctx, root: _GroupResult, qualify_p: float | None) -> ResultRecord:
    plan = ctx.plan
    out = _Assembly()
    document = _fragment(ctx, root, out, ()) or {}
    entity = EntityInfo(key=ctx.key, label=ctx.label, qualify_p=qualify_p)
    # what was dropped for failing validation is reported as dropped
    answers: dict[str, FieldAnswer] = {}
    for key, answer in out.answers.items():
        if any(answer.path[: len(path)] == path for path in out.dropped):
            answer = dataclasses.replace(
                answer,
                value=None,
                source="default",
                rejected_reason=answer.rejected_reason or "invalid_optional_group",
            )
        answers[key] = answer
    errors = list(out.errors)
    model: BaseModel | None = None
    try:
        model = plan.schema.model_validate(document)
    except ValidationError as exc:
        errors.extend(_error_strings(exc, skip=out.fatal))
    if model is None or out.fatal or errors:
        fields = {
            **answers,
            **{
                note.spec.pretty: _note_answer(note, None)
                for note in out.notes.values()
            },
        }
        record = ResultRecord(
            schema_name=plan.name,
            json_doc_id=None,
            kind="insert",
            value=None,
            after=_plain(document),
            rejected=tuple(out.rejected),
            fields=fields,
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
            )
        return record
    # Report the values as the model holds them (enums, sets, nested models, ...).
    final: dict[str, FieldAnswer] = {}
    for key, answer in answers.items():
        if answer.value is not None:
            with contextlib.suppress(AttributeError):
                answer = dataclasses.replace(answer, value=_lookup(model, answer.path))
        final[key] = answer
    for note in out.notes.values():
        value = None
        if note.present:
            with contextlib.suppress(AttributeError):
                value = _lookup(model, note.spec.path)
        final[note.spec.pretty] = _note_answer(note, value)
    return ResultRecord(
        schema_name=plan.name,
        json_doc_id=None,
        kind="insert",
        value=model,
        after=model.model_dump(),
        rejected=tuple(out.rejected),
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


# --------------------------------------------------------------------------- #
# Updates: one existing record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UpdateOutcome:
    """What ``update_one`` produced.

    Attributes:
        record: The update record (also when it made no change), or ``None``
            when the presence gate found nothing about the record in the input
            or updates are disabled.
        skipped: The skipped entity, when updates are disabled.
        handled_id: The existing record's id when the record was dealt with
            (updated or skipped); ``None`` when it stays untouched.
        extracted: Whether any extraction request was made.
    """

    record: ResultRecord | None
    skipped: SkippedEntity | None
    handled_id: str | None
    extracted: bool


def _group_node(result: _GroupResult, key: str) -> GroupNode:
    """The tree the diff walks, from the groups as they were asked and answered."""
    group = result.group
    node = GroupNode(key, group, present=result.present)
    if not result.present:
        return node
    for leaf in result.leaves:
        spec = leaf.spec
        state = leaf.state
        handled = (
            state is not None and (state.reject is not None or not state.resolved.ok)
        ) or (leaf.resolved is not None and not leaf.resolved.ok)
        node.leaves.append(
            LeafNode(
                key=_key(spec),
                spec=spec,
                value=leaf.final(),
                handled=handled,
                identity=spec.identity or str(spec.path[-1]) in group.identity,
                spans=tuple(state.spans) if state is not None else (),
            )
        )
    for child in result.children:
        node.children.append(_group_node(child, _key(child.group)))
    for union in result.unions:
        union_key = _key(union.union)
        member = (
            _group_node(union.members[union.chosen.label], union_key)
            if union.chosen is not None
            else None
        )
        node.unions.append(
            UnionNode(
                key=union_key,
                union=union.union,
                chosen=union.chosen,
                member=member,
                handled=not union.resolved.ok,
            )
        )
    return node


def _answers(result: _GroupResult, out: _Assembly, base: tuple[str, ...]) -> None:
    """Describe every asked field of an update (no document is built).

    The pointers of the rejections are document pointers (aliases), like the
    operations'.
    """
    group = result.group
    if group.spec is not None and group.spec.clef is not None:
        out.notes[group.pretty] = _Note(
            group.spec,
            {"noul": result.presence},
            None,
            result.present,
            None if result.present else "absent",
        )
    if not result.present:
        return
    for leaf in result.leaves:
        spec = leaf.spec
        where = json_pointer((*base, _key(spec)))
        if leaf.state is not None:
            answer, rejects = _clairic_answer(leaf.state, where)
        elif leaf.resolved is not None:
            answer, rejects = _clef_answer(spec, leaf.resolved, where)
        else:
            answer, rejects = _filled_answer(spec, leaf.value), []
        out.answers[spec.pretty] = answer
        out.rejected.extend(rejects)
    for child in result.children:
        _answers(child, out, (*base, _key(child.group)))
    for union in result.unions:
        below = (*base, _key(union.union))
        chosen = _union_note(union, out, json_pointer(below))
        if chosen is not None:
            _answers(union.members[chosen.label], out, below)


async def _load_cross(ctx: Ctx, st: FieldState) -> None:
    """Resolve the input-only answer of a field (``ctx`` is the input-only one)."""
    pair = ctx.crossed.get(st.spec.wire_name)
    if pair is None:
        return
    answer, anchored = pair
    st.cross = await resolve(ctx, st.spec, answer, anchored, st.type_answer)


def _validate_document(
    plan: SchemaPlan, document: Mapping[str, Any]
) -> tuple[BaseModel | None, ValidationError | None]:
    try:
        return plan.schema.model_validate(document), None
    except ValidationError as exc:
        return None, exc


@dataclass
class _Verdict:
    """What became of one computed operation."""

    change: Change
    reason: str | None = None  # why it was not applied
    p: float | None = None  # the change gate's probability, if it was asked


async def _judge_changes(
    ctx: Ctx, before: dict[str, Any], changes: list[Change], out: list[RejectedOp]
) -> list[_Verdict]:
    """Run the near-duplicate check and the change gate over the operations."""
    verdicts = [_Verdict(change) for change in changes]
    dedups = await dedup_elements(ctx, before, changes)
    for index, found in dedups.items():
        if found.duplicate:
            verdict = verdicts[index]
            verdict.reason = "duplicate"
            verdict.p = found.p
            change = verdict.change
            out.append(
                RejectedOp(
                    change.op,
                    change.path,
                    "duplicate",
                    f"{change.value!r} is the same fact as {found.other!r}",
                    found.p,
                )
            )
    live = [v for v in verdicts if v.reason is None]
    gated = await gate_changes(ctx, before, [v.change for v in live])
    for verdict, answer in zip(live, gated, strict=True):
        verdict.p = answer.p
        if answer.passed:
            continue
        change = verdict.change
        if answer.p is None:
            verdict.reason = "gate_missing"
            detail = "the change gate gave no answer"
        else:
            verdict.reason = "gate_noul"
            detail = (
                f"the input does not support the change (p={answer.p:.2f} < "
                f"{answer.needed})"
            )
        out.append(RejectedOp(change.op, change.path, verdict.reason, detail, answer.p))
    return verdicts


def _owner(changes: list[Change], op: PatchOp) -> Change | None:
    """The computed change an applied (possibly repaired) operation came from."""
    path = op.repaired_from or op.path
    return next((c for c in changes if c.path == path), None)


async def _finish_update(
    ctx: Ctx,
    root: _GroupResult,
    before: dict[str, Any],
    target: ExistingRecord,
    entity: EntityInfo,
) -> ResultRecord:
    run, plan = ctx.run, ctx.plan
    out = _Assembly()
    _answers(root, out, ())
    nodes = _group_node(root, "")
    diff = diff_group(nodes, before, allow_null_removals=run.gates.allow_null_removals)
    rejected = [*out.rejected, *diff.rejected]
    verdicts = await _judge_changes(ctx, before, diff.changes, rejected)
    passed = [v.change for v in verdicts if v.reason is None]
    ordered = sort_changes(passed)
    applied = apply_ops(before, [change.as_op() for change in ordered])
    reasons: dict[int, str] = {}
    for op, error in applied.failed:
        change = next(
            (c for c in ordered if c.path == op.path and c.value == op.value), None
        )
        rejected.append(RejectedOp(op.op, op.path, "apply_failed", error, None))
        if change is not None:
            reasons[id(change)] = "apply_failed"
    model, failure = _validate_document(plan, applied.doc)
    if failure is not None:
        bad = ops_for_errors(applied.applied, [e["loc"] for e in failure.errors()])
        if bad:
            for op in bad:
                detail = next(
                    (
                        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                        for e in failure.errors()
                        if e["loc"]
                    ),
                    "",
                )
                rejected.append(RejectedOp(op.op, op.path, "validation", detail, None))
                owner = _owner(ordered, op)
                if owner is not None:
                    reasons[id(owner)] = "validation"
            keep = [op for op in applied.applied if not any(op is b for b in bad)]
            applied = apply_ops(before, keep)
            model, failure = _validate_document(plan, applied.doc)
    # the fields: how each answer fared
    answers = dict(out.answers)
    for verdict in verdicts:
        change = verdict.change
        reason = verdict.reason or reasons.get(id(change))
        for name in change.fields:
            answer = answers.get(name)
            if answer is None:
                continue
            answers[name] = dataclasses.replace(
                answer,
                gate_p=verdict.p,
                value=None if reason else answer.value,
                rejected_reason=reason or answer.rejected_reason,
            )
    valid = failure is None and model is not None
    final: dict[str, FieldAnswer] = {}
    for key, answer in answers.items():
        if valid and model is not None and answer.value is not None:
            with contextlib.suppress(AttributeError):
                answer = dataclasses.replace(answer, value=_lookup(model, answer.path))
        final[key] = answer
    for note in out.notes.values():
        value = None
        if valid and model is not None and note.present:
            with contextlib.suppress(AttributeError):
                value = _lookup(model, note.spec.path)
        final[note.spec.pretty] = _note_answer(note, value)
    errors = tuple(_error_strings(failure)) if failure is not None else ()
    record = ResultRecord(
        schema_name=plan.name,
        json_doc_id=target.record_id,
        kind="update",
        value=model if valid else None,
        before=before,
        after=applied.doc,
        patches=tuple(applied.applied),
        rejected=tuple(rejected),
        fields=final,
        entity=entity,
        valid=valid,
        errors=errors,
    )
    if not valid and run.strict:
        raise ExtractionValidationError(
            f"{plan.name} {target.record_id!r} failed validation after the update: "
            + "; ".join(errors),
            errors=list(errors),
            record=record,
            calls=tuple(run.calls),
        )
    return record


async def update_one(
    run: Run,
    plan: SchemaPlan,
    text: str,
    target: ExistingRecord,
    *,
    label: str,
    enable_updates: bool = True,
    skip_presence: bool = False,
) -> UpdateOutcome:
    """Update one existing record of a schema from ``text``.

    The steps:

    1. the presence gate (skipped when the schema is forced): a record the input
       does not talk about is left untouched, with no further request;
    2. with ``enable_updates`` off the record is skipped (``updates_disabled``);
    3. every Clairic group is asked with the record shown after the input, and
       asked again without it (the cross-check, call kind ``crosscheck``); an
       answer whose span lies in the record is rejected (``echo_span``);
    4. the answers that would change the record are cross-checked, and the
       verification ladder runs on the input alone (a re-ask never sees the
       record);
    5. ``_diff`` computes the operations, each is judged by the change gate and
       applied, and the patched document is validated.

    A run in which no operation is applied is still an ``update`` record with an
    empty ``patches`` and ``after == before``.

    Args:
        run: The invoke's context.
        plan: The compiled schema.
        text: The user's input.
        target: The record to update (any keys; they are normalised to the
            schema's validation keys).
        label: How prompts name the entity (``"the Contact"``).
        enable_updates: Whether updating is allowed.
        skip_presence: Skip the presence gate (the schema was forced).

    Returns:
        The outcome.

    Raises:
        ExtractionValidationError: In strict mode, if the patched record does
            not validate.
        StateTruncatedError: In strict mode, if Clairic keeps truncating.
        TransportError: If a request failed (it is in ``run.calls`` first).
    """
    before = normalise_record(plan, target.data)
    block = ExistingBlock(
        target.record_id, plan.name, json.dumps(before, ensure_ascii=False)
    )
    ctx = Ctx(
        run=run,
        plan=plan,
        text=text,
        label=label,
        existing=block,
        crosscheck=run.verify.input_only_crosscheck,
    )
    _note_unsupported(run, plan, updating=True)
    qualify_p: float | None = None
    if not skip_presence:
        qualify_p = await _presence(ctx)
        if qualify_p < run.presence.threshold_absent:
            return UpdateOutcome(None, None, None, False)
    entity = EntityInfo(key=ctx.key, label=ctx.label, qualify_p=qualify_p)
    if not enable_updates:
        skipped = SkippedEntity(
            plan.name,
            entity,
            "updates_disabled",
            f"the input is about the existing record {target.record_id!r}, but "
            "enable_updates is off",
        )
        return UpdateOutcome(None, skipped, target.record_id, False)

    root = await _run_unit(ctx, plan.root)
    plain = ctx.input_only()
    states = list(_leaf_states(root))
    by_wire = {st.spec.wire_name: st for st in states}
    unchanged: set[str] = set()
    new_at: dict[str, list[int]] = {}
    for leaf, old in leaves_with_old(_group_node(root, ""), before):
        state = by_wire.get(leaf.spec.wire_name)
        if state is None or not state.has_value():
            continue
        value = state.resolved.value
        if isinstance(value, list):
            fresh = new_positions(leaf.spec, value, old)
            if fresh:
                new_at[leaf.spec.wire_name] = fresh
            else:
                unchanged.add(leaf.spec.wire_name)
        elif not is_change(leaf.spec, value, old):
            unchanged.add(leaf.spec.wire_name)
    candidates = [
        st for st in states if st.has_value() and st.spec.wire_name not in unchanged
    ]
    if ctx.crosscheck and candidates:
        await gather(*(_load_cross(plain, st) for st in candidates))
        await crosscheck(plain, candidates, new_at)
    await run_ladder(plain, [st for st in states if st.spec.wire_name not in unchanged])
    _settle(root)
    record = await _finish_update(plain, root, before, target, entity)
    return UpdateOutcome(record, None, target.record_id, True)
