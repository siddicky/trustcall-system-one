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
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from trustcall_system_one._compile import (
    CollectionPlan,
    FieldGroup,
    FieldSpec,
    SchemaPlan,
    UnionGroup,
    UnionMember,
)
from trustcall_system_one._diff import (
    MISSING,
    ApplyResult,
    Change,
    DiffResult,
    GroupNode,
    LeafNode,
    UnionNode,
    apply_ops,
    diff_group,
    document_path,
    is_change,
    jsonable,
    leaves_with_old,
    list_paths,
    lookup,
    new_positions,
    ops_for_errors,
    pointer,
    sort_changes,
)
from trustcall_system_one._errors import ExtractionValidationError
from trustcall_system_one._existing import ExistingRecord, normalise_record
from trustcall_system_one._gates import dedup_elements, gate_changes
from trustcall_system_one._match import candidate_options, identity_compatible
from trustcall_system_one._names import Path
from trustcall_system_one._nested import (
    COLLECTIONS,
    NestedInsert,
    NestedUpdate,
    doc_key,
    guard,
    guard_warning,
    nest_for,
    nested_update,
)
from trustcall_system_one._render import Focus, RenderedState
from trustcall_system_one._repair import (
    Hit,
    Outcome,
    Pending,
    Problem,
    RepairState,
    Target,
    locate,
    problems_of,
    repair_loop,
)
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
    Nest,
    Placement,
    Resolved,
    accept,
    adopt_cross,
    ask_clairic,
    contest_cross,
    crosscheck,
    resolve,
    resolve_clef,
    run_ladder,
    usable_cross,
)
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ClairicAnswer,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
)

if TYPE_CHECKING:
    from trustcall_system_one._collection import Scope

__all__ = [
    "InsertOutcome",
    "UpdateOutcome",
    "ask_presence",
    "insert_one",
    "update_one",
]

Span = tuple[int, int]
Extracted = dict[str, tuple[ClairicAnswer | None, RenderedState | None]]


@dataclass(frozen=True)
class InsertOutcome:
    """What ``insert_one`` produced.

    Attributes:
        record: The record, or ``None`` when the presence gate found no entity.
        extracted: Whether any extraction request was made.
        attempts: Extraction rounds: 0 when nothing was extracted, else 1 plus
            the repair rounds.
        skipped: The skipped entity (``validation_failed``) when the record is
            invalid.
        nested_skipped: The items of the record's nested lists that were found
            and not used (``not_qualified``, ``validation_failed``), each with
            its ``EntityInfo.parent``.
    """

    record: ResultRecord | None
    extracted: bool
    attempts: int = 0
    skipped: SkippedEntity | None = None
    nested_skipped: tuple[SkippedEntity, ...] = ()


def _note_plan(run: Run, plan: SchemaPlan) -> None:
    """Record the compile warnings of a plan (a recursive field is named once)."""
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


async def ask_presence(run: Run, plan: SchemaPlan, text: str, label: str) -> float:
    """Ask Clef how likely the input is to contain an entity of the schema."""
    return await _presence(Ctx(run=run, plan=plan, text=text, label=label))


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
    collections: list[NestedInsert] = field(default_factory=list)


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
        ctx.state(),
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
        ctx.state(),
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
        ctx.warn(
            f"{ctx.plan.name}.{where}: the choice between "
            f"{', '.join(m.label for m in union.members)} was not confident "
            f"({resolved.reason}); the field is treated as unstated"
        )
        return result
    ctx.warn(
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
        repaired=resolved.repaired and not state.reject,
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
        repaired=resolved.repaired and reason is None,
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
    problems: list[Problem] = field(default_factory=list)
    swallowed: list[Problem] = field(default_factory=list)

    def mark(self) -> tuple[int, frozenset[tuple[Any, ...]], int]:
        """Where the errors stand, to undo what a dropped fragment added."""
        return len(self.errors), frozenset(self.fatal), len(self.problems)

    def rollback(self, mark: tuple[int, frozenset[tuple[Any, ...]], int]) -> None:
        """Forget the errors a fragment raised once it is dropped.

        The errors are not lost for the repair loop: a fragment that is dropped
        can still be repaired, so what was wrong inside it moves to ``swallowed``.
        """
        del self.errors[mark[0] :]
        self.fatal = set(mark[1])
        self.swallowed.extend(self.problems[mark[2] :])
        del self.problems[mark[2] :]


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
    if answer.value is not None:
        doc[_key(spec)] = answer.value
    elif not spec.has_default:
        # a required list the input says nothing about is empty, not null (a null
        # fails validation: "Input should be a valid list"); a nullable one keeps None
        empty = _is_list_field(spec) and not spec.nullable
        doc[_key(spec)] = [] if empty else None


def _is_list_field(spec: FieldSpec) -> bool:
    """Whether a field is a Clairic ``list`` (a list, set, tuple or frozenset)."""
    return spec.clairic is not None and spec.clairic.type == "list"


def _validated(
    ctx: Ctx,
    model: type[BaseModel],
    document: dict[str, Any],
    path: Path,
    out: _Assembly,
    *,
    what: str,
    optional: bool,
    trail: tuple[str, ...],
) -> BaseModel | None:
    """Validate a sub-document on its own, so the instance is what the parent gets.

    A sub-model that is optional and invalid is dropped (it is absent, with a
    warning and a rejected op); a required one reports its errors under its own
    path and the record is invalid. Either way the errors are kept as ``Problem``
    objects, located by ``trail`` (the document keys down to the sub-document),
    for the repair loop.
    """
    try:
        return model.model_validate(document)
    except ValidationError as exc:
        label = ".".join(str(part) for part in path)
        errors = _error_strings(exc, label)
        problems = problems_of(exc, trail)
        if optional:
            out.swallowed.extend(problems)
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
            ctx.warn(
                f"{ctx.plan.name}.{label}: the {what} failed validation "
                f"({'; '.join(errors)}); it is treated as absent"
            )
        else:
            out.errors.extend(errors)
            out.problems.extend(problems)
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
                trail=(*trail, _key(child.group)),
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
    for nested in result.collections:
        _place_collection(nested, doc, out)
    placed = {nested.spec.path for nested in result.collections}
    for spec in group.specs:
        # a collection that was not extracted (a limit, a list of a union of
        # models) is empty, as one the input says nothing about is
        if spec.engine in COLLECTIONS and spec.path not in placed:
            _empty_collection(spec, doc)
    return doc


def _place_collection(
    nested: NestedInsert, doc: dict[str, Any], out: _Assembly
) -> None:
    """Put the items of a collection field into the parent's document.

    The items are in first-mention order (a mapping is keyed by the extracted
    key). A field with nothing in it takes its default; a required list or mapping
    that the input says nothing about is empty, not null (a null fails
    validation); a nullable one keeps ``None``.
    """
    spec = nested.spec
    out.answers.update(nested.answers)
    out.rejected.extend(nested.rejected)
    value = nested.document()
    if value:
        doc[doc_key(spec)] = value
    else:
        _empty_collection(spec, doc)


def _empty_collection(spec: FieldSpec, doc: dict[str, Any]) -> None:
    """A collection field with nothing in it: its default, else empty (or ``None``)."""
    if not spec.has_default:
        doc[doc_key(spec)] = None if spec.nullable else _empty_of(spec)


def _empty_of(spec: FieldSpec) -> list[Any] | dict[str, Any]:
    return {} if spec.engine == "mapping" else []


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
        trail=(*trail, _key(union)),
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


_LOOKUP_ERRORS = (AttributeError, IndexError, KeyError, TypeError)


def _lookup(model: BaseModel, path: Path) -> Any:
    """Follow a path of field names, list positions and mapping keys through a model."""
    value: Any = model
    for part in path:
        if isinstance(value, Mapping) or isinstance(part, int):
            value = value[part]
        else:
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


@dataclass
class _Built:
    """A candidate record: the document, whether it validates and what is wrong.

    Attributes:
        out: What walking the results produced (answers, rejections, notes).
        document: The nested document that was validated.
        model: The validated model, ``None`` when it did not validate.
        errors: The error strings of what failed.
        blocking: The errors that make the record invalid (the root's, and those
            of a required union member).
        swallowed: The errors of optional sub-models and union members that were
            dropped as absent: the record validates without them, but the repair
            loop can still fix what was wrong inside.
    """

    out: _Assembly
    document: dict[str, Any]
    model: BaseModel | None
    errors: list[str]
    blocking: list[Problem]
    swallowed: list[Problem]

    @property
    def failed(self) -> bool:
        """Whether the record is invalid."""
        return self.model is None or bool(self.out.fatal) or bool(self.errors)

    @property
    def problems(self) -> list[Problem]:
        """Everything the repair loop may try to fix."""
        return [*self.blocking, *self.swallowed]


def _build(ctx: Ctx, root: _GroupResult) -> _Built:
    """Build the document the answers make and validate it."""
    plan = ctx.plan
    out = _Assembly()
    document = _fragment(ctx, root, out, ()) or {}
    errors = list(out.errors)
    blocking = list(out.problems)
    model: BaseModel | None = None
    try:
        model = plan.schema.model_validate(document)
    except ValidationError as exc:
        errors.extend(_error_strings(exc, skip=out.fatal))
        blocking.extend(
            p
            for p in problems_of(exc)
            if not (p.type == "missing" and p.loc in out.fatal)
        )
    return _Built(out, document, model, errors, blocking, list(out.swallowed))


def _record(ctx: Ctx, built: _Built, qualify_p: float | None) -> ResultRecord:
    """The result record of a built candidate (valid or not)."""
    plan, out = ctx.plan, built.out
    entity = ctx.info or EntityInfo(key=ctx.key, label=ctx.label, qualify_p=qualify_p)
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
    model = built.model
    if built.failed or model is None:
        fields = {
            **answers,
            **{
                note.spec.pretty: _note_answer(note, None)
                for note in out.notes.values()
            },
        }
        return ResultRecord(
            schema_name=plan.name,
            json_doc_id=None,
            kind="insert",
            value=None,
            after=_plain(built.document),
            rejected=tuple(out.rejected),
            fields=fields,
            entity=entity,
            valid=False,
            errors=tuple(built.errors),
        )
    # Report the values as the model holds them (enums, sets, nested models, ...).
    final: dict[str, FieldAnswer] = {}
    for key, answer in answers.items():
        if answer.value is not None:
            with contextlib.suppress(*_LOOKUP_ERRORS):
                answer = dataclasses.replace(answer, value=_lookup(model, answer.path))
        final[key] = answer
    for note in out.notes.values():
        value = None
        if note.present:
            with contextlib.suppress(*_LOOKUP_ERRORS):
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


def _all_leaves(result: _GroupResult) -> Iterator[_Leaf]:
    """Every leaf below a group, every union member that was asked included."""
    yield from result.leaves
    for child in result.children:
        yield from _all_leaves(child)
    for union in result.unions:
        for member in union.members.values():
            yield from _all_leaves(member)


def _chosen_of(result: _GroupResult, base: tuple[str, ...]) -> dict[str, str]:
    """The document pointer of every union that stands, to the member chosen."""
    found: dict[str, str] = {}
    if not result.present:
        return found
    for union in result.unions:
        at = (*base, _key(union.union))
        if union.chosen is not None:
            found[pointer(at)] = union.chosen.label
            found.update(_chosen_of(union.members[union.chosen.label], at))
    for child in result.children:
        found.update(_chosen_of(child, (*base, _key(child.group))))
    return found


def _note_leaf(leaf: _Leaf, text: str) -> None:
    """Add a line to the history of a field's answer."""
    if leaf.state is not None:
        leaf.state.notes.append(text)
    elif leaf.resolved is not None:
        leaf.resolved = dataclasses.replace(
            leaf.resolved, notes=(*leaf.resolved.notes, text)
        )


def _install(leaf: _Leaf, outcome: Outcome) -> bool:
    """Put a repair outcome into a leaf. Returns whether a new value went in.

    An outcome that went in can be taken back and put in again through its
    ``undo`` and ``redo``.
    """
    saved = (leaf.spec, leaf.state, leaf.resolved)
    outcome.restores = leaf.final()
    if _put(leaf, outcome):
        new = (leaf.spec, leaf.state, leaf.resolved)

        def undo() -> None:
            leaf.spec, leaf.state, leaf.resolved = saved
            _note_leaf(leaf, f"reverted, it was not needed: {outcome.note}")

        def redo() -> None:
            leaf.spec, leaf.state, leaf.resolved = new

        outcome.undo, outcome.redo = undo, redo
        return True
    return False


def _put(leaf: _Leaf, outcome: Outcome) -> bool:
    if outcome.state is not None:
        earlier = leaf.state
        if earlier is not None and earlier.reject:
            outcome.state.notes.append(
                f"the first answer had been rejected ({earlier.reject}) before "
                "the repair"
            )
        leaf.spec, leaf.state, leaf.resolved = outcome.spec, outcome.state, None
        return True
    if outcome.resolved is not None:
        leaf.spec, leaf.state, leaf.resolved = outcome.spec, None, outcome.resolved
        return True
    _note_leaf(leaf, outcome.note)
    return False


def _drop_leaf(leaf: _Leaf, detail: str) -> None:
    """Give up on a field's value: it is null (or its default) from now on."""
    if leaf.state is not None:
        leaf.state.set_reject("validation", detail)
    elif leaf.resolved is not None:
        leaf.resolved = dataclasses.replace(
            leaf.resolved, ok=False, reason="validation", detail=detail
        )


def _problem_text(problems: list[Problem]) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in p.loc)}: {p.msg}" if p.loc else p.msg
        for p in problems
    )


async def _finish_insert(
    ctx: Ctx, root: _GroupResult, qualify_p: float | None
) -> tuple[ResultRecord, SkippedEntity | None, int]:
    """Assemble, validate and (if needed) repair the candidate of an insert.

    Returns:
        The record, the skipped entity when the record is invalid (only with
        the repair loop on), and the number of repair rounds that were used.

    Raises:
        ExtractionValidationError: In strict mode, if the record stays invalid.
    """
    run, plan = ctx.run, ctx.plan
    state = RepairState()
    built = _build(ctx, root)
    chosen = _chosen_of(root, ())
    leaves = {
        leaf.spec.wire_name: leaf
        for leaf in _all_leaves(root)
        if not leaf.filled and (leaf.state is not None or leaf.resolved is not None)
    }
    if run.repair.enabled and built.problems:

        def target_for(hit: Hit) -> Target | None:
            leaf = leaves.get(hit.entry.spec.wire_name)
            if leaf is None:
                return None
            kind = leaf.state.type_answer if leaf.state is not None else None
            return Target(hit, leaf.spec, leaf.final(), kind)

        async def apply(outcomes: list[Outcome]) -> None:
            for outcome in outcomes:
                leaf = leaves.get(outcome.target.wire)
                if leaf is not None:
                    _install(leaf, outcome)

        async def refresh() -> Pending:
            nonlocal built
            built = _build(ctx, root)
            return Pending(built.problems, chosen)

        await repair_loop(
            ctx,
            state,
            Pending(built.problems, chosen),
            refresh=refresh,
            target_for=target_for,
            apply=apply,
        )
        if built.failed:
            # an optional field that still fails goes back to its default (or null)
            for hit in locate(plan, built.blocking, chosen):
                leaf = leaves.get(hit.entry.spec.wire_name)
                if leaf is None or not hit.direct or leaf.spec.required:
                    continue
                _drop_leaf(leaf, _problem_text(hit.problems))
        # the answers now carry what the loop noted and dropped
        built = _build(ctx, root)
    record = _record(ctx, built, qualify_p)
    if record.valid:
        return record, None, state.rounds
    # with the loop off the record is just invalid, as it was before M6
    skipped = (
        SkippedEntity(
            plan.name,
            record.entity or EntityInfo(key=ctx.key, label=ctx.label),
            "validation_failed",
            "; ".join(record.errors),
        )
        if run.repair.enabled
        else None
    )
    if run.strict:
        raise ExtractionValidationError(
            f"{plan.name} failed validation: " + "; ".join(record.errors),
            errors=list(record.errors),
            record=record,
            calls=tuple(run.calls),
        )
    return record, skipped, state.rounds


def _sites(
    result: _GroupResult, base: tuple[str, ...]
) -> Iterator[tuple[_GroupResult, tuple[str, ...]]]:
    """Every group of an entity that stands, with the document keys down to it.

    The groups are the entity's own, its sub-models that were stated and the
    chosen member of each union: the places where a collection field can hold
    items.
    """
    if not result.present:
        return
    yield result, base
    for child in result.children:
        yield from _sites(child, (*base, _key(child.group)))
    for union in result.unions:
        if union.chosen is not None:
            yield from _sites(
                union.members[union.chosen.label], (*base, _key(union.union))
            )


def _lineage_name(ctx: Ctx) -> str:
    """How the entity of a context is named in the warnings of its items."""
    return f"{ctx.info.label} ({ctx.info.key})" if ctx.info is not None else ctx.label


def _collection_jobs(
    ctx: Ctx, root: _GroupResult
) -> list[tuple[_GroupResult, tuple[str, ...], FieldSpec, CollectionPlan]]:
    """The collection fields to extract: in groups that stand and within limits."""
    jobs: list[tuple[_GroupResult, tuple[str, ...], FieldSpec, CollectionPlan]] = []
    seen: set[tuple[Any, str]] = set()
    for result, base in _sites(root, ()):
        for spec in result.group.specs:
            if spec.engine not in COLLECTIONS:
                continue
            if spec.engine == "collection" and len(spec.models) != 1:
                ctx.run.warn(
                    f"{ctx.plan.name}.{spec.pretty}: unsupported_list_item: a list "
                    "of a union of models is not extracted"
                )
                continue
            plan = ctx.plan.collection(spec)
            why = guard(ctx, spec, plan)
            if why:
                key = (result.group.model, why + spec.pretty)
                if key not in seen:
                    seen.add(key)
                    ctx.run.warn(guard_warning(ctx, spec, plan, why))
                continue
            jobs.append((result, base, spec, plan))
    return jobs


def _collection_scope(
    ctx: Ctx,
    parent: EntityInfo,
    base: tuple[str, ...],
    spec: FieldSpec,
    plan: CollectionPlan,
    existing: tuple[Any, ...] = (),
) -> "Scope":
    """The collection scope of a collection field of the entity ``ctx`` is about."""
    # the collection engine calls this module: import it late
    from trustcall_system_one._collection import Scope

    return Scope(
        plan=plan.item,
        source=ctx.text,
        cardinality=plan.many,
        parent=parent,
        pointer=pointer((*base, doc_key(spec))),
        existing=existing,
        lineage=(*ctx.lineage, _lineage_name(ctx)),
        nest=nest_for(ctx, spec, plan),
    )


async def _insert_nested(ctx: Ctx, root: _GroupResult) -> list[NestedInsert]:
    """Extract the items of every collection field of an inserted entity.

    For each collection field of a group that stands, the collection engine runs
    scoped to this entity (discovery, coreference, qualification and counting
    ask about the items *of this parent*), and every item is inserted by the full
    single-entity pipeline. The results are attached to their groups, to be put
    into the parent's document on assembly.
    """
    from trustcall_system_one._collection import insert_collection

    jobs = _collection_jobs(ctx, root)
    if not jobs:
        return []
    parent = ctx.info or EntityInfo(key=ctx.key, label=ctx.label)

    async def one(
        job: tuple[_GroupResult, tuple[str, ...], FieldSpec, CollectionPlan],
    ) -> NestedInsert:
        _, base, spec, plan = job
        scope = _collection_scope(ctx, parent, base, spec, plan)
        done = await insert_collection(ctx.run, scope)
        return NestedInsert.build(
            spec,
            plan,
            base,
            done.records,
            done.skipped,
            extracted=done.extracted,
            attempts=done.attempts,
        )

    built = await gather(*(one(job) for job in jobs))
    for job, nested in zip(jobs, built, strict=True):
        job[0].collections.append(nested)
    return built


async def insert_one(
    run: Run,
    plan: SchemaPlan,
    text: str,
    *,
    label: str,
    skip_presence: bool = False,
    presence: float | None = None,
    entity: EntityInfo | None = None,
    focus: Focus | None = None,
    focus_plain: Focus | None = None,
    placement: Placement | None = None,
) -> InsertOutcome:
    """Extract one entity of a schema from ``text``.

    Nested sub-models and unions of models are expanded into groups (see
    ``compile_schema``): an optional sub-model is gated by its own presence
    question, a union is resolved by one Clef choice and only the chosen
    member's fields are asked. Each collection field (``list[Model]``,
    ``dict[str, X]``) of a group that stands goes through the collection engine,
    scoped to this entity (the items *of this parent*), once the entity's own
    fields are known; the items are put into the document in first-mention order.

    Args:
        run: The invoke's context.
        plan: The compiled schema.
        text: The user's input.
        label: How prompts name the entity (``"the Person"``).
        skip_presence: Skip the presence gate (the schema was forced).
        presence: The presence probability, when it has been asked already.
        entity: The entity of a collection this record is about (``label`` then
            is its name); its key and mentions go into the prompts and the
            record. The presence gate is not asked for it.
        focus: The focus header of the Clairic requests (a collection entity's).
        focus_plain: The same without offsets, for requests that carry only
            windows of the input.
        placement: For an item of a nested list, whom it belongs to.

    Returns:
        The outcome: no record when the gate finds no entity, else an insert
        record that is valid or not.

    Raises:
        ExtractionValidationError: In strict mode, if validation fails.
        StateTruncatedError: In strict mode, if Clairic keeps truncating.
        TransportError: If a request failed (it is in ``run.calls`` first).
    """
    placed = placement or Placement()
    if entity is None:
        ctx = Ctx(run=run, plan=plan, text=text, label=label)
    else:
        ctx = Ctx(
            run=run,
            plan=plan,
            text=text,
            label=label,
            key=entity.key,
            mentions=tuple(dict.fromkeys(m.text for m in entity.mentions)),
            info=entity,
            focus=focus,
            focus_plain=focus_plain,
        )
        skip_presence = True
    ctx = dataclasses.replace(
        ctx,
        parent=placed.parent,
        lineage=placed.lineage,
        nest=placed.nest
        if placed.nest.chain
        else Nest(placed.nest.depth, (plan.schema,)),
        scope=placed.scope,
    )
    _note_plan(run, plan)
    qualify_p: float | None = presence
    if entity is not None and qualify_p is None:
        qualify_p = entity.qualify_p
    if not skip_presence and qualify_p is None:
        qualify_p = await _presence(ctx)
    absent = qualify_p is not None and qualify_p < run.presence.threshold_absent
    if absent and entity is None:
        return InsertOutcome(None, extracted=False)

    root = await _run_unit(ctx, plan.root)
    await run_ladder(ctx, list(_leaf_states(root)))
    _settle(root)
    nested = await _insert_nested(ctx, root)
    record, skipped, rounds = await _finish_insert(ctx, root, qualify_p)
    return InsertOutcome(
        record,
        True,
        max([1 + rounds, *(n.attempts for n in nested)]),
        skipped,
        tuple(item for n in nested for item in n.skipped),
    )


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
        skipped: The skipped entity, when updates are disabled; for a collection
            entity whose match was refused (``identity_conflict``) it comes next
            to the record, which then holds no change.
        handled: ``(schema name, record id)`` of the existing record when it was
            dealt with (updated or skipped; the schema name is ``"__any__"`` for
            such a record); ``None`` when it stays untouched.
        extracted: Whether any extraction request was made.
        attempts: Extraction rounds: 0 when nothing was extracted, else 1 plus
            the repair rounds.
        nested_skipped: The items of the record's nested lists that were found
            and not used, each with its ``EntityInfo.parent``.
    """

    record: ResultRecord | None
    skipped: SkippedEntity | None
    handled: tuple[str, str] | None
    extracted: bool
    attempts: int = 0
    nested_skipped: tuple[SkippedEntity, ...] = ()


def _json_default(model: type[BaseModel], spec: FieldSpec) -> Any:
    """The field's default as a JSON value, or ``MISSING`` when it has none."""
    default = _default_of(model, spec)
    if default is PydanticUndefined:
        return MISSING
    try:
        return jsonable(default)
    except (TypeError, ValueError):
        return MISSING


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
                default=_json_default(group.model, spec),
                ref=leaf,
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
    rejection: RejectedOp | None = None  # how it is reported, if it was thrown away
    stage: int = 0  # 0: the near-duplicate check, 1: the change gate


async def _judge_changes(
    ctx: Ctx, before: dict[str, Any], changes: list[Change]
) -> list[_Verdict]:
    """Run the near-duplicate check and the change gate over the operations."""
    if ctx.vetoed:  # every operation is refused, nothing is asked
        return [
            _Verdict(
                change,
                "identity_conflict",
                None,
                RejectedOp(
                    change.op, change.path, "identity_conflict", ctx.vetoed, None
                ),
            )
            for change in changes
        ]
    verdicts = [_Verdict(change) for change in changes]
    dedups = await dedup_elements(ctx, before, changes)
    for index, found in dedups.items():
        if found.duplicate:
            verdict = verdicts[index]
            verdict.reason = "duplicate"
            verdict.p = found.p
            change = verdict.change
            verdict.rejection = RejectedOp(
                change.op,
                change.path,
                "duplicate",
                f"{change.value!r} is the same fact as {found.other!r}",
                found.p,
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
        verdict.stage = 1
        verdict.rejection = RejectedOp(
            change.op, change.path, verdict.reason, detail, answer.p
        )
    return verdicts


def _dotted(path: tuple[str | int, ...]) -> str:
    return ".".join(str(part) for part in path) or "(record)"


def _chosen_members(node: GroupNode, base: tuple[str, ...]) -> dict[str, str]:
    """The document pointer of every union the update decided, to its member."""
    found: dict[str, str] = {}
    for union in node.unions:
        below = (*base, union.key)
        if union.chosen is not None and union.member is not None:
            found[pointer(below)] = union.chosen.label
            found.update(_chosen_members(union.member, below))
    for child in node.children:
        if child.present:
            found.update(_chosen_members(child, (*base, child.key)))
    return found


def _problems(
    plan: SchemaPlan, failure: ValidationError, chosen: dict[str, str]
) -> list[tuple[tuple[str | int, ...], str]]:
    """The validation errors as ``(document path, message)``.

    The location of an error below a union of models is mapped to the document
    (``document_path``) and an error about a member that was not chosen is left
    out. If that leaves nothing, every error is kept.
    """
    mapped: list[tuple[tuple[str | int, ...], str]] = []
    everything: list[tuple[tuple[str | int, ...], str]] = []
    for error in failure.errors():
        loc = tuple(error["loc"])
        everything.append((loc, error["msg"]))
        path = document_path(plan, loc, chosen)
        if path is not None:
            mapped.append((path, error["msg"]))
    return mapped or everything


def _owner(changes: list[Change], op: PatchOp) -> Change | None:
    """The computed change an applied (possibly repaired) operation came from."""
    path = op.repaired_from or op.path
    return next((c for c in changes if c.path == path), None)


@dataclass
class _Eval:
    """One evaluation of an update: its operations, applied and validated.

    Attributes:
        diff: What the diff found.
        verdicts: The verdict on every computed operation, in diff order.
        ordered: The operations that passed, in the order they are applied.
        lists: Pointers of the list fields (for ``apply_ops``).
        applied: The result of applying them.
        reasons: Why an operation (by ``id``) did not make it, when apply said so.
        apply_rejected: The rejections of operations that could not be applied.
        chosen: The union members the update decided, by union pointer.
        model: The validated patched record (``None`` if it is invalid).
        failure: The validation error, if there is one.
    """

    diff: DiffResult
    verdicts: list[_Verdict]
    ordered: list[Change]
    lists: frozenset[str]
    applied: ApplyResult
    reasons: dict[int, str]
    apply_rejected: list[RejectedOp]
    chosen: dict[str, str]
    model: BaseModel | None
    failure: ValidationError | None


def _is_new(spec: FieldSpec, value: Any, old: Any, default: Any) -> bool:
    """Whether an answer would change the record (add to it or replace a value)."""
    if value is None:
        return False
    if isinstance(value, list):
        return bool(new_positions(spec, value, old))
    return is_change(spec, value, old, default)


def _walk(doc: Any, path: tuple[str | int, ...]) -> Any:
    """Follow keys and list indexes through a document (``MISSING`` if absent)."""
    value = doc
    for part in path:
        if isinstance(part, int) and isinstance(value, list):
            if not 0 <= part < len(value):
                return MISSING
            value = value[part]
        elif isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return MISSING
    return value


def _held_element(
    record: Mapping[str, Any],
    patched: Mapping[str, Any],
    ops: Sequence[PatchOp],
    path: tuple[str | int, ...],
) -> bool:
    """Whether a path points at a list element the record already held.

    An error about such an element is not the update's: it was there before, and
    it costs no repair request. The element is the record's only when the patched
    document holds exactly the record's element at that path and no applied
    operation wrote an ancestor of the path (a ``replace /pet`` hands the
    patched document a new pet whose ``toys/0`` merely sits where the old pet's
    ``toys/0`` was).

    Args:
        record: The record before the update.
        patched: The document the operations produced.
        ops: The operations that were applied.
        path: Document path of the error.
    """
    if not any(isinstance(part, int) for part in path):
        return False
    here = pointer(path)
    if any(op.path != here and here.startswith(op.path + "/") for op in ops):
        return False
    old = _walk(record, path)
    return old is not MISSING and _walk(patched, path) == old


def _change_key(change: Change) -> tuple[str, str, str]:
    """What identifies an operation across evaluations (its kind, place, value)."""
    return (
        change.op,
        change.path,
        json.dumps(change.value, sort_keys=True, default=str),
    )


async def _evaluate(
    ctx: Ctx,
    root: _GroupResult,
    before: dict[str, Any],
    cache: dict[tuple[str, str, str], _Verdict],
    nested: NestedUpdate,
) -> _Eval:
    """Diff the answers against the record, judge, apply and validate.

    ``cache`` holds the verdicts of the operations that were judged already, so a
    repair round only pays for the operations its new answers produced. The
    operations of the nested items join the record's own: a new item (``add``)
    is judged like any other addition, while an item's field changes were
    attributed and gated by the item's own update, and a removal was decided by
    the removal question, so they stand as they are.
    """
    run, plan = ctx.run, ctx.plan
    nodes = _group_node(root, "")
    diff = diff_group(nodes, before, allow_null_removals=run.gates.allow_null_removals)
    changes = [*diff.changes, *nested.changes]
    for settled in nested.fixed:
        cache.setdefault(_change_key(settled), _Verdict(settled))
    fresh = [c for c in changes if _change_key(c) not in cache]
    if fresh:
        for verdict in await _judge_changes(ctx, before, fresh):
            cache[_change_key(verdict.change)] = verdict
    verdicts = [cache[_change_key(c)] for c in changes]
    passed = [v.change for v in verdicts if v.reason is None]
    ordered = sort_changes(passed)
    lists = list_paths(nodes) | frozenset(nested.lists)
    applied = apply_ops(before, [change.as_op() for change in ordered], lists=lists)
    reasons: dict[int, str] = {}
    apply_rejected: list[RejectedOp] = []
    for op, error in applied.failed:
        change = next(
            (c for c in ordered if c.path == op.path and c.value == op.value), None
        )
        apply_rejected.append(RejectedOp(op.op, op.path, "apply_failed", error, None))
        if change is not None:
            reasons[id(change)] = "apply_failed"
    model, failure = _validate_document(plan, applied.doc)
    return _Eval(
        diff,
        verdicts,
        ordered,
        lists,
        applied,
        reasons,
        apply_rejected,
        _chosen_members(nodes, ()),
        model,
        failure,
    )


async def _repair_update(
    ctx: Ctx,
    root: _GroupResult,
    before: dict[str, Any],
    cache: dict[tuple[str, str, str], _Verdict],
    ev: _Eval,
    nested: NestedUpdate,
) -> tuple[_Eval, int, list[RejectedOp]]:
    """Re-ask the fields an update's validation errors implicate.

    A field is re-asked when an operation of this update touches it (the error
    is the update's own) or a required value is missing (that one is re-asked
    input-only, attributed and gated like the rest); any other error the record
    already had is left alone. The re-asked value goes through the
    ladder (``run_round``), then the diff, the near-duplicate check and the
    change gate again (only the operations it changed are judged anew).

    Returns:
        The last evaluation, the number of repair rounds used and the
        operations a repair refused to revert.
    """
    state = RepairState()
    refused: list[RejectedOp] = []
    leaves = {
        leaf.spec.wire_name: leaf
        for leaf in _all_leaves(root)
        if not leaf.filled and (leaf.state is not None or leaf.resolved is not None)
    }

    def _refuse_reversion(outcome: Outcome) -> None:
        """A repair that goes back to the record's value is not a repair.

        The field was asked again because the update's operation for it broke
        validation; an answer that is the record's own value would silently undo
        what the input said. The repair counts as failed, the field keeps the
        answer the operation came from, and the operation is reported rejected.
        """
        value = (
            outcome.state.resolved.value
            if outcome.state is not None
            else outcome.resolved.value
            if outcome.resolved is not None
            else None
        )
        if value is None:
            return
        entry = outcome.target.hit.entry
        spec, previous = outcome.target.spec, outcome.target.previous
        old = lookup(before, entry.path)
        default = _json_default(entry.owner, entry.spec)
        if not _is_new(spec, previous, old, default) or _is_new(
            outcome.spec, value, old, default
        ):
            return
        ops = ops_for_errors(ev.applied.applied, [entry.path])
        op = ops[0] if ops else None
        detail = (
            f"the repair gave {value!r}, the record's own value, for a field the "
            "update meant to change"
        )
        refused.append(
            RejectedOp(
                op.op if op else None,
                op.path if op else pointer(entry.path),
                "validation",
                detail,
            )
        )
        outcome.state = outcome.resolved = None
        outcome.note = f"{outcome.note.split(':')[0]} failed: {detail}"

    def pre_existing(problem: Problem) -> bool:
        path = document_path(ctx.plan, problem.loc, ev.chosen)
        return path is not None and _held_element(
            before, ev.applied.doc, ev.applied.applied, path
        )

    def pending() -> Pending:
        # errors of what the record held before the update are not the update's
        # to repair: they cost no request
        found = problems_of(ev.failure) if ev.failure is not None else []
        return Pending([p for p in found if not pre_existing(p)], ev.chosen)

    def target_for(hit: Hit) -> Target | None:
        leaf = leaves.get(hit.entry.spec.wire_name)
        if leaf is None:
            return None
        touched = bool(ops_for_errors(ev.applied.applied, [hit.entry.path]))
        if not touched and not any(p.type == "missing" for p in hit.problems):
            return None
        # the answer the operation came from: never the patched document, which
        # holds the record
        kind = leaf.state.type_answer if leaf.state is not None else None
        return Target(hit, leaf.spec, leaf.final(), kind, show_input=False)

    async def apply(outcomes: list[Outcome]) -> None:
        installed: list[_Leaf] = []
        for outcome in outcomes:
            leaf = leaves.get(outcome.target.wire)
            if leaf is None:
                continue
            _refuse_reversion(outcome)
            if _install(leaf, outcome):
                installed.append(leaf)
        clef = [leaf for leaf in installed if leaf.resolved is not None]
        if not clef:
            return
        # a Clef answer always resolves: ask whether the input states it
        await _check_stated(
            ctx, _group_node(root, ""), before, only={id(leaf) for leaf in clef}
        )
        for leaf in clef:
            resolved = leaf.resolved
            if resolved is not None and resolved.value is None:
                leaf.resolved = dataclasses.replace(
                    resolved,
                    repaired=False,
                    notes=(*resolved.notes, "the input does not state it"),
                )

    async def refresh() -> Pending:
        nonlocal ev
        ev = await _evaluate(ctx, root, before, cache, nested)
        return pending()

    await repair_loop(
        ctx,
        state,
        pending(),
        refresh=refresh,
        target_for=target_for,
        apply=apply,
    )
    return ev, state.rounds, refused


async def _finish_update(
    ctx: Ctx,
    root: _GroupResult,
    before: dict[str, Any],
    target: ExistingRecord,
    entity: EntityInfo,
    nested: NestedUpdate,
) -> tuple[ResultRecord, int]:
    """Diff, judge, apply and validate an update; repair what fails validation.

    Returns:
        The record and the number of repair rounds that were used.
    """
    run, plan = ctx.run, ctx.plan
    cache: dict[tuple[str, str, str], _Verdict] = {}
    ev = await _evaluate(ctx, root, before, cache, nested)
    rounds = 0
    refused: list[RejectedOp] = []
    if run.repair.enabled and ev.failure is not None:
        ev, rounds, refused = await _repair_update(ctx, root, before, cache, ev, nested)
    out = _Assembly()
    _answers(root, out, ())
    out.answers.update(nested.answers)
    out.rejected.extend(nested.rejected)
    judged = sorted(
        (v for v in ev.verdicts if v.rejection is not None), key=lambda v: v.stage
    )
    rejected = [
        *out.rejected,
        *ev.diff.rejected,
        *(v.rejection for v in judged if v.rejection is not None),
        *ev.apply_rejected,
        *refused,
    ]
    verdicts, ordered, lists, chosen = ev.verdicts, ev.ordered, ev.lists, ev.chosen
    applied, reasons = ev.applied, dict(ev.reasons)
    model, failure = ev.model, ev.failure
    if failure is not None:
        problems = _problems(plan, failure, chosen)
        # which error is about which operation
        bad: list[PatchOp] = []
        why: dict[int, str] = {}

        def blame(op: PatchOp, text: str) -> None:
            if id(op) not in why:
                bad.append(op)
                why[id(op)] = text

        for path, message in problems:
            if _held_element(before, applied.doc, applied.applied, path):
                continue  # the record had that element: not this update's error
            for op in ops_for_errors(applied.applied, [path]):
                blame(op, f"{_dotted(path)}: {message}")
        # a model-level error whose message names fields is about those fields
        for hit in locate(plan, problems_of(failure), chosen):
            if hit.direct or not hit.named:
                continue
            for problem in hit.problems:
                for op in ops_for_errors(applied.applied, [hit.entry.path]):
                    blame(op, problem.msg)
        for op in bad:
            rejected.append(RejectedOp(op.op, op.path, "validation", why[id(op)], None))
            owner = _owner(ordered, op)
            if owner is not None:
                reasons[id(owner)] = "validation"
        if bad:
            keep = [op for op in applied.applied if not any(op is b for b in bad)]
            applied = apply_ops(before, keep, lists=lists)
            model, failure = _validate_document(plan, applied.doc)
    problems = _problems(plan, failure, chosen) if failure is not None else []
    for path, message in problems:  # what is still wrong after the drops
        rejected.append(
            RejectedOp(None, pointer(path), "validation", f"{_dotted(path)}: {message}")
        )
    if failure is not None:
        # a model-level error that names nothing: every operation that was part
        # of the failing document is reported, not just the record as a whole
        general = [
            p
            for p in problems_of(failure)
            if not p.loc
            or (
                (hits := locate(plan, [p], chosen))
                and not any(h.direct for h in hits)
                and not any(h.named for h in hits)
            )
        ]
        if general:
            for op in applied.applied:
                rejected.append(
                    RejectedOp(op.op, op.path, "validation", general[0].msg, None)
                )
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
        # an item's answer is not looked up again: removals shift the positions
        if (
            model is not None
            and valid
            and answer.value is not None
            and key not in nested.answers
        ):
            with contextlib.suppress(*_LOOKUP_ERRORS):
                answer = dataclasses.replace(answer, value=_lookup(model, answer.path))
        final[key] = answer
    for note in out.notes.values():
        value = None
        if valid and model is not None and note.present:
            with contextlib.suppress(*_LOOKUP_ERRORS):
                value = _lookup(model, note.spec.path)
        final[note.spec.pretty] = _note_answer(note, value)
    errors = tuple(_error_strings(failure)) if failure is not None else ()
    if problems:  # the errors of members that were not chosen are noise
        errors = tuple(f"{_dotted(path)}: {message}" for path, message in problems)
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
    return record, rounds


def _has_cross_answer(ctx: Ctx, st: FieldState) -> bool:
    """Whether the input-only pass gave a non-null raw answer for the field."""
    pair = ctx.crossed.get(st.spec.wire_name)
    return pair is not None and pair[0] is not None and pair[0].value is not None


async def _check_stated(
    ctx: Ctx,
    nodes: GroupNode,
    before: dict[str, Any],
    only: Collection[int] | None = None,
) -> None:
    """Ask whether the input states each Clef answer that would change the record.

    A ``bool``, ``Literal`` or ``Enum`` answer always resolves, whether or not the
    input says anything about the field, so it cannot be told from the model's
    default leaning. For every such answer that would change or add to the
    record, one Clef ``noul`` per group (call kind ``stated``) asks "Does
    ``input`` state <the field>?"; below ``verify.clef_stated_min`` the answer is
    made a null (noted ``not_stated``), so it is reported as ``not_mentioned``.
    With ``only`` (the ``id`` of leaves) just those answers are checked.
    """
    groups: dict[tuple[Any, ...], list[_Leaf]] = {}
    for node, old in leaves_with_old(nodes, before):
        leaf = node.ref
        if not isinstance(leaf, _Leaf) or leaf.filled or leaf.resolved is None:
            continue
        if only is not None and id(leaf) not in only:
            continue
        resolved = leaf.resolved
        if not resolved.ok or resolved.value is None or leaf.spec.engine != "clef":
            continue
        if is_change(node.spec, resolved.value, old, node.default):
            key = (node.spec.path[:-1], node.spec.member_labels)
            groups.setdefault(key, []).append(leaf)
    if not groups:
        return
    minimum = ctx.run.verify.clef_stated_min

    async def one(leaves: list[_Leaf]) -> None:
        questions = {
            f"stated.{leaf.spec.wire_name}": NoulQuestion(
                type="noul",
                instructions=f"Does `input` state {ctx.desc(leaf.spec)}?",
                criteria=NoulCriteria(
                    true="The input states it, directly or by clear implication.",
                    false="The input does not state it.",
                ),
            )
            for leaf in leaves
        }
        answers = await ctx.run.call_clef(
            questions,
            ctx.entity_state(),
            meta=ctx.meta(
                "stated",
                field=".".join(str(p) for p in leaves[0].spec.path[:-1]) or None,
            ),
        )
        for leaf in leaves:
            answer = answers.get(f"stated.{leaf.spec.wire_name}")
            p = answer.noul if isinstance(answer, NoulAnswer) else None
            if p is None or p < minimum:
                resolved = leaf.resolved
                assert resolved is not None
                leaf.resolved = dataclasses.replace(
                    resolved, value=None, notes=(*resolved.notes, "not_stated")
                )

    await gather(*(one(leaves) for leaves in groups.values()))


def _existing_items(
    plan: CollectionPlan, record: Mapping[str, Any], spec: FieldSpec
) -> tuple[list[ExistingRecord], list[str], int]:
    """The items a record holds in a collection field, as existing records.

    A list element's id is its index (as text); a mapping entry's id is its key
    and its record the entry ``{"key", "value"}``. An element that is not an
    object cannot be matched or patched and is skipped (it keeps its index).

    Returns:
        The records, the keys of a mapping and the size of a list.
    """
    value = record.get(doc_key(spec))
    name = plan.item.name
    if plan.is_dict:
        if not isinstance(value, Mapping):
            return [], [], 0
        keys = [str(k) for k in value]
        return (
            [
                ExistingRecord(str(k), name, {"key": str(k), "value": v})
                for k, v in value.items()
            ],
            keys,
            0,
        )
    if not isinstance(value, list):
        return [], [], 0
    return (
        [
            ExistingRecord(str(i), name, dict(element))
            for i, element in enumerate(value)
            if isinstance(element, Mapping)
        ],
        [],
        len(value),
    )


async def _update_nested(
    ctx: Ctx, root: _GroupResult, before: dict[str, Any], parent: EntityInfo
) -> NestedUpdate:
    """The operations that the nested items of an updated record need.

    For each collection field of a group that stands, the collection engine runs
    scoped to this entity with the record's items as the existing ones: each
    discovered item is matched to one (or to none), an item matched to one is
    updated by the full update path (its operations are rebased into the parent's
    patch list), a new item is inserted, and with ``enable_item_removals`` the
    items the input says to remove are removed. Items the input does not mention
    are untouched, and nothing is computed from the old value ("two more" is read
    as what the text states, never added to the quantity of the record).
    """
    from trustcall_system_one._collection import apply_collection

    jobs = _collection_jobs(ctx, root)
    out = NestedUpdate()
    if not jobs:
        return out

    async def one(
        job: tuple[_GroupResult, tuple[str, ...], FieldSpec, CollectionPlan],
    ) -> NestedUpdate:
        _, base, spec, plan = job
        holder = lookup(before, base)
        items, keys, size = _existing_items(
            plan, holder if isinstance(holder, Mapping) else {}, spec
        )
        scope = _collection_scope(ctx, parent, base, spec, plan, tuple(items))
        done = await apply_collection(
            ctx.run,
            scope,
            enable_inserts=ctx.run.enable_inserts,
            enable_updates=True,
            enable_deletes=ctx.run.enable_item_removals,
        )
        return nested_update(
            spec,
            plan,
            base,
            done.records,
            done.skipped,
            existing=size,
            keys=keys,
            extracted=done.extracted,
            attempts=done.attempts,
        )

    for part in await gather(*(one(job) for job in jobs)):
        out.merge(part)
    return out


def _identity_conflict(
    plan: SchemaPlan,
    names: Sequence[str],
    by_wire: Mapping[str, FieldState],
    before: Mapping[str, Any],
    mentions: Sequence[str] = (),
) -> str:
    """Why the entity cannot be the record it was matched to, or ``""``.

    The identity the input gives the entity (its extracted, verified value) and the
    record's own must be forms of one name (``identity_compatible``). With several
    identity fields the match stands as soon as one of them fits; a field the input
    gave no value for says nothing. A schema without identity never conflicts.

    A mention of the entity that fits the record's identity and does NOT fit the
    extracted one settles it: coreference linked the names, so an entity that the
    input calls both "Sarah Johnson" and "Sarah Miller" is the record "Sarah
    Johnson", and the new name is left to the change gate. A short form that fits
    both ("Emma" next to "Emma Watson", for the record "Emma Thompson") is no
    evidence.
    """
    seen: list[tuple[str, str, str]] = []
    for name in names:
        spec = next((s for s in plan.specs if s.path == (name,)), None)
        state = by_wire.get(spec.wire_name) if spec is not None else None
        if spec is None or state is None or not state.has_value():
            continue
        new, old = state.resolved.value, before.get(spec.alias or name)
        if not isinstance(new, str) or not isinstance(old, str):
            continue
        # a mention that fits the record and does not fit what the input calls the
        # entity is the other name of the same entity ("Sarah Johnson" next to
        # "Sarah Miller"); a short form of both ("Emma" for Emma Watson and Emma
        # Thompson) says nothing
        if identity_compatible(new, old) or any(
            identity_compatible(mention, old) and not identity_compatible(mention, new)
            for mention in mentions
        ):
            return ""
        seen.append((name, new, old))
    if not seen:
        return ""
    return "identity conflict: " + "; ".join(
        f"the input calls the entity {new!r} ({name}) but the record says {old!r}"
        for name, new, old in seen
    )


async def update_one(
    run: Run,
    plan: SchemaPlan,
    text: str,
    target: ExistingRecord,
    *,
    label: str,
    enable_updates: bool = True,
    skip_presence: bool = False,
    presence: float | None = None,
    entity: EntityInfo | None = None,
    focus: Focus | None = None,
    focus_plain: Focus | None = None,
    identity: Sequence[str] = (),
    placement: Placement | None = None,
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
        presence: The presence probability, when it has been asked already (to
            decide which schema an ``"__any__"`` record belongs to).
        entity: The entity of a collection that was matched to this record
            (``label`` then is its name): its key and mentions go into the
            prompts and the record, so attribution and the change gate are about
            that entity only. The presence gate is not asked for it.
        focus: The focus header of the Clairic requests (a collection entity's).
        focus_plain: The same without offsets, for requests that carry only
            windows of the input.
        identity: The identity fields of a collection entity. When what the
            input says the entity is called and what the record is called share
            no name (``identity_compatible``), the match was wrong: the whole
            update is refused (every operation is ``RejectedOp(reason=
            "identity_conflict")``, nothing is gated or applied) and the outcome
            carries an ``ambiguous_match`` skip next to the no-change record.
        placement: For an item of a nested list, whom it belongs to.

    Returns:
        The outcome.

    Raises:
        ExtractionValidationError: In strict mode, if the patched record does
            not validate.
        StateTruncatedError: In strict mode, if Clairic keeps truncating.
        TransportError: If a request failed (it is in ``run.calls`` first).
    """
    before = normalise_record(plan, target.data)
    handle = (target.schema_name, target.record_id)
    block = ExistingBlock(
        target.record_id,
        plan.name,
        json.dumps(before, ensure_ascii=False),
        data=before,
    )
    ctx = Ctx(
        run=run,
        plan=plan,
        text=text,
        label=label,
        existing=block,
        crosscheck=run.verify.input_only_crosscheck,
    )
    if entity is not None:
        ctx = dataclasses.replace(
            ctx,
            key=entity.key,
            mentions=tuple(dict.fromkeys(m.text for m in entity.mentions)),
            info=entity,
            focus=focus,
            focus_plain=focus_plain,
        )
        skip_presence = True
    placed = placement or Placement()
    ctx = dataclasses.replace(
        ctx,
        parent=placed.parent,
        lineage=placed.lineage,
        nest=placed.nest
        if placed.nest.chain
        else Nest(placed.nest.depth, (plan.schema,)),
        scope=placed.scope,
    )
    _note_plan(run, plan)
    qualify_p: float | None = presence
    if entity is not None and qualify_p is None:
        qualify_p = entity.qualify_p
    if not skip_presence and qualify_p is None:
        qualify_p = await _presence(ctx)
    if (
        entity is None
        and qualify_p is not None
        and qualify_p < run.presence.threshold_absent
    ):
        return UpdateOutcome(None, None, None, False)
    info = entity or EntityInfo(key=ctx.key, label=ctx.label, qualify_p=qualify_p)
    if not enable_updates:
        skipped = SkippedEntity(
            plan.name,
            info,
            "updates_disabled",
            f"the input is about the existing record {target.record_id!r}, but "
            "enable_updates is off",
        )
        return UpdateOutcome(None, skipped, handle, False)

    root = await _run_unit(ctx, plan.root)
    plain = ctx.input_only()
    nodes = _group_node(root, "")
    states = list(_leaf_states(root))
    by_wire = {st.spec.wire_name: st for st in states}
    unchanged: set[str] = set()
    new_at: dict[str, list[int]] = {}
    adoptable: list[tuple[FieldState, Any, LeafNode]] = []
    for leaf, old in leaves_with_old(nodes, before):
        state = by_wire.get(leaf.spec.wire_name)
        if state is None:
            continue
        if state.has_value():
            value = state.resolved.value
            if isinstance(value, list):
                fresh = new_positions(leaf.spec, value, old)
                if fresh:
                    new_at[leaf.spec.wire_name] = fresh
                else:
                    unchanged.add(leaf.spec.wire_name)
            elif not is_change(leaf.spec, value, old, leaf.default):
                unchanged.add(leaf.spec.wire_name)
        if not state.has_value() or leaf.spec.wire_name in unchanged:
            adoptable.append((state, old, leaf))
    candidates = [
        st for st in states if st.has_value() and st.spec.wire_name not in unchanged
    ]

    async def verify_clairic() -> None:
        if ctx.crosscheck:
            # the input-only reading of a field is needed to check a change, and
            # to take over where the primary reading gave nothing new
            wanted = {
                id(st): st
                for st in [*candidates, *(item[0] for item in adoptable)]
                if _has_cross_answer(ctx, st)
            }
            await gather(*(_load_cross(plain, st) for st in wanted.values()))
            await crosscheck(plain, candidates, new_at)

            async def take(st: FieldState, leaf: LeafNode, *, contested: bool) -> None:
                adopted = (
                    await contest_cross(plain, st) if contested else adopt_cross(st)
                )
                if adopted:
                    unchanged.discard(leaf.spec.wire_name)

            takes = []
            for st, old, leaf in adoptable:
                found = usable_cross(st)
                if found is None:
                    continue
                if isinstance(found.value, list):
                    changes = bool(new_positions(leaf.spec, found.value, old))
                else:
                    changes = is_change(leaf.spec, found.value, old, leaf.default)
                if not changes:
                    continue
                # the primary pass read the record's own value AND pointed at the
                # input for it: a genuine reading, so the two are adjudicated; a
                # null, an echo or a span-less value is simply given way to
                contested = leaf.spec.wire_name in unchanged and bool(st.spans)
                takes.append(take(st, leaf, contested=contested))
            await gather(*takes)
        await run_ladder(
            plain, [st for st in states if st.spec.wire_name not in unchanged]
        )

    await gather(verify_clairic(), _check_stated(plain, nodes, before))
    _settle(root)
    conflict = (
        _identity_conflict(
            plan,
            identity,
            by_wire,
            before,
            [m.text for m in entity.mentions] if entity is not None else (),
        )
        if entity is not None
        else ""
    )
    nested = NestedUpdate()
    if conflict:
        plain = dataclasses.replace(plain, vetoed=conflict)
    else:
        nested = await _update_nested(plain, root, before, info)
    record, rounds = await _finish_update(plain, root, before, target, info, nested)
    refused: SkippedEntity | None = None
    if conflict:
        options = dict(candidate_options(info.match_probs))
        options.setdefault(
            target.record_id, info.match_probs.get(target.record_id, 0.0)
        )
        refused = SkippedEntity(
            plan.name,
            info,
            "ambiguous_match",
            f"{conflict}; the update of record {target.record_id!r} was refused",
            tuple(sorted(options.items(), key=lambda item: (-item[1], item[0]))),
        )
    return UpdateOutcome(
        record,
        refused,
        handle,
        True,
        max(1 + rounds, nested.attempts),
        tuple(nested.skipped),
    )
