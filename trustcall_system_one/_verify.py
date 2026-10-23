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
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from trustcall_system_one._chunk import (
    Merged,
    agree,
    chunk_size,
    identity_state,
    merge_field,
    part_of,
    settled,
    split_ranges,
)
from trustcall_system_one._coerce import Coerced, coerce, coerce_union
from trustcall_system_one._compile import (
    FieldGroup,
    FieldSpec,
    Role,
    SchemaPlan,
    UnionGroup,
    render_instructions,
)
from trustcall_system_one._coref import has_context
from trustcall_system_one._errors import StateTruncatedError
from trustcall_system_one._escalate import tiebreak
from trustcall_system_one._render import (
    Focus,
    RenderedState,
    mention_windows,
    narrow_text,
    narrowed_spans,
    render_clairic,
)
from trustcall_system_one._run import Answer, Question, Run, gather
from trustcall_system_one._sentences import split_sentences
from trustcall_system_one._types import CallKind, EntityInfo
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
    "LINEAGE_SEP",
    "Asked",
    "Ctx",
    "ExistingBlock",
    "FieldState",
    "Nest",
    "Placement",
    "Resolved",
    "accept",
    "adopt_cross",
    "agree",
    "align_list_spans",
    "ask_clairic",
    "compact_block",
    "contest_cross",
    "crosscheck",
    "display",
    "reask",
    "resolve",
    "resolve_clef",
    "run_ladder",
    "usable_cross",
]

Span = tuple[int, int]

LINEAGE_SEP = " \u203a "
"""Joins the names of an item and its ancestors in a warning."""
REASK_SUFFIX = " Answer only from the text near these mentions."
_SHORT_CODES = frozenset({"abstain", "unstated", "type_uncertain", "missing_answer"})
_NONE_OF_THESE = "none_of_these"
_ARTICLE = re.compile(r"^(?:the|an|a)\s+", re.IGNORECASE)
_NOT_STATED = "The input states no value for this."


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExistingBlock:
    """The existing record as it is shown to Clairic (updates).

    Attributes:
        record_id: The record's id (the ``id`` attribute of the block).
        schema_name: The schema name (the ``schema`` attribute).
        text: The record as JSON.
        data: The record itself (to compact the block, see ``compact_block``).
        compacted: Whether ``text`` already is a compacted rendering.
    """

    record_id: str
    schema_name: str
    text: str
    data: Mapping[str, Any] | None = None
    compacted: bool = False


@dataclass(frozen=True)
class Nest:
    """Where a plan sits inside the document it is part of.

    A collection's item plan is a plan of its own, so how deep it is nested and
    which item models are on the way down is not in the plan; the engine carries
    it to keep ``limits.max_depth`` and ``limits.max_recursion`` honest for
    collections of collections.

    Attributes:
        depth: How many path segments lie between the document root and this
            plan's root.
        chain: The item models from the root down to this plan, the root model
            first.
    """

    depth: int = 0
    chain: tuple[type[BaseModel], ...] = ()


@dataclass(frozen=True)
class Placement:
    """Where a nested item sits: its parent, ancestors, nesting and call scope.

    Attributes:
        parent: The entity the item belongs to.
        lineage: The names of its ancestors including the parent, outermost first.
        nest: How deep the item's plan is nested (see ``Nest``).
        scope: ``CallMeta.scope`` of its calls.
        others: Where the input mentions the entities of the same collection (the
            item's siblings and the item itself): a sentence that names another
            of them ends the reach of an entity's sentences.
    """

    parent: EntityInfo | None = None
    lineage: tuple[str, ...] = ()
    nest: Nest = field(default_factory=Nest)
    scope: str = ""
    others: tuple[Span, ...] = ()


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
        attribution_hints: Attribution probabilities already measured for a
            field's value (by wire name, with the value shown as text), so the
            ladder does not ask the same question again.
        existing: The record shown after the input in every Clairic request of
            this context (updates); ``None`` for an insert and for the
            input-only passes (see ``input_only``).
        crosscheck: Whether every Clairic pass is repeated without the record
            (call kind ``crosscheck``); the answers land in ``crossed``.
        crossed: The cross-check answers by wire name, with the state to map
            their spans with.
        echoes: Notes by wire name about chunked answers that were thrown away
            because their span lay in the existing record.
        trailer: Blocks written after the input in every Clairic request of this
            context (the repair loop's description of a rejected answer).
        info: The entity of a collection (key, label, mentions with their spans):
            its Clef states carry the mentions with spans, so two entities that
            share a label stay apart, and its record reports it.
        focus: The focus header of every Clairic request of this context, when it
            is not the default one (a collection entity's names the schema, the
            entity and its mentions with their offsets).
        focus_plain: The same header without offsets, used when the state holds
            only windows of the input (the offsets would not match it).
        vetoed: Why every operation of an update is refused without being judged
            (an identity conflict between the entity and the record it was matched
            to); empty when nothing is.
        parent: For an item of a nested list, the entity it belongs to: the Clef
            states name it (``entity.parent``) so the items of two parents stay
            apart.
        lineage: The names of the ancestors of a nested item, outermost first
            (``"Emma Thompson (e0)"``); run warnings start with them.
        nest: Where the plan sits in the document (see ``Nest``).
        others: Mention spans of the entities of the same collection (see
            ``Placement.others``).
        scope: ``CallMeta.scope`` of this context's calls: the parent key and the
            pointer of the collection (``"e0/pets"``), empty at the top level.
        identity_hint: For a single record being updated (a ``One``), what the
            record is called (its identity values, else its id): the Clef states
            name it next to the label ("the Bill INV-2291"), so a figure that
            belongs to another invoice in the same input is not attributed to
            this one.
        identity: For an update of a collection entity, the names of its identity
            fields (a dotted path for a nested one), as ``Many(identity=...)`` or the
            schema gave them; the diff protects them from a token-subset overwrite.
        sentence_fields: Names of the fields that were read by sentences in this
            context (shared by its input-only copy), for the novelty check.
        fresh: Whether the entity is NEW: an insert, with no record held for it. The
            verification ladder of an update runs on the input-only copy of a context
            that holds a record, so there it is False. An amount the input asks to
            ADD to a new entity ("two more of the blue mugs") has nothing to add to,
            so it is the entity's whole amount (see ``_attribution_question``).
    """

    run: Run
    plan: SchemaPlan
    text: str
    label: str
    key: str = "e0"
    mentions: tuple[str, ...] = ()
    attribution_hints: dict[str, tuple[str, float]] = field(default_factory=dict)
    existing: ExistingBlock | None = None
    crosscheck: bool = False
    crossed: dict[str, tuple[ClairicAnswer | None, RenderedState | None]] = field(
        default_factory=dict
    )
    echoes: dict[str, list[str]] = field(default_factory=dict)
    trailer: str = ""
    info: EntityInfo | None = None
    focus: Focus | None = None
    focus_plain: Focus | None = None
    vetoed: str = ""
    parent: EntityInfo | None = None
    lineage: tuple[str, ...] = ()
    nest: Nest = field(default_factory=Nest)
    scope: str = ""
    others: tuple[Span, ...] = ()
    identity_hint: str = ""
    identity: tuple[str, ...] = ()
    sentence_fields: set[str] = field(default_factory=set)
    fresh: bool = False

    def input_only(self) -> "Ctx":
        """The same context without the record: every request is about the input.

        The verification ladder of an update runs on this, so that a re-ask can
        never read the value back from the record it is checking. The
        attribution hints are shared with the original.
        """
        return dataclasses.replace(self, existing=None, crosscheck=False)

    def meta(self, kind: CallKind, *, field: str | None = None) -> CallMeta:
        """Routing context for a call. The entity is its key, not its label."""
        return CallMeta(
            kind=kind,
            schema=self.plan.name,
            entity=self.key,
            field=field,
            scope=self.scope,
        )

    def desc(self, spec: FieldSpec) -> str:
        """The field's description, put through ``verify.instruction_template``.

        A nested field is named through its owners, so ``author.name`` of a Book
        reads "the Book's author's name" and not "the Book's name". A leading
        article of the description is dropped so the template does not double it.
        """
        leaf = _ARTICLE.sub("", spec.description, count=1)
        chain = "'s ".join([*spec.owners, leaf])
        return self.run.verify.instruction_template.format(label=self.label, desc=chain)

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
        # a list field takes no null_threshold (the API refuses it)
        if null_threshold is not None and spec.clairic.type != "list":
            update["null_threshold"] = null_threshold
        return spec.clairic.model_copy(update=update)

    def clef_question(self, spec: FieldSpec) -> Question:
        """Return a copy of the spec's Clef question about this entity.

        Rebuilt from the field's parts (so a question the compiler clipped is not
        a problem) with the templated description. The presence question of an
        optional sub-model, the choice of a union of models and the questions of
        Clef-typed fields all come through here.
        """
        assert spec.clef is not None
        role: Role = (
            "union_type"
            if spec.clairic is not None
            else "presence"
            if spec.engine == "submodel"
            else spec.clef.type
        )
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
        existing = self.existing
        # offsets in the focus header are offsets in the whole input: with only
        # parts of it in the state they would point at the wrong text
        partial = bool(windows) and list(windows or ()) != [(0, len(self.text))]
        return render_clairic(
            self.text,
            label=self.label,
            mentions=self.mentions,
            existing=existing.text if existing is not None else None,
            existing_id=existing.record_id if existing is not None else None,
            existing_schema=existing.schema_name if existing is not None else None,
            windows=windows or None,
            trailer=self.trailer,
            focus=self.focus_plain if partial and self.focus_plain else self.focus,
        )

    def warn(self, message: str) -> None:
        """Record a run warning; one about a collection entity names the entity.

        An item of a nested list is named through its ancestors:
        ``Emma Thompson (e0) \u203a Rex (e1): ...``.
        """
        names = list(self.lineage)
        if self.info is not None:
            names.append(f"{self.info.label} ({self.info.key})")
        if names:
            message = f"{LINEAGE_SEP.join(names)}: {message}"
        self.run.warn(message)

    def mention_spans(self) -> list[Span]:
        """Where the input mentions a collection entity (empty for a ``One``)."""
        if self.info is None:
            return []
        return [m.span for m in self.info.mentions if m.span is not None]

    def parent_spans(self) -> list[Span]:
        """Where the input mentions the parent of a nested item."""
        if self.parent is None:
            return []
        return [m.span for m in self.parent.mentions if m.span is not None]

    def _mention_entries(self, info: EntityInfo, *, spans: bool) -> list[Any]:
        """The ``mentions`` of an entity in a Clef state: text, span, context."""
        context = self.run.coref.context_chars
        entries: list[Any] = []
        total = len(info.mentions)
        for index, m in enumerate(info.mentions):
            entry: dict[str, Any] = {"text": m.text}
            if m.span is not None:
                if spans:
                    entry["span"] = list(m.span)
                if has_context(index, total):  # the first 8 and the last 2
                    start, end = m.span
                    entry["context"] = self.text[
                        max(0, start - context) : end + context
                    ]
            entries.append(entry)
        return entries

    def parent_dict(self, *, spans: bool = True) -> dict[str, Any] | None:
        """The parent of a nested item for a Clef state: ``{"label", "mentions"}``."""
        if self.parent is None:
            return None
        return {
            "label": self.parent.label,
            "mentions": self._mention_entries(self.parent, spans=spans),
        }

    def entity_dict(self, *, spans: bool = True) -> dict[str, Any]:
        """The ``entity`` of a Clef state: ``{"label", "mentions"}``.

        The mentions are the texts the input uses; for a collection entity each
        is an object with its text, its span in the input and a short context
        around it (as coreference states have), so two entities that share a label
        (two people called Emma) can still be told apart. An item of a nested list
        also carries its ``parent`` (the same, for the entity it belongs to).

        Args:
            spans: Include the spans. They are offsets in the whole input, so a
                state whose input was narrowed leaves them out (the context
                still places the mention).
        """
        mentions: list[Any] = list(self.mentions)
        if self.info is not None and self.info.mentions:
            mentions = self._mention_entries(self.info, spans=spans)
        label = self.label
        if self.identity_hint and self.identity_hint.casefold() not in label.casefold():
            label = f"{label} {self.identity_hint}"
        entity: dict[str, Any] = {"label": label, "mentions": mentions}
        parent = self.parent_dict(spans=spans)
        if parent is not None:
            entity["parent"] = parent
        return entity

    def fitted(
        self,
        text: str | None = None,
        spans: Sequence[Span] | None = None,
        *,
        entity: bool = True,
        record: Any = None,
        proposed_change: Any = None,
        windows: Sequence[Span] | None = None,
        budget: int | None = None,
    ) -> dict[str, Any]:
        """A Clef state about this entity, kept under the Clef budget.

        When the input is over budget it is narrowed to windows around ``spans``
        and around the entity's own mentions, so the question can still see who
        it is about. The entity's spans are offsets in the whole input and are
        left out of a state whose input was narrowed (or was given already
        narrowed as ``text``).

        Args:
            text: The input as it goes into the state; the whole input by default.
                Nothing anchors a text given like this.
            spans: Further spans worth keeping when the state is narrowed.
            windows: Windows of the whole input to show (``narrow_text``); the
                spans and the entity's mentions are re-based into the narrowed
                text, and if that is still over budget it is cut around them
                rather than at its start.
            entity: Whether to include ``entity`` at all.
            record: An existing record (updates).
            proposed_change: A change under review (updates).
            budget: A smaller token budget for the state (see ``Run.clef_state``).
        """
        whole = [*(spans or ()), *self.mention_spans(), *self.parent_spans()]
        if windows:
            shown = narrow_text(self.text, windows)
            anchors = narrowed_spans(len(self.text), windows, whole)
        else:
            shown = self.text if text is None else text
            # spans are offsets in the whole input: there is nothing to anchor on
            # in a text that was narrowed already
            anchors = whole if shown == self.text else []
        state = self.run.clef_state(
            shown,
            entity=self.entity_dict() if entity else None,
            record=record,
            proposed_change=proposed_change,
            spans=list(dict.fromkeys(anchors)) or None,
            budget=budget,
        )
        if entity and state["input"] != self.text:
            state["entity"] = self.entity_dict(spans=False)
        return state

    def state(self, spans: list[Span] | None = None) -> dict[str, Any]:
        """The Clef state of a question about the input: ``{input}``.

        A collection entity's questions also carry the entity (``{input,
        entity}``), because the input names several and the question must say
        which one it is about. Everything else (a ``One``) stays ``{input}``.

        Args:
            spans: Spans to keep if the input is over the Clef budget (the
                entity's own mentions are kept as well).
        """
        return self.fitted(spans=spans, entity=self.info is not None)

    def _nearest_mentions(self, spans: Sequence[Span]) -> list[Span]:
        """For each span, the entity's mention closest to it."""
        mentions = self.mention_spans()
        if not mentions:
            return []
        return list(
            dict.fromkeys(
                min(mentions, key=lambda m: max(m[0] - span[1], span[0] - m[1], 0))
                for span in spans
            )
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
                check; the entity's mention nearest to each keeps a window too).
            anchors: Spans to keep if the full input is over the Clef budget.
        """
        if not narrow_to:
            return self.fitted(spans=anchors)
        keep = [*narrow_to, *self._nearest_mentions(narrow_to)]
        windows = mention_windows(len(self.text), keep, self.run.verify.window_chars)
        return self.fitted(windows=windows)


# --------------------------------------------------------------------------- #
# Asking Clairic, in chunks when the state is too long
# --------------------------------------------------------------------------- #

Asked = dict[str, tuple[ClairicAnswer | None, RenderedState | None]]
"""Per field: the answer and the state to map its spans with."""


def _root_key(plan: SchemaPlan, name: str) -> str:
    """The document key of a top-level field (its alias when it has one)."""
    root = plan.root
    items: list[FieldSpec | FieldGroup | UnionGroup] = [
        *root.specs,
        *root.children,
        *root.unions,
    ]
    for item in items:
        if str(item.path[-1]) == name:
            return item.alias if item.alias is not None else name
    return name


def compact_block(ctx: Ctx, fields: Iterable[str]) -> ExistingBlock | None:
    """The existing block cut down to the identity and the fields being asked.

    Args:
        ctx: The entity context (it holds the block and the plan).
        fields: Wire names of the fields of the request.

    Returns:
        A compacted block (top-level keys of the identity fields and of the
        asked fields only), or ``None`` when there is nothing to compact or it
        would not be shorter.
    """
    block = ctx.existing
    if block is None or block.compacted or block.data is None:
        return None
    plan = ctx.plan
    by_wire = {spec.wire_name: spec for spec in plan.all_specs() if spec.wire_name}
    keep = {_root_key(plan, name.split(".", 1)[0]) for name in plan.identity}
    keep.update(
        _root_key(plan, str(by_wire[wire].path[0]))
        for wire in fields
        if wire in by_wire
    )
    data = {key: value for key, value in block.data.items() if key in keep}
    text = json.dumps(data, ensure_ascii=False)
    if len(text) >= len(block.text):
        return None
    return dataclasses.replace(block, text=text, compacted=True)


def _record_is_the_problem(
    ctx: Ctx,
    rendered: RenderedState,
    response: ClairicResponse,
    anchored: RenderedState | None,
    covered: int,
) -> bool:
    """Whether a truncated response was cut by the record, not by the input.

    True when the whole input was read (so the cut is inside the record) or the
    record takes up at least half of what was read.
    """
    block = ctx.existing
    if block is None or block.compacted or anchored is None:
        return False
    if anchored.input_chars >= covered:
        return True
    read = _chars_read(rendered, response, anchored)
    return read is not None and len(block.text) * 2 >= read


async def ask_clairic(
    ctx: Ctx,
    fields: dict[str, ClairicField],
    *,
    meta: CallMeta,
    windows: list[Span] | None = None,
    descriptions: Mapping[str, str] | None = None,
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

    When chunks give different non-null values for a scalar, every candidate is
    kept: a Clef ``adjudicate`` question over the distinct candidates picks one
    at ``verify.adjudicate_min``; failing that the candidates are checked for
    attribution to the entity in confidence order and the first that passes is
    kept; failing that the most confident one stands and the ladder judges it.

    Args:
        ctx: The entity context.
        fields: Clairic fields keyed by wire name.
        meta: Routing context (the call kind).
        windows: Ranges of the input to ask about; ``None`` for all of it.
        descriptions: Field descriptions by wire name, for the questions that
            settle disagreeing chunks (the instructions are used otherwise).

    Returns:
        Each field's answer and the state to map its spans with. For a chunked
        ask the spans are already in original-input offsets.
    """
    run = ctx.run
    rendered = ctx.rendered(windows=windows)
    response = await run.call_clairic(fields, rendered, meta=meta)
    anchored = run.anchor(rendered, response)
    response = align_list_spans(response, anchored)
    if not response.truncated:
        return {name: (response.answers.get(name), anchored) for name in fields}

    ranges = windows or [(0, len(ctx.text))]
    overlap = run.verify.window_chars
    covered = sum(end - start for start, end in ranges)
    if _record_is_the_problem(ctx, rendered, response, anchored, covered):
        compact = compact_block(ctx, fields)
        if compact is not None:
            ctx.warn(
                f"record_compacted: the existing record of {ctx.plan.name} did not "
                "fit the Clairic request with the input; it was cut down to its "
                "identity and the fields being asked"
            )
            return await ask_clairic(
                dataclasses.replace(ctx, existing=compact),
                fields,
                meta=meta,
                windows=windows,
                descriptions=descriptions,
            )
    reserved = (len(ctx.existing.text) if ctx.existing is not None else 0) + len(
        ctx.trailer
    )
    size = chunk_size(
        _room(_chars_read(rendered, response, anchored), reserved), covered, overlap
    )
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
            _room(_chars_read(ctx.rendered(windows=[chunk]), resp, anch), reserved)
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
        ctx.warn(
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
        ctx.warn(f"clairic_truncated: {message}")
    if not responses:  # could not split: use what the full request returned
        return {name: (response.answers.get(name), anchored) for name in fields}
    responses = [
        (chunk, _without_echoes(ctx, resp, anch), anch)
        for chunk, resp, anch in responses
    ]
    identity = identity_state(ctx.text)
    names = list(fields)
    outcomes = {
        name: merge_field(
            [
                part_of(resp.answers[name], anch, chunk[0])
                for chunk, resp, anch in responses
                if name in resp.answers
            ]
        )
        for name in names
    }
    settled_answers = await _settle_all(
        ctx,
        outcomes,
        {
            name: (descriptions or {}).get(name) or fields[name].instructions or name
            for name in names
        },
    )
    return {name: (settled_answers[name], identity) for name in names}


_LETTERS = "abcdefghij"


async def _settle_all(
    ctx: Ctx,
    outcomes: dict[str, Merged],
    descriptions: dict[str, str],
) -> dict[str, ClairicAnswer | None]:
    """Pick one answer per field where chunks disagree about a scalar.

    All disagreeing fields share one ``adjudicate`` request (their questions are
    ``adjudicate.<wire name>``), and the fields it leaves open share one
    attribution request. A decisive ``neither`` makes the field null.
    """
    result: dict[str, ClairicAnswer | None] = {
        name: merged.answer for name, merged in outcomes.items()
    }
    conflicts = {name: merged for name, merged in outcomes.items() if merged.conflict}
    if not conflicts:
        return result
    run = ctx.run
    shown = {
        name: merged.candidates[: len(_LETTERS)] for name, merged in conflicts.items()
    }
    anchors = [
        part.span
        for merged in conflicts.values()
        for part in merged.candidates
        if part.span is not None
    ]
    questions: dict[str, Question] = {
        f"adjudicate.{name}": ChoiceQuestion(
            type="choice",
            instructions=(
                f"Which value does `input` state for {descriptions[name]} "
                "of `entity.label`?"
            ),
            criteria={
                **{
                    _LETTERS[i]: display(part.answer.value)
                    for i, part in enumerate(parts)
                },
                "neither": "The input states none of these",
            },
        )
        for name, parts in shown.items()
    }
    verdicts = await run.call_clef(
        questions,
        ctx.entity_state(anchors=anchors or None),
        meta=ctx.meta("adjudicate", field=",".join(conflicts)),
    )
    pending: dict[str, Merged] = {}
    for name, merged in conflicts.items():
        values = ", ".join(display(part.answer.value) for part in merged.candidates)
        where = f"the chunks of {ctx.plan.name} gave different values for {name}"
        verdict = verdicts.get(f"adjudicate.{name}")
        p = (
            verdict.probabilities.get(verdict.choice)
            if isinstance(verdict, ChoiceAnswer)
            else None
        )
        if (
            isinstance(verdict, ChoiceAnswer)
            and p is not None
            and p >= run.verify.adjudicate_min
        ):
            if verdict.choice in _LETTERS[: len(shown[name])]:
                ctx.warn(f"chunk_conflict: {where} ({values}); adjudicated")
                result[name] = settled(shown[name][_LETTERS.index(verdict.choice)])
                continue
            if verdict.choice == "neither" and merged.answer is not None:
                ctx.warn(
                    f"chunk_conflict: {where} ({values}); the input states none "
                    "of them, so the field is null"
                )
                result[name] = merged.answer.model_copy(
                    update={
                        "value": None,
                        "span": None,
                        "spans": None,
                        "p_null": max(merged.answer.p_null or 0.0, 0.95),
                    }
                )
                continue
        pending[name] = merged
    if not pending:
        return result
    passed: set[str] = set()
    if run.verify.attribution:
        checks = await run.call_clef(
            {
                f"attr.{name}.{i}": _attribution_question(
                    descriptions[name], part.answer.value, element=False
                )
                for name, merged in pending.items()
                for i, part in enumerate(merged.candidates)
            },
            ctx.entity_state(
                anchors=[
                    part.span
                    for merged in pending.values()
                    for part in merged.candidates
                    if part.span is not None
                ]
                or None
            ),
            meta=ctx.meta("attribution", field=",".join(pending)),
            decisive=True,
        )
        for name, merged in pending.items():
            values = ", ".join(display(part.answer.value) for part in merged.candidates)
            for i, part in enumerate(merged.candidates):
                p = _p(checks, f"attr.{name}.{i}")
                if p is not None and p >= run.verify.attribution_keep:
                    ctx.warn(
                        f"chunk_conflict: the chunks of {ctx.plan.name} gave "
                        f"different values for {name} ({values}); the first that "
                        "is stated about the entity was kept"
                    )
                    result[name] = settled(part)
                    # the ladder would ask this very question next: hand it over
                    ctx.attribution_hints[name] = (display(part.answer.value), p)
                    passed.add(name)
                    break
    for name, merged in pending.items():
        if name not in passed:
            values = ", ".join(display(part.answer.value) for part in merged.candidates)
            ctx.warn(
                f"chunk_conflict: the chunks of {ctx.plan.name} gave different "
                f"values for {name} ({values}) and no check settled it; the most "
                "confident one was kept"
            )
    return result


def _room(read: int | None, reserved: int) -> int | None:
    """How many input characters fit a request once the record's room is taken.

    The record follows the input whole in every request, so it is part of what
    Clairic has to read: the characters it was seen to read cover the input
    *and* (when the state was cut inside the input) none of the record.
    """
    return None if read is None else max(0, read - reserved)


def _fold(text: Any) -> str:
    """Case- and whitespace-insensitive form of a piece of text."""
    return " ".join(str(text).split()).casefold()


def _aligned(answer: ClairicAnswer, anchored: RenderedState) -> ClairicAnswer:
    """Pair every element of a text list answer with a span that covers it.

    Clairic's ``value`` and ``spans`` need not be index-aligned: the first live
    run gave a span for a second "Sarah" that the values left out, so the spans
    after it belonged to the wrong elements. Each element gets the first unused
    span whose covered text equals it (case and whitespace aside), else the first
    free occurrence of its text in the input block of the returned state; if an
    element has none, the answer keeps no spans. An answer whose spans already
    line up is returned as it is.
    """
    value, spans = answer.value, answer.spans
    if not isinstance(value, list) or not value or not spans:
        return answer
    if not all(isinstance(item, str) for item in value):
        return answer
    if any(len(pair) != 2 for pair in spans):
        return answer
    text = anchored.text

    def covered(pair: Sequence[int]) -> str:
        return text[pair[0] : pair[1]]

    if len(spans) == len(value) and all(
        _fold(item) == _fold(covered(pair))
        for item, pair in zip(value, spans, strict=True)
    ):
        return answer
    used: set[int] = set()
    taken: set[tuple[int, int]] = {(pair[0], pair[1]) for pair in spans}
    chosen: list[list[int]] = []
    low, high = anchored.input_span
    for item in value:
        key = _fold(item)
        pick = next(
            (
                i
                for i, pair in enumerate(spans)
                if i not in used and _fold(covered(pair)) == key
            ),
            None,
        )
        if pick is not None:
            used.add(pick)
            chosen.append(list(spans[pick]))
            continue
        at = text.find(item, low, high) if item.strip() else -1
        while at >= 0 and (at, at + len(item)) in taken:
            at = text.find(item, at + 1, high)
        if at < 0:
            fixed = answer.model_copy(update={"spans": None})
            fixed._raw_spans = [list(pair) for pair in spans]
            return fixed
        taken.add((at, at + len(item)))
        chosen.append([at, at + len(item)])
    fixed = answer.model_copy(update={"spans": chosen})
    fixed._raw_spans = [list(pair) for pair in spans]
    return fixed


def align_list_spans(
    response: ClairicResponse, anchored: RenderedState | None
) -> ClairicResponse:
    """Re-pair the spans of the response's text-list answers with their elements.

    See ``_aligned``. The spans the model returned stay available as the
    answer's private ``_raw_spans`` for callers that want all of them.

    Args:
        response: Clairic's response.
        anchored: The state its spans are offsets into (``None``: nothing to do).

    Returns:
        The response (a copy when an answer changed).
    """
    if anchored is None:
        return response
    answers = {
        name: _aligned(answer, anchored) for name, answer in response.answers.items()
    }
    if all(answers[name] is response.answers[name] for name in answers):
        return response
    return response.model_copy(update={"answers": answers})


def _without_echoes(
    ctx: Ctx, response: ClairicResponse, anchored: RenderedState | None
) -> ClairicResponse:
    """Null the answers of a chunk whose span lies in the existing record.

    The chunks are merged in original-input offsets, where the record is not
    visible any more, so an echo has to be caught here. The answer becomes a
    confident null and ``ctx.echoes`` remembers why, so a field that ends up
    null is reported as ``echo_span`` and not as unstated.
    """
    if anchored is None or anchored.existing_span is None:
        return response
    answers = dict(response.answers)
    changed = False
    for name, answer in response.answers.items():
        spans: list[list[int]] = []
        if answer.span is not None and len(answer.span) == 2:
            spans.append(answer.span)
        if answer.spans is not None:
            spans.extend(pair for pair in answer.spans if len(pair) == 2)
        echoed = [pair for pair in spans if anchored.in_existing((pair[0], pair[1]))]
        if not echoed:
            continue
        if isinstance(answer.value, list) and answer.spans is not None:
            keep = [
                i
                for i, pair in enumerate(answer.spans)
                if not (len(pair) == 2 and anchored.in_existing((pair[0], pair[1])))
            ]
            if keep and len(answer.spans) == len(answer.value):
                answers[name] = answer.model_copy(
                    update={
                        "value": [answer.value[i] for i in keep],
                        "spans": [answer.spans[i] for i in keep],
                    }
                )
                ctx.echoes.setdefault(name, []).append(
                    f"{len(answer.value) - len(keep)} element(s) were read from "
                    "the existing record"
                )
                changed = True
                continue
        answers[name] = answer.model_copy(
            update={"value": None, "span": None, "spans": None, "p_null": 0.99}
        )
        ctx.echoes.setdefault(name, []).append("the span lies in the existing record")
        changed = True
    return response.model_copy(update={"answers": answers}) if changed else response


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
        anchored = ctx.run.anchor(rendered, response)
        return chunk, align_list_spans(response, anchored), anchored

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
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return str(int(value))  # a total of 1080.0 is asked about as 1080
    return str(value)


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
    repaired: bool = False


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
        cross: The input-only cross-check answer (updates), resolved the same way.
        crosscheck_agree: Whether the cross-check agreed (``None`` when none
            ran for this field).
        sentences: Whether the elements are sentences read by Clef (the
            ``sentences`` list strategy). They come from the input alone, so there
            is no input-only cross-check to run, and Clairic's list type, which
            cannot read such a field, is never asked to re-read it.
        settle: The answer is a repair of a field that is null, so that the record
            is invalid without it (``RepairConfig.keep_uncertain``): a value whose
            attribution stays between the drop and the keep bar after the second
            look stands.
        escalated: The escalation kind that proposed the value (``relative_dates``,
            ``list_rows``), empty for an answer of the default setup. There is no
            Clairic reading to cross-check it against, and no tie-break to ask the
            model that proposed it: the attribution check (Clef) has to agree.
    """

    spec: FieldSpec
    resolved: Resolved
    type_answer: ChoiceAnswer | None = None
    reject: str | None = None
    reject_detail: str = ""
    reject_p: float | None = None
    attribution_p: float | tuple[float | None, ...] | None = None
    reask: bool = False
    dropped: list[tuple[int, str, str, float | None]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cross: Resolved | None = None
    crosscheck_agree: bool | None = None
    sentences: bool = False
    settle: bool = False
    escalated: str = ""

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


def _reference_day(run: Run) -> date | None:
    """The invoke's reference date as a ``date`` (``None`` without one)."""
    if not run.reference_date:
        return None
    try:
        return date.fromisoformat(run.reference_date[:10])
    except ValueError:
        return None


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
        coerced = coerce(spec, answer, gates, today=_reference_day(ctx.run))
    notes: list[str] = []
    echo_lost: list[tuple[int, str, str]] = []
    if coerced.ok and rendered is not None and rendered.existing_span is not None:
        coerced, echo_lost, everything = _drop_echoes(rendered, coerced)
        if everything:
            return Resolved(
                ok=False,
                reason="echo_span",
                detail="the answer's span lies in the existing record, not the input",
                raw=raw,
                p_null=coerced.p_null,
                confidence=coerced.confidence,
                guess=coerced.guess,
                lost=tuple(echo_lost),
            )
    relocated: dict[int, Span] = {}
    if coerced.ok and rendered is not None:
        coerced, outside, everything, relocated = _drop_outside(
            rendered, coerced, ctx.text, notes
        )
        echo_lost.extend(outside)
        if everything:
            return Resolved(
                ok=False,
                reason="span_outside_input",
                detail="the answer's span is not inside the input block (it lies "
                "in the focus header or the blocks around the input)",
                raw=raw,
                p_null=coerced.p_null,
                confidence=coerced.confidence,
                guess=coerced.guess,
                lost=tuple(echo_lost),
            )
    span: Span | None = (
        relocated.get(0) if not isinstance(coerced.value, list) else None
    )
    if span is None:
        span = _map_span(rendered, coerced.span, notes)
    spans: tuple[Span, ...] | None = None
    if coerced.spans is not None:
        mapped = [
            relocated.get(i) or _map_span(rendered, pair, notes)
            for i, pair in enumerate(coerced.spans)
        ]
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
    lost.extend(echo_lost)
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


def _frame(rendered: RenderedState, pair: Span) -> bool:
    """Whether a span touches the header or the trailing blocks of the state.

    ``rendered.text`` is the state Clairic says it read. Everything before the
    input block's content (the ``<focus>`` header and the sentinel) and
    everything after it (the closing sentinel, the ``<existing>`` block) is the
    frame; a span that touches it is not a quote of the input, whatever else it
    covers.
    """
    text = rendered.text
    start = rendered.input_span[0]
    end = len(text)
    # where the input block ends in the state Clairic read: its closing tag, or,
    # if the tag is not there as sent, whatever else follows the input
    for marker in ("\n</input>", "\n<existing"):
        found = text.find(marker, rendered.input_span[1])
        if found >= 0:
            end = min(end, found)
    if rendered.existing_span is not None:
        end = min(end, rendered.existing_span[0])
    return pair[0] < start or pair[1] > end


def _occurrences(source: str, value: str) -> list[int]:
    """Where ``value`` occurs in ``source``: exactly, else with other whitespace."""
    found = [m.start() for m in re.finditer(re.escape(value), source)]
    if found or not value.split():
        return found
    flexible = r"\s+".join(re.escape(word) for word in value.split())
    return [m.start() for m in re.finditer(flexible, source)]


def _relocate(
    rendered: RenderedState, source: str, value: Any, pair: Span
) -> Span | None:
    """Find a text answer in the original input when its span cannot be mapped.

    ``state_text`` may differ from the state that was sent (line endings
    normalised, a part cut) so that spans after the first difference map to
    nothing. The value is searched in the original input instead: the occurrence
    nearest to where the span would be (counted from the end of the last piece
    that did map), else the first.

    Returns:
        The span in the original input, or ``None`` when the value is not text
        or the input does not contain it.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    starts = _occurrences(source, value)
    if not starts:
        return None
    guess: int | None = None
    if rendered.segments:
        seg_start, seg_end, original = rendered.segments[-1]
        if pair[0] >= seg_end:
            guess = original + (seg_end - seg_start) + (pair[0] - seg_end)
    at = min(starts, key=lambda s: abs(s - guess)) if guess is not None else starts[0]
    length = len(value)
    if source[at : at + length] != value:  # whitespace differed: take the match
        flexible = r"\s+".join(re.escape(word) for word in value.split())
        match = re.compile(flexible).match(source, at)
        length = match.end() - at if match else length
    return (at, at + length)


def _drop_outside(
    rendered: RenderedState, coerced: Coerced, source: str, notes: list[str]
) -> tuple[Coerced, list[tuple[int, str, str]], bool, dict[int, Span]]:
    """Throw away what is not read from the input block.

    A text answer's span must not touch the frame of the state Clairic read (see
    ``_frame``): one that starts in the ``<focus>`` header (which names the entity)
    and ends in the input is not a quote of the input and would put header text in
    the value. A span that merely cannot be mapped (``state_text`` differs from
    what was sent from some point on: CRLF turned into LF) says nothing against the
    value: if the value is found in the original input it stays and its span is
    relocated there. A list is judged element by element, like ``_drop_echoes``.

    Args:
        rendered: The state the answer was asked on, as Clairic returned it.
        coerced: The answer.
        source: The original input.
        notes: Notes of the field (a ``span_relocated`` note is added).

    Returns:
        The answer without such elements, the dropped elements as ``(position in the
        model's answer list, "span_outside_input", detail)``, whether nothing is
        left of the answer and the relocated spans by position in the answer that
        is returned (``0`` for a scalar).
    """
    value = coerced.value

    def judge(item: Any, pair: Span) -> tuple[bool, Span | None]:
        """Whether the element stays, and where it was relocated to, if it was."""
        if rendered.to_original(pair) is not None:
            return True, None
        if _frame(rendered, pair):
            return False, None
        moved = _relocate(rendered, source, item, pair)
        if moved is None:
            return not isinstance(item, str), None  # a number has nothing to find
        return True, moved

    if isinstance(value, list):
        spans = coerced.spans
        if spans is None or len(spans) != len(value):
            return coerced, [], False, {}
        moves: dict[int, Span] = {}
        keep: list[int] = []
        for i, pair in enumerate(spans):
            stays, moved = judge(value[i], pair)
            if stays:
                keep.append(i)
                if moved is not None:
                    moves[i] = moved
        if len(keep) == len(value):
            if moves:
                notes.append("span_relocated")
            return coerced, [], False, moves
        before = coerced.kept if coerced.kept is not None else tuple(range(len(value)))
        lost = [
            (
                before[i],
                "span_outside_input",
                f"{value[i]!r} points outside the input block",
            )
            for i in range(len(value))
            if i not in keep
        ]
        if not keep:
            return coerced, lost, True, {}
        kept = replace(
            coerced,
            value=[value[i] for i in keep],
            spans=tuple(spans[i] for i in keep),
            kept=tuple(before[i] for i in keep),
        )
        if moves:
            notes.append("span_relocated")
        return (
            kept,
            lost,
            False,
            {k: moves[i] for k, i in enumerate(keep) if i in moves},
        )
    if coerced.span is not None:
        stays, moved = judge(value, coerced.span)
        if not stays:
            return coerced, [], True, {}
        if moved is not None:
            notes.append("span_relocated")
            return coerced, [], False, {0: moved}
    return coerced, [], False, {}


def _drop_echoes(
    rendered: RenderedState, coerced: Coerced
) -> tuple[Coerced, list[tuple[int, str, str]], bool]:
    """Throw away what was read from the existing record.

    A scalar whose span lies in the ``<existing>`` block is an echo. For a list
    each element is judged by its own span; the ones that echo are dropped and
    reported as ``(position in the model's answer list, "echo_span", detail)``.

    Returns:
        The answer without the echoed elements, the dropped elements and whether
        nothing is left of the answer.
    """
    value = coerced.value
    if isinstance(value, list):
        spans = coerced.spans
        if spans is None or len(spans) != len(value):
            return coerced, [], False
        keep = [i for i, pair in enumerate(spans) if not rendered.in_existing(pair)]
        if len(keep) == len(value):
            return coerced, [], False
        before = coerced.kept if coerced.kept is not None else tuple(range(len(value)))
        lost = [
            (before[i], "echo_span", f"{value[i]!r} was read from the existing record")
            for i in range(len(value))
            if i not in keep
        ]
        if not keep:
            return coerced, lost, True
        kept = replace(
            coerced,
            value=[value[i] for i in keep],
            spans=tuple(spans[i] for i in keep),
            kept=tuple(before[i] for i in keep),
        )
        return kept, lost, False
    if coerced.span is not None and rendered.in_existing(coerced.span):
        return coerced, [], True
    return coerced, [], False


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
        ctx.state(list(spans) if spans else None),
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


def _answered_by(answer: Answer | None) -> str | None:
    """Who answered, when it was a fallback (the answer's ``provider`` extra)."""
    provider = ((answer.model_extra if answer is not None else None) or {}).get(
        "provider"
    )
    return str(provider) if provider else None


def resolve_clef(spec: FieldSpec, answer: Answer | None, ctx: Ctx) -> Resolved:
    """Turn a Clef answer into a value, apply the thresholds and note who answered.

    The notes of a field answered by a fallback (``answered_by:decisions``, and
    ``hard_answer`` for a bare pick without probabilities) end up in
    ``FieldAnswer.warnings``; see ``_resolve_clef`` for the rest.
    """
    resolved = _resolve_clef(spec, answer, ctx)
    provider = _answered_by(answer)
    if provider is None:
        return resolved
    extra = (answer.model_extra if answer is not None else None) or {}
    hard = bool(extra.get("hard"))
    notes = (*resolved.notes, f"answered_by:{provider}", *(("hard_answer",) * hard))
    return dataclasses.replace(resolved, notes=notes)


def _resolve_clef(spec: FieldSpec, answer: Answer | None, ctx: Ctx) -> Resolved:
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
        descriptions={spec.wire_name: spec.scoped},
    )
    answer, anchored = asked[spec.wire_name]
    return await resolve(ctx, spec, answer, anchored, st.type_answer)


async def reask(ctx: Ctx, st: FieldState) -> Resolved:
    """Ask one field again, alone, over the whole input (no record, no mentions).

    Args:
        ctx: The entity context (an update's input-only one).
        st: The field's state.

    Returns:
        The new answer, resolved; the state is not changed.
    """
    return await _reask(ctx, st, spans=None)


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
            f"Which value does `input` state for {spec.scoped} of `entity.label`?"
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


def _fuzzy(spec: FieldSpec) -> bool:
    """Whether answers for this field may agree on text similarity alone.

    Free text does; dates, times, durations, numbers and codes compare exactly.
    """
    clairic = spec.clairic
    if clairic is None:
        return False
    if clairic.type == "list":
        return clairic.items == "text" and spec.item_kind in ("str", "choice")
    return clairic.type == "text"


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
        and agree(first.value, checked.value, fuzzy=_fuzzy(spec))
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
        questions, ctx.state(), meta=ctx.meta("null_check")
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

_ATTR_TRUE = (
    "The input states this about the entity `entity.label`, or it follows by "
    "simple arithmetic or date math from figures the input states about "
    "`entity.label`."
)
_ATTR_FALSE = (
    "The input does not state it about `entity.label`, or it concerns another entity."
)


_YES = "Yes."
_NO = "No."


_LIST_OF = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?:list|array|set|collection|sequence|group)\s+of\s+"
    r"(?:all\s+)?(?:the\s+)?",
    re.IGNORECASE,
)
_OWNER = re.compile(r"^(?:(?:a|an|the)\s+)?\w+(?:\s\w+)?['\u2019]s\s+", re.IGNORECASE)
_HEAD_END = re.compile(
    r"\s(?:of|for|that|who|which|worth|to|in|about|from|with|on)\s", re.IGNORECASE
)
_IRREGULAR_PLURALS = frozenset({"people", "children", "staff", "data", "men", "women"})
_SCOPE_SUFFIX = re.compile(r"^(.*?)(\s*\([^()]*\))?$", re.DOTALL)


def _element_phrase(description: str) -> str:
    """What a list element is, for "is X ___ of the entity?": a noun phrase.

    The field's description is cleaned of what would double up in the sentence: a
    leading article, "list of" ("one of the list of tags"), the owner of a
    possessive ("one of the person's nickname"). A plural ("the contacts") gives
    "one of the contacts"; a singular ("the person's nickname") gives "a nickname".
    A parenthesised scope ("tags (owner → child)") is kept at the end.
    """
    flat = " ".join(description.split())
    found = _SCOPE_SUFFIX.match(flat)
    core, scope = (found.group(1), found.group(2) or "") if found else (flat, "")
    core = core.strip(" .!?:;")
    core = _ARTICLE.sub("", core, count=1)
    core = _LIST_OF.sub("", core, count=1)
    core = _OWNER.sub("", core, count=1)
    if not core:
        return "one of the things the field asks for"
    head = _HEAD_END.split(f" {core} ", maxsplit=1)[0].split()
    last = head[-1].casefold() if head else ""
    plural = last in _IRREGULAR_PLURALS or (
        last.endswith("s") and not last.endswith(("ss", "us", "is"))
    )
    if plural:
        return f"one of the {core}{scope}"
    article = "an" if core[:1].lower() in "aeiou" else "a"
    return f"{article} {core}{scope}"


_ORDERED_NAME = re.compile(r"^(?:\w+_)?(?:quantity|qty)(?:_\w+)?$")
# a bare count word is a requested quantity only on a line of an order or the like
_UNIT_NAME = re.compile(
    r"^(?:units?|counts?|pieces?|pcs|seats?|tickets?|copies|boxes|items?"
    r"|(?:num|nr|no|n)_\w+|number_of_\w+)$"
)
_ORDER_LIKE = re.compile(
    r"\b(?:line|lines|item|items|order|orders|product|products|cart|basket|booking|"
    r"bookings|reservation|purchase|invoice|shipment|ticket|tickets)\b",
    re.IGNORECASE,
)
_NOT_REQUESTED = re.compile(
    r"\b(?:price|prices|cost|costs|fee|rate|total|totals|sum|subtotal|id|ids|"
    r"identifier|sku|code|to date|so far|sold|shipped|delivered|received|stock|"
    r"inventory|available|pages?)\b|\bnumber of the\b",
    re.IGNORECASE,
)
_STATED_NUMBER = re.compile(r"\b(?:years?|ages?|aged|old|born|yob)\b", re.IGNORECASE)
_ORDERED_UNITS = re.compile(
    r"\b(?:units?|items?|pieces?|copies|pcs|quantity|qty|number|count|how many)\b"
    r".*\b(?:ordered|wanted|requested|needed|"
    r"to\s+(?:order|buy|add|purchase|book|reserve))\b"
    r"|\b(?:ordered|requested|wanted)\s+(?:quantity|units?|items?)\b"
    # a head count is asked for by whoever books: "how many tickets to book"; a
    # count that is only stated ("how many children she has") is not
    r"|\bhow many\b.*\bto\s+(?:book|order|attend|reserve|buy|bring)\b"
    r"|\bnumber of\s+(?:guests|attendees|passengers|people|diners|participants)\b",
    re.IGNORECASE,
)


def _snake(name: str) -> str:
    """A field name as lower-case words joined by underscores (``qtyOrdered``)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def relative_candidate(spec: FieldSpec) -> bool:
    """Whether a changed number could be a RELATIVE edit ("two more") of its old value.

    Any number field except those whose name or description says what is only ever
    stated: a price, cost, total, id, a count to date, a year or an age. The check
    does not rely on the field being NAMED like a quantity (``units``, ``count``,
    ``num_items`` carried the data loss again when it did).
    """
    if spec.clairic is None or spec.clairic.type != "number":
        return False
    words = _snake(str(spec.path[-1])).replace("_", " ")
    return not (
        _NOT_REQUESTED.search(words)
        or _NOT_REQUESTED.search(spec.description)
        or _STATED_NUMBER.search(words)
        or _STATED_NUMBER.search(spec.description)
    )


def requestable(spec: FieldSpec, plan: SchemaPlan | None = None) -> bool:
    """Whether the input may ASK for the field's value instead of stating it.

    An ORDERED quantity can be requested ("add two more of the blue mugs": a
    quantity of 2 that no sentence states): a number field named ``quantity`` or
    ``qty`` (or ``units``, ``count``, ``pieces``, ``seats``, ``tickets``, ``copies``,
    ``boxes``, ``num_*`` when the schema is a line of an order, a booking or the
    like: ``plan``), or whose description is a number of units, items or copies
    ordered, wanted or to order. A total, a price, a cost, an id, a count to date
    ("copies sold to date", "total units shipped so far", "page count") or "the number
    of the invoice" is stated: asking "state or ask for" about a derived total made
    Clef less sure of it (0.88 became 0.52).
    """
    if spec.clairic is None or spec.clairic.type != "number":
        return False
    name = _snake(str(spec.path[-1]))
    if _NOT_REQUESTED.search(name.replace("_", " ")) or _NOT_REQUESTED.search(
        spec.description
    ):
        return False
    if _ORDERED_NAME.match(name) is not None:
        return True
    if (
        plan is not None
        and _UNIT_NAME.match(name) is not None
        and _ORDER_LIKE.search(f"{plan.title} {plan.description}") is not None
    ):
        return True
    return _ORDERED_UNITS.search(spec.description) is not None


def _today_note(ctx: Ctx, spec: FieldSpec) -> str:
    """Say what day it is to the attribution of a date.

    The note reads ``Today is Monday, 2026-03-23.``

    Only for a date or datetime field (or a list of them). The day is the invoke's
    ``Run.today``: its reference date, or the UTC date pinned when it began (so the
    note is there without a reference date too, and is the date TypeSafe resolved a
    relative date against). The attribution requests of date fields therefore carry
    the date in every run, which changes their request keys: a cassette recorded
    before this note no longer replays those requests strictly, and an invoke without
    a ``reference_date`` asks a different question each day, so a strict replay needs
    one (or the cassette has to be recorded again).

    A date that the input gives as "yesterday", "three days ago" or "last Tuesday" was
    resolved against the reference date; a question that asks whether the input states
    that date cannot be answered by a model that does not know what day it is (live:
    0.03 for ``2026-03-17`` over "Completed last Tuesday", 0.09 for yesterday's date).
    """
    clairic = spec.clairic
    if clairic is None:
        return ""
    kind = clairic.items if clairic.type == "list" else clairic.type
    if kind not in ("date", "datetime"):
        return ""
    today = date.fromisoformat(ctx.run.today)
    return f"Today is {today:%A}, {today.isoformat()}. "


_CLAUSE = re.compile(r"^(?:how|whether|when|where|why|what|who|whom|which)\b", re.I)
"""A description that is a clause ("how the person relates to the speaker"), not a
noun phrase: "<description> of the entity" would hang the entity on its last noun
("the speaker of Olivia Davis")."""

_FRESH_NOTE = (
    " `entity.label` is new, nothing of it was held before: an amount the input asks "
    'to add ("two more") is the whole amount of it.'
)


def _attribution_question(
    description: str,
    value: Any,
    *,
    element: bool,
    ask_for: bool = False,
    today: str = "",
    fresh: bool = False,
) -> NoulQuestion:
    """The question "is this value stated about the entity?" for a value or element.

    A list element is asked as "is X one of the <field> of the entity?": the
    field is named, since the element alone says nothing about what it is for
    (live: a non-contact scored higher than a contact when it was not). The answers
    are plain yes and no. A quantity the input may request rather than state (an
    order's "add two more") is asked as "does the input ask for it".
    """
    text = display(value)
    if element:
        return NoulQuestion(
            type="noul",
            instructions=(
                f'{today}According to `input`, is "{text}" '
                f"{_element_phrase(description)} "
                "of `entity.label`? Something the input asks to include counts."
            ),
            criteria=NoulCriteria(true=_YES, false=_NO),
        )
    if ask_for:
        # "Does the input ask for 2 of the entity (as the number of units ordered)?"
        # is the shape that told 2 (0.78 to 0.86) from 3 (0.02 to 0.04) and kept a
        # stated 1 (0.92 to 0.98); the longer "state or ask for X to be V" did not
        noun = _ARTICLE.sub("", description.strip(), count=1)
        instructions = f"Does `input` ask for {text} of `entity.label` (as the {noun})?"
        if fresh:
            # the engine knows the entity is new; the model reading the input does not
            # (live: "two more of the blue mugs" on an order that holds none, 0.48)
            instructions += _FRESH_NOTE
    else:
        clause = _CLAUSE.match(description.strip()) is not None
        about = "for `entity.label`, " if clause else ""
        of = "" if clause else " of `entity.label`"
        instructions = (
            f"{today}Does `input` state that {about}{description}{of} is "
            f'"{text}", or does it follow by simple arithmetic or date math from '
            "figures the input states about `entity.label`?"
        )
    return NoulQuestion(
        type="noul",
        instructions=instructions,
        criteria=NoulCriteria(true=_ATTR_TRUE, false=_ATTR_FALSE),
    )


_PASSAGE_MAX = 240
"""Longest passage a re-check quotes, in characters."""


_CLAUSE_BREAK = re.compile(
    r"(?:[,;:]\s+|\s+(?:and|but|while|whereas|then|so)\s+)", re.IGNORECASE
)


def _head(text: str, limit: int) -> str:
    """The start of ``text`` in at most ``limit`` characters, cut at a word end."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if not text[limit].isspace():
        space = cut.rfind(" ")
        cut = cut[:space] if space > 0 else cut
    return cut.rstrip(" ,;:-")


def _tail(text: str, limit: int) -> str:
    """The end of ``text`` in at most ``limit`` characters, cut at a word start."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    if not text[-limit - 1].isspace():
        space = cut.find(" ")
        cut = cut[space + 1 :] if space >= 0 else cut
    return cut.lstrip(" ,;:-")


def _fit_passage(before: str, mid: str, after: str) -> str:
    """A passage of at most ``_PASSAGE_MAX`` characters around ``mid``.

    The clause is kept whole when it fits. Otherwise the start of the clause (its
    subject) is kept, then a gap, then the words next to the span, all cut at word
    boundaries; a span longer than the cap is cut itself.
    """
    if len(before) + len(mid) + len(after) <= _PASSAGE_MAX:
        return before + mid + after
    gap = " … "
    # a span longer than the cap is cut itself, keeping room for the clause's subject
    reserve = min(len(before.strip()), 60) + len(gap) if before.strip() else 0
    mid = _head(mid, _PASSAGE_MAX - reserve)
    room = _PASSAGE_MAX - len(mid)
    post = _head(after, room // 4)
    room -= len(post)
    if len(before) <= room:
        return before + mid + post
    subject = _head(before, min(60, max(0, room - len(gap)) // 2))
    near = _tail(before, room - len(subject) - len(gap))
    return f"{subject}{gap}{near}{mid}{post}" if near else f"{subject}{gap}{mid}{post}"


def _passage(text: str, span: Span | None) -> str | None:
    """The clause of the input a span lies in, as the passage a re-check quotes.

    The sentence the span is in, cut to the clause around it when the sentence has
    several ("Please add two more of the blue mugs and drop the quantity of the tea
    towels to 1" gives the first: another clause's number would otherwise pull the
    answer about the first one down, live 0.4793). A clause over ``_PASSAGE_MAX``
    characters keeps its subject and the words around the span, cut at word
    boundaries; the passage never exceeds the cap, whatever the span's length.
    """
    if span is None:
        return None
    for sentence in split_sentences(text):
        if sentence.start <= span[0] < sentence.end:
            start, end = sentence.start, sentence.end
            low, high = start, end
            for cut in _CLAUSE_BREAK.finditer(text, start, end):
                if cut.end() <= span[0]:
                    low = cut.end()
                elif cut.start() >= span[1] and high == end:
                    high = cut.start()
            if len(text[low:high].split()) >= 3:  # a clause with something in it
                start, end = low, high
            at = min(max(span[0], start), end)
            to = min(max(span[1], at), end)

            def clean(piece: str) -> str:
                # collapse runs of white space; keep one at an edge so words stay apart
                squeezed = " ".join(piece.split())
                lead = " " if piece[:1].isspace() and squeezed else ""
                trail = " " if piece[-1:].isspace() and squeezed else ""
                return f"{lead}{squeezed}{trail}"

            passage = _fit_passage(
                clean(text[start:at]), clean(text[at:to]), clean(text[to:end])
            )
            return _head(passage.strip().replace('"', "'"), _PASSAGE_MAX)
    return None


def _recheck_question(
    question: NoulQuestion, text: str, span: Span | None
) -> NoulQuestion:
    """The same attribution question, asked differently: the second look.

    A middle-band answer is checked once more after a re-ask, but on a short input
    the re-ask changes neither the value nor the state (the input is the same whole
    text), so the request would be byte-identical and Clef would give the same
    middle-band answer: the value would always be dropped. The second look quotes
    the passage the re-ask found the value in (or, with no span, asks to read the
    input again for the entity only), so it is a different question.
    """
    quote = _passage(text, span)
    lead = (
        f'Reading the passage "{quote}" of `input`: '
        if quote
        else "Reading `input` once more, and only what it says about `entity.label`: "
    )
    return question.model_copy(
        update={"instructions": f"{lead}{question.instructions}"}
    )


def _is_list(st: FieldState) -> bool:
    return isinstance(st.resolved.value, list)


def _attr_id(st: FieldState, index: int | None) -> str:
    base = f"attr.{st.spec.wire_name}"
    return base if index is None else f"{base}.{index}"


def _attribution_questions(
    states: list[FieldState], plan: SchemaPlan, ctx: Ctx
) -> dict[str, Question]:
    questions: dict[str, Question] = {}
    for st in states:
        today = _today_note(ctx, st.spec)
        if _is_list(st):
            for index, element in enumerate(st.resolved.value):
                questions[_attr_id(st, index)] = _attribution_question(
                    st.spec.scoped, element, element=True, today=today
                )
        else:
            questions[_attr_id(st, None)] = _attribution_question(
                st.spec.scoped,
                st.resolved.value,
                element=False,
                ask_for=requestable(st.spec, plan),
                today=today,
                fresh=ctx.fresh,
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
    questions = _attribution_questions(targets, ctx.plan, ctx)
    known: dict[str, Answer] = {}
    for st in targets:
        hint = ctx.attribution_hints.get(st.spec.wire_name)
        qid = _attr_id(st, None)
        if (
            hint is not None
            and not _is_list(st)
            and qid in questions
            and hint[0] == display(st.resolved.value)
        ):
            del questions[qid]
            known[qid] = NoulAnswer(type="noul", noul=hint[1])
    if not questions and not known:
        return
    anchors = [span for st in targets for span in st.spans]
    answers: dict[str, Answer] = {}
    if questions:
        answers = await ctx.run.call_clef(
            questions,
            ctx.entity_state(anchors=anchors or None),
            meta=ctx.meta("attribution"),
            decisive=True,
        )
    answers.update(known)
    await gather(*(_judge(ctx, st, answers) for st in targets))


def _note_attribution_provider(st: FieldState, answers: dict[str, Answer]) -> None:
    """Say on the field when a fallback gave the attribution it was judged by."""
    keys = (
        [_attr_id(st, i) for i in range(len(st.resolved.value))]
        if _is_list(st)
        else [_attr_id(st, None)]
    )
    for key in keys:
        provider = _answered_by(answers.get(key))
        if provider and f"attribution_by:{provider}" not in st.notes:
            st.notes.append(f"attribution_by:{provider}")


async def _judge(ctx: Ctx, st: FieldState, answers: dict[str, Answer]) -> None:
    verify = ctx.run.verify
    _note_attribution_provider(st, answers)
    if not _is_list(st):
        p = _p(answers, _attr_id(st, None))
        st.attribution_p = p
        if p is None and ctx.run.was_refused(
            ctx.meta("attribution"), _attr_id(st, None)
        ):
            # the model declined to answer: that is final, and not asked again
            st.set_reject(
                "attribution_missing",
                "the model refused the attribution question (decisions_refusal)",
                None,
            )
            return
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
    probabilities: list[float | None] = []
    keep: list[bool] = []
    pending: list[int] = []
    for index in range(len(elements)):
        answered = _p(answers, _attr_id(st, index))
        probabilities.append(answered)
        if answered is not None and answered >= verify.attribution_keep:
            keep.append(True)
        elif answered is not None and answered < verify.attribution_drop:
            keep.append(False)
            _drop(st, index, elements[index], answered, missing=False)
        elif answered is None and ctx.run.was_refused(
            ctx.meta("attribution"), _attr_id(st, index)
        ):  # declined: final, not asked again
            keep.append(False)
            _drop(st, index, elements[index], None, missing=True)
        else:  # in the middle band, or unanswered (which is not a pass)
            keep.append(False)
            pending.append(index)
    st.attribution_p = tuple(probabilities)
    if pending and st.sentences:
        # Clairic's list type cannot re-read a sentence: not stated clearly enough
        for index in pending:
            was = probabilities[index]
            _drop(st, index, elements[index], was, missing=was is None)
    elif pending:
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
            _attr_id(st, None): _recheck_question(
                _attribution_question(
                    st.spec.scoped,
                    checked.value,
                    element=False,
                    ask_for=requestable(st.spec, ctx.plan),
                    today=_today_note(ctx, st.spec),
                    fresh=ctx.fresh,
                ),
                ctx.text,
                checked.span,
            )
        },
        ctx.entity_state(spans or None),
        meta=ctx.meta("attribution", field=st.spec.wire_name),
        decisive=True,
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
    elif p >= verify.attribution_drop and (
        broken := await _tiebreak_scalar(ctx, st, checked, p)
    ):
        if broken >= verify.attribution_keep:
            _adopt(st, checked)
        else:
            st.set_reject(
                "attribution",
                "the escalation model says the input does not state it either",
                broken,
            )
    elif st.settle and p >= verify.attribution_drop:
        # neither confirmed nor refuted, and a null here loses the whole record
        _adopt(st, checked)
        st.notes.append(
            f"attribution_uncertain: p={p:.2f} (kept: a required field, the same "
            "value twice)"
        )
    else:
        st.set_reject(
            "attribution", "still not stated about the entity after a re-ask", p
        )


async def _tiebreak_scalar(
    ctx: Ctx, st: FieldState, checked: Resolved, p: float
) -> float | None:
    """The escalation tier's answer to an attribution the second look left in the band.

    ``None`` when the tier is off, out of calls, failed or has nothing to say (the
    value's fate is then what it was without the tier). Never for a removal: the
    probability it returns is at most ``prob_cap``, below every removal gate.
    """
    if not ctx.run.escalation.runs("attribution_tiebreak"):
        return None
    question = _attribution_question(
        st.spec.scoped,
        checked.value,
        element=False,
        ask_for=requestable(st.spec, ctx.plan),
        today=_today_note(ctx, st.spec),
        fresh=ctx.fresh,
    )
    q = await tiebreak(ctx, st, str(question.instructions), span=checked.span)
    if q is not None:
        st.attribution_p = q
        st.notes.append(f"attribution_tiebreak: clef {p:.2f} -> escalation {q:.2f}")
    return q


async def _recheck_elements(
    ctx: Ctx,
    st: FieldState,
    elements: list[Any],
    spans: list[Span] | None,
    pending: list[int],
    keep: list[bool],
    probabilities: list[float | None],
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
                if agree(elements[index], other, fuzzy=_fuzzy(st.spec)):
                    candidates[index] = (other, found_spans[j] if found_spans else None)
                    del found[j]
                    if found_spans:
                        del found_spans[j]
                    break
    for index in pending:
        if index not in candidates:
            was = probabilities[index]
            _drop(st, index, elements[index], was, missing=was is None)
    if not candidates:
        return
    questions: dict[str, Question] = {
        _attr_id(st, index): _recheck_question(
            _attribution_question(
                st.spec.scoped, value, element=True, today=_today_note(ctx, st.spec)
            ),
            ctx.text,
            span,
        )
        for index, (value, span) in candidates.items()
    }
    narrowed = [span for _, span in candidates.values() if span is not None]
    answers = await ctx.run.call_clef(
        questions,
        ctx.entity_state(narrowed or None),
        meta=ctx.meta("attribution", field=st.spec.wire_name),
        decisive=True,
    )
    for index, (value, _) in candidates.items():
        p = _p(answers, _attr_id(st, index))
        if p is not None:
            probabilities[index] = p
        if p is not None and verify.attribution_drop <= p < verify.attribution_keep:
            asked = await tiebreak(
                ctx,
                st,
                _attribution_question(
                    st.spec.scoped,
                    value,
                    element=True,
                    today=_today_note(ctx, st.spec),
                ).instructions,
                span=candidates[index][1],
                detail=f"[{index}]",
            )
            if asked is not None:
                st.notes.append(f"attribution_tiebreak: clef {p:.2f} -> {asked:.2f}")
                p = probabilities[index] = asked
        if p is not None and p >= verify.attribution_keep:
            keep[index] = True
            elements[index] = value
        else:
            _drop(st, index, value, p, missing=p is None)
    st.attribution_p = tuple(probabilities)


# --------------------------------------------------------------------------- #
# Input-only cross-check (updates)
# --------------------------------------------------------------------------- #


def _cross_raw(st: FieldState) -> Any:
    """What the input-only pass gave for a field, whatever its confidence.

    Agreement with the primary answer is judged on this: a hesitant reading that
    says the same thing still shows the input gives the value.
    """
    if st.cross is None or not st.cross.ok:
        return None
    return st.cross.value


def usable_cross(st: FieldState) -> Resolved | None:
    """The input-only answer if it passes the answer-level thresholds and is not null.

    Only a usable answer may be taken as a field's value (see ``adopt_cross``).
    """
    if st.cross is None:
        return None
    checked = _effective(st.spec, st.cross)
    return checked if checked is not None and checked.value is not None else None


def adopt_cross(st: FieldState) -> bool:
    """Take the input-only reading as the field's answer.

    Used when the primary pass gave nothing usable (null, an echo of the record,
    or just the record's own value) and the input alone says something. The
    reason the primary answer was thrown away stays in the field's notes.

    Returns:
        Whether there was a usable cross-check answer to take.
    """
    checked = usable_cross(st)
    if checked is None:
        return False
    if st.reject:
        st.notes.append(f"{st.reject}: replaced by the input-only reading")
    _adopt(st, checked)
    st.crosscheck_agree = False
    st.notes.append("adopted_from_crosscheck")
    return True


async def contest_cross(ctx: Ctx, st: FieldState) -> bool:
    """Settle a primary reading that equals the record against a different one.

    The primary pass read the record's own value and pointed at text in the
    input for it (a genuine reading, not an echo), while the input-only pass
    reads something else. Neither is taken on faith: Clef is asked which of the
    two the input states (call kind ``adjudicate``, the same question as for two
    disagreeing values). Only a decisive win of the input-only value takes it
    over (see ``adopt_cross``); anything else leaves the field as the record
    has it.

    Args:
        ctx: The entity context (the input-only one, for an update).
        st: The field; its primary answer equals the record's value.

    Returns:
        Whether the input-only reading was adopted.
    """
    found = usable_cross(st)
    if found is None:
        return False
    verdict, _ = await _adjudicate(ctx, st, st.resolved.value, found.value)
    if verdict == "b" and adopt_cross(st):
        st.notes.append("adjudicated_to_crosscheck")
        return True
    st.notes.append(
        "crosscheck_overruled: the input-only reading differs, but the input was "
        "not shown to state it rather than the record's value"
    )
    return False


def _cross_list(st: FieldState, new: list[int] | None) -> None:
    """Check the new elements of a list against the input-only pass.

    A new element stands only if the cross-check list has an agreeing element;
    the others are dropped as ``echo_crosscheck``. A list has no "two different
    values" case: an element the second pass did not give is simply unsupported.
    """
    elements = list(st.resolved.value)
    spans = list(st.resolved.spans) if st.resolved.spans else None
    found = _cross_raw(st)
    others = list(found) if isinstance(found, list) else []
    fuzzy = _fuzzy(st.spec)
    wanted = range(len(elements)) if new is None else new
    unsupported = [
        index
        for index in wanted
        if not any(agree(elements[index], other, fuzzy=fuzzy) for other in others)
    ]
    st.crosscheck_agree = not unsupported
    if not unsupported:
        return
    for index in unsupported:
        st.dropped.append(
            (_origin(st, index), "echo_crosscheck", display(elements[index]), None)
        )
    survivors = [i for i in range(len(elements)) if i not in unsupported]
    if not survivors:
        st.set_reject(
            "echo_crosscheck",
            "the input alone gives none of these elements",
            None,
        )
        return
    st.resolved = replace(
        st.resolved,
        value=[elements[i] for i in survivors],
        spans=tuple(spans[i] for i in survivors) if spans else None,
        origin=tuple(_origin(st, i) for i in survivors),
    )
    st.notes.append(
        f"dropped {len(unsupported)} element(s): not given by the input alone"
    )


async def _cross_scalar(ctx: Ctx, st: FieldState) -> None:
    spec = st.spec
    found = _cross_raw(st)
    if found is None:
        st.crosscheck_agree = False
        st.set_reject(
            "echo_crosscheck",
            "the input alone does not give this value (the second pass, "
            "without the existing record, found none)",
            None,
        )
        return
    if agree(st.resolved.value, found, fuzzy=_fuzzy(spec)):
        st.crosscheck_agree = True
        return
    st.crosscheck_agree = False
    verdict, p = await _adjudicate(ctx, st, st.resolved.value, found)
    if verdict == "a":
        return
    if verdict == "b" and usable_cross(st) is not None:
        st.notes.append("adjudicated_to_crosscheck")
        adopt_cross(st)
        return
    st.set_reject(
        "crosscheck_disagree",
        "the pass with the record and the input-only pass gave different values "
        "and the adjudication was not decisive",
        p,
    )


async def crosscheck(
    ctx: Ctx,
    states: Sequence[FieldState],
    new_elements: Mapping[str, list[int]] | None = None,
) -> None:
    """Check each candidate change against the input-only pass.

    ``FieldState.cross`` holds the answer of the second Clairic pass over the
    same field, made without the existing record. For a scalar: the cross-check
    agreeing (normalised equality, text similarity only for free text, as in
    ``agree``) lets the value proceed; a null cross-check while the primary
    answer is not null rejects it as ``echo_crosscheck`` (it came from the
    record); two different values go to a Clef ``choice`` ``{a: primary,
    b: cross-check, neither}``: ``a`` keeps the primary value, ``b`` takes the
    cross-check's, anything else (or a probability below
    ``verify.adjudicate_min``) rejects it as ``crosscheck_disagree``. For a list
    each new element needs an agreeing element in the cross-check.

    Args:
        ctx: The entity context.
        states: The fields that would change the record.
        new_elements: For list fields (by wire name), the positions of the
            elements that are new relative to the record; all of them if absent.
    """
    lookup = new_elements or {}

    async def one(st: FieldState) -> None:
        if st.sentences or st.escalated:  # no Clairic reading to check against
            st.crosscheck_agree = True
        elif _is_list(st):
            _cross_list(st, lookup.get(st.spec.wire_name))
        else:
            await _cross_scalar(ctx, st)

    await gather(*(one(st) for st in states if st.has_value()))


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
                and not st.sentences
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
