"""Matching entities to existing records, one-to-one, and flagging deletes.

Steps 8 and 9 of "Collection engine" in ``docs/DESIGN.md``:

1. every entity gets ONE Clef ``choice`` (call kind ``match``, question id
   ``match``) over the existing records and ``__new__``; its state is the
   entity-anchored ``{input, entity: {label, mentions}}`` and the records are the
   options' descriptions (``rec_<i>``: the record as a JSON object). More than 254
   records are cut to the 254 most similar to the entity by identity
   (``match_prefiltered``); a request that would be over the Clef state budget
   renders the records compactly and, if that is not enough, offers fewer of them
   (``match_compacted``);
2. the answers are turned into ONE assignment of entities to records by the
   Hungarian algorithm over ``-log p`` (``hungarian``, ``assign``), with a
   ``__new__`` column per entity so that any number of them can be new. Two
   entities that both claim a record with probability 0.5 or more are a collision
   (``match_collision``): the loser takes its best feasible alternative, or, with
   ``match.collision="merge"``, the two are first checked for being one entity;
3. an assignment stands only if its option's probability reaches
   ``match.accept`` and the answer's confidence ``match.min_confidence``; else the
   entity is ambiguous (``match.on_ambiguous``);
4. deletes: one Clef ``noul`` per existing record, "does the input state that this
   record should be removed?" (call kind ``delete``).

``hungarian`` and ``assign`` are pure and tested against brute force.
"""

import difflib
import itertools
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from trustcall_system_one._compile import SchemaPlan
from trustcall_system_one._config import MatchConfig
from trustcall_system_one._coref import (
    NEW,
    canonical_label,
    identity_compatible,
    name_tokens,
    nicknames,
    strong_compatible,
)
from trustcall_system_one._errors import AmbiguousMatchError
from trustcall_system_one._render import estimate_tokens, mention_windows
from trustcall_system_one._run import Run, gather
from trustcall_system_one._types import EntityInfo, Mention, SkippedEntity
from trustcall_system_one._verify import Ctx, Placement
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
)
from trustcall_system_one.transport import CallMeta

__all__ = [
    "MAX_OPTIONS",
    "NEW",
    "Candidate",
    "MatchAnswer",
    "MatchOutcome",
    "assign",
    "candidate_options",
    "collisions",
    "compact_record",
    "flag_deletes",
    "hungarian",
    "identity_compatible",
    "identity_keys",
    "identity_values",
    "match_entities",
    "similarity",
    "token_set_ratio",
]

MAX_OPTIONS = 254
"""Records one match question offers (Clef allows 255 options; ``__new__`` is one)."""
CLAIM_MIN = 0.5
"""Probability from which an entity's most likely record counts as its claim."""
_FLOOR = 1e-9
_TIE_EPS = 1e-12
_LIST_ELEMENTS = 3
_TEXT_CHARS = 160
_SHOWN_CANDIDATES = 10
_NOTICEABLE = 0.01
_DIMINUTIVE = 0.9
_PRUNE = 1e-3
"""Below this probability the options of an entity are not told apart."""


@dataclass(frozen=True)
class Candidate:
    """An existing record, as matching sees it.

    Attributes:
        record_id: The record's id.
        data: The record with its keys normalised to the schema's validation keys.
    """

    record_id: str
    data: Mapping[str, Any] = field(hash=False)


@dataclass(frozen=True)
class MatchAnswer:
    """One entity's match answer.

    Attributes:
        probs: Probability of each option the model was offered: a record id, or
            ``"__new__"``. Empty when the model gave no answer.
        confidence: The answer's confidence (``None`` without an answer).
    """

    probs: Mapping[str, float] = field(default_factory=dict, hash=False)
    confidence: float | None = None

    def p(self, option: str) -> float:
        """The probability of an option (0 for one that was not offered)."""
        return self.probs.get(option, 0.0)


@dataclass(frozen=True)
class MatchOutcome:
    """What matching decided.

    Attributes:
        entities: The entities after any collision merges, each with its
            ``match_probs`` filled in, in order of first mention.
        targets: The existing record id each entity is to update, or ``None`` for
            an entity that is new, by entity key. An ambiguous entity is only here
            (as new) with ``on_ambiguous="insert"``.
        skipped: The ambiguous entities (``ambiguous_match``).
        confirmed: The keys of the entities matched by the pairwise confirmation
            (``match_confirm``) rather than by the choice question.
    """

    entities: tuple[EntityInfo, ...] = ()
    targets: Mapping[str, str | None] = field(default_factory=dict, hash=False)
    skipped: tuple[SkippedEntity, ...] = ()
    confirmed: frozenset[str] = frozenset()


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def hungarian(cost: Sequence[Sequence[float]]) -> list[int]:
    """Minimum-cost assignment of rows to distinct columns (Kuhn-Munkres).

    The matrix may be rectangular as long as it has at least as many columns as
    rows. Pure Python, O(rows^2 * columns); the result is deterministic.

    Args:
        cost: ``cost[i][j]`` is the cost of giving column ``j`` to row ``i``.

    Returns:
        For each row the column it gets.

    Raises:
        ValueError: If there are more rows than columns, or the rows differ in
            length.
    """
    rows = len(cost)
    if rows == 0:
        return []
    cols = len(cost[0])
    if any(len(row) != cols for row in cost):
        raise ValueError("every row of the cost matrix must have the same length")
    if rows > cols:
        raise ValueError(f"{rows} rows cannot take distinct columns of only {cols}")
    inf = math.inf
    # 1-based potentials and matching; column 0 is a sentinel
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    owner = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for row in range(1, rows + 1):
        owner[0] = row
        col0 = 0
        slack = [inf] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[col0] = True
            current = owner[col0]
            delta, col1 = inf, 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                reduced = cost[current - 1][col - 1] - u[current] - v[col]
                if reduced < slack[col]:
                    slack[col], way[col] = reduced, col0
                if slack[col] < delta:
                    delta, col1 = slack[col], col
            for col in range(cols + 1):
                if used[col]:
                    u[owner[col]] += delta
                    v[col] -= delta
                else:
                    slack[col] -= delta
            col0 = col1
            if owner[col0] == 0:
                break
        while col0:
            previous = way[col0]
            owner[col0] = owner[previous]
            col0 = previous
    result = [-1] * rows
    for col in range(1, cols + 1):
        if owner[col]:
            result[owner[col] - 1] = col - 1
    return result


def assign(
    probabilities: Sequence[Mapping[str, float]],
    records: Sequence[str],
    *,
    prune: float = 0.0,
) -> list[str]:
    """One-to-one assignment of entities to records, any number of them new.

    The cost of giving an entity an option is ``-log(max(p, floor))``, the floor
    being ``prune`` (at least 1e-9): every probability below it costs the same. The
    columns are the records plus one ``__new__`` column per entity (so every entity
    can be new, and ``__new__`` costs each entity its own ``__new__`` probability);
    a record goes to at most one entity, and the entity that loses a contested
    record takes the best option that is still free. Exact ties go to the earlier
    entity and the earlier record, so the result never depends on dictionary order.

    The result is the exact optimum of that cost matrix. A record that every entity
    gives no more than the floor is left out of the matrix: it costs each of them
    exactly what its own ``__new__`` column costs at most, so it is never worth
    taking, and leaving it out keeps the matrix small with hundreds of records. A
    larger ``prune`` makes more of the small probabilities equal (a stronger claim
    still always beats a weaker one above it).

    Args:
        probabilities: For each entity the probability of each option (a record id
            or ``"__new__"``); an option that is missing has probability 0.
        records: The record ids that are columns, in order.
        prune: The floor below which probabilities are not told apart.

    Returns:
        For each entity the record id it gets, or ``"__new__"``.
    """
    n = len(probabilities)
    if n == 0:
        return []
    floor = max(prune, _FLOOR)

    def cost(probs: Mapping[str, float], option: str) -> float:
        return -math.log(max(probs.get(option, 0.0), floor))

    ids = [
        rid
        for rid in records
        if any(probs.get(rid, 0.0) > floor for probs in probabilities)
    ]
    matrix: list[list[float]] = []
    for i, probs in enumerate(probabilities):
        row = [cost(probs, rid) for rid in ids]
        row.extend([cost(probs, NEW)] * n)
        # break exact ties towards the earlier entity (the lower column)
        weight = (n - i) * _TIE_EPS
        matrix.append([c + weight * j for j, c in enumerate(row)])
    return [ids[j] if j < len(ids) else NEW for j in hungarian(matrix)]


def collisions(
    probabilities: Sequence[Mapping[str, float]], claim_min: float = CLAIM_MIN
) -> dict[str, list[int]]:
    """The records that two or more entities claim.

    An entity claims its most likely option when that is a record and has
    probability ``claim_min`` or more.

    Args:
        probabilities: For each entity the probability of each option.
        claim_min: The probability a claim needs.

    Returns:
        The claimants (entity positions) of each contested record id.
    """
    claims: dict[str, list[int]] = {}
    for index, probs in enumerate(probabilities):
        if not probs:
            continue
        best = max(probs, key=lambda option: (probs[option], option == NEW))
        if best != NEW and probs[best] >= claim_min:
            claims.setdefault(best, []).append(index)
    return {rid: who for rid, who in claims.items() if len(who) > 1}


# --------------------------------------------------------------------------- #
# Similarity and rendering
# --------------------------------------------------------------------------- #


def token_set_ratio(first: str, second: str) -> float:
    """How alike two texts are as sets of words, from 0 to 1.

    The words both have are compared with each text's own extra words, so
    "Sarah" fits "Sarah Johnson" fully and word order does not matter.
    """
    a, b = set(name_tokens(first)), set(name_tokens(second))
    if not a or not b:
        return 0.0
    both = " ".join(sorted(a & b))
    left = " ".join(filter(None, [both, *sorted(a - b)]))
    right = " ".join(filter(None, [both, *sorted(b - a)]))

    def ratio(x: str, y: str) -> float:
        return difflib.SequenceMatcher(None, x, y).ratio() if x and y else 0.0

    return max(ratio(both, left), ratio(both, right), ratio(left, right))


def similarity(values: Sequence[str], mentions: Sequence[str]) -> float:
    """How well a record's identity values fit an entity's mentions, from 0 to 1.

    The best token-set ratio over every value and mention; two words that are
    forms of one given name (Bob and Robert) count as 0.9.

    Args:
        values: The record's identity value(s).
        mentions: The texts the input uses for the entity.
    """
    best = 0.0
    for value in values:
        mine = name_tokens(value)
        for mention in mentions:
            score = token_set_ratio(value, mention)
            if score < _DIMINUTIVE and any(
                nicknames(x, y) for x in mine for y in name_tokens(mention)
            ):
                score = _DIMINUTIVE
            best = max(best, score)
    return best


def identity_keys(plan: SchemaPlan, names: Sequence[str]) -> tuple[str, ...]:
    """The keys of a normalised record that hold the identity fields."""
    keys: list[str] = []
    for name in names:
        spec = next((s for s in plan.specs if s.path == (name,)), None)
        keys.append(spec.alias if spec is not None and spec.alias else name)
    return tuple(keys)


def identity_values(data: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    """The record's identity values as text (else its first text values)."""
    values: list[str] = []
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(str(v) for v in value if isinstance(v, (str, int, float)))
        elif value is not None and not isinstance(value, (dict, bool)):
            values.append(str(value))
    if values:
        return values
    return [v for v in data.values() if isinstance(v, str)][:6]


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _TEXT_CHARS:
        return value[:_TEXT_CHARS] + "…"
    return value


def compact_record(
    data: Mapping[str, Any], identity: Sequence[str], render_fields: int
) -> dict[str, Any]:
    """A record cut down to what tells it apart.

    Keeps the identity fields, up to ``render_fields`` further non-null scalar
    fields, and the first 3 elements of every list; long texts are shortened.

    Args:
        data: The record (normalised keys).
        identity: Keys of the identity fields.
        render_fields: How many non-identity scalar fields to keep.

    Returns:
        A new dict in the record's own key order.
    """
    out: dict[str, Any] = {}
    kept = 0
    for key, value in data.items():
        if key in identity:
            out[key] = (
                [_clip(v) for v in value[:_LIST_ELEMENTS]]
                if isinstance(value, list)
                else _clip(value)
            )
        elif isinstance(value, list):
            out[key] = [_clip(v) for v in value[:_LIST_ELEMENTS]]
        elif value is not None and not isinstance(value, dict) and kept < render_fields:
            out[key] = _clip(value)
            kept += 1
    return out


def _size(value: Any) -> float:
    return estimate_tokens(value)


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #


def _option(index: int) -> str:
    return f"rec_{index}"


@dataclass
class _Report:
    """What the requests had to do to fit, for one warning per kind."""

    prefiltered: int = 0
    compacted: int = 0
    dropped: int = 0
    offered: list[int] = field(default_factory=list)


class _Matcher:
    """Asks the match question of each entity of one collection."""

    def __init__(
        self,
        run: Run,
        plan: SchemaPlan,
        text: str,
        noun: str,
        candidates: Sequence[Candidate],
        identity: Sequence[str],
        placement: Placement | None = None,
    ) -> None:
        self.run, self.plan, self.text, self.noun = run, plan, text, noun
        self.placement = placement or Placement()
        self.candidates = list(candidates)
        self.identity = tuple(identity)
        self.report = _Report()
        self._values = [identity_values(c.data, self.identity) for c in candidates]
        self._full = [dict(c.data) for c in candidates]
        self._small = [
            compact_record(c.data, self.identity, run.match.render_fields)
            for c in candidates
        ]

    def ctx(self, info: EntityInfo) -> Ctx:
        """The entity-anchored context a match state is built from."""
        return Ctx(
            run=self.run,
            plan=self.plan,
            text=self.text,
            label=info.label,
            key=info.key,
            mentions=tuple(dict.fromkeys(m.text for m in info.mentions)),
            info=info,
            parent=self.placement.parent,
            lineage=self.placement.lineage,
            nest=self.placement.nest,
            scope=self.placement.scope,
        )

    def _ranked(self, info: EntityInfo) -> list[int]:
        """Record indexes, the most similar to the entity first (ties: input order)."""
        mentions = [info.label, *(m.text for m in info.mentions)]
        scores = [similarity(values, mentions) for values in self._values]
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    def _question(self, options: Mapping[str, Any]) -> ChoiceQuestion:
        return ChoiceQuestion(
            type="choice",
            instructions="Which listed record is `entity.label`?",
            criteria=dict(options),
        )

    def _criteria(self, indexes: Sequence[int], rows: Sequence[Any]) -> dict[str, Any]:
        options: dict[str, Any] = {_option(i): rows[i] for i in sorted(indexes)}
        options[NEW] = f"A {self.plan.title} not among the listed records"
        return options

    async def ask(self, info: EntityInfo) -> MatchAnswer:
        """Ask which record an entity is (or ``__new__``)."""
        budget = self.run.limits.clef_state_token_budget
        ctx = self.ctx(info)
        keep = self._ranked(info)
        if len(keep) > MAX_OPTIONS:
            self.report.prefiltered = max(self.report.prefiltered, len(keep))
            keep = keep[:MAX_OPTIONS]
        state = ctx.fitted()
        instructions = "Which listed record is `entity.label`?"
        base = _size({"state": state, "instructions": instructions}) + _size(
            {NEW: f"A {self.plan.title} not among the listed records"}
        )
        rows: Sequence[Any] = self._full
        total = base + sum(_size({_option(i): self._full[i]}) for i in keep)
        if total > budget:
            rows = self._small
            self.report.compacted += 1
            sizes = [_size({_option(i): self._small[i]}) for i in keep]
            running, fitting = base, 0
            for size in sizes:  # keep is in rank order: the best fit first
                if running + size > budget:
                    break
                running += size
                fitting += 1
            if fitting == 0:  # one record is always offered
                running, fitting = base + sizes[0], 1
            if fitting < len(keep):
                self.report.dropped += 1
                keep = keep[:fitting]
            if running > budget:
                # not even one record fits beside the whole state: narrow the input
                room = budget - (running - base) - _size(instructions)
                state = ctx.fitted(budget=max(1_000, int(room)))
        self.report.offered.append(len(keep))
        answers = await self.run.call_clef(
            {"match": self._question(self._criteria(keep, rows))},
            state,
            meta=ctx.meta("match"),
        )
        answer = answers.get("match")
        if not isinstance(answer, ChoiceAnswer):
            return MatchAnswer()
        probs: dict[str, float] = {}
        for index in sorted(keep):
            probs[self.candidates[index].record_id] = answer.probabilities.get(
                _option(index), 0.0
            )
        probs[NEW] = answer.probabilities.get(NEW, 0.0)
        return MatchAnswer(probs, answer.confidence)

    def sole_fit(self, info: EntityInfo) -> int | None:
        """The index of the ONE record the entity is a strong name fit for, if any.

        ``None`` when no record fits, when two or more fit at all (``Emma`` with
        Emma Thompson and Emma Watson: a name is no evidence there), when the one
        that fits only does so as a prefix, a typo or an initial of a one-word
        mention (``strong_compatible``), or when a mention of two words or more
        contradicts the record ("Emma Davis" next to "Emma Thompson").
        """
        mentions = [info.label, *(m.text for m in info.mentions)]
        fits = [
            index
            for index, values in enumerate(self._values)
            if any(identity_compatible(m, v) for m in mentions for v in values)
        ]
        if len(fits) != 1:
            return None
        (only,) = fits
        # a full name that contradicts the record is another person ("Emma Davis"
        # next to "Emma Thompson"), whatever a short form of it would fit
        for text in mentions:
            if len(name_tokens(text)) >= 2 and not any(
                identity_compatible(text, v) for v in self._values[only]
            ):
                return None
        strong = any(
            strong_compatible(m, v) for m in mentions for v in self._values[only]
        )
        return only if strong else None

    async def confirm(self, info: EntityInfo, index: int, kind: str) -> float | None:
        """Ask whether the entity is the same kind of thing as one record.

        One Clef ``noul`` in a request of its own (call kind ``match_confirm``) over
        the input (windows around the entity's mentions), the entity and the record.
        """
        record = self._full[index]
        budget = self.run.limits.clef_state_token_budget
        if _size(record) > budget // 2:
            record = self._small[index]
        ctx = self.ctx(info)
        spans = [m.span for m in info.mentions if m.span is not None]
        windows = (
            mention_windows(len(self.text), spans, self.run.verify.window_chars)
            if spans
            else None
        )
        who = ", ".join(v for v in self._values[index] if v.strip()) or "no name"
        question = NoulQuestion(
            type="noul",
            instructions=(
                f"In `input`, is `entity.label` the same {kind} as the existing "
                f"record `record` ({who})?"
            ),
            criteria=NoulCriteria(
                true=f"Yes: `entity.label` and `record` are the same {kind}.",
                false=f"No: `entity.label` is not the {kind} that `record` describes, "
                f"but a different one.",
            ),
        )
        state = ctx.fitted(windows=windows, record=record)
        answers = await self.run.call_clef(
            {"confirm": question},
            state,
            meta=ctx.meta("match_confirm", field=self.candidates[index].record_id),
            decisive=True,
        )
        answer = answers.get("confirm")
        return answer.noul if isinstance(answer, NoulAnswer) else None

    def warnings(self) -> list[str]:
        """One warning per thing the requests had to do to fit."""
        report, name, total = self.report, self.plan.name, len(self.candidates)
        out: list[str] = []
        if report.prefiltered:
            out.append(
                f"match_prefiltered: {report.prefiltered} existing {name} records "
                f"are more than the {MAX_OPTIONS} options of one Clef choice; each "
                f"entity was matched against its {MAX_OPTIONS} most similar records "
                "by identity"
            )
        if report.compacted:
            out.append(
                f"match_compacted: the {total} existing {name} records did not fit "
                "one Clef request in full, so they were rendered compactly (the "
                "identity fields, a few other fields and the first elements of "
                "each list)"
                + (
                    f"; for {report.dropped} entity(ies) only the most similar "
                    f"records were offered (as few as {min(report.offered)})"
                    if report.dropped
                    else ""
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Matching a collection
# --------------------------------------------------------------------------- #

MergeCheck = Callable[[EntityInfo, EntityInfo], Awaitable[float | None]]


def _merged(group: Sequence[EntityInfo]) -> EntityInfo:
    """One entity made of several: their mentions in order, the earliest key."""
    mentions: list[Mention] = sorted(
        {m for e in group for m in e.mentions}, key=lambda m: (m.span or (0, 0), m.text)
    )
    first = min(
        group,
        key=lambda e: e.first_mention.span or (0, 0) if e.first_mention else (0, 0),
    )
    return EntityInfo(
        key=first.key,
        label=canonical_label([m.text for m in mentions]),
        mentions=tuple(mentions),
        first_mention=mentions[0] if mentions else first.first_mention,
        qualify_p=first.qualify_p,
        parent=first.parent,
    )


def _name(info: EntityInfo) -> str:
    return f"{info.label} ({info.key})"


def _collision_note(
    infos: Sequence[EntityInfo],
    table: Sequence[Mapping[str, float]],
    chosen: Sequence[str],
    rid: str,
    who: Sequence[int],
) -> str:
    """The ``match_collision`` warning for one record that several entities claim."""
    names = " and ".join(_name(infos[i]) for i in who)
    odds = ", ".join(f"{table[i].get(rid, 0.0):.2f}" for i in who)
    winners = [_name(infos[i]) for i in who if chosen[i] == rid]
    fell = [
        f"{_name(infos[i])} fell back to "
        + ("a new record" if chosen[i] == NEW else f"the record {chosen[i]!r}")
        for i in who
        if chosen[i] != rid
    ]
    return (
        f"match_collision: {names} both match the record {rid!r} (p={odds}); "
        f"one-to-one assignment gave it to {winners[0] if winners else 'neither'}"
        + (f", and {'; '.join(fell)}" if fell else "")
    )


def candidate_options(probs: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    shown = sorted(
        ((option, p) for option, p in probs.items() if p >= _NOTICEABLE),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(shown[:_SHOWN_CANDIDATES])


async def _merge_collisions(
    run: Run,
    infos: Sequence[EntityInfo],
    contested: Mapping[str, list[int]],
    merge_check: MergeCheck,
) -> list[EntityInfo] | None:
    """Merge the entities that claim one record, where they are one entity.

    Every pair of claimants of a record gets a merge check; two sets of claimants
    are joined, strongest check first, only if every pair across them passed.

    Returns:
        The entities after the merges, or ``None`` if nothing merged.
    """
    pairs = [
        (a, b) for who in contested.values() for a, b in itertools.combinations(who, 2)
    ]
    checked = await gather(*(merge_check(infos[a], infos[b]) for a, b in pairs))
    threshold = run.coref.merge_threshold
    checks = dict(zip(pairs, checked, strict=True))
    passed = sorted(
        (pair for pair, p in checks.items() if p is not None and p >= threshold),
        key=lambda pair: (-(checks[pair] or 0.0), pair),
    )
    sets: dict[int, list[int]] = {i: [i] for i in range(len(infos))}
    owner = {i: i for i in range(len(infos))}
    for a, b in passed:
        ra, rb = owner[a], owner[b]
        if ra == rb:
            continue
        across = [(min(x, y), max(x, y)) for x in sets[ra] for y in sets[rb]]
        if all(pair in checks and pair in passed for pair in across):
            for member in sets[rb]:
                owner[member] = ra
            sets[ra] = sorted(sets[ra] + sets.pop(rb))
    if len(sets) == len(infos):
        return None
    return [_merged([infos[i] for i in members]) for members in sets.values()]


def _compatible_top(
    config: MatchConfig,
    matcher: "_Matcher",
    ids: Sequence[str],
    info: EntityInfo,
    answer: MatchAnswer,
    pick: str,
) -> bool:
    """Whether a name-compatible record that is Clef's clear top choice is enough.

    Clef's choice ``confidence`` is not the probability of the chosen option (live:
    p 0.84 with confidence 0.47), and a record whose identity is a form of the
    entity's own name (``identity_compatible``) is the strongest evidence there is
    that it is the one. So when the pick is a record, its probability is at least
    ``match.accept_compatible`` and beats every other option (``__new__`` and the
    runner-up) by ``match.compatible_margin``, it is the ONLY record any of the
    entity's mentions fits, and the fit stands on its own (``strong_compatible``: a
    mention of two words or more, or one that is a word or diminutive of the
    record's identity; "Dan" is no more than the start of "Daniela"), the entity is
    matched without the ``match.min_confidence`` floor. Everything else (the
    assignment, the identity conflict guard) is unchanged.
    """
    if pick == NEW or not answer.probs or pick not in ids:
        return False
    here = answer.p(pick)
    rivals = [p for option, p in answer.probs.items() if option != pick]
    rivals.append(answer.p(NEW))
    if here < config.accept_compatible or here - max(rivals) < config.compatible_margin:
        return False  # not clearly above every other option
    mentions = [info.label, *(m.text for m in info.mentions)]
    fits = [
        index
        for index, values in enumerate(matcher._values)
        if any(
            identity_compatible(mention, value)
            for mention in mentions
            for value in values
        )
    ]
    if len(fits) != 1 or ids[fits[0]] != pick:
        return False
    return any(
        strong_compatible(mention, value)
        for mention in mentions
        for value in matcher._values[fits[0]]
    )


async def match_entities(
    run: Run,
    plan: SchemaPlan,
    text: str,
    noun: str,
    entities: Sequence[EntityInfo],
    candidates: Sequence[Candidate],
    identity: Sequence[str],
    merge_check: MergeCheck,
    placement: Placement | None = None,
    kind: str | None = None,
) -> MatchOutcome:
    """Match every entity to an existing record or to ``__new__``.

    Args:
        run: The invoke's context (``run.match`` configures the step).
        plan: The schema of the entities and records.
        text: The user's input.
        noun: What an entity is called in prompts.
        entities: The entities that qualify, in order of first mention.
        candidates: The existing records (normalised keys).
        identity: Keys of the identity fields, for the prefilter and for compact
            rendering.
        merge_check: Asks the coreference merge check for two entities (used with
            ``match.collision="merge"``).
        placement: For the items of a nested list, whom they belong to: the
            match state names the parent, so the candidates are the items of
            that parent only.
        kind: What kind of thing an entity is ("person"), for the confirmation
            question; defaults to ``noun``.

    Returns:
        The decisions.

    Raises:
        AmbiguousMatchError: With ``match.on_ambiguous="raise"``, for the first
            ambiguous entity.
    """
    if not entities:
        return MatchOutcome()
    config = run.match
    matcher = _Matcher(run, plan, text, noun, candidates, identity, placement)
    ids = [c.record_id for c in candidates]
    infos = list(entities)
    answers: dict[str, MatchAnswer] = {}
    while True:
        todo = [e for e in infos if e.key not in answers]
        for info, answer in zip(
            todo, await gather(*(matcher.ask(e) for e in todo)), strict=True
        ):
            answers[info.key] = answer
        table = [answers[e.key].probs for e in infos]
        chosen = assign(table, ids, prune=_PRUNE)
        contested = collisions(table)
        if contested and config.collision == "merge":
            merged = await _merge_collisions(run, infos, contested, merge_check)
            if merged is not None:
                before = {e.key: e for e in infos}
                for info in merged:  # a merged entity must be asked again
                    if before.get(info.key) != info:
                        answers.pop(info.key, None)
                        run.warn(
                            f"match_merged: {info.label} ({info.key}) was found as "
                            "two entities that both matched one record, and the "
                            "merge check says they are one; they were merged and "
                            "matched again"
                        )
                infos = merged
                continue
        for rid, who in sorted(contested.items()):
            run.warn(_collision_note(infos, table, chosen, rid, who))
        break
    for message in matcher.warnings():
        run.warn(message)

    out: list[EntityInfo] = []
    targets: dict[str, str | None] = {}
    skipped: list[SkippedEntity] = []
    sure_ones: list[bool] = []
    for info, pick in zip(infos, chosen, strict=True):
        answer = answers[info.key]
        p = answer.p(pick)
        sure_ones.append(
            (
                bool(answer.probs)
                and p >= config.accept
                and answer.confidence is not None
                and answer.confidence >= config.min_confidence
            )
            or _compatible_top(config, matcher, ids, info, answer, pick)
        )
    confirmed = await _confirmations(
        run, matcher, infos, chosen, sure_ones, ids, answers, kind or noun
    )
    for info, pick, sure in zip(infos, chosen, sure_ones, strict=True):
        answer = answers[info.key]
        info = replace(info, match_probs=dict(answer.probs))
        out.append(info)
        p = answer.p(pick)
        if info.key in confirmed:
            targets[info.key] = confirmed[info.key]
            continue
        if sure:
            targets[info.key] = None if pick == NEW else pick
            continue
        candidates_ = candidate_options(answer.probs)
        detail = f"{info.label!r} could not be matched with enough certainty: " + (
            f"the best option {pick!r} has p={p:.2f} (accept at "
            f"{config.accept:g}) and the answer's confidence is "
            f"{answer.confidence:.2f} (needs {config.min_confidence:g})"
            if answer.confidence is not None
            else "the model gave no answer"
        )
        if config.on_ambiguous == "raise":
            raise AmbiguousMatchError(
                f"{plan.name} entity {info.label!r}: {detail}",
                schema_name=plan.name,
                label=info.label,
                candidates=candidates_,
            )
        if config.on_ambiguous == "insert" and answer.p(NEW) >= config.accept:
            targets[info.key] = None  # new is a likely reading: insert it
            continue
        skipped.append(
            SkippedEntity(plan.name, info, "ambiguous_match", detail, candidates_)
        )
    return MatchOutcome(tuple(out), targets, tuple(skipped), frozenset(confirmed))


async def _confirmations(
    run: Run,
    matcher: _Matcher,
    infos: Sequence[EntityInfo],
    chosen: Sequence[str],
    sure_ones: Sequence[bool],
    ids: Sequence[str],
    answers: Mapping[str, MatchAnswer],
    kind: str,
) -> dict[str, str]:
    """Confirm, pairwise, the entities the choice question did not settle.

    Clef's record choice answers ``__new__`` at 0.83 for "Emma" against the record
    "Emma Thompson" while the record-presence check agrees with the record at 0.87,
    and gives the README's Emma only 0.506. An entity is settled when the choice
    picks a record and is sure (``accept``, ``min_confidence``), or picks ``__new__``
    at ``match.new_final`` (0.9) or more: by that probability alone, since the
    choice's confidence is calibrated differently by every provider. Otherwise, when
    EXACTLY ONE record is a strong name fit for the entity (``_Matcher.sole_fit``),
    one Clef ``noul`` in its own request asks whether the entity is the same kind of
    thing as that record. At ``match.confirm_accept``, and above the choice's own
    probability of ``__new__``, the entity is matched to the record; otherwise the
    outcome stays as it was. A record already assigned to another entity, or wanted
    by a more likely one, is not taken.

    Returns:
        The record id by entity key, for the entities that were confirmed.
    """
    floor = run.match.confirm_accept
    if floor <= 0:
        return {}
    taken = {
        pick
        for pick, sure in zip(chosen, sure_ones, strict=True)
        if sure and pick != NEW
    }
    wanted: list[tuple[EntityInfo, int]] = []
    for info, pick, sure in zip(infos, chosen, sure_ones, strict=True):
        if pick == NEW:
            # a __new__ is final from `new_final`, by its probability alone: the
            # choice's confidence means something else on every provider
            if answers[info.key].p(NEW) >= run.match.new_final:
                continue
        elif sure:
            continue  # a sure record is final
        index = matcher.sole_fit(info)
        if index is not None and ids[index] not in taken:
            wanted.append((info, index))
    if not wanted:
        return {}
    asked = await gather(
        *(matcher.confirm(info, index, kind) for info, index in wanted)
    )
    confirmed: dict[str, str] = {}
    ranked = sorted(
        (
            (p, info, index)
            for (info, index), p in zip(wanted, asked, strict=True)
            if p is not None and p >= floor and p > answers[info.key].p(NEW)
        ),
        key=lambda item: -item[0],
    )
    picked = {info.key: pick for info, pick in zip(infos, chosen, strict=True)}
    for p, info, index in ranked:
        if ids[index] in taken:
            continue
        taken.add(ids[index])
        confirmed[info.key] = ids[index]
        was = picked[info.key]
        run.warn(
            f"match_confirmed: {info.label} ({info.key}) is the record "
            f"{ids[index]!r} on a pairwise check (p={p:.2f}); the choice gave "
            f"{was!r} (p={answers[info.key].p(was):.2f})"
        )
    return confirmed


# --------------------------------------------------------------------------- #
# Deletes
# --------------------------------------------------------------------------- #


async def flag_deletes(
    run: Run,
    plan: SchemaPlan,
    text: str,
    candidates: Sequence[Candidate],
    identity: Sequence[str],
    placement: Placement | None = None,
) -> dict[str, float | None]:
    """Ask, per existing record, whether the input says it should be deleted.

    The records go into requests of up to ``run.clef_batch()`` (64, or 32 on
    Experiential Labs), in order, one Clef ``noul`` each (call
    kind ``delete``): the state is ``{input, records: {r0: <record>, r1: ...}}`` and
    the question ``delete.r<k>`` names ``records.r<k>`` (``k`` is the record's place
    in ``candidates``). A request that would be over the Clef state budget takes
    fewer records, and a record that does not fit beside the input alone is rendered
    compactly (``match_compacted``). ``CallMeta.entity`` is not set;
    ``CallMeta.field`` holds the ids of the request's records, comma-joined, in
    question order.

    Args:
        run: The invoke's context.
        plan: The schema of the records.
        text: The user's input.
        candidates: The existing records (normalised keys).
        identity: Keys of the identity fields (kept when compacting).
        placement: For the items of a nested list, whom they belong to: the state
            carries the ``parent`` and the question asks about removing the item
            from it.

    Returns:
        The probability by record id (``None`` for a missing answer).
    """
    budget = run.limits.clef_state_token_budget
    delete_batch = run.clef_batch()
    placed = placement or Placement()
    parent_ctx = Ctx(
        run=run, plan=plan, text=text, label="", parent=placed.parent, nest=placed.nest
    )
    parent = parent_ctx.parent_dict()
    shared: dict[str, Any] = {"records": {}}
    if parent is not None:
        shared["parent"] = parent
    base = estimate_tokens(run.clef_state(text, extra=shared))
    batches: list[dict[int, dict[str, Any]]] = []
    used = base
    compacted = 0
    for index, candidate in enumerate(candidates):
        record: dict[str, Any] = dict(candidate.data)
        size = _size({f"r{index}": record})
        if base + size > budget:
            record = compact_record(candidate.data, identity, run.match.render_fields)
            size = _size({f"r{index}": record})
            compacted += 1
        if (
            not batches
            or len(batches[-1]) >= delete_batch
            or (used + size > budget and batches[-1])
        ):
            batches.append({})
            used = base
        batches[-1][index] = record
        used += size
    if compacted:
        run.warn(
            f"match_compacted: {compacted} existing {plan.name} record(s) did not "
            "fit one Clef request in full, so the delete question saw them "
            "rendered compactly"
        )
    question_for = {
        index: NoulQuestion(
            type="noul",
            instructions=(
                f"Does `input` state that `records.r{index}` should be deleted"
                + (" from `parent.label`" if parent is not None else "")
                + " \u2014 e.g. the person/item was removed, has died, or should no "
                "longer be tracked? A change to its details is NOT deletion."
            ),
            criteria=NoulCriteria(
                true="The input says the record should be deleted.",
                false="The input does not say that about the record: at most its "
                "details change.",
            ),
        )
        for index in range(len(candidates))
    }

    async def ask(batch: dict[int, dict[str, Any]]) -> dict[str, float | None]:
        extra: dict[str, Any] = {"records": {f"r{i}": r for i, r in batch.items()}}
        if parent is not None:
            extra["parent"] = parent
        state = run.clef_state(text, extra=extra, spans=parent_ctx.parent_spans())
        if parent is not None and state["input"] != text:
            state["parent"] = parent_ctx.parent_dict(spans=False)
        ids = [candidates[i].record_id for i in batch]
        answers = await run.call_clef(
            {f"delete.r{i}": question_for[i] for i in batch},
            state,
            meta=CallMeta(
                kind="delete",
                schema=plan.name,
                field=",".join(ids),
                scope=placed.scope,
            ),
        )
        found: dict[str, float | None] = {}
        for i in batch:
            answer = answers.get(f"delete.r{i}")
            found[candidates[i].record_id] = (
                answer.noul if isinstance(answer, NoulAnswer) else None
            )
        return found

    parts = await gather(*(ask(batch) for batch in batches))
    merged: dict[str, float | None] = {}
    for part in parts:
        merged.update(part)
    return {c.record_id: merged.get(c.record_id) for c in candidates}
