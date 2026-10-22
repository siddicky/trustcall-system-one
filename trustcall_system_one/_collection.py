"""The collection engine: every entity of a schema that the input mentions.

``collect`` finds the entities (steps 1-5 of "Collection engine" in
``docs/DESIGN.md``) and ``insert_collection`` extracts each of them with the
single-entity insert pipeline (step 6) concurrently:

1. the *identity* of an entity: ``Many(identity=...)``, else the schema's own
   (``SchemaPlan.identity``), else none (discovery is then driven by the schema's
   description and a warning says so);
2. *discovery*, three requests in parallel: a presence question (Clef), the list
   of mentions (Clairic) and the count of distinct entities (Clairic);
3. *coreference* (``_coref``): the mentions are clustered into entities;
4. *qualification*: one Clef question per entity, "is this a <Schema>?", which
   filters out what merely looks like one (a dog called Sunny is not a Person);
5. *count reconciliation*: the entities found are compared with the count; a
   count above them triggers one broadened discovery, a count below them
   cluster-level merge checks, and anything that stays inconsistent is a warning
   (an entity is never invented).

A ``Scope`` says what is collected from where, so the same engine can serve a
nested list later (``parent``, ``pointer`` and ``existing`` are the parts that
differ from a top-level ``Many``); at the top level ``parent`` is ``None``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from trustcall_system_one._cardinality import Many
from trustcall_system_one._compile import SchemaPlan
from trustcall_system_one._coref import (
    Clustering,
    PairCache,
    canonical_label,
    cluster_mentions,
    has_context,
    mention_entry,
    ordered_pair,
    plural,
)
from trustcall_system_one._entity import InsertOutcome, insert_one
from trustcall_system_one._render import RenderedState, discovery_focus, entity_focus
from trustcall_system_one._run import CLEF_BATCH, Run, gather
from trustcall_system_one._types import (
    EntityInfo,
    Mention,
    ResultRecord,
    SkippedEntity,
)
from trustcall_system_one._verify import Ctx, ask_clairic
from trustcall_system_one._wire import (
    ClairicAnswer,
    ClairicField,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
)
from trustcall_system_one.transport import CallMeta

__all__ = [
    "CollectionOutcome",
    "CollectionResult",
    "Scope",
    "clean_mention",
    "collect",
    "identity_fields",
    "insert_collection",
    "parse_mentions",
]

Span = tuple[int, int]

QUALIFY_MIN = 0.5
"""Qualification probability below which an entity is not a ``<Schema>``."""
MERGE_CANDIDATE_MIN = 0.2
"""Best pairwise probability from which two clusters are checked for a merge."""

PRONOUNS = frozenset(
    {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "that", "this", "these", "those",
        "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
        "nobody", "noone", "one", "ones", "each", "both", "either", "neither",
        "another", "other", "others", "all", "none", "there", "here",
    }
)  # fmt: skip
"""English closed-class words that are never a mention of an entity."""

_NOT_NAMES = PRONOUNS | {"the", "a", "an"}
_CONNECTORS = frozenset(
    {
        "and",
        "or",
        "nor",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
    }
    | {"vs", "versus", "et", "und", "y"}
)
"""Words that join two names into a list or a phrase, never part of one name."""
_DETERMINER = re.compile(r"^(?:the|an|a|this|that|these|those)\s+", re.IGNORECASE)
_POSSESSIVE = re.compile("['\u2019]s$", re.IGNORECASE)
_EDGE = " \t\r\n\"'\u201c\u201d\u2018\u2019`,;:.!?()[]{}"


# --------------------------------------------------------------------------- #
# Scope and outcome
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scope:
    """What a collection is collected from, and for.

    Attributes:
        plan: The schema of the entities.
        source: The text the entities are found in.
        cardinality: The ``Many`` that asked for them (its ``identity`` and
            ``label``).
        parent: The entity the collection belongs to; ``None`` at the top level.
        pointer: JSON Pointer of the collection inside its parent's document
            (empty at the top level).
        existing: Existing items to match the entities against (records at the
            top level, list elements for a nested list). Matching arrives with
            M8; this milestone only inserts, so it must be empty.
    """

    plan: SchemaPlan
    source: str
    cardinality: Many
    parent: EntityInfo | None = None
    pointer: str = ""
    existing: tuple[Any, ...] = ()

    @property
    def noun(self) -> str:
        """What one entity is called in prompts."""
        return self.cardinality.label or self.plan.title


@dataclass(frozen=True)
class CollectionOutcome:
    """The entities a collection found.

    Attributes:
        entities: The entities that qualify, in order of first mention.
        skipped: The entities that were found and do not qualify
            (``SkippedEntity(reason="not_qualified")``).
        warnings: Notes about anything degraded (also added to the run's).
        presence: The presence probability, when it was asked.
        count: The count of entities Clairic gave, when it gave one.
    """

    entities: tuple[EntityInfo, ...] = ()
    skipped: tuple[SkippedEntity, ...] = ()
    warnings: tuple[str, ...] = ()
    presence: float | None = None
    count: int | None = None


@dataclass(frozen=True)
class CollectionResult:
    """What ``insert_collection`` produced.

    Attributes:
        records: One insert record per qualifying entity, in order of first
            mention (an invalid record is among them, with its skip).
        skipped: Entities not turned into a valid record, in order of first
            mention.
        extracted: Whether any extraction request was made.
        attempts: Extraction rounds of the entity that needed the most.
        outcome: The discovery outcome.
    """

    records: tuple[ResultRecord, ...] = ()
    skipped: tuple[SkippedEntity, ...] = ()
    extracted: bool = False
    attempts: int = 0
    outcome: CollectionOutcome = field(default_factory=CollectionOutcome)


# --------------------------------------------------------------------------- #
# Mentions
# --------------------------------------------------------------------------- #


def _capitalised(word: str) -> bool:
    return bool(word) and (word[0].isupper() or word[0].isdigit())


def _only_function_words(words: Sequence[str]) -> bool:
    """Whether every word is a pronoun or article, so the mention names nobody.

    An all-capitals word of two letters or more ("US", "IT") is an acronym, never a
    pronoun; and several capitalised words ("The Who") are a proper name even when
    each one is a function word.
    """
    if len(words) > 1 and all(_capitalised(w) for w in words):
        return False
    return all(
        not (w.isupper() and len(w) >= 2)
        and w.strip("'\u2019").casefold() in _NOT_NAMES
        for w in words
    )


def clean_mention(text: str, span: Span | None = None) -> Mention | None:
    """Make a discovered mention a clean one, or drop it.

    Surrounding punctuation, a trailing possessive (``Emma's``) and a leading
    lower-case determiner (``the``, ``a``, ``this`` ...) are stripped, the span
    shrinking with the text. A capitalised determiner is part of a proper name
    when what follows is capitalised too ("The Hague", "An Nguyen", "This Is Us")
    and stays; it goes when what follows is a common phrase ("The cousin"). A
    mention that is only pronouns ("she", "them", "someone") is dropped, but an
    acronym ("US", "IT") and a name made of function words ("The Who") are not. A
    possessive pronoun that makes a descriptive reference ("my cousin") is not a
    determiner here and stays.

    Args:
        text: The mention as Clairic returned it. When ``span`` is given the
            text is what the input has there.
        span: Its ``(start, end)`` in the original input.

    Returns:
        The cleaned mention with its span moved to match, or ``None``.
    """
    lo, hi = 0, len(text)

    def current() -> str:
        return text[lo:hi]

    def trim() -> None:
        nonlocal lo, hi
        while lo < hi and text[lo] in _EDGE:
            lo += 1
        while lo < hi and text[hi - 1] in _EDGE:
            hi -= 1

    trim()
    cut = _POSSESSIVE.search(current())
    if cut is not None and cut.start() > 0:
        hi = lo + cut.start()
        trim()
    lead = _DETERMINER.match(current())
    if lead is not None and lead.end() < len(current()):
        rest = current()[lead.end() :].split()
        name = lead.group(0)[0].isupper() and all(_capitalised(w) for w in rest)
        if not name:
            lo += lead.end()
            trim()
    kept = current()
    words = [w for w in re.split(r"[\s-]+", kept) if w]
    if not words or _only_function_words(words):
        return None
    if span is None:
        return Mention(kept)
    return Mention(kept, (span[0] + lo, span[0] + hi))


def _overlap(first: Span, second: Span) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _name_gap(gap: str, *, words: bool) -> bool:
    """Whether the text between two parts of a name is name material.

    Spaces, hyphens inside a hyphenated name, initials ("J.") and, when ``words``
    allows it, capitalised words (a middle name) are. A separator (``, ; : & /``, a
    line break, the word "and"), a lower-case word and a possessive ("'s brother")
    are not: they set one mention apart from the next, or make a phrase.
    """
    if re.search(r"[,;:&/\n\r]", gap):
        return False
    for token in re.split(r"[ \t\u00a0\-]+", gap):
        if not token or (len(token) <= 2 and token[0].isupper() and token[1:] in "."):
            continue
        plain = token.replace("'", "").replace("\u2019", "")
        if not (words and token[0].isupper() and plain.isalnum()):
            return False
        if token.casefold() in _NOT_NAMES or token.casefold() in _CONNECTORS:
            return False
    return True


def _one_name(source: str, whole: Span, parts: Sequence[Span]) -> bool:
    """Whether ``whole`` is ONE name that ``parts`` (inside it) merely quote.

    Only when at most one part is itself a multi-token name (two make a list of
    people: "Emma Thompson Michael Chen"), and the text between the parts, and
    before and after them, is only spaces, hyphens and initials. Capitalised words
    ("Mary Ann Smith" over "Mary" and "Smith") also count when every part is a
    single token. Separators and lower-case or possessive words ("'s brother")
    make a phrase that bridges separate mentions instead.
    """
    single = [len(source[start:end].split()) == 1 for start, end in parts]
    if single.count(False) > 1:
        return False
    words = all(single)
    cursor = whole[0]
    for start, end in sorted(parts):
        if not _name_gap(source[cursor:start], words=words):
            return False
        cursor = end
    return _name_gap(source[cursor : whole[1]], words=words)


def _resolve_overlaps(mentions: Sequence[Mention], source: str) -> list[Mention]:
    """Keep one reading where mentions overlap, never merging two mentions into one.

    Shorter mentions are placed first. A longer one replaces the mentions it
    contains when that is one mention ("Emma Thompson" for "Emma") or when it is one
    name that several of them quote ("Emma Thompson" over "Emma" and "Thompson",
    see ``_one_name``); a span that bridges separate mentions ("Emma's brother
    Tom" over "Emma" and "Tom") replaces neither, and is dropped, as is a mention
    inside a longer one, a partial overlap or a repeat of a span.
    """
    kept: list[Mention] = []
    for mention in sorted(
        mentions, key=lambda m: (m.span[1] - m.span[0]) if m.span else 0
    ):
        if mention.span is None:
            kept.append(mention)
            continue
        new = mention.span
        clashes = [k for k in kept if k.span and _overlap(k.span, new)]
        if not clashes:
            kept.append(mention)
            continue
        spans = [k.span for k in clashes if k.span]
        inside = all(new[0] <= a and b <= new[1] for a, b in spans)
        if not inside or spans == [new]:
            continue
        if len(clashes) == 1 or _one_name(source, new, spans):
            kept = [k for k in kept if k not in clashes]
            kept.append(mention)
    return sorted(kept, key=lambda m: m.span or (0, 0))


def _span_of(
    answer: ClairicAnswer, index: int, anchored: RenderedState | None
) -> Span | None:
    spans = answer.spans
    if spans is None or index >= len(spans) or len(spans[index]) != 2:
        return None
    if anchored is None:
        return None
    return anchored.to_original((spans[index][0], spans[index][1]))


def _find_unused(source: str, text: str, cursor: int, used: set[Span]) -> int:
    """The first occurrence of ``text`` after ``cursor`` that no item has, else any."""
    for start in (cursor, 0):
        at = source.find(text, start)
        while at >= 0:
            if (at, at + len(text)) not in used:
                return at
            at = source.find(text, at + 1)
    return -1


def parse_mentions(
    source: str,
    answer: ClairicAnswer | None,
    anchored: RenderedState | None,
    run: Run | None = None,
) -> list[Mention]:
    """The mentions in a discovery answer, as clean mentions in input order.

    Each item is located in the ORIGINAL input: by the span Clairic gave, mapped
    out of the rendered state, or, failing that, by searching for its text after
    the previous mention. An item that is not in the input is dropped (Clairic is
    extractive, so it can only be a slip). Pronoun-only mentions are dropped and
    determiners and possessives stripped (``clean_mention``). Of two overlapping
    mentions the longer is kept only if it contains exactly one of them (a span
    that bridges two mentions replaces neither), and two mentions never share a
    span.

    Args:
        source: The original input.
        answer: Clairic's ``list`` answer.
        anchored: The state to map the answer's spans with.
        run: The invoke's context, to warn about items that could not be placed.

    Returns:
        The mentions, ordered by position.
    """
    if answer is None or not isinstance(answer.value, list):
        return []
    found: list[Mention] = []
    used: set[Span] = {
        (pair[0], pair[1])
        for index in range(len(answer.value))
        if (pair := _span_of(answer, index, anchored)) is not None
    }
    cursor = 0
    for index, item in enumerate(answer.value):
        text = str(item)
        if not text.strip():
            continue
        span = _span_of(answer, index, anchored)
        if span is not None and source[span[0] : span[1]].strip() != text.strip():
            lo, hi = max(0, span[0] - 24), min(len(source), span[1] + 24)
            near = source.find(text, lo, hi)
            span = None if near < 0 else (near, near + len(text))
        if span is None:
            at = _find_unused(source, text, cursor, used)
            if at < 0:
                if run is not None:
                    run.warn(
                        f"mention_not_in_input: discovery returned {text!r}, which "
                        "the input does not contain; it was ignored"
                    )
                continue
            span = (at, at + len(text))
        used.add(span)
        cleaned = clean_mention(source[span[0] : span[1]], span)
        if cleaned is None or cleaned.span is None:
            continue
        found.append(cleaned)
        cursor = span[1]
    return _resolve_overlaps(found, source)


# --------------------------------------------------------------------------- #
# Step 1: identity
# --------------------------------------------------------------------------- #


def _humane(name: str) -> str:
    return name.replace("_", " ").strip()


def identity_fields(plan: SchemaPlan, many: Many) -> tuple[str, ...]:
    """The fields that tell entities of a schema apart.

    ``Many(identity=...)`` wins; else the schema's own identity (a field marked
    ``s1_field(identity=True)``, a ``str`` field named like an identity, the first
    required ``str``); else none.

    Args:
        plan: The compiled schema.
        many: The cardinality wrapper.

    Returns:
        The field names, possibly empty.

    Raises:
        ValueError: If ``Many(identity=...)`` names a field the schema lacks.
    """
    if many.identity is None:
        return plan.identity
    names = (many.identity,) if isinstance(many.identity, str) else tuple(many.identity)
    known = {str(spec.path[0]) for spec in plan.specs if spec.path}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"Many({plan.name}, identity=...) names {unknown}, which "
            f"{plan.name} has no field of; its fields are {sorted(known)}"
        )
    return names


def _identity_phrase(plan: SchemaPlan, names: Sequence[str]) -> str:
    """How the identity reads in a question: ``name (full name)``."""
    parts: list[str] = []
    for name in names:
        spec = next((s for s in plan.specs if s.path == (name,)), None)
        text = _humane(name)
        if spec is not None and spec.description:
            said = " ".join(spec.description.split()).rstrip(" .")
            if said and said.casefold() != text.casefold() and len(said) <= 80:
                text += f" ({said})"
        parts.append(text)
    return " and ".join(parts)


def _described(plan: SchemaPlan) -> str:
    return " ".join(plan.description.split()).rstrip(" .!?:;")


# --------------------------------------------------------------------------- #
# Step 2: presence, discovery, count
# --------------------------------------------------------------------------- #

_BROADER = (
    " Also include descriptive references such as 'my cousin' when no name is given."
)


async def _presence(run: Run, scope: Scope) -> tuple[float | None, bool]:
    """Ask Clef whether the input mentions any entity.

    Returns:
        The probability, and whether the state was cut to fit the Clef budget (the
        question has nothing to anchor on, so a cut state is read from the start
        of the input and its answer is not to be trusted).
    """
    plan = scope.plan
    described = _described(plan)
    question = NoulQuestion(
        type="noul",
        instructions=(
            f"Does `input` mention any {scope.noun}"
            + (f" ({described})" if described else "")
            + "?"
        ),
        criteria=NoulCriteria(
            true=f"The input mentions at least one {scope.noun}.",
            false=f"The input mentions no {scope.noun}.",
        ),
    )
    state = run.clef_state(scope.source)
    answers = await run.call_clef(
        {"presence": question},
        state,
        meta=CallMeta(kind="presence", schema=plan.name),
    )
    answer = answers.get("presence")
    cut = state["input"] != scope.source
    return (answer.noul if isinstance(answer, NoulAnswer) else None), cut


def _discovery_ctx(run: Run, scope: Scope) -> Ctx:
    return Ctx(
        run=run,
        plan=scope.plan,
        text=scope.source,
        label=scope.noun,
        key="",
        focus=discovery_focus(scope.plan.name, scope.noun),
    )


async def _discover(
    run: Run, scope: Scope, phrase: str, *, broaden: bool = False
) -> list[Mention]:
    """Ask Clairic for every mention of an entity, and clean what comes back."""
    noun = scope.noun
    if phrase:
        what = f"Every mention of a {noun} by their {phrase}"
    else:
        described = _described(scope.plan)
        what = f"Every mention of a {noun}" + (f" ({described})" if described else "")
    instructions = (
        f"{what}, in the order they appear in the input. Include repeated "
        "mentions and shortened ones (a first name alone). Do not list pronouns. "
        "Each item is the exact words the input uses."
    ) + (_BROADER if broaden else "")
    fields = {
        "mentions": ClairicField(type="list", items="text", instructions=instructions)
    }
    ctx = _discovery_ctx(run, scope)
    asked = await ask_clairic(
        ctx,
        fields,
        meta=CallMeta(kind="discover", schema=scope.plan.name),
    )
    answer, anchored = asked["mentions"]
    return parse_mentions(scope.source, answer, anchored, run)


async def _count(run: Run, scope: Scope) -> int | None:
    """Ask Clairic how many distinct entities the input mentions."""
    noun = scope.noun
    field_ = ClairicField(
        type="number",
        integer=True,
        instructions=(
            f"How many distinct {plural(noun)} does the input mention? Count each "
            f"individual {noun} once, however many times they are mentioned."
        ),
    )
    ctx = _discovery_ctx(run, scope)
    response = await run.call_clairic(
        {"count": field_},
        ctx.rendered(),
        meta=CallMeta(kind="count", schema=scope.plan.name),
    )
    answer = response.answers.get("count")
    if response.truncated:
        run.warn(
            f"count_unreliable: the input of {scope.plan.name} did not fit one "
            "Clairic request, so the count of entities was ignored"
        )
        return None
    if answer is None or answer.value is None or isinstance(answer.value, bool):
        return None
    try:
        count = round(float(answer.value))
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


# --------------------------------------------------------------------------- #
# Steps 3-5: clusters, qualification, reconciliation
# --------------------------------------------------------------------------- #


@dataclass
class _Group:
    """A cluster of mentions that is one entity (before it has a key)."""

    mentions: tuple[Mention, ...]
    qualify_p: float | None = None

    @property
    def label(self) -> str:
        return canonical_label([m.text for m in self.mentions])

    @property
    def first(self) -> Span:
        return self.mentions[0].span or (0, 0)

    @property
    def signature(self) -> frozenset[Mention]:
        return frozenset(self.mentions)


def _provisional(index: int) -> str:
    return f"e{index}"


async def _qualify(
    run: Run, scope: Scope, items: Sequence[tuple[int, _Group]]
) -> list[float | None]:
    """Ask, in requests of at most 64 entities, whether each is really a ``<Schema>``.

    A request has the state ``{"input", "entities": {"e0": {"label", "mentions"},
    ...}}`` with the entities it asks about, and each is a question
    ``qualify.e<n>`` (``n`` is the entity's place in the list being resolved); the
    call has no entity of its own. A mention carries its span, or, where the input
    had to be narrowed and the span would point at nothing, its text and a short
    context.
    """
    if not items:
        return []
    plan = scope.plan
    described = _described(plan)
    context = run.coref.context_chars

    def entity(group: _Group, *, spans: bool) -> dict[str, Any]:
        total = len(group.mentions)
        mentions: list[dict[str, Any]] = []
        for index, m in enumerate(group.mentions):
            entry: dict[str, Any] = {"text": m.text}
            if m.span is not None:
                if spans:
                    entry["span"] = list(m.span)
                elif has_context(index, total):
                    start, end = m.span
                    entry["context"] = scope.source[
                        max(0, start - context) : end + context
                    ]
            mentions.append(entry)
        return {"label": group.label, "mentions": mentions}

    async def ask(chunk: Sequence[tuple[int, _Group]]) -> dict[str, float | None]:
        anchors = [m.span for _, g in chunk for m in g.mentions if m.span is not None]
        state = run.clef_state(
            scope.source,
            spans=anchors,
            extra={
                "entities": {_provisional(n): entity(g, spans=True) for n, g in chunk}
            },
        )
        if state["input"] != scope.source:  # offsets no longer point into the input
            state["entities"] = {
                _provisional(n): entity(g, spans=False) for n, g in chunk
            }
        questions = {
            f"qualify.{_provisional(n)}": NoulQuestion(
                type="noul",
                instructions=(
                    f"Is `entities.{_provisional(n)}.label` a {scope.noun}"
                    + (f" ({described})" if described else "")
                    + "?"
                ),
                criteria=NoulCriteria(
                    true=f"The entity is a {scope.noun}.",
                    false=f"The entity is something else, not a {scope.noun}.",
                ),
            )
            for n, _ in chunk
        }
        answers = await run.call_clef(
            questions, state, meta=CallMeta(kind="qualify", schema=plan.name)
        )
        found: dict[str, float | None] = {}
        for n, _ in chunk:
            answer = answers.get(f"qualify.{_provisional(n)}")
            found[_provisional(n)] = (
                answer.noul if isinstance(answer, NoulAnswer) else None
            )
        return found

    chunks = [items[k : k + CLEF_BATCH] for k in range(0, len(items), CLEF_BATCH)]
    parts = await gather(*(ask(chunk) for chunk in chunks))
    merged = {key: p for part in parts for key, p in part.items()}
    return [merged[_provisional(n)] for n, _ in items]


async def _group(
    run: Run,
    scope: Scope,
    mentions: Sequence[Mention],
    pair_cache: PairCache,
    known: dict[frozenset[Mention], float | None],
) -> tuple[list[_Group], Clustering]:
    """Cluster the mentions and qualify every cluster that is new."""
    clustering = await cluster_mentions(
        run,
        source=scope.source,
        mentions=mentions,
        noun=scope.noun,
        schema=scope.plan.name,
        cache=pair_cache,
    )
    groups = [
        _Group(tuple(mentions[i] for i in cluster)) for cluster in clustering.clusters
    ]
    fresh = [(n, g) for n, g in enumerate(groups) if g.signature not in known]
    for (_, g), p in zip(fresh, await _qualify(run, scope, fresh), strict=True):
        known[g.signature] = p
    for g in groups:
        g.qualify_p = known[g.signature]
    return groups, clustering


def _qualified(groups: Sequence[_Group]) -> list[_Group]:
    return [g for g in groups if g.qualify_p is None or g.qualify_p >= QUALIFY_MIN]


def _best_pair(a: _Group, b: _Group, cache: PairCache) -> float | None:
    """The highest probability asked for a mention of ``a`` and one of ``b``.

    ``None`` when no such pair was asked at all: that is not a "no".
    """
    asked = [
        p
        for x in a.mentions
        for y in b.mentions
        if (p := cache.get(ordered_pair(x, y))) is not None
    ]
    return max(asked) if asked else None


async def _merge_check(run: Run, scope: Scope, a: _Group, b: _Group) -> float | None:
    question = NoulQuestion(
        type="noul",
        instructions=f"Do `a` and `b` refer to the same {scope.noun}?",
        criteria=NoulCriteria(
            true=f"Both refer to one and the same {scope.noun}.",
            false=f"They are two different {plural(scope.noun)}.",
        ),
    )

    def side(group: _Group) -> dict[str, Any]:
        return {
            "label": group.label,
            "mentions": [
                mention_entry(scope.source, m, run.coref.context_chars)
                for m in group.mentions
            ],
        }

    state = run.clef_state(
        scope.source,
        spans=[m.span for g in (a, b) for m in g.mentions if m.span],
        extra={"a": side(a), "b": side(b)},
    )
    answers = await run.call_clef(
        {"merge": question}, state, meta=CallMeta(kind="coref", schema=scope.plan.name)
    )
    answer = answers.get("merge")
    return answer.noul if isinstance(answer, NoulAnswer) else None


async def _merge(
    run: Run,
    scope: Scope,
    groups: list[_Group],
    count: int,
    cache: PairCache,
) -> tuple[list[_Group], int]:
    """Merge qualifying clusters that are one entity, until the count fits.

    Every pair of clusters whose best pairwise probability is at least 0.2, or
    that no pairwise question was ever asked about (a nickname and its name two
    clusters apart), gets a merge check. Two sets of clusters are joined,
    strongest check first, only if EVERY pair across them was checked and passed:
    A~B and B~C never make A and C one entity by themselves (the missing check
    is asked for first).

    Returns:
        The clusters after the merges and how many merge checks were asked.
    """
    live = _qualified(groups)
    threshold = run.coref.merge_threshold
    checks: dict[tuple[int, int], float | None] = {}

    async def check(pairs: Sequence[tuple[int, int]]) -> None:
        answers = await gather(
            *(_merge_check(run, scope, live[a], live[b]) for a, b in pairs)
        )
        checks.update(zip(pairs, answers, strict=True))

    candidates = [
        (a, b)
        for a in range(len(live))
        for b in range(a + 1, len(live))
        if (best := _best_pair(live[a], live[b], cache)) is None
        or best >= MERGE_CANDIDATE_MIN
    ]
    if not candidates:
        return groups, 0
    await check(candidates)

    def passed(pair: tuple[int, int]) -> bool:
        p = checks.get(pair)
        return p is not None and p >= threshold

    approved = sorted(
        (pair for pair in candidates if passed(pair)),
        key=lambda pair: (-(checks[pair] or 0.0), pair),
    )
    sets: dict[int, list[int]] = {i: [i] for i in range(len(live))}
    owner = {i: i for i in range(len(live))}
    for a, b in approved:
        if len(sets) <= count:
            break
        ra, rb = owner[a], owner[b]
        if ra == rb:
            continue
        across = [(min(x, y), max(x, y)) for x in sets[ra] for y in sets[rb]]
        missing = [pair for pair in across if pair not in checks]
        if missing:
            await check(missing)
        if all(passed(pair) for pair in across):
            for member in sets[rb]:
                owner[member] = ra
            sets[ra] = sorted(sets[ra] + sets.pop(rb))
    out: list[_Group] = []
    for members in sets.values():
        mentions = sorted(
            (m for i in members for m in live[i].mentions),
            key=lambda m: m.span or (0, 0),
        )
        out.append(_Group(tuple(mentions), live[members[0]].qualify_p))
    # what did not qualify stays, as it was
    kept_ids = {id(g) for g in live}
    out.extend(g for g in groups if id(g) not in kept_ids)
    return out, len(checks)


def _union(
    first: Sequence[Mention], second: Sequence[Mention], source: str
) -> list[Mention]:
    """The first discovery's mentions plus what a broadened one found elsewhere.

    A broadened mention only ADDS: one that overlaps a first-pass mention is left
    out, so "Emma's brother Tom" can never take the place of (or merge) the
    mentions "Emma" and "Tom" that were found first.
    """
    fresh = [
        m
        for m in second
        if m.span is None
        or not any(f.span is not None and _overlap(f.span, m.span) for f in first)
    ]
    return sorted(
        [*first, *_resolve_overlaps(fresh, source)], key=lambda m: m.span or (0, 0)
    )


async def collect(
    scope: Scope, run: Run, *, skip_presence: bool = False
) -> CollectionOutcome:
    """Find the distinct entities of a schema in the input.

    Args:
        scope: What to find and where.
        run: The invoke's context (``run.discovery`` and ``run.coref`` configure
            the steps).
        skip_presence: Do not ask whether the input mentions any entity (the
            schema was forced).

    Returns:
        The entities that qualify, the ones that do not, and the warnings.

    Raises:
        ValueError: If ``Many(identity=...)`` names a field the schema lacks.
        NotImplementedError: If ``scope.existing`` is not empty (matching
            against existing records arrives with M8).
    """
    if scope.existing:
        raise NotImplementedError(
            f"matching {scope.plan.name} entities against existing records "
            "arrives in M8"
        )
    plan, config = scope.plan, run.discovery
    warnings: list[str] = []

    def warn(message: str) -> None:
        warnings.append(message)
        run.warn(message)

    names = identity_fields(plan, scope.cardinality)
    if not names:
        warn(
            f"no_identity_field: {plan.name} has no field that identifies an "
            "entity, so discovery relies on the schema's description; name one "
            "with Many(..., identity=...) or s1_field(identity=True)"
        )
    phrase = _identity_phrase(plan, names)

    jobs: list[Any] = [_discover(run, scope, phrase), _count(run, scope)]
    if not skip_presence:
        jobs.append(_presence(run, scope))
    results = await gather(*jobs)
    mentions: list[Mention] = results[0]
    count: int | None = results[1]
    presence: float | None = None
    cut = False
    if not skip_presence:
        presence, cut = results[2]

    def empty() -> CollectionOutcome:
        return CollectionOutcome((), (), tuple(warnings), presence, count)

    if presence is not None and presence < config.presence_absent:
        if not (cut and mentions):
            return empty()
        warn(
            f"presence_unreliable: the input of {plan.name} did not fit one Clef "
            f"request, so its presence answer ({presence:.2f}) came from the start "
            f"of the input only; the {len(mentions)} mention(s) discovery found "
            "were kept"
        )
    if presence is not None and presence < config.presence_present and not mentions:
        return empty()
    if not mentions and not count:
        if presence is not None:
            warn(
                "no_mentions_found: the input seems to mention "
                f"{plural(scope.noun)} but discovery found no mention of one"
            )
        return empty()

    pair_cache: PairCache = {}
    known: dict[frozenset[Mention], float | None] = {}
    groups, _ = await _group(run, scope, mentions, pair_cache, known)
    live = _qualified(groups)

    if count is not None and count != len(live):
        act = (
            config.broaden_on_count_mismatch
            and max(count, len(live)) <= config.count_trust_max
            and not (count == 0 and live)  # "none" is no answer when entities qualify
        )
        if act and count > len(live):
            wider = await _discover(run, scope, phrase, broaden=True)
            merged = _union(mentions, wider, scope.source)
            if merged != mentions:
                mentions = merged
                groups, _ = await _group(run, scope, mentions, pair_cache, known)
                live = _qualified(groups)
        checked = 0
        if act and count < len(live):
            groups, checked = await _merge(run, scope, groups, count, pair_cache)
            live = _qualified(groups)
        why = (
            ""
            if act
            else (
                "; the count is not trusted above "
                f"{config.count_trust_max} entities, so nothing was changed"
                if max(count, len(live)) > config.count_trust_max
                else "; a count of 0 is not trusted when entities qualify, so "
                "nothing was changed"
                if count == 0
                else "; broadening is switched off, so nothing was changed"
            )
        )
        if count > len(live):
            warn(
                f"count_exceeds_discovered: the count is {count} but only "
                f"{len(live)} {scope.noun}(s) were found"
                + (
                    ", even when descriptive references were allowed; no entity "
                    "was invented"
                    if act
                    else why
                )
            )
        elif count < len(live):
            warn(
                f"count_below_discovered: the count is {count} but "
                f"{len(live)} {scope.noun}(s) were found"
                + (
                    (
                        f"; {checked} merge check(s) did not join them further"
                        if checked
                        else "; pairwise coreference found them different, so no "
                        "merge check was asked"
                    )
                    if act
                    else why
                )
            )

    ordered = sorted(groups, key=lambda g: g.first)
    entities: list[EntityInfo] = []
    skipped: list[SkippedEntity] = []
    for index, group in enumerate(ordered):
        info = EntityInfo(
            key=_provisional(index),
            label=group.label,
            mentions=group.mentions,
            first_mention=group.mentions[0],
            qualify_p=group.qualify_p,
            parent=scope.parent.key if scope.parent is not None else None,
        )
        if group in live:
            entities.append(info)
        else:
            p = group.qualify_p
            skipped.append(
                SkippedEntity(
                    plan.name,
                    info,
                    "not_qualified",
                    f"{group.label!r} is not a {scope.noun} (p={p:.2f})"
                    if p is not None
                    else f"{group.label!r} is not a {scope.noun}",
                )
            )
    return CollectionOutcome(
        tuple(entities), tuple(skipped), tuple(warnings), presence, count
    )


# --------------------------------------------------------------------------- #
# Step 6: extraction of every entity
# --------------------------------------------------------------------------- #


async def insert_collection(
    run: Run, scope: Scope, *, skip_presence: bool = False
) -> CollectionResult:
    """Find the entities of a schema and insert one record for each.

    Every entity runs the full single-entity insert pipeline (groups, unions, Clef
    typed fields, verification ladder with attribution, repair loop) concurrently.
    The focus header names the entity and its mentions, instructions are
    templated with its label, and the presence gate is not asked again.

    Args:
        run: The invoke's context.
        scope: What to extract and where from.
        skip_presence: Skip the presence question (the schema was forced).

    Returns:
        The records in order of first mention, the entities that were skipped
        and the discovery outcome.

    Raises:
        ExtractionValidationError: In strict mode, if a record stays invalid.
        TransportError: If a request failed.
    """
    outcome = await collect(scope, run, skip_presence=skip_presence)
    plan = scope.plan

    async def extract(info: EntityInfo) -> InsertOutcome:
        focus = entity_focus(
            plan.name,
            scope.noun,
            info.label,
            [(m.text, m.span) for m in info.mentions],
        )
        plain = entity_focus(
            plan.name, scope.noun, info.label, [(m.text, None) for m in info.mentions]
        )
        return await insert_one(
            run,
            plan,
            scope.source,
            label=info.label,
            entity=info,
            focus=focus,
            focus_plain=plain,
        )

    done = await gather(*(extract(info) for info in outcome.entities))
    records: list[ResultRecord] = []
    skips: list[tuple[str, SkippedEntity]] = [
        (s.entity.key, s) for s in outcome.skipped
    ]
    attempts = 0
    for info, item in zip(outcome.entities, done, strict=True):
        if item.record is not None:
            records.append(item.record)
        if item.skipped is not None:
            skips.append((info.key, item.skipped))
        attempts = max(attempts, item.attempts)
    skips.sort(key=lambda pair: int(pair[0][1:]))
    return CollectionResult(
        records=tuple(records),
        skipped=tuple(s for _, s in skips),
        extracted=bool(outcome.entities),
        attempts=attempts,
        outcome=outcome,
    )
