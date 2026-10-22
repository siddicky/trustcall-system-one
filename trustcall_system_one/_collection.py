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

With existing records of the schema ``apply_collection`` goes on (steps 8 and 9,
``_match``): every entity is matched to a record or to ``__new__`` (one-to-one),
matched records are updated with the full update path, new entities inserted, and,
with ``enable_deletes``, records the input says to remove are deleted.

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
from trustcall_system_one._entity import (
    InsertOutcome,
    UpdateOutcome,
    insert_one,
    update_one,
)
from trustcall_system_one._existing import ExistingRecord, normalise_record
from trustcall_system_one._match import (
    Candidate,
    MatchOutcome,
    flag_deletes,
    identity_keys,
    match_entities,
)
from trustcall_system_one._render import (
    Parent,
    RenderedState,
    discovery_focus,
    entity_focus,
    mention_windows,
)
from trustcall_system_one._run import CLEF_BATCH, Run, gather
from trustcall_system_one._types import (
    CallKind,
    EntityInfo,
    Mention,
    ResultRecord,
    SkippedEntity,
)
from trustcall_system_one._verify import (
    LINEAGE_SEP,
    Ctx,
    Nest,
    Placement,
    ask_clairic,
)
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
    "apply_collection",
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
_PARTICLES = frozenset(
    {"van", "von", "de", "der", "den", "da", "del", "della", "di", "du"}
    | {"la", "le", "bin", "ibn", "al", "ter", "ten"}
)
"""Lower-case name particles ("Vincent van Gogh") that may lie inside one name."""
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
            Discovery, qualification, matching and the removal question then all
            ask about the items *of this parent*.
        pointer: JSON Pointer of the collection inside its parent's document
            (empty at the top level).
        existing: Existing items to match the entities against (the
            ``ExistingRecord``s of the schema at the top level; for a nested list
            one per element, its id the element's index, or the key of a
            mapping entry). ``collect`` does not use them; ``apply_collection``
            matches the entities it found against them.
        lineage: The names of the ancestors of a nested collection including its
            parent, outermost first (``("Emma Thompson (e0)",)``); warnings start
            with them.
        nest: How deep the item plan is nested in the whole document.
    """

    plan: SchemaPlan
    source: str
    cardinality: Many
    parent: EntityInfo | None = None
    pointer: str = ""
    existing: tuple[Any, ...] = ()
    lineage: tuple[str, ...] = ()
    nest: Nest = field(default_factory=Nest)

    @property
    def noun(self) -> str:
        """What one entity is called in prompts."""
        return self.cardinality.label or self.plan.title

    @property
    def name(self) -> str:
        """The collection's field name (``"pets"``), empty at the top level."""
        return self.pointer.rsplit("/", 1)[-1]

    @property
    def call_scope(self) -> str:
        """``CallMeta.scope`` of the collection's calls (``"e0/pets"``)."""
        return f"{self.parent.key}{self.pointer}" if self.parent is not None else ""

    def meta(
        self, kind: CallKind, *, entity: str | None = None, field: str | None = None
    ) -> CallMeta:
        """Routing context of a call of this collection."""
        return CallMeta(
            kind=kind,
            schema=self.plan.name,
            entity=entity,
            field=field,
            scope=self.call_scope,
        )

    def placement(self) -> Placement:
        """Where an item of this collection sits, for its own pipeline."""
        return Placement(self.parent, self.lineage, self.nest, self.call_scope)

    def scoped(self, message: str) -> str:
        """A warning, started with the ancestors and the field of a nested list."""
        if self.parent is None:
            return message
        return f"{LINEAGE_SEP.join([*self.lineage, self.name])}: {message}"

    def parent_entry(self, context_chars: int, *, spans: bool = True) -> dict[str, Any]:
        """The parent as a Clef state shows it: ``{"label", "mentions"}``.

        A mention carries its text, its span (unless the input was narrowed and
        the offsets would point at nothing) and, for the first 8 and the last 2,
        a short context around it.
        """
        assert self.parent is not None
        total = len(self.parent.mentions)
        mentions: list[dict[str, Any]] = []
        for index, m in enumerate(self.parent.mentions):
            entry: dict[str, Any] = {"text": m.text}
            if m.span is not None:
                if spans:
                    entry["span"] = list(m.span)
                if has_context(index, total):
                    start, end = m.span
                    entry["context"] = self.source[
                        max(0, start - context_chars) : end + context_chars
                    ]
            mentions.append(entry)
        return {"label": self.parent.label, "mentions": mentions}

    def parent_spans(self) -> list[Span]:
        """Where the input mentions the parent."""
        if self.parent is None:
            return []
        return [m.span for m in self.parent.mentions if m.span is not None]

    def parent_focus(self) -> Parent | None:
        """The parent for a focus header: its label and mention spans."""
        if self.parent is None:
            return None
        return self.parent.label, [m.span for m in self.parent.mentions]


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
        records: One record per qualifying entity that was inserted or updated,
            in order of first mention (an invalid record is among them, with its
            skip), then the deletes in record order.
        skipped: Entities not turned into a valid record, in order of first
            mention.
        extracted: Whether any extraction request was made.
        attempts: Extraction rounds of the entity that needed the most.
        outcome: The discovery outcome.
        handled: ``(schema name, record id)`` of every existing record that was
            updated, deleted or skipped for a reason about itself
            (``updates_disabled``, ``conflict_update_delete``); the rest are
            untouched. The schema name is ``"__any__"`` for such a record.
        deleted: Ids of the records that were deleted.
    """

    records: tuple[ResultRecord, ...] = ()
    skipped: tuple[SkippedEntity, ...] = ()
    extracted: bool = False
    attempts: int = 0
    outcome: CollectionOutcome = field(default_factory=CollectionOutcome)
    handled: tuple[tuple[str, str], ...] = ()
    deleted: tuple[str, ...] = ()


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


def _name_gap(gap: str, *, words: bool, particles: bool = False) -> bool:
    """Whether the text between two parts of a name is name material.

    Spaces, hyphens inside a hyphenated name, initials ("J.") and, when ``words``
    allows it, capitalised words (a middle name) are, and so are lower-case name
    particles (van, von, de, bin ...) when ``particles`` allows them. A separator
    (``, ; : & /``, a line break, the word "and"), any other lower-case word and a
    possessive ("'s brother") are not: they set one mention apart from the next, or
    make a phrase.
    """
    if re.search(r"[,;:&/\n\r]", gap):
        return False
    for token in re.split(r"[ \t\u00a0\-]+", gap):
        if not token or (len(token) <= 2 and token[0].isupper() and token[1:] in "."):
            continue
        plain = token.replace("'", "").replace("\u2019", "")
        if particles and token in _PARTICLES:
            continue
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
    # a particle ("van") lies inside a name only between capitalised single tokens
    particles = words and all(source[start : start + 1].isupper() for start, _ in parts)
    cursor = whole[0]
    for start, end in sorted(parts):
        if not _name_gap(source[cursor:start], words=words, particles=particles):
            return False
        cursor = end
    return _name_gap(source[cursor : whole[1]], words=words, particles=particles)


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


def _fold(text: str) -> str:
    """Case- and whitespace-insensitive form of a piece of text."""
    return " ".join(text.split()).casefold()


def _occurrences(source: str, text: str) -> list[Span]:
    """Every whole-word occurrence of ``text`` in ``source``.

    Exact first; if there is none, with any whitespace between the words and
    case ignored (Clairic read a state whose line endings may have changed).
    """
    words = text.split()
    if not words:
        return []
    lead = r"(?<!\w)" if words[0][0].isalnum() else ""
    tail = r"(?!\w)" if words[-1][-1].isalnum() else ""
    for flags, glue in ((0, re.escape(text.strip())), (re.IGNORECASE, None)):
        body = glue or r"\s+".join(re.escape(w) for w in words)
        found = [
            (m.start(), m.end())
            for m in re.finditer(f"{lead}{body}{tail}", source, flags)
        ]
        if found:
            return found
    return []


def _part_of_a_name(source: str, span: Span) -> bool:
    """Whether a capitalised word follows the span: it is probably a longer name."""
    after = re.match(r"[ \t][A-Z]", source[span[1] : span[1] + 2])
    return bool(source[span[0] : span[0] + 1].isupper() and after)


def _parent_region(
    source: str, run: Run | None, scope: Scope | None
) -> list[Span] | None:
    """Where the items of a nested list are looked for by text: around the parent.

    The input may talk about other parents elsewhere, and their items must not be
    taken for this one's. ``None``: no restriction (a top-level collection, or a
    parent whose mentions are unknown).
    """
    if scope is None or scope.parent is None or not scope.parent_spans():
        return None
    chars = run.verify.window_chars if run is not None else 600
    return mention_windows(len(source), scope.parent_spans(), chars)


def _search(source: str, text: str, region: Sequence[Span] | None) -> list[Span]:
    """Occurrences of ``text``, those inside the region first (and alone, if any)."""
    found = _occurrences(source, text)
    if region is None:
        return found
    inside = [o for o in found if any(a <= o[0] and o[1] <= b for a, b in region)]
    return inside or found


def _answered_spans(
    answer: ClairicAnswer, anchored: RenderedState | None
) -> list[Span]:
    """The spans the model gave for a list answer, in original-input offsets.

    The values and the spans of a list answer are not index-aligned (a second
    mention can be among the spans while the values leave it out), so the spans
    are never paired with the values by position: each value finds its own by the
    text it covers. The spans as the model returned them are used when
    ``align_list_spans`` kept them aside.
    """
    raw = answer._raw_spans if answer._raw_spans is not None else answer.spans
    if anchored is None or raw is None:
        return []
    found: list[Span] = []
    for pair in raw:
        if len(pair) != 2:
            continue
        mapped = anchored.to_original((pair[0], pair[1]))
        if mapped is not None:
            found.append(mapped)
    return found


def parse_mentions(
    source: str,
    answer: ClairicAnswer | None,
    anchored: RenderedState | None,
    run: Run | None = None,
    scope: Scope | None = None,
) -> list[Mention]:
    """The mentions in a discovery answer, as clean mentions in input order.

    Each distinct value is located in the ORIGINAL input by its text, never by its
    index in the answer (``value`` and ``spans`` are not aligned): every span the
    model returned whose covered text equals the value (case and whitespace
    aside) is a mention of it, so a repeated mention that the values list once is
    kept; a value no span covers (or listed more often than spans cover it) takes
    every whole-word occurrence in the input. A value that is not in the input is
    dropped (Clairic is extractive, so it can only be a slip). Pronoun-only
    mentions are dropped and determiners and possessives stripped
    (``clean_mention``). Of two overlapping mentions the longer is kept only if it
    contains exactly one of them (a span that bridges two mentions replaces
    neither), and two mentions never share a span.

    Args:
        source: The original input.
        answer: Clairic's ``list`` answer.
        anchored: The state to map the answer's spans with.
        run: The invoke's context, to warn about items that could not be placed.
        scope: The collection, to name it in such a warning (nested lists only).

    Returns:
        The mentions, ordered by position.
    """
    if answer is None or not isinstance(answer.value, list):
        return []
    found: list[Mention] = []
    spans = _answered_spans(answer, anchored)
    values = {_fold(str(i)): str(i) for i in answer.value if str(i).strip()}
    listed: dict[str, int] = {}
    for item in answer.value:
        if str(item).strip():
            listed[_fold(str(item))] = listed.get(_fold(str(item)), 0) + 1
    # where the model pointed at a longer listed value: a shorter value's extra
    # occurrences are not looked for inside it
    longer: dict[str, list[Span]] = {}
    for key in values:
        holders = [other for other in values if len(other) > len(key) and key in other]
        longer[key] = [s for s in spans if _fold(source[s[0] : s[1]]) in holders]
    region = _parent_region(source, run, scope)
    used: set[Span] = set()
    done: set[str] = set()
    for item in answer.value:
        text = str(item)
        key = _fold(text)
        if not key or key in done:
            continue
        done.add(key)
        covering = [s for s in spans if _fold(source[s[0] : s[1]]) == key]
        if len(covering) < listed[key]:
            free = [o for o in _search(source, text, region) if o not in covering]
            if covering:  # the model pointed at some: only as many more as listed
                free = [
                    o
                    for o in free
                    if not any(h[0] <= o[0] and o[1] <= h[1] for h in longer[key])
                ]
                free.sort(key=lambda o: (_part_of_a_name(source, o), o))
                free = sorted(free[: listed[key] - len(covering)])
            covering = [*covering, *free]
        if not covering:
            if run is not None:
                message = (
                    f"mention_not_in_input: discovery returned {text!r}, which "
                    "the input does not contain; it was ignored"
                )
                run.warn(scope.scoped(message) if scope is not None else message)
            continue
        for span in covering:
            if span in used:
                continue
            used.add(span)
            cleaned = clean_mention(source[span[0] : span[1]], span)
            if cleaned is not None and cleaned.span is not None:
                found.append(cleaned)
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
    nested = scope.parent is not None
    context = run.coref.context_chars
    question = NoulQuestion(
        type="noul",
        instructions=(
            f"Does `input` mention any {scope.noun}"
            + (f" ({described})" if described else "")
            + (" belonging to `parent.label`" if nested else "")
            + "?"
        ),
        criteria=NoulCriteria(
            true=f"The input mentions at least one {scope.noun}"
            + (" belonging to the parent." if nested else "."),
            false=f"The input mentions no {scope.noun}"
            + (" belonging to the parent." if nested else "."),
        ),
    )
    state = run.clef_state(
        scope.source,
        spans=scope.parent_spans() or None,
        extra={"parent": scope.parent_entry(context)} if nested else None,
    )
    cut = state["input"] != scope.source
    if nested and cut:  # the offsets point into the whole input, not this one
        state["parent"] = scope.parent_entry(context, spans=False)
    answers = await run.call_clef(
        {"presence": question}, state, meta=scope.meta("presence")
    )
    answer = answers.get("presence")
    return (answer.noul if isinstance(answer, NoulAnswer) else None), cut


def _discovery_ctx(run: Run, scope: Scope) -> Ctx:
    return Ctx(
        run=run,
        plan=scope.plan,
        text=scope.source,
        label=scope.noun,
        key="",
        focus=discovery_focus(scope.plan.name, scope.noun, scope.parent_focus()),
        parent=scope.parent,
        lineage=scope.lineage,
        nest=scope.nest,
        scope=scope.call_scope,
    )


async def _discover(
    run: Run, scope: Scope, phrase: str, *, broaden: bool = False
) -> list[Mention]:
    """Ask Clairic for every mention of an entity, and clean what comes back."""
    noun = scope.noun
    of = f" belonging to {scope.parent.label}" if scope.parent is not None else ""
    if phrase:
        what = f"Every mention of a {noun}{of} by their {phrase}"
    else:
        described = _described(scope.plan)
        what = f"Every mention of a {noun}{of}" + (
            f" ({described})" if described else ""
        )
    instructions = (
        f"{what}, in the order they appear in the input. Include repeated "
        "mentions and shortened ones (a first name alone). Do not list pronouns. "
        "Each item is the exact words the input uses."
    ) + (_BROADER if broaden else "")
    fields = {
        "mentions": ClairicField(type="list", items="text", instructions=instructions)
    }
    ctx = _discovery_ctx(run, scope)
    asked = await ask_clairic(ctx, fields, meta=scope.meta("discover"))
    answer, anchored = asked["mentions"]
    return parse_mentions(scope.source, answer, anchored, run, scope)


async def _count(run: Run, scope: Scope) -> int | None:
    """Ask Clairic how many distinct entities the input mentions."""
    noun = scope.noun
    of = f" belonging to {scope.parent.label}" if scope.parent is not None else ""
    field_ = ClairicField(
        type="number",
        integer=True,
        instructions=(
            f"How many distinct {plural(noun)}{of} does the input mention? Count "
            f"each individual {noun} once, however many times they are mentioned."
        ),
    )
    ctx = _discovery_ctx(run, scope)
    response = await run.call_clairic(
        {"count": field_}, ctx.rendered(), meta=scope.meta("count")
    )
    answer = response.answers.get("count")
    if response.truncated:
        run.warn(
            scope.scoped(
                f"count_unreliable: the input of {scope.plan.name} did not fit one "
                "Clairic request, so the count of entities was ignored"
            )
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

    nested = scope.parent is not None

    async def ask(chunk: Sequence[tuple[int, _Group]]) -> dict[str, float | None]:
        anchors = [m.span for _, g in chunk for m in g.mentions if m.span is not None]
        extra: dict[str, Any] = {
            "entities": {_provisional(n): entity(g, spans=True) for n, g in chunk}
        }
        if nested:
            extra["parent"] = scope.parent_entry(context)
        state = run.clef_state(
            scope.source,
            spans=[*anchors, *scope.parent_spans()],
            extra=extra,
        )
        if state["input"] != scope.source:  # offsets no longer point into the input
            state["entities"] = {
                _provisional(n): entity(g, spans=False) for n, g in chunk
            }
            if nested:
                state["parent"] = scope.parent_entry(context, spans=False)
        belonging = " belonging to `parent.label`" if nested else ""
        questions = {
            f"qualify.{_provisional(n)}": NoulQuestion(
                type="noul",
                instructions=(
                    f"Is `entities.{_provisional(n)}.label` a {scope.noun}"
                    + (f" ({described})" if described else "")
                    + belonging
                    + "?"
                ),
                criteria=NoulCriteria(
                    true=f"The entity is a {scope.noun}"
                    + (" belonging to the parent." if nested else "."),
                    false="The entity is something else, not a "
                    + f"{scope.noun}"
                    + (" of the parent." if nested else "."),
                ),
            )
            for n, _ in chunk
        }
        answers = await run.call_clef(questions, state, meta=scope.meta("qualify"))
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
        scope=scope.call_scope,
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
    answers = await run.call_clef({"merge": question}, state, meta=scope.meta("coref"))
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
    """
    plan, config = scope.plan, run.discovery
    warnings: list[str] = []

    def warn(message: str) -> None:
        message = scope.scoped(message)
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


def _focuses(scope: Scope, info: EntityInfo) -> tuple[Any, Any]:
    """The focus header of an entity's Clairic requests, with and without offsets."""
    plan = scope.plan
    parent = scope.parent_focus()
    focus = entity_focus(
        plan.name,
        scope.noun,
        info.label,
        [(m.text, m.span) for m in info.mentions],
        parent,
    )
    plain = entity_focus(
        plan.name,
        scope.noun,
        info.label,
        [(m.text, None) for m in info.mentions],
        (parent[0], []) if parent is not None else None,
    )
    return focus, plain


async def _insert_entity(run: Run, scope: Scope, info: EntityInfo) -> InsertOutcome:
    focus, plain = _focuses(scope, info)
    return await insert_one(
        run,
        scope.plan,
        scope.source,
        label=info.label,
        entity=info,
        focus=focus,
        focus_plain=plain,
        placement=scope.placement(),
    )


async def _update_entity(
    run: Run,
    scope: Scope,
    info: EntityInfo,
    target: ExistingRecord,
    *,
    enable_updates: bool,
) -> UpdateOutcome:
    focus, plain = _focuses(scope, info)
    return await update_one(
        run,
        scope.plan,
        scope.source,
        target,
        label=info.label,
        enable_updates=enable_updates,
        entity=info,
        focus=focus,
        focus_plain=plain,
        identity=identity_fields(scope.plan, scope.cardinality),
        placement=scope.placement(),
    )


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
    done = await gather(*(_insert_entity(run, scope, e) for e in outcome.entities))
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
        # the items of this entity's nested lists follow it
        skips.extend((info.key, nested) for nested in item.nested_skipped)
        attempts = max(attempts, item.attempts)
    skips.sort(key=lambda pair: int(pair[0][1:]))
    return CollectionResult(
        records=tuple(records),
        skipped=tuple(s for _, s in skips),
        extracted=bool(outcome.entities),
        attempts=attempts,
        outcome=outcome,
    )


# --------------------------------------------------------------------------- #
# Steps 8-9: records the input is about
# --------------------------------------------------------------------------- #


_LIKELY_RECORD = 0.25
"""Probability from which a record is an ambiguous entity's likely match."""


@dataclass
class _Work:
    """What happened to one entity once it was matched."""

    info: EntityInfo
    record: ResultRecord | None = None
    skipped: SkippedEntity | None = None
    attempts: int = 0
    extracted: bool = False
    handled: tuple[str, str] | None = None
    nested_skipped: tuple[SkippedEntity, ...] = ()


def _delete_record(plan: SchemaPlan, candidate: Candidate, p: float) -> ResultRecord:
    """The result record of a deletion: the record as it was, and nothing after."""
    return ResultRecord(
        schema_name=plan.name,
        json_doc_id=candidate.record_id,
        kind="delete",
        value=None,
        before=dict(candidate.data),
        after=None,
        delete_p=p,
    )


async def apply_collection(
    run: Run,
    scope: Scope,
    *,
    skip_presence: bool = False,
    enable_inserts: bool = False,
    enable_updates: bool = True,
    enable_deletes: bool = False,
) -> CollectionResult:
    """Find the entities of a schema and deal with them against its existing records.

    Without existing records this is ``insert_collection``. With them:

    1. ``collect`` finds the entities (discovery, coreference, qualification);
    2. each entity is matched to an existing record or to ``__new__``, one-to-one
       (``_match``); with ``enable_deletes`` a Clef question per record asks
       whether the input says to remove it, concurrently;
    3. an entity matched to a record is updated by the full update path of
       ``update_one`` (presence not asked again, attribution scoped to the entity),
       unless ``enable_updates`` is off (``updates_disabled``); a new
       entity is inserted if ``enable_inserts`` (else ``inserts_disabled``); an
       entity that could not be matched is ``ambiguous_match``. If the record is
       also flagged for deletion the update runs and decides: with no operation
       through the gate the record is deleted (the input only says "remove it"),
       with at least one it is a ``conflict_update_delete`` and neither happens;
    4. the records flagged for deletion are deleted (``kind="delete"``, ``after``
       ``None``), last, in record order; records nothing happened to are untouched.

    Args:
        run: The invoke's context.
        scope: What to find and where; ``scope.existing`` holds the schema's
            ``ExistingRecord``s.
        skip_presence: Skip the presence question (the schema was forced).
        enable_inserts: Insert entities that match no record.
        enable_updates: Update the records entities were matched to.
        enable_deletes: Delete records the input says to remove.

    Returns:
        The records (inserts and updates in order of first mention, then the
        deletes), the skipped entities and which records were dealt with.

    Raises:
        AmbiguousMatchError: With ``match.on_ambiguous="raise"``.
        ExtractionValidationError: In strict mode, if a record stays invalid.
        TransportError: If a request failed.
    """
    existing: tuple[ExistingRecord, ...] = tuple(scope.existing)
    if not existing:
        return await insert_collection(run, scope, skip_presence=skip_presence)
    plan = scope.plan
    outcome = await collect(scope, run, skip_presence=skip_presence)
    names = identity_fields(plan, scope.cardinality)
    keys = identity_keys(plan, names)
    candidates = [
        Candidate(r.record_id, normalise_record(plan, r.data)) for r in existing
    ]
    by_id = {r.record_id: r for r in existing}
    mentioned = not (
        not outcome.entities
        and outcome.presence is not None
        and outcome.presence < run.discovery.presence_absent
    )

    async def merge_check(a: EntityInfo, b: EntityInfo) -> float | None:
        return await _merge_check(
            run,
            scope,
            _Group(a.mentions, a.qualify_p),
            _Group(b.mentions, b.qualify_p),
        )

    async def matching() -> MatchOutcome:
        return await match_entities(
            run,
            plan,
            scope.source,
            scope.noun,
            outcome.entities,
            candidates,
            keys,
            merge_check,
            scope.placement(),
        )

    async def deleting() -> dict[str, float | None]:
        if not (enable_deletes and mentioned):
            return {}
        return await flag_deletes(
            run, plan, scope.source, candidates, keys, scope.placement()
        )

    jobs: list[Any] = [matching(), deleting()]
    results = await gather(*jobs)
    matched: MatchOutcome = results[0]
    delete_p: dict[str, float | None] = results[1]
    flagged = {
        rid
        for rid, p in delete_p.items()
        if p is not None and p >= run.match.delete_threshold
    }

    async def flagged_work(info: EntityInfo, record: ExistingRecord) -> _Work:
        """An entity matched to a record the input also says to delete.

        The update runs first. If no operation survives the gate (the input says
        "remove it" and nothing else about it) the record is deleted; if the update
        has at least one gated operation the input both changes and removes the
        record, which is a conflict: neither happens. An identity conflict (a wrong
        match) is reported as such and holds the delete (``match_delete_held``).
        """
        handle = (record.schema_name, record.record_id)
        if not enable_updates:  # nothing could be changed: the removal stands
            return _Work(info, handled=handle)
        done = await _update_entity(run, scope, info, record, enable_updates=True)
        kept = done.record.patches if done.record is not None else ()
        if done.skipped is not None and done.skipped.reason == "ambiguous_match":
            return _Work(
                info,
                done.record,
                done.skipped,
                done.attempts,
                done.extracted,
                done.handled,
                done.nested_skipped,
            )
        if not kept:
            return _Work(
                info,
                attempts=done.attempts,
                extracted=done.extracted,
                handled=handle,
            )
        p = delete_p[record.record_id] or 0.0
        return _Work(
            info,
            skipped=SkippedEntity(
                plan.name,
                info,
                "conflict_update_delete",
                f"{info.label!r} matches the record {record.record_id!r}, and the "
                f"input also says that record should be deleted (p={p:.2f}) while "
                f"changing it ({len(kept)} operation(s) passed the gate); it was "
                "neither updated nor deleted",
                ((record.record_id, info.match_probs.get(record.record_id, 0.0)),),
            ),
            attempts=done.attempts,
            extracted=done.extracted,
            handled=handle,
        )

    async def work(info: EntityInfo) -> _Work:
        target = matched.targets[info.key]
        if target is not None:
            record = by_id[target]
            if target in flagged:
                return await flagged_work(info, record)
            done = await _update_entity(
                run, scope, info, record, enable_updates=enable_updates
            )
            return _Work(
                info,
                done.record,
                done.skipped,
                done.attempts,
                done.extracted,
                done.handled,
                done.nested_skipped,
            )
        if not (enable_inserts or not existing):
            return _Work(
                info,
                skipped=SkippedEntity(
                    plan.name,
                    info,
                    "inserts_disabled",
                    f"{info.label!r} is not among the existing "
                    f"{plan.name} records, but enable_inserts is off",
                    tuple((rid, p) for rid, p in info.match_probs.items() if p >= 0.01),
                ),
            )
        inserted = await _insert_entity(run, scope, info)
        return _Work(
            info,
            inserted.record,
            inserted.skipped,
            inserted.attempts,
            inserted.extracted,
            nested_skipped=inserted.nested_skipped,
        )

    todo = [e for e in matched.entities if e.key in matched.targets]
    done_all = await gather(*(work(e) for e in todo))
    records: list[tuple[int, ResultRecord]] = []
    skips: list[tuple[int, SkippedEntity]] = [
        (int(s.entity.key[1:]), s) for s in (*outcome.skipped, *matched.skipped)
    ]
    handled: list[tuple[str, str]] = []
    attempts = 0
    extracted = False
    for item in done_all:
        order = int(item.info.key[1:])
        if item.record is not None:
            records.append((order, item.record))
        if item.skipped is not None:
            skips.append((order, item.skipped))
        skips.extend((order, nested) for nested in item.nested_skipped)
        if item.handled is not None:
            handled.append(item.handled)
        attempts = max(attempts, item.attempts)
        extracted = extracted or item.extracted
    conflicts = {
        w.handled[1]
        for w in done_all
        if w.skipped is not None and w.skipped.reason == "conflict_update_delete"
        if w.handled is not None
    }
    # a record an ambiguous entity most likely is must not be deleted on the same
    # input: the entity may be the very thing the input says to delete or update
    unsure: dict[str, str] = {}
    for _, skip in skips:
        if skip.reason != "ambiguous_match":
            continue
        best = max(
            ((rid, p) for rid, p in skip.candidates if rid in by_id),
            key=lambda pair: pair[1],
            default=None,
        )
        if best is not None and best[1] >= _LIKELY_RECORD:
            unsure[best[0]] = skip.entity.label
    deletes: list[ResultRecord] = []
    for candidate in candidates:
        rid = candidate.record_id
        if rid in flagged and rid in unsure:
            run.warn(
                scope.scoped(
                    f"match_delete_held: the record {rid!r} is flagged for deletion, "
                    f"but {unsure[rid]!r} could not be matched with certainty and "
                    "the record is its most likely match, so it was not deleted"
                )
            )
            continue
        if rid in flagged and rid not in conflicts:
            deletes.append(_delete_record(plan, candidate, delete_p[rid] or 0.0))
            record = by_id[rid]
            handled.append((record.schema_name, rid))
    records.sort(key=lambda pair: pair[0])
    skips.sort(key=lambda pair: pair[0])
    return CollectionResult(
        records=(*(r for _, r in records), *deletes),
        skipped=tuple(s for _, s in skips),
        extracted=extracted,
        attempts=attempts,
        outcome=outcome,
        handled=tuple(dict.fromkeys(handled)),
        deleted=tuple(r.json_doc_id for r in deletes if r.json_doc_id is not None),
    )
