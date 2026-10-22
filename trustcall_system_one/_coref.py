"""Mention clustering: which mentions of an input name the same entity.

The unit is the *mention* (a piece of the input with its span), never the
surface form: two people who are both called "Emma" are two mentions with the
same text and may be two entities.

Two strategies, picked by ``CorefConfig.strategy``:

* ``pairwise``: one Clef ``noul`` per pair of mentions; above
  ``CorefConfig.all_pairs_below`` mentions only per PLAUSIBLE pair (they share a
  name token, one is a short form, a nickname or the initials of the other, a
  single name stands near a longer one, or one side is a descriptive reference
  such as "my cousin"). A pair that is not asked is "not asked", never a "no": it
  stays out of the pair cache, and a count-driven merge can still check it. The
  questions go out in requests of at most ``Run.clef_batch()`` (64; 32 on
  Experiential Labs), each with a structured state
  ``{"input", "mentions": {"m0": {"text", "span", "context"}}}``
  that holds only the mentions its questions name; the requests run
  concurrently. The answers are clustered with average linkage, which settles
  inconsistent triangles (A~B 0.9, B~C 0.9, A~C 0.05 stays two entities because
  the average over ``{A, B}`` and ``{C}`` is below the threshold).
* ``incremental``: the mentions are walked in order, in batches of 16: each batch
  is one request of Clef choices over the clusters that existed when it started
  (plus ``__new__``; each cluster is described once, in ``state.clusters``), and
  the mentions of one batch that start new clusters are then compared with each
  other pairwise. It needs about a request per 16 mentions, so ``auto`` switches
  to it for long lists. Pairs of mentions from different batches are never
  compared, so only the pairs inside a batch land in the pair cache.
"""

import difflib
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from trustcall_system_one._render import (
    CHOICE_OPTION_TOKENS,
    estimate_json_tokens,
    estimate_request_tokens,
)
from trustcall_system_one._run import Run, gather
from trustcall_system_one._types import Mention
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
)
from trustcall_system_one.transport import CallMeta

__all__ = [
    "NEW",
    "Clustering",
    "PairCache",
    "agglomerate",
    "article",
    "canonical_label",
    "cluster_mentions",
    "has_context",
    "identity_compatible",
    "mention_entry",
    "name_part",
    "name_tokens",
    "near",
    "nicknames",
    "normalise",
    "ordered_pair",
    "plausible",
    "plural",
    "strong_compatible",
]

logger = logging.getLogger(__name__)

NEW = "__new__"
"""The option of an incremental assignment that starts a new cluster."""
_MAX_OPTIONS = 255
_BATCH = 16
"""Mentions an incremental request assigns at once."""
_MIN_OPTIONS = 24
"""Fewest earlier entities a batch is matched against before the batch is made
smaller instead (a choice question carries every option once per question)."""
_INPUT_SHARE = 0.5
"""Largest share of the Clef budget the input may take in a coreference request; the
rest is for the mentions, the earlier entities and the questions."""
_FLOOR = 1e-12
_POSSESSIVES = frozenset({"my", "our", "your", "his", "her", "their", "its"})

PairCache = dict[tuple[Mention, Mention], float]
"""Pair probabilities already asked, by the two mentions (the earlier first)."""


@dataclass(frozen=True)
class Clustering:
    """The result of clustering a list of mentions.

    Attributes:
        clusters: The clusters as tuples of indexes into the mention list, each
            in increasing order, the clusters ordered by their first mention.
        pairs: Probability that two mentions are the same entity, by the pair of
            their indexes (smaller first), for the pairs that were asked; a pair
            that is not here was not asked (not "no"). Empty for the incremental
            strategy, apart from the pairs compared inside a batch.
        strategy: The strategy that ran.
    """

    clusters: tuple[tuple[int, ...], ...]
    pairs: Mapping[tuple[int, int], float]
    strategy: Literal["pairwise", "incremental"]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def normalise(text: str) -> str:
    """Fold a mention for comparison: case, spacing and edge punctuation."""
    return " ".join(re.sub("[^\\w\\s'\u2019-]", " ", text).casefold().split())


def plural(noun: str) -> str:
    """Pluralise a schema title for a prompt (``Class`` becomes ``Classes``).

    Only the regular English rules: consonant + y -> ies, a sibilant ending ->
    es, anything else + s. The result is for prompts, so "Persons" is fine.
    """
    lower = noun.lower()
    if len(noun) > 1 and lower.endswith("y") and lower[-2] not in "aeiou":
        return noun[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


_AN_LETTERS = frozenset("AEFHILMNORSX")
_A_BEFORE_VOWEL = ("uni", "use", "usu", "uti", "eu", "one", "once", "ubiq")
_AN_BEFORE_CONSONANT = ("hour", "honest", "honor", "honour", "heir")


def article(noun: str) -> str:
    """``a`` or ``an`` for a noun phrase, by how it is said: "a user", "an hour".

    An acronym is said letter by letter ("an LLM", "a CEO").
    """
    found = re.match(r"[A-Za-z]+", noun.strip())
    word = found.group(0) if found else ""
    if not word:
        return "a"
    lower = word.lower()
    if len(word) > 1 and word.isupper():
        return "an" if word[0] in _AN_LETTERS else "a"
    if lower.startswith(_AN_BEFORE_CONSONANT):
        return "an"
    if lower.startswith(_A_BEFORE_VOWEL):
        return "a"
    return "an" if lower[0] in "aeiou" else "a"


def _capitalised(token: str) -> bool:
    word = token.lstrip("\"'([{\u201c\u2018")
    return bool(word) and (word[0].isupper() or word[0].isdigit())


def name_part(text: str) -> str | None:
    """The proper-noun part of a mention, or ``None`` for a descriptive one.

    A name starts with a capital ("Emma", "Dr. Who", "The Hague"). A mention led by
    a possessive ("my old college roommate", "her cousin Olivia Davis") is a
    description of someone, and only a capitalised run after it is a name ("Olivia
    Davis"); a lower-case word ("cousin") is no name at all.

    Args:
        text: A mention as written.

    Returns:
        The name, or ``None`` when the mention only describes the entity.
    """
    tokens = text.split()
    if not tokens:
        return None
    start = 0
    if tokens[0].casefold() in _POSSESSIVES:
        start = 1
        while start < len(tokens) and not _capitalised(tokens[start]):
            start += 1
        if start >= len(tokens):
            return None
    if not _capitalised(tokens[start]):
        return None
    return " ".join(tokens[start:])


def canonical_label(texts: Sequence[str]) -> str:
    """The name a cluster of mentions is called by.

    A name beats a description ("Emma" over "my old college roommate"; "her cousin
    Olivia Davis" is called "Olivia Davis", see ``name_part``). Among names the one
    with the most tokens wins ("Emma Thompson" over "Emma"), then the normalised
    form that occurs most often, then the earliest.

    Args:
        texts: The mention texts of one cluster, in input order.

    Returns:
        The chosen label (a mention, or the name inside one).

    Raises:
        ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("a cluster has at least one mention")
    shown = [name_part(text) or text for text in texts]
    named = [name_part(text) is not None for text in texts]
    counts: dict[str, int] = {}
    for label in shown:
        counts[normalise(label)] = counts.get(normalise(label), 0) + 1
    best = min(
        range(len(texts)),
        key=lambda i: (
            not named[i],
            -len(shown[i].split()),
            -counts[normalise(shown[i])],
            i,
        ),
    )
    return shown[best]


def name_tokens(text: str) -> list[str]:
    """The case-folded words of a name or title (hyphens split words)."""
    return [t for t in normalise(text).replace("-", " ").split() if t]


def _initials(tokens: Sequence[str]) -> str:
    return "".join(t[0] for t in tokens)


_NICKNAMES = tuple(
    frozenset(group.split())
    for group in (
        "robert rob bob bobby robbie",
        "william will bill billy willy liam",
        "elizabeth liz beth betty eliza lizzie liza",
        "margaret maggie peggy meg marge margie",
        "michael mike mickey mick",
        "james jim jimmy jamie",
        "katherine catherine kate katie kathy cathy kat",
        "richard dick rick rich richie ricky",
        "anthony tony",
        "susan sue susie suzy",
        "jennifer jen jenny",
        "christopher chris",
        "christine christina chris tina",
        "alexander alex",
        "alexandra alex",
        "samuel sam",
        "samantha sam",
        "daniel dan danny",
        "thomas tom tommy",
        "joseph joe joey",
        "nicholas nick",
        "benjamin ben",
        "matthew matt",
        "edward ed eddie ted teddy",
        "theodore ted theo",
        "john jack johnny jon",
        "jonathan jon",
        "henry harry hank",
        "charles charlie chuck",
        "patricia pat patty trish",
        "deborah debbie deb",
        "rebecca becky becca",
        "stephen steven steve",
        "andrew andy drew",
        "david dave",
        "timothy tim",
        "frederick fred freddie",
        "gregory greg",
        "joshua josh",
        "kenneth ken kenny",
        "lawrence larry",
        "victoria vicky",
        "abigail abby",
        "dorothy dot dottie",
        "eleanor ellie nell",
        "nathaniel nate nat",
        "peter pete",
        "philip phil",
        "ronald ron ronnie",
        "walter walt wally",
        "zachary zack zach",
    )
)
"""Groups of English given names that are short forms of one another."""
_NEAR_CHARS = 200
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def nicknames(first: str, second: str) -> bool:
    """Whether two given names are forms of one name (``Bob`` and ``Robert``)."""
    return any(first in group and second in group for group in _NICKNAMES)


_TYPO = 0.8


def _token_fits(short: str, long: str) -> bool:
    """Whether two name tokens can be the same word written two ways."""
    if short == long or nicknames(short, long):
        return True
    if len(short) == 1 or len(long) == 1:  # an initial
        return short[0] == long[0]
    if any(ch.isdigit() for ch in short + long):
        return False  # numbers must be equal
    if (
        len(short) >= 2
        and len(long) >= 2
        and (long.startswith(short) or short.startswith(long))
    ):
        return True  # Ann/Anne, Em/Emma
    if min(len(short), len(long)) < 4:
        return False
    return difflib.SequenceMatcher(None, short, long).ratio() >= _TYPO


def identity_compatible(first: str, second: str) -> bool:
    """Whether two identity values can name one entity.

    They can when every word of the one with fewer words is a form of a different
    word of the other: the same word, a diminutive (Liz and Elizabeth), an initial,
    a prefix (Ann and Anne), a small typo (Thomson and Thompson), or the initials
    of the whole name ("JS" for John Smith). "Sarah" fits "Sarah Johnson"; "Emma
    Watson" does not fit "Emma Thompson" (a shared first name is not enough), and
    two ids that differ in a number never fit. A blank value (empty, whitespace or
    only punctuation) fits nothing: there is nothing to compare.

    Args:
        first: One identity value (a name, a title, an id).
        second: The other.
    """
    a, b = name_tokens(first), name_tokens(second)
    if not a or not b:
        return False  # nothing to compare is no evidence: a blank fits no one
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if (
        len(short) == 1
        and len(short[0]) >= 2
        and short[0] == "".join(t[0] for t in long)
    ):
        return True
    free = list(long)
    for token in short:
        for other in free:
            if _token_fits(token, other):
                free.remove(other)
                break
        else:
            return False
    return True


def strong_compatible(mention: str, value: str) -> bool:
    """Whether a mention fits an identity value well enough to stand on its own.

    ``identity_compatible`` also accepts a one-word mention that is only the start
    of a word, a typo of it or an initial ("Dan" for "Daniela Cruz", "Jo" for
    "John Doe"), which is a coin flip between two people. A fit stands on its own
    when the mention has two words or more ("Emma Thomson" for "Emma Thompson"),
    or its single word is a word of the value or a diminutive of one ("Liz" for
    "Elizabeth Parker").

    Args:
        mention: How the input names the entity.
        value: An identity value of the record.
    """
    if not identity_compatible(mention, value):
        return False
    words = name_tokens(mention)
    if len(words) >= 2:
        return True
    return any(words[0] == t or nicknames(words[0], t) for t in name_tokens(value))


def near(source: str, first: Mention, second: Mention) -> bool:
    """Whether two mentions stand in one sentence or within about 200 characters."""
    if first.span is None or second.span is None:
        return False
    (_, a1), (b0, _) = sorted([first.span, second.span])
    if b0 - a1 <= _NEAR_CHARS:
        return True
    return _SENTENCE_END.search(source, a1, b0) is None


def plausible(first: str, second: str, *, near: bool = False) -> bool:
    """Whether two mentions are worth asking about when not every pair is.

    They are when they share a name token ("Emma" and "Emma Thompson"), when one
    token is the start of another ("Ann" and "Anne", "Em" and "Emma"), when two
    tokens are forms of one given name ("Bob" and "Robert"), when one is the
    initials of the other ("JS", "J. Smith" and "John Smith"), when a single name
    stands near a longer one (``near``: "Bob" in the sentence after "Robert
    Smith"), or when either side only describes its entity ("my cousin" could be
    anyone). Everything else, "Emma Thompson" and "Michael Chen", is not asked.

    Args:
        first: One mention as written.
        second: The other.
        near: Whether the two stand in one sentence or within about 200 characters.
    """
    if name_part(first) is None or name_part(second) is None:
        return True
    a, b = name_tokens(first), name_tokens(second)
    if set(a) & set(b):
        return True
    for x in a:
        for y in b:
            if len(x) >= 2 and len(y) >= 2 and (x.startswith(y) or y.startswith(x)):
                return True
            if nicknames(x, y):
                return True
    if near and (len(a) == 1) != (len(b) == 1):
        return True
    for short, full in ((a, b), (b, a)):
        letters = "".join(short)
        if len(short) == 1 and len(letters) >= 2 and letters == _initials(full):
            return True  # "JS" for John Smith
        initials = [t for t in short if len(t) == 1]
        # "J. Smith": every one-letter token starts some token of the other
        if (
            initials
            and len(short) <= len(full)
            and all(any(f.startswith(t) for f in full) for t in initials)
        ):
            return True
    return False


def agglomerate(
    n: int, probability: Callable[[int, int], float], threshold: float
) -> list[list[int]]:
    """Average-linkage agglomerative clustering of ``n`` items.

    The two clusters with the highest average pairwise probability are merged
    until no pair of clusters averages ``threshold`` or more. Ties are broken
    towards the earlier first mention (the lower item index), so the result does
    not depend on dictionary or set order.

    Args:
        n: Number of items (mentions), numbered ``0..n-1`` in input order.
        probability: ``probability(i, j)`` for ``i < j``.
        threshold: Merge while the best average is at least this.

    Returns:
        The clusters, each sorted, ordered by their smallest item.
    """
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    link = {
        (a, b): probability(a, b) for a in range(n) for b in range(a + 1, n)
    }  # sums of the pairwise probabilities between two clusters
    while link:
        best: tuple[float, int, int] | None = None
        for (a, b), total in link.items():
            average = total / (len(members[a]) * len(members[b]))
            if average < threshold - _FLOOR:
                continue
            key = (-round(average, 12), a, b)
            if best is None or key < best:
                best = key
        if best is None:
            break
        _, a, b = best
        for c in members:
            if c in (a, b):
                continue
            kept, gone = (min(a, c), max(a, c)), (min(b, c), max(b, c))
            link[kept] = link[kept] + link.pop(gone)
        del link[(a, b)]
        members[a] = sorted(members[a] + members.pop(b))
    return [members[key] for key in sorted(members)]


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #


def has_context(index: int, total: int) -> bool:
    """Whether the ``index``-th of ``total`` mentions of an entity shows its context.

    Only the first 8 and the last 2 do: an entity mentioned dozens of times needs
    no more to be told apart, and the state stays small.
    """
    return index < 8 or index >= total - 2


def mention_entry(
    source: str, mention: Mention, context_chars: int
) -> dict[str, object]:
    """One mention as the coreference states show it.

    Returns:
        ``{"text", "span", "context"}``: the mention as written, its span in the
        input and ``context_chars`` of the input on each side of it.
    """
    entry: dict[str, object] = {"text": mention.text}
    if mention.span is not None:
        start, end = mention.span
        entry["span"] = [start, end]
        entry["context"] = source[max(0, start - context_chars) : end + context_chars]
    return entry


def _spans(mentions: Sequence[Mention]) -> list[tuple[int, int]]:
    return [m.span for m in mentions if m.span is not None]


def _pair_id(i: int, j: int) -> str:
    return f"pair.m{i}.m{j}"


def ordered_pair(first: Mention, second: Mention) -> tuple[Mention, Mention]:
    """The pair with the earlier mention first (the cache's key)."""
    a = first.span or (0, 0)
    b = second.span or (0, 0)
    return (first, second) if (a, first.text) <= (b, second.text) else (second, first)


def _state(
    run: Run,
    source: str,
    shown: Mapping[int, Mention],
    clusters: Mapping[str, object] | None = None,
    *,
    share: float = 1.0,
) -> dict[str, object]:
    """A coreference state holding only the mentions ``shown`` (by index).

    ``clusters`` (incremental assignment) are the earlier entities the mentions are
    matched with, each described once. ``share`` is how much of the Clef budget the
    state may take: past it the input is narrowed to windows around the mentions
    shown (``Run.clef_state``), and the spans of the mentions are re-based into the
    narrowed input.
    """
    extra: dict[str, object] = {
        "mentions": {
            f"m{i}": mention_entry(source, m, run.coref.context_chars)
            for i, m in shown.items()
        }
    }
    if clusters is not None:
        extra["clusters"] = dict(clusters)
    return run.clef_state(
        source,
        spans=_spans(list(shown.values())),
        extra=extra,
        budget=int(run.clef_budget() * share),
    )


async def _pairwise(
    run: Run,
    *,
    source: str,
    mentions: Sequence[Mention],
    noun: str,
    schema: str,
    cache: PairCache,
    scope: str = "",
) -> dict[tuple[int, int], float]:
    """Ask about the pairs of mentions that the cache does not know yet.

    Up to ``coref.all_pairs_below`` mentions every pair is asked; above it only the
    plausible ones. A pair that is not asked is left OUT of the cache and of the
    result: it is "not asked", which is not a confident no.
    """
    n = len(mentions)
    everything = n <= run.coref.all_pairs_below
    missing: dict[str, tuple[int, int]] = {}
    unasked = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ordered_pair(mentions[i], mentions[j]) in cache:
                continue
            if everything or plausible(
                mentions[i].text,
                mentions[j].text,
                near=near(source, mentions[i], mentions[j]),
            ):
                missing[_pair_id(i, j)] = (i, j)
            else:
                unasked += 1
    if unasked:
        logger.debug(
            "coref: %d of %d pairs of %s mentions share no name token, short form, "
            "nickname or neighbourhood and were not asked about",
            unasked,
            n * (n - 1) // 2,
            schema,
        )
    if missing:
        questions = {
            qid: NoulQuestion(
                type="noul",
                instructions=(
                    f"Do `mentions.m{i}` and `mentions.m{j}` refer to the same {noun}?"
                ),
                criteria=NoulCriteria(
                    true=f"Both mentions refer to the same {noun}.",
                    false=f"The mentions refer to two different {plural(noun)}.",
                ),
            )
            for qid, (i, j) in missing.items()
        }
        ids = list(missing)
        batch = run.clef_batch()
        chunks = [ids[k : k + batch] for k in range(0, len(ids), batch)]

        async def ask(chunk: list[str]) -> dict[str, object]:
            shown = {index: mentions[index] for qid in chunk for index in missing[qid]}
            return dict(
                await run.call_clef(
                    {qid: questions[qid] for qid in chunk},
                    _state(run, source, dict(sorted(shown.items())), share=0.75),
                    meta=CallMeta(kind="coref", schema=schema, scope=scope),
                )
            )

        parts = await gather(*(ask(chunk) for chunk in chunks))
        answers = {qid: answer for part in parts for qid, answer in part.items()}
        for qid, (i, j) in missing.items():
            answer = answers.get(qid)
            # a question that came back unanswered stays unknown
            if isinstance(answer, NoulAnswer):
                cache[ordered_pair(mentions[i], mentions[j])] = answer.noul
    return {
        (i, j): cache[key]
        for i in range(n)
        for j in range(i + 1, n)
        if (key := ordered_pair(mentions[i], mentions[j])) in cache
    }


def _cluster_entry(
    members: Sequence[Mention], *, spans: bool = True
) -> dict[str, object]:
    """A cluster as the incremental state shows it: its label and up to 3 mentions.

    ``spans=False`` leaves the offsets out (the input was narrowed, so they would
    point at the wrong text).
    """
    shown = list(members) if len(members) <= 3 else [*members[:2], members[-1]]
    return {
        "label": canonical_label([m.text for m in members]),
        "mentions": [
            {"text": m.text, "span": list(m.span)}
            if m.span and spans
            else {"text": m.text}
            for m in shown
        ],
    }


def _decide(
    answer: object, options: Mapping[str, object], threshold: float
) -> int | None:
    """The cluster a choice answer joins, or ``None`` for a new one."""
    if not isinstance(answer, ChoiceAnswer) or answer.choice == NEW:
        return None
    chosen = answer.choice
    if chosen in options and answer.probabilities.get(chosen, 0.0) >= threshold:
        return int(chosen[1:])
    return None


def _lookup_in(
    pairs: Mapping[tuple[int, int], float], mentions: Sequence[Mention]
) -> Callable[[int, int], float]:
    """Pair probabilities for clustering; an unanswered pair of identical words is 1.

    A pair that is not in ``pairs`` was not asked (it is implausible) or its
    question got no answer (the request was refused). Two mentions with the same
    words are always asked, so for them a missing pair is an unanswered one, and the
    same words are the same entity rather than two duplicates.
    """

    def lookup(i: int, j: int) -> float:
        known = pairs.get((i, j))
        if known is not None:
            return known
        same = normalise(mentions[i].text) == normalise(mentions[j].text)
        return 1.0 if same else 0.0

    return lookup


def _options_within_budget(
    run: Run,
    noun: str,
    batch: Sequence[int],
    base: Mapping[str, object],
    described: Mapping[str, Mapping[str, object]],
) -> int:
    """How many of the earlier entities (most recent first) one request can offer.

    A choice question lists every option, and a request asks one question per
    mention of the batch, so an option costs once in ``state.clusters`` and once per
    question: 16 questions over 254 entities are 4,000 options. The count is the
    longest prefix of ``described`` that keeps the whole request, state and
    questions, within ``Run.clef_budget``.
    """
    skeleton = {
        f"assign.m{k}": ChoiceQuestion(
            type="choice",
            instructions=(
                f"Which {noun} does `mentions.m{k}` refer to? Pick the earlier "
                f"{noun} it names (see `clusters`), or {NEW} if it is a different "
                f"{noun}."
            ),
            criteria={"c0": "x", NEW: f"A {noun} that none of the others is."},
        )
        for k in batch
    }
    used = estimate_request_tokens({**base, "clusters": {}}, skeleton)
    budget = run.clef_budget()
    kept = 0
    for option, entry in described.items():
        cost = estimate_json_tokens({option: entry}) + len(batch) * (
            estimate_json_tokens({option: entry["label"]}) + CHOICE_OPTION_TOKENS * 1.1
        )
        if used + cost > budget:
            break
        used += cost
        kept += 1
    return kept


async def _incremental(
    run: Run,
    *,
    source: str,
    mentions: Sequence[Mention],
    noun: str,
    schema: str,
    cache: PairCache,
    scope: str = "",
) -> list[list[int]]:
    """Assign the mentions, a batch at a time, to a cluster so far or a new one."""
    clusters: list[list[int]] = [[0]] if mentions else []
    capped_by_budget = False
    capped_by_count = False
    budget = run.clef_budget()
    position = 1
    while position < len(mentions):
        size = _BATCH
        while True:
            batch = list(range(position, min(len(mentions), position + size)))
            base = _state(
                run, source, {k: mentions[k] for k in batch}, share=_INPUT_SHARE
            )
            narrowed = base["input"] != source
            # the most recently mentioned entities are the likeliest to be named again
            recent = sorted(range(len(clusters)), key=lambda c: max(clusters[c]))
            candidates = recent[-(_MAX_OPTIONS - 1) :][::-1]
            described_all = {
                f"c{c}": _cluster_entry(
                    [mentions[i] for i in clusters[c]], spans=not narrowed
                )
                for c in candidates
            }
            keep = _options_within_budget(run, noun, batch, base, described_all)
            if keep >= min(len(candidates), _MIN_OPTIONS) or len(batch) == 1:
                break
            size = max(1, len(batch) // 2)  # fewer questions, each with more options
        position = batch[-1] + 1
        capped_by_count = capped_by_count or len(clusters) > len(candidates)
        while True:
            shown = sorted(candidates[:keep])
            described = {f"c{c}": described_all[f"c{c}"] for c in shown}
            state = {**base, "clusters": described}
            criteria: dict[str, str | None] = {
                option: str(entry["label"]) for option, entry in described.items()
            }
            criteria[NEW] = f"A {noun} that none of the others is."
            questions = {
                f"assign.m{k}": ChoiceQuestion(
                    type="choice",
                    instructions=(
                        f"Which {noun} does `mentions.m{k}` refer to? Pick the "
                        f"earlier {noun} it names (see `clusters`), or {NEW} if it "
                        f"is a different {noun}."
                    ),
                    criteria=criteria,
                )
                for k in batch
            }
            # the per-option sum above is an approximation: check the whole request
            if keep == 0 or estimate_request_tokens(state, questions) <= budget:
                break
            keep = min(keep - 1, int(keep * 0.95))
        capped_by_budget = capped_by_budget or keep < len(candidates)
        # a choice needs an option beside `__new__`; with none that fits the budget
        # the mentions of the batch start entities of their own
        answers = (
            await run.call_clef(
                questions,
                state,
                meta=CallMeta(
                    kind="coref",
                    schema=schema,
                    field=f"m{batch[0]}-m{batch[-1]}",
                    scope=scope,
                ),
            )
            if shown
            else {}
        )
        fresh: list[int] = []
        joins: dict[int, list[int]] = {}
        twins: dict[int, list[int]] = {}  # unanswered repeats of a fresh mention
        first_of: dict[str, int] = {}
        for k in batch:
            answer = answers.get(f"assign.m{k}")
            target = _decide(answer, criteria, run.coref.merge_threshold)
            if target is None and answer is None:
                # no answer (the request was refused, or nothing fitted the budget):
                # the same words are the same entity rather than a duplicate
                text = normalise(mentions[k].text)
                target = next(
                    (
                        c
                        for c, members in enumerate(clusters)
                        if any(normalise(mentions[i].text) == text for i in members)
                    ),
                    None,
                )
                if target is None and text in first_of:
                    twins.setdefault(first_of[text], []).append(k)
                    continue
                if target is None:
                    first_of[text] = k
            if target is None:
                fresh.append(k)
            else:
                joins.setdefault(target, []).append(k)
        for target, members in joins.items():
            clusters[target].extend(members)
        # the mentions that start a cluster in this batch may be one entity
        if len(fresh) == 1:
            clusters.append([fresh[0]])
        elif fresh:
            sub = [mentions[k] for k in fresh]
            pairs = await _pairwise(
                run,
                source=source,
                mentions=sub,
                noun=noun,
                schema=schema,
                cache=cache,
                scope=scope,
            )
            groups = agglomerate(
                len(sub), _lookup_in(pairs, sub), run.coref.merge_threshold
            )
            clusters.extend([fresh[i] for i in group] for group in groups)
        for first, repeats in twins.items():
            next(c for c in clusters if first in c).extend(repeats)
    if capped_by_budget:
        run.warn(
            f"coref_options_capped: so many entities of {schema} were found that a "
            "batch of mentions could not be matched against all of them within the "
            "Clef request budget (limits.clef_state_token_budget); each batch was "
            "matched against the most recently mentioned ones, and a mention of an "
            "earlier one may have started a new entity"
        )
    elif capped_by_count:
        run.warn(
            f"coref_options_capped: more than {_MAX_OPTIONS - 1} entities of "
            f"{schema} were found; each batch of mentions was matched against the "
            "most recently mentioned ones only"
        )
    return [sorted(c) for c in sorted(clusters, key=min)]


async def cluster_mentions(
    run: Run,
    *,
    source: str,
    mentions: Sequence[Mention],
    noun: str,
    schema: str,
    cache: PairCache | None = None,
    scope: str = "",
) -> Clustering:
    """Group mentions that name the same entity.

    Args:
        run: The invoke's context (it carries ``run.coref``).
        source: The input the mentions were found in.
        mentions: The mentions in input order.
        noun: What the entities are called in questions (``"Person"``).
        schema: Schema name, for the call metadata.
        cache: Pair probabilities from an earlier call on a list that shares
            mentions with this one (the broadened re-discovery): those pairs are
            not asked again, and new answers are added.
        scope: ``CallMeta.scope`` of the calls (the parent and field of a nested
            list).

    Returns:
        The clusters and the probabilities of the pairs that were asked.
    """
    n = len(mentions)
    if n == 0:
        return Clustering((), {}, "pairwise")
    if n == 1:
        return Clustering(((0,),), {}, "pairwise")
    config = run.coref
    shared: PairCache = cache if cache is not None else {}
    strategy = config.strategy
    if strategy == "auto":
        strategy = "pairwise" if n <= config.max_pairwise_mentions else "incremental"
    if strategy == "incremental":
        groups = await _incremental(
            run,
            source=source,
            mentions=mentions,
            noun=noun,
            schema=schema,
            cache=shared,
            scope=scope,
        )
        return Clustering(tuple(tuple(g) for g in groups), {}, "incremental")
    pairs = await _pairwise(
        run,
        source=source,
        mentions=mentions,
        noun=noun,
        schema=schema,
        cache=shared,
        scope=scope,
    )
    groups = agglomerate(n, _lookup_in(pairs, mentions), config.merge_threshold)
    return Clustering(tuple(tuple(g) for g in groups), pairs, "pairwise")
