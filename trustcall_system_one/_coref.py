"""Mention clustering: which mentions of an input name the same entity.

The unit is the *mention* (a piece of the input with its span), never the
surface form: two people who are both called "Emma" are two mentions with the
same text and may be two entities.

Two strategies, picked by ``CorefConfig.strategy``:

* ``pairwise``: one Clef ``noul`` per PLAUSIBLE pair of mentions (they share a
  name token, one is a short form or the initials of the other, or one side is a
  descriptive reference such as "my cousin"; every other pair is taken as 0
  without asking). The questions go out in requests of at most 64, each with a
  structured state ``{"input", "mentions": {"m0": {"text", "span", "context"}}}``
  that holds only the mentions its questions name; the requests run
  concurrently. The answers are clustered with average linkage, which settles
  inconsistent triangles (A~B 0.9, B~C 0.9, A~C 0.05 stays two entities because
  the average over ``{A, B}`` and ``{C}`` is below the threshold).
* ``incremental``: the mentions are walked in order, in batches of 16: each batch
  is one request of Clef choices over the clusters that existed when it started
  (plus ``__new__``), and the mentions of one batch that start new clusters are
  then compared with each other pairwise. It needs about a request per 16
  mentions, so ``auto`` switches to it for long lists.
"""

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from trustcall_system_one._run import CLEF_BATCH, Run, gather
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
    "canonical_label",
    "cluster_mentions",
    "mention_entry",
    "name_part",
    "normalise",
    "ordered_pair",
    "plausible",
    "plural",
]

logger = logging.getLogger(__name__)

NEW = "__new__"
"""The option of an incremental assignment that starts a new cluster."""
_MAX_OPTIONS = 255
_BATCH = 16
"""Mentions an incremental request assigns at once."""
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
            their indexes (smaller first); empty for the incremental strategy.
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


def _name_tokens(text: str) -> list[str]:
    return [t for t in normalise(text).replace("-", " ").split() if t]


def _initials(tokens: Sequence[str]) -> str:
    return "".join(t[0] for t in tokens)


def plausible(first: str, second: str) -> bool:
    """Whether two mentions are worth asking about at all.

    They are when they share a name token ("Emma" and "Emma Thompson"), when one
    token is the start of another ("Ann" and "Anne", "Em" and "Emma"), when one is
    the initials of the other ("JS", "J. Smith" and "John Smith"), or when either
    side only describes its entity ("my cousin" could be anyone). Everything else,
    "Emma Thompson" and "Michael Chen", is not asked: the answer is no.

    Args:
        first: One mention as written.
        second: The other.
    """
    if name_part(first) is None or name_part(second) is None:
        return True
    a, b = _name_tokens(first), _name_tokens(second)
    if set(a) & set(b):
        return True
    for x in a:
        for y in b:
            if len(x) >= 2 and len(y) >= 2 and (x.startswith(y) or y.startswith(x)):
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
    run: Run, source: str, shown: Mapping[int, Mention], key: str = "mentions"
) -> dict[str, object]:
    """A coreference state holding only the mentions ``shown`` (by index)."""
    return run.clef_state(
        source,
        spans=_spans(list(shown.values())),
        extra={
            key: {
                f"m{i}": mention_entry(source, m, run.coref.context_chars)
                for i, m in shown.items()
            }
        },
    )


async def _pairwise(
    run: Run,
    *,
    source: str,
    mentions: Sequence[Mention],
    noun: str,
    schema: str,
    cache: PairCache,
) -> dict[tuple[int, int], float]:
    """Ask about every plausible pair of mentions that the cache does not know."""
    n = len(mentions)
    missing: dict[str, tuple[int, int]] = {}
    unasked = 0
    for i in range(n):
        for j in range(i + 1, n):
            key = ordered_pair(mentions[i], mentions[j])
            if key in cache:
                continue
            if plausible(mentions[i].text, mentions[j].text):
                missing[_pair_id(i, j)] = (i, j)
            else:
                cache[key] = 0.0
                unasked += 1
    if unasked:
        logger.debug(
            "coref: %d of %d pairs of %s mentions share no name token, short form "
            "or description and are taken as different entities without asking",
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
        chunks = [ids[k : k + CLEF_BATCH] for k in range(0, len(ids), CLEF_BATCH)]

        async def ask(chunk: list[str]) -> dict[str, object]:
            shown = {index: mentions[index] for qid in chunk for index in missing[qid]}
            return dict(
                await run.call_clef(
                    {qid: questions[qid] for qid in chunk},
                    _state(run, source, dict(sorted(shown.items()))),
                    meta=CallMeta(kind="coref", schema=schema),
                )
            )

        parts = await gather(*(ask(chunk) for chunk in chunks))
        answers = {qid: answer for part in parts for qid, answer in part.items()}
        for qid, (i, j) in missing.items():
            answer = answers.get(qid)
            # a question that came back unanswered does not merge anything
            cache[ordered_pair(mentions[i], mentions[j])] = (
                answer.noul if isinstance(answer, NoulAnswer) else 0.0
            )
    return {
        (i, j): cache[ordered_pair(mentions[i], mentions[j])]
        for i in range(n)
        for j in range(i + 1, n)
    }


def _describe(members: Sequence[Mention]) -> str:
    shown = list(members)
    gap = False
    if len(shown) > 8:
        shown, gap = [*shown[:4], *shown[-4:]], True
    parts = [
        f'"{m.text}"' + (f" ({m.span[0]}-{m.span[1]})" if m.span else "") for m in shown
    ]
    if gap:
        parts.insert(4, "...")
    return f"{canonical_label([m.text for m in members])}: mentioned as " + ", ".join(
        parts
    )


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


def _lookup_in(pairs: Mapping[tuple[int, int], float]) -> Callable[[int, int], float]:
    return lambda i, j: pairs[(i, j)]


async def _incremental(
    run: Run,
    *,
    source: str,
    mentions: Sequence[Mention],
    noun: str,
    schema: str,
    cache: PairCache,
) -> list[list[int]]:
    """Assign the mentions, a batch at a time, to a cluster so far or a new one."""
    clusters: list[list[int]] = [[0]] if mentions else []
    capped = False
    position = 1
    while position < len(mentions):
        batch = list(range(position, min(len(mentions), position + _BATCH)))
        position += _BATCH
        shown = list(range(len(clusters)))
        if len(shown) > _MAX_OPTIONS - 1:  # the most recently mentioned ones fit
            shown = sorted(shown, key=lambda c: max(clusters[c]))[-(_MAX_OPTIONS - 1) :]
            shown.sort()
            capped = True
        criteria: dict[str, str | None] = {
            f"c{c}": _describe([mentions[i] for i in clusters[c]]) for c in shown
        }
        criteria[NEW] = (
            f"A {noun} that none of the others is, one not mentioned before."
        )
        questions = {
            f"assign.m{k}": ChoiceQuestion(
                type="choice",
                instructions=(
                    f"Which {noun} does `mentions.m{k}` refer to? Pick the earlier "
                    f"{noun} it names, or {NEW} if it is a different {noun}."
                ),
                criteria=criteria,
            )
            for k in batch
        }
        answers = await run.call_clef(
            questions,
            _state(run, source, {k: mentions[k] for k in batch}),
            meta=CallMeta(
                kind="coref", schema=schema, field=f"m{batch[0]}-m{batch[-1]}"
            ),
        )
        fresh: list[int] = []
        joins: dict[int, list[int]] = {}
        for k in batch:
            target = _decide(
                answers.get(f"assign.m{k}"), criteria, run.coref.merge_threshold
            )
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
                run, source=source, mentions=sub, noun=noun, schema=schema, cache=cache
            )
            groups = agglomerate(len(sub), _lookup_in(pairs), run.coref.merge_threshold)
            clusters.extend([fresh[i] for i in group] for group in groups)
    if capped:
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

    Returns:
        The clusters and, for the pairwise strategy, every pair probability.
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
        )
        return Clustering(tuple(tuple(g) for g in groups), {}, "incremental")
    pairs = await _pairwise(
        run,
        source=source,
        mentions=mentions,
        noun=noun,
        schema=schema,
        cache=shared,
    )
    groups = agglomerate(n, lambda i, j: pairs[(i, j)], config.merge_threshold)
    return Clustering(tuple(tuple(g) for g in groups), pairs, "pairwise")
