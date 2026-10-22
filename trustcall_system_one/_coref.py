"""Mention clustering: which mentions of an input name the same entity.

The unit is the *mention* (a piece of the input with its span), never the
surface form: two people who are both called "Emma" are two mentions with the
same text and may be two entities.

Two strategies, picked by ``CorefConfig.strategy``:

* ``pairwise``: one Clef ``noul`` per pair of mentions, all pairs sharing ONE
  structured state ``{"input", "mentions": {"m0": {"text", "span", "context"}}}``
  (at most 64 questions per request, the requests run concurrently). The answers
  are clustered with average linkage, which settles inconsistent triangles
  (A~B 0.9, B~C 0.9, A~C 0.05 stays two entities because the average over
  ``{A, B}`` and ``{C}`` is below the threshold).
* ``incremental``: the mentions are walked in order and each is assigned with one
  Clef ``choice`` over the clusters so far plus ``__new__``. It needs one request
  per mention instead of one per pair, so ``auto`` switches to it for long lists.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from trustcall_system_one._run import Run
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
    "normalise",
    "ordered_pair",
]

NEW = "__new__"
"""The option of an incremental assignment that starts a new cluster."""
_MAX_OPTIONS = 255
_FLOOR = 1e-12

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


def canonical_label(texts: Sequence[str]) -> str:
    """The name a cluster of mentions is called by.

    The mention with the most tokens wins ("Emma Thompson" over "Emma"); among
    equals the normalised form that occurs most often, then the earliest.

    Args:
        texts: The mention texts of one cluster, in input order.

    Returns:
        The chosen mention text, as written.

    Raises:
        ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("a cluster has at least one mention")
    counts: dict[str, int] = {}
    for text in texts:
        counts[normalise(text)] = counts.get(normalise(text), 0) + 1
    best = min(
        range(len(texts)),
        key=lambda i: (
            -len(texts[i].split()),
            -counts[normalise(texts[i])],
            i,
        ),
    )
    return texts[best]


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


async def _pairwise(
    run: Run,
    *,
    source: str,
    mentions: Sequence[Mention],
    noun: str,
    schema: str,
    cache: PairCache,
) -> dict[tuple[int, int], float]:
    """Ask about every pair of mentions that the cache does not know yet."""
    n = len(mentions)
    missing: dict[str, tuple[int, int]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if ordered_pair(mentions[i], mentions[j]) not in cache:
                missing[_pair_id(i, j)] = (i, j)
    if missing:
        questions = {
            qid: NoulQuestion(
                type="noul",
                instructions=(
                    f"Do `mentions.m{i}` and `mentions.m{j}` refer to the same {noun}?"
                ),
                criteria=NoulCriteria(
                    true=f"Both mentions refer to the same {noun}.",
                    false=f"The mentions refer to two different {noun}s.",
                ),
            )
            for qid, (i, j) in missing.items()
        }
        state = run.clef_state(
            source,
            spans=_spans(mentions),
            extra={
                "mentions": {
                    f"m{i}": mention_entry(source, m, run.coref.context_chars)
                    for i, m in enumerate(mentions)
                }
            },
        )
        answers = await run.call_clef(
            questions, state, meta=CallMeta(kind="coref", schema=schema)
        )
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
    cited = ", ".join(
        f'"{m.text}"' + (f" ({m.span[0]}-{m.span[1]})" if m.span else "")
        for m in members
    )
    return f"{canonical_label([m.text for m in members])}: mentioned as {cited}"


async def _incremental(
    run: Run,
    *,
    source: str,
    mentions: Sequence[Mention],
    noun: str,
    schema: str,
) -> list[list[int]]:
    """Assign the mentions one by one to a cluster so far, or to a new one."""
    clusters: list[list[int]] = []
    capped = False
    for k, mention in enumerate(mentions):
        if not clusters:
            clusters.append([k])
            continue
        shown = list(range(len(clusters)))
        if len(shown) > _MAX_OPTIONS - 1:  # the most recent clusters fit
            shown = shown[-(_MAX_OPTIONS - 1) :]
            capped = True
        criteria: dict[str, str | None] = {
            f"c{c}": _describe([mentions[i] for i in clusters[c]]) for c in shown
        }
        criteria[NEW] = (
            f"A {noun} that none of the others is, one not mentioned before."
        )
        question = ChoiceQuestion(
            type="choice",
            instructions=(
                f"Which {noun} does `mention` refer to? Pick the earlier {noun} it "
                f"names, or {NEW} if it is a different {noun}."
            ),
            criteria=criteria,
        )
        qid = f"assign.m{k}"
        state = run.clef_state(
            source,
            spans=_spans([mention]),
            extra={"mention": mention_entry(source, mention, run.coref.context_chars)},
        )
        answers = await run.call_clef(
            {qid: question},
            state,
            meta=CallMeta(kind="coref", schema=schema, field=f"m{k}"),
        )
        answer = answers.get(qid)
        target: int | None = None
        if isinstance(answer, ChoiceAnswer) and answer.choice != NEW:
            chosen = answer.choice
            if (
                chosen in criteria
                and answer.probabilities.get(chosen, 0.0) >= run.coref.merge_threshold
            ):
                target = int(chosen[1:])
        if target is None:
            clusters.append([k])
        else:
            clusters[target].append(k)
    if capped:
        run.warn(
            f"coref_options_capped: more than {_MAX_OPTIONS - 1} entities of "
            f"{schema} were found; each mention was matched against the most "
            "recent ones only"
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
    strategy = config.strategy
    if strategy == "auto":
        strategy = "pairwise" if n <= config.max_pairwise_mentions else "incremental"
    if strategy == "incremental":
        groups = await _incremental(
            run, source=source, mentions=mentions, noun=noun, schema=schema
        )
        return Clustering(tuple(tuple(g) for g in groups), {}, "incremental")
    pairs = await _pairwise(
        run,
        source=source,
        mentions=mentions,
        noun=noun,
        schema=schema,
        cache=cache if cache is not None else {},
    )
    groups = agglomerate(n, lambda i, j: pairs[(i, j)], config.merge_threshold)
    return Clustering(tuple(tuple(g) for g in groups), pairs, "pairwise")
