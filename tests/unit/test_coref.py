"""Mention clustering: average linkage, the 64-question split, incremental mode."""

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

import pytest

from trustcall_system_one import CallMeta, CorefConfig, Mention
from trustcall_system_one._coref import (
    NEW,
    agglomerate,
    canonical_label,
    cluster_mentions,
    normalise,
    ordered_pair,
)
from trustcall_system_one._run import Run
from trustcall_system_one.testing import (
    FakeTransport,
    choice,
    clef_response,
    noul,
)

Body = dict[str, Any]

# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_normalise_folds_case_spacing_and_edge_punctuation() -> None:
    assert normalise("  Emma   THOMPSON, ") == "emma thompson"
    assert normalise("Emma's") == "emma's"


def test_the_canonical_label_is_the_mention_with_the_most_tokens() -> None:
    assert canonical_label(["Emma", "Emma Thompson", "Emma"]) == "Emma Thompson"


def test_among_equal_token_counts_the_most_frequent_form_wins() -> None:
    assert canonical_label(["Em", "Emma", "emma", "Emmy", "EMMA"]) == "Emma"


def test_among_equal_counts_and_frequencies_the_earliest_wins() -> None:
    assert canonical_label(["Emmy", "Emma"]) == "Emmy"
    assert canonical_label(["Dr Who", "Mr Lee"]) == "Dr Who"


def test_a_cluster_needs_a_mention() -> None:
    with pytest.raises(ValueError, match="at least one"):
        canonical_label([])


def table(values: dict[tuple[int, int], float]) -> Callable[[int, int], float]:
    return lambda i, j: values.get((i, j), 0.0)


def test_an_inconsistent_triangle_is_settled_by_the_average() -> None:
    # A~B 0.9 and B~C 0.9 but A~C 0.05: {A, B} then averages 0.475 against C
    p = table({(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.05})
    assert agglomerate(3, p, 0.5) == [[0, 1], [2]]


def test_a_consistent_chain_becomes_one_cluster() -> None:
    p = table({(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.8})
    assert agglomerate(3, p, 0.5) == [[0, 1, 2]]


def test_ties_go_to_the_earlier_mention() -> None:
    # 0~1 and 1~2 tie; the earlier pair merges first, then 2 averages 0.45 < 0.5
    p = table({(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.0})
    assert agglomerate(3, p, 0.5) == [[0, 1], [2]]
    # the other tie-break would have given [[0], [1, 2]]


def test_the_threshold_is_inclusive_and_nothing_is_merged_below_it() -> None:
    assert agglomerate(2, table({(0, 1): 0.5}), 0.5) == [[0, 1]]
    assert agglomerate(2, table({(0, 1): 0.49}), 0.5) == [[0], [1]]
    assert agglomerate(0, table({}), 0.5) == []


# --------------------------------------------------------------------------- #
# Clustering with a fake Clef
# --------------------------------------------------------------------------- #

SOURCE = "Ann met Bob. Ann saw Cy. Then Bob left."


def mentions_of(source: str, *texts: str) -> list[Mention]:
    """The mentions (in the order given), each at its next occurrence."""
    found, cursor = [], 0
    for text in texts:
        at = source.index(text, cursor)
        found.append(Mention(text, (at, at + len(text))))
        cursor = at + len(text)
    return found


def run_with(handler: Callable[[str, Body, CallMeta], Any], **coref: Any) -> Run:
    return Run(transport=FakeTransport(handler), coref=CorefConfig(**coref))


def by_span(
    answer: Callable[[tuple[int, int], tuple[int, int]], float],
) -> Callable[[str, Body, CallMeta], Any]:
    """A coref handler that answers pair questions from the two mentions' spans."""

    def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        assert meta.kind == "coref"
        mentions = payload["state"]["mentions"]
        answers = {}
        for qid in payload["questions"]:
            first, second = re.findall(r"m\d+", qid)
            answers[qid] = noul(
                answer(tuple(mentions[first]["span"]), tuple(mentions[second]["span"]))
            )
        return clef_response(answers, model=payload["model"])

    return handler


async def test_the_pairs_share_one_structured_state_and_ids() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann", "Cy", "Bob")
    ann, bob = {0, 2}, {1, 4}

    def who(span: tuple[int, int]) -> str:
        index = next(i for i, m in enumerate(mentions) if m.span == span)
        return "ann" if index in ann else "bob" if index in bob else "cy"

    handler = by_span(lambda a, b: 0.96 if who(a) == who(b) else 0.03)
    run = run_with(handler)
    clustering = await cluster_mentions(
        run, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    assert clustering.clusters == ((0, 2), (1, 4), (3,))
    assert clustering.strategy == "pairwise"
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    ((_, payload, meta),) = transport.calls  # 10 pairs: one request
    assert meta == CallMeta(kind="coref", schema="Person")
    assert sorted(payload["questions"]) == sorted(
        f"pair.m{i}.m{j}" for i in range(5) for j in range(i + 1, 5)
    )
    state = payload["state"]
    assert state["input"] == SOURCE
    assert sorted(state["mentions"]) == ["m0", "m1", "m2", "m3", "m4"]
    assert state["mentions"]["m2"]["text"] == "Ann"
    assert state["mentions"]["m2"]["span"] == [13, 16]
    assert state["mentions"]["m2"]["context"] == SOURCE[0 : 16 + 120]
    first = payload["questions"]["pair.m0.m2"]
    assert first["instructions"] == (
        "Do `mentions.m0` and `mentions.m2` refer to the same Person?"
    )
    assert set(first["criteria"]) == {"true", "false"}
    assert clustering.pairs[(0, 2)] == 0.96 and clustering.pairs[(0, 1)] == 0.03


async def test_the_context_is_limited_to_context_chars_each_side() -> None:
    text = "x" * 300 + "Ann" + "y" * 300 + "Ann" + "z" * 300
    mentions = mentions_of(text, "Ann", "Ann")
    run = run_with(by_span(lambda a, b: 0.9), context_chars=10)
    await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    context = transport.calls[0][1]["state"]["mentions"]["m0"]["context"]
    assert context == "x" * 10 + "Ann" + "y" * 10


async def test_an_inconsistent_triangle_through_the_clef_questions() -> None:
    text = "Ann Lee. Ann. Anne Lee."
    mentions = mentions_of(text, "Ann Lee", "Ann", "Anne Lee")
    # A~B 0.9, B~C 0.9, A~C 0.05
    table_ = {(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.05}

    def answer(a: tuple[int, int], b: tuple[int, int]) -> float:
        index = {m.span: i for i, m in enumerate(mentions)}
        return table_[(index[a], index[b])]

    run = run_with(by_span(answer))
    clustering = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert clustering.clusters == ((0, 1), (2,))


async def test_two_distinct_emmas_stay_two_entities_despite_the_same_text() -> None:
    text = "Emma sings. Emma Thompson dances. Emma paints."
    mentions = mentions_of(text, "Emma", "Emma Thompson", "Emma")
    # the first and third "Emma" are two different people; "Emma Thompson" is the
    # first one
    truth = {mentions[0].span: "a", mentions[1].span: "a", mentions[2].span: "b"}
    run = run_with(by_span(lambda a, b: 0.96 if truth[a] == truth[b] else 0.03))
    clustering = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert clustering.clusters == ((0, 1), (2,))
    assert mentions[0].text == mentions[2].text  # the same surface form


async def test_more_than_64_questions_are_split_and_run_concurrently() -> None:
    names = [f"Pn{i:02d}" for i in range(12)]  # 66 pairs
    text = " ".join(names)
    mentions = mentions_of(text, *names)
    in_flight = 0
    peak = 0

    async def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return clef_response(
            {qid: noul(0.03) for qid in payload["questions"]}, model=payload["model"]
        )

    run = run_with(handler)
    clustering = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert len(clustering.clusters) == 12
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    sizes = sorted(len(payload["questions"]) for _, payload, _ in transport.calls)
    assert sizes == [2, 64]
    assert peak == 2  # both requests were in flight together
    states = [json_state(payload) for _, payload, _ in transport.calls]
    assert states[0] == states[1]  # one shared structured state
    ids = [qid for _, payload, _ in transport.calls for qid in payload["questions"]]
    assert len(set(ids)) == 66


def json_state(payload: Body) -> str:
    return json.dumps(payload["state"], sort_keys=True)


async def test_pairs_already_asked_are_not_asked_again() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann", "Cy", "Bob")
    run = run_with(by_span(lambda a, b: 0.03))
    cache: dict[tuple[Mention, Mention], float] = {}
    await cluster_mentions(
        run,
        source=SOURCE,
        mentions=mentions[:3],
        noun="Person",
        schema="Person",
        cache=cache,
    )
    assert len(cache) == 3
    await cluster_mentions(
        run,
        source=SOURCE,
        mentions=mentions,
        noun="Person",
        schema="Person",
        cache=cache,
    )
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    second = sorted(transport.calls[1][1]["questions"])
    # only the pairs that involve the two new mentions (m3 and m4)
    assert second == sorted(
        f"pair.m{i}.m{j}" for i in range(5) for j in range(i + 1, 5) if j >= 3
    )
    assert ordered_pair(mentions[2], mentions[0]) == (mentions[0], mentions[2])


async def test_zero_and_one_mention_cost_no_request() -> None:
    run = run_with(by_span(lambda a, b: 1.0))
    empty = await cluster_mentions(
        run, source=SOURCE, mentions=[], noun="Person", schema="Person"
    )
    single = await cluster_mentions(
        run,
        source=SOURCE,
        mentions=mentions_of(SOURCE, "Ann"),
        noun="Person",
        schema="Person",
    )
    assert empty.clusters == () and single.clusters == ((0,),)
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    assert transport.calls == []


# --------------------------------------------------------------------------- #
# Incremental assignment
# --------------------------------------------------------------------------- #


def assigner(
    truth: dict[tuple[int, int], str], p: float = 0.95
) -> Callable[[str, Body, CallMeta], Any]:
    """An ``assign`` handler: the cluster that already holds the same person."""

    def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        assert meta.kind == "coref"
        ((qid, question),) = payload["questions"].items()
        assert qid.startswith("assign.m")
        mention = payload["state"]["mention"]
        who = truth[tuple(mention["span"])]
        options = list(question["criteria"])
        picked = NEW
        for option, text in question["criteria"].items():
            if option == NEW:
                continue
            spans = [(int(a), int(b)) for a, b in re.findall(r"\((\d+)-(\d+)\)", text)]
            if any(truth[s] == who for s in spans):
                picked = option
        rest = (1 - p) / (len(options) - 1)
        probabilities = {o: (p if o == picked else rest) for o in options}
        return clef_response(
            {qid: choice(picked, probabilities)}, model=payload["model"]
        )

    return handler


async def test_above_the_threshold_the_mentions_are_assigned_one_by_one() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann", "Cy", "Bob")
    truth = {
        m.span: who
        for m, who in zip(mentions, ["ann", "bob", "ann", "cy", "bob"], strict=True)
        if m.span
    }
    run = run_with(assigner(truth), max_pairwise_mentions=4)  # 5 mentions > 4
    clustering = await cluster_mentions(
        run, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    assert clustering.strategy == "incremental"
    assert clustering.clusters == ((0, 2), (1, 4), (3,))
    assert clustering.pairs == {}
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    metas = [meta for _, _, meta in transport.calls]
    assert [m.field for m in metas] == ["m1", "m2", "m3", "m4"]  # in order
    payload = transport.calls[3][1]  # the last mention, against three clusters
    assert list(payload["questions"]["assign.m4"]["criteria"]) == [
        "c0",
        "c1",
        "c2",
        NEW,
    ]
    assert payload["state"]["mention"]["text"] == "Bob"
    assert "Ann" in payload["questions"]["assign.m4"]["criteria"]["c0"]
    assert payload["questions"]["assign.m4"]["instructions"].startswith(
        "Which Person does `mention` refer to?"
    )


async def test_the_strategy_can_be_forced_either_way() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann")
    truth = {m.span: who for m, who in zip(mentions, "aba", strict=True) if m.span}
    forced = run_with(assigner(truth), strategy="incremental")
    result = await cluster_mentions(
        forced, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    assert result.clusters == ((0, 2), (1,)) and result.strategy == "incremental"

    pairwise = run_with(
        by_span(lambda a, b: 0.96 if truth[a] == truth[b] else 0.03),
        strategy="pairwise",
        max_pairwise_mentions=2,  # three mentions are above it, but it was forced
    )
    again = await cluster_mentions(
        pairwise, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    assert again.clusters == ((0, 2), (1,)) and again.strategy == "pairwise"


async def test_an_assignment_below_the_threshold_starts_a_new_cluster() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Ann")

    def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        ((qid, _),) = payload["questions"].items()
        # the cluster is the likeliest option, but only at 0.4
        answer = choice("c0", {"c0": 0.4, NEW: 0.3})
        return clef_response({qid: answer}, model=payload["model"])

    run = run_with(handler, strategy="incremental")
    result = await cluster_mentions(
        run, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    assert result.clusters == ((0,), (1,))


async def test_the_options_are_capped_at_255_with_a_warning() -> None:
    names = [f"Q{i:03d}" for i in range(258)]
    text = " ".join(names)
    mentions = mentions_of(text, *names)
    sizes: list[int] = []

    def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        ((qid, question),) = payload["questions"].items()
        options = list(question["criteria"])
        sizes.append(len(options))
        answer = choice(NEW, {o: (1.0 if o == NEW else 0.0) for o in options})
        return clef_response({qid: answer}, model=payload["model"])

    run = run_with(handler, strategy="incremental")
    result = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert len(result.clusters) == 258  # every mention is a new entity
    assert max(sizes) == 255  # 254 clusters and __new__
    assert any("coref_options_capped" in w for w in run.warnings)
