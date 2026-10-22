"""Mention clustering: average linkage, the 64-question split, incremental mode."""

import asyncio
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
    has_context,
    name_part,
    near,
    nicknames,
    normalise,
    ordered_pair,
    plausible,
    plural,
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


def test_a_name_beats_a_description_in_the_label() -> None:
    assert canonical_label(["Emma", "My old college roommate"]) == "Emma"
    assert canonical_label(["my cousin", "Kim"]) == "Kim"
    assert canonical_label(["Emma", "My old college roommate", "Emma"]) == "Emma"


def test_the_name_inside_a_descriptive_mention_is_the_label() -> None:
    assert canonical_label(["her cousin Olivia Davis"]) == "Olivia Davis"
    assert canonical_label(["Olivia", "her cousin Olivia Davis"]) == "Olivia Davis"
    assert canonical_label(["Dr. Who", "the doctor"]) == "Dr. Who"


def test_a_cluster_of_only_descriptions_is_called_by_the_longest() -> None:
    assert canonical_label(["my cousin", "my old college roommate"]) == (
        "my old college roommate"
    )


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Emma", "Emma"),
        ("The Hague", "The Hague"),
        ("her cousin Olivia Davis", "Olivia Davis"),
        ("my neighbour", None),
        ("My old college roommate", None),
        ("cousin", None),
        ("", None),
    ],
)
def test_name_part(text: str, name: str | None) -> None:
    assert name_part(text) == name


@pytest.mark.parametrize(
    ("first", "second", "asked"),
    [
        ("Emma Thompson", "Emma", True),  # a shared token
        ("Ann", "Anne Lee", True),  # the start of a token
        ("JS", "John Smith", True),  # initials
        ("J. Smith", "John Smith", True),
        ("my cousin", "Michael Chen", True),  # a description could be anyone
        ("Emma Thompson", "Michael Chen", False),
        ("Sarah", "Olivia Davis", False),
        ("Ann", "Bob", False),
    ],
)
def test_only_plausible_pairs_are_asked(first: str, second: str, asked: bool) -> None:
    assert plausible(first, second) is asked
    assert plausible(second, first) is asked


@pytest.mark.parametrize(
    ("noun", "plural_"),
    [
        ("Person", "Persons"),
        ("Class", "Classes"),
        ("Company", "Companies"),
        ("Day", "Days"),
        ("Box", "Boxes"),
        ("Match", "Matches"),
        ("Wish", "Wishes"),
        ("Quiz", "Quizes"),
    ],
)
def test_plural(noun: str, plural_: str) -> None:
    assert plural(noun) == plural_


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


async def test_a_request_holds_only_the_mentions_its_questions_name() -> None:
    far = " ".join(["and so it went"] * 30)  # Bob Roe is nowhere near the Anns
    text = f"Ann Lee met Ann. Anne Lee came. {far} Bob Roe waved."
    mentions = mentions_of(text, "Ann Lee", "Ann", "Anne Lee", "Bob Roe")
    truth = {0: "a", 1: "a", 2: "a", 3: "b"}
    index = {m.span: i for i, m in enumerate(mentions)}
    run = run_with(
        by_span(lambda x, y: 0.96 if truth[index[x]] == truth[index[y]] else 0.03),
        all_pairs_below=0,  # only the plausible pairs
    )
    clustering = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert clustering.clusters == ((0, 1, 2), (3,))
    assert clustering.strategy == "pairwise"
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    ((_, payload, meta),) = transport.calls  # the Bob Roe pairs are not asked
    assert meta == CallMeta(kind="coref", schema="Person")
    assert sorted(payload["questions"]) == ["pair.m0.m1", "pair.m0.m2", "pair.m1.m2"]
    state = payload["state"]
    assert state["input"] == text
    assert sorted(state["mentions"]) == ["m0", "m1", "m2"]  # not m3
    assert state["mentions"]["m1"] == {
        "text": "Ann",
        "span": [12, 15],
        "context": text[0 : 15 + 120],
    }
    first = payload["questions"]["pair.m0.m1"]
    assert first["instructions"] == (
        "Do `mentions.m0` and `mentions.m1` refer to the same Person?"
    )
    assert set(first["criteria"]) == {"true", "false"}
    assert clustering.pairs[(0, 1)] == 0.96
    assert (0, 3) not in clustering.pairs  # not asked: nothing in common


async def test_pairs_that_cannot_be_the_same_entity_are_not_asked() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann", "Cy", "Bob")
    run = run_with(by_span(lambda a, b: 0.96), all_pairs_below=0)
    clustering = await cluster_mentions(
        run, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    ((_, payload, _),) = transport.calls
    assert sorted(payload["questions"]) == ["pair.m0.m2", "pair.m1.m4"]
    assert clustering.clusters == ((0, 2), (1, 4), (3,))


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
    text = " ".join(["Ann"] * 12)  # 66 pairs, every one plausible
    mentions = [Mention("Ann", (4 * i, 4 * i + 3)) for i in range(12)]
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
    by_size = {len(p["questions"]): p for _, p, _ in transport.calls}
    assert len(by_size[64]["state"]["mentions"]) == 12
    # the small request holds only the three mentions its two questions name
    assert sorted(by_size[2]["state"]["mentions"]) == ["m10", "m11", "m9"]
    ids = [qid for _, payload, _ in transport.calls for qid in payload["questions"]]
    assert len(set(ids)) == 66


async def test_pairs_already_asked_are_not_asked_again() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann", "Cy", "Bob")
    run = run_with(by_span(lambda a, b: 0.03), all_pairs_below=0)
    cache: dict[tuple[Mention, Mention], float] = {}
    await cluster_mentions(
        run,
        source=SOURCE,
        mentions=mentions[:3],
        noun="Person",
        schema="Person",
        cache=cache,
    )
    assert len(cache) == 1  # only the pair that was asked is known
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
    assert sorted(transport.calls[0][1]["questions"]) == ["pair.m0.m2"]
    # only the new plausible pair, the two Bobs
    assert sorted(transport.calls[1][1]["questions"]) == ["pair.m1.m4"]
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
    """A handler for ``assign`` batches (and, within a batch, ``pair`` questions)."""

    def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        assert meta.kind == "coref"
        mentions = payload["state"]["mentions"]
        answers = {}
        for qid, question in payload["questions"].items():
            if qid.startswith("pair."):
                a, b = re.findall(r"m\d+", qid)
                same = (
                    truth[tuple(mentions[a]["span"])]
                    == truth[tuple(mentions[b]["span"])]
                )
                answers[qid] = noul(0.96 if same else 0.03)
                continue
            assert qid.startswith("assign.m")
            who = truth[tuple(mentions[qid.removeprefix("assign.")]["span"])]
            options = list(question["criteria"])
            picked = NEW
            for option in question["criteria"]:
                if option == NEW:
                    continue
                held = payload["state"]["clusters"][option]["mentions"]
                if any(truth[tuple(m["span"])] == who for m in held):
                    picked = option
            rest = (1 - p) / (len(options) - 1)
            answers[qid] = choice(
                picked, {o: (p if o == picked else rest) for o in options}
            )
        return clef_response(answers, model=payload["model"])

    return handler


async def test_above_the_threshold_the_mentions_are_assigned_in_batches() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann", "Cy", "Bob")
    truth = {
        m.span: who
        for m, who in zip(mentions, ["ann", "bob", "ann", "cy", "bob"], strict=True)
        if m.span
    }
    run = run_with(
        assigner(truth), max_pairwise_mentions=4, all_pairs_below=0
    )  # 5 mentions > 4
    clustering = await cluster_mentions(
        run, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    assert clustering.strategy == "incremental"
    assert clustering.clusters == ((0, 2), (1, 4), (3,))
    assert clustering.pairs == {}
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    # ONE request assigns mentions 1 to 4 against the cluster of mention 0 ...
    first = transport.calls[0][1]
    assert transport.calls[0][2].field == "m1-m4"
    assert sorted(first["questions"]) == [f"assign.m{k}" for k in (1, 2, 3, 4)]
    assert list(first["questions"]["assign.m1"]["criteria"]) == ["c0", NEW]
    assert sorted(first["state"]["mentions"]) == ["m1", "m2", "m3", "m4"]
    assert first["state"]["mentions"]["m4"]["text"] == "Bob"
    assert "Ann" in first["questions"]["assign.m1"]["criteria"]["c0"]
    assert first["questions"]["assign.m1"]["instructions"].startswith(
        "Which Person does `mentions.m1` refer to?"
    )
    # ... and the mentions that started new clusters are then compared pairwise
    second = transport.calls[1][1]
    assert sorted(second["questions"]) == ["pair.m0.m2"]  # the two Bobs (m1, m4)
    assert second["state"]["mentions"]["m0"]["text"] == "Bob"
    assert len(transport.calls) == 2


async def test_each_batch_sees_the_clusters_the_earlier_ones_made() -> None:
    names = [f"Qq{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(20)]
    text = " ".join(names)
    mentions = mentions_of(text, *names)
    truth = {m.span: m.text for m in mentions if m.span}
    run = run_with(assigner(truth), strategy="incremental", all_pairs_below=0)
    result = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert len(result.clusters) == 20
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    fields = [meta.field for _, _, meta in transport.calls]
    assert fields == ["m1-m16", "m17-m19"]  # 16 mentions per request, in order
    second = transport.calls[1][1]["questions"]["assign.m17"]["criteria"]
    assert list(second) == [f"c{i}" for i in range(17)] + [NEW]


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


async def test_the_options_are_capped_at_255_with_the_latest_mentioned() -> None:
    names = [f"Q{i:03d}" for i in range(258)]
    text = " ".join(names)
    mentions = mentions_of(text, *names)
    seen: list[list[str]] = []

    def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        answers = {}
        for qid, question in payload["questions"].items():
            options = list(question["criteria"])
            seen.append(options)
            answers[qid] = choice(NEW, {o: (1.0 if o == NEW else 0.0) for o in options})
        return clef_response(answers, model=payload["model"])

    run = run_with(handler, strategy="incremental", all_pairs_below=0)
    result = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert len(result.clusters) == 258  # every mention is a new entity
    assert max(len(options) for options in seen) == 255  # 254 clusters and __new__
    last = seen[-1]
    assert "c256" in last and "c0" not in last  # the oldest mention dropped out
    assert any("coref_options_capped" in w for w in run.warnings)


# --------------------------------------------------------------------------- #
# Which pairs are asked
# --------------------------------------------------------------------------- #


def test_nicknames_are_forms_of_one_name() -> None:
    assert nicknames("bob", "robert") and nicknames("robert", "rob")
    assert nicknames("liz", "elizabeth") and nicknames("peggy", "margaret")
    assert nicknames("tom", "thomas") and nicknames("chris", "christine")
    assert not nicknames("bob", "william") and not nicknames("emma", "emma")


def test_a_nickname_makes_a_pair_plausible_however_far_apart() -> None:
    assert plausible("Bob", "Robert Smith") and plausible("Robert Smith", "Bob")
    assert plausible("Liz", "Elizabeth Warren")
    assert plausible("Bill", "William")  # two single names
    assert not plausible("Bob", "Michael Chen")
    assert not plausible("Zed", "Robert Smith")


def test_a_single_name_near_a_longer_one_is_plausible() -> None:
    assert not plausible("Zed", "Robert Smith")
    assert plausible("Zed", "Robert Smith", near=True)
    assert not plausible("Zed Roe", "Robert Smith", near=True)  # both long
    assert not plausible("Zed", "Ann", near=True)  # both single


def test_near_is_one_sentence_or_about_200_characters() -> None:
    far = "x " * 150
    text = f"Robert Smith is my uncle. {far}. Bob loves sailing. Zed Roe came."
    smith = Mention("Robert Smith", (0, 12))
    bob = Mention("Bob", (text.index("Bob"), text.index("Bob") + 3))
    assert not near(text, smith, bob)  # a sentence break and over 200 characters
    assert near("Robert Smith spoke. Bob listened.", smith, Mention("Bob", (20, 23)))
    same_sentence = "Robert Smith, " + "x " * 150 + "and Bob came."
    at = same_sentence.index("Bob")
    assert near(same_sentence, smith, Mention("Bob", (at, at + 3)))  # no break
    assert not near(text, smith, Mention("Bob"))  # no span, no answer


async def test_up_to_the_threshold_every_pair_is_asked() -> None:
    names = [f"Ab{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(24)]
    text = " ".join(names)
    mentions = mentions_of(text, *names)
    run = run_with(by_span(lambda a, b: 0.03))  # all_pairs_below defaults to 24
    result = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    sizes = sorted(len(p["questions"]) for _, p, _ in transport.calls)
    assert sum(sizes) == 276 and sizes == [20, 64, 64, 64, 64]  # 5 requests
    assert len(result.pairs) == 276 and len(result.clusters) == 24


async def test_above_the_threshold_only_plausible_pairs_are_asked() -> None:
    names = [f"Ab{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(25)]
    text = " ".join(names)
    mentions = mentions_of(text, *names)
    run = run_with(by_span(lambda a, b: 0.03))
    result = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    assert transport.calls == []  # no two share anything
    assert result.pairs == {} and len(result.clusters) == 25


async def test_the_threshold_is_configurable() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Cy")
    run = run_with(by_span(lambda a, b: 0.03), all_pairs_below=2)  # 3 mentions
    await cluster_mentions(
        run, source=SOURCE, mentions=mentions, noun="Person", schema="Person"
    )
    transport = run.transport
    assert isinstance(transport, FakeTransport)
    assert transport.calls == []


async def test_a_nickname_is_asked_about_far_above_the_threshold() -> None:
    far = " ".join(["and so it went"] * 40)
    text = f"Elizabeth Warren spoke. {far} Later Liz left. {far} Zed Roe came."
    mentions = mentions_of(text, "Elizabeth Warren", "Liz", "Zed Roe")
    asked: list[str] = []

    def handler(model: str, payload: Body, meta: CallMeta) -> Any:
        asked.extend(payload["questions"])
        return by_span(lambda a, b: 0.96 if a == mentions[0].span else 0.03)(
            model, payload, meta
        )

    run = run_with(handler, all_pairs_below=0)
    result = await cluster_mentions(
        run, source=text, mentions=mentions, noun="Person", schema="Person"
    )
    assert asked == ["pair.m0.m1"]  # Liz and Elizabeth; Zed Roe is far from both
    assert result.clusters == ((0, 1), (2,))


async def test_a_pair_that_was_not_asked_is_not_in_the_cache() -> None:
    mentions = mentions_of(SOURCE, "Ann", "Bob", "Ann")
    run = run_with(by_span(lambda a, b: 0.96), all_pairs_below=0)
    cache: dict[tuple[Mention, Mention], float] = {}
    await cluster_mentions(
        run,
        source=SOURCE,
        mentions=mentions,
        noun="Person",
        schema="Person",
        cache=cache,
    )
    assert list(cache.values()) == [0.96]  # Ann~Ann; nothing stands for Ann~Bob


def test_only_the_first_eight_and_the_last_two_mentions_show_a_context() -> None:
    shown = [i for i in range(12) if has_context(i, 12)]
    assert shown == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
    assert all(has_context(i, 5) for i in range(5))
