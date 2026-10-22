"""Matching: the assignment algorithm, the prefilter and the Clef-budget rendering."""

import itertools
import math
import random
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    AmbiguousMatchError,
    CallMeta,
    EntityInfo,
    Limits,
    MatchConfig,
    Mention,
    compile_schema,
)
from trustcall_system_one._match import (
    MAX_OPTIONS,
    NEW,
    Candidate,
    assign,
    collisions,
    compact_record,
    flag_deletes,
    hungarian,
    identity_compatible,
    identity_keys,
    match_entities,
    similarity,
    token_set_ratio,
)
from trustcall_system_one._render import estimate_tokens
from trustcall_system_one._run import Run
from trustcall_system_one.testing import FakeTransport, choice, clef_response, noul


class Person(BaseModel):
    """Someone the speaker knows."""

    name: str
    relationship: Optional[str] = None  # noqa: UP045
    notes: list[str] = []


# --------------------------------------------------------------------------- #
# Hungarian algorithm
# --------------------------------------------------------------------------- #


def total(cost: list[list[float]], cols: list[int]) -> float:
    return sum(cost[i][j] for i, j in enumerate(cols))


def brute(cost: list[list[float]]) -> float:
    rows, width = len(cost), len(cost[0])
    return min(
        sum(cost[i][j] for i, j in enumerate(pick))
        for pick in itertools.permutations(range(width), rows)
    )


def test_a_small_matrix_is_solved_optimally() -> None:
    cost = [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]
    cols = hungarian(cost)
    assert sorted(cols) == [0, 1, 2]
    assert total(cost, cols) == 5.0  # (0,1) (1,0) (2,2)
    assert cols == [1, 0, 2]


def test_a_rectangular_matrix_leaves_columns_unused() -> None:
    cost = [[9.0, 2.0, 7.0, 8.0], [6.0, 4.0, 3.0, 7.0]]
    cols = hungarian(cost)
    assert len(set(cols)) == 2
    assert total(cost, cols) == pytest.approx(brute(cost)) == 5.0


def test_degenerate_shapes() -> None:
    assert hungarian([]) == []
    assert hungarian([[3.0, 1.0, 2.0]]) == [1]
    assert hungarian([[1.0], [2.0]][:1]) == [0]


def test_more_rows_than_columns_or_ragged_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="distinct columns"):
        hungarian([[1.0], [2.0]])
    with pytest.raises(ValueError, match="same length"):
        hungarian([[1.0, 2.0], [1.0]])


def test_exact_ties_resolve_the_same_way_every_time() -> None:
    cost = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    first = hungarian(cost)
    assert all(hungarian(cost) == first for _ in range(5))
    assert sorted(first) == [0, 1, 2]


def test_hungarian_agrees_with_brute_force_on_random_matrices() -> None:
    rng = random.Random(8)
    for _ in range(300):
        rows = rng.randint(1, 5)
        width = rng.randint(rows, 6)
        # small integers make ties common, floats make them rare
        if rng.random() < 0.5:
            cost = [
                [float(rng.randint(0, 4)) for _ in range(width)] for _ in range(rows)
            ]
        else:
            cost = [[rng.uniform(-3, 21) for _ in range(width)] for _ in range(rows)]
        cols = hungarian(cost)
        assert len(set(cols)) == rows and all(0 <= c < width for c in cols)
        assert total(cost, cols) == pytest.approx(brute(cost), abs=1e-9)


def test_a_large_matrix_is_solved_in_reasonable_time() -> None:
    rng = random.Random(1)
    cost = [[rng.random() for _ in range(160)] for _ in range(80)]
    cols = hungarian(cost)
    assert len(set(cols)) == 80
    # every row's column is at least as good as the one a greedy pass would leave
    assert total(cost, cols) <= sum(min(row) for row in cost) + 80


# --------------------------------------------------------------------------- #
# assign: entities to records, any number of them new
# --------------------------------------------------------------------------- #


def test_every_entity_takes_its_most_likely_option_when_nothing_clashes() -> None:
    probs = [
        {"a": 0.9, "b": 0.05, NEW: 0.05},
        {"a": 0.1, "b": 0.8, NEW: 0.1},
        {"a": 0.02, "b": 0.03, NEW: 0.95},
    ]
    assert assign(probs, ["a", "b"]) == ["a", "b", NEW]


def test_any_number_of_entities_can_be_new() -> None:
    probs = [{NEW: 0.9, "a": 0.1} for _ in range(12)]
    assert assign(probs, ["a"]) == [NEW] * 12
    assert assign(probs, []) == [NEW] * 12
    assert assign([], ["a"]) == []


def test_the_loser_of_a_collision_falls_to_its_best_feasible_option() -> None:
    probs = [
        {"a": 0.9, "b": 0.05, NEW: 0.05},
        {"a": 0.6, "b": 0.3, NEW: 0.1},  # wants "a" too, then "b"
    ]
    assert assign(probs, ["a", "b"]) == ["a", "b"]
    # with "b" taken by a stronger claim the loser becomes new
    probs.append({"a": 0.01, "b": 0.97, NEW: 0.02})
    assert assign(probs, ["a", "b"]) == ["a", NEW, "b"]


def test_the_assignment_minimises_the_total_not_the_greedy_choice() -> None:
    # both want "a"; taking it for the first entity (as a greedy pass in order
    # would) leaves the second with a 0.1 option, while giving it to the second and
    # "b" to the first is far better in total
    probs = [
        {"a": 0.9, "b": 0.85, NEW: 0.0},
        {"a": 0.9, "b": 0.1, NEW: 0.0},
    ]
    assert assign(probs, ["a", "b"]) == ["b", "a"]


def test_exact_ties_go_to_the_earlier_entity() -> None:
    probs = [{"a": 0.7, NEW: 0.3}, {"a": 0.7, NEW: 0.3}]
    for _ in range(5):
        assert assign(probs, ["a"]) == ["a", NEW]
    three = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
    assert assign(three, ["a", "b"]) == ["a", "b", NEW]


def test_a_missing_answer_costs_every_option_the_same() -> None:
    probs = [{}, {"a": 0.9, NEW: 0.1}]
    assert assign(probs, ["a"])[1] == "a"


def test_assign_agrees_with_brute_force_on_random_tables() -> None:
    rng = random.Random(21)

    def cost(p: float) -> float:
        return -math.log(max(p, 1e-9))

    for _ in range(200):
        n, m = rng.randint(1, 4), rng.randint(0, 4)
        ids = [f"r{i}" for i in range(m)]
        table = []
        for _ in range(n):
            weights = [rng.random() ** 2 for _ in range(m + 1)]
            scale = sum(weights)
            table.append(
                {
                    **{
                        rid: w / scale for rid, w in zip(ids, weights[:-1], strict=True)
                    },
                    NEW: weights[-1] / scale,
                }
            )
        chosen = assign(table, ids)
        used = [c for c in chosen if c != NEW]
        assert len(used) == len(set(used))
        got = sum(cost(table[i].get(c, 0.0)) for i, c in enumerate(chosen))
        best = math.inf
        for picks in itertools.product([*ids, NEW], repeat=n):
            taken = [p for p in picks if p != NEW]
            if len(taken) != len(set(taken)):
                continue
            best = min(
                best, sum(cost(table[i].get(p, 0.0)) for i, p in enumerate(picks))
            )
        assert got == pytest.approx(best, abs=1e-6)


def test_records_nobody_gives_a_probability_are_left_out_of_the_matrix() -> None:
    ids = [f"r{i}" for i in range(3000)]
    table = [
        {"r5": 0.9, "r6": 0.05, NEW: 0.05},
        {"r5": 0.6, "r2999": 0.3, NEW: 0.1},
        {NEW: 0.99},
    ]
    # every other record has probability 0: with 3000 of them the full matrix would
    # be slow; leaving them out decides exactly as the full matrix would
    pruned = assign(table, ids, prune=1e-3)
    assert pruned == ["r5", "r2999", NEW]
    assert pruned == assign(table, ["r5", "r6", "r2999"])
    assert assign(table, ids[:10], prune=1e-3) == ["r5", NEW, NEW]


def test_pruning_is_the_exact_optimum_of_the_clipped_matrix() -> None:
    rng = random.Random(77)

    def clipped(p: float, floor: float) -> float:
        return -math.log(max(p, floor))

    for _ in range(400):
        floor = rng.choice([1e-3, 1e-2, 0.05])
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        ids = [f"r{i}" for i in range(m)]
        table = []
        for _ in range(n):
            # many tiny probabilities, some of them below the floor
            weights = [
                rng.choice([0, 1e-4, 5e-4, 0.002, 0.02, 0.3, 1, 3])
                for _ in range(m + 1)
            ]
            scale = sum(weights) or 1
            options = [*ids, NEW]
            table.append({o: w / scale for o, w in zip(options, weights, strict=True)})
        chosen = assign(table, ids, prune=floor)
        used = [c for c in chosen if c != NEW]
        assert len(used) == len(set(used))
        got = sum(clipped(table[i].get(c, 0.0), floor) for i, c in enumerate(chosen))
        best = math.inf
        for picks in itertools.product([*ids, NEW], repeat=n):
            taken = [p for p in picks if p != NEW]
            if len(taken) != len(set(taken)):
                continue
            best = min(
                best,
                sum(clipped(table[i].get(p, 0.0), floor) for i, p in enumerate(picks)),
            )
        assert got == pytest.approx(best, abs=1e-6)


def test_a_contested_record_goes_to_the_stronger_claim_with_pruning_on() -> None:
    # the weaker claim is above the floor, so it is still told apart from nothing
    table = [
        {"a": 0.3, "b": 0.02, NEW: 0.68},
        {"a": 0.6, "b": 0.02, NEW: 0.38},
    ]
    assert assign(table, ["a", "b"], prune=1e-3) == [NEW, "a"]
    # below the floor nothing separates the options: the earlier entity keeps its tie
    tiny = [{"a": 0.0005, NEW: 0.5}, {"a": 0.0002, NEW: 0.5}]
    assert assign(tiny, ["a"], prune=1e-3) == [NEW, NEW]


# --------------------------------------------------------------------------- #
# Collisions
# --------------------------------------------------------------------------- #


def test_two_claims_of_0_5_or_more_on_one_record_collide() -> None:
    probs = [
        {"a": 0.9, NEW: 0.1},
        {"a": 0.6, NEW: 0.4},
        {"a": 0.4, "b": 0.3, NEW: 0.3},  # a claim of 0.4 is not one
        {"b": 0.8, NEW: 0.2},
    ]
    assert collisions(probs) == {"a": [0, 1]}


def test_a_new_entity_or_an_unanswered_one_claims_nothing() -> None:
    assert collisions([{NEW: 0.9, "a": 0.1}, {NEW: 0.8, "a": 0.2}, {}]) == {}
    assert (
        collisions([{"a": 0.5, NEW: 0.5}, {"a": 0.5, NEW: 0.5}]) == {}
    )  # new wins ties


def test_the_claim_threshold_can_be_changed() -> None:
    probs = [{"a": 0.45, NEW: 0.1}, {"a": 0.4, NEW: 0.2}]
    assert collisions(probs) == {}
    assert collisions(probs, 0.4) == {"a": [0, 1]}


# --------------------------------------------------------------------------- #
# Similarity, the prefilter's ranking
# --------------------------------------------------------------------------- #


def test_token_set_ratio_ignores_order_and_extra_words() -> None:
    assert token_set_ratio("Sarah", "Sarah Johnson") == 1.0
    assert token_set_ratio("Johnson, Sarah", "sarah johnson") == 1.0
    assert token_set_ratio("Emma Thompson", "Michael Chen") < 0.4
    assert token_set_ratio("", "Emma") == 0.0
    assert token_set_ratio("Emma Thomson", "Emma Thompson") > 0.8


def test_a_diminutive_counts_as_a_close_match() -> None:
    assert similarity(["Robert Smith"], ["Bob"]) == 0.9
    assert similarity(["Robert Smith"], ["Bob", "Robert"]) == 1.0
    assert similarity(["Liz Warren"], ["Elizabeth Warren"]) >= 0.9
    assert similarity(["Zed Qux"], ["Bob"]) < 0.5
    assert similarity([], ["Bob"]) == 0.0


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        ("Sarah", "Sarah Johnson", True),
        ("Liz", "Elizabeth Smith", True),
        ("Bob", "Robert Smith", True),
        ("J. Smith", "John Smith", True),
        ("JS", "John Smith", True),
        ("Ann", "Anne Lee", True),
        ("Emma Thomson", "Emma Thompson", True),  # a typo
        ("Dr. Emma Thompson", "Emma Thompson", True),
        ("emma thompson", "Emma  Thompson", True),
        ("Acme Corp", "ACME Corporation", True),
        ("Emma Watson", "Emma Thompson", False),  # a shared first name is not enough
        ("Olivia Davis", "Emma Thompson", False),
        ("SO-10442", "SO-10443", False),  # ids that differ in a number
        ("Tom Ford", "Tim Ford", False),
    ],
)
def test_identity_compatible(first: str, second: str, same: bool) -> None:
    assert identity_compatible(first, second) is same
    assert identity_compatible(second, first) is same


@pytest.mark.parametrize("blank", ["", "   ", "...", "\t\n"])
def test_a_blank_identity_fits_nothing(blank: str) -> None:
    assert not identity_compatible(blank, "Emma")
    assert not identity_compatible("Emma", blank)
    assert not identity_compatible(blank, blank)


def test_compact_records_keep_identity_scalars_and_list_heads() -> None:
    record = {
        "name": "Emma Thompson",
        "a": 1,
        "b": "two",
        "c": None,
        "d": 3.5,
        "e": True,
        "f": "late",
        "nested": {"x": 1},
        "notes": [f"note {i}" for i in range(10)],
        "bio": "x" * 500,
    }
    small = compact_record(record, ["name"], 4)
    assert list(small) == ["name", "a", "b", "d", "e", "notes"]
    assert small["notes"] == ["note 0", "note 1", "note 2"]
    assert compact_record(record, ["name"], 0) == {
        "name": "Emma Thompson",
        "notes": ["note 0", "note 1", "note 2"],
    }
    long = compact_record(record, ["bio"], 0)
    assert long["bio"].endswith("…") and len(long["bio"]) < 200


def test_identity_keys_follow_aliases() -> None:
    class Aliased(BaseModel):
        full_name: str = Field(alias="fullName")

    plan = compile_schema(Aliased)
    assert identity_keys(plan, ["full_name"]) == ("fullName",)
    assert identity_keys(compile_schema(Person), ["name"]) == ("name",)


# --------------------------------------------------------------------------- #
# Asking: the prefilter, the budget, the answers
# --------------------------------------------------------------------------- #

TEXT = "Emma Thompson and Bob went out. Later Bob called Emma."


def entity(key: str, label: str, *texts: str, after: str = "") -> EntityInfo:
    mentions = []
    cursor = TEXT.index(after) if after else 0
    for text in texts:
        at = TEXT.index(text, cursor)
        mentions.append(Mention(text, (at, at + len(text))))
        cursor = at + len(text)
    return EntityInfo(
        key=key,
        label=label,
        mentions=tuple(mentions),
        first_mention=mentions[0],
        qualify_p=0.97,
    )


def named(key: str, label: str) -> EntityInfo:
    """An entity whose single mention has no span (its text is not in ``TEXT``)."""
    mention = Mention(label, None)
    return EntityInfo(
        key=key, label=label, mentions=(mention,), first_mention=mention, qualify_p=0.97
    )


class Recorder:
    """A transport handler that answers match questions from a name table."""

    def __init__(self, table: dict[str, str | None], p: float = 0.92) -> None:
        self.table, self.p = table, p
        self.questions: list[dict[str, Any]] = []
        self.states: list[Any] = []
        self.meta: list[CallMeta] = []
        self.transport = FakeTransport(self.handle)

    def handle(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        if meta.kind == "match_confirm":  # not under test here: "a different one"
            return clef_response(
                {q: noul(0.05) for q in payload["questions"]}, model=payload["model"]
            )
        assert meta.kind == "match"
        self.meta.append(meta)
        self.states.append(payload["state"])
        question = payload["questions"]["match"]
        self.questions.append(question)
        label = payload["state"]["entity"]["label"]
        wanted = self.table.get(label)
        options = dict(question["criteria"])
        pick = NEW
        for option, record in options.items():
            if (
                option != NEW
                and isinstance(record, dict)
                and record.get("name") == wanted
            ):
                pick = option
        rest = (1 - self.p) / (len(options) - 1)
        probs = {o: (self.p if o == pick else rest) for o in options}
        return clef_response({"match": choice(pick, probs)}, model=payload["model"])


def candidates(*names: str, extra: dict[str, Any] | None = None) -> list[Candidate]:
    return [
        Candidate(
            str(i), {"name": name, "relationship": None, "notes": [], **(extra or {})}
        )
        for i, name in enumerate(names)
    ]


async def never(a: EntityInfo, b: EntityInfo) -> float | None:
    raise AssertionError("no merge check expected")


async def run_match(
    recorder: Recorder,
    entities: list[EntityInfo],
    records: list[Candidate],
    *,
    match: MatchConfig | None = None,
    limits: Limits | None = None,
) -> tuple[Any, Run]:
    run = Run(
        transport=recorder.transport,
        match=match or MatchConfig(),
        limits=limits or Limits(),
    )
    outcome = await match_entities(
        run, compile_schema(Person), TEXT, "Person", entities, records, ("name",), never
    )
    return outcome, run


async def test_each_entity_gets_one_choice_over_the_records_and_new() -> None:
    records = candidates("Zed", "Emma Thompson", "Bob Roe")
    recorder = Recorder({"Emma Thompson": "Emma Thompson", "Bob": None})
    outcome, run = await run_match(
        recorder,
        [
            entity("e0", "Emma Thompson", "Emma Thompson", "Emma"),
            entity("e1", "Bob", "Bob"),
        ],
        records,
    )
    assert [m.kind for m in recorder.meta] == ["match", "match"]
    assert [m.entity for m in recorder.meta] == ["e0", "e1"]
    first = recorder.questions[0]
    assert first["type"] == "choice"
    assert first["instructions"] == "Which listed record is `entity.label`?"
    assert first["criteria"]["rec_1"] == records[1].data  # the record, as an object
    assert first["criteria"][NEW] == "A Person not among the listed records"
    assert list(first["criteria"]) == ["rec_0", "rec_1", "rec_2", NEW]
    state = recorder.states[0]
    assert state["input"] == TEXT
    assert state["entity"]["label"] == "Emma Thompson"
    assert [m["text"] for m in state["entity"]["mentions"]] == ["Emma Thompson", "Emma"]
    assert all("span" in m for m in state["entity"]["mentions"])
    assert outcome.targets == {"e0": "1", "e1": None}
    emma = outcome.entities[0]
    assert emma.match_probs["1"] == 0.92 and emma.match_probs[NEW] == pytest.approx(
        0.08 / 3
    )
    assert set(emma.match_probs) == {"0", "1", "2", NEW}
    assert run.warnings == [] and outcome.skipped == ()


async def test_no_entities_means_no_requests() -> None:
    recorder = Recorder({})
    outcome, _ = await run_match(recorder, [], candidates("Zed"))
    assert outcome.entities == () and recorder.meta == []


async def test_300_records_are_cut_to_the_254_most_similar_and_keep_the_match() -> None:
    names = [f"Zed Qux{i:03d}" for i in range(300)]
    names[287] = "Emma Thompson"
    names[299] = "Robert Smith"  # Bob's record: a diminutive
    records = candidates(*names)
    recorder = Recorder({"Emma Thompson": "Emma Thompson", "Bob": "Robert Smith"})
    outcome, run = await run_match(
        recorder,
        [entity("e0", "Emma Thompson", "Emma Thompson"), entity("e1", "Bob", "Bob")],
        records,
    )
    for question in recorder.questions:
        assert len(question["criteria"]) == MAX_OPTIONS + 1  # 254 records + __new__
    assert "rec_287" in recorder.questions[0]["criteria"]
    assert "rec_299" in recorder.questions[1]["criteria"]
    assert outcome.targets == {"e0": "287", "e1": "299"}
    (warning,) = [w for w in run.warnings if w.startswith("match_prefiltered")]
    assert "300 existing Person records" in warning and "254" in warning
    assert len(outcome.entities[0].match_probs) == MAX_OPTIONS + 1


async def test_exactly_254_records_are_not_prefiltered() -> None:
    records = candidates(*[f"Zed Qux{i:03d}" for i in range(254)])
    recorder = Recorder({})
    _, run = await run_match(recorder, [entity("e0", "Bob", "Bob")], records)
    assert len(recorder.questions[0]["criteria"]) == 255
    assert run.warnings == []


def bulky(names: list[str]) -> list[Candidate]:
    return candidates(
        *names,
        extra={
            "relationship": "colleague " * 40,
            "notes": [f"note {i} " + "word " * 60 for i in range(40)],
        },
    )


async def test_records_over_the_clef_budget_are_rendered_compactly() -> None:
    records = bulky(["Zed Qux", "Emma Thompson", "Zoe Quill", "Yan Quirk", "Xi Qi"])
    recorder = Recorder({"Emma Thompson": "Emma Thompson"})
    outcome, run = await run_match(
        recorder,
        [entity("e0", "Emma Thompson", "Emma Thompson")],
        records,
        limits=Limits(clef_state_token_budget=2_500),
    )
    criteria = recorder.questions[0]["criteria"]
    assert len([o for o in criteria if o != NEW]) == 5  # all of them still fit
    for option, record in criteria.items():
        if option != NEW:
            assert list(record) == ["name", "relationship", "notes"]
            assert len(record["notes"]) == 3
            assert len(record["relationship"]) <= 161
    assert outcome.targets == {"e0": "1"}
    (warning,) = [w for w in run.warnings if w.startswith("match_compacted")]
    assert "5 existing Person records" in warning and "most similar" not in warning


async def test_a_state_that_still_does_not_fit_offers_fewer_records_best_first() -> (
    None
):
    names = [f"Zed Qux{i:02d}" for i in range(60)]
    names[41] = "Emma Thompson"
    records = bulky(names)
    recorder = Recorder({"Emma Thompson": "Emma Thompson"})
    outcome, run = await run_match(
        recorder,
        [entity("e0", "Emma Thompson", "Emma Thompson")],
        records,
        limits=Limits(clef_state_token_budget=2_500),
    )
    offered = [o for o in recorder.questions[0]["criteria"] if o != NEW]
    assert 1 <= len(offered) < 60
    assert "rec_41" in offered  # the most similar record is always kept
    assert outcome.targets == {"e0": "41"}
    (warning,) = [w for w in run.warnings if w.startswith("match_compacted")]
    assert "only the most similar" in warning
    # records that were not offered have no probability
    assert len(outcome.entities[0].match_probs) == len(offered) + 1


async def test_a_record_that_does_not_fit_beside_a_huge_input_narrows_the_input() -> (
    None
):
    long_text = TEXT + " " + "filler words go here. " * 600
    records = bulky(["Emma Thompson", "Zed Qux"])
    recorder = Recorder({"Emma Thompson": "Emma Thompson"})
    run = Run(
        transport=recorder.transport,
        limits=Limits(clef_state_token_budget=2_500),
    )
    info = entity("e0", "Emma Thompson", "Emma Thompson", "Emma")
    outcome = await match_entities(
        run,
        compile_schema(Person),
        long_text,
        "Person",
        [info],
        records,
        ("name",),
        never,
    )
    assert outcome.targets == {"e0": "0"}
    sent = recorder.states[0]["input"]
    assert len(sent) < len(long_text) and "Emma Thompson" in sent
    assert any("clef_state_narrowed" in w for w in run.warnings)


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


class Fixed:
    """Match answers by entity label: probabilities by record name (or ``__new__``)."""

    def __init__(
        self,
        table: dict[str, dict[str, float]],
        confidence: float | dict[str, float] | None = None,
        confirm: dict[str, float] | None = None,
    ) -> None:
        self.table, self.confidence = table, confidence
        self.confirm = confirm or {}  # the pairwise answer by entity label
        self.confirmed: list[tuple[str, dict[str, Any]]] = []
        self.transport = FakeTransport(self.handle)

    def handle(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        label = payload["state"]["entity"]["label"]
        if meta.kind == "match_confirm":
            self.confirmed.append((label, payload))
            p = self.confirm.get(label, 0.05)
            return clef_response(
                {q: noul(p) for q in payload["questions"]}, model=payload["model"]
            )
        options = dict(payload["questions"]["match"]["criteria"])
        wanted = self.table[label]
        probs = {
            option: wanted.get(NEW if option == NEW else options[option]["name"], 0.0)
            for option in options
        }
        best = max(probs, key=lambda o: probs[o])
        confidence = (
            self.confidence.get(label)
            if isinstance(self.confidence, dict)
            else self.confidence
        )
        return clef_response(
            {"match": choice(best, probs, confidence)}, model=payload["model"]
        )


async def run_fixed(
    fixed: Fixed,
    entities: list[EntityInfo],
    records: list[Candidate],
    match: MatchConfig | None = None,
) -> tuple[Any, Run]:
    run = Run(transport=fixed.transport, match=match or MatchConfig())
    outcome = await match_entities(
        run, compile_schema(Person), TEXT, "Person", entities, records, ("name",), never
    )
    return outcome, run


async def test_an_assignment_below_the_accept_threshold_is_ambiguous() -> None:
    fixed = Fixed({"Emma Thompson": {"Ann": 0.5, "Zed": 0.3, NEW: 0.2}})
    outcome, _ = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson")],
        candidates("Ann", "Zed"),
    )
    assert outcome.targets == {}
    (skip,) = outcome.skipped
    assert skip.reason == "ambiguous_match" and skip.entity.label == "Emma Thompson"
    assert skip.candidates == (("0", 0.5), ("1", 0.3), (NEW, 0.2))
    assert "p=0.50" in skip.detail and "0.6" in skip.detail
    assert skip.entity.match_probs == {"0": 0.5, "1": 0.3, NEW: 0.2}


async def test_a_low_confidence_answer_is_ambiguous_even_if_the_option_is_likely() -> (
    None
):
    fixed = Fixed({"Emma Thompson": {"Ann": 0.9, NEW: 0.1}}, confidence=0.3)
    outcome, _ = await run_fixed(
        fixed, [entity("e0", "Emma Thompson", "Emma Thompson")], candidates("Ann")
    )
    (skip,) = outcome.skipped
    assert skip.reason == "ambiguous_match" and "confidence is 0.30" in skip.detail


async def test_the_thresholds_are_configurable() -> None:
    fixed = Fixed({"Emma Thompson": {"Ann": 0.5, "Zed": 0.3, NEW: 0.2}})
    outcome, _ = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson")],
        candidates("Ann", "Zed"),
        MatchConfig(accept=0.5),
    )
    assert outcome.targets == {"e0": "0"} and outcome.skipped == ()
    sure = Fixed({"Emma Thompson": {"Ann": 0.9, NEW: 0.1}}, confidence=0.3)
    outcome, _ = await run_fixed(
        sure,
        [entity("e0", "Emma Thompson", "Emma Thompson")],
        candidates("Ann"),
        MatchConfig(min_confidence=0.3),
    )
    assert outcome.targets == {"e0": "0"}


async def test_a_name_compatible_top_record_is_accepted_without_the_confidence() -> (
    None
):
    # live: p 0.84 with a choice confidence of 0.47 skipped a plain rename
    fixed = Fixed(
        {"Emma Thompson": {"Emma Thompson": 0.84, NEW: 0.16}}, confidence=0.47
    )
    outcome, _ = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson", "Emma")],
        candidates("Emma Thompson", "Bob Roe"),
    )
    assert outcome.targets == {"e0": "0"} and outcome.skipped == ()
    # a record that no mention fits stays ambiguous with the same answer
    other = Fixed({"Bob": {"Emma Thompson": 0.84, NEW: 0.16}}, confidence=0.47)
    outcome, _ = await run_fixed(
        other, [entity("e0", "Bob", "Bob")], candidates("Emma Thompson", "Zed Qux")
    )
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"


async def test_the_readme_emma_at_0_506_is_confirmed_pairwise() -> None:
    """Live 5/6: Emma, 0.506 for the record against 0.4683 for __new__, confidence
    0.30. The margin makes it a coin flip for the choice; the pairwise question
    decides it."""
    table = {"Emma": {"Emma Thompson": 0.506, NEW: 0.4683, "Bob Roe": 0.02}}
    pool = candidates("Emma Thompson", "Bob Roe")
    fixed = Fixed(table, confidence=0.30, confirm={"Emma": 0.88})
    outcome, run = await run_fixed(fixed, [entity("e0", "Emma", "Emma")], pool)
    assert outcome.targets == {"e0": "0"} and outcome.skipped == ()
    ((label, payload),) = fixed.confirmed
    assert label == "Emma"
    assert payload["questions"]["confirm"]["instructions"] == (
        "In `input`, is `entity.label` the same Person as the existing record "
        "`record` (Emma Thompson)?"
    )
    assert payload["state"]["record"]["name"] == "Emma Thompson"
    assert payload["state"]["entity"]["label"] == "Emma"
    assert any(w.startswith("match_confirmed") for w in run.warnings)
    # a low answer keeps the choice's outcome: ambiguous
    low = Fixed(table, confidence=0.30, confirm={"Emma": 0.4})
    outcome, _ = await run_fixed(low, [entity("e0", "Emma", "Emma")], pool)
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"
    # the knob moves the bar and 0 switches the step off
    knob = MatchConfig(confirm_accept=0.95)
    outcome, _ = await run_fixed(fixed, [entity("e0", "Emma", "Emma")], pool, knob)
    assert outcome.targets == {}
    off = Fixed(table, confidence=0.30, confirm={"Emma": 0.99})
    outcome, _ = await run_fixed(
        off, [entity("e0", "Emma", "Emma")], pool, MatchConfig(confirm_accept=0)
    )
    assert outcome.targets == {} and off.confirmed == []
    # and the compatible shortcut still takes a clear margin without any question
    clear = Fixed({"Emma": {"Emma Thompson": 0.58, NEW: 0.38, "Bob Roe": 0.04}}, 0.30)
    outcome, _ = await run_fixed(clear, [entity("e0", "Emma", "Emma")], pool)
    assert outcome.targets == {"e0": "0"} and clear.confirmed == []
    below = Fixed({"Emma": {"Emma Thompson": 0.45, NEW: 0.3, "Bob Roe": 0.25}})
    outcome, _ = await run_fixed(
        below,
        [entity("e0", "Emma", "Emma")],
        pool,
        MatchConfig(accept_compatible=0.4, confirm_accept=0),
    )
    assert outcome.targets == {"e0": "0"}


async def test_a_short_form_the_choice_calls_new_is_confirmed_pairwise() -> None:
    """short_form (live 7): __new__ 0.831 with confidence 0.4377 (not sure) for
    "Emma" against the record Emma Thompson; the pairwise answer was 0.8472."""
    fixed = Fixed(
        {"Emma": {"Emma Thompson": 0.169, NEW: 0.831}},
        confidence=0.4377,
        confirm={"Emma": 0.8472},
    )
    outcome, _ = await run_fixed(
        fixed, [entity("e0", "Emma", "Emma")], candidates("Emma Thompson")
    )
    assert outcome.targets == {"e0": "0"} and outcome.confirmed == {"e0"}
    # the pairwise answer must beat the choice's own p(__new__)
    weak = Fixed(
        {"Emma": {"Emma Thompson": 0.169, NEW: 0.831}},
        confidence=0.4377,
        confirm={"Emma": 0.80},
    )
    outcome, _ = await run_fixed(
        weak, [entity("e0", "Emma", "Emma")], candidates("Emma Thompson")
    )
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"


async def test_a_final_new_is_never_overridden_by_the_pairwise_answer() -> None:
    """A `__new__` at 0.9 or more is final (different_emma 0.984, new_hire 0.929, a
    sure 0.92): a gullible pairwise 0.99 does not turn a new person into the record."""
    for p_new in (0.984, 0.929, 0.92, 0.9):
        fixed = Fixed(
            {"Maria": {"Maria Costa": round(1 - p_new, 3), NEW: p_new}},
            confidence=p_new,  # Experiential reports the top probability itself
            confirm={"Maria": 0.99},
        )
        outcome, _ = await run_fixed(
            fixed, [named("e0", "Maria")], candidates("Maria Costa")
        )
        assert outcome.targets == {"e0": None} and fixed.confirmed == [], p_new


async def test_a_new_below_the_final_probability_may_be_confirmed_on_any_provider() -> (
    None
):
    """The sureness of `__new__` is its probability, not the choice's confidence:
    Cloudflare gave short_form a confidence of 0.44, Experiential reports 0.8176 (the
    top probability) for the same answer, and both are confirmed."""
    cases = [
        ("cloudflare", 0.831, 0.4377, 0.8472),  # live 7
        ("experiential", 0.8176, 0.8176, 0.85),  # live 8
        ("maria on experiential", 0.696, 0.696, 0.75),  # must beat 0.7 and 0.696
    ]
    for name, p_new, confidence, p_confirm in cases:
        fixed = Fixed(
            {"Emma": {"Emma Thompson": round(1 - p_new, 4), NEW: p_new}},
            confidence=confidence,
            confirm={"Emma": p_confirm},
        )
        outcome, _ = await run_fixed(
            fixed, [named("e0", "Emma")], candidates("Emma Thompson")
        )
        assert outcome.targets == {"e0": "0"}, name
    # it must beat max(confirm_accept, p(__new__)): 0.69 is under the floor
    weak = Fixed(
        {"Emma": {"Emma Thompson": 0.3, NEW: 0.696}},
        confidence=0.696,
        confirm={"Emma": 0.69},
    )
    outcome, _ = await run_fixed(
        weak, [named("e0", "Emma")], candidates("Emma Thompson")
    )
    assert outcome.targets == {"e0": None}  # the choice's own outcome stands


async def test_the_final_probability_is_a_knob() -> None:
    table = {"Emma": {"Emma Thompson": 0.08, NEW: 0.92}}
    fixed = Fixed(table, confidence=0.92, confirm={"Emma": 0.95})
    outcome, _ = await run_fixed(
        fixed,
        [named("e0", "Emma")],
        candidates("Emma Thompson"),
        MatchConfig(new_final=0.95),
    )
    assert outcome.targets == {"e0": "0"}  # 0.92 is no longer final
    outcome, _ = await run_fixed(
        Fixed(table, confidence=0.92, confirm={"Emma": 0.95}),
        [named("e0", "Emma")],
        candidates("Emma Thompson"),
    )
    assert outcome.targets == {"e0": None}  # at the default 0.9 it is


async def test_two_first_names_are_confirmed_one_by_one_when_unsure() -> None:
    fixed = Fixed(
        {
            "Maria": {"Maria Costa": 0.15, "Jon Petit": 0.01, NEW: 0.84},
            "Jon": {"Maria Costa": 0.01, "Jon Petit": 0.852, NEW: 0.138},
        },
        confidence={"Maria": 0.4, "Jon": 0.616},
        confirm={"Maria": 0.93},
    )
    outcome, _ = await run_fixed(
        fixed,
        [named("e0", "Maria"), named("e1", "Jon")],
        candidates("Maria Costa", "Jon Petit"),
    )
    assert outcome.targets == {"e0": "0", "e1": "1"}
    assert [label for label, _ in fixed.confirmed] == ["Maria"]  # Jon was sure


async def test_a_full_name_that_contradicts_the_record_is_never_confirmed() -> None:
    from trustcall_system_one._match import _Matcher

    run = Run(transport=FakeTransport(lambda m, p, meta: None))
    matcher = _Matcher(
        run,
        compile_schema(Person),
        TEXT,
        "Person",
        candidates("Emma Thompson"),
        ("name",),
    )
    emma = named("e0", "Emma")
    davis = EntityInfo(
        key="e1",
        label="Emma Davis",
        mentions=(Mention("Emma Davis", None), Mention("Emma", None)),
        first_mention=Mention("Emma Davis", None),
        qualify_p=0.97,
    )
    assert matcher.sole_fit(emma) == 0
    assert matcher.sole_fit(davis) is None  # "Emma Davis" is not "Emma Thompson"


async def test_a_new_person_who_is_not_a_strong_fit_is_never_asked_about() -> None:
    fixed = Fixed(
        {"Emma Watson": {"Emma Thompson": 0.1, NEW: 0.88}},
        confirm={"Emma Watson": 0.99},
    )
    outcome, _ = await run_fixed(
        fixed,
        [named("e0", "Emma Watson")],
        candidates("Emma Thompson", "Bob Roe"),
    )
    assert outcome.targets == {"e0": None} and fixed.confirmed == []


async def test_two_compatible_records_are_never_confirmed() -> None:
    fixed = Fixed(
        {"Emma": {"Emma Thompson": 0.1, "Emma Watson": 0.1, NEW: 0.8}},
        confirm={"Emma": 0.99},
    )
    outcome, _ = await run_fixed(
        fixed,
        [entity("e0", "Emma", "Emma")],
        candidates("Emma Thompson", "Emma Watson"),
    )
    assert outcome.targets == {"e0": None} and fixed.confirmed == []


async def test_a_different_person_is_not_forced_onto_the_record_by_a_low_answer() -> (
    None
):
    fixed = Fixed(
        {"Emma": {"Emma Thompson": 0.1, NEW: 0.85}},
        confidence=0.3,
        confirm={"Emma": 0.12},
    )
    outcome, _ = await run_fixed(
        fixed, [entity("e0", "Emma", "Emma")], candidates("Emma Thompson")
    )
    # the choice was not sure, the pairwise answer is low: the old outcome stands
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"
    assert len(fixed.confirmed) == 1
    # with on_ambiguous="insert" the likely-new reading is a new person
    inserting = MatchConfig(on_ambiguous="insert", accept=0.6)
    outcome, _ = await run_fixed(
        fixed, [entity("e0", "Emma", "Emma")], candidates("Emma Thompson"), inserting
    )
    assert outcome.targets == {"e0": None}


async def test_a_record_another_entity_holds_is_not_confirmed_for_a_second() -> None:
    fixed = Fixed(
        {
            "Emma Thompson": {"Emma Thompson": 0.95, NEW: 0.05},
            "Emma": {"Emma Thompson": 0.1, NEW: 0.88},
        },
        confidence=0.9,
        confirm={"Emma": 0.99},
    )
    outcome, _ = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson"), named("e1", "Emma")],
        candidates("Emma Thompson"),
    )
    assert outcome.targets == {"e0": "0", "e1": None} and fixed.confirmed == []


async def test_the_compatible_shortcut_is_no_coin_flip() -> None:
    # 0.51 against __new__ 0.49 with a low confidence: not a decision
    flip = Fixed(
        {"Emma": {"Emma Thompson": 0.51, NEW: 0.49}},
        confidence=0.02,
    )
    pool = candidates("Emma Thompson", "Bob Roe")
    outcome, _ = await run_fixed(flip, [entity("e0", "Emma", "Emma")], pool)
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"
    # the margin is over the runner-up too, not only over __new__
    runner = Fixed(
        {"Emma": {"Emma Thompson": 0.52, "Emma Watson": 0.45, NEW: 0.03}},
        confidence=0.02,
    )
    outcome, _ = await run_fixed(
        runner,
        [entity("e0", "Emma", "Emma")],
        candidates("Emma Thompson", "Emma Watson"),
    )
    assert outcome.targets == {}
    # a margin of 0.1 is enough, and the knob moves it
    clear = Fixed({"Emma": {"Emma Thompson": 0.58, NEW: 0.42}}, confidence=0.02)
    outcome, _ = await run_fixed(clear, [entity("e0", "Emma", "Emma")], pool)
    assert outcome.targets == {"e0": "0"}
    wide = MatchConfig(compatible_margin=0.3)
    outcome, _ = await run_fixed(clear, [entity("e0", "Emma", "Emma")], pool, wide)
    assert outcome.targets == {}
    off = MatchConfig(compatible_margin=0.0)
    outcome, _ = await run_fixed(flip, [entity("e0", "Emma", "Emma")], pool, off)
    assert outcome.targets == {"e0": "0"}


async def test_a_prefix_alone_does_not_accept_a_longer_name() -> None:
    # "Dan" is the start of "Daniela", and a nickname of Daniel: not enough
    fixed = Fixed({"Dan": {"Daniela Cruz": 0.7, NEW: 0.2}}, confidence=0.1)
    pool = candidates("Daniela Cruz", "Bob Roe")
    outcome, _ = await run_fixed(fixed, [named("e0", "Dan")], pool)
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"
    # a typo of a two-word name is a fit; so is a single word the record has
    typo = Fixed({"Emma Thomson": {"Emma Thompson": 0.7, NEW: 0.2}}, confidence=0.1)
    outcome, _ = await run_fixed(
        typo,
        [named("e0", "Emma Thomson")],
        candidates("Emma Thompson", "Bob Roe"),
    )
    assert outcome.targets == {"e0": "0"}
    word = Fixed({"Daniela": {"Daniela Cruz": 0.7, NEW: 0.2}}, confidence=0.1)
    outcome, _ = await run_fixed(word, [named("e0", "Daniela")], pool)
    assert outcome.targets == {"e0": "0"}
    liz = Fixed({"Liz": {"Elizabeth Parker": 0.7, NEW: 0.2}}, confidence=0.1)
    outcome, _ = await run_fixed(
        liz,
        [named("e0", "Liz")],
        candidates("Elizabeth Parker", "Bob Roe"),
    )
    assert outcome.targets == {"e0": "0"}


@pytest.mark.parametrize(
    ("mention", "value", "strong"),
    [
        ("Dan", "Daniela Cruz", False),
        ("Jo", "John Doe", False),
        ("JS", "John Smith", False),
        ("Emma Thomson", "Emma Thompson", True),
        ("Emma", "Emma Thompson", True),
        ("Liz", "Elizabeth Parker", True),
        ("J. Smith", "John Smith", True),
        ("Emma Watson", "Emma Thompson", False),
        ("", "Emma", False),
    ],
)
def test_strong_compatible(mention: str, value: str, strong: bool) -> None:
    from trustcall_system_one._coref import strong_compatible

    assert strong_compatible(mention, value) is strong


async def test_a_blank_identity_record_is_never_the_compatible_one() -> None:
    recs = [
        Candidate("0", {"name": "", "relationship": None, "notes": []}),
        Candidate("1", {"name": "  ", "relationship": None, "notes": []}),
    ]
    fixed = Fixed({"Zara Khan": {"0": 0.7, NEW: 0.2}}, confidence=0.1)
    outcome, _ = await run_fixed(fixed, [named("e0", "Zara Khan")], recs)
    assert outcome.targets == {}


async def test_the_compatible_shortcut_needs_the_top_and_only_fitting_record() -> None:
    # Clef's top option is __new__ ("Bob" as a new person): not rescued
    new_top = Fixed({"Bob": {"Bob Roe": 0.15, NEW: 0.84}}, confidence=0.3)
    outcome, _ = await run_fixed(
        new_top, [entity("e0", "Bob", "Bob")], candidates("Bob Roe", "Zed Qux")
    )
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"
    # two records fit a first name: Clef has to decide it the normal way
    two = Fixed(
        {"Emma": {"Emma Thompson": 0.55, "Emma Watson": 0.35, NEW: 0.1}},
        confidence=0.2,
    )
    outcome, _ = await run_fixed(
        two,
        [entity("e0", "Emma", "Emma")],
        candidates("Emma Thompson", "Emma Watson"),
    )
    assert outcome.targets == {} and outcome.skipped[0].reason == "ambiguous_match"


async def test_a_new_entity_needs_the_accept_threshold_too() -> None:
    fixed = Fixed({"Bob": {"Ann": 0.3, NEW: 0.55}})
    outcome, _ = await run_fixed(fixed, [entity("e0", "Bob", "Bob")], candidates("Ann"))
    (skip,) = outcome.skipped
    assert skip.reason == "ambiguous_match" and skip.candidates[0] == (NEW, 0.55)


async def test_on_ambiguous_insert_makes_a_likely_new_entity_new() -> None:
    # not sure of the answer (low confidence), but "new" is what it says
    fixed = Fixed({"Emma Thompson": {"Ann": 0.05, NEW: 0.95}}, confidence=0.3)
    outcome, _ = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson")],
        candidates("Ann"),
        MatchConfig(on_ambiguous="insert"),
    )
    assert outcome.targets == {"e0": None} and outcome.skipped == ()


async def test_on_ambiguous_insert_skips_an_entity_that_is_not_likely_new() -> None:
    for table in (
        {"Ann": 0.5, NEW: 0.5},  # new is no more likely than accept
        {"Ann": 0.9, NEW: 0.1},  # a record is likely: low confidence only
    ):
        fixed = Fixed(
            {"Emma Thompson": table},
            confidence=0.3 if table["Ann"] > 0.8 else None,
        )
        outcome, _ = await run_fixed(
            fixed,
            [entity("e0", "Emma Thompson", "Emma Thompson")],
            candidates("Ann"),
            MatchConfig(on_ambiguous="insert"),
        )
        assert outcome.targets == {}
        (skip,) = outcome.skipped
        assert skip.reason == "ambiguous_match"


async def test_on_ambiguous_raise_names_the_entity_and_the_options() -> None:
    fixed = Fixed({"Emma Thompson": {"Ann": 0.5, "Zed": 0.3, NEW: 0.2}})
    with pytest.raises(AmbiguousMatchError, match="Emma Thompson") as caught:
        await run_fixed(
            fixed,
            [entity("e0", "Emma Thompson", "Emma Thompson")],
            candidates("Ann", "Zed"),
            MatchConfig(on_ambiguous="raise"),
        )
    assert caught.value.schema_name == "Person"
    assert caught.value.label == "Emma Thompson"
    assert caught.value.candidates == (("0", 0.5), ("1", 0.3), (NEW, 0.2))


async def test_a_missing_answer_is_ambiguous() -> None:
    # FakeTransport refuses an answer of the wrong type, so "no answer" is a choice
    # that gives every offered option probability 0
    def empty(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        options = payload["questions"]["match"]["criteria"]
        return clef_response(
            {"match": choice(NEW, {o: 0.0 for o in options}, 0.0)},
            model=payload["model"],
        )

    run = Run(transport=FakeTransport(empty))
    outcome = await match_entities(
        run,
        compile_schema(Person),
        TEXT,
        "Person",
        [entity("e0", "Bob", "Bob")],
        candidates("Ann"),
        ("name",),
        never,
    )
    (skip,) = outcome.skipped
    assert skip.reason == "ambiguous_match"


# --------------------------------------------------------------------------- #
# Collisions between entities
# --------------------------------------------------------------------------- #


async def test_a_collision_warns_and_the_loser_falls_to_the_next_option() -> None:
    fixed = Fixed(
        {
            "Emma Thompson": {"Ann": 0.9, "Zed": 0.05, NEW: 0.05},
            "Bob": {"Ann": 0.7, "Zed": 0.25, NEW: 0.05},
        }
    )
    outcome, run = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson"), entity("e1", "Bob", "Bob")],
        candidates("Ann", "Zed"),
        MatchConfig(accept=0.2),
    )
    assert outcome.targets == {"e0": "0", "e1": "1"}  # Bob falls to Zed
    (warning,) = [w for w in run.warnings if w.startswith("match_collision")]
    assert "Emma Thompson (e0)" in warning and "Bob (e1)" in warning
    assert "'0'" in warning and "p=0.90, 0.70" in warning
    assert "Bob (e1) fell back to the record '1'" in warning


async def test_a_collision_loser_with_no_good_option_is_ambiguous() -> None:
    fixed = Fixed(
        {
            "Emma Thompson": {"Ann": 0.9, NEW: 0.1},
            "Bob": {"Ann": 0.7, NEW: 0.3},
        }
    )
    outcome, run = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson"), entity("e1", "Bob", "Bob")],
        candidates("Ann"),
    )
    assert outcome.targets == {"e0": "0"}
    (skip,) = outcome.skipped
    assert skip.entity.label == "Bob" and skip.reason == "ambiguous_match"
    assert skip.candidates == (("0", 0.7), (NEW, 0.3))
    assert any(w.startswith("match_collision") for w in run.warnings)


async def test_a_weak_second_claim_is_no_collision() -> None:
    fixed = Fixed(
        {
            "Emma Thompson": {"Ann": 0.9, NEW: 0.1},
            "Bob": {"Ann": 0.4, NEW: 0.6},
        }
    )
    outcome, run = await run_fixed(
        fixed,
        [entity("e0", "Emma Thompson", "Emma Thompson"), entity("e1", "Bob", "Bob")],
        candidates("Ann"),
    )
    assert outcome.targets == {"e0": "0", "e1": None}
    assert run.warnings == []


async def test_merge_collision_merges_two_entities_that_are_one() -> None:
    fixed = Fixed(
        {
            "Emma Thompson": {"Ann": 0.9, NEW: 0.1},
            "Emma": {"Ann": 0.8, NEW: 0.2},
        }
    )
    asked: list[tuple[str, str]] = []

    async def merge(a: EntityInfo, b: EntityInfo) -> float | None:
        asked.append((a.key, b.key))
        return 0.9

    run = Run(transport=fixed.transport, match=MatchConfig(collision="merge"))
    # the merged entity is asked again, under its own label
    fixed.table["Emma Thompson"] = {"Ann": 0.95, NEW: 0.05}
    outcome = await match_entities(
        run,
        compile_schema(Person),
        TEXT,
        "Person",
        [
            entity("e0", "Emma Thompson", "Emma Thompson"),
            entity("e1", "Emma", "Emma", after="Later"),
        ],
        candidates("Ann"),
        ("name",),
        merge,
    )
    assert asked == [("e0", "e1")]
    (merged,) = outcome.entities
    assert merged.key == "e0" and merged.label == "Emma Thompson"
    assert [m.text for m in merged.mentions] == ["Emma Thompson", "Emma"]
    assert outcome.targets == {"e0": "0"}
    assert any(w.startswith("match_merged") for w in run.warnings)
    assert not any(w.startswith("match_collision") for w in run.warnings)


async def test_merge_collision_that_fails_the_check_warns_like_warn() -> None:
    fixed = Fixed(
        {
            "Emma Thompson": {"Ann": 0.9, NEW: 0.1},
            "Bob": {"Ann": 0.7, NEW: 0.3},
        }
    )

    async def merge(a: EntityInfo, b: EntityInfo) -> float | None:
        return 0.1

    run = Run(transport=fixed.transport, match=MatchConfig(collision="merge"))
    outcome = await match_entities(
        run,
        compile_schema(Person),
        TEXT,
        "Person",
        [entity("e0", "Emma Thompson", "Emma Thompson"), entity("e1", "Bob", "Bob")],
        candidates("Ann"),
        ("name",),
        merge,
    )
    assert len(outcome.entities) == 2
    assert outcome.targets == {"e0": "0"}
    assert any(w.startswith("match_collision") for w in run.warnings)


async def test_a_missing_merge_answer_does_not_merge() -> None:
    fixed = Fixed(
        {"Emma Thompson": {"Ann": 0.9, NEW: 0.1}, "Bob": {"Ann": 0.7, NEW: 0.3}}
    )

    async def merge(a: EntityInfo, b: EntityInfo) -> float | None:
        return None

    run = Run(transport=fixed.transport, match=MatchConfig(collision="merge"))
    outcome = await match_entities(
        run,
        compile_schema(Person),
        TEXT,
        "Person",
        [entity("e0", "Emma Thompson", "Emma Thompson"), entity("e1", "Bob", "Bob")],
        candidates("Ann"),
        ("name",),
        merge,
    )
    assert len(outcome.entities) == 2


# --------------------------------------------------------------------------- #
# Deletes
# --------------------------------------------------------------------------- #


class DeleteLog:
    """A handler for ``delete`` requests that records them."""

    def __init__(self, p: Any = 0.1) -> None:
        self.p = p
        self.seen: list[tuple[CallMeta, dict[str, Any]]] = []
        self.transport = FakeTransport(self.handle)

    def handle(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        self.seen.append((meta, payload))
        records = payload["state"]["records"]
        return clef_response(
            {
                qid: noul(
                    self.p(records[qid.removeprefix("delete.")])
                    if callable(self.p)
                    else self.p
                )
                for qid in payload["questions"]
            },
            model=payload["model"],
        )


async def test_the_delete_questions_are_batched_in_one_structured_state() -> None:
    log = DeleteLog(lambda record: 0.9 if record["name"] == "Zed" else 0.1)
    run = Run(transport=log.transport)
    records = candidates("Ann", "Zed")
    found = await flag_deletes(run, compile_schema(Person), TEXT, records, ("name",))
    assert found == {"0": 0.1, "1": 0.9}
    ((meta, payload),) = log.seen  # one request for both
    assert meta.kind == "delete" and meta.entity is None and meta.field == "0,1"
    assert set(payload["state"]) == {"input", "records"}
    assert payload["state"]["input"] == TEXT
    assert payload["state"]["records"] == {
        "r0": records[0].data,
        "r1": records[1].data,
    }
    assert list(payload["questions"]) == ["delete.r0", "delete.r1"]
    question = payload["questions"]["delete.r1"]
    assert question["type"] == "noul"
    assert question["instructions"] == (
        "Does `input` state that `records.r1` should be deleted \u2014 e.g. the "
        "person/item was removed, has died, or should no longer be tracked? A "
        "change to its details is NOT deletion."
    )


async def test_more_than_64_records_go_in_requests_of_64() -> None:
    log = DeleteLog()
    run = Run(transport=log.transport)
    records = candidates(*[f"Zed Qux{i:03d}" for i in range(150)])
    found = await flag_deletes(run, compile_schema(Person), TEXT, records, ("name",))
    assert len(found) == 150
    sizes = sorted(len(p["questions"]) for _, p in log.seen)
    assert sizes == [22, 64, 64]
    # every record is asked once, under its place in the whole list
    asked = sorted(
        int(qid.removeprefix("delete.r")) for _, p in log.seen for qid in p["questions"]
    )
    assert asked == list(range(150))
    for meta, payload in log.seen:
        assert len((meta.field or "").split(",")) == len(payload["questions"])


async def test_records_that_do_not_fit_one_request_are_split_by_the_budget() -> None:
    log = DeleteLog()
    run = Run(transport=log.transport, limits=Limits(clef_state_token_budget=2_000))
    records = candidates(
        *[f"Zed Qux{i}" for i in range(40)],
        extra={"relationship": "colleague " * 30},
    )
    await flag_deletes(run, compile_schema(Person), TEXT, records, ("name",))
    assert len(log.seen) > 1
    for _, payload in log.seen:
        assert estimate_tokens(payload["state"]) <= 2_000


async def test_a_huge_record_is_compacted_for_the_delete_question() -> None:
    log = DeleteLog()
    run = Run(transport=log.transport, limits=Limits(clef_state_token_budget=1_000))
    records = bulky(["Ann"])
    await flag_deletes(run, compile_schema(Person), TEXT, records, ("name",))
    ((_, payload),) = log.seen
    assert len(payload["state"]["records"]["r0"]["notes"]) == 3
    assert any(w.startswith("match_compacted") for w in run.warnings)


# --------------------------------------------------------------------------- #
# MatchConfig
# --------------------------------------------------------------------------- #


def test_the_defaults() -> None:
    config = MatchConfig()
    assert (config.accept, config.min_confidence, config.render_fields) == (0.6, 0.5, 4)
    assert config.accept_compatible == 0.5
    assert config.compatible_margin == 0.1
    assert config.confirm_accept == 0.7
    assert config.new_final == 0.9
    assert (config.collision, config.on_ambiguous, config.delete_threshold) == (
        "warn",
        "skip",
        0.85,
    )
    assert hash(config) == hash(MatchConfig())
    assert MatchConfig(accept=0.7) != config


@pytest.mark.parametrize(
    "bad",
    [
        {"accept": 0.0},
        {"accept": 1.5},
        {"min_confidence": -0.1},
        {"accept_compatible": 1.5},
        {"compatible_margin": -0.1},
        {"compatible_margin": 1.5},
        {"confirm_accept": 1.5},
        {"new_final": 0.0},
        {"new_final": 1.5},
        {"delete_threshold": float("nan")},
        {"render_fields": -1},
        {"render_fields": True},
        {"collision": "both"},
        {"on_ambiguous": "ignore"},
    ],
)
def test_invalid_settings_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MatchConfig(**bad)
