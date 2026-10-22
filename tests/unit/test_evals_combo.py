"""``evals combo`` and ``evals search``: spec runs, suites and the greedy search."""

import asyncio
import dataclasses
import importlib
import json
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from evals import cli, suites
from evals.cases import CASES
from evals.model import Case, CaseResult
from evals.runner import (
    Settings,
    SuiteResult,
    aggregate,
    combo_labels,
    missing_credentials,
    run_suite,
    usage_by_provider,
)
from evals.search import (
    DEFAULT_ORDER,
    GRID,
    Objective,
    SearchResult,
    Trial,
    compare,
    estimate_cost,
    leaderboard_markdown,
    load_previous,
    run_search,
)
from trustcall_system_one import CallMeta, Transport, TransportResponse
from trustcall_system_one._calibrate import shift_probability
from trustcall_system_one.combos import PROVIDERS, ComboSpec

SUBSET = [
    "readme_people",
    "colleague_bleed",
    "rename",
    "short_form",
    "computed_total",
    "bleed_update",
    "total_correction",
    "notes_update_contact",
    "relative_edit",
    "delete_request",
]
ATTRIBUTION = ("attribution", "stated", "null_check")
GATES = ("gate", "novelty", "list_dedup", "relative_edit")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
        "EXPERIENTIAL_API_KEY",
        "TYPESAFE_API_KEY",
        "TRUSTCALL_S1_PROVIDER",
        "TRUSTCALL_S1_CLEF_PROVIDER",
        "TRUSTCALL_S1_CLEF_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)


class Skewed:
    """A provider that is wrong (or shy) about some call kinds.

    ``flip`` turns every yes/no probability of the kinds into its opposite;
    ``shift`` moves its logit (a provider that is shy about its yeses).
    """

    def __init__(
        self,
        inner: Transport,
        kinds: tuple[str, ...] = (),
        *,
        flip: bool = False,
        shift: float = 0.0,
    ) -> None:
        self.inner, self.kinds, self.flip, self.shift = inner, set(kinds), flip, shift

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        reply = await self.inner.run(model, payload, meta=meta, timeout=timeout)
        if meta.kind not in self.kinds or "answers" not in reply.body:
            return reply
        answers = {}
        for qid, answer in reply.body["answers"].items():
            if answer.get("type") == "noul":
                p = answer["noul"]
                p = 1 - p if self.flip else shift_probability(p, self.shift)
                answer = {**answer, "noul": p}
            answers[qid] = answer
        return dataclasses.replace(reply, body={**reply.body, "answers": answers})


def scripted(
    flip: Mapping[str, tuple[str, ...]] | None = None,
    shy: Mapping[str, tuple[tuple[str, ...], float]] | None = None,
) -> Callable[[Case], Mapping[str, Transport]]:
    """Every provider answers like the case's world, except as said."""

    def members(case: Case) -> Mapping[str, Transport]:
        assert case.world is not None
        out: dict[str, Transport] = {}
        for name in PROVIDERS:
            kinds, shift = (shy or {}).get(name, ((), 0.0))
            out[name] = Skewed(
                case.world().transport(),
                (flip or {}).get(name, ()) + kinds,
                flip=name in (flip or {}),
                shift=shift,
            )
        return out

    return members


def subset() -> list[Case]:
    return [c for c in CASES if c.id in SUBSET]


def evaluator(
    members: Callable[[Case], Mapping[str, Transport]] | None = None,
    cases: list[Case] | None = None,
) -> Callable[[ComboSpec, str], Any]:
    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        settings = Settings(
            overrides=spec.options, combo=spec, members=members, tag=label
        )
        return await run_suite(cases or subset(), settings, config=label)

    return evaluate


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #


def test_the_default_suite_is_the_29_dev_cases() -> None:
    assert [c.id for c in suites.load_suite()] == [c.id for c in CASES]
    assert len(suites.load_suite("default")) == 29
    assert cli.select_cases(["rename"], None, "default")[0].id == "rename"
    with pytest.raises(ValueError, match="unknown suite"):
        suites.load_suite("nope")


def test_a_missing_holdout_file_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "evals.holdout_cases":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(
        ValueError, match=r"evals/holdout_cases\.py, which does not exist"
    ):
        suites.load_suite("holdout")
    assert cli.main(["list", "--suite", "holdout"]) == 2
    assert "holdout_cases.py" in capsys.readouterr().err
    assert cli.main(["run", "--suite", "holdout", "--quiet"]) == 2


def test_a_holdout_file_is_loaded_when_it_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("evals.holdout_cases")
    fake.CASES = [CASES[0]]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "evals.holdout_cases", fake)
    assert [c.id for c in suites.load_suite("holdout")] == [CASES[0].id]
    assert [c.id for c in cli.select_cases(None, None, "holdout")] == [CASES[0].id]
    empty = types.ModuleType("evals.holdout_cases")
    empty.CASES = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "evals.holdout_cases", empty)
    with pytest.raises(ValueError, match="no cases"):
        suites.load_suite("holdout")


def test_the_suite_is_recorded_in_the_results(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert cli.main(["run", "--cases", "rename", "--quiet", "--out", str(out)]) == 0
    assert json.loads((out / "results.json").read_text())["suite"] == "default"


# --------------------------------------------------------------------------- #
# combo
# --------------------------------------------------------------------------- #

SPEC: dict[str, Any] = {
    "name": "mix",
    "extraction": "cloudflare",
    "extraction_by_type": {"date": "typesafe", "number": "typesafe"},
    "decision": "cloudflare",
    "by_kind": {"attribution": "decisions", "gate": "typesafe"},
    "options": {"verify.attribution_keep": 0.7},
}


def write_spec(
    tmp_path: Path, spec: Mapping[str, Any] = SPEC, name: str = "spec.json"
) -> Path:
    file = tmp_path / name
    file.write_text(json.dumps(spec))
    return file


def test_a_combo_runs_offline_and_records_its_spec_and_labels(tmp_path: Path) -> None:
    out = tmp_path / "out"
    spec = write_spec(tmp_path)
    code = cli.main(
        ["combo", "--spec", str(spec), "--tag", "t1", "--quiet", "--out", str(out)]
    )
    assert code == 0
    data = json.loads((out / "results.json").read_text())
    assert data["aggregate"]["passed"] == data["aggregate"]["cases"] == 29
    assert data["suite"] == "default" and data["tag"] == "t1"
    combo = data["combo"]
    assert combo["spec"]["by_kind"] == {"attribution": "decisions", "gate": "typesafe"}
    assert combo["spec"]["options"] == {"verify.attribution_keep": 0.7}
    assert combo["labels"]["extraction"].startswith(
        "cloudflare clairic (date: typesafe"
    )
    assert "attribution: experiential decisions" in combo["labels"]["decisions"]
    assert combo["companies"] == [
        "Cloudflare",
        "Experiential Labs",
        "OpenAI",
        "TypeSafe",
    ]
    assert data["overrides"] == {"verify.attribution_keep": 0.7}
    assert "Combination:" in (out / "summary.md").read_text()
    # the split Clairic calls are logged per provider
    assert data["cases"][0]["by_provider"].keys() >= {"cloudflare", "typesafe"}


def test_combo_set_overrides_the_specs_options_and_is_recorded(tmp_path: Path) -> None:
    out = tmp_path / "out"
    spec = write_spec(tmp_path)
    cli.main(
        [
            "combo",
            "--spec",
            str(spec),
            "--quiet",
            "--out",
            str(out),
            "--cases",
            "rename",
            "--set",
            "gates.add_threshold=0.5",
        ]
    )
    data = json.loads((out / "results.json").read_text())
    assert data["combo"]["spec"]["options"]["gates.add_threshold"] == 0.5
    assert data["tag"] == "mix"  # the spec's name when there is no --tag


def test_record_goes_to_a_directory_named_by_the_tag_and_replays_with_calibration(
    tmp_path: Path,
) -> None:
    spec = write_spec(tmp_path)
    record = tmp_path / "cassettes"
    first = tmp_path / "first"
    args = ["combo", "--spec", str(spec), "--quiet", "--cases", "colleague_bleed"]
    assert (
        cli.main([*args, "--tag", "base", "--record", str(record), "--out", str(first)])
        == 0
    )
    assert list((record / "base").glob("*.jsonl"))  # DIR/NAME
    # replay the raw answers: the same combination reproduces the same result
    again = tmp_path / "again"
    assert cli.main([*args, "--replay", str(record / "base"), "--out", str(again)]) == 0
    one = json.loads((first / "results.json").read_text())["aggregate"]
    two = json.loads((again / "results.json").read_text())["aggregate"]
    assert (one["passed"], one["requests"]) == (two["passed"], two["requests"])
    # a calibration is applied on top of the recorded raw answers, per provider
    calibrated = write_spec(
        tmp_path,
        {**SPEC, "calibration": {"decisions": {"attribution": 0.5}}},
        name="calibrated.json",
    )
    third = tmp_path / "third"
    replay = ["--replay", str(record / "base"), "--out", str(third)]
    cli.main(
        [
            "combo",
            "--spec",
            str(calibrated),
            "--quiet",
            "--cases",
            "colleague_bleed",
            *replay,
        ]
    )
    cases = json.loads((third / "results.json").read_text())["cases"]
    warnings = [w for c in cases for w in c["warnings"]]
    assert any(w.startswith("calibration_applied: attribution") for w in warnings)


def test_a_live_combo_needs_the_credentials_of_its_providers_and_never_says_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    jev = write_spec(tmp_path, {"extraction": "typesafe", "decision": "typesafe"})
    code = cli.main(
        ["combo", "--spec", str(jev), "--live", "--quiet", "--out", str(tmp_path)]
    )
    err = capsys.readouterr().err
    assert code == 2 and "TYPESAFE_API_KEY" in err and "CLOUDFLARE" not in err
    assert "the text goes to TypeSafe" in err
    mixed = write_spec(tmp_path)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-secret-1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "key-secret-2")
    code = cli.main(
        ["combo", "--spec", str(mixed), "--live", "--quiet", "--out", str(tmp_path)]
    )
    err = capsys.readouterr().err
    assert code == 2 and "CLOUDFLARE_API_TOKEN" in err and "EXPERIENTIAL_API_KEY" in err
    assert "secret" not in err
    spec = ComboSpec.model_validate(SPEC)
    assert missing_credentials(spec) == ["CLOUDFLARE_API_TOKEN", "EXPERIENTIAL_API_KEY"]


def test_a_combo_is_not_mixed_with_the_provider_settings() -> None:
    settings = Settings(combo=ComboSpec(), provider="typesafe")
    with pytest.raises(ValueError, match="names its own providers"):
        asyncio.run(run_suite(subset(), settings))


def test_the_environment_does_not_change_a_combo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "decisions")
    suite = asyncio.run(
        run_suite(subset()[:2], Settings(combo=ComboSpec(), members=scripted()))
    )
    assert suite.decisions.startswith("cloudflare clef")
    assert all(r.passed for r in suite.results)


def test_labels_and_usage_by_provider() -> None:
    extraction, decisions = combo_labels(ComboSpec.model_validate(SPEC))
    assert extraction == "cloudflare clairic (date: typesafe jev; number: typesafe jev)"
    assert decisions.startswith("cloudflare clef (attribution: experiential decisions")
    suite = asyncio.run(
        run_suite(
            subset()[:3],
            Settings(combo=ComboSpec.model_validate(SPEC), members=scripted()),
        )
    )
    for r in suite.results:
        assert r.by_provider and r.output_tokens >= 0
        assert sum(p["input_tokens"] for p in r.by_provider.values()) == r.input_tokens
        assert sum(p["requests"] for p in r.by_provider.values()) == r.requests
    assert usage_by_provider([]) == {}


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #


def obj(
    passed: int, f1: float = 1.0, va: float = 1.0, bleed: int = 0, wd: int = 0
) -> Objective:
    return Objective(passed, f1, va, bleed, wd)


def test_the_objective_is_lexicographic() -> None:
    best = obj(10, 0.9, 0.9)
    assert compare(obj(11, 0.1, 0.1), best)[0]  # passes first
    assert compare(obj(10, 0.91, 0.1), best)[0]  # then F1
    assert compare(obj(10, 0.9, 0.91), best)[0]  # then value accuracy
    assert not compare(obj(10, 0.9, 0.9), best)[0]  # a tie is not an improvement
    assert not compare(obj(9, 1.0, 1.0), best)[0]
    assert compare(obj(10, 0.9, 0.9, bleed=0), obj(10, 0.9, 0.9, bleed=1))[
        0
    ]  # fewer bleed
    assert compare(obj(10, 0.9, 0.9, wd=0), obj(10, 0.9, 0.9, wd=2))[0]


def test_bleed_or_wrong_deletes_going_up_is_a_veto() -> None:
    best = obj(10, 0.9, 0.9)
    ok, why = compare(obj(29, 1.0, 1.0, bleed=1), best)
    assert not ok and why == "veto: bleed 0 -> 1"
    ok, why = compare(obj(29, 1.0, 1.0, wd=1), best)
    assert not ok and why == "veto: wrong deletes 0 -> 1"
    assert compare(obj(11, bleed=1), obj(10, bleed=1))[0]  # not up: not a veto


def test_the_objective_reads_an_aggregate() -> None:
    agg = {
        "passed": 3,
        "op_f1": 0.5,
        "value_accuracy": None,
        "bleed": 1,
        "wrong_deletes": 2,
    }
    assert Objective.of(agg) == Objective(3, 0.5, 1.0, 1, 2)
    suite = asyncio.run(
        run_suite(subset()[:2], Settings(combo=ComboSpec(), members=scripted()))
    )
    assert aggregate(suite.results)["op_f1"] == 1.0


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def result(by_provider: dict[str, dict[str, int]], reported: float = 0.0) -> CaseResult:
    return CaseResult(case="c", by_provider=by_provider, reported_cost_usd=reported)


def test_cost_is_estimated_from_tokens_and_prices() -> None:
    rows = [
        result(
            {"typesafe": {"requests": 1, "input_tokens": 2_000_000, "output_tokens": 0}}
        ),
        result(
            {
                "cloudflare": {
                    "requests": 1,
                    "input_tokens": 1_000_000,
                    "output_tokens": 500_000,
                }
            }
        ),
    ]
    prices = {"typesafe": (1.0, 2.0), "cloudflare": (0.1, 0.4)}
    usd, partial = estimate_cost(rows, prices)
    assert usd == pytest.approx(2.0 + 0.1 + 0.2) and partial is False


def test_an_unpriced_provider_makes_the_estimate_partial_unless_it_reported() -> None:
    use = {"experiential": {"requests": 1, "input_tokens": 100, "output_tokens": 10}}
    assert estimate_cost([result(use)], {}) == (0.0, True)
    usd, partial = estimate_cost([result(use, reported=0.5)], {})
    assert (usd, partial) == (0.5, False)  # it reported its own price
    priced = estimate_cost([result(use, reported=0.5)], {"decisions": (1.0, 1.0)})
    assert priced[1] is True or priced[0] == 0.5  # an unrelated price changes nothing


# --------------------------------------------------------------------------- #
# The search, on scripted providers that are good at different things
# --------------------------------------------------------------------------- #

BAD_AT = {
    "cloudflare": ATTRIBUTION + GATES,  # wrong about both
    "decisions": GATES,  # good at attribution
    "typesafe": ATTRIBUTION,  # good at the change gates
    "experiential-clef": ATTRIBUTION + GATES,
}


def test_the_search_picks_the_provider_that_is_better_at_each_kind(
    tmp_path: Path,
) -> None:
    lines: list[str] = []
    result = asyncio.run(
        run_search(
            ComboSpec(),
            evaluator(scripted(flip=BAD_AT)),
            providers=["cloudflare", "decisions", "typesafe"],
            order=["attribution", "gates"],
            log=lines.append,
        )
    )
    best = result.best.spec
    for kind in ATTRIBUTION:
        assert best.by_kind[kind] == "decisions"
    for kind in GATES:
        assert best.by_kind[kind] == "typesafe"
    assert result.best.aggregate["passed"] == len(SUBSET)
    base = result.trials[0]
    assert base.outcome == "base" and base.aggregate["passed"] < len(SUBSET)
    assert [t.change for t in result.trials] == [
        "base",
        "attribution=decisions",
        "attribution=typesafe",
        "gates=decisions",
        "gates=typesafe",
    ]
    assert [t.outcome for t in result.trials] == [
        "base",
        "kept",
        "veto",  # typesafe is wrong about attribution: facts bleed onto other records
        "rejected",
        "kept",
    ]
    board = leaderboard_markdown(result)
    for column in (
        "pass",
        "op F1",
        "value acc",
        "bleed",
        "wrong del",
        "requests",
        "tokens",
        "cost",
        "s/case",
    ):
        assert column in board
    assert board.count("\n| ") >= len(result.trials)
    assert "best kept" in board and "attribution=typesafe" in board


def test_the_search_goes_through_the_order_and_a_better_provider_for_one_type() -> None:
    result = asyncio.run(
        run_search(
            ComboSpec(),
            evaluator(scripted(flip={"cloudflare": ATTRIBUTION})),
            providers=["cloudflare", "typesafe"],
            order=["attribution", "extract:date", "extract:text"],
        )
    )
    assert result.best.spec.by_kind["attribution"] == "typesafe"
    changes = [t.change for t in result.trials]
    assert "extract:date=typesafe" in changes and "extract:text=typesafe" in changes
    # they tie with the base on the scripted worlds, so the extraction stays put
    assert result.best.spec.extraction_by_type == {}


def test_a_candidate_that_raises_bleed_is_vetoed_whatever_else_it_gains() -> None:
    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        great = spec.by_kind.get("gate") == "typesafe"
        results = [
            CaseResult(
                case=f"c{i}",
                n_expected=1,
                matched_expected=1,
                n_actual=1,
                matched_actual=1,
                violations=[("bleed", "x")] if great and i == 0 else [],
            )
            for i in range(3)
        ]
        # c1 and c2 only pass with the "good" gate; c0 bleeds when it is on
        for r in results[1:]:
            if not great:
                r.matched_expected = 0
        return SuiteResult(label, {}, results, [], "offline", combo=spec)

    result = asyncio.run(
        run_search(
            ComboSpec(), evaluate, providers=["cloudflare", "typesafe"], order=["gates"]
        )
    )
    veto = result.trials[1]
    assert veto.outcome == "veto" and veto.note.startswith("veto: bleed 0 -> 1")
    assert result.best is result.trials[0]  # the base stays


def test_a_failed_case_is_reported_and_never_retried() -> None:
    sent: list[str] = []

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        sent.append(spec.key())
        boom = CaseResult(case="flaky", error="UncertainOutcomeError: timed out")
        ok = CaseResult(
            case="fine", n_expected=1, matched_expected=1, n_actual=1, matched_actual=1
        )
        return SuiteResult(label, {}, [boom, ok], [], "live", combo=spec)

    result = asyncio.run(
        run_search(
            ComboSpec(),
            evaluate,
            providers=["cloudflare", "typesafe"],
            order=["gates", "coref"],
        )
    )
    assert len(sent) == len(set(sent)) == len(result.trials)  # each combination once
    assert result.trials[0].failures == [
        {
            "case": "flaky",
            "error": "UncertainOutcomeError: timed out",
            "missing": [],
            "extra": [],
            "violations": [],
        }
    ]
    board = leaderboard_markdown(result)
    assert "flaky: UncertainOutcomeError: timed out" in board
    assert "never retried" in board


def test_a_trial_that_raises_is_recorded_and_the_search_goes_on() -> None:
    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        if spec.by_kind.get("gate") == "typesafe":
            raise RuntimeError("the network is down")
        return SuiteResult(label, {}, [CaseResult(case="c", n_expected=0)], [], "live")

    result = asyncio.run(
        run_search(
            ComboSpec(),
            evaluate,
            providers=["typesafe", "decisions"],
            order=["gates", "coref"],
        )
    )
    error = next(t for t in result.trials if t.outcome == "error")
    assert error.error == "RuntimeError: the network is down"
    assert any(t.change == "coref=typesafe" for t in result.trials)  # it went on
    assert "the network is down" in leaderboard_markdown(result)


def test_a_failing_base_stops_the_search() -> None:
    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        raise RuntimeError("no credentials")

    with pytest.raises(RuntimeError, match="no credentials"):
        asyncio.run(run_search(ComboSpec(), evaluate, providers=["typesafe"]))


def flat(cost: float = 0.0) -> Callable[[ComboSpec, str], Any]:
    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        usage = {
            "cloudflare": {
                "requests": 1,
                "input_tokens": int(cost * 1e6),
                "output_tokens": 0,
            }
        }
        row = CaseResult(case="c", n_expected=0, by_provider=usage)
        return SuiteResult(label, {}, [row], [], "offline", combo=spec)

    return evaluate


def test_max_runs_stops_the_search_between_trials() -> None:
    result = asyncio.run(
        run_search(ComboSpec(), flat(), providers=list(PROVIDERS), max_runs=3)
    )
    assert len(result.trials) == 3
    assert result.stopped == "--max-runs 3 reached"


def test_the_budget_stops_before_a_trial_that_could_overspend() -> None:
    result = asyncio.run(
        run_search(
            ComboSpec(),
            flat(cost=1.0),
            providers=list(PROVIDERS),
            prices={"cloudflare": (1.0, 1.0)},
            budget_usd=3.5,
        )
    )
    assert len(result.trials) == 3  # $1 each: a fourth could take it to $4
    assert result.spent_usd == pytest.approx(3.0)
    assert result.stopped.startswith("--budget-usd 3.5")


def test_a_resumed_search_reuses_what_was_evaluated(tmp_path: Path) -> None:
    calls: list[str] = []

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        calls.append(spec.key())
        return await evaluator(scripted(flip=BAD_AT))(spec, label)  # type: ignore[no-any-return]

    def go(**kw: Any) -> SearchResult:
        return asyncio.run(
            run_search(
                ComboSpec(),
                evaluate,
                providers=["cloudflare", "decisions", "typesafe"],
                order=["attribution", "gates"],
                **kw,
            )
        )

    from evals.search import write_search

    first = go(max_runs=3)
    write_search(tmp_path, first, {"suite": "default", "cases": SUBSET})
    assert len(calls) == 3 and first.stopped
    header, previous = load_previous(tmp_path)
    assert header["cases"] == SUBSET and len(previous) == 3
    calls.clear()
    second = go(previous=previous)
    assert len(calls) == 2  # only the two that were left
    assert [t.cached for t in second.trials] == [True, True, True, False, False]
    assert second.best.spec.by_kind["gate"] == "typesafe"
    fresh = go()
    assert [t.outcome for t in fresh.trials] == [t.outcome for t in second.trials]


def test_the_search_report_files_are_written_and_resumable(tmp_path: Path) -> None:
    out = tmp_path / "s"
    cases = ["search", "--cases", "colleague_bleed,rename", "--quiet"]
    where = ["--providers", "cloudflare,typesafe"]
    assert cli.main([*cases, "--out", str(out), "--order", "gates", *where]) == 0
    assert {p.name for p in out.iterdir()} >= {
        "search.json",
        "leaderboard.md",
        "best.json",
        "trials",
    }
    data = json.loads((out / "search.json").read_text())
    assert data["suite"] == "default" and data["cases"] == ["colleague_bleed", "rename"]
    assert [t["change"] for t in data["trials"]] == ["base", "gates=typesafe"]
    assert (out / "trials").exists() and len(list((out / "trials").iterdir())) == 2
    best = json.loads((out / "best.json").read_text())
    assert ComboSpec.model_validate(best).providers() == ("cloudflare",)
    # resuming on the same cases does not run the two trials again
    code = cli.main([*cases, "--resume", str(out), "--order", "gates,coref", *where])
    assert code == 0
    data = json.loads((out / "search.json").read_text())
    assert [t["cached"] for t in data["trials"]] == [True, True, False]
    # on other cases it refuses
    assert (
        cli.main(["search", "--cases", "rename", "--resume", str(out), "--quiet"]) == 2
    )


def test_search_is_dev_suite_only_and_live_needs_named_providers(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["search", "--suite", "holdout", "--quiet"]) == 2
    assert "dev suite only" in capsys.readouterr().err
    assert cli.main(["search", "--live", "--quiet"]) == 2
    assert "--live needs --providers" in capsys.readouterr().err
    code = cli.main(
        ["search", "--live", "--providers", "typesafe", "--budget-usd", "1", "--quiet"]
    )
    err = capsys.readouterr().err
    assert code == 2 and "--price" in err
    assert cli.main(["search", "--price", "gpt=1:2", "--quiet"]) == 2
    assert cli.main(["search", "--grid", "1:0:1", "--quiet"]) == 2
    assert cli.main(["search", "--order", "nonsense", "--quiet"]) == 2
    assert cli.main(["search", "--providers", "gpt", "--quiet"]) == 2


def test_the_default_order_and_grid() -> None:
    assert DEFAULT_ORDER[:7] == (
        "presence",
        "coref",
        "match",
        "attribution",
        "gates",
        "clef_fields",
        "verify",
    )
    assert [o for o in DEFAULT_ORDER if o.startswith("extract:")] == [
        f"extract:{t}"
        for t in ("text", "number", "date", "time", "datetime", "duration", "list")
    ]
    assert GRID == (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


# --------------------------------------------------------------------------- #
# Calibration search
# --------------------------------------------------------------------------- #


def test_the_calibration_search_finds_the_shift_that_undoes_a_shy_provider() -> None:
    # typesafe's attribution and gates are shy by 2.5 logits: its yeses of 0.95 come
    # out as 0.61, below the 0.75 the engine keeps. +1.0 on each brings them back.
    members = scripted(shy={"typesafe": (ATTRIBUTION + GATES, -2.5)})
    base = ComboSpec.model_validate(
        {"decision": "typesafe", "extraction": "cloudflare"}
    )
    result = asyncio.run(
        run_search(
            base,
            evaluator(members),
            providers=["typesafe"],
            order=[],
            calibrate=True,
            calibrate_kinds=["attribution", "gate"],
        )
    )
    cal = result.best.spec.calibration["typesafe"]
    assert cal == {"attribution": 1.0, "gate": 1.0}  # the first shift that helps
    assert result.trials[0].aggregate["passed"] < len(SUBSET)
    assert result.best.aggregate["passed"] == len(SUBSET)
    shifts = {t.change for t in result.trials if t.dimension.startswith("calibrate")}
    assert "cal typesafe.attribution+1.5" in shifts
    assert "cal typesafe.attribution+0" not in shifts  # the current value is skipped
    assert len(shifts) >= 6  # the grid


def test_calibration_is_searched_on_the_dev_suite_only_through_the_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["search", "--calibrate", "--suite", "holdout", "--quiet"]) == 2
    assert "dev suite only" in capsys.readouterr().err


def test_a_calibration_that_the_spec_refuses_is_not_run() -> None:
    base = ComboSpec.model_validate({"only": "cloudflare"})
    result = asyncio.run(
        run_search(base, flat(), providers=["typesafe"], order=["gates"])
    )
    assert [t.change for t in result.trials] == ["base"]
    assert result.not_run and "only='cloudflare'" in result.not_run[0]
    assert "Not run" in leaderboard_markdown(result)


def test_trials_round_trip_through_json() -> None:
    suite = asyncio.run(
        run_search(ComboSpec(), flat(), providers=["typesafe"], order=["coref"])
    )
    for trial in suite.trials:
        again = Trial.from_dict(json.loads(json.dumps(trial.as_dict())))
        assert again.spec.key() == trial.spec.key() and again.change == trial.change


def test_search_files_never_hold_a_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals.search import write_search

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-secret-123")

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        if spec.by_kind.get("coref") == "typesafe":
            raise RuntimeError(
                "failed: https://api.cloudflare.com/client/v4/accounts/"
                "acct-secret-123/ai/run/x"
            )
        return SuiteResult(label, {}, [CaseResult(case="c", n_expected=0)], [], "live")

    result = asyncio.run(
        run_search(ComboSpec(), evaluate, providers=["typesafe"], order=["coref"])
    )
    assert any(t.outcome == "error" for t in result.trials)
    write_search(tmp_path, result, {"suite": "default", "cases": ["c"]})
    for file in tmp_path.iterdir():
        assert "acct-secret-123" not in file.read_text(), file.name
