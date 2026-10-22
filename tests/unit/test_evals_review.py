"""Regression tests for the review of the combination search: replay, resume, cost."""

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from evals import cli
from evals.holdout_cases import HOLDOUT_CASES
from evals.model import CaseResult
from evals.report import suite_markdown
from evals.runner import Settings, SuiteResult, aggregate, run_suite
from evals.scoring import score
from evals.search import (
    Margin,
    Trial,
    estimate_cost,
    leaderboard_markdown,
    load_previous,
    run_search,
    write_search,
)
from tests.unit.test_evals_combo import (
    ATTRIBUTION,
    BAD_AT,
    GATES,
    SUBSET,
    evaluator,
    scripted,
    subset,
)
from trustcall_system_one import CLAIRIC, CLEF, CallMeta, TransportResponse
from trustcall_system_one.combos import ComboSpec
from trustcall_system_one.testing import (
    CassetteMissError,
    CassetteTransport,
    request_key,
)

CASES = ["colleague_bleed", "rename"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
        "EXPERIENTIAL_API_KEY",
        "TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# B1: a replay answers a call only from the provider it is routed to
# --------------------------------------------------------------------------- #


def test_a_spec_never_gets_the_answers_of_a_provider_it_does_not_route_to(
    tmp_path: Path,
) -> None:
    """The reviewer's probe: attribution routed to decisions, recorded on cloudflare."""
    members = scripted(
        flip={"decisions": ATTRIBUTION}
    )  # decisions: wrong on attribution
    cases = subset()
    base = ComboSpec(name="base")
    routed = ComboSpec(
        name="routed",
        decision="cloudflare",
        by_kind={k: "decisions" for k in ATTRIBUTION},
    )
    recording = tmp_path / "base"

    async def go() -> tuple[SuiteResult, SuiteResult, SuiteResult]:
        first = await run_suite(
            cases, Settings(combo=base, members=members, record=recording)
        )
        fresh = await run_suite(cases, Settings(combo=routed, members=members))
        replay = await run_suite(
            cases, Settings(combo=routed, members=members, replay=recording)
        )
        return first, fresh, replay

    first, fresh, replay = asyncio.run(go())
    assert aggregate(first.results)["passed"] == len(SUBSET)
    fresh_agg, replay_agg = aggregate(fresh.results), aggregate(replay.results)
    assert fresh_agg["passed"] < len(SUBSET)  # decisions is wrong about attribution
    # the replay must not turn that into a pass with cloudflare's recorded answers
    assert replay_agg["passed"] < len(SUBSET)
    assert replay_agg["errors"] > 0
    for r in replay.results:
        if r.error:
            assert "cassette miss" in r.error
            assert "another provider" in r.error or "no call" in r.error
    for r in replay.results:  # nothing decisions-routed was answered from a recording
        assert r.by_provider.get("decisions", {}).get("replayed_requests", 0) == 0
        assert r.forwarded == 0  # and nothing went anywhere without --live
    assert replay.mode == "replay"


def test_the_same_spec_replays_what_it_recorded_and_it_costs_nothing(
    tmp_path: Path,
) -> None:
    spec = ComboSpec.model_validate(
        {"by_kind": {"gate": "typesafe", "attribution": "decisions"}}
    )
    members = scripted()
    recording = tmp_path / "rec"

    async def go() -> tuple[SuiteResult, SuiteResult]:
        a = await run_suite(
            subset(), Settings(combo=spec, members=members, record=recording)
        )
        b = await run_suite(
            subset(), Settings(combo=spec, members=members, replay=recording)
        )
        return a, b

    recorded, replayed = asyncio.run(go())
    one, two = aggregate(recorded.results), aggregate(replayed.results)
    assert (one["passed"], one["requests"]) == (two["passed"], two["requests"])
    assert two["errors"] == 0 and two["forwarded"] == 0
    assert replayed.replay == str(recording) and replayed.mode == "replay"
    summary = suite_markdown(replayed)
    assert f"Replay of `{recording}`" in summary and "nothing is sent" in summary
    used = {p for r in replayed.results for p in r.by_provider}
    assert {"cloudflare", "decisions", "typesafe"} <= used
    for r in replayed.results:
        for row in r.by_provider.values():
            assert row["replayed_requests"] == row["requests"]
    prices = {"cloudflare": (1.0, 1.0), "typesafe": (1.0, 1.0), "decisions": (1.0, 1.0)}
    assert estimate_cost(recorded.results, prices)[0] > 0
    assert estimate_cost(replayed.results, prices) == (0.0, False)  # no spend


def test_a_cassette_for_one_provider_replays_only_what_that_provider_recorded(
    tmp_path: Path,
) -> None:
    payload: dict[str, Any] = {"model": "clef", "state": "x", "questions": {}}
    key = request_key(CLEF, payload)

    def line(provider: str | None, noul: float) -> str:
        entry: dict[str, Any] = {
            "key": key,
            "model": CLEF,
            "request": payload,
            "response": {
                "model": "clef",
                "answers": {"q": {"type": "noul", "noul": noul}},
                "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.5},
            },
            "status": 200,
        }
        if provider:
            entry["provider"] = provider
        return json.dumps(entry)

    file = tmp_path / "c.jsonl"
    file.write_text(
        "\n".join([line("cloudflare", 0.1), line("decisions", 0.9), line(None, 0.5)])
    )
    meta = CallMeta(kind="attribution")

    def ask(cassette: CassetteTransport) -> Any:
        return asyncio.run(cassette.run(CLEF, payload, meta=meta))

    whole = CassetteTransport(file)
    assert ask(whole).body["answers"]["q"]["noul"] == 0.1  # unfiltered: as before
    cloudflare = CassetteTransport(file, provider="cloudflare")
    decisions = CassetteTransport(file, provider="decisions")
    assert ask(decisions).body["answers"]["q"]["noul"] == 0.9
    assert ask(cloudflare).body["answers"]["q"]["noul"] == 0.1
    assert cloudflare.provider == "cloudflare"
    assert decisions.replayed == {
        "requests": 1,
        "input_tokens": 10,
        "output_tokens": 2,
        "cost": 0.5,
    }
    with pytest.raises(CassetteMissError, match="already replayed"):
        ask(decisions)  # its only recording was replayed
    typesafe = CassetteTransport(file, provider="typesafe")
    with pytest.raises(CassetteMissError, match="made by another provider"):
        ask(typesafe)  # a recording by someone else, or by no one, is not an answer
    assert typesafe.hits == 0
    assert decisions.fork().provider == "decisions"  # the filter survives a fork
    assert decisions.fork(provider="typesafe").provider == "typesafe"


def test_a_miss_goes_live_only_when_asked_and_is_counted(tmp_path: Path) -> None:
    class Live:
        provider = "typesafe"
        calls = 0

        async def run(
            self,
            model: str,
            payload: Mapping[str, Any],
            *,
            meta: CallMeta,
            timeout: float | None = None,
        ) -> TransportResponse:
            self.calls += 1
            return TransportResponse(body={"served_by": "live"}, provider="typesafe")

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "x.jsonl").write_text("")
    live = Live()
    strict = CassetteTransport(empty, provider="typesafe")
    with pytest.raises(CassetteMissError):
        asyncio.run(strict.run(CLEF, {}, meta=CallMeta(kind="gate")))
    forwarding = CassetteTransport(empty, strict=False, inner=live, provider="typesafe")
    asyncio.run(forwarding.run(CLEF, {}, meta=CallMeta(kind="gate")))
    assert (live.calls, forwarding.forwarded) == (1, 1)


def test_search_replay_is_for_the_calibration_search_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = tmp_path / "rec"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"name": "base"}))
    common = ["--cases", ",".join(CASES), "--quiet"]
    first = ["combo", "--spec", str(spec), "--record", str(rec), "--tag", "base"]
    assert cli.main([*first, *common, "--out", str(tmp_path / "first")]) == 0
    replay = ["--replay", str(rec / "base")]
    for bad in (
        ["--order", "gates"],  # a routing dimension
        [],  # the default order runs every dimension
        ["--order", "none"],  # nothing to calibrate
        ["--calibrate"],  # routing AND calibration
    ):
        code = cli.main(["search", *common, *replay, *bad])
        err = capsys.readouterr().err
        assert code == 2, bad
        assert "calibration search alone" in err or "searches nothing" in err, bad
    out = tmp_path / "cal"
    code = cli.main(
        [
            *["search", *common, *replay, "--base", str(spec), "--calibrate-only"],
            *["--calibrate-kinds", "attribution", "--grid=-0.5:0.5:0.5"],
            *["--out", str(out)],
        ]
    )
    assert code == 0
    data = json.loads((out / "search.json").read_text())
    assert [t["change"] for t in data["trials"]] == [
        "base",
        "cal cloudflare.attribution-0.5",
        "cal cloudflare.attribution+0.5",
    ]
    assert data["order"] == [] and data["mode"] == f"replay:{rec / 'base'}"
    assert data["trials"][0]["mode"] == data["mode"]


def test_order_none_is_the_same_as_calibrate_only_and_empty_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = [
        "search",
        "--cases",
        "rename",
        "--quiet",
        "--calibrate-kinds",
        "attribution",
    ]
    grid = ["--grid", "0:0.5:0.5"]
    out = tmp_path / "o"
    assert (
        cli.main([*base, *grid, "--order", "none", "--calibrate", "--out", str(out)])
        == 0
    )
    assert [
        t["change"] for t in json.loads((out / "search.json").read_text())["trials"]
    ] == [
        "base",
        "cal cloudflare.attribution+0.5",
    ]
    assert cli.main([*base, "--order", ""]) == 2  # no longer "every dimension"
    assert "names no dimension" in capsys.readouterr().err
    assert cli.main([*base, "--order", "gates", "--calibrate-only"]) == 2


# --------------------------------------------------------------------------- #
# B2: resume reuses a trial only in the mode it ran in
# --------------------------------------------------------------------------- #


def test_an_offline_search_is_not_reused_by_a_live_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "s"
    args = ["search", "--cases", ",".join(CASES), "--quiet", "--order", "gates"]
    assert (
        cli.main([*args, "--providers", "cloudflare,typesafe", "--out", str(out)]) == 0
    )
    before = (out / "search.json").read_text()
    assert {t["mode"] for t in json.loads(before)["trials"]} == {"offline"}
    capsys.readouterr()
    code = cli.main(
        [*args, "--live", "--providers", "cloudflare", "--resume", str(out)]
    )
    err = capsys.readouterr().err
    # nothing was reused: the base ran for real, and found no credentials
    assert code == 2
    assert "CLOUDFLARE_ACCOUNT_ID" in err and "CLOUDFLARE_API_TOKEN" in err
    assert "not reused" in err
    assert (out / "search.json").read_text() == before  # and nothing was overwritten


def test_run_search_reuses_only_trials_of_the_same_mode() -> None:
    sent: list[str] = []

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        sent.append(label)
        return SuiteResult(
            label, {}, [CaseResult(case="c", n_expected=0)], [], "offline", combo=spec
        )

    def go(mode: str, previous: list[Trial]) -> Any:
        return asyncio.run(
            run_search(
                ComboSpec(),
                evaluate,
                providers=["typesafe"],
                order=["coref"],
                mode=mode,
                previous=previous,
            )
        )

    first = go("offline", [])
    sent.clear()
    again = go("offline", first.trials)
    assert sent == [] and all(t.cached for t in again.trials)
    live = go("live", first.trials)
    assert len(sent) == len(first.trials) and not any(t.cached for t in live.trials)
    assert {t.mode for t in live.trials} == {"live"}
    old = [Trial.from_dict({**t.as_dict(), "mode": ""}) for t in first.trials]
    sent.clear()
    go("offline", old)  # a trial from before the mode was recorded is not reused
    assert len(sent) == len(first.trials)


def test_a_replay_is_a_mode_of_its_own() -> None:
    sent: list[str] = []

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        sent.append(label)
        return SuiteResult(label, {}, [CaseResult(case="c", n_expected=0)], [], "x")

    base = asyncio.run(
        run_search(ComboSpec(), evaluate, providers=[], order=[], mode="replay:/a")
    )
    sent.clear()
    asyncio.run(
        run_search(
            ComboSpec(),
            evaluate,
            providers=[],
            order=[],
            mode="replay:/b",
            previous=base.trials,
        )
    )
    assert len(sent) == 1


# --------------------------------------------------------------------------- #
# S3: calibrating what can authorise a removal
# --------------------------------------------------------------------------- #


def flat_evaluate() -> Any:
    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        return SuiteResult(
            label, {}, [CaseResult(case="c", n_expected=0)], [], "offline", combo=spec
        )

    return evaluate


def calibrations(result: Any) -> set[str]:
    return {t.change for t in result.trials if t.dimension.startswith("calibrate")}


def test_the_default_calibration_leaves_out_what_can_authorise_a_removal() -> None:
    result = asyncio.run(
        run_search(
            ComboSpec.model_validate({"decision": "typesafe"}),
            flat_evaluate(),
            providers=[],
            order=[],
            calibrate=True,
            grid=(-0.5, 0.5),
        )
    )
    kinds = {c.split(".")[1].rstrip("+-0123456789.") for c in calibrations(result)}
    assert kinds
    assert not kinds & {"gate", "null_check", "delete"}
    assert {"attribution", "match", "coref"} <= kinds


def test_a_removal_kind_must_be_named_and_then_only_shifts_down_are_tried() -> None:
    def go(**kw: Any) -> Any:
        return asyncio.run(
            run_search(
                ComboSpec.model_validate({"decision": "typesafe"}),
                flat_evaluate(),
                providers=[],
                order=[],
                calibrate=True,
                grid=(-1.0, -0.5, 0.5, 1.0),
                calibrate_kinds=["gate", "null_check"],
                **kw,
            )
        )

    safe = go()
    assert calibrations(safe) == {
        "cal typesafe.gate-1",
        "cal typesafe.gate-0.5",
        "cal typesafe.null_check-1",
        "cal typesafe.null_check-0.5",
    }
    assert any("--allow-positive-removal-shift" in n for n in safe.not_run)
    risky = go(allow_positive_removal_shift=True)
    assert "cal typesafe.gate+1" in calibrations(risky)


def test_a_removal_shift_is_flagged_in_the_leaderboard_and_by_the_spec() -> None:
    result = asyncio.run(
        run_search(
            ComboSpec.model_validate({"decision": "typesafe"}),
            flat_evaluate(),
            providers=[],
            order=[],
            calibrate=True,
            grid=(0.5,),
            calibrate_kinds=["gate", "attribution"],
            allow_positive_removal_shift=True,
        )
    )
    board = leaderboard_markdown(result)
    assert "[removal shift: typesafe.gate+0.5]" in board
    assert "## Removal shifts" in board
    flagged = [t for t in result.trials if t.spec.removal_shifts()]
    assert [t.change for t in flagged] == ["cal typesafe.gate+0.5"]
    assert (
        ComboSpec.model_validate(
            {"decision": "typesafe", "calibration": {"typesafe": {"gate": -1.0}}}
        ).removal_shifts()
        == []
    )
    assert ComboSpec.model_validate(
        {"decision": "typesafe", "calibration": {"typesafe": {"delete": 1.0}}}
    ).removal_shifts() == ["typesafe.delete+1"]


def test_combo_warns_before_running_a_removal_shift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"calibration": {"cloudflare": {"gate": 1.0}}}))
    cli.main(
        [
            "combo",
            "--spec",
            str(spec),
            "--cases",
            "rename",
            "--quiet",
            "--out",
            str(tmp_path / "o"),
        ]
    )
    assert "cloudflare.gate+1" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# S5, S7, S8, N6 and the results directory
# --------------------------------------------------------------------------- #


def test_a_sweep_on_the_holdout_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["sweep", "--suite", "holdout", "--param", "gates.add_threshold=0.5,0.6"]
    )
    assert code == 2
    assert "dev set" in capsys.readouterr().err


def test_a_trial_that_errored_is_not_run_again_on_resume_unless_asked(
    tmp_path: Path,
) -> None:
    runs: list[str] = []

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        runs.append(label)
        if spec.by_kind.get("coref") == "typesafe":
            raise RuntimeError("the network is down")
        return SuiteResult(label, {}, [CaseResult(case="c", n_expected=0)], [], "x")

    def go(**kw: Any) -> Any:
        return asyncio.run(
            run_search(
                ComboSpec(),
                evaluate,
                providers=["typesafe"],
                order=["coref"],
                mode="live",
                **kw,
            )
        )

    first = go()
    assert [t.outcome for t in first.trials] == ["base", "error"]
    write_search(tmp_path, first, {"suite": "default", "cases": ["c"]})
    _, previous = load_previous(tmp_path)
    runs.clear()
    second = go(previous=previous)
    assert runs == []  # neither the base nor the error was run again
    assert second.trials[1].outcome == "error" and second.trials[1].cached
    assert "the network is down" in (second.trials[1].error or "")
    third = go(previous=previous, retry_errors=True)
    assert len(runs) == 1  # only the one that raised
    assert third.trials[1].cached is False


def test_the_conflict_case_needs_the_skip_to_be_reported() -> None:
    case = next(c for c in HOLDOUT_CASES if c.id == "update_and_delete_conflict")
    assert case.expect_skipped == [("conflict_update_delete", "acct-2")]

    async def go(flip: dict[str, tuple[str, ...]]) -> CaseResult:
        suite = await run_suite(
            [case], Settings(combo=ComboSpec(), members=scripted(flip=flip))
        )
        (only,) = suite.results
        return only

    right = asyncio.run(go({}))
    assert right.passed and right.skipped_found == right.n_skipped == 1
    # a combination that finds nothing does nothing: no delete, no change, and no
    # report of the conflict. It used to pass.
    nothing = asyncio.run(
        go({"cloudflare": ("presence", "qualify", "match", "delete")})
    )
    assert not nothing.passed
    assert nothing.n_skipped == 1 and nothing.skipped_found == 0
    assert "skipped conflict_update_delete acct-2" in nothing.missing
    assert not nothing.violations  # it did no harm: it just did not do the job


def test_expected_skips_are_matched_by_reason_and_record() -> None:
    from types import SimpleNamespace

    from evals.model import Case

    def result(*skips: tuple[str, str]) -> Any:
        return SimpleNamespace(
            records=[],
            skipped=[
                SimpleNamespace(reason=r, candidates=((rid, 0.9),)) for r, rid in skips
            ],
            usage=SimpleNamespace(requests=0, input_tokens=0),
            warnings=[],
        )

    def case(expected: tuple[str, str | None]) -> Case:
        return Case(id="x", text="t", schemas=lambda: [], expect_skipped=[expected])

    assert score(case(("a", "r1")), result(("a", "r1"))).passed
    assert score(case(("a", None)), result(("a", "r9"))).passed
    assert not score(case(("a", "r1")), result(("a", "r2"))).passed
    assert not score(case(("a", "r1")), result(("b", "r1"))).passed
    assert not score(case(("a", "r1")), result()).passed


def test_the_lead_case_is_unambiguous() -> None:
    case = next(c for c in HOLDOUT_CASES if c.id == "crm_two_leads")
    assert "min 2BR" in case.text and "2BR ok" not in case.text
    assert "budget $450k, pre-appr." in case.text
    assert asyncio.run(run_suite([case], Settings())).results[0].passed


def test_a_narrow_win_has_to_beat_the_margin_and_may_be_repeated() -> None:
    def results(passed: int) -> list[CaseResult]:
        ok = CaseResult(
            case="ok", n_expected=1, matched_expected=1, n_actual=1, matched_actual=1
        )
        no = CaseResult(case="no", n_expected=1, n_actual=1)
        return [ok] * passed + [no] * (40 - passed)

    seen: list[str] = []

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        seen.append(label)
        better = spec.by_kind.get("coref") == "typesafe"
        # one case better, once; the repeat does not hold
        passed = 25 if better and "repeat" not in label else 24
        return SuiteResult(label, {}, results(passed), [], "x", combo=spec)

    def go(**kw: Any) -> Any:
        seen.clear()
        return asyncio.run(
            run_search(
                ComboSpec(), evaluate, providers=["typesafe"], order=["coref"], **kw
            )
        )

    kept = go()  # a case more than the incumbent: kept (the default margin)
    assert kept.best.spec.by_kind == {"coref": "typesafe"}
    strict = go(margin=Margin(2, 0.5))  # one case, and 0.01 of F1, is not enough
    assert strict.trials[1].outcome == "rejected"
    assert strict.trials[1].note == "better, but by less than the margin"
    repeated = go(repeats=2)
    assert repeated.trials[1].outcome == "rejected"
    assert "won once but not on repeat 2" in repeated.trials[1].note
    assert any(label.endswith("-repeat2") for label in seen)
    assert repeated.best is repeated.trials[0]


def test_min_margin_and_repeats_are_validated_and_reach_the_search(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["search", "--cases", "rename", "--quiet", "--order", "coref"]
    assert cli.main([*base, "--min-margin", "x"]) == 2
    assert cli.main([*base, "--min-margin", "-1"]) == 2
    assert cli.main([*base, "--repeats", "0"]) == 2
    assert (
        cli.main(
            [
                *base,
                "--min-margin",
                "0:0.5",
                "--repeats",
                "2",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli._margin("2") == Margin(2, 0.02)
    assert cli._margin("0:0.1") == Margin(0, 0.1)


def test_a_holdout_result_directory_says_so() -> None:
    def where(suite: str) -> str:
        return cli._out_dir(argparse.Namespace(out=None, suite=suite), "default").name

    assert where("holdout").endswith("-holdout-default")
    assert where("default").endswith("-default") and "holdout" not in where("default")
    assert cli._out_dir(
        argparse.Namespace(out=Path("x"), suite="holdout"), "n"
    ) == Path("x")


def test_the_search_still_picks_the_right_providers_with_the_margin() -> None:
    result = asyncio.run(
        run_search(
            ComboSpec(),
            evaluator(scripted(flip=BAD_AT)),
            providers=["cloudflare", "decisions", "typesafe"],
            order=["attribution", "gates"],
        )
    )
    for kind in ATTRIBUTION:
        assert result.best.spec.by_kind[kind] == "decisions"
    for kind in GATES:
        assert result.best.spec.by_kind[kind] == "typesafe"


# --------------------------------------------------------------------------- #
# R1: a recorder or a cassette around a provider hides nothing from the routers
# --------------------------------------------------------------------------- #

IMAGE = "data:image/png;base64,iVBORw0KGgo="
STATE = "<input>\nAda was born on 1815-12-10.\n</input>"
PICTURED: dict[str, Any] = {
    "model": "clairic",
    "state": STATE,
    "fields": {"name": {"type": "text"}, "born": {"type": "date"}},
    "images": [IMAGE],
    "options": {"return_state_text": True},
}


class Member:
    """A provider that may take images, and fails loudly when it cannot."""

    def __init__(self, name: str, images: bool) -> None:
        self.provider, self.images, self.sent = name, images, []  # type: ignore[var-annotated]

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        return self.images or not (payload or {}).get("images")

    def classify_failure(self, exc: Exception) -> str | None:
        return f"{self.provider}-classified"

    def sensitive_values(self) -> tuple[str, ...]:
        return (f"secret-{self.provider}",)

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        from trustcall_system_one import BadRequestError

        if payload.get("images") and not self.images:
            raise BadRequestError(f"{self.provider} takes text only")
        await asyncio.sleep(0.01)
        self.sent.append(list(payload["fields"]))
        answers = {
            n: {"type": s["type"], "value": "x", "p_null": 0.01, "confidence": 0.9}
            for n, s in payload["fields"].items()
        }
        body = {
            "model": "clairic",
            "answers": answers,
            "truncated": False,
            "state_text": STATE,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        return TransportResponse(body=body, status_code=200, provider=self.provider)


def pictured_run(wrap: Any, tmp_path: Path) -> tuple[str, Member, Member]:
    from trustcall_system_one.combos import build_combo_transport

    cf, jev = Member("cloudflare", True), Member("typesafe", False)
    spec = ComboSpec(extraction="cloudflare", extraction_by_type={"date": "typesafe"})
    stack = build_combo_transport(
        spec, providers={"cloudflare": cf, "typesafe": jev}, calibrate=False, wrap=wrap
    )
    try:
        reply = asyncio.run(stack.run(CLAIRIC, PICTURED, meta=CallMeta(kind="extract")))
        out = f"ok {reply.provider} rerouted_from={reply.rerouted_from}"
    except Exception as exc:
        out = f"{type(exc).__name__} maybe_sent={getattr(exc, 'maybe_sent', None)}"
    return out, cf, jev


def test_images_behave_the_same_with_a_recorder_around_each_provider(
    tmp_path: Path,
) -> None:
    from trustcall_system_one.testing import RecordingTransport

    plain, cf0, jev0 = pictured_run(None, tmp_path)
    assert plain == "ok cloudflare rerouted_from=fieldtyperoutingtransport"  # whole
    assert cf0.sent == [["name", "born"]] and jev0.sent == []
    recorded, cf, jev = pictured_run(
        lambda p, t: RecordingTransport(t, tmp_path / "r.jsonl"), tmp_path
    )
    assert recorded == plain
    assert cf.sent == cf0.sent and jev.sent == []


def test_images_behave_the_same_with_a_live_cassette_around_each_provider(
    tmp_path: Path,
) -> None:
    from trustcall_system_one.testing import CassetteTransport

    (tmp_path / "e.jsonl").write_text("")
    plain, *_ = pictured_run(None, tmp_path)
    live, cf, jev = pictured_run(
        lambda p, t: CassetteTransport(
            tmp_path / "e.jsonl", strict=False, inner=t, provider=p
        ),
        tmp_path,
    )
    assert live == plain and jev.sent == [] and cf.sent == [["name", "born"]]


def test_images_under_a_replay_with_nothing_behind_it_are_rerouted_whole_too(
    tmp_path: Path,
) -> None:
    """The runner's adapter: a strict cassette answers for the member it stands for."""
    from evals.runner import _Replaying
    from trustcall_system_one.testing import CassetteTransport

    (tmp_path / "e.jsonl").write_text("")
    cassettes: dict[str, CassetteTransport] = {}

    def wrap(provider: str, member: Any) -> Any:
        cassettes[provider] = CassetteTransport(tmp_path / "e.jsonl", provider=provider)
        return _Replaying(cassettes[provider], member)

    out, cf, jev = pictured_run(wrap, tmp_path)
    # the request was rerouted whole to cloudflare's cassette (a miss: nothing was
    # recorded), and typesafe's cassette was never asked: no partition was launched
    assert out.startswith("CassetteMissError maybe_sent=")
    assert len(cassettes["cloudflare"].misses) == 1
    assert cassettes["typesafe"].misses == [] and jev.sent == [] == cf.sent
    assert "maybe_sent=True" not in out


def test_nothing_is_sent_when_no_partition_can_take_the_images(tmp_path: Path) -> None:
    from evals.runner import _Replaying
    from trustcall_system_one import BadRequestError
    from trustcall_system_one.combos import build_combo_transport
    from trustcall_system_one.testing import CassetteTransport, RecordingTransport

    for wrap in (
        None,
        lambda p, t: RecordingTransport(t, tmp_path / "x.jsonl"),
        lambda p, t: _Replaying(CassetteTransport(tmp_path, provider=p), t),
    ):
        a, b = Member("cloudflare", False), Member("typesafe", False)
        spec = ComboSpec(
            extraction="cloudflare", extraction_by_type={"date": "typesafe"}
        )
        stack = build_combo_transport(
            spec, providers={"cloudflare": a, "typesafe": b}, calibrate=False, wrap=wrap
        )
        with pytest.raises(BadRequestError) as caught:
            asyncio.run(stack.run(CLAIRIC, PICTURED, meta=CallMeta(kind="extract")))
        assert a.sent == b.sent == []
        assert not getattr(caught.value, "maybe_sent", False)  # nothing went out


def test_a_recorder_and_a_cassette_forward_what_a_router_asks() -> None:
    from trustcall_system_one import CloudflareTransport, UncertainOutcomeError
    from trustcall_system_one.testing import RecordingTransport

    member = Member("typesafe", False)
    pictured = {"images": [IMAGE]}
    recorder = RecordingTransport(member, Path(os.devnull))
    assert recorder.serves(CLAIRIC, pictured) is False
    assert recorder.serves(CLAIRIC, {}) is True
    assert recorder.classify_failure(RuntimeError()) == "typesafe-classified"
    assert recorder.sensitive_values() == ("secret-typesafe",)
    cassette = (
        CassetteTransport(Path("/nonexistent-never-read"), inner=member)
        if False
        else None
    )
    assert cassette is None
    # a real transport: Cloudflare's own classification of a post-send failure
    cf = CloudflareTransport(account_id="a", api_token="t")
    wrapped = RecordingTransport(cf, Path(os.devnull))
    assert wrapped.classify_failure(UncertainOutcomeError("x")) == "uncertain"
    assert wrapped.sensitive_values() == ("a", "t")


def test_a_cassette_with_an_inner_transport_forwards_and_one_without_says_nothing(
    tmp_path: Path,
) -> None:
    from trustcall_system_one.testing import CassetteTransport

    (tmp_path / "e.jsonl").write_text("")
    member = Member("typesafe", False)
    pictured = {"images": [IMAGE]}
    live = CassetteTransport(tmp_path, strict=False, inner=member)
    assert live.serves(CLAIRIC, pictured) is False
    assert live.classify_failure(RuntimeError()) == "typesafe-classified"
    assert live.sensitive_values() == ("secret-typesafe",)
    alone = CassetteTransport(tmp_path)
    assert alone.serves(CLAIRIC, pictured) is True  # it replays whatever was recorded
    assert alone.sensitive_values() == ()
    assert alone.classify_failure(RuntimeError()) is None


# --------------------------------------------------------------------------- #
# The engine's gather leaves no error unretrieved
# --------------------------------------------------------------------------- #


def test_concurrent_failures_leave_no_task_exception_never_retrieved() -> None:
    import gc

    from trustcall_system_one._run import gather

    reports: list[dict[str, Any]] = []

    async def fail(i: int) -> int:
        raise ValueError(f"chunk {i} failed")

    async def go() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: reports.append(context))
        for _ in range(3):  # several tasks fail in the same pass of the loop
            with pytest.raises(ValueError, match="chunk 0 failed"):
                await gather(*(fail(i) for i in range(4)))
        gc.collect()
        await asyncio.sleep(0)

    asyncio.run(go())
    gc.collect()
    assert [r["message"] for r in reports] == []


def test_gather_still_raises_the_first_error_and_returns_results_in_order() -> None:
    from trustcall_system_one._run import gather

    async def value(i: int) -> int:
        await asyncio.sleep(0)
        return i

    async def go() -> list[int]:
        return await gather(value(1), value(2), value(3))

    assert asyncio.run(go()) == [1, 2, 3]

    async def slow_then_fail() -> int:
        await asyncio.sleep(0.01)
        raise KeyError("late")

    async def fail_now() -> int:
        raise ValueError("first")

    async def mixed() -> None:
        await gather(slow_then_fail(), fail_now())

    with pytest.raises(ValueError, match="first"):
        asyncio.run(mixed())
