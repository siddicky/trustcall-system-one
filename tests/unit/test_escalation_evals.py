"""The escalation tier in the evals: ``--escalate``, combination specs and the search.

Offline, the escalation model is a scripted ``FakeEscalation`` (``Settings.escalation``)
or, by default, a stand-in that never answers, so a run with the tier on is the same
run. Nothing is sent anywhere.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from evals.cli import main as evals_main
from evals.model import Case, Op, Val
from evals.report import suite_markdown
from evals.runner import (
    Settings,
    aggregate,
    escalation_plan,
    escalation_usage,
    missing_credentials,
    run_suite,
    usage_by_provider,
)
from evals.search import (
    EFFORT_CANDIDATES,
    EFFORT_DIMENSION,
    estimate_cost,
    leaderboard_markdown,
    run_search,
)
from trustcall_system_one import One
from trustcall_system_one.combos import ComboSpec, EscalateSpec
from trustcall_system_one.testing import (
    FakeEscalation,
    FakeTransport,
    FakeWorld,
    clairic_answer,
)

TODAY = "2026-10-23"
TEXT = "Task: Replace the pump. Completed three days ago by the night crew."


class Task(BaseModel):
    """A task report."""

    title: str = Field(description="the task title")
    done_on: dt.date = Field(description="the day the task was completed")


class _World:
    """The scripted truth of the case, with Clairic failing on the relative date."""

    def transport(self) -> FakeTransport:
        world = FakeWorld(TEXT)
        world.entity(
            "the Task",
            schema=Task,
            mentions=[],
            fields={"title": "Replace the pump", "done_on": dt.date(2026, 10, 20)},
        )
        inner: Any = world.handler

        def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
            out = inner(model, payload, meta)
            if meta.kind in ("extract", "crosscheck", "reask", "repair"):
                for wire, answer in out["answers"].items():
                    if answer["type"] == "date":
                        out["answers"][wire] = clairic_answer(
                            "date", None, p_null=0.99, confidence=0.9
                        )
            return out

        return FakeTransport(handler)


CASE = Case(
    id="relative_done",
    text=TEXT,
    schemas=lambda: [One(Task)],
    options={"enable_inserts": True, "reference_date": TODAY},
    expect_ops=[Op("insert")],
    expect_values=[
        Val("Task", "/title", "Replace the pump"),
        Val("Task", "/done_on", dt.date(2026, 10, 20)),
    ],
    world=_World,
)
ANSWER = {"phrase": "three days ago", "iso_date": "2026-10-20", "time": None}
NO_ANSWER = {"phrase": None, "iso_date": None, "time": None}


def luna(
    needs: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max"),
    **kw: Any,
) -> Any:
    """An escalation model that answers only at the efforts in ``needs``."""

    def build(case: Case) -> FakeEscalation:
        return FakeEscalation(
            lambda payload, meta: (
                ANSWER
                if payload.get("reasoning_effort", "none") in needs
                else NO_ANSWER
            ),
            **kw,
        )

    return build


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "TRUSTCALL_S1_ESCALATE",
        "TRUSTCALL_S1_ESCALATE_EFFORT",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# The CLI flag and the environment opt-in
# --------------------------------------------------------------------------- #


def results(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "results.json").read_text())
    return data


def test_escalate_flag_runs_offline_and_records_the_tier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = evals_main(
        [
            "run",
            "--cases",
            "readme_people,rename",
            "--escalate",
            "openai",
            "--escalate-effort",
            "low",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert status == 0
    data = results(tmp_path)
    assert data["escalation"] == "openai gpt-6-luna effort=low"
    assert data["escalate_effort"] == "low"
    assert data["aggregate"]["escalation_calls"] == 0  # the stand-in never answers
    assert data["aggregate"]["passed"] == data["aggregate"]["cases"]
    summary = (tmp_path / "summary.md").read_text()
    assert "Escalation tier: openai gpt-6-luna effort=low" in summary
    assert "goes to OpenAI" in summary
    assert "nothing is sent to OpenAI" in capsys.readouterr().err


def test_without_the_flag_nothing_is_recorded_about_a_tier(tmp_path: Path) -> None:
    assert (
        evals_main(["run", "--cases", "rename", "--out", str(tmp_path), "--quiet"]) == 0
    )
    data = results(tmp_path)
    assert data["escalation"] == "" and data["escalate_effort"] == ""
    assert "Escalation tier" not in (tmp_path / "summary.md").read_text()


def test_an_effort_without_the_tier_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = evals_main(["run", "--cases", "rename", "--escalate-effort", "high"])
    assert status == 2
    assert "needs --escalate openai" in capsys.readouterr().err


def test_an_unknown_effort_or_provider_is_refused_by_the_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (
        ["run", "--escalate", "gemini"],
        ["run", "--escalate", "openai", "--escalate-effort", "extreme"],
    ):
        with pytest.raises(SystemExit):
            evals_main(argv)
    assert "invalid choice" in capsys.readouterr().err


def test_the_environment_opt_in_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 30)  # a key alone: nothing
    assert escalation_plan(Settings(), None) is None
    assert (
        evals_main(["run", "--cases", "rename", "--out", str(tmp_path), "--quiet"]) == 0
    )
    assert results(tmp_path)["escalation"] == ""
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "openai")
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE_EFFORT", "medium")
    plan = escalation_plan(Settings(), None)
    assert plan is not None and plan.reasoning_effort == "medium"
    assert escalation_plan(Settings(escalate_effort="high"), None) == EscalateSpec(
        reasoning_effort="high"
    )


async def test_live_escalation_needs_the_key_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t" * 40)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await run_suite([CASE], Settings(live=True, escalate="openai"))
    spec = ComboSpec(escalate="openai")
    assert "OPENAI_API_KEY" in missing_credentials(spec)
    assert "OPENAI_API_KEY" not in missing_credentials(ComboSpec())


# --------------------------------------------------------------------------- #
# A case the default setup fails, fixed by the tier
# --------------------------------------------------------------------------- #


async def test_a_suite_fails_the_case_without_the_tier_and_passes_with_it() -> None:
    off = await run_suite([CASE], Settings())
    assert not off.results[0].passed and aggregate(off.results)["escalation_calls"] == 0
    on = await run_suite(
        [CASE], Settings(escalate="openai", escalation=luna(reasoning_tokens=30))
    )
    (result,) = on.results
    assert result.passed
    totals = aggregate(on.results)
    assert totals["escalation_calls"] == 1 and totals["escalation_failed"] == 0
    assert totals["escalation_reasoning_tokens"] == 30
    assert (
        totals["escalation_output_tokens"] == 35
    )  # reasoning tokens are output tokens
    assert totals["escalation_mean_latency_ms"] > 0
    assert result.escalation["by_kind"] == {"relative_dates": 1}
    # the call log files it under its own provider, with its latency
    row = result.by_provider["openai"]
    assert row["requests"] == 1 and row["latency_ms"] > 0 and row["output_tokens"] == 35
    assert (
        on.escalation == "openai gpt-6-luna effort=none"
        and on.escalate_effort == "none"
    )
    assert "Escalation calls: 1" in suite_markdown(
        on
    ) and "reasoning 30" in suite_markdown(on)


async def test_the_effort_reaches_the_request_and_the_results() -> None:
    seen: list[str] = []

    def build(case: Case) -> FakeEscalation:
        def handler(payload: dict[str, Any], meta: Any) -> dict[str, Any]:
            seen.append(payload["reasoning_effort"])
            return ANSWER

        return FakeEscalation(handler)

    suite = await run_suite(
        [CASE], Settings(escalate="openai", escalate_effort="high", escalation=build)
    )
    assert seen == ["high"] and suite.escalate_effort == "high"


def test_usage_helpers_count_latency_and_escalation_calls() -> None:
    from trustcall_system_one import CallRecord

    calls = [
        CallRecord(
            model="m",
            kind="attribution",
            schema="S",
            entity=None,
            field=None,
            scope="",
            n_items=1,
            usage={"input_tokens": 10, "output_tokens": 1},
            duration_ms=7.0,
            provider="cloudflare",
        ),
        CallRecord(
            model="gpt-6-luna",
            kind="escalate",
            schema="S",
            entity=None,
            field="list_rows",
            scope="",
            n_items=1,
            usage={"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 40},
            duration_ms=900.0,
            provider="openai",
        ),
        CallRecord(
            model="gpt-6-luna",
            kind="escalate",
            schema="S",
            entity=None,
            field="relative_dates:done_on",
            scope="",
            n_items=1,
            error="UncertainOutcomeError: timed out",
            duration_ms=60000.0,
            provider="openai",
        ),
    ]
    rows = usage_by_provider(calls)
    assert rows["cloudflare"]["latency_ms"] == 7.0
    assert rows["openai"]["latency_ms"] == 60900.0 and rows["openai"]["requests"] == 2
    used = escalation_usage(calls)
    assert used["calls"] == 2 and used["failed"] == 1
    assert used["reasoning_tokens"] == 40 and used["output_tokens"] == 50
    assert used["by_kind"] == {"list_rows": 1, "relative_dates": 1}


# --------------------------------------------------------------------------- #
# ComboSpec.escalate
# --------------------------------------------------------------------------- #


def test_a_spec_names_the_tier_as_a_string_or_an_object() -> None:
    short = ComboSpec.model_validate({"escalate": "openai"})
    assert short.escalate == EscalateSpec()
    full = ComboSpec.model_validate(
        {
            "escalate": {
                "provider": "openai",
                "model": "gpt-6-luna",
                "reasoning_effort": "low",
                "kinds": ["relative_dates", "list_rows"],
                "limits": {"list_rows": 2},
                "max_calls_per_invoke": 5,
                "prob_cap": 0.7,
            }
        }
    )
    config = full.escalate.config()  # type: ignore[union-attr]
    assert config.enabled and config.reasoning_effort == "low"
    assert config.kinds == {"relative_dates", "list_rows"}
    assert config.limit_for("list_rows") == 2 and config.prob_cap == 0.7
    assert config.max_calls_per_invoke == 5
    assert "escalate openai gpt-6-luna effort=low" in full.describe()
    assert full.key() != short.key() != ComboSpec().key()  # the tier is part of the key
    assert "OpenAI" in full.companies()
    assert "OpenAI" not in ComboSpec().companies()


def test_a_spec_refuses_what_the_tier_cannot_do() -> None:
    for bad in (
        {"escalate": {"reasoning_effort": "extreme"}},
        {"escalate": {"kinds": ["magic"]}},
        {"escalate": {"prob_cap": 0.9}},
        {"escalate": {"provider": "gemini"}},
        {"escalate": {"mystery": 1}},
        {"only": "typesafe", "escalate": "openai"},
        {"by_kind": {"escalate": "cloudflare"}},
        {"decision": "typesafe", "calibration": {"typesafe": {"escalate": 1.0}}},
    ):
        with pytest.raises(ValueError):
            ComboSpec.model_validate(bad)


def test_the_spec_effort_changes_its_canonical_key() -> None:
    low = ComboSpec(escalate=EscalateSpec(reasoning_effort="low"))
    high = low.replace(escalate=EscalateSpec(reasoning_effort="high"))
    assert low.key() != high.key()


async def test_a_combo_run_records_the_tier_in_the_results(tmp_path: Path) -> None:
    spec = ComboSpec(name="tiered", escalate=EscalateSpec(reasoning_effort="medium"))
    suite = await run_suite([CASE], Settings(combo=spec, escalation=luna()))
    assert suite.results[0].passed
    assert suite.escalation == "openai gpt-6-luna effort=medium"
    from evals.report import combo_record

    record = combo_record(suite)
    assert record is not None
    assert record["labels"]["escalation"] == "openai gpt-6-luna effort=medium"
    assert "OpenAI" in record["companies"]
    assert record["spec"]["escalate"]["reasoning_effort"] == "medium"


# --------------------------------------------------------------------------- #
# Sweeping the effort: same margin and veto, dev suite only
# --------------------------------------------------------------------------- #


async def _search(base: ComboSpec, build: Any, **kwargs: Any) -> Any:
    async def evaluate(spec: ComboSpec, label: str) -> Any:
        return await run_suite(
            [CASE], Settings(combo=spec, escalation=build), config=label
        )

    return await run_search(
        base, evaluate, providers=[], order=[EFFORT_DIMENSION], **kwargs
    )


async def test_the_search_sweeps_the_effort_and_keeps_the_cheapest_that_wins() -> None:
    # the model only answers from "medium" up: low is no gain, medium wins, and high
    # (same pass count, dearer) does not beat the incumbent by the margin
    build = luna(
        needs=("medium", "high", "xhigh", "max"),
        reasoning_tokens={"medium": 200, "high": 900},
    )
    base = ComboSpec(escalate=EscalateSpec())
    result = await _search(base, build, prices={"openai": (1.0, 4.0)})
    changes = {t.change: t.outcome for t in result.trials}
    assert changes["base"] == "base"
    assert changes["escalate.effort=low"] == "rejected"
    assert changes["escalate.effort=medium"] == "kept"
    assert changes["escalate.effort=high"] == "rejected"
    assert "escalate.effort=none" not in changes  # that is the base itself
    assert result.best.spec.escalate is not None
    assert result.best.spec.escalate.reasoning_effort == "medium"
    assert result.best.aggregate["passed"] == 1
    # reasoning tokens are output tokens: more effort costs more
    by_effort = {t.spec.escalate.reasoning_effort: t for t in result.trials}
    assert by_effort["high"].cost_usd > by_effort["medium"].cost_usd > 0
    assert by_effort["medium"].aggregate["escalation_reasoning_tokens"] == 200
    board = leaderboard_markdown(result)
    assert (
        "esc calls" in board and "esc reasoning tok" in board and "esc ms/call" in board
    )
    assert "escalate.effort=medium" in board


async def test_the_effort_sweep_keeps_the_veto_and_the_margin() -> None:
    # an escalation that answers at every effort leaves nothing to gain: nothing moves
    result = await _search(ComboSpec(escalate=EscalateSpec()), luna())
    assert result.best.n == 0
    assert all(t.outcome in ("base", "rejected") for t in result.trials)
    assert [t.spec.escalate.reasoning_effort for t in result.trials[1:]] == [
        e for e in EFFORT_CANDIDATES if e != "none"
    ]


async def test_the_effort_sweep_needs_a_tier_in_the_base() -> None:
    result = await _search(ComboSpec(), luna())
    assert len(result.trials) == 1
    assert any("has no escalate to sweep" in line for line in result.not_run)


def test_the_search_cost_counts_reasoning_through_output_tokens() -> None:
    from evals.model import CaseResult

    row = {"requests": 1, "input_tokens": 1000, "output_tokens": 500}
    usd, partial = estimate_cost(
        [CaseResult(case="x", by_provider={"openai": dict(row)})],
        {"openai": (2.0, 8.0)},
    )
    assert usd == pytest.approx(1000 * 2 / 1e6 + 500 * 8 / 1e6) and not partial
    _, unpriced = estimate_cost(
        [CaseResult(case="x", by_provider={"openai": dict(row)})], {}
    )
    assert unpriced  # a tier with no price is a partial estimate, never a made-up one


def test_the_search_is_refused_on_the_holdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = evals_main(["search", "--suite", "holdout", "--order", EFFORT_DIMENSION])
    assert status == 2 and "dev suite only" in capsys.readouterr().err
