"""Review fixes of the Experiential transport: attempt budget, cancelled calls, cost.

The first two come from probes (a 429 storm that starved the last send of time,
and a sibling request that was cancelled while on the wire and left no record).
"""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.cases import CASES
from evals.cli import _parser, _settings
from evals.model import CaseResult
from evals.report import suite_markdown
from evals.runner import (
    Settings,
    _have_credentials,
    _run_case,
    aggregate,
    count_unknown_outcomes,
    run_suite,
)
from tests.unit.test_experiential import (
    BODY,
    KEY,
    META,
    PAYLOAD,
    Tag,
    call,
    make,
    ok,
)
from trustcall_system_one import (
    CLEF,
    BadRequestError,
    CallMeta,
    CallRecord,
    ExperientialTransport,
    ModelResponseError,
    Prices,
    RoutingTransport,
    TransportError,
    TransportResponse,
    UncertainOutcomeError,
    Usage,
    create_extractor,
)
from trustcall_system_one._run import CANCELLED_AFTER_SEND, Run
from trustcall_system_one._wire import ClefRequest, ClefUsage, parse_clef
from trustcall_system_one.testing import scrub

# --------------------------------------------------------------------------- #
# 1. A re-send is not made with less time than an attempt needs
# --------------------------------------------------------------------------- #


class Clocked:
    """A fake clock, a sleep that advances it and a handler that logs each send."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.now = 0.0
        self.responses = list(responses)
        self.sent: list[tuple[float, float]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.sent.append((self.now, request.extensions["timeout"]["read"]))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    async def sleep(self, delay: float) -> None:
        self.now += delay

    def transport(self, **kwargs: Any) -> ExperientialTransport:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handler), timeout=60.0
        )
        return ExperientialTransport(
            api_key=KEY,
            client=client,
            clock=lambda: self.now,
            sleep=self.sleep,
            **kwargs,
        )


def too_busy(wait: int = 59) -> httpx.Response:
    return httpx.Response(
        429, json={"error": "slow"}, headers={"Retry-After": str(wait)}
    )


async def test_the_last_send_is_not_made_with_a_starved_read_timeout() -> None:
    # probe exp_starved_timeout: 429 waits of 59 s used the 300 s budget up and
    # the 6th send went out with a ~5 s read timeout
    clock = Clocked(too_busy())  # 429 with Retry-After, every time
    transport = clock.transport()
    with pytest.raises(TransportError) as info:
        await call(transport)
    error = info.value
    assert not isinstance(error, UncertainOutcomeError)  # nothing was sent
    assert "deadline" in str(error)
    assert "not sent" in str(error)
    assert error.retries == 4
    assert [at for at, _ in clock.sent] == [0.0, 59.0, 118.0, 177.0, 236.0]
    assert all(read == 60.0 for _, read in clock.sent)  # never a shortened one
    assert clock.now == 236.0  # and it did not wait for a send it would not make


async def test_with_the_check_off_the_old_behaviour_is_there() -> None:
    clock = Clocked(*[too_busy()] * 5, httpx.ReadTimeout("slow"))
    transport = clock.transport(min_attempt_budget=0)
    with pytest.raises(UncertainOutcomeError):
        await call(transport)
    reads = [read for _, read in clock.sent]
    assert min(reads) < 10  # the starved send the check exists to prevent


async def test_the_budget_can_be_set_and_a_per_call_timeout_is_the_default() -> None:
    clock = Clocked(too_busy(100), ok())
    transport = clock.transport(min_attempt_budget=150, max_retry_after=200)
    # 300 s: after a 100 s wait 200 s are left, enough for a 150 s budget
    result = await call(transport)
    assert result.retries == 1
    clock = Clocked(too_busy(100), ok())
    transport = clock.transport(min_attempt_budget=250, max_retry_after=200)
    with pytest.raises(TransportError, match="not sent"):
        await call(transport)
    assert len(clock.sent) == 1
    # without the knob, the call's own timeout is what an attempt needs
    clock = Clocked(too_busy(250), ok())
    transport = clock.transport(max_retry_after=300)
    with pytest.raises(TransportError, match="not sent"):
        await transport.run(CLEF, PAYLOAD, meta=META, timeout=100)
    clock = Clocked(too_busy(250), ok())
    transport = clock.transport(max_retry_after=300)
    result = await transport.run(CLEF, PAYLOAD, meta=META, timeout=40)
    assert result.retries == 1


async def test_the_first_attempt_always_goes_out_even_on_a_short_deadline() -> None:
    clock = Clocked(ok())
    transport = clock.transport(call_deadline=5.0)
    assert (await call(transport)).body == BODY
    assert len(clock.sent) == 1


async def test_the_send_check_catches_time_lost_while_waiting() -> None:
    # a sleep that overshoots: the pre-wait check passed, the pre-send one does not
    clock = Clocked(httpx.ConnectError("refused"), ok())

    async def slow_sleep(delay: float) -> None:
        clock.now += delay + 280

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(clock.handler), timeout=60.0
    )
    transport = ExperientialTransport(
        api_key=KEY, client=client, clock=lambda: clock.now, sleep=slow_sleep
    )
    with pytest.raises(TransportError, match="not sent"):
        await call(transport)
    assert len(clock.sent) == 1


def test_the_budget_must_be_valid() -> None:
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="min_attempt_budget"):
            ExperientialTransport(min_attempt_budget=bad)


# --------------------------------------------------------------------------- #
# 2. A call cancelled after it may have been sent is recorded
# --------------------------------------------------------------------------- #


def two_chunks() -> Mapping[str, Any]:
    """70 questions: more than the 64 one request takes, so two requests."""
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Is {i} even?"} for i in range(70)
    }
    items = list(questions.items())
    merged: dict[str, Any] = {}
    for part in (items[:64], items[64:]):
        merged.update(
            ClefRequest.model_validate(
                {"model": "clef", "state": "s", "questions": dict(part)}
            ).questions
        )
    return merged


async def test_a_sibling_cancelled_on_the_wire_is_recorded() -> None:
    # probe exp_sibling_cancel
    on_wire: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        first = next(iter(json.loads(request.content)["questions"]))
        on_wire.append(first)
        if first == "q0":
            await asyncio.sleep(0.05)
            raise httpx.ReadTimeout("slow", request=request)
        await asyncio.sleep(30)  # the sibling: sent, the server is working on it
        raise AssertionError("unreachable")

    transport = ExperientialTransport(
        api_key=KEY, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    run = Run(transport=transport)
    with pytest.raises(UncertainOutcomeError):
        await run.call_clef(two_chunks(), "s", meta=CallMeta(kind="attribution"))
    assert len(on_wire) == 2  # both requests were sent
    assert len(run.calls) == 2  # and both are on record
    errors = sorted(c.error or "" for c in run.calls)
    assert errors[0].startswith("UncertainOutcomeError")
    assert errors[1] == CANCELLED_AFTER_SEND
    assert run.usage.requests == 0  # nothing answered


class Hangs:
    """A transport (not an HTTP one) whose call never finishes."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.entered.set()
        await asyncio.sleep(30)
        raise AssertionError("unreachable")


async def test_cancelling_the_caller_records_the_call_for_any_transport() -> None:
    transport = Hangs()
    run = Run(transport=transport)
    task = asyncio.ensure_future(
        run.call_clef(
            {"q": ClefRequest.model_validate(PAYLOAD).questions["review"]},
            "s",
            meta=CallMeta(kind="gate", schema="Person"),
        )
    )
    await transport.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (record,) = run.calls
    assert record.error == "cancelled after send; outcome unknown"
    assert (record.kind, record.schema) == ("gate", "Person")
    assert record.duration_ms >= 0


# --------------------------------------------------------------------------- #
# 4. Reported cost
# --------------------------------------------------------------------------- #

MODEL = "@cf/cloudflare/clef"


def clef_usage(cost: Any = 0.01, tokens: int = 100) -> ClefUsage:
    return ClefUsage.model_validate(
        {"input_tokens": tokens, "output_tokens": 0, "cost": cost}
    )


def test_the_clef_usage_model_reads_an_optional_cost() -> None:
    assert clef_usage(2.652e-05).cost == 2.652e-05
    assert ClefUsage(input_tokens=1, output_tokens=2).cost is None
    parsed = parse_clef(BODY).usage
    assert parsed.cost == BODY["usage"]["cost"]
    assert parsed.model_extra == {"is_byok": False}  # other keys are kept


@pytest.mark.parametrize("cost", [[], "free", True, None])
def test_a_cost_of_the_wrong_type_does_not_invalidate_the_response(cost: Any) -> None:
    usage = ClefUsage.model_validate(
        {"input_tokens": 1, "output_tokens": 0, "cost": cost}
    )
    assert usage.cost is None


def test_usage_sums_reported_costs_apart_from_tokens() -> None:
    usage = Usage()
    usage.add_call(MODEL, "gate", clef_usage(0.25))
    usage.add_call(MODEL, "gate", clef_usage(0.5))
    usage.add_call(MODEL, "gate", ClefUsage(input_tokens=1, output_tokens=0))
    assert usage.reported_cost_usd == 0.75
    assert (usage.requests, usage.reported_requests) == (3, 2)
    assert usage.by_model[MODEL].reported_cost_usd == 0.75
    assert usage.by_kind["gate"].reported_requests == 2
    total = usage + usage
    assert total.reported_cost_usd == 1.5
    assert total.by_model[MODEL].reported_requests == 4


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf"), True, "0.1"])
def test_a_cost_that_is_not_a_price_is_not_reported(cost: Any) -> None:
    usage = Usage()
    usage.add_call(
        MODEL,
        "gate",
        ClefUsage.model_validate(
            {"input_tokens": 1, "output_tokens": 0, "cost": 0.0}
        ).model_copy(update={"cost": cost}),
    )
    assert usage.reported_cost_usd == 0.0
    assert usage.reported_requests == 0


def test_a_model_whose_calls_all_reported_a_cost_is_priced() -> None:
    usage = Usage()
    usage.add_call(MODEL, "gate", clef_usage(0.25))
    usage.add_call(MODEL, "gate", clef_usage(0.5))
    report = Prices().cost(usage)
    assert report.complete
    assert report.unpriced_models == ()
    assert report.reported_usd == 0.75
    assert report.usd == 0.0  # the token estimate stays apart


def test_a_model_with_a_call_that_reported_none_is_not_priced() -> None:
    usage = Usage()
    usage.add_call(MODEL, "gate", clef_usage(0.25))
    usage.add_call(MODEL, "gate", ClefUsage(input_tokens=10, output_tokens=0))
    report = Prices().cost(usage)
    assert not report.complete
    assert report.unpriced_models == (MODEL,)
    assert report.reported_usd == 0.0  # a partial sum would mislead


def test_a_configured_price_and_reported_costs_are_two_figures() -> None:
    usage = Usage()
    usage.add_call(MODEL, "gate", clef_usage(0.25, tokens=1_000_000))
    report = Prices({"clef": 2.0}).cost(usage)
    assert (report.usd, report.reported_usd, report.complete) == (2.0, 0.25, True)


async def test_the_cost_and_request_id_reach_the_call_record() -> None:
    headers = {"x-correlation-id": "corr-7", "x-request-id": "req-7"}
    transport, _ = make(httpx.Response(200, json=BODY, headers=headers))
    run = Run(transport=transport)
    answers = await run.call_clef(
        {"review": ClefRequest.model_validate(PAYLOAD).questions["review"]},
        "s",
        meta=CallMeta(kind="gate"),
    )
    assert answers["review"].type == "choice"
    (record,) = run.calls
    assert record.cost_usd == BODY["usage"]["cost"]
    assert record.request_id == "corr-7"
    assert run.usage.reported_cost_usd == BODY["usage"]["cost"]
    assert run.usage.reported_requests == 1
    assert Prices().cost(run.usage).reported_usd == BODY["usage"]["cost"]
    assert CallRecord(model=MODEL, kind="gate").cost_usd is None


# --------------------------------------------------------------------------- #
# 5. RoutingTransport.aclose closes every member
# --------------------------------------------------------------------------- #


class Breaks(Tag):
    async def aclose(self) -> None:
        self.closed.append("aclose")
        raise RuntimeError(f"{self.name} would not close")


async def test_aclose_closes_every_member_then_raises_the_first_error() -> None:
    first, second, third = Breaks("first"), Breaks("second"), Tag("third")
    router = RoutingTransport({"clairic": first, "clef": second}, default=third)
    with pytest.raises(RuntimeError, match="first would not close"):
        await router.aclose()
    assert first.closed == ["aclose"]
    assert second.closed == ["aclose"]
    assert third.closed == ["aclose"]


# --------------------------------------------------------------------------- #
# 6. The request id on failures of a 2xx and on responses
# --------------------------------------------------------------------------- #

IDS = {"x-correlation-id": "corr-42", "x-request-id": "req-42"}


async def test_a_success_carries_the_request_id() -> None:
    transport, _ = make(httpx.Response(200, json=BODY, headers=IDS))
    assert (await call(transport)).request_id == "corr-42"
    only_request_id = make(
        httpx.Response(200, json=BODY, headers={"x-request-id": "r"})
    )
    assert (await call(only_request_id[0])).request_id == "r"
    assert (await call(make(ok())[0])).request_id is None


async def test_a_2xx_that_is_not_a_clef_response_names_the_request() -> None:
    transport, _ = make(httpx.Response(200, json={"model": "clef"}, headers=IDS))
    with pytest.raises(ModelResponseError) as info:
        await call(transport)
    assert info.value.request_id == "corr-42"
    assert "corr-42" in str(info.value)


async def test_a_failure_inside_a_200_names_the_request() -> None:
    body = {"error": {"message": "no credit"}}
    transport, _ = make(httpx.Response(200, json=body, headers=IDS))
    with pytest.raises(BadRequestError, match="no credit") as info:
        await call(transport)
    assert info.value.request_id == "corr-42"
    assert "corr-42" in str(info.value)
    assert info.value.retries == 0


async def test_non_2xx_errors_carry_the_request_id_too() -> None:
    transport, _ = make(httpx.Response(503, json={"error": "down"}, headers=IDS))
    with pytest.raises(UncertainOutcomeError) as info:
        await call(transport)
    assert info.value.request_id == "corr-42"


async def test_a_failed_call_is_logged_with_its_request_id() -> None:
    transport, _ = make(httpx.Response(503, json={"error": "down"}, headers=IDS))
    run = Run(transport=transport)
    with pytest.raises(UncertainOutcomeError):
        await run.call_clef(
            {"review": ClefRequest.model_validate(PAYLOAD).questions["review"]},
            "s",
            meta=CallMeta(kind="gate"),
        )
    (record,) = run.calls
    assert record.request_id == "corr-42"
    assert record.status_code == 503
    assert record.error is not None and record.error.startswith("UncertainOutcome")


async def test_an_unparsable_answer_is_logged_with_its_request_id() -> None:
    bad = {**BODY, "answers": {"review": {"type": "choice"}}}
    transport, _ = make(httpx.Response(200, json=bad, headers=IDS))
    # the transport already refuses it, so the log entry comes from the failure
    run = Run(transport=transport)
    with pytest.raises(ModelResponseError):
        await run.call_clef(
            {"review": ClefRequest.model_validate(PAYLOAD).questions["review"]},
            "s",
            meta=CallMeta(kind="gate"),
        )
    assert [c.request_id for c in run.calls] == ["corr-42"]


# --------------------------------------------------------------------------- #
# Nits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "error", [OSError("tls reset"), RuntimeError("anyio went wrong"), ValueError("x")]
)
async def test_any_error_from_the_send_is_an_uncertain_outcome(
    error: Exception,
) -> None:
    transport, recorder = make(error, ok())
    with pytest.raises(UncertainOutcomeError) as info:
        await call(transport)
    assert info.value.__cause__ is error
    assert len(recorder.requests) == 1


async def test_a_payload_that_cannot_be_encoded_fails_before_sending() -> None:
    transport, recorder = make(ok())
    with pytest.raises(TypeError):
        await transport.run(CLEF, {**PAYLOAD, "state": {"x": {1, 2}}}, meta=META)
    assert recorder.requests == []  # nothing was sent, so it is not "uncertain"


def test_scrub_widens_to_keys_with_underscores_and_dashes() -> None:
    dashed = "xpl_" + "ab-cd_ef-12_34-56_78-9a"
    cleaned = scrub(f"key={dashed} end")
    assert dashed not in cleaned and "ab-cd" not in cleaned
    assert cleaned == "key=<redacted> end"


def test_have_credentials_strips_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", "  \n")
    assert not _have_credentials("experiential")
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", f" {KEY}\n")
    assert _have_credentials("experiential")
    assert _have_credentials("cloudflare")


def test_a_bad_provider_variable_fails_when_the_extractor_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str

    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "openai")
    with pytest.raises(ValueError, match="TRUSTCALL_S1_CLEF_PROVIDER"):
        create_extractor([Person])
    # an explicit transport makes the variable irrelevant
    create_extractor([Person], transport=Tag("x")).close()


def test_clef_provider_without_live_warns(capsys: pytest.CaptureFixture[str]) -> None:
    args = _parser().parse_args(["run", "--clef-provider", "experiential"])
    _settings(args, {})
    assert "no effect without --live" in capsys.readouterr().err
    args = _parser().parse_args(["run", "--live", "--clef-provider", "experiential"])
    _settings(args, {})
    assert capsys.readouterr().err == ""


async def test_a_replay_report_shows_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = await run_suite(
        CASES[:1], Settings(replay=tmp_path, clef_provider="experiential")
    )
    assert suite.mode == "replay"
    assert suite.clef_provider == "experiential"
    text = suite_markdown(suite)
    assert "Clef provider: experiential (replay only: nothing was sent)" in text
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "cloudflare")
    again = await run_suite(CASES[:1], Settings(replay=tmp_path))
    assert again.clef_provider == "cloudflare"
    offline = await run_suite(CASES[:1], Settings())
    assert offline.clef_provider == ""


class Dies:
    """A live transport whose first request ends in an uncertain outcome."""

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        raise UncertainOutcomeError("timed out; outcome unknown", status_code=504)


async def test_a_failed_case_counts_its_requests_and_unknown_outcomes() -> None:
    result = await _run_case(CASES[0], Settings(live=True), Dies())
    assert result.error is not None and "UncertainOutcomeError" in result.error
    assert result.requests >= 1
    assert result.unknown_outcomes >= 1
    agg = aggregate([result, CaseResult(case="b", unknown_outcomes=2)])
    assert agg["unknown_outcomes"] == result.unknown_outcomes + 2


def test_unknown_outcomes_are_counted_from_the_call_log() -> None:
    calls = [
        CallRecord(model=MODEL, kind="gate", error="UncertainOutcomeError: x"),
        CallRecord(model=MODEL, kind="gate", error=CANCELLED_AFTER_SEND),
        CallRecord(model=MODEL, kind="gate", error="AuthError: nope"),
        CallRecord(model=MODEL, kind="gate"),
    ]
    assert count_unknown_outcomes(calls) == 2
