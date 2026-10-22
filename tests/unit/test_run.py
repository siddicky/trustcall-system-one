"""Tests for the per-invoke Run: calls, call log, usage and warnings."""

import asyncio
import contextlib
from collections.abc import Mapping
from typing import Any

import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CallMeta,
    Limits,
    ModelResponseError,
    RateLimitError,
    TransportResponse,
    VerifyConfig,
)
from trustcall_system_one._render import render_clairic
from trustcall_system_one._run import Run, gather
from trustcall_system_one._wire import (
    ClairicField,
    NoulAnswer,
    NoulQuestion,
)
from trustcall_system_one.testing import (
    FakeTransport,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    route,
)

PNG = "data:image/png;base64,iVBORw0KGgo="
FIELDS = {"name": ClairicField(type="text", instructions="the name")}
QUESTION = NoulQuestion(type="noul", instructions="Does `input` say yes?")


def clairic_ok(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
    return clairic_response(
        {name: clairic_answer("text", "Ada") for name in payload["fields"]},
        state_text=payload["state"],
        input_tokens=250,
    )


def clef_ok(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
    return clef_response({qid: noul(0.9) for qid in payload["questions"]})


def run_with(handler: Any, **kwargs: Any) -> tuple[Run, FakeTransport]:
    transport = FakeTransport(handler)
    return Run(transport=transport, **kwargs), transport


# --------------------------------------------------------------------------- #
# Clairic
# --------------------------------------------------------------------------- #


async def test_clairic_request_is_built_from_the_run_settings() -> None:
    run, transport = run_with(
        route(extract=clairic_ok), images=[PNG], reference_date="2026-10-22"
    )
    rendered = render_clairic("Ada", label="the Person")
    response = await run.call_clairic(
        FIELDS, rendered, meta=CallMeta(kind="extract", schema="Person", entity="e0")
    )
    assert response.answers["name"].value == "Ada"
    ((model, payload, meta),) = transport.calls
    assert model == CLAIRIC
    assert payload["model"] == "clairic"
    assert payload["state"] == rendered.text
    assert payload["options"] == {"return_state_text": True}
    assert payload["reference_date"] == "2026-10-22"
    assert payload["images"] == [PNG]
    assert meta.kind == "extract"


async def test_a_plain_string_state_is_accepted() -> None:
    run, transport = run_with(route(extract=clairic_ok))
    await run.call_clairic(FIELDS, "just text", meta=CallMeta(kind="extract"))
    assert transport.calls[0][1]["state"] == "just text"
    assert "images" not in transport.calls[0][1]
    assert "reference_date" not in transport.calls[0][1]


async def test_call_is_logged_with_items_usage_and_timing() -> None:
    run, _ = run_with(route(extract=clairic_ok, clef_fields=clef_ok))
    meta = CallMeta(
        kind="extract", schema="Person", entity="e0", field="name", scope="s"
    )
    await run.call_clairic(FIELDS, "x", meta=meta)
    await run.call_clef(
        {"q": QUESTION}, {"input": "x"}, meta=CallMeta(kind="clef_fields")
    )
    first, second = run.calls
    assert (first.model, first.kind, first.schema, first.entity, first.field) == (
        CLAIRIC,
        "extract",
        "Person",
        "e0",
        "name",
    )
    assert first.scope == "s"
    assert first.n_items == 1
    assert first.usage["input_tokens"] == 250
    assert first.retries == 0
    assert first.error is None
    assert first.status_code == 200
    assert first.duration_ms >= 0
    assert (second.model, second.kind, second.n_items) == (CLEF, "clef_fields", 1)
    assert run.usage.requests == 2
    assert run.usage.by_model[CLAIRIC].input_tokens == 250
    assert run.usage.by_kind["extract"].requests == 1
    assert run.usage.by_kind["clef_fields"].requests == 1


async def test_anchor_warns_when_the_input_is_missing_from_state_text() -> None:
    def odd(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clairic_response(
            {"name": clairic_answer("text", "Ada")}, state_text="something else"
        )

    run, _ = run_with(route(extract=odd))
    rendered = render_clairic("Ada Lovelace", label="x")
    response = await run.call_clairic(FIELDS, rendered, meta=CallMeta(kind="extract"))
    assert run.anchor(rendered, response) is None
    assert run.warnings and "state_text_mismatch" in run.warnings[0]
    run.anchor(rendered, response)
    assert len(run.warnings) == 1  # warnings are not repeated


# --------------------------------------------------------------------------- #
# Clef
# --------------------------------------------------------------------------- #


async def test_more_than_64_questions_are_split_and_run_concurrently() -> None:
    in_flight = 0
    peak = 0
    release = asyncio.Event()

    async def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight == 3:
            release.set()
        await release.wait()
        in_flight -= 1
        return clef_response({qid: noul(0.9) for qid in payload["questions"]})

    run, transport = run_with(route(clef_fields=handler))
    questions = {f"q{i}": QUESTION for i in range(150)}
    answers = await run.call_clef(
        questions, {"input": "x"}, meta=CallMeta(kind="clef_fields")
    )
    assert sorted(answers) == sorted(questions)
    assert all(isinstance(a, NoulAnswer) for a in answers.values())
    sizes = [len(payload["questions"]) for _, payload, _ in transport.calls]
    assert sizes == [64, 64, 22]
    assert peak == 3  # all three requests were in flight together
    assert sorted(c.n_items for c in run.calls) == [22, 64, 64]
    assert run.usage.requests == 3


async def test_exactly_64_questions_are_one_request() -> None:
    run, transport = run_with(route(clef_fields=clef_ok))
    await run.call_clef(
        {f"q{i}": QUESTION for i in range(64)},
        {"input": "x"},
        meta=CallMeta(kind="clef_fields"),
    )
    assert len(transport.calls) == 1


async def test_no_questions_make_no_request() -> None:
    run, transport = run_with(route(clef_fields=clef_ok))
    assert (
        await run.call_clef({}, {"input": "x"}, meta=CallMeta(kind="clef_fields")) == {}
    )
    assert transport.calls == []


async def test_clef_requests_carry_the_images_and_the_model() -> None:
    def flash(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {qid: noul(0.9) for qid in payload["questions"]}, model="clef-flash"
        )

    run, transport = run_with(
        route(clef_fields=flash), images=[PNG], clef_model="@cf/cloudflare/clef-flash"
    )
    await run.call_clef(
        {"q": QUESTION}, {"input": "x"}, meta=CallMeta(kind="clef_fields")
    )
    ((model, payload, _),) = transport.calls
    assert model == "@cf/cloudflare/clef-flash"
    assert payload["model"] == "clef-flash"
    assert payload["images"] == [PNG]


async def test_per_call_model_override() -> None:
    run, transport = run_with(route(clef_fields=clef_ok))
    await run.call_clef(
        {"q": QUESTION}, {"input": "x"}, meta=CallMeta(kind="clef_fields"), model=CLEF
    )
    assert transport.calls[0][0] == CLEF


async def test_a_clef_call_near_the_context_limit_warns() -> None:
    def big(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {qid: noul(0.9) for qid in payload["questions"]}, input_tokens=65_000
        )

    run, _ = run_with(route(clef_fields=big))
    await run.call_clef(
        {"q": QUESTION}, {"input": "x"}, meta=CallMeta(kind="clef_fields")
    )
    assert run.warnings and "clef_state_possibly_truncated" in run.warnings[0]
    run2, _ = run_with(route(clef_fields=clef_ok))
    await run2.call_clef(
        {"q": QUESTION}, {"input": "x"}, meta=CallMeta(kind="clef_fields")
    )
    assert run2.warnings == []


async def test_clef_state_is_narrowed_over_the_budget() -> None:
    run, _ = run_with(
        route(),
        limits=Limits(clef_state_token_budget=1000),
        verify=VerifyConfig(window_chars=100),
    )
    text = "a" * 5000 + " Ada " + "b" * 5000
    state = run.clef_state(text, entity={"label": "Ada"}, spans=[(5001, 5004)])
    assert "Ada" in state["input"] and len(state["input"]) < 1000
    assert state["entity"] == {"label": "Ada"}
    assert run.warnings and "clef_state_narrowed" in run.warnings[0]
    small = run.clef_state("short")
    assert small == {"input": "short"}


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #


class Failing:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        raise self.error


async def test_failed_call_is_logged_first_and_then_propagates() -> None:
    error = RateLimitError("slow down", status_code=429, retry_after=2.0)
    error.retries = 6
    run = Run(transport=Failing(error))
    with pytest.raises(RateLimitError):
        await run.call_clairic(FIELDS, "x", meta=CallMeta(kind="extract", schema="P"))
    (record,) = run.calls
    assert record.kind == "extract"
    assert record.schema == "P"
    assert record.retries == 6
    assert record.status_code == 429
    assert record.error is not None and "slow down" in record.error
    assert record.usage == {}
    assert run.usage.requests == 0

    with pytest.raises(RateLimitError):
        await run.call_clef(
            {"q": QUESTION}, {"input": "x"}, meta=CallMeta(kind="presence")
        )
    assert [c.kind for c in run.calls] == ["extract", "presence"]
    assert run.calls[1].model == CLEF


async def test_unparsable_response_is_logged_and_raises() -> None:
    class Sloppy:
        async def run(
            self,
            model: str,
            payload: Mapping[str, Any],
            *,
            meta: CallMeta,
            timeout: float | None = None,
        ) -> TransportResponse:
            return TransportResponse(
                body={"nonsense": True}, retries=2, status_code=200
            )

    run = Run(transport=Sloppy())
    with pytest.raises(ModelResponseError):
        await run.call_clairic(FIELDS, "x", meta=CallMeta(kind="extract"))
    (record,) = run.calls
    assert record.retries == 2
    assert record.error is not None and "ModelResponseError" in record.error
    assert run.usage.requests == 0


async def test_a_failing_sibling_cancels_the_other_requests() -> None:
    cancelled = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def boom() -> None:
        await asyncio.sleep(0)
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await gather(slow(), boom())
    assert cancelled.is_set()


async def test_gather_returns_results_in_order() -> None:
    async def value(x: Any) -> Any:
        await asyncio.sleep(0)
        return x

    assert await gather(value(1), value(2)) == [1, 2]
    assert await gather() == []


async def test_gather_raises_the_real_error_not_a_siblings_cancellation() -> None:
    shared: asyncio.Future[None] = asyncio.ensure_future(asyncio.sleep(10))

    async def waits() -> None:
        await shared  # cancelled out from under it by the failing sibling

    async def boom() -> None:
        shared.cancel()
        await asyncio.sleep(0)  # the sibling ends up cancelled first
        await asyncio.sleep(0)
        raise ValueError("the real error")

    with pytest.raises(ValueError, match="the real error"):
        await gather(waits(), boom())
    with pytest.raises(ValueError, match="the real error"):
        await gather(boom(), waits())


async def test_gather_re_raises_the_callers_own_cancellation() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def child() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()

    task = asyncio.ensure_future(gather(child(), child()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


async def test_gather_with_a_cancelled_sibling_and_no_error_raises_cancelled() -> None:
    async def victim() -> None:
        await asyncio.sleep(10)

    async def cancel_it(task: asyncio.Future[None]) -> None:
        await asyncio.sleep(0)
        task.cancel()

    future = asyncio.ensure_future(victim())
    with pytest.raises(asyncio.CancelledError):
        await gather(future, cancel_it(future))


async def test_gather_keeps_the_callers_cancellation_over_a_sibling_error() -> None:
    """The caller is cancelled and a child fails while it is being stopped."""
    started = asyncio.Event()

    async def fails_when_stopped() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise ValueError("cleanup failed") from None

    async def waits() -> None:
        await asyncio.sleep(10)

    task = asyncio.ensure_future(gather(fails_when_stopped(), waits()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _cancelled_child_with_error_behind_it() -> None:
    """``gather`` of a child that ends cancelled and one that errors when stopped."""
    shared: asyncio.Future[None] = asyncio.ensure_future(asyncio.sleep(10))
    release = asyncio.Event()

    async def waits() -> None:
        await shared  # ends cancelled: the child cancellation

    async def boom() -> None:
        shared.cancel()
        try:
            await release.wait()
        except asyncio.CancelledError:
            raise ValueError("the real error") from None

    await gather(waits(), boom())


async def test_gather_substitutes_the_real_error_for_a_cancelled_sibling() -> None:
    with pytest.raises(ValueError, match="the real error"):
        await _cancelled_child_with_error_behind_it()


@pytest.mark.skipif(
    not hasattr(asyncio.Task, "cancelling"), reason="Task.cancelling needs 3.11"
)
async def test_gather_does_not_substitute_while_the_caller_is_being_cancelled() -> None:
    """A swallowed cancellation leaves ``cancelling() > 0``: gather stays cancelled."""

    async def caller() -> None:
        # swallowed, not uncancelled: the task is still "being cancelled"
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(10)
        await _cancelled_child_with_error_behind_it()

    task = asyncio.ensure_future(caller())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
