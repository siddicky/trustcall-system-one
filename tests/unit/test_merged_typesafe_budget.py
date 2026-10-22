"""The TypeSafe provider and the M11 request budget and error containment, together.

Each side was written and reviewed alone; these tests pin what they promise when
combined:

* the engine has ONE effective Clef budget (``Run.clef_budget``), the smaller of the
  library's knob and what ``TypeSafeTransport`` says Jev takes, and no request is
  built over it;
* a refusal that ``TypeSafeTransport`` makes locally ("too large for Jev's context",
  before anything is sent) is a size refusal the engine contains: a
  ``request_rejected`` warning, not an abort;
* an error that may have been run and charged (``maybe_sent``, ``partial_usage``) is
  never contained or resent, even when it looks like a size refusal.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from tests.unit.jev_mock import Bridge, JevBackend
from tests.unit.test_request_budget import Part, table_world
from trustcall_system_one import (
    BadRequestError,
    CorefConfig,
    Limits,
    Many,
    RequestTooLargeError,
    TypeSafeTransport,
    create_extractor,
)
from trustcall_system_one._errors import is_request_rejection, is_size_rejection
from trustcall_system_one._render import estimate_request_tokens
from trustcall_system_one._run import Run
from trustcall_system_one._wire import ClefUsage, NoulCriteria, NoulQuestion
from trustcall_system_one.testing import FakeTransport
from trustcall_system_one.transport import CallMeta, state_token_budget

KEY = "test-key-for-typesafe"
KNOB = 56_000
JEV = 24_000
META = CallMeta(kind="gate")


def typesafe(handler: Any) -> TypeSafeTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TypeSafeTransport(api_key=KEY, client=client, max_retries=0)


def question(text: str) -> NoulQuestion:
    return NoulQuestion(
        type="noul", instructions=text, criteria=NoulCriteria(true="yes", false="no")
    )


def jev_answer(request: httpx.Request) -> httpx.Response:
    """A Jev that answers every question of a request with ``noul`` 0.9."""
    body = json.loads(request.content)
    answers = {q: {"type": "noul", "noul": 0.9} for q in body["questions"]}
    usage = {"input_tokens": 1000, "output_tokens": 5}
    return httpx.Response(
        200, json={"model": "jev-1.14.0", "answers": answers, "usage": usage}
    )


class Spy(Bridge):
    """A ``Bridge`` that keeps the Clef requests the ENGINE built."""

    def __init__(self, inner: Any, backend: JevBackend) -> None:
        super().__init__(inner, backend)
        self.clef: list[tuple[str, float]] = []

    async def run(self, model: str, payload: Any, **kwargs: Any) -> Any:
        if "questions" in payload:
            size = estimate_request_tokens(payload["state"], payload["questions"])
            self.clef.append((kwargs["meta"].kind, size))
        return await super().run(model, payload, **kwargs)


# --------------------------------------------------------------------------- #
# (a) one effective budget: min(56k, 24k)
# --------------------------------------------------------------------------- #


def test_the_effective_budget_is_the_smaller_of_the_knob_and_jevs() -> None:
    transport = typesafe(jev_answer)
    assert state_token_budget(transport, "clef") == JEV
    assert Run(transport=transport).clef_budget() == min(KNOB, JEV) == JEV
    # the knob wins when it is the smaller one; a transport with no budget says nothing
    tight = Run(transport=transport, limits=Limits(clef_state_token_budget=5_000))
    assert tight.clef_budget() == 5_000
    assert Run(transport=FakeTransport(lambda *_: None)).clef_budget() == KNOB


async def test_no_request_exceeds_the_budget_on_a_table_like_input() -> None:
    world, text, names = table_world(120)
    # the control: on a transport that says nothing the same input DOES build
    # requests over Jev's budget (only the 56k knob applies)
    control = world.transport()
    await create_extractor(
        [Many(Part)], transport=control, coref=CorefConfig(strategy="incremental")
    ).ainvoke(text)
    built = [
        estimate_request_tokens(body["state"], body["questions"])
        for _, body, _ in control.calls
        if "questions" in body
    ]
    assert max(built) > JEV

    world, text, names = table_world(120)
    backend = JevBackend(world)
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    spy = Spy(TypeSafeTransport(api_key=KEY, client=client), backend)
    extractor = create_extractor(
        [Many(Part)], transport=spy, coref=CorefConfig(strategy="incremental")
    )
    result = await extractor.ainvoke(text)
    assert spy.clef and max(size for _, size in spy.clef) <= JEV
    assert backend.requests  # it really went through the Jev wire
    assert (
        sorted(r.value.name for r in result.records if isinstance(r.value, Part))
        == names
    )
    assert not [w for w in result.warnings if w.startswith("request_rejected")]


# --------------------------------------------------------------------------- #
# (b) a local "too large" refusal of TypeSafe is contained
# --------------------------------------------------------------------------- #

HUGE = "x " + "-" * 120_000  # the engine's estimate sees ~17k tokens, Jev's ~40k


async def test_a_local_too_large_refusal_is_a_rejected_request_not_an_abort() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return jev_answer(request)

    run = Run(transport=typesafe(handler))
    big = {"big": question(HUGE)}
    assert estimate_request_tokens({"input": "t"}, big) <= run.clef_budget()  # passes
    answers = await run.call_clef(big, {"input": "t"}, meta=META)
    assert answers == {}  # its question is unanswered, the call returned
    assert not sent  # refused before anything was put on the wire
    rejected = [w for w in run.warnings if w.startswith("request_rejected")]
    assert len(rejected) == 1 and "RequestTooLargeError" in rejected[0]

    # the same request is never tried again
    assert await run.call_clef(big, {"input": "t"}, meta=META) == {}
    assert not sent

    # strict mode raises it
    strict = Run(transport=typesafe(handler), strict=True)
    with pytest.raises(RequestTooLargeError):
        await strict.call_clef(big, {"input": "t"}, meta=META)
    assert not sent


async def test_the_other_questions_of_a_refused_request_are_still_asked() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return jev_answer(request)

    run = Run(transport=typesafe(handler))
    asked = {"big": question(HUGE), "small": question("Is it small?")}
    answers = await run.call_clef(asked, {"input": "t"}, meta=META)
    assert set(answers) == {"small"}  # the one that fits was resent on its own
    assert [list(json.loads(r.content)["questions"]) for r in sent] == [["small"]]
    assert any(w.startswith("clef_request_resplit") for w in run.warnings)
    assert any(w.startswith("request_rejected") for w in run.warnings)


def test_the_local_refusals_of_to_jev_clef_are_typed_size_refusals() -> None:
    from trustcall_system_one._typesafe import to_jev_clef

    huge = {"type": "noul", "instructions": HUGE, "criteria": {"true": "y"}}
    payload = {"model": "clef", "state": {"input": "t"}, "questions": {"q": huge}}
    with pytest.raises(RequestTooLargeError) as alone:
        to_jev_clef(
            payload, "jev-latest", question_budget=30_000, request_budget=60_000
        )
    assert is_size_rejection(alone.value) and is_request_rejection(alone.value)
    # a state with nothing left to cut: many short strings, none above the floor
    state = {f"k{i}": "word " * 30 for i in range(900)}
    small = {"type": "noul", "instructions": "Is it?", "criteria": {"true": "y"}}
    stuck = {"model": "clef", "state": state, "questions": {"q": small}}
    with pytest.raises(RequestTooLargeError) as no_room:
        to_jev_clef(stuck, "jev-latest", question_budget=30_000, request_budget=60_000)
    assert is_size_rejection(no_room.value)
    # other local refusals (an image, a malformed request) are not size refusals
    with pytest.raises(BadRequestError) as other:
        to_jev_clef({"model": "clef"}, "jev-latest")
    assert not isinstance(other.value, RequestTooLargeError)
    assert not is_size_rejection(other.value)


# --------------------------------------------------------------------------- #
# (c) a charged error is never contained or resent
# --------------------------------------------------------------------------- #


def _split_transport() -> tuple[TypeSafeTransport, dict[str, int]]:
    """A Clef request that Jev gets as several requests; the second is refused 400."""
    seen = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 2:
            await asyncio.sleep(0.2)  # the other parts have answered by then
            return httpx.Response(
                400, json={"detail": "max_tokens_exceeded: context length"}
            )
        return jev_answer(request)

    transport = typesafe(handler)
    transport._request_tokens = 3420  # a split into several Jev requests
    return transport, seen


async def test_a_typesafe_error_with_partial_usage_is_never_contained_or_resent() -> (
    None
):
    transport, seen = _split_transport()
    run = Run(transport=transport)
    asked = {f"q{i}": question(f"Is it {i}?") for i in range(6)}
    with pytest.raises(BadRequestError) as caught:
        await run.call_clef(asked, {"input": "word " * 2000}, meta=META)
    error: Any = caught.value
    # the transport says it was charged for the parts that answered ...
    assert error.maybe_sent is True
    assert error.partial_usage.input_tokens == 1000 * (seen["n"] - 1)
    # ... so, although it reads like a size refusal, it is not one
    assert "max_tokens_exceeded" in f"{error} {error.body}"
    assert not is_size_rejection(error) and not is_request_rejection(error)
    assert not run._contains(error)
    # raised, not contained; not split or sent again; the charge is in the log
    assert seen["n"] > 2
    total = seen["n"]
    assert not run.rejected and not run.warnings
    assert run.usage.input_tokens == 1000 * (total - 1)  # filed, once
    assert seen["n"] == total


@pytest.mark.parametrize("mark", ["maybe_sent", "partial_usage"])
async def test_a_charged_local_refusal_is_not_contained_either(mark: str) -> None:
    """Even the typed local refusal, if anything says it may have been charged."""
    seen = {"n": 0}

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        seen["n"] += 1
        error = RequestTooLargeError("too large for Jev")
        if mark == "maybe_sent":
            error.maybe_sent = True
        else:
            error.partial_usage = ClefUsage(  # type: ignore[attr-defined]
                input_tokens=1000, output_tokens=5
            )
        raise error

    run = Run(transport=FakeTransport(handler))
    asked = {f"q{i}": question(f"Is it {i}?") for i in range(4)}
    with pytest.raises(RequestTooLargeError):
        await run.call_clef(asked, {"input": "t"}, meta=META)
    assert seen["n"] == 1  # one attempt: not split into halves, not repeated
    assert not run.rejected
    assert not [w for w in run.warnings if w.startswith("request_rejected")]


async def test_a_charge_behind_a_fallbacks_primary_error_is_never_contained() -> None:
    from trustcall_system_one import UncertainOutcomeError

    primary = UncertainOutcomeError("timed out after the parts were sent")
    primary.maybe_sent = True
    refused = BadRequestError("HTTP 413", status_code=413)
    refused.primary_error = primary
    assert not is_size_rejection(refused)
    run = Run(transport=FakeTransport(lambda *_: None))
    assert not run._contains(refused)
