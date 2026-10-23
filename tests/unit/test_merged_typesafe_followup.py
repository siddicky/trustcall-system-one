"""The second re-check of the merged TypeSafe provider and M11 branch.

Cancellation stays a cancellation, a clipped docstring cannot fool the total
heuristic, a shared stitch failure does not inflate the call log, a question that is
too large alone is isolated at the real limits, and the warnings of a narrowing are
recorded and replayed with a cassette.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from tests.unit.test_merged_typesafe_review import (
    ASKED,
    META,
    STATE,
    Gate,
    jev_answer,
    question,
    split_transport,
    typesafe,
)
from trustcall_system_one import (
    CLEF,
    BadRequestError,
    RequestTooLargeError,
    TypeSafeConfig,
    UncertainOutcomeError,
    compile_schema,
)
from trustcall_system_one._candidates import _is_total, own_description
from trustcall_system_one._compile import _doc_for_context
from trustcall_system_one._run import CANCELLED_AFTER_SEND, Run
from trustcall_system_one._typesafe import _own_copy
from trustcall_system_one._wire import ClairicField
from trustcall_system_one.testing import (
    CassetteTransport,
    FakeTransport,
    RecordingTransport,
    noul,
)
from trustcall_system_one.testing import clef_response as clef_reply
from trustcall_system_one.transport import (
    CallMeta,
    TransportResponse,
    _blame,
)

# --------------------------------------------------------------------------- #
# A. A caller's cancellation is never turned into an error
# --------------------------------------------------------------------------- #


def test_blame_leaves_a_cancellation_alone() -> None:
    cancelled = asyncio.CancelledError()
    cancelled.sibling_errors = [  # type: ignore[attr-defined]
        UncertainOutcomeError("a sibling timed out")
    ]
    cancelled.cancelled_siblings = 2  # type: ignore[attr-defined]
    assert _blame(cancelled) is cancelled
    interrupt = KeyboardInterrupt()
    interrupt.sibling_errors = [UncertainOutcomeError("x")]  # type: ignore[attr-defined]
    assert _blame(interrupt) is interrupt


def test_blame_does_not_annotate_a_sibling_error_in_place() -> None:
    refusal = RequestTooLargeError("too large")
    timed_out = UncertainOutcomeError("a sibling timed out")
    refusal.sibling_errors = [timed_out]  # type: ignore[attr-defined]
    chosen = _blame(refusal)
    assert chosen is not timed_out  # a copy was annotated
    assert isinstance(chosen, UncertainOutcomeError)
    assert chosen.maybe_sent is True
    assert "sibling_errors" not in vars(timed_out)
    assert "maybe_sent" not in vars(timed_out)


@pytest.mark.parametrize("a_part_times_out", [False, True])
async def test_cancelling_a_split_call_stays_a_cancellation(
    a_part_times_out: bool,
) -> None:
    sent: list[list[str]] = []
    arrived = Gate(3)
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        qids = list(json.loads(request.content)["questions"])
        sent.append(qids)
        await arrived.wait()
        if (
            a_part_times_out and qids[0] == "q0"
        ):  # it times out as the call is cancelled
            await release.wait()
            raise httpx.ReadTimeout("read timed out", request=request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    run = Run(transport=split_transport(handler))
    task = asyncio.ensure_future(run.call_clef(ASKED, STATE, meta=META))
    await arrived.open.wait()  # all three parts are on the wire
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):  # not an UncertainOutcomeError
        await task
    assert len(sent) == 3  # nothing was sent again
    errors = [c.error or "" for c in run.calls]
    assert errors  # the parts that were sent are logged, none of them as a refusal
    assert not [e for e in errors if e.startswith("RequestTooLarge")]
    if not a_part_times_out:  # every part was cancelled in flight: each is logged
        assert errors.count(CANCELLED_AFTER_SEND) == 4  # the call and its 3 parts


# --------------------------------------------------------------------------- #
# B. A docstring clipped inside a parenthesis
# --------------------------------------------------------------------------- #


class ClippedTimes(BaseModel):
    """Total time in service (including overhauls, APU hours and every other counted
    interval that the maintenance programme tracks for the airframe and its
    engines, as recorded) for the aircraft."""

    airframe_hours: float = Field(description="Airframe hours")


class ClippedAircraft(BaseModel):
    """An aircraft."""

    times: ClippedTimes


def test_a_clipped_docstring_leaves_no_open_parenthesis() -> None:
    doc = ClippedTimes.__doc__ or ""
    context = _doc_for_context(" ".join(doc.split()))
    assert context.count("(") == context.count(")")
    assert context.startswith("Total time in service")
    # a balanced docstring is kept as it is
    assert _doc_for_context("Hours (and minutes)") == "Hours (and minutes)"


def test_a_clipped_docstring_does_not_turn_on_sums() -> None:
    plan = compile_schema(ClippedAircraft)
    fields = [
        s.clairic
        for s in plan.all_specs()
        if s.clairic is not None and s.clairic.type == "number"
    ]
    assert fields
    for field in fields:
        text = field.instructions or ""
        assert text.count("(") == text.count(")")  # the context is well formed
        assert "Airframe hours" in text
        assert not _is_total(field)


@pytest.mark.parametrize(
    "text",
    [
        "Aircraft → times (Total time (including overhauls…): Airframe hours.",
        "Aircraft → times (Total time in service (including…): Airframe hours.",
    ],
)
def test_own_description_survives_an_unbalanced_context(text: str) -> None:
    assert own_description(text) == "Airframe hours."


# --------------------------------------------------------------------------- #
# C. A shared stitch failure does not inflate the call log
# --------------------------------------------------------------------------- #


def test_a_failure_copied_for_a_waiter_drops_the_leaders_notes() -> None:
    leader = RequestTooLargeError("refused")
    leader.sibling_errors = [UncertainOutcomeError("x")]  # type: ignore[attr-defined]
    leader.cancelled_siblings = 3  # type: ignore[attr-defined]
    leader._blamed = leader  # type: ignore[attr-defined]
    leader.maybe_sent = True
    mine = _own_copy(leader)
    assert mine is not leader
    for name in ("sibling_errors", "cancelled_siblings", "_blamed", "maybe_sent"):
        assert name not in vars(mine), name
    assert vars(leader)["sibling_errors"] and vars(leader)["cancelled_siblings"] == 3


async def test_waiting_for_a_slot_is_not_a_request_that_was_sent() -> None:
    """A stitch split into many batches: most are still waiting for a limiter slot
    when the first fails. Only the requests that were sent are in the call log."""
    sent: list[int] = []
    slots = typesafe(jev_answer).limiter.limit("typesafe")
    gate = Gate(slots)  # as many requests as the limiter lets out at once

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(1)
        first = len(sent) == 1
        await gate.wait()
        if first:
            return httpx.Response(400, json={"detail": "max_tokens_exceeded"})
        await asyncio.Event().wait()  # in flight, never answered
        raise AssertionError("unreachable")

    config = TypeSafeConfig(request_token_budget=300, verify_selection=False)
    transport = typesafe(handler, config=config)
    text = "\n".join(
        f"Line {i} of the notes about the part\nthat continues here" for i in range(30)
    )
    first = {"notes": ClairicField(type="text", instructions="The notes")}
    second = {"summary": ClairicField(type="text", instructions="The summary")}
    leader, waiter = Run(transport=transport), Run(transport=transport)
    results = await asyncio.wait_for(
        asyncio.gather(
            leader.call_clairic(first, text, meta=CallMeta(kind="extract")),
            waiter.call_clairic(second, text, meta=CallMeta(kind="extract")),
            return_exceptions=True,
        ),
        30,
    )
    # the freed slot let ONE more batch out; the rest never left the queue
    assert slots < len(sent) <= slots + 1
    assert isinstance(results[0], BadRequestError)
    error: Any = results[0]
    assert error.maybe_sent is True
    assert error.cancelled_siblings == len(sent) - 1  # only those that were in flight
    # one row per request that was actually sent, for the leader's call
    assert len(leader.calls) == len(sent)
    # the waiter sent nothing of its own: one row for its own failed call
    assert len(waiter.calls) <= 1
    assert results[1] is not results[0]  # it never gets the leader's error object


# --------------------------------------------------------------------------- #
# D. A question too large alone is isolated at the real limits
# --------------------------------------------------------------------------- #


async def test_a_question_over_the_budget_is_refused_alone_by_default() -> None:
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(list(json.loads(request.content)["questions"]))
        return jev_answer(request)

    run = Run(transport=typesafe(handler))  # the real limits: nothing is lowered
    asked = {
        "a": question("Is a so?"),
        "big": question("word " * 22_500),  # ~25k tokens: over 24k, under Jev's 30k
        "c": question("Is c so?"),
    }
    answers = await run.call_clef(asked, {"input": "t"}, meta=META)
    assert set(answers) == {"a", "c"}
    assert sent == [["a", "c"]]  # the siblings, together, once
    rejected = [w for w in run.warnings if w.startswith("request_rejected")]
    assert (
        len(rejected) == 1 and "'big'" in rejected[0] and "(1 item(s))" in rejected[0]
    )
    assert any(w.startswith("clef_request_isolated") for w in run.warnings)
    assert not any(w.startswith("clef_request_resplit") for w in run.warnings)


async def test_every_oversized_question_is_isolated_in_turn() -> None:
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(list(json.loads(request.content)["questions"]))
        return jev_answer(request)

    run = Run(transport=typesafe(handler))
    asked = {
        "big1": question("word " * 22_500),
        "a": question("Is a so?"),
        "big2": question("word " * 22_500),
        "b": question("Is b so?"),
    }
    answers = await run.call_clef(asked, {"input": "t"}, meta=META)
    assert set(answers) == {"a", "b"}
    assert sent == [["a", "b"]]
    assert len([w for w in run.warnings if w.startswith("request_rejected")]) == 2


async def test_a_lone_oversized_question_is_dropped_and_strict_raises() -> None:
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return jev_answer(request)

    run = Run(transport=typesafe(handler))
    big = {"big": question("word " * 22_500)}
    assert await run.call_clef(big, {"input": "t"}, meta=META) == {}
    assert not sent
    strict = Run(transport=typesafe(lambda r: jev_answer(r)), strict=True)
    with pytest.raises(RequestTooLargeError):
        await strict.call_clef(big, {"input": "t"}, meta=META)


# --------------------------------------------------------------------------- #
# E. The warnings of a narrowing are recorded and replayed
# --------------------------------------------------------------------------- #

NOTE = "typesafe_state_narrowed: clef_state_narrowed: cut to windows"


class Narrowing:
    """A transport that narrows the state and says so."""

    async def run(
        self,
        model: str,
        payload: Any,
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        body = clef_reply(
            {q: noul(0.9) for q in payload["questions"]}, model=payload["model"]
        )
        return TransportResponse(body=body, warnings=(NOTE,))


async def test_a_cassette_replays_the_warnings_it_recorded(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    payload = {
        "model": "clef",
        "state": {"input": "t"},
        "questions": {"q": question("Is it so?").model_dump(mode="json")},
    }
    live = Run(transport=RecordingTransport(Narrowing(), path))
    await live.call_clef({"q": question("Is it so?")}, {"input": "t"}, meta=META)
    assert NOTE in live.warnings
    recorded = json.loads(path.read_text().splitlines()[0])
    assert recorded["warnings"] == [NOTE]

    replayed = Run(transport=CassetteTransport(path))
    await replayed.call_clef({"q": question("Is it so?")}, {"input": "t"}, meta=META)
    assert NOTE in replayed.warnings  # the replay says what the recording said

    # a cassette written before the field existed replays as it always did
    del recorded["warnings"]
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps(recorded) + "\n")
    quiet = Run(transport=CassetteTransport(old))
    await quiet.call_clef({"q": question("Is it so?")}, {"input": "t"}, meta=META)
    assert not quiet.warnings
    reply = await CassetteTransport(old).run(CLEF, payload, meta=META)
    assert reply.warnings == ()


async def test_a_call_without_warnings_writes_no_field(tmp_path: Path) -> None:
    path = tmp_path / "plain.jsonl"

    def answer(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_reply(
            {q: noul(0.9) for q in payload["questions"]}, model=payload["model"]
        )

    run = Run(transport=RecordingTransport(FakeTransport(answer), path))
    await run.call_clef({"q": question("Is it so?")}, {"input": "t"}, meta=META)
    assert "warnings" not in json.loads(path.read_text().splitlines()[0])
