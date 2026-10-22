"""Regression tests of the review of the merged TypeSafe provider and M11 branch.

Every race is forced, not waited for: the handlers of the scripted Jev park on an
``asyncio.Event`` that is set only once every request of the call has arrived, so
the failures of the siblings land in the same tick, in an order the test chooses.
"""

import asyncio
import json
import random
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    CLEF,
    CLEF_FLASH,
    BadRequestError,
    FallbackTransport,
    RoutingTransport,
    TypeSafeConfig,
    TypeSafeTransport,
    UncertainOutcomeError,
    compile_schema,
)
from trustcall_system_one._candidates import _is_total, own_description
from trustcall_system_one._errors import may_have_run
from trustcall_system_one._render import estimate_request_tokens
from trustcall_system_one._run import CANCELLED_AFTER_SEND, Run
from trustcall_system_one._wire import ClairicField, NoulCriteria, NoulQuestion
from trustcall_system_one.testing import (
    CassetteTransport,
    FakeTransport,
    RecordingTransport,
    noul,
)
from trustcall_system_one.testing import clef_response as clef_reply
from trustcall_system_one.transport import CallMeta, state_token_budget

KEY = "test-key-for-typesafe"
META = CallMeta(kind="gate")
Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def typesafe(handler: Any, **kwargs: Any) -> TypeSafeTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TypeSafeTransport(api_key=KEY, client=client, max_retries=0, **kwargs)


def question(text: str) -> NoulQuestion:
    return NoulQuestion(
        type="noul", instructions=text, criteria=NoulCriteria(true="yes", false="no")
    )


def jev_answer(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    answers = {q: {"type": "noul", "noul": 0.9} for q in body["questions"]}
    usage = {"input_tokens": 1000, "output_tokens": 5}
    return httpx.Response(
        200, json={"model": "jev-1.14.0", "answers": answers, "usage": usage}
    )


class Gate:
    """Holds every request until ``n`` of them are waiting, then lets them all go."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.arrived = 0
        self.open = asyncio.Event()

    async def wait(self) -> None:
        self.arrived += 1
        if self.arrived >= self.n:
            self.open.set()
        await self.open.wait()


# --------------------------------------------------------------------------- #
# 1. A refusal that lands with failed siblings is not contained
# --------------------------------------------------------------------------- #

PARTS = [["q0", "q1"], ["q2", "q3"], ["q4", "q5"]]


def clef_split(plan: dict[str, str]) -> tuple[Handler, Gate, list[list[str]]]:
    """A scripted Jev for a Clef request split in three: what each part does.

    ``plan`` maps the first question of a part to ``400`` (max_tokens_exceeded),
    ``timeout`` (a read timeout: the request may have been charged), ``hang`` (never
    answers) or ``ok``. All but the ``hang`` parts act in the same tick.
    """
    sent: list[list[str]] = []
    gate = Gate(sum(1 for how in plan.values() if how != "hang"))

    async def handler(request: httpx.Request) -> httpx.Response:
        qids = list(json.loads(request.content)["questions"])
        sent.append(qids)
        how = plan.get(qids[0], "ok")
        if how == "hang":
            await asyncio.Event().wait()
        await gate.wait()
        if how == "400":
            return httpx.Response(400, json={"detail": "max_tokens_exceeded"})
        if how == "timeout":
            raise httpx.ReadTimeout("read timed out", request=request)
        return jev_answer(request)

    return handler, gate, sent


def split_transport(handler: Handler) -> TypeSafeTransport:
    transport = typesafe(handler)
    transport._request_tokens = 2400  # three parts
    return transport


ASKED = {f"q{i}": question(f"Is it {i}?") for i in range(6)}
STATE = {"input": "word " * 2000}


async def call(run: Run) -> Any:
    return await asyncio.wait_for(run.call_clef(ASKED, STATE, meta=META), 30)


@pytest.mark.parametrize(
    "plan",
    [
        {"q0": "400", "q2": "timeout", "q4": "ok"},  # the refusal wakes first
        {"q0": "timeout", "q2": "400", "q4": "ok"},  # the timeout wakes first
        {"q0": "ok", "q2": "timeout", "q4": "400"},  # the refusal wakes last
        {"q0": "400", "q2": "timeout", "q4": "timeout"},
    ],
    ids=["400-first", "timeout-first", "400-last", "two-timeouts"],
)
async def test_a_size_refusal_beside_a_timed_out_part_is_raised_not_contained(
    plan: dict[str, str],
) -> None:
    handler, _gate, sent = clef_split(plan)
    run = Run(transport=split_transport(handler))
    with pytest.raises(Exception) as caught:
        await call(run)
    error = caught.value
    assert may_have_run(error)  # it says a sibling may have been charged
    assert isinstance(error, UncertainOutcomeError)  # the uncertain error wins
    assert error.maybe_sent is True
    assert sent == PARTS  # nothing was sent a second time
    assert not run.rejected
    assert not [w for w in run.warnings if w.startswith("clef_request_")]
    assert not [w for w in run.warnings if w.startswith("request_rejected")]
    # nothing is invisible: the refused part is in the log with its status, and
    # every part that timed out is logged as the uncertain outcome it was
    statuses = [c.status_code for c in run.calls]
    assert 400 in statuses
    uncertain = [c for c in run.calls if (c.error or "").startswith("UncertainOutcome")]
    assert len(uncertain) == sum(1 for how in plan.values() if how == "timeout")


async def test_a_size_refusal_beside_a_part_that_was_cancelled_is_not_contained() -> (
    None
):
    handler, _gate, sent = clef_split({"q0": "400", "q2": "hang", "q4": "ok"})
    run = Run(transport=split_transport(handler))
    with pytest.raises(BadRequestError) as caught:
        await call(run)
    error: Any = caught.value
    assert may_have_run(error) and error.maybe_sent is True
    assert error.cancelled_siblings == 1
    assert sorted(sent) == PARTS  # the hung part was sent once, never again
    assert not [w for w in run.warnings if w.startswith("request_rejected")]
    assert [c.error for c in run.calls].count(CANCELLED_AFTER_SEND) == 1
    assert "were cancelled" in str(error)


async def test_a_refusal_of_every_part_is_still_one_uncontained_error() -> None:
    """Even when no sibling may have run, a split call with a failed sibling is not a
    single refused request: it is raised, the engine does not resend any part."""
    handler, _gate, sent = clef_split({"q0": "400", "q2": "400", "q4": "400"})
    run = Run(transport=split_transport(handler))
    with pytest.raises(BadRequestError) as caught:
        await call(run)
    assert may_have_run(caught.value)
    assert sent == PARTS
    assert not run.rejected and not run.warnings


async def test_a_single_refused_request_is_still_contained() -> None:
    """The change is only for calls of several requests."""
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        qids = list(json.loads(request.content)["questions"])
        sent.append(qids)
        if len(qids) > 3:
            return httpx.Response(400, json={"detail": "max_tokens_exceeded"})
        return jev_answer(request)

    run = Run(transport=typesafe(handler))
    answers = await call(run)
    assert set(answers) == set(ASKED)  # halved after the refusal, then answered
    assert sent == [[f"q{i}" for i in range(6)], ["q0", "q1", "q2"], ["q3", "q4", "q5"]]
    assert any(w.startswith("clef_request_resplit") for w in run.warnings)


async def test_clairic_batches_a_size_refusal_beside_timeouts_is_raised() -> None:
    sent: list[int] = []
    gate = Gate(4)  # the text below makes four one-question requests

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(len(json.loads(request.content)["questions"]))
        first = len(sent) == 1
        await gate.wait()
        if first:
            return httpx.Response(400, json={"detail": "max_tokens_exceeded"})
        raise httpx.ReadTimeout("read timed out", request=request)

    config = TypeSafeConfig(
        request_token_budget=600, stitch=False, verify_selection=False
    )
    run = Run(transport=typesafe(handler, config=config))
    text = "Contacts: " + "; ".join(
        f"Person{i} Smith at Acme{i} Ltd on 2024-0{1 + i % 9}-1{i % 9}"
        for i in range(60)
    )
    fields = {
        "name": ClairicField(type="text", instructions="The name of the person"),
        "company": ClairicField(type="text", instructions="The company"),
    }
    with pytest.raises(Exception) as caught:
        await asyncio.wait_for(
            run.call_clairic(fields, text, meta=CallMeta(kind="extract")), 30
        )
    error: Any = caught.value
    assert may_have_run(error) and error.maybe_sent is True
    assert isinstance(error, UncertainOutcomeError)
    assert sent == [1, 1, 1, 1]  # nothing was sent again
    assert not [w for w in run.warnings if w.startswith("request_rejected")]
    assert not run.rejected
    statuses = [c.status_code for c in run.calls]
    assert 400 in statuses  # the refused batch is logged
    timed_out = [c for c in run.calls if (c.error or "").startswith("UncertainOutcome")]
    assert len(timed_out) == 3  # and so is every batch that timed out


async def test_clairic_a_size_refusal_after_a_charged_stage_is_raised() -> None:
    """A refusal that follows a request that answered (charged): the older probe."""
    from tests.unit.test_typesafe_transport import (
        EMAIL_FIELD,
        Jev,
        clairic_payload,
        picks,
    )
    from trustcall_system_one._wire import ClairicRequest

    jev = Jev(picks(email="dana.personal@gmail.com"))
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 2:
            return httpx.Response(400, json={"detail": "max_tokens_exceeded"})
        return jev(request)

    run = Run(transport=typesafe(handler))
    request = ClairicRequest.model_validate(clairic_payload(EMAIL_FIELD))
    with pytest.raises(BadRequestError) as caught:
        await run.call_clairic(
            request.fields, request.state, meta=CallMeta(kind="extract")
        )
    assert may_have_run(caught.value)
    assert seen["n"] == 2
    assert run.usage.input_tokens > 0  # what the stitch cost is in the log
    assert not [w for w in run.warnings if w.startswith("request_rejected")]


# --------------------------------------------------------------------------- #
# 2. The transport reads the state the engine built, byte for byte
# --------------------------------------------------------------------------- #

_WORDS = [
    "the",
    "pilot",
    "reported",
    "that",
    "aircraft",
    "maintenance",
    "was",
    "completed",
    "before",
    "departure",
    "and",
    "the",
    "crew",
    "confirmed",
    "configuration",
    "inspection",
    "documentation",
]
NAMES = ["Captain Alvarez", "Officer Brennan"]


def prose(n_chars: int) -> str:
    """Long-word prose: the old chars/3 estimate cut it, the engine's does not."""
    rng = random.Random(1)
    body = " ".join(
        " ".join(rng.choice(_WORDS) for _ in range(12)) + "." for _ in range(12_000)
    )
    return f"{body[:n_chars]} {NAMES[0]} flew with {NAMES[1]}."


def padded(rows: int) -> str:
    """A space-padded spec table (the shape of the M11 corpus)."""

    def row(a: str, b: str, c: str) -> str:
        return f"| {a:<40} | {b:<30} | {c:<20} |"

    lines = [
        row("Item", "Part", "Action"),
        "|" + "-" * 42 + "|" + "-" * 32 + "|" + "-" * 22 + "|",
        *(
            row("Engine overhaul interval", f"Component {i}", "Inspect")
            for i in range(rows)
        ),
    ]
    note = f"Note: {NAMES[0]} replaced the pump; {NAMES[1]} tested it."
    return "\n".join([*lines, note])


def table_like(rows: int) -> str:
    """A dense, digit-heavy table: the engine counts it denser than chars/3."""
    lines = [
        f"| {i:>4} | PN-{i:05d} | {i * 37 % 1000:>6} | {i * 91 % 9999:>7} FH "
        f"| {NAMES[i % 2]} |"
        for i in range(rows)
    ]
    return "\n".join(lines)


def mentions_of(text: str) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    spans = [(text.rfind(n), text.rfind(n) + len(n)) for n in NAMES]
    return [
        {"text": n, "span": list(s)} for n, s in zip(NAMES, spans, strict=True)
    ], spans


def largest_unnarrowed(build: Callable[[int], str], steps: range) -> str:
    """The largest input the engine still leaves whole at TypeSafe's budget."""
    scratch = Run(transport=typesafe(jev_answer))
    best = ""
    for n in steps:
        text = build(n)
        mentions, spans = mentions_of(text)
        state = scratch.clef_state(text, extra={"mentions": mentions}, spans=spans)
        if state["input"] != text:
            break
        best = text
    assert best
    return best


@pytest.mark.parametrize(
    ("build", "steps"),
    [
        (prose, range(40_000, 200_000, 4_000)),
        (padded, range(100, 2_000, 25)),
        (table_like, range(50, 1_500, 25)),
    ],
    ids=["long-word-prose", "space-padded-table", "dense-table"],
)
async def test_jev_reads_exactly_the_state_the_engine_fitted(
    build: Callable[[int], str], steps: range
) -> None:
    text = largest_unnarrowed(build, steps)
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return jev_answer(request)

    run = Run(transport=typesafe(handler))
    mentions, spans = mentions_of(text)
    state = run.clef_state(text, extra={"mentions": mentions}, spans=spans)
    asked = {"q0": question("Do mention 0 and mention 1 name the same person?")}
    assert estimate_request_tokens(state, asked) <= run.clef_budget()
    answers = await run.call_clef(asked, state, meta=CallMeta(kind="coref"))
    assert set(answers) == {"q0"}
    (body,) = bodies
    assert body["state"] == state  # not one character was cut on the way
    assert not run.warnings  # and nothing was narrowed behind the engine's back
    sent = body["state"]
    for mention in sent["mentions"]:  # so the spans still read the names
        start, end = mention["span"]
        assert sent["input"][start:end] == mention["text"]


async def test_a_narrowing_the_transport_has_to_make_is_a_warning() -> None:
    """A request that did not come through the engine's budget is still never cut
    silently: the transport says so, and the engine records it."""
    filler = "filler words go here " * 6000
    text = f"{filler}{NAMES[0]} met {NAMES[1]}. {filler}"
    mentions, _ = mentions_of(text)
    state = {"input": text, "mentions": mentions}
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return jev_answer(request)

    run = Run(transport=typesafe(handler))
    transport: Any = run.transport
    transport.state_token_budget = 10_000_000  # engine would not narrow: it is told so
    run.limits = type(run.limits)(clef_state_token_budget=10_000_000)
    asked = {"q0": question("Same person?")}
    await run.call_clef(asked, state, meta=CallMeta(kind="coref"))
    (body,) = bodies
    assert len(body["state"]["input"]) < len(text)
    assert [w for w in run.warnings if w.startswith("typesafe_state_narrowed")]
    for mention in body["state"]["mentions"]:  # the spans were re-based
        start, end = mention["span"]
        assert body["state"]["input"][start:end] == mention["text"]


# --------------------------------------------------------------------------- #
# 3. The budget is the one of the model a request goes to
# --------------------------------------------------------------------------- #


async def test_the_budget_follows_the_model_through_a_routing_transport() -> None:
    jev_bodies: list[dict[str, Any]] = []

    def jev_handler(request: httpx.Request) -> httpx.Response:
        jev_bodies.append(json.loads(request.content))
        return jev_answer(request)

    seen: list[float] = []

    def default(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        seen.append(estimate_request_tokens(payload["state"], payload["questions"]))
        return clef_reply(
            {q: noul(0.9) for q in payload["questions"]}, model=payload["model"]
        )

    routing = RoutingTransport(
        {"clef-flash": typesafe(jev_handler)}, default=FakeTransport(default)
    )
    run = Run(transport=routing, gate_model=CLEF_FLASH)
    assert state_token_budget(routing, CLEF) is None
    assert run.clef_budget(CLEF) == 56_000  # the default transport says nothing
    assert run.clef_budget(CLEF_FLASH) == 24_000  # Jev's
    assert run.clef_budget() == 24_000  # a state for either model fits both

    state = {"input": "word " * 30_000}  # about 33k estimated tokens
    asked = {"q0": question("Is it so?")}
    assert 24_000 < estimate_request_tokens(state, asked) < 56_000
    await run.call_clef(asked, state, meta=CallMeta(kind="gate"), model=CLEF_FLASH)
    (body,) = jev_bodies
    assert estimate_request_tokens(body["state"], body["questions"]) <= 24_000
    assert any("clef_state_truncated" in w for w in run.warnings)

    run.warnings.clear()
    await run.call_clef(asked, state, meta=CallMeta(kind="gate"), model=CLEF)
    assert seen and seen[-1] > 24_000  # the model on the big transport is not cut
    assert not run.warnings
    # a state built for a model the run reads with is fitted to that model
    fitted = run.clef_state("word " * 30_000, model=CLEF_FLASH)
    assert estimate_request_tokens(fitted) <= 24_000


# --------------------------------------------------------------------------- #
# 4. Request ids do not defeat the circuit breaker
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("style", ["request id", "correlation id", "attribute"])
async def test_refusals_that_differ_only_in_their_request_id_trip_the_breaker(
    style: str,
) -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        rid = uuid.uuid4().hex
        message = "TypeSafe API returned HTTP 400: max_tokens_exceeded"
        if style != "attribute":
            message += f" ({style} {rid})"
        error = BadRequestError(message, status_code=400)
        error.request_id = rid
        raise error

    run = Run(transport=FakeTransport(handler))
    contained = 0
    with pytest.raises(BadRequestError):
        for i in range(12):  # a different question each time: a different request
            await run.call_clef(
                {f"q{i}": question(f"Is {i} so?")}, {"input": "t"}, meta=META
            )
            contained += 1
    assert contained == 8  # the ninth identical refusal is raised
    assert len(run.refusals) == 1  # one refusal, however many ids it came with


# --------------------------------------------------------------------------- #
# 5. A parent's docstring is not the field asking for a total
# --------------------------------------------------------------------------- #


class Times(BaseModel):
    """Total time in service, including overhauls."""

    airframe_hours: float = Field(description="Airframe hours")
    cycles: int = Field(description="Landing cycles")


class Aircraft(BaseModel):
    """An aircraft."""

    registration: str
    times: Times


class Invoice(BaseModel):
    """An invoice."""

    total: float = Field(description="The total amount due")


class Bill(BaseModel):
    """A bill."""

    inner: Invoice


def number_fields(
    model: type[BaseModel],
) -> dict[tuple[str | int, ...], ClairicField]:
    plan = compile_schema(model)
    return {
        spec.path: spec.clairic
        for spec in plan.all_specs()
        if spec.clairic is not None and spec.clairic.type == "number"
    }


def test_a_parent_docstring_with_total_does_not_turn_on_sums() -> None:
    fields = number_fields(Aircraft)
    assert set(fields) == {("times", "airframe_hours"), ("times", "cycles")}
    for field in fields.values():
        assert "Total time in service" in (field.instructions or "")  # the context
        assert not _is_total(field)


def test_a_field_that_itself_asks_for_a_total_still_does() -> None:
    top = number_fields(Invoice)[("total",)]
    assert _is_total(top)
    nested = number_fields(Bill)[("inner", "total")]
    assert "Bill" in (nested.instructions or "") and _is_total(nested)


@pytest.mark.parametrize(
    ("text", "own"),
    [
        (
            "Aircraft → times (Total time, including x): Airframe hours.",
            "Airframe hours.",
        ),
        ("A → b → c (Doc: with colon (nested)): The cycles", "The cycles"),
        ("Total cost", "Total cost"),  # no context: the whole text
        ("Invoice: the total amount", "Invoice: the total amount"),  # no arrow
        ("A → b: Pick: the total", "Pick: the total"),  # the first ": " after it
    ],
)
def test_own_description_strips_only_the_context(text: str, own: str) -> None:
    assert own_description(text) == own


# --------------------------------------------------------------------------- #
# 7. Wrappers say the budget of the model, per model
# --------------------------------------------------------------------------- #


class Budgeted:
    """A stand-in transport with a budget and, optionally, models it serves."""

    def __init__(self, budget: int | None, serves: set[str] | None = None) -> None:
        self.budget = budget
        self.models = serves

    def state_token_budget(self, model: str) -> int | None:
        return self.budget

    def serves(self, model: str, payload: Any = None) -> bool:
        return self.models is None or model in self.models

    async def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("not called")


def test_a_fallback_that_does_not_serve_the_model_does_not_narrow_it() -> None:
    primary = Budgeted(24_000)
    decisions = Budgeted(8_000, serves={CLEF})
    both = FallbackTransport(primary, decisions)
    assert both.state_token_budget(CLEF) == 8_000  # it would be asked: it takes less
    assert both.state_token_budget(CLEF_FLASH) == 24_000  # it never would
    silent = FallbackTransport(Budgeted(None), Budgeted(8_000, serves=set()))
    assert silent.state_token_budget(CLEF) is None


def test_a_cassette_replays_each_model_with_the_budget_it_was_recorded_with(
    tmp_path: Path,
) -> None:
    async def record(path: Path) -> None:
        def answer(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
            return clef_reply(
                {q: noul(0.9) for q in payload["questions"]}, model=payload["model"]
            )

        pro, flash = FakeTransport(answer), FakeTransport(answer)
        pro.state_token_budget = 20_000  # type: ignore[attr-defined]
        flash.state_token_budget = 8_000  # type: ignore[attr-defined]
        router = RoutingTransport({"clef": pro, "clef-flash": flash}, default=pro)
        recording = RecordingTransport(router, path)
        for model in (CLEF, CLEF_FLASH):
            payload = {
                "model": "clef-flash" if model == CLEF_FLASH else "clef",
                "state": {"input": "t"},
                "questions": {"q": question("Is it so?").model_dump(mode="json")},
            }
            await recording.run(model, payload, meta=META)

    path = tmp_path / "c.jsonl"
    asyncio.run(record(path))
    cassette = CassetteTransport(path)
    assert cassette.state_token_budget(CLEF) == 20_000
    assert cassette.state_token_budget(CLEF_FLASH) == 8_000
    assert cassette.fork().state_token_budget(CLEF_FLASH) == 8_000
    # a miss that goes to an inner transport is no larger than that one takes
    inner = Budgeted(5_000)
    both = CassetteTransport(path, strict=False, inner=inner)
    assert both.state_token_budget(CLEF) == 5_000
    assert both.state_token_budget(CLEF_FLASH) == 5_000
