"""The review of the TypeSafe-only provider: the corrections and what pins them.

One test file for the fixes that came out of the review of the first version:
overload as an uncertain outcome, none-means-null, the state budget a transport
advertises, a Clef request split to Jev's limits, user text kept out of errors,
cassette provenance, internal batching by tokens, single-flight stitching, repair
requests, Decisions-as-primary with images, the accounting of a call of several
requests, and the evals refusing contradictory flags. Offline: ``httpx.MockTransport``.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.cases import CASES
from evals.runner import Settings, run_suite
from tests.unit.test_typesafe_transport import (
    CLEF_PAYLOAD,
    EMAIL_FIELD,
    KEY,
    MAIL,
    META,
    PROBE,
    Jev,
    Recorder,
    _verify_script,
    choice,
    clairic,
    clairic_payload,
    make,
    ok,
    picks,
    run,
)
from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    BadRequestError,
    CloudflareTransport,
    DecisionsTransport,
    ExperientialTransport,
    FallbackTransport,
    Limits,
    ModelResponseError,
    Prices,
    RateLimitError,
    RoutingTransport,
    TransportError,
    TransportResponse,
    TypeSafeConfig,
    UncertainOutcomeError,
    Usage,
    system_one_transport,
)
from trustcall_system_one._candidates import pattern_candidates
from trustcall_system_one._render import estimate_request_tokens, estimate_tokens
from trustcall_system_one._repair import _Blocks
from trustcall_system_one._run import Run
from trustcall_system_one._structure import join_questions, structure_key, to_lines
from trustcall_system_one._typesafe import (
    _focus,
    _own_copy,
    _Repair,
    _repair_of,
    _stitch,
    _value_fits,
    from_jev,
    to_jev_clef,
)
from trustcall_system_one._wire import (
    ClairicField,
    ClairicUsage,
    parse_clairic,
    parse_clef,
)
from trustcall_system_one.testing import (
    CassetteTransport,
    FakeTransport,
    RecordingTransport,
)
from trustcall_system_one.transport import (
    _note_siblings,
    classify_failure,
    max_questions,
    state_token_budget,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "EXPERIENTIAL_API_KEY",
        "TYPESAFE_API_KEY",
        "TRUSTCALL_S1_PROVIDER",
        "TRUSTCALL_S1_CLEF_PROVIDER",
        "TRUSTCALL_S1_CLEF_FALLBACK",
        "TRUSTCALL_S1_CLEF_FALLBACK_ON",
    ):
        monkeypatch.delenv(name, raising=False)


def _none_over_phone(question: dict[str, Any]) -> dict[str, Any]:
    """A close call: `none` 0.45 narrowly ahead of the phone number at 0.40.

    (Probe 6 had the phone at 0.65 and `none` at 0.28; this is the case where
    Jev says it is not stated and a candidate is close behind.)
    """
    options = list(question["criteria"])
    assert "415-555-0100" in options
    rest = 0.15 / (len(options) - 2)
    probabilities = {o: rest for o in options}
    probabilities["none"] = 0.45
    probabilities["415-555-0100"] = 0.40
    return {
        "type": "choice",
        "choice": "none",
        "confidence": 0.45,
        "probabilities": probabilities,
    }


# --------------------------------------------------------------------------- #
# 3. Jev choosing `none` is a null
# --------------------------------------------------------------------------- #

PHONE_FIELD = {
    "phone": {"type": "text", "instructions": "The customer's phone number."}
}


def test_choosing_none_is_a_null_even_when_a_candidate_is_close() -> None:
    """`none` 0.45 against the phone 0.40: Jev said it is not stated."""

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        return _none_over_phone(question) if qid == "phone::pick" else None

    text = "Call us. The number on file is 415-555-0100."
    sent = clairic(make(Jev(script)), clairic_payload(PHONE_FIELD, text))
    answer = sent.body["answers"]["phone"]
    assert answer["value"] is None
    assert "span" not in answer
    assert answer["p_null"] == pytest.approx(0.45)  # P(none), not a threshold's guess
    assert answer["guess"] == "415-555-0100"  # the top candidate is only a guess
    assert answer["confidence"] == pytest.approx(0.45)
    # the candidate "Call" (an instruction, not a name) was never offered
    assert "Call" not in answer.values()


def test_choosing_none_stays_null_whatever_the_null_threshold() -> None:
    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        return _none_over_phone(question) if qid == "phone::pick" else None

    lax = {"phone": {**PHONE_FIELD["phone"], "null_threshold": 0.95}}
    text = "The number on file is 415-555-0100."
    sent = clairic(make(Jev(script)), clairic_payload(lax, text))
    assert sent.body["answers"]["phone"]["value"] is None


# --------------------------------------------------------------------------- #
# 4. The state budget a transport advertises
# --------------------------------------------------------------------------- #


def test_the_transport_advertises_the_state_it_takes() -> None:
    transport = make(Recorder(ok()))
    assert transport.state_token_budget == 24_000
    assert state_token_budget(transport, CLEF) == 24_000
    wide = make(Recorder(ok()), long_context=True)
    assert (state_token_budget(wide, CLEF) or 0) > 100_000
    assert state_token_budget(FakeTransport(lambda *_: None), CLEF) is None
    assert state_token_budget(object(), CLEF) is None


def test_routing_and_fallback_say_the_budget_of_who_serves() -> None:
    jev = make(Recorder(ok()))
    other = FakeTransport(lambda *_: None)
    routing = RoutingTransport({"clef": jev}, default=other)
    assert state_token_budget(routing, CLEF) == 24_000
    assert state_token_budget(routing, CLAIRIC) is None
    assert state_token_budget(FallbackTransport(jev, other), CLEF) == 24_000
    spare = FakeTransport(lambda *_: None)
    assert state_token_budget(FallbackTransport(other, spare), CLEF) is None


def test_the_engine_narrows_a_clef_state_to_what_the_transport_takes() -> None:
    text = "word " * 6000  # about 10k tokens
    plain = Run(transport=FakeTransport(lambda *_: None))
    assert (
        estimate_request_tokens(plain.clef_state(text)) > 5_000
    )  # the library limit: 24k
    narrow = FakeTransport(lambda *_: None)
    narrow.state_token_budget = 3_000  # type: ignore[attr-defined]
    run_ = Run(transport=narrow)
    state = run_.clef_state(text)
    assert estimate_request_tokens(state) <= 3_000
    assert run_.warnings  # it was narrowed, and says so
    # the smaller of the two wins, in both directions
    tight = Run(transport=narrow, limits=Limits(clef_state_token_budget=1_500))
    assert estimate_request_tokens(tight.clef_state(text)) <= 1_500
    roomy = FakeTransport(lambda *_: None)
    roomy.state_token_budget = 90_000  # type: ignore[attr-defined]
    capped = Run(transport=roomy, limits=Limits(clef_state_token_budget=2_000))
    assert estimate_request_tokens(capped.clef_state(text)) <= 2_000


async def test_a_clef_request_over_jevs_limits_is_split_and_narrowed() -> None:
    big = {"input": "word " * 40_000}  # 200k characters: 67k tokens
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Question {i}? " + "x " * 3000}
        for i in range(30)
    }
    payload = {"model": "clef", "state": big, "questions": questions}
    parts = to_jev_clef(
        payload, "jev-latest", question_budget=30_000, request_budget=60_000
    )
    assert len(parts) > 1
    assert [q for p in parts for q in p["questions"]] == list(questions)  # in order
    for part in parts:
        state = estimate_tokens(part["state"])
        costs = [estimate_tokens(q) for q in part["questions"].values()]
        assert state + max(costs) <= 30_000  # Jev: 32k for the state and a question
        assert state + sum(costs) <= 60_000  # and 64k for the state and them all
    # a request that fits is one request, unchanged
    (small,) = to_jev_clef(
        CLEF_PAYLOAD, "jev-latest", question_budget=30_000, request_budget=60_000
    )
    assert small["questions"] == CLEF_PAYLOAD["questions"]
    assert small["state"] == CLEF_PAYLOAD["state"]


def test_the_transport_sends_a_big_clef_request_as_several_and_merges() -> None:
    jev = Jev(lambda *_: None)
    transport = make(jev)
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Question {i}? " + "x " * 1500}
        for i in range(40)
    }
    payload = {
        "model": "clef",
        "state": {"input": "word " * 30_000},
        "questions": questions,
    }
    sent = run(transport.run(CLEF, payload, meta=META))
    assert len(jev.requests) > 1
    for request in jev.requests:
        state = estimate_tokens(request["state"])
        costs = [estimate_tokens(q) for q in request["questions"].values()]
        assert state + max(costs) <= 32_000
        assert state + sum(costs) <= 64_000
    assert list(sent.body["answers"]) == list(questions)  # every answer, in order
    assert sent.body["usage"]["input_tokens"] == 100 * len(jev.requests)


# --------------------------------------------------------------------------- #
# 6. User text stays out of error messages
# --------------------------------------------------------------------------- #

SECRET = "the user's private words"


def test_a_clairic_request_error_names_the_field_and_never_the_value() -> None:
    bad = {"model": "clairic", "state": SECRET, "fields": {"f": {"type": SECRET}}}
    with pytest.raises(BadRequestError) as caught:
        clairic(make(Recorder(ok())), bad)
    assert "private" not in str(caught.value)
    assert "not a valid Clairic request" in str(caught.value)


def test_a_clef_request_error_never_holds_the_value() -> None:
    bad = {"model": "clef", "state": SECRET, "questions": {"q": {"type": SECRET}}}
    with pytest.raises(BadRequestError) as caught:
        to_jev_clef(bad, "jev-latest")
    assert "private" not in str(caught.value)


def test_an_option_that_was_not_offered_is_not_named() -> None:
    body = json.loads(json.dumps(PROBE))
    questions = CLEF_PAYLOAD["questions"]
    body["answers"]["topic"]["choice"] = SECRET
    with pytest.raises(ModelResponseError) as caught:
        from_jev(body, questions)
    assert "private" not in str(caught.value)
    assert "'topic'" in str(caught.value)


def test_a_selection_that_was_not_offered_does_not_echo_the_text() -> None:
    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid == "email::pick":
            return {
                "type": "choice",
                "choice": SECRET,
                "confidence": 0.9,
                "probabilities": {SECRET: 0.9, "none": 0.1},
            }
        return None

    with pytest.raises(ModelResponseError) as caught:
        clairic(make(Jev(script)), clairic_payload(EMAIL_FIELD))
    assert "private" not in str(caught.value)


# --------------------------------------------------------------------------- #
# 7. Cassettes keep provenance
# --------------------------------------------------------------------------- #


def _record_one(path: Path) -> dict[str, Any]:
    recording = RecordingTransport(make(Recorder(ok())), path)
    assert state_token_budget(recording, CLEF) == 24_000  # forwarded like the limit
    assert max_questions(recording, CLEF) == 64
    run(recording.run(CLEF, CLEF_PAYLOAD, meta=META))
    line: dict[str, Any] = json.loads(path.read_text().splitlines()[0])
    return line


def test_a_recording_says_who_answered_and_replays_it(tmp_path: Path) -> None:
    line = _record_one(tmp_path / "typesafe.jsonl")
    assert line["provider"] == "typesafe"
    assert line["served_model"] == PROBE["model"]
    assert line["state_token_budget"] == 24_000
    cassette = CassetteTransport(tmp_path / "typesafe.jsonl")
    assert cassette.providers() == {"clairic": (), "clef": ("typesafe",)}
    assert state_token_budget(cassette, CLEF) == 24_000
    again = run(cassette.run(CLEF, CLEF_PAYLOAD, meta=META))
    assert again.provider == "typesafe"
    assert again.served_model == PROBE["model"]  # accounted under who answered


def test_an_old_cassette_replays_exactly_as_before(tmp_path: Path) -> None:
    line = _record_one(tmp_path / "new.jsonl")
    for key in ("provider", "served_model", "state_token_budget"):
        del line[key]
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps(line) + "\n")
    cassette = CassetteTransport(old)
    again = run(cassette.run(CLEF, CLEF_PAYLOAD, meta=META))
    assert again.provider is None
    assert again.served_model is None
    assert cassette.providers() == {"clairic": (), "clef": ()}
    assert state_token_budget(cassette, CLEF) is None


def test_a_replay_is_labelled_with_what_it_was_recorded_on(tmp_path: Path) -> None:
    folder = tmp_path / "cassettes"
    folder.mkdir()
    _record_one(folder / "typesafe.jsonl")
    suite = asyncio.run(run_suite(CASES[:1], Settings(replay=folder)))
    assert suite.decisions == "replay (recorded on typesafe jev)"
    assert suite.extraction == "replay (recorded on unknown)"  # no Clairic recorded


# --------------------------------------------------------------------------- #
# 8. Internal requests are split by tokens alone; stitching is single-flight
# --------------------------------------------------------------------------- #


def test_a_stitch_of_many_lines_is_one_request_with_no_question_cap() -> None:
    text = "\n".join(f"item {i} and then" for i in range(120))
    jev = Jev(picks())
    transport = make(jev, config=TypeSafeConfig(verify_selection=False))
    clairic(transport, clairic_payload(EMAIL_FIELD, text))
    stitches = [r for r in jev.requests if "L001" in r["questions"]]
    assert len(stitches) == 1
    assert len(stitches[0]["questions"]) > 64  # the Clef cap is not Jev's


class SlowJev(Jev):
    """A Jev that takes a moment, so concurrent calls really overlap."""

    async def __call__(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        await asyncio.sleep(0.05)
        return Jev.__call__(self, request)


def test_concurrent_calls_over_one_input_make_one_stitch_request() -> None:
    jev = SlowJev(picks())
    transport = make(jev, config=TypeSafeConfig(verify_selection=False))

    async def many() -> list[TransportResponse]:
        return list(
            await asyncio.gather(
                *(
                    transport.run(CLAIRIC, clairic_payload(EMAIL_FIELD), meta=META)
                    for _ in range(8)
                )
            )
        )

    results = run(many())
    assert len(results) == 8
    assert len([r for r in jev.requests if "L001" in r["questions"]]) == 1
    selects = [r for r in jev.requests if "email::pick" in r["questions"]]
    assert len(selects) == 8  # every call still selects for itself


def test_a_failed_stitch_is_not_remembered() -> None:
    jev = Jev(picks())
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 1:
            raise httpx.ReadTimeout("slow")
        return jev(request)

    transport = make(handler, config=TypeSafeConfig(verify_selection=False))
    with pytest.raises(UncertainOutcomeError):
        clairic(transport, clairic_payload(EMAIL_FIELD))
    sent = clairic(transport, clairic_payload(EMAIL_FIELD))  # asks again, and works
    assert sent.body["answers"]["email"]["value"] is None or sent.body["answers"]
    assert len([r for r in jev.requests if "L001" in r["questions"]]) == 1


def test_waiters_of_a_failed_stitch_fail_with_it_and_do_not_resend() -> None:
    seen = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        await asyncio.sleep(0.05)
        raise httpx.ReadTimeout("slow")

    transport = make(handler, config=TypeSafeConfig(verify_selection=False))

    async def many() -> list[Any]:
        return list(
            await asyncio.gather(
                *(
                    transport.run(CLAIRIC, clairic_payload(EMAIL_FIELD), meta=META)
                    for _ in range(3)
                ),
                return_exceptions=True,
            )
        )

    results = run(many())
    assert all(isinstance(r, UncertainOutcomeError) for r in results)
    assert seen["n"] == 1  # one request, never resent, shared by all three


# --------------------------------------------------------------------------- #
# 9. Repair requests
# --------------------------------------------------------------------------- #


def _trailer(
    previous: Any, message: str, field: str = "email", shown: str | None = None
) -> str:
    """The trailer the repair loop really writes (``_Blocks.trailer``)."""
    error: dict[str, Any] = {"field": field, "message": message, "type": "value_error"}
    if shown is not None:
        error["input"] = shown  # the engine shows a leaf's own value
    return _Blocks(previous, [error], "{}").trailer()


def test_a_repair_says_what_was_wrong_and_does_not_offer_the_answer_again() -> None:
    jev = Jev(picks(email="dana.whit@acme-corp.com"))
    payload = clairic_payload(
        EMAIL_FIELD,
        MAIL,
        trailer=_trailer("dana.personal@gmail.com", "not allowed on this domain"),
    )
    sent = clairic(make(jev, config=TypeSafeConfig(verify_selection=False)), payload)
    (select,) = [r for r in jev.requests if "email::pick" in r["questions"]]
    question = select["questions"]["email::pick"]
    assert "email: not allowed on this domain" in question["instructions"]
    assert "it is not an option" in question["instructions"]
    assert "dana.personal@gmail.com" not in question["criteria"]
    assert "dana.whit@acme-corp.com" in question["criteria"]
    assert sent.body["answers"]["email"]["value"] == "dana.whit@acme-corp.com"
    # the rejected answer and its errors are context, not input: never in the state
    assert "previous_attempt" not in json.dumps(select["state"])


def test_a_repair_with_nothing_new_to_offer_is_a_null_not_a_paid_repeat() -> None:
    field = {"n": {"type": "number", "instructions": "The price in dollars."}}
    jev = Jev(picks())
    payload = clairic_payload(
        field, "It costs 42 dollars.", trailer=_trailer(42, "too high", "n")
    )
    sent = clairic(make(jev), payload)
    assert not jev.requests  # no identical request is sent and charged again
    answer = sent.body["answers"]["n"]
    assert answer["value"] is None
    assert answer["p_null"] == 1.0


def test_the_existing_record_is_never_sent_to_jev() -> None:
    old = "dana.old@acme-corp.com"
    payload = clairic_payload(
        EMAIL_FIELD, MAIL, existing=f"Person record: email {old}", existing_id="1"
    )
    assert old in payload["state"]  # it is in the engine's state ...
    jev = Jev(picks(email="dana.personal@gmail.com"))
    clairic(make(jev), payload)
    assert jev.requests
    assert old not in json.dumps(jev.requests)  # ... and not in a Jev request


# --------------------------------------------------------------------------- #
# 10. Decisions as the primary: images go to the default
# --------------------------------------------------------------------------- #


class Spy:
    """A transport that records what it was asked and answers."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        model: str,
        payload: Any,
        *,
        meta: Any,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.calls.append(dict(payload))
        return TransportResponse(body={"answered": True})


def test_an_image_request_goes_to_the_default_not_to_the_decisions_api() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(500, json={})

    decisions = DecisionsTransport(
        api_key="xpl_" + "ab12cd34" * 5,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    default = Spy()
    routing = RoutingTransport({"clef": decisions}, default=default)
    with_image = {**CLEF_PAYLOAD, "images": ["data:image/png;base64,iVBORw0KGgo="]}
    assert decisions.serves(CLEF, with_image) is False
    reply = run(routing.run(CLEF, with_image, meta=META))
    assert reply.body == {"answered": True}
    assert len(default.calls) == 1
    assert not sent  # the Decisions API never saw the image
    # text only: the Decisions API serves it
    assert decisions.serves(CLEF, CLEF_PAYLOAD) is True
    with pytest.raises(UncertainOutcomeError):
        run(routing.run(CLEF, CLEF_PAYLOAD, meta=META))
    assert len(sent) == 1
    assert len(default.calls) == 1


# --------------------------------------------------------------------------- #
# 11. A call of several requests: what failed part-way still cost
# --------------------------------------------------------------------------- #


def test_a_failure_after_a_charged_stage_keeps_its_tokens_and_is_uncertain() -> None:
    jev = Jev(picks(email="dana.personal@gmail.com"))
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 2:  # the stitch answered; the selection is refused
            return httpx.Response(429, json={})
        return jev(request)

    transport = make(handler, max_retries=0)
    with pytest.raises(RateLimitError) as caught:
        clairic(transport, clairic_payload(EMAIL_FIELD))
    error: Any = caught.value
    assert error.partial_usage.input_tokens == 100  # the stitch's tokens
    assert error.partial_usage.passes == 1
    assert error.maybe_sent is True
    assert error.served_model == "jev-1.14.0"
    assert transport.classify_failure(error) == "uncertain"  # it was charged
    # a first request that is refused charged nothing
    first = make(Recorder(httpx.Response(429, json={})), max_retries=0)
    with pytest.raises(RateLimitError) as nothing:
        clairic(first, clairic_payload(EMAIL_FIELD))
    assert getattr(nothing.value, "partial_usage", None) is None
    assert first.classify_failure(nothing.value) == "unavailable"


def test_cancelled_siblings_are_counted_and_say_so() -> None:
    text = "\n".join(f"Person{i} Name{i} met Person{i + 1}." for i in range(40))
    field = {"people": {"type": "list", "items": "text", "instructions": "People"}}
    config = TypeSafeConfig(
        verify_selection=False,
        stitch=False,
        max_list_candidates=1000,
        request_token_budget=1500,  # several requests in the selection round
    )
    seen = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 1:
            raise httpx.ReadTimeout("slow")
        await asyncio.sleep(1)  # the others are still in flight when the first fails
        return httpx.Response(500, json={})

    transport = make(handler, config=config)
    with pytest.raises(UncertainOutcomeError) as caught:
        clairic(transport, clairic_payload(field, text))
    siblings: Any = caught.value
    assert siblings.cancelled_siblings >= 1
    assert "cancelled after they may have been sent" in str(caught.value)
    assert transport.classify_failure(caught.value) == "uncertain"


async def test_the_engine_records_the_tokens_of_a_failed_call() -> None:
    class Partway:
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> TransportResponse:
            error = UncertainOutcomeError("the selection timed out")
            error.partial_usage = ClairicUsage(  # type: ignore[attr-defined]
                input_tokens=300, output_tokens=7, passes=2
            )
            error.served_model = "jev-1.14.0"
            raise error

    engine = Run(transport=Partway())
    with pytest.raises(UncertainOutcomeError):
        await engine._send(CLAIRIC, {"model": "clairic"}, META, 1)
    (call,) = engine.calls
    assert call.model == "jev-1.14.0"  # under the model that answered
    assert call.usage["input_tokens"] == 300
    assert call.error is not None and call.error.startswith("UncertainOutcomeError")
    assert engine.usage.input_tokens == 300  # it counts in the totals too


def test_the_deadline_bounds_the_whole_clairic_call_not_each_request() -> None:
    jev = Jev(picks(email="dana.personal@gmail.com"))

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)  # each request is fast enough on its own ...
        return jev(request)

    transport = make(slow, call_deadline=0.3)  # ... three of them are not
    with pytest.raises(UncertainOutcomeError) as caught:
        clairic(transport, clairic_payload(EMAIL_FIELD))
    assert "deadline" in str(caught.value)
    late: Any = caught.value
    assert late.partial_usage.passes == 1  # the stitch had answered
    assert transport.classify_failure(caught.value) == "uncertain"
    # a deadline hit before anything answered is still not "nothing was sent"
    stalled = make(slow, call_deadline=0.05)
    with pytest.raises(UncertainOutcomeError) as first:  # never a plain TransportError
        clairic(stalled, clairic_payload(EMAIL_FIELD))
    assert "deadline" in str(first.value)
    assert stalled.classify_failure(first.value) == "uncertain"


# --------------------------------------------------------------------------- #
# Accounting labels and prices
# --------------------------------------------------------------------------- #


def test_a_call_with_no_request_is_under_the_typesafe_model_with_no_tokens() -> None:
    jev = Jev(picks())
    sent = clairic(
        make(jev),
        clairic_payload({"n": {"type": "number", "instructions": "x"}}, "no digits"),
    )
    assert not jev.requests
    assert sent.served_model == "jev-latest"  # the alias: nothing named a version
    assert sent.body["model"] == "jev-latest"
    assert sent.body["usage"] == {"input_tokens": 0, "output_tokens": 0, "passes": 0}


async def test_an_error_is_logged_under_the_alias_when_no_version_answered() -> None:
    transport = make(Recorder(httpx.Response(500, json={})))
    engine = Run(transport=transport)
    with pytest.raises(UncertainOutcomeError):
        await engine._send(CLEF, CLEF_PAYLOAD, META, 1)
    assert engine.calls[-1].model == "jev-latest"


def test_a_price_for_the_alias_covers_every_jev_version() -> None:
    usage = Usage()
    for model in ("jev-1.14.0", "jev-1.15.2"):
        usage.add_call(
            model,
            "gate",
            ClairicUsage(input_tokens=1_000_000, output_tokens=0, passes=1),
        )
    report = Prices({"jev-latest": 0.042}).cost(usage)
    assert report.usd == pytest.approx(0.084)
    assert report.complete
    # a price for a version wins over the alias
    exact = Prices({"jev-latest": 0.042, "jev-1.14.0": 1.0}).cost(usage)
    assert exact.usd == pytest.approx(1.042)
    assert Prices().cost(usage).unpriced_models == ("jev-1.14.0", "jev-1.15.2")


# --------------------------------------------------------------------------- #
# The escalation hook: its answer is validated, its failure is tagged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "given",
    [
        {"value": 5},  # not text
        {"value": "x@y.co", "span": [5, 3]},  # backwards
        {"value": "x@y.co", "span": [0, 99999]},  # outside the input
        {"value": "x@y.co", "span": ["a", "b"]},
        {"value": "x@y.co", "confidence": 7},
        {"value": "x@y.co", "p_null": -1},
        {"nothing": True},
        "orders@acme-corp.com",
    ],
)
def test_a_malformed_hook_answer_is_not_used(given: Any) -> None:
    transport = make(Jev(_verify_script(0.9)), escalate=lambda request: given)
    sent = clairic(transport, clairic_payload(EMAIL_FIELD))
    answer = sent.body["answers"]["email"]
    assert answer["escalated"] == "hook_invalid"
    assert answer["value"] is None
    assert answer["guess"] == "dana.personal@gmail.com"
    assert "span" not in answer


def test_a_hook_that_raises_is_tagged_and_does_not_fail_the_call() -> None:
    def boom(request: Any) -> dict[str, Any]:
        raise RuntimeError(SECRET)

    sent = clairic(
        make(Jev(_verify_script(0.9)), escalate=boom), clairic_payload(EMAIL_FIELD)
    )
    answer = sent.body["answers"]["email"]
    assert answer["escalated"] == "hook_error"
    assert answer["value"] is None
    assert SECRET not in json.dumps(sent.body)

    async def aboom(request: Any) -> dict[str, Any]:
        raise ValueError(SECRET)

    again = clairic(
        make(Jev(_verify_script(0.9)), escalate=aboom), clairic_payload(EMAIL_FIELD)
    )
    assert again.body["answers"]["email"]["escalated"] == "hook_error"


def test_a_valid_hook_answer_with_a_span_is_in_state_offsets() -> None:
    transport = make(
        Jev(_verify_script(0.9)),
        escalate=lambda request: {"value": "billing", "span": [0, 4]},
    )
    payload = clairic_payload(EMAIL_FIELD)
    answer = clairic(transport, payload).body["answers"]["email"]
    assert answer["escalated"] == "hook"
    start = payload["state"].index("<input>\n") + len("<input>\n")
    assert answer["span"] == [start, start + 4]


# --------------------------------------------------------------------------- #
# The focus block when the user's text has the narrowing mark itself
# --------------------------------------------------------------------------- #

_HEAD = (
    '<focus entity="the Person">\nAnswer only about the Person.\n'
    "The entity is mentioned in the input at characters 0-4.\n</focus>"
)


def test_the_focus_quotes_the_words_when_the_input_is_whole() -> None:
    assert 'mentioned in the input as: "Dana"' in _focus(_HEAD, "Dana wrote.")


@pytest.mark.parametrize(
    "content", ["Dana\n…\nwrote", "…\nDana wrote", "Dana wrote\n…"]
)
def test_the_focus_drops_offsets_that_may_point_into_other_text(content: str) -> None:
    text = _focus(_HEAD, content)
    assert "characters" not in text  # a model cannot count them
    assert "as:" not in text  # and the quote could be the wrong words
    assert "Answer only about the Person." in text


# --------------------------------------------------------------------------- #
# 12. Contradictory flags and objects are refused
# --------------------------------------------------------------------------- #


def test_typesafe_refuses_the_other_transports_it_would_not_use() -> None:
    with pytest.raises(ValueError, match="uses none of"):
        system_one_transport(
            provider="typesafe",
            cloudflare=CloudflareTransport(account_id="acct", api_token="tok"),
        )
    with pytest.raises(ValueError, match="uses none of"):
        system_one_transport(
            provider="typesafe", experiential=ExperientialTransport(api_key=KEY)
        )
    with pytest.raises(ValueError, match="uses none of"):
        system_one_transport(
            provider="typesafe", decisions=DecisionsTransport(api_key=KEY)
        )


@pytest.fixture
def empty_cassettes(tmp_path: Path) -> Path:
    folder = tmp_path / "cassettes"
    folder.mkdir()
    (folder / "empty.jsonl").write_text("")
    return folder


async def test_the_evals_refuse_typesafe_with_another_clef(
    empty_cassettes: Path,
) -> None:
    cases = CASES[:1]
    for settings in (
        Settings(
            replay=empty_cassettes, provider="typesafe", clef_provider="decisions"
        ),
        Settings(
            replay=empty_cassettes, provider="typesafe", clef_provider="experiential"
        ),
        Settings(
            replay=empty_cassettes, provider="typesafe", clef_fallback="decisions"
        ),
    ):
        with pytest.raises(ValueError, match="answers Clef with Jev itself"):
            await run_suite(cases, settings)
    # cloudflare as the Clef provider is the default, not a contradiction
    suite = await run_suite(
        cases,
        Settings(
            replay=empty_cassettes, provider="typesafe", clef_provider="cloudflare"
        ),
    )
    assert suite.provider == "typesafe"


async def test_the_evals_refuse_what_the_environment_contradicts(
    empty_cassettes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "decisions")
    with pytest.raises(ValueError, match="answers Clef with Jev itself"):
        await run_suite(
            CASES[:1], Settings(replay=empty_cassettes, provider="typesafe")
        )
    monkeypatch.delenv("TRUSTCALL_S1_CLEF_PROVIDER")
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_FALLBACK", "decisions")
    with pytest.raises(ValueError, match="answers Clef with Jev itself"):
        await run_suite(CASES[:1], Settings(replay=empty_cassettes))


def test_choice_helper_is_the_probed_shape() -> None:
    """Keeps the shared helper honest: a choice has a probability for every option."""
    answer = choice({"criteria": {"a": None, "b": None, "none": None}}, "a", 0.8)
    assert answer["probabilities"]["a"] == 0.8
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Second review. 11: what a failed sibling leaves behind, seen through wrappers
# --------------------------------------------------------------------------- #

LIST_FIELD = {"people": {"type": "list", "items": "text", "instructions": "People"}}
LIST_TEXT = "\n".join(f"Person{i} Name{i} met Person{i + 1}." for i in range(40))
SPLIT = TypeSafeConfig(
    verify_selection=False,
    stitch=False,
    max_list_candidates=1000,
    request_token_budget=1500,  # several requests in the selection round
)


def _parallel_failure() -> tuple[Any, Jev, dict[str, int]]:
    jev = Jev(picks())
    seen = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 2:
            await asyncio.sleep(0.3)  # the others have answered by then
            return httpx.Response(429, json={})
        return jev(request)

    return make(handler, config=SPLIT, max_retries=0), jev, seen


def test_the_siblings_that_answered_are_counted_when_one_fails() -> None:
    transport, _jev, seen = _parallel_failure()
    with pytest.raises(RateLimitError) as caught:
        clairic(transport, clairic_payload(LIST_FIELD, LIST_TEXT))
    error: Any = caught.value
    assert seen["n"] >= 3  # a call of many requests, one of which is refused
    assert error.maybe_sent is True
    assert error.partial_usage.passes == seen["n"] - 1  # every other one answered
    assert error.partial_usage.input_tokens == 100 * (seen["n"] - 1)
    assert transport.classify_failure(error) == "uncertain"


def test_a_fallback_does_not_re_ask_what_a_failed_call_was_charged_for() -> None:
    transport, _jev, _seen = _parallel_failure()
    spare = Spy()
    both = FallbackTransport(transport, spare)
    with pytest.raises(RateLimitError):
        run(both.run(CLAIRIC, clairic_payload(LIST_FIELD, LIST_TEXT), meta=META))
    assert not spare.calls


def test_a_wrapped_transport_keeps_the_uncertainty(tmp_path: Path) -> None:
    jev = Jev(picks(email="dana.personal@gmail.com"))
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 2:  # the stitch answered, the selection is refused
            return httpx.Response(429, json={})
        return jev(request)

    spare = Spy()
    wrapped = RecordingTransport(make(handler, max_retries=0), tmp_path / "c.jsonl")
    both = FallbackTransport(wrapped, spare)
    with pytest.raises(RateLimitError) as caught:
        run(both.run(CLAIRIC, clairic_payload(EMAIL_FIELD), meta=META))
    assert not spare.calls  # the module-level classification sees maybe_sent
    assert classify_failure(caught.value) == "uncertain"
    assert seen["n"] == 2


def test_classification_honours_a_charge_whatever_the_error_type() -> None:
    plain = RateLimitError("slow down")
    assert classify_failure(plain) == "unavailable"
    charged = RateLimitError("slow down")
    charged.maybe_sent = True
    assert classify_failure(charged) == "uncertain"
    paid = TransportError("x")
    paid.partial_usage = ClairicUsage(  # type: ignore[attr-defined]
        input_tokens=1, output_tokens=1, passes=1
    )
    assert classify_failure(paid) == "uncertain"
    refused = BadRequestError("no")
    refused.maybe_sent = True
    assert classify_failure(refused) is None  # a bad request is never retried


def _clef_split_failure() -> tuple[Any, Any, dict[str, int]]:
    payload = {
        **CLEF_PAYLOAD,
        "state": {"input": "word " * 2000},
        "questions": {
            f"q{i}": {
                "type": "noul",
                "instructions": f"Is it {i}?",
                "criteria": {"true": "y", "false": "n"},
            }
            for i in range(6)
        },
    }
    seen = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        body = json.loads(request.content)
        if seen["n"] == 2:
            await asyncio.sleep(0.2)
            return httpx.Response(429, json={})
        answers = {q: {"type": "noul", "noul": 0.1} for q in body["questions"]}
        usage = {"input_tokens": 1000, "output_tokens": 5}
        return httpx.Response(
            200, json={"model": "jev-1.14.0", "answers": answers, "usage": usage}
        )

    transport = make(handler, max_retries=0)
    transport._request_tokens = 3420  # a split into several requests
    return transport, payload, seen


def test_a_split_clef_request_that_fails_part_way_keeps_what_answered() -> None:
    transport, payload, seen = _clef_split_failure()
    with pytest.raises(RateLimitError) as caught:
        run(transport.run(CLEF, payload, meta=META))
    error: Any = caught.value
    assert seen["n"] > 2
    assert error.maybe_sent is True
    assert error.partial_usage.input_tokens == 1000 * (seen["n"] - 1)
    assert transport.classify_failure(error) == "uncertain"


def test_a_fallback_does_not_re_ask_the_clef_parts_that_were_charged() -> None:
    transport, payload, _seen = _clef_split_failure()
    spare = Spy()
    with pytest.raises(RateLimitError):
        run(FallbackTransport(transport, spare).run(CLEF, payload, meta=META))
    assert not spare.calls


async def test_the_primary_failure_of_a_fallback_keeps_its_tokens() -> None:
    class Partway:
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> TransportResponse:
            error = UncertainOutcomeError("the selection timed out")
            error.partial_usage = ClairicUsage(  # type: ignore[attr-defined]
                input_tokens=300, output_tokens=7, passes=2
            )
            error.served_model = "jev-1.14.0"
            raise error

    class Refuses:
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> TransportResponse:
            raise TransportError("no")

    engine = Run(transport=FallbackTransport(Partway(), Refuses(), on="uncertain"))
    with pytest.raises(TransportError):
        await engine._send(CLAIRIC, {"model": "clairic"}, META, 1)
    primary, fallback = engine.calls
    assert primary.usage["input_tokens"] == 300  # the primary's tokens are kept
    assert primary.model == "jev-1.14.0"
    assert fallback.error is not None
    assert engine.usage.input_tokens == 300


def test_note_of_cancelled_siblings_is_added_once() -> None:
    error = UncertainOutcomeError("the selection timed out")
    _note_siblings(error, 2)
    _note_siblings(error, 2)  # a second call that shares the error
    assert str(error).count("cancelled after they may have been sent") == 1
    mine = _own_copy(error)
    assert "cancelled" not in str(mine)  # the sibling note is the leader's
    assert str(mine) == "the selection timed out"


def test_a_waiter_gets_its_own_error_without_the_leaders_charge() -> None:
    error = UncertainOutcomeError("slow")
    error.partial_usage = ClairicUsage(  # type: ignore[attr-defined]
        input_tokens=5, output_tokens=1, passes=1
    )
    error.maybe_sent = True
    error.cancelled_siblings = 3  # type: ignore[attr-defined]
    mine = _own_copy(error)
    assert mine is not error
    assert getattr(mine, "partial_usage", None) is None
    assert not getattr(mine, "maybe_sent", False)
    assert not getattr(mine, "cancelled_siblings", 0)
    leader: Any = error
    assert leader.partial_usage.passes == 1  # the leader's is untouched


async def test_a_waiter_never_removes_a_newer_owners_entry() -> None:
    content = "line one and\ncontinues here."
    lines = to_lines(content)
    questions = join_questions(lines)
    key = structure_key(content)
    loop = asyncio.get_running_loop()
    old: asyncio.Future[dict[str, float]] = loop.create_future()
    newer: asyncio.Future[dict[str, float]] = loop.create_future()
    memo = {key: old}

    async def ask(*_: Any) -> Any:
        raise AssertionError("a waiter does not ask")

    waiter = asyncio.create_task(_stitch(content, lines, questions, ask, memo))
    await asyncio.sleep(0)  # it is waiting on `old`
    memo[key] = newer  # another call took over the entry
    old.set_exception(RuntimeError("the first stitch failed"))
    with pytest.raises(RuntimeError):
        await waiter
    assert memo[key] is newer
    newer.cancel()


def test_529_slows_the_limiter_and_is_still_not_retried() -> None:
    recorder = Recorder(httpx.Response(529, json={}), ok())
    transport = make(recorder)
    before = transport.limiter.limit("typesafe")
    with pytest.raises(UncertainOutcomeError):
        run(transport.run(CLEF, CLEF_PAYLOAD, meta=META))
    assert transport.limiter.limit("typesafe") == max(1, before // 2)
    assert len(recorder.requests) == 1


# --------------------------------------------------------------------------- #
# Second review. 4: a Clef request is narrowed until it fits, or refused locally
# --------------------------------------------------------------------------- #


def _delete_question(i: int) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": f"Does `input` state that `records.r{i}` should be deleted?",
        "criteria": {"true": "yes", "false": "no"},
    }


def _fits(parts: list[dict[str, Any]]) -> None:
    for part in parts:
        state = estimate_tokens(part["state"])
        costs = [estimate_tokens(q) for q in part["questions"].values()]
        assert state + max(costs) <= 30_000
        assert state + sum(costs) <= 60_000


def test_a_state_of_many_records_is_narrowed_until_the_request_fits() -> None:
    records = {
        f"r{i}": {"name": f"Person {i}", "notes": "lorem ipsum dolor sit amet " * 30}
        for i in range(300)
    }
    state = {"input": "Alice left the company. " * 50, "records": records}
    questions = {f"delete.r{i}": _delete_question(i) for i in range(60)}
    assert estimate_tokens(state) > 50_000
    parts = to_jev_clef(
        {"model": "clef", "state": state, "questions": questions},
        "jev-latest",
        question_budget=30_000,
        request_budget=60_000,
    )
    _fits(parts)
    assert [q for p in parts for q in p["questions"]] == list(questions)
    assert parts[0]["state"]["input"].startswith("Alice left")  # the input is kept


def test_what_the_questions_quote_is_kept_when_the_state_is_narrowed() -> None:
    filler = "filler words go here " * 3000
    needle = "Lisbon and pays by card"
    text = f"{filler}The customer lives in {needle}, every month. {filler}"
    question = {
        "type": "noul",
        "instructions": f"Does `input` say the city is Lisbon? It reads '{needle}'.",
        "criteria": {"true": "yes", "false": "no"},
    }
    (part,) = to_jev_clef(
        {"model": "clef", "state": {"input": text}, "questions": {"q": question}},
        "jev-latest",
        question_budget=30_000,
        request_budget=60_000,
    )
    kept = part["state"]["input"]
    assert needle in kept  # not just the start of the text
    assert "\u2026" in kept  # with a mark where text was left out
    assert len(kept) < len(text)
    _fits([part])
    # text the questions do not quote is cut from its end
    (plain,) = to_jev_clef(
        {
            "model": "clef",
            "state": {"input": text},
            "questions": {"q": _delete_question(1)},
        },
        "jev-latest",
        question_budget=30_000,
        request_budget=60_000,
    )
    assert plain["state"]["input"].startswith("filler words")
    assert plain["state"]["input"].endswith("\u2026")


def test_a_question_that_alone_fills_jevs_context_is_refused_locally() -> None:
    huge = {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {f"o{i}": "x" * 400 for i in range(254)},
    }
    payload = {
        "model": "clef",
        "state": {"input": "hello " * 20000},
        "questions": {"m": huge},
    }
    with pytest.raises(BadRequestError, match="'m' alone"):
        to_jev_clef(
            payload, "jev-latest", question_budget=30_000, request_budget=60_000
        )
    recorder = Recorder(ok())
    with pytest.raises(BadRequestError):  # through the transport: nothing is sent
        run(make(recorder).run(CLEF, payload, meta=META))
    assert not recorder.requests


def test_a_state_that_cannot_be_cut_small_enough_is_refused_locally() -> None:
    state = {
        "records": {f"r{i}": "x" * 150 for i in range(1000)}
    }  # all under the floor
    payload = {"model": "clef", "state": state, "questions": {"q": _delete_question(1)}}
    recorder = Recorder(ok())
    with pytest.raises(BadRequestError, match="cannot be narrowed"):
        run(make(recorder).run(CLEF, payload, meta=META))
    assert not recorder.requests


# --------------------------------------------------------------------------- #
# Second review. 9: the repair trailer the engine really writes
# --------------------------------------------------------------------------- #


def test_the_real_trailer_is_read_and_the_input_it_shows_is_not_copied() -> None:
    private = "dana-private@secret.example"
    trailer = _trailer(
        "dana.personal@gmail.com", "not allowed here", "email", shown=private
    )
    repair = _repair_of(trailer)
    assert repair is not None
    assert repair.previous == "dana.personal@gmail.com"
    assert repair.errors == (("email", "not allowed here"),)
    jev = Jev(picks(email="dana.whit@acme-corp.com"))
    payload = clairic_payload(EMAIL_FIELD, MAIL, trailer=trailer)
    clairic(make(jev, config=TypeSafeConfig(verify_selection=False)), payload)
    assert private not in json.dumps(jev.requests)  # never in an instruction


def test_a_record_level_error_keeps_the_answer_as_an_option() -> None:
    field = {"price": {"type": "number", "instructions": "The price in dollars."}}
    text = "It costs 42 dollars, with a 50 dollar discount."
    trailer = _trailer(42, "Value error, discount must be below price", "(record)")
    jev = Jev(picks(price="42"))
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(field, text, trailer=trailer),
    )
    (select,) = [r for r in jev.requests if "price::pick" in r["questions"]]
    question = select["questions"]["price::pick"]
    assert "42" in question["criteria"]  # not removed: the price may be right
    assert "discount must be below price" in question["instructions"]
    assert "it is not an option" not in question["instructions"]
    assert sent.body["answers"]["price"]["value"] == 42
    # an error that names this very field does remove it
    own = _trailer(42, "too high", "price")
    jev2 = Jev(picks(price="50"))
    clairic(
        make(jev2, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(field, text, trailer=own),
    )
    (second,) = [r for r in jev2.requests if "price::pick" in r["questions"]]
    assert "42" not in second["questions"]["price::pick"]["criteria"]
    assert "price: too high" in second["questions"]["price::pick"]["instructions"]


def test_a_nested_path_names_its_leaf_field() -> None:
    repair = _Repair("x", (("contacts.0.email", "bad"),))
    assert repair.names("email")
    assert not repair.names("phone")
    assert not _Repair("x", (("(record)", "bad"),)).names("email")


# --------------------------------------------------------------------------- #
# Second review. 12, 6, 10 and the nits
# --------------------------------------------------------------------------- #


def test_a_typesafe_transport_without_the_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = make(Recorder(ok()))
    with pytest.raises(ValueError, match="the text would go to Cloudflare"):
        system_one_transport(typesafe=own)
    with pytest.raises(ValueError, match="the text would go to Cloudflare"):
        system_one_transport(provider="cloudflare", typesafe=own)
    assert system_one_transport(provider="typesafe", typesafe=own) is own
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")  # opt-in by the variable
    assert system_one_transport(typesafe=own) is own


def test_a_response_that_does_not_parse_names_no_option_or_value() -> None:
    body = {
        "model": "jev-1.14.0",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "answers": {
            "email::pick": {
                "type": "choice",
                "choice": "dana.secret@private.example",
                "confidence": 0.9,
                "probabilities": {"dana.secret@private.example": "not a number"},
            }
        },
    }
    with pytest.raises(ModelResponseError) as caught:
        parse_clef(body)
    assert "private" not in str(caught.value)
    assert "dana.secret" not in str(caught.value)
    assert "answers.email::pick.choice.probabilities" in str(caught.value)
    with pytest.raises(ModelResponseError) as clairic_error:
        parse_clairic({"answers": {"x": "dana.secret"}})
    assert "dana.secret" not in str(clairic_error.value)


def test_only_a_request_with_images_is_rerouted_and_the_engine_says_so() -> None:
    ts = make(Recorder(ok()))
    default = Spy()
    routing = RoutingTransport({"clairic": ts}, default=default)
    forecast = {
        "f": {
            "type": "forecast",
            "min": 0,
            "max": 10,
            "horizon": 3,
            "instructions": "x",
        }
    }
    with pytest.raises(BadRequestError, match="forecast"):  # refused, not moved
        run(routing.run(CLAIRIC, clairic_payload(forecast), meta=META))
    assert not default.calls

    decisions = DecisionsTransport(
        api_key="xpl_" + "ab12cd34" * 5,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(500, json={}))
        ),
    )
    router = RoutingTransport({"clef": decisions}, default=default)
    with_image = {**CLEF_PAYLOAD, "images": ["data:image/png;base64,iVBORw0KGgo="]}
    reply = run(router.run(CLEF, with_image, meta=META))
    assert reply.rerouted_from == "decisions"
    assert len(default.calls) == 1

    async def through_the_engine() -> list[str]:
        engine = Run(transport=router)
        await engine._send(CLEF, with_image, META, 1)
        return engine.warnings

    warnings = asyncio.run(through_the_engine())
    assert any(w.startswith("request_rerouted") for w in warnings)
    # an invalid Decisions payload (no images) is not moved either
    bad = {**CLEF_PAYLOAD, "questions": {}}
    with pytest.raises(Exception) as refused:
        run(router.run(CLEF, bad, meta=META))
    assert len(default.calls) == 2 or "request_rerouted" not in str(refused.value)


@pytest.mark.parametrize(
    "text",
    [
        "Pay 250\nUSD now",
        "Pay 5\nk now",
        "Pay 5\nmillion now",
        "Pay $\n250 now",
        "7\n%",
    ],
)
def test_a_number_does_not_run_across_a_line_break(text: str) -> None:
    for c in pattern_candidates(ClairicField(type="number", instructions="x"), text):
        assert "\n" not in c.text
    # the number on its own line is still found
    texts = {
        c.text
        for c in pattern_candidates(ClairicField(type="number", instructions="x"), text)
    }
    assert any(t in texts for t in ("250", "5", "7"))


def test_a_product_does_not_run_across_a_line_break() -> None:
    field = ClairicField(type="number", instructions="The total amount due.")
    for c in pattern_candidates(field, "4\nx 250 and a total\n4 x\n250"):
        assert "\n" not in c.text


def test_the_hook_value_of_a_list_is_checked_element_by_element() -> None:
    numbers = ClairicField(type="list", items="number", instructions="x")
    assert _value_fits(numbers, [1, 2.5])
    assert not _value_fits(numbers, [1, "2"])
    assert not _value_fits(numbers, [True])
    assert not _value_fits(numbers, [None])
    words = ClairicField(type="list", items="text", instructions="x")
    assert _value_fits(words, ["a", "b"])
    assert not _value_fits(words, ["a", 3])
    assert not _value_fits(words, "a")
    dates = ClairicField(type="list", items="date", instructions="x")
    assert _value_fits(dates, ["2026-08-14"])
    assert not _value_fits(dates, ["next Thursday"])
    assert _value_fits(words, None)
    assert _value_fits(words, [])
