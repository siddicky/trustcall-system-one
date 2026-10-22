"""TypeSafeTransport: Clef translation, Clairic by selection, errors, retries, limits.

Every request goes to ``httpx.MockTransport``, which also validates it against the
published TypeSafe OpenAPI schema (``docs/schemas/typesafe-openapi.json``). The key is
made up and deliberately not hexadecimal. The example bodies under ``docs/schemas``
(``typesafe-*.example.json``) are the live probes' responses.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CLEF_FLASH,
    AuthError,
    BadRequestError,
    CallMeta,
    ModelResponseError,
    RateLimitError,
    TransportError,
    TransportResponse,
    TypeSafeConfig,
    TypeSafeTransport,
    UncertainOutcomeError,
)
from trustcall_system_one._render import render_clairic
from trustcall_system_one._wire import parse_clairic, parse_clef
from trustcall_system_one.transport import classify_failure, max_questions

ROOT = Path(__file__).resolve().parents[2] / "docs/schemas"
KEY = "apikey_" + "ghjk-mnpq" * 3
URL = "https://api.typesafe.ai/v1/systemone"
META = CallMeta(kind="attribution", schema="Person", entity="Ada Lovelace", field="n")
OPENAPI = json.loads((ROOT / "typesafe-openapi.json").read_text())


def _load(name: str) -> Any:
    return json.loads((ROOT / name).read_text())


def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        {"$ref": f"#/components/schemas/{name}", "components": OPENAPI["components"]}
    )


JEV_REQUEST = _validator("SystemOneRequest")
JEV_RESPONSE = _validator("SystemOneResponse")
CLEF_OUTPUT = jsonschema.Draft202012Validator(_load("clef-output.json"))
CLEF_INPUT = jsonschema.Draft202012Validator(_load("clef-input.json"))
CLAIRIC_OUTPUT = jsonschema.Draft202012Validator(_load("clairic-output.json"))
PROBE = _load("typesafe-systemone-response.example.json")
SPAN_REQUEST = _load("typesafe-span-ids-request.example.json")
SPAN_RESPONSE = _load("typesafe-span-ids-response.example.json")
ERROR_422 = _load("typesafe-error-422.example.json")

CLEF_PAYLOAD: dict[str, Any] = {
    "model": "clef",
    "state": {"input": "I was charged twice. Please refund the duplicate."},
    "questions": {
        "refund_requested": {"type": "noul", "instructions": "Does the customer ask?"},
        "topic": {
            "type": "choice",
            "instructions": "What is it about?",
            "criteria": {
                "billing": "Charges",
                "shipping": "Delivery",
                "account": None,
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["Can wait", "This week", "Today"],
        },
    },
}


class Recorder:
    """A MockTransport handler: queued responses (the last repeats) and sleeps."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.delays: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def make(
    handler: Callable[[httpx.Request], Any] | Recorder,
    **kwargs: Any,
) -> TypeSafeTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("api_key", KEY)
    if isinstance(handler, Recorder):
        kwargs.setdefault("sleep", handler.sleep)
    return TypeSafeTransport(client=client, **kwargs)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def ok(body: Any = PROBE, **headers: str) -> httpx.Response:
    return httpx.Response(200, json=body, headers=headers)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "TYPESAFE_BASE_URL",
        "TYPESAFE_DEFAULT_MODEL",
        "TRUSTCALL_S1_PROVIDER",
        "TRUSTCALL_S1_CLEF_PROVIDER",
        "TRUSTCALL_S1_CLEF_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Clef
# --------------------------------------------------------------------------- #


def test_clef_request_goes_through_unchanged_except_the_model() -> None:
    assert not list(CLEF_INPUT.iter_errors(CLEF_PAYLOAD))
    recorder = Recorder(ok())
    transport = make(recorder)
    sent = run(transport.run(CLEF, CLEF_PAYLOAD, meta=META))
    (request,) = recorder.requests
    assert str(request.url) == URL
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    body = recorder.bodies[0]
    assert not list(JEV_REQUEST.iter_errors(body))
    assert body["model"] == "jev-latest"
    assert body["state"] == CLEF_PAYLOAD["state"]
    assert body["questions"] == CLEF_PAYLOAD["questions"]
    assert list(body["questions"]) == ["refund_requested", "topic", "urgency"]
    assert sent.provider == "typesafe"
    assert sent.status_code == 200
    assert sent.served_model == "jev-1.14.0"


def test_clef_answers_come_back_valid_for_clef() -> None:
    sent = run(make(Recorder(ok())).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert not list(JEV_RESPONSE.iter_errors(PROBE))
    assert not list(CLEF_OUTPUT.iter_errors(sent.body))
    parsed = parse_clef(sent.body)
    assert list(parsed.answers) == ["refund_requested", "topic", "urgency"]
    score = sent.body["answers"]["urgency"]
    # a score level is its position in `criteria`; the legend comes back as strings
    assert score["legend"] == {
        "0": "Can wait",
        "1": "Needs attention this week",
        "2": "Needs attention today",
    }
    assert set(score["probabilities"]) == {"0", "1", "2"}
    assert score["score"] == 2.0
    assert sent.body["usage"] == {"input_tokens": 440, "output_tokens": 70}
    assert "cost" not in sent.body["usage"]


def test_clef_flash_and_bare_names_are_served_too() -> None:
    recorder = Recorder(ok())
    transport = make(recorder, model="jev-1.14.0")
    for model in (CLEF_FLASH, "clef-flash", "clef"):
        payload = {**CLEF_PAYLOAD, "model": "clef-flash"}
        run(transport.run(model, payload, meta=META))
    assert {b["model"] for b in recorder.bodies} == {"jev-1.14.0"}


def test_the_request_id_header_is_kept() -> None:
    sent = run(
        make(Recorder(ok(**{"X-Typesafe-Request-Id": "req-123"}))).run(
            CLEF, CLEF_PAYLOAD, meta=META
        )
    )
    assert sent.request_id == "req-123"


def test_clef_images_are_refused_before_anything_is_sent() -> None:
    recorder = Recorder(ok())
    payload = {**CLEF_PAYLOAD, "images": ["data:image/png;base64,iVBORw0KGgo="]}
    with pytest.raises(BadRequestError, match="images"):
        run(make(recorder).run(CLEF, payload, meta=META))
    assert not recorder.requests
    assert make(Recorder(ok())).serves(CLEF, payload) is False
    assert make(Recorder(ok())).serves(CLEF, CLEF_PAYLOAD) is True


def test_only_the_system_one_models_are_served() -> None:
    recorder = Recorder(ok())
    with pytest.raises(BadRequestError, match="Clairic and Clef"):
        run(make(recorder).run("@cf/meta/llama", CLEF_PAYLOAD, meta=META))
    assert not recorder.requests


def test_an_invalid_clef_request_is_refused_locally() -> None:
    recorder = Recorder(ok())
    with pytest.raises(BadRequestError, match="not a valid Clef request"):
        run(make(recorder).run(CLEF, {"model": "clef", "state": "x"}, meta=META))
    assert not recorder.requests


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b["answers"].pop("topic"),
        lambda b: b["answers"].__setitem__("topic", b["answers"]["refund_requested"]),
        lambda b: b["answers"]["topic"].update(choice="refunds"),
        lambda b: b["answers"]["urgency"]["probabilities"].update({"7": 0.0}),
    ],
    ids=["missing", "wrong type", "unknown option", "unknown level"],
)
def test_an_answer_that_does_not_fit_its_question_is_a_model_response_error(
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    body = json.loads(json.dumps(PROBE))
    mutate(body)
    with pytest.raises(ModelResponseError):
        run(make(Recorder(ok(body))).run(CLEF, CLEF_PAYLOAD, meta=META))


def test_a_response_that_is_not_a_system_one_response() -> None:
    with pytest.raises(ModelResponseError, match="not a JSON object"):
        run(
            make(Recorder(httpx.Response(200, json=[1]))).run(
                CLEF, CLEF_PAYLOAD, meta=META
            )
        )
    with pytest.raises(ModelResponseError, match="non-JSON"):
        run(
            make(Recorder(httpx.Response(200, text="<html>"))).run(
                CLEF, CLEF_PAYLOAD, meta=META
            )
        )
    with pytest.raises(ModelResponseError):
        run(make(Recorder(ok({"model": "jev"}))).run(CLEF, CLEF_PAYLOAD, meta=META))


# --------------------------------------------------------------------------- #
# Errors and retries
# --------------------------------------------------------------------------- #


def test_a_missing_key_is_an_auth_error_before_anything_is_sent() -> None:
    recorder = Recorder(ok())
    transport = make(recorder, api_key=None)
    with pytest.raises(AuthError, match="TYPESAFE_API_KEY"):
        run(transport.run(CLEF, CLEF_PAYLOAD, meta=META))
    assert not recorder.requests


def test_the_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    recorder = Recorder(ok())
    run(make(recorder, api_key=None).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert recorder.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_base_url_and_model_come_from_the_sdk_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://example.test/")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-preview")
    recorder = Recorder(ok())
    run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert str(recorder.requests[0].url) == "https://example.test/v1/systemone"
    assert recorder.bodies[0]["model"] == "jev-preview"


def test_long_context_sends_the_documented_header() -> None:
    recorder = Recorder(ok())
    run(make(recorder, long_context=True).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert recorder.requests[0].headers["typesafe-danger-zone-long-context"] == "true"
    plain = Recorder(ok())
    run(make(plain).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert "typesafe-danger-zone-long-context" not in plain.requests[0].headers


def test_401_is_an_auth_error_and_the_message_has_no_key() -> None:
    recorder = Recorder(httpx.Response(401, json={"detail": "Invalid API key"}))
    with pytest.raises(AuthError) as caught:
        run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert KEY not in str(caught.value)
    assert caught.value.status_code == 401
    assert len(recorder.requests) == 1


def test_422_names_the_field_and_not_the_users_text() -> None:
    body = {
        "detail": [
            {
                "type": "too_short",
                "loc": ["body", "questions", "bad", "choice", "criteria"],
                "msg": "Dictionary should have at least 1 item after validation, not 0",
                "input": {"secret": "the user's private words"},
            },
            {
                "type": "string_too_long",
                # past `criteria` the keys are option ids: the user's own words
                "loc": ["body", "questions", "q", "choice", "criteria", "private opt"],
                "msg": "String should have at most 255 characters",
                "input": "the user's private words",
                "ctx": {"max_length": 255, "given": "the user's private words"},
            },
        ]
    }
    recorder = Recorder(httpx.Response(422, json=body))
    with pytest.raises(BadRequestError) as caught:
        run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    message = str(caught.value)
    assert "body.questions.bad.choice.criteria" in message
    assert "at least 1 item" in message
    assert "private" not in message
    assert caught.value.status_code == 422
    # the body that stays on the error has type, loc (cut at `criteria`) and msg
    kept: Any = caught.value.body
    assert kept["detail"][0]["loc"] == [
        "body",
        "questions",
        "bad",
        "choice",
        "criteria",
    ]
    assert kept["detail"][1]["loc"] == ["body", "questions", "q", "choice", "criteria"]
    assert all(set(e) == {"type", "loc", "msg"} for e in kept["detail"])
    assert "private" not in json.dumps(kept)
    assert "private" not in json.dumps(caught.value.cf_errors)
    assert "private" not in repr(caught.value)
    assert len(recorder.requests) == 1
    # the probed body has the same shape
    assert ERROR_422["detail"][0]["loc"][-1] == "criteria"


def test_429_with_retry_after_waits_and_retries() -> None:
    recorder = Recorder(
        httpx.Response(429, headers={"Retry-After": "2"}, json={}), ok()
    )
    sent = run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert recorder.delays == [2.0]
    assert sent.retries == 1
    assert len(recorder.requests) == 2


def test_429_with_retry_after_ms_is_honoured() -> None:
    recorder = Recorder(
        httpx.Response(429, headers={"retry-after-ms": "1500"}, json={}), ok()
    )
    run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert recorder.delays == [1.5]


def test_429_without_a_hint_is_retried_with_backoff() -> None:
    recorder = Recorder(
        httpx.Response(429, json={}), httpx.Response(429, json={}), ok()
    )
    sent = run(
        make(recorder, backoff_base=0.1, backoff_max=1.0).run(
            CLEF, CLEF_PAYLOAD, meta=META
        )
    )
    assert sent.retries == 2
    assert len(recorder.delays) == 2
    assert all(0.1 <= d <= 1.0 for d in recorder.delays)


def test_a_rate_limit_that_outlasts_the_retries_is_a_rate_limit_error() -> None:
    recorder = Recorder(httpx.Response(429, json={"detail": "slow down"}))
    with pytest.raises(RateLimitError) as caught:
        run(make(recorder, max_retries=2).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert caught.value.status_code == 429
    assert caught.value.retries == 2
    assert len(recorder.requests) == 3
    assert classify_failure(caught.value) == "unavailable"  # nothing ran


def test_529_may_have_been_run_so_it_is_uncertain_and_never_resent() -> None:
    """Overloaded is a 5xx: the server may have run the model, and may have charged."""
    recorder = Recorder(httpx.Response(529, json={"detail": "overloaded"}), ok())
    with pytest.raises(UncertainOutcomeError) as caught:
        run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert caught.value.status_code == 529
    assert len(recorder.requests) == 1  # exactly one request: no retry
    assert recorder.delays == []
    assert "may have charged" in str(caught.value)
    assert classify_failure(caught.value) == "uncertain"


def test_a_retry_after_longer_than_the_limit_fails_at_once() -> None:
    recorder = Recorder(httpx.Response(429, headers={"Retry-After": "999"}, json={}))
    with pytest.raises(RateLimitError):
        run(make(recorder, max_retry_after=10.0).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert len(recorder.requests) == 1


def test_a_connect_error_is_retried_because_nothing_was_sent() -> None:
    recorder = Recorder(httpx.ConnectError("refused"), ok())
    sent = run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert sent.retries == 1
    assert len(recorder.requests) == 2


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("dropped"),
        httpx.ReadError("reset"),
        httpx.Response(500, json={"detail": "boom"}),
        httpx.Response(503, json={}),
        httpx.Response(408, json={}),
        httpx.Response(529, json={}),
    ],
    ids=["read timeout", "dropped", "read error", "500", "503", "408", "529"],
)
def test_a_request_that_may_have_arrived_is_never_resent(
    failure: httpx.Response | Exception,
) -> None:
    recorder = Recorder(failure, ok())
    with pytest.raises(UncertainOutcomeError) as caught:
        run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
    assert len(recorder.requests) == 1
    assert "may have charged" in str(caught.value)
    assert classify_failure(caught.value) == "uncertain"


def test_other_client_errors_are_bad_requests() -> None:
    for status in (400, 413, 415):
        recorder = Recorder(
            httpx.Response(status, json={"detail": "max_tokens_exceeded"})
        )
        with pytest.raises(BadRequestError, match="max_tokens_exceeded"):
            run(make(recorder).run(CLEF, CLEF_PAYLOAD, meta=META))
        assert len(recorder.requests) == 1


def test_sensitive_values_lists_an_explicit_key() -> None:
    assert make(Recorder(ok())).sensitive_values() == (KEY,)
    assert make(Recorder(ok()), api_key=None).sensitive_values() == ()


def test_the_question_limit_is_the_clef_schemas_not_the_apis() -> None:
    # the docs state no limit on the number of questions, only the context
    transport = make(Recorder(ok()))
    assert max_questions(transport, CLEF) == 64
    assert max_questions(transport, CLAIRIC) == 64


def test_the_limiter_is_shared_by_both_models_and_halves_on_429() -> None:
    recorder = Recorder(
        httpx.Response(429, headers={"Retry-After": "0"}, json={}), ok()
    )
    transport = make(recorder)
    before = transport.limiter.limit("typesafe")
    run(transport.run(CLEF, CLEF_PAYLOAD, meta=META))
    assert transport.limiter.limit("typesafe") == max(1, before // 2)


# --------------------------------------------------------------------------- #
# Clairic by selection
# --------------------------------------------------------------------------- #

MAIL = (
    "From: Dana Whit <dana.whit@acme-corp.com>\n"
    "To: billing@acme-corp.com\n"
    "Cc: orders@acme-corp.com\n"
    "Reply-To: dana.personal@gmail.com\n"
    "\n"
    "Hi team - please don't use the billing alias for this one. Send my receipt to "
    "my personal address instead. Thanks, Dana."
)


def clairic_payload(
    fields: dict[str, dict[str, Any]],
    text: str = MAIL,
    *,
    state_text: bool = True,
    reference_date: str | None = None,
    **render: Any,
) -> dict[str, Any]:
    rendered = render_clairic(text, label="the Person", **render)
    payload: dict[str, Any] = {
        "model": "clairic",
        "state": rendered.text,
        "fields": fields,
    }
    if state_text:
        payload["options"] = {"return_state_text": True}
    if reference_date:
        payload["reference_date"] = reference_date
    return payload


class Jev:
    """A scripted Jev: ``script(qid, question, body)`` answers a question (or not)."""

    def __init__(
        self,
        script: Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any] | None],
    ) -> None:
        self.script = script
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert not list(JEV_REQUEST.iter_errors(body))
        self.requests.append(body)
        answers = {}
        for qid, question in body["questions"].items():
            given = self.script(qid, question, body)
            answers[qid] = given if given is not None else default_answer(question)
        reply = {
            "model": "jev-1.14.0",
            "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        assert not list(JEV_RESPONSE.iter_errors(reply))
        return httpx.Response(200, json=reply)

    def stages(self) -> list[list[str]]:
        return [list(r["questions"]) for r in self.requests]


def choice(question: dict[str, Any], pick: str, p: float = 0.93) -> dict[str, Any]:
    options = list(question["criteria"])
    rest = (1 - p) / max(1, len(options) - 1)
    return {
        "type": "choice",
        "choice": pick,
        "confidence": p,
        "probabilities": {o: (p if o == pick else rest) for o in options},
    }


def default_answer(question: dict[str, Any]) -> dict[str, Any]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.04}
    return choice(
        question,
        "none" if "none" in question["criteria"] else next(iter(question["criteria"])),
        0.9,
    )


def picks(**by_field: str | None) -> Callable[..., dict[str, Any] | None]:
    """Script: choose the named option of ``<field>::pick`` (``None``: ``none``)."""

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        name, _, part = qid.partition("::")
        if part == "pick" and name in by_field:
            return choice(question, by_field[name] or "none")
        return None

    return script


def clairic(transport: TypeSafeTransport, payload: dict[str, Any]) -> TransportResponse:
    sent: TransportResponse = run(transport.run(CLAIRIC, payload, meta=META))
    return sent


EMAIL_FIELD = {
    "email": {
        "type": "text",
        "instructions": "The email address where Dana Whit wants the receipt sent.",
    }
}


def test_a_text_field_is_a_choice_among_found_spans_with_a_none_option() -> None:
    jev = Jev(picks(email="dana.personal@gmail.com"))
    payload = clairic_payload(EMAIL_FIELD)
    sent = clairic(make(jev, config=TypeSafeConfig(verify_selection=False)), payload)
    stage = [r for r in jev.requests if "email::pick" in r["questions"]]
    assert len(stage) == 1
    question = stage[0]["questions"]["email::pick"]
    criteria = question["criteria"]
    # the cookbook's shape: the span is the option id and needs no description
    assert criteria["dana.personal@gmail.com"] is None
    assert criteria["dana.whit@acme-corp.com"] is None
    assert list(criteria)[-1] == "none"
    assert criteria["none"] == "None of these is the requested value."
    # structure: blocks and sentences by id, with short descriptions
    ids = [k for k in criteria if k[0] in "BS" and k[1:4].isdigit()]
    assert ids
    assert all(len(criteria[i]) < 140 for i in ids)
    assert "Which option is it?" in question["instructions"]
    state = stage[0]["state"]
    assert "Answer only about the Person" in state["focus"]
    assert state["input"].startswith("B000| [S000] From:")
    answer = sent.body["answers"]["email"]
    assert answer["value"] == "dana.personal@gmail.com"
    assert answer["type"] == "text"
    assert answer["guess"] == "dana.personal@gmail.com"
    assert answer["p_null"] < 0.01  # Jev shared 0.07 over the other options
    assert answer["confidence"] == 0.93
    # the span is an exact offset into the state that was sent
    start, end = answer["span"]
    assert sent.body["state_text"] == payload["state"]
    assert payload["state"][start:end] == "dana.personal@gmail.com"


def test_the_response_is_a_valid_clairic_response() -> None:
    jev = Jev(picks(email="dana.personal@gmail.com"))
    sent = clairic(make(jev), clairic_payload(EMAIL_FIELD))
    assert not list(CLAIRIC_OUTPUT.iter_errors(sent.body))
    parsed = parse_clairic(sent.body)
    assert parsed.truncated is False
    assert parsed.answers["email"].value == "dana.personal@gmail.com"
    assert parsed.usage.passes == len(jev.requests)
    assert sent.provider == "typesafe"
    assert sent.served_model == "jev-1.14.0"
    assert sent.body["jev"]["requests"] == len(jev.requests)
    assert "cost" not in sent.body["usage"]


def test_state_text_is_returned_only_when_asked() -> None:
    sent = clairic(make(Jev(picks())), clairic_payload(EMAIL_FIELD, state_text=False))
    assert "state_text" not in sent.body


def test_none_chosen_is_a_null_with_the_top_candidate_as_guess() -> None:
    jev = Jev(picks(email=None))
    sent = clairic(make(jev), clairic_payload(EMAIL_FIELD))
    answer = sent.body["answers"]["email"]
    assert answer["value"] is None
    assert "span" not in answer
    assert answer["p_null"] > 0.5
    assert answer["guess"] is not None
    assert not list(CLAIRIC_OUTPUT.iter_errors(sent.body))


def test_the_null_threshold_decides_between_value_and_null() -> None:
    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid == "email::pick":
            options = list(question["criteria"])
            probs = {o: 0.0 for o in options}
            probs["dana.personal@gmail.com"] = 0.6
            probs["none"] = 0.4
            return {
                "type": "choice",
                "choice": "dana.personal@gmail.com",
                "confidence": 0.2,
                "probabilities": probs,
            }
        return None

    config = TypeSafeConfig(verify_selection=False)
    plain = clairic(make(Jev(script), config=config), clairic_payload(EMAIL_FIELD))
    assert plain.body["answers"]["email"]["value"] == "dana.personal@gmail.com"
    strict = {"email": {**EMAIL_FIELD["email"], "null_threshold": 0.3}}
    sent = clairic(make(Jev(script), config=config), clairic_payload(strict))
    answer = sent.body["answers"]["email"]
    assert answer["value"] is None
    assert answer["guess"] == "dana.personal@gmail.com"
    assert answer["p_null"] == pytest.approx(0.4)


def test_a_field_with_no_candidate_is_a_confident_null_without_a_request() -> None:
    jev = Jev(picks())
    sent = clairic(
        make(jev),
        clairic_payload(
            {"n": {"type": "number", "instructions": "x"}}, "no digits here"
        ),
    )
    assert sent.body["answers"]["n"]["value"] is None
    assert sent.body["answers"]["n"]["p_null"] == 1.0
    assert not jev.requests


def test_all_fields_are_asked_in_one_request_over_the_same_state() -> None:
    fields = {
        "email": EMAIL_FIELD["email"],
        "name": {"type": "text", "instructions": "The sender's name"},
        "when": {"type": "date", "instructions": "The date it was sent"},
        "tags": {"type": "list", "items": "text", "instructions": "People named"},
    }
    jev = Jev(picks())
    clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(fields),
    )
    selecting = [r for r in jev.requests if "name::pick" in r["questions"]]
    assert len(selecting) == 1
    questions = selecting[0]["questions"]
    assert {
        "email::pick",
        "name::pick",
        "when::mode",
        "when::month",
        "tags::in0",
    } <= set(questions)
    # one state for all of them
    assert (
        len(
            {
                json.dumps(r["state"], sort_keys=True)
                for r in jev.requests
                if "email::pick" in r["questions"]
            }
        )
        == 1
    )


def test_the_stitch_request_runs_once_per_input_and_is_remembered() -> None:
    jev = Jev(picks())
    transport = make(jev, config=TypeSafeConfig(verify_selection=False))
    clairic(transport, clairic_payload(EMAIL_FIELD))
    stitches = [r for r in jev.requests if "L001" in r["questions"]]
    assert len(stitches) == 1
    assert stitches[0]["state"]["input"].startswith("L000| From:")
    clairic(transport, clairic_payload(EMAIL_FIELD))
    assert len([r for r in jev.requests if "L001" in r["questions"]]) == 1  # remembered
    other = clairic_payload(EMAIL_FIELD, MAIL + "\nPS see you.")
    clairic(transport, other)
    assert len([r for r in jev.requests if "L001" in r["questions"]]) == 2


def test_a_single_line_input_needs_no_stitch_request() -> None:
    jev = Jev(picks())
    clairic(
        make(jev), clairic_payload(EMAIL_FIELD, "Write to dana@acme-corp.com please.")
    )
    assert not [r for r in jev.requests if "L001" in r["questions"]]


def test_stitching_can_be_turned_off() -> None:
    jev = Jev(picks())
    clairic(
        make(jev, config=TypeSafeConfig(stitch=False)), clairic_payload(EMAIL_FIELD)
    )
    assert not [r for r in jev.requests if "L001" in r["questions"]]


WRAPPED = "Please call us about the\ninvoice that is overdue.\nThanks."


def test_a_sentence_across_a_hard_wrap_is_one_block_with_exact_offsets() -> None:
    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid == "L001":
            return {"type": "noul", "noul": 0.6}  # picks up mid-sentence
        if qid == "ask::pick":
            return choice(question, "S000")
        return None

    jev = Jev(script)
    payload = clairic_payload(
        {"ask": {"type": "text", "instructions": "What to do"}}, WRAPPED
    )
    sent = clairic(make(jev, config=TypeSafeConfig(verify_selection=False)), payload)
    select = next(r for r in jev.requests if "ask::pick" in r["questions"])
    assert select["state"]["input"].splitlines() == [
        "B000| [S000] Please call us about the invoice that is overdue.",
        "B001| [S001] Thanks.",
    ]
    answer = sent.body["answers"]["ask"]
    assert answer["value"] == "Please call us about the invoice that is overdue."
    start, end = answer["span"]
    assert (
        payload["state"][start:end]
        == "Please call us about the\ninvoice that is overdue."
    )


def test_a_notes_list_asks_one_yes_no_per_block_or_sentence() -> None:
    notes = "Dana moved to Berlin. She likes hiking.\nShe started on Monday."

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid.startswith("notes::in"):
            cand = question["instructions"]["candidate"]
            hit = cand.get("id") in ("S001", "S002")
            return {"type": "noul", "noul": 0.9 if hit else 0.02}
        return None

    jev = Jev(script)
    field = {
        "notes": {
            "type": "list",
            "items": "text",
            "instructions": "Things worth remembering",
        }
    }
    payload = clairic_payload(field, notes)
    sent = clairic(make(jev), payload)
    select = next(r for r in jev.requests if "notes::in0" in r["questions"])
    asked = select["questions"]
    refs = [
        q["instructions"]["candidate"]
        for qid, q in asked.items()
        if qid.startswith("notes::in")
    ]
    assert "notes::kind" in asked  # asked in the same request: items or statements
    # a sentence is named by its id, the text is in the state
    assert {"id": "S001"} in refs
    assert all("text" not in r for r in refs if r.get("id") in ("S000", "S001", "S002"))
    answer = sent.body["answers"]["notes"]
    assert answer["value"] == ["She likes hiking.", "She started on Monday."]
    state = payload["state"]
    assert [state[a:b] for a, b in answer["spans"]] == answer["value"]
    assert answer["confidence"] == 0.9
    assert "p_null" not in answer


def test_a_sentence_and_its_part_do_not_both_stay_in_a_list() -> None:
    text = "Dana likes hiking and climbing."

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid.startswith("notes::in"):
            cand = question["instructions"]["candidate"]
            return {
                "type": "noul",
                "noul": 0.9 if cand.get("id", "").startswith("S000") else 0.01,
            }
        return None

    field = {"notes": {"type": "list", "items": "text", "instructions": "Facts"}}
    sent = clairic(make(Jev(script)), clairic_payload(field, text))
    assert sent.body["answers"]["notes"]["value"] == ["Dana likes hiking and climbing."]


def test_an_empty_list_is_null_with_the_confidence_of_the_noes() -> None:
    field = {"people": {"type": "list", "items": "text", "instructions": "People"}}
    sent = clairic(make(Jev(picks())), clairic_payload(field, "Dana wrote."))
    answer = sent.body["answers"]["people"]
    assert answer["value"] is None
    assert answer["confidence"] == pytest.approx(0.96)
    assert not list(CLAIRIC_OUTPUT.iter_errors(sent.body))


def test_a_number_field_picks_among_found_numbers_and_returns_the_number() -> None:
    text = "Invoice INV-2291: 4 pallets at $250.00 each, shipping $80.00."
    field = {"total": {"type": "number", "instructions": "The total amount due"}}
    jev = Jev(picks(total="1080 (the sum of 1000 + 80)"))
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(field, text),
    )
    select = next(r for r in jev.requests if "total::pick" in r["questions"])
    question = select["questions"]["total::pick"]
    criteria = question["criteria"]
    # the sum is offered under a name that says what it is, not as the sentence
    assert "4 pallets at $250.00 each, shipping $80.00" not in criteria
    assert criteria["1080 (the sum of 1000 + 80)"] == (
        "the total of the amounts written in "
        "'4 pallets at $250.00 each, shipping $80.00'"
    )
    assert "is the total of the amounts" in question["instructions"]
    assert criteria["$250.00"] == "250 USD"
    answer = sent.body["answers"]["total"]
    assert answer["value"] == 1080
    payload_state = clairic_payload(field, text)["state"]
    start, end = answer["span"]
    assert payload_state[start:end] == "4 pallets at $250.00 each, shipping $80.00"


def test_the_estimate_type_is_a_number_too() -> None:
    field = {
        "n": {"type": "estimate", "instructions": "How many", "min": 0, "max": 100}
    }
    sent = clairic(
        make(Jev(picks(n="12"))), clairic_payload(field, "We shipped 12 units.")
    )
    assert sent.body["answers"]["n"]["value"] == 12
    assert sent.body["answers"]["n"]["type"] == "estimate"


def _part_script(
    parts: dict[str, str], p: float = 0.97
) -> Callable[..., dict[str, Any] | None]:
    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        _, _, part = qid.partition("::")
        if part in parts:
            return choice(question, parts[part], p)
        return None

    return script


def test_a_date_is_read_as_parts_and_resolved_in_code() -> None:
    text = "Please return the signed form by August 14, 2026."
    field = {"due": {"type": "date", "instructions": "The deadline to return the form"}}
    script = _part_script(
        {"mode": "absolute", "month": "August", "day": "14", "year": "2026"}
    )
    jev = Jev(script)
    payload = clairic_payload(field, text, reference_date="2026-07-30")
    sent = clairic(make(jev), payload)
    (request,) = jev.requests  # one request: no structure and no selection to verify
    assert list(request["questions"]) == [
        "due::mode",
        "due::month",
        "due::day",
        "due::year",
        "due::day_anchor",
        "due::weekday",
        "due::week_offset",
    ]
    assert (
        "How is the deadline to return the form written?"
        in request["questions"]["due::mode"]["instructions"]
    )
    answer = sent.body["answers"]["due"]
    assert answer["value"] == "2026-08-14"
    assert answer["confidence"] == pytest.approx(0.97)
    assert answer["p_null"] == pytest.approx(0.01, abs=0.01)
    assert "span" not in answer
    assert answer["needs_review"] is False
    assert not list(CLAIRIC_OUTPUT.iter_errors(sent.body))


def test_a_year_that_the_text_does_not_state_is_inferred_and_flagged() -> None:
    text = "Please return the signed form by August 14."
    field = {"due": {"type": "date", "instructions": "The deadline to return the form"}}
    script = _part_script(
        {"mode": "absolute", "month": "August", "day": "14", "year": "none"}
    )
    payload = clairic_payload(field, text, reference_date="2026-07-30")
    answer = clairic(make(Jev(script)), payload).body["answers"]["due"]
    assert answer["value"] == "2026-08-14"
    assert answer["needs_review"] is True
    assert answer["confidence"] == pytest.approx(0.8)  # under the re-ask bar
    assert "inferred year" in answer["note"]


def test_a_relative_date_uses_the_reference_date() -> None:
    script = _part_script(
        {
            "mode": "relative",
            "day_anchor": "weekday",
            "weekday": "Thursday",
            "week_offset": "next",
        }
    )
    field = {"due": {"type": "date", "instructions": "The review date"}}
    payload = clairic_payload(
        field, "Review next Thursday.", reference_date="2026-07-30"
    )
    assert (
        clairic(make(Jev(script)), payload).body["answers"]["due"]["value"]
        == "2026-08-06"
    )


def test_a_date_that_is_not_stated_is_null_and_flagged_for_review() -> None:
    field = {"due": {"type": "date", "instructions": "The kickoff call date"}}
    jev = Jev(_part_script({"mode": "none"}, 0.46))
    sent = clairic(
        make(jev),
        clairic_payload(field, "A date: 2026-01-01.", reference_date="2026-07-30"),
    )
    answer = sent.body["answers"]["due"]
    assert answer["value"] is None
    assert answer["needs_review"] is True
    assert answer["confidence"] == pytest.approx(0.46)
    assert answer["p_null"] >= 0.5


def test_time_datetime_and_duration_fields() -> None:
    text = "The call is on August 14, 2027 at 2:30 pm and lasts 2 hours 30 minutes."
    fields = {
        "at": {"type": "time", "instructions": "The call time"},
        "when": {"type": "datetime", "instructions": "When the call is"},
        "length": {
            "type": "duration",
            "instructions": "How long it lasts",
            "unit": "minutes",
        },
    }

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        picks = {
            "at::mode": "stated",
            "at::hour": "14",
            "at::minute": "30",
            "when::mode": "absolute",
            "when::month": "August",
            "when::day": "14",
            "when::year": "2027",
            "when::t.mode": "stated",
            "when::t.hour": "14",
            "when::t.minute": "30",
            "length::mode": "stated",
            "length::hours": "2",
            "length::minutes": "30",
        }
        if qid in picks:
            return choice(question, picks[qid], 0.97)
        return None

    sent = clairic(make(Jev(script)), clairic_payload(fields, text))
    answers = sent.body["answers"]
    assert answers["at"]["value"] == "14:30:00"
    assert answers["when"]["value"] == "2027-08-14T14:30:00"
    assert answers["length"]["value"] == 150
    assert not list(CLAIRIC_OUTPUT.iter_errors(sent.body))


def test_a_list_of_dates_asks_the_parts_of_each_accepted_mention() -> None:
    text = "Due August 14, 2027 and again on 2028-01-05."

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        _, _, rest = qid.partition("::")
        if rest.startswith("in"):
            return {"type": "noul", "noul": 0.9}
        if "::e0::" in qid or "::e1::" in qid:
            first = qid.startswith("dates::e0")
            part = qid.rsplit("::", 1)[1]
            value = {
                True: {
                    "mode": "absolute",
                    "month": "August",
                    "day": "14",
                    "year": "2027",
                },
                False: {
                    "mode": "absolute",
                    "month": "January",
                    "day": "5",
                    "year": "2028",
                },
            }[first]
            if part in value:
                return choice(question, value[part])
        return None

    field = {"dates": {"type": "list", "items": "date", "instructions": "Due dates"}}
    jev = Jev(script)
    sent = clairic(make(jev), clairic_payload(field, text))
    answer = sent.body["answers"]["dates"]
    assert answer["value"] == ["2027-08-14", "2028-01-05"]
    assert len(answer["spans"]) == 2
    assert len(jev.requests) == 2  # the mentions, then the parts of both


def test_more_than_254_candidates_are_chosen_in_rounds() -> None:
    text = " ".join(f"item{i} and" for i in range(300)) + " done"
    text = "\n".join(f"Code: AB-{1000 + i}" for i in range(300))
    field = {"code": {"type": "text", "instructions": "The code of order 777"}}
    seen: list[int] = []

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid.startswith("code::pick") or qid == "code::final":
            seen.append(len(question["criteria"]))
            if qid == "code::final" or "AB-1255" in question["criteria"]:
                return choice(question, "AB-1255")
            return choice(question, "none")
        return None

    jev = Jev(script)
    config = TypeSafeConfig(verify_selection=False, stitch=False)
    sent = clairic(make(jev, config=config), clairic_payload(field, text))
    assert max(seen) <= 255  # a Jev choice takes 255 options, one of them `none`
    assert (
        len(
            [
                q
                for r in jev.requests
                for q in r["questions"]
                if q.startswith("code::pick")
            ]
        )
        >= 2
    )
    final = [q for r in jev.requests for q in r["questions"] if q == "code::final"]
    assert len(final) == 1
    assert sent.body["answers"]["code"]["value"] == "AB-1255"


def test_a_long_input_is_cut_and_reported_truncated() -> None:
    text = "Write to dana@acme-corp.com please. " + "filler words here. " * 100
    config = TypeSafeConfig(max_input_chars=400, verify_selection=False, stitch=False)
    payload = clairic_payload(EMAIL_FIELD, text)
    sent = clairic(make(Jev(picks(email="dana@acme-corp.com")), config=config), payload)
    assert sent.body["truncated"] is True
    state_text = sent.body["state_text"]
    assert payload["state"].startswith(state_text)
    assert len(state_text) < len(payload["state"])
    start, end = sent.body["answers"]["email"]["span"]
    assert state_text[start:end] == "dana@acme-corp.com"
    parse_clairic(sent.body)


def test_requests_are_split_by_the_context_budget_alone() -> None:
    text = "\n".join(
        f"Person{i} Name{i} met Person{i + 1} Name{i + 1}." for i in range(60)
    )
    field = {"people": {"type": "list", "items": "text", "instructions": "People"}}
    jev = Jev(picks())
    config = TypeSafeConfig(
        verify_selection=False, stitch=False, max_list_candidates=1000
    )
    clairic(make(jev, config=config), clairic_payload(field, text))
    sizes = [len(r["questions"]) for r in jev.requests]
    # Jev states no limit on the number of questions: they go by the context alone
    assert len(sizes) == 1
    assert sizes[0] > 64
    jev2 = Jev(picks())
    tiny = TypeSafeConfig(
        verify_selection=False,
        stitch=False,
        max_list_candidates=1000,
        request_token_budget=2000,
    )
    clairic(make(jev2, config=tiny), clairic_payload(field, text))
    assert len(jev2.requests) > len(jev.requests)


def test_usage_adds_up_over_the_requests() -> None:
    jev = Jev(picks(email="dana.personal@gmail.com"))
    sent = clairic(make(jev), clairic_payload(EMAIL_FIELD))
    n = len(jev.requests)
    assert n >= 2  # stitch, selection, verification
    assert sent.body["usage"] == {
        "input_tokens": 100 * n,
        "output_tokens": 10 * n,
        "passes": n,
    }
    assert sent.body["jev"]["stages"] == {"stitch": 1, "select": 1, "verify": 1}


def test_images_and_forecasts_are_refused() -> None:
    recorder = Recorder(ok())
    transport = make(recorder)
    payload = clairic_payload(EMAIL_FIELD)
    with pytest.raises(BadRequestError, match="images"):
        clairic(
            transport, {**payload, "images": ["data:image/png;base64,iVBORw0KGgo="]}
        )
    fc = {
        "f": {
            "type": "forecast",
            "min": 0,
            "max": 10,
            "horizon": 3,
            "instructions": "x",
        }
    }
    with pytest.raises(BadRequestError, match="forecast"):
        clairic(transport, clairic_payload(fc))
    assert not recorder.requests
    assert transport.serves(CLAIRIC, clairic_payload(fc)) is False
    assert transport.serves(CLAIRIC, payload) is True
    with pytest.raises(BadRequestError, match="not a valid Clairic request"):
        clairic(transport, {"model": "clairic", "state": "x"})


def test_an_error_from_one_request_of_a_clairic_call_is_that_requests() -> None:
    recorder = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(UncertainOutcomeError):
        clairic(make(recorder), clairic_payload(EMAIL_FIELD))
    assert len(recorder.requests) == 1  # the stitch; not resent
    with pytest.raises(TransportError):
        clairic(
            make(Recorder(httpx.Response(401, json={}))), clairic_payload(EMAIL_FIELD)
        )


# --------------------------------------------------------------------------- #
# Verification and escalation (SDE cascade)
# --------------------------------------------------------------------------- #


def _verify_script(
    p_wrong: float, pick: str = "dana.personal@gmail.com"
) -> Callable[..., dict[str, Any] | None]:
    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid == "email::pick":
            return choice(question, pick)
        if qid == "email::hallucinated":
            return {"type": "noul", "noul": p_wrong}
        if qid == "email::off_target":
            return {"type": "noul", "noul": 0.1}
        return None

    return script


def test_the_selected_value_is_verified_with_the_cookbooks_questions() -> None:
    jev = Jev(_verify_script(0.05))
    sent = clairic(make(jev), clairic_payload(EMAIL_FIELD))
    verify = next(r for r in jev.requests if "email::hallucinated" in r["questions"])
    assert set(verify["questions"]) == {"email::hallucinated", "email::off_target"}
    q = verify["questions"]["email::hallucinated"]
    assert q["instructions"]["extracted_field"] == "dana.personal@gmail.com"
    assert q["instructions"]["field_spec"]["path"] == "email"
    assert q["instructions"]["main_question"].startswith(
        "Is the `extracted_field` unsupported by"
    )
    assert (
        q["criteria"]["false"]
        == "the `extracted_field` is supported by the source text"
    )
    assert verify["state"]["source_text"] == MAIL
    assert verify["state"]["extraction"] == {"email": "dana.personal@gmail.com"}
    answer = sent.body["answers"]["email"]
    assert answer["value"] == "dana.personal@gmail.com"
    assert answer["p_wrong"] == pytest.approx(0.1)
    assert "escalated" not in answer


def test_a_field_that_fires_is_returned_as_a_null_for_the_ladder() -> None:
    sent = clairic(make(Jev(_verify_script(0.95))), clairic_payload(EMAIL_FIELD))
    answer = sent.body["answers"]["email"]
    assert answer["escalated"] == "unavailable"
    assert answer["value"] is None
    assert answer["guess"] == "dana.personal@gmail.com"
    assert answer["p_null"] >= 0.95
    assert answer["confidence"] <= 0.05 + 1e-9
    assert "span" not in answer


def test_the_fire_threshold_is_a_setting() -> None:
    config = TypeSafeConfig(fire_threshold=0.99)
    sent = clairic(
        make(Jev(_verify_script(0.95)), config=config), clairic_payload(EMAIL_FIELD)
    )
    assert sent.body["answers"]["email"]["value"] == "dana.personal@gmail.com"
    assert "escalated" not in sent.body["answers"]["email"]


def test_the_verification_can_be_turned_off() -> None:
    jev = Jev(_verify_script(0.95))
    clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(EMAIL_FIELD),
    )
    assert not [r for r in jev.requests if "email::hallucinated" in r["questions"]]


def test_a_null_answer_is_not_verified() -> None:
    jev = Jev(picks(email=None))
    clairic(make(jev), clairic_payload(EMAIL_FIELD))
    assert not [r for r in jev.requests if "email::hallucinated" in r["questions"]]


def test_the_escalation_hook_answers_a_doubted_field() -> None:
    seen: list[Any] = []

    def hook(request: Any) -> dict[str, Any]:
        seen.append(request)
        return {"value": "orders@acme-corp.com", "span": [50, 70], "confidence": 0.9}

    transport = make(Jev(_verify_script(0.9)), escalate=hook)
    payload = clairic_payload(EMAIL_FIELD)
    sent = clairic(transport, payload)
    (request,) = seen
    assert request.field == "email"
    assert request.selected == "dana.personal@gmail.com"
    assert request.input == MAIL
    assert request.spec["type"] == "text"
    assert request.p_wrong == pytest.approx(0.9)
    assert set(request.signals) == {"hallucinated", "off_target"}
    answer = sent.body["answers"]["email"]
    assert answer["value"] == "orders@acme-corp.com"
    assert answer["escalated"] == "hook"
    assert answer["confidence"] == 0.9
    base = payload["state"].index("<input>\n") + len("<input>\n")
    assert answer["span"] == [
        base + 50,
        base + 70,
    ]  # input offsets become state offsets
    assert answer["p_null"] == 0.0


def test_an_async_hook_may_say_not_stated_or_decline() -> None:
    async def says_null(request: Any) -> dict[str, Any]:
        return {"value": None}

    sent = clairic(
        make(Jev(_verify_script(0.9)), escalate=says_null), clairic_payload(EMAIL_FIELD)
    )
    answer = sent.body["answers"]["email"]
    assert answer["value"] is None
    assert answer["p_null"] == 1.0

    async def declines(request: Any) -> None:
        return None

    sent = clairic(
        make(Jev(_verify_script(0.9)), escalate=declines), clairic_payload(EMAIL_FIELD)
    )
    assert sent.body["answers"]["email"]["escalated"] == "unavailable"


def test_the_hook_is_not_called_when_nothing_fires() -> None:
    calls: list[Any] = []
    transport = make(Jev(_verify_script(0.1)), escalate=lambda r: calls.append(r))
    clairic(transport, clairic_payload(EMAIL_FIELD))
    assert not calls


def test_config_is_validated() -> None:
    with pytest.raises(ValueError, match="fire_threshold"):
        TypeSafeConfig(fire_threshold=1.5)
    with pytest.raises(ValueError, match="context_chars"):
        TypeSafeConfig(context_chars=0)
    defaults = TypeSafeConfig()
    assert (defaults.join_after_dangling, defaults.join_after_terminal) == (0.2, 0.5)
    assert defaults.review_below == 0.60
    assert defaults.fire_threshold == 0.7


def test_the_probed_span_id_example_has_the_shape_the_adapter_reads() -> None:
    """Probe 6: the shipped scheme, a span as the option id, plus `none`."""
    request = SPAN_REQUEST["questions"]["phone::pick"]
    criteria = request["criteria"]
    assert "(415) 555-0177" in criteria  # the span itself is the id, with spaces
    assert "+1 415 555 0142" in criteria  # ... and punctuation
    assert criteria["none"] == "None of these is the requested value."
    assert list(criteria)[-1] == "none"
    answer = SPAN_RESPONSE["answers"]["phone::pick"]
    assert answer["choice"] == "(415) 555-0177"  # it answered with the span, verbatim
    assert answer["probabilities"]["(415) 555-0177"] == 0.65
    assert answer["probabilities"]["none"] == 0.28
    assert answer["confidence"] == 0.6
    none = SPAN_RESPONSE["answers"]["email::pick"]  # a field the input does not state
    assert none["choice"] == "none"
    assert none["probabilities"]["none"] == 1.0
    assert not list(JEV_REQUEST.iter_errors(SPAN_REQUEST))
    assert not list(JEV_RESPONSE.iter_errors(SPAN_RESPONSE))
    parsed = parse_clef(SPAN_RESPONSE)
    assert set(parsed.answers) == {"phone::pick", "email::pick"}
