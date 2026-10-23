"""OpenAIResponsesTransport (the escalation tier's client), offline.

Every request goes to ``httpx.MockTransport``. The key below is made up. What is known
about the live API comes from two probes of ``gpt-6-luna`` (a strict-schema request at
``reasoning.effort`` ``none``, with and without logprobs); their scrubbed bodies are
``docs/schemas/openai-responses-*.example.json``.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from trustcall_system_one import (
    AuthError,
    BadRequestError,
    CallMeta,
    ModelResponseError,
    OpenAIResponsesTransport,
    RateLimitError,
    TransportError,
    UncertainOutcomeError,
    create_extractor,
    escalation_from_env,
    system_one_transport,
)
from trustcall_system_one.openai_transport import EFFORT_TIMEOUTS, REASONING_EFFORTS
from trustcall_system_one.testing import RecordingTransport, scrub
from trustcall_system_one.transport import classify_failure

KEY = "sk-proj-" + "ab12cd34" * 6
URL = "https://api.openai.com/v1/responses"
META = CallMeta(kind="escalate", schema="Report", entity="e0", field="relative_dates")
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": ["string", "null"]}},
    "required": ["city"],
    "additionalProperties": False,
}
USER_TEXT = "Dana Whitfield moved to Berlin and her salary is 123456 euros."
PAYLOAD: dict[str, Any] = {
    "input": f"Text: {USER_TEXT} Which city?",
    "schema": SCHEMA,
    "schema_name": "city",
}
DOCS = Path(__file__).resolve().parents[2] / "docs/schemas"
JSON_BODY: dict[str, Any] = json.loads(
    (DOCS / "openai-responses-json.example.json").read_text()
)
LOGPROBS_BODY: dict[str, Any] = json.loads(
    (DOCS / "openai-responses-logprobs.example.json").read_text()
)


class _Doc(BaseModel):
    """A document."""

    name: str


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "TRUSTCALL_S1_ESCALATE",
        "TRUSTCALL_S1_ESCALATE_EFFORT",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


class Recorder:
    """A MockTransport handler that records requests and the sleeps."""

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

    def body(self, index: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[index].content)
        return data


def make(
    recorder: Recorder, **kwargs: Any
) -> tuple[OpenAIResponsesTransport, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    kwargs.setdefault("api_key", KEY)
    transport = OpenAIResponsesTransport(client=client, sleep=recorder.sleep, **kwargs)
    return transport, client


def ok(body: dict[str, Any] | None = None, **headers: str) -> httpx.Response:
    return httpx.Response(200, json=body or JSON_BODY, headers=headers)


def error(
    status: int,
    code: str | None = None,
    kind: str = "invalid_request_error",
    message: str = "boom",
    **headers: str,
) -> httpx.Response:
    body = {"error": {"message": message, "type": kind, "param": None, "code": code}}
    return httpx.Response(status, json=body, headers=headers)


async def run(
    transport: OpenAIResponsesTransport,
    payload: dict[str, Any] | None = None,
    **kw: Any,
) -> Any:
    return await transport.run("gpt-6-luna", payload or PAYLOAD, meta=META, **kw)


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #


async def test_the_request_is_a_strict_json_schema_call_with_effort_none() -> None:
    rec = Recorder(ok())
    transport, _ = make(rec)
    await run(transport)
    request = rec.requests[0]
    assert str(request.url) == URL and request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = rec.body()
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning"] == {"effort": "none"}
    assert body["input"] == PAYLOAD["input"]
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "city",
            "strict": True,
            "schema": SCHEMA,
        }
    }
    assert body["store"] is False  # OpenAI is asked not to keep the text
    assert "include" not in body and "top_logprobs" not in body  # logprobs are opt-in


async def test_instructions_and_a_call_effort_go_in_the_body() -> None:
    rec = Recorder(ok())
    transport, _ = make(rec, reasoning_effort="low")
    await run(transport, {**PAYLOAD, "instructions": "Be literal."})
    assert rec.body()["instructions"] == "Be literal."
    assert rec.body()["reasoning"] == {"effort": "low"}
    await run(transport, {**PAYLOAD, "reasoning_effort": "high"})
    assert rec.body(1)["reasoning"] == {"effort": "high"}


async def test_the_key_comes_from_the_environment_and_a_missing_one_is_an_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(ok())
    transport, _ = make(rec, api_key=None)
    with pytest.raises(AuthError, match="OPENAI_API_KEY"):
        await run(transport)
    assert rec.requests == []  # nothing was sent
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    await run(transport)
    assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, "https://api.openai.com/v1/responses"),
        ("https://proxy.example/", "https://proxy.example/v1/responses"),
        ("https://proxy.example/v1", "https://proxy.example/v1/responses"),
        ("https://proxy.example/v1/responses", "https://proxy.example/v1/responses"),
    ],
)
async def test_base_url_argument_and_environment(
    given: str | None, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Recorder(ok())
    transport, _ = make(rec, base_url=given)
    await run(transport)
    assert str(rec.requests[0].url) == expected
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example")
    env, _ = make(Recorder(ok()))
    assert env._base_url == "https://env.example/v1/responses"
    explicit, _ = make(Recorder(ok()), base_url="https://arg.example")
    assert explicit._base_url == "https://arg.example/v1/responses"  # the argument wins


async def test_logprobs_are_opt_in_and_returned_untouched() -> None:
    rec = Recorder(ok(LOGPROBS_BODY))
    transport, _ = make(rec)
    reply = await run(transport, {**PAYLOAD, "top_logprobs": 5})
    body = rec.body()
    assert body["include"] == ["message.output_text.logprobs"]
    assert body["top_logprobs"] == 5
    assert reply.body["parsed"] == {"answer": "yes"}
    # under a strict schema top_logprobs holds only the chosen token, so there is no
    # P(no) to compare with: the transport hands the entries over and computes nothing
    assert len(reply.body["logprobs"]) == 5
    assert all(len(e["top_logprobs"]) == 1 for e in reply.body["logprobs"])
    assert "probabilities" not in reply.body


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


async def test_the_answer_is_the_output_text_item_and_usage_is_read() -> None:
    transport, _ = make(Recorder(ok(x_request_id="req_1")))
    reply = await run(transport)
    assert reply.body["parsed"] == {"city": "Berlin"}
    assert reply.body["no_answer"] is None
    assert reply.body["usage"] == {
        "input_tokens": 52,
        "output_tokens": 12,
        "reasoning_tokens": 0,
    }
    assert reply.provider == "openai" and reply.served_model == "gpt-6-luna"
    assert reply.status_code == 200 and reply.retries == 0


async def test_a_reasoning_item_before_the_message_is_ignored() -> None:
    body = json.loads(json.dumps(JSON_BODY))
    body["output"].insert(
        0,
        {
            "type": "reasoning",
            "id": "<id>",
            "summary": [{"type": "summary_text", "text": "I thought about Dana"}],
        },
    )
    body["usage"]["output_tokens"] = 240
    body["usage"]["output_tokens_details"]["reasoning_tokens"] = 228
    transport, _ = make(Recorder(ok(body)), reasoning_effort="medium")
    reply = await run(transport)
    assert reply.body["parsed"] == {"city": "Berlin"}
    # reasoning tokens are counted (they are output tokens); the summary is never kept
    assert reply.body["usage"]["reasoning_tokens"] == 228
    assert reply.body["usage"]["output_tokens"] == 240
    assert "I thought about" not in json.dumps(reply.body)


async def test_a_refusal_is_no_answer_not_an_error() -> None:
    body = json.loads(json.dumps(JSON_BODY))
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "I can't help."}]
    transport, _ = make(Recorder(ok(body)))
    reply = await run(transport)
    assert reply.body["parsed"] is None and reply.body["no_answer"] == "refusal"
    assert reply.body["usage"]["input_tokens"] == 52  # it was charged


@pytest.mark.parametrize(
    ("status", "details", "expected"),
    [
        ("incomplete", {"reason": "max_output_tokens"}, "status incomplete"),
        ("failed", None, "status failed"),
    ],
)
async def test_a_response_that_is_not_completed_is_no_answer(
    status: str, details: dict[str, str] | None, expected: str
) -> None:
    body = json.loads(json.dumps(JSON_BODY))
    body["status"] = status
    body["incomplete_details"] = details
    body["output"] = []
    rec = Recorder(ok(body))
    transport, _ = make(rec)
    reply = await run(transport)
    assert reply.body["parsed"] is None
    assert reply.body["no_answer"].startswith(expected)
    assert len(rec.requests) == 1  # the response arrived: nothing is sent again


async def test_text_that_is_not_json_is_a_model_response_error_without_the_text() -> (
    None
):
    body = json.loads(json.dumps(JSON_BODY))
    body["output"][0]["content"][0]["text"] = f"not json: {USER_TEXT}"
    transport, _ = make(Recorder(ok(body)))
    with pytest.raises(ModelResponseError) as caught:
        await run(transport)
    assert USER_TEXT not in str(caught.value) and "not json" not in str(caught.value)


# --------------------------------------------------------------------------- #
# Retries and errors: the library's rules
# --------------------------------------------------------------------------- #


async def test_a_429_is_retried_and_honours_retry_after() -> None:
    rec = Recorder(error(429, "rate_limit_exceeded", **{"Retry-After": "3"}), ok())
    transport, _ = make(rec)
    reply = await run(transport)
    assert len(rec.requests) == 2 and rec.delays == [3.0]
    assert reply.retries == 1 and reply.body["parsed"] == {"city": "Berlin"}


@pytest.mark.parametrize(
    ("headers", "wait"),
    [
        ({"x-ratelimit-reset-requests": "20ms"}, 0.02),
        ({"x-ratelimit-reset-tokens": "6m0s"}, 360.0),
        (
            {  # the exhausted resource decides
                "x-ratelimit-reset-requests": "1s",
                "x-ratelimit-reset-tokens": "2s",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-remaining-tokens": "100",
            },
            1.0,
        ),
        ({"x-ratelimit-reset-requests": "1.5s", "Retry-After": "7"}, 7.0),
    ],
)
async def test_a_429_reads_the_ratelimit_reset_hints(
    headers: dict[str, str], wait: float
) -> None:
    rec = Recorder(error(429, "rate_limit_exceeded", **headers), ok())
    transport, _ = make(rec, max_retry_after=1000.0, call_deadline=None)
    await run(transport)
    assert rec.delays == [pytest.approx(wait)]


async def test_a_429_without_a_hint_backs_off_and_is_retried() -> None:
    rec = Recorder(error(429, "rate_limit_exceeded"), error(429), ok())
    transport, _ = make(rec)
    reply = await run(transport)
    assert len(rec.requests) == 3 and reply.retries == 2


async def test_a_429_that_never_clears_is_a_rate_limit_error() -> None:
    rec = Recorder(error(429, "rate_limit_exceeded", **{"Retry-After": "1"}))
    transport, _ = make(rec, max_retries=2)
    with pytest.raises(RateLimitError) as caught:
        await run(transport)
    assert len(rec.requests) == 3
    assert classify_failure(caught.value) == "unavailable"  # nothing ran


async def test_an_exhausted_quota_is_not_retried() -> None:
    rec = Recorder(error(429, "insufficient_quota", "insufficient_quota"), ok())
    transport, _ = make(rec)
    with pytest.raises(RateLimitError, match="insufficient_quota"):
        await run(transport)
    assert len(rec.requests) == 1


async def test_a_retry_after_beyond_the_cap_fails_at_once() -> None:
    rec = Recorder(error(429, "rate_limit_exceeded", **{"Retry-After": "600"}), ok())
    transport, _ = make(rec)
    with pytest.raises(RateLimitError):
        await run(transport)
    assert len(rec.requests) == 1 and rec.delays == []


async def test_a_connect_error_is_retried_because_nothing_was_sent() -> None:
    rec = Recorder(httpx.ConnectError("refused"), ok())
    transport, _ = make(rec)
    reply = await run(transport)
    assert len(rec.requests) == 2 and reply.retries == 1


@pytest.mark.parametrize("status", [500, 502, 503, 529, 408])
async def test_a_5xx_or_408_is_uncertain_after_exactly_one_request(
    status: int,
) -> None:
    rec = Recorder(error(status, "server_error", "server_error"), ok())
    transport, _ = make(rec)
    with pytest.raises(UncertainOutcomeError) as caught:
        await run(transport)
    assert len(rec.requests) == 1 and rec.delays == []
    assert caught.value.status_code == status
    assert classify_failure(caught.value) == "uncertain"
    assert "may have charged" in str(caught.value)


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("slow"),
        httpx.WriteTimeout("slow"),
        httpx.RemoteProtocolError("dropped"),
        httpx.ReadError("reset"),
    ],
)
async def test_a_timeout_or_a_dropped_connection_is_uncertain_and_never_resent(
    failure: Exception,
) -> None:
    rec = Recorder(failure, ok())
    transport, _ = make(rec)
    with pytest.raises(UncertainOutcomeError) as caught:
        await run(transport)
    assert len(rec.requests) == 1 and rec.delays == []
    assert caught.value.retries == 0


async def test_a_timeout_at_a_high_effort_is_still_uncertain_and_never_resent() -> None:
    rec = Recorder(httpx.ReadTimeout("thinking too long"), ok())
    transport, _ = make(rec, reasoning_effort="max")
    with pytest.raises(UncertainOutcomeError):
        await run(transport)
    assert len(rec.requests) == 1


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_are_auth_errors(status: int) -> None:
    rec = Recorder(
        error(
            status,
            "invalid_api_key",
            "invalid_request_error",
            "Incorrect API key: " + KEY,
        )
    )
    transport, _ = make(rec)
    with pytest.raises(AuthError) as caught:
        await run(transport)
    assert len(rec.requests) == 1
    assert KEY not in str(caught.value) and KEY not in json.dumps(caught.value.body)
    assert classify_failure(caught.value) is None  # never falls back


async def test_a_4xx_is_a_bad_request_with_type_and_code_but_no_user_text() -> None:
    rec = Recorder(
        httpx.Response(
            400,
            json={
                "error": {
                    "message": f"Invalid input: {USER_TEXT}",
                    "type": "invalid_request_error",
                    "param": "text.format.schema",
                    "code": "invalid_json_schema",
                }
            },
            headers={"x-request-id": "req_abc123"},
        )
    )
    transport, _ = make(rec)
    with pytest.raises(BadRequestError) as caught:
        await run(transport)
    exc = caught.value
    text = str(exc) + json.dumps(exc.body) + json.dumps(exc.cf_errors)
    assert "invalid_request_error" in str(exc) and "invalid_json_schema" in str(exc)
    assert exc.cf_errors == [
        {
            "type": "invalid_request_error",
            "code": "invalid_json_schema",
            "param": "text.format.schema",
        }
    ]
    assert "req_abc123" in str(exc) and exc.request_id == "req_abc123"
    for fragment in ("Dana", "123456", "Invalid input", USER_TEXT):
        assert fragment not in text
    assert classify_failure(exc) is None
    assert len(rec.requests) == 1


async def test_an_error_object_with_hostile_identifiers_is_not_echoed() -> None:
    rec = Recorder(
        httpx.Response(
            400,
            json={
                "error": {
                    "type": f"see {USER_TEXT}",
                    "code": USER_TEXT,
                    "param": USER_TEXT,
                    "message": USER_TEXT,
                }
            },
        )
    )
    transport, _ = make(rec)
    with pytest.raises(BadRequestError) as caught:
        await run(transport)
    assert "Dana" not in str(caught.value) + json.dumps(caught.value.body)


async def test_a_non_openai_model_is_refused_before_anything_is_sent() -> None:
    rec = Recorder(ok())
    transport, _ = make(rec)
    with pytest.raises(BadRequestError):
        await transport.run("@cf/cloudflare/clef", PAYLOAD, meta=META)
    assert rec.requests == []
    assert transport.serves("gpt-6-luna") and not transport.serves("clairic")
    assert not transport.serves("@cf/cloudflare/clef")


def test_the_limits_and_the_key_it_holds_are_exposed() -> None:
    transport = OpenAIResponsesTransport(api_key=KEY)
    assert 1 <= transport.max_questions <= 16  # a small number
    assert transport.state_token_budget >= 1
    assert transport.sensitive_values() == (KEY,)
    assert OpenAIResponsesTransport().sensitive_values() == ()
    assert transport.provider == "openai" and transport.model == "gpt-6-luna"
    with pytest.raises(ValueError, match="state_token_budget"):
        OpenAIResponsesTransport(state_token_budget=0)
    assert (
        transport.classify_failure(TransportError("never connected")) == "unavailable"
    )


# --------------------------------------------------------------------------- #
# Reasoning effort
# --------------------------------------------------------------------------- #


def test_the_efforts_are_the_six_levels_and_default_to_none() -> None:
    assert REASONING_EFFORTS == ("none", "low", "medium", "high", "xhigh", "max")
    assert OpenAIResponsesTransport().reasoning_effort == "none"
    for effort in REASONING_EFFORTS:
        assert (
            OpenAIResponsesTransport(reasoning_effort=effort).reasoning_effort == effort
        )
    with pytest.raises(ValueError, match="reasoning_effort"):
        OpenAIResponsesTransport(reasoning_effort="extreme")


async def test_a_bad_call_effort_is_refused_before_sending() -> None:
    rec = Recorder(ok())
    transport, _ = make(rec)
    with pytest.raises(ValueError, match="reasoning_effort"):
        await run(transport, {**PAYLOAD, "reasoning_effort": "extreme"})
    assert rec.requests == []


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
async def test_the_per_call_timeout_follows_the_effort(effort: str) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return ok()

    transport = OpenAIResponsesTransport(
        api_key=KEY, reasoning_effort=effort, client=None
    )
    # the transport's own client is the one that carries the effort's timeout
    transport._injected_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=transport._timeout
    )
    await run(transport)
    assert seen[0]["read"] == EFFORT_TIMEOUTS[effort]
    # a call at a higher effort than the transport's gets the longer timeout
    await run(transport, {**PAYLOAD, "reasoning_effort": "max"})
    assert seen[1]["read"] >= EFFORT_TIMEOUTS[effort]


def test_the_effort_timeouts_grow_and_can_be_overridden() -> None:
    values = [EFFORT_TIMEOUTS[e] for e in REASONING_EFFORTS]
    assert values == sorted(values) and values[0] == 60.0
    own = OpenAIResponsesTransport(
        reasoning_effort="high", timeout=httpx.Timeout(42.0, connect=3.0)
    )
    assert own._timeout.read == 42.0 and own._timeout.connect == 3.0
    assert OpenAIResponsesTransport(reasoning_effort="max")._call_deadline == max(
        300.0, 2.5 * EFFORT_TIMEOUTS["max"]
    )
    assert OpenAIResponsesTransport(call_deadline=None)._call_deadline is None
    assert OpenAIResponsesTransport(call_deadline=90)._call_deadline == 90.0


# --------------------------------------------------------------------------- #
# The key never leaves, and nothing routes here unless configured
# --------------------------------------------------------------------------- #


async def test_recording_never_writes_the_key(tmp_path: Path) -> None:
    rec = Recorder(ok())
    transport, _ = make(rec)
    path = tmp_path / "cassette.jsonl"
    recorder = RecordingTransport(transport, path)
    await run_via(recorder)
    text = path.read_text()
    assert KEY not in text and "Authorization" not in text
    assert json.loads(text.splitlines()[0])["provider"] == "openai"
    # a call whose request or response HOLDS the key is not written at all
    held = {**PAYLOAD, "input": f"the key is {KEY}"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await recorder.run("gpt-6-luna", held, meta=META)
    assert KEY not in path.read_text()
    assert any("holds a credential" in str(w.message) for w in caught)


async def run_via(transport: Any) -> Any:
    return await transport.run("gpt-6-luna", PAYLOAD, meta=META)


def test_the_openai_key_shapes_are_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert KEY not in scrub(f"error talking to openai with {KEY} today")
    assert "sk-abcdefghijklmnopqrstuv" not in scrub("key sk-abcdefghijklmnopqrstuv.")
    assert scrub("task-list sk-short") == "task-list sk-short"  # not a key
    monkeypatch.setenv("OPENAI_API_KEY", "opaque-key-without-the-prefix-123456")
    assert "opaque-key-without" not in scrub(
        "leaked opaque-key-without-the-prefix-123456"
    )


def test_a_key_in_the_environment_alone_routes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    assert escalation_from_env() is None
    assert (
        getattr(system_one_transport(provider="cloudflare"), "escalation", None) is None
    )
    extractor = create_extractor([_Doc])
    assert extractor.core.escalation(None) is None
    assert extractor.core.escalation(extractor.escalation) is None


def test_the_explicit_choices_turn_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "openai")
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE_EFFORT", "low")
    config = escalation_from_env()
    assert config is not None and config.enabled and config.reasoning_effort == "low"
    core = create_extractor([_Doc]).core
    chosen = core.escalation(None)
    assert chosen is not None and isinstance(chosen.transport, OpenAIResponsesTransport)
    assert chosen.transport.reasoning_effort == "low"
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "gemini")
    with pytest.raises(ValueError, match="TRUSTCALL_S1_ESCALATE"):
        escalation_from_env()
    with pytest.raises(ValueError, match="TRUSTCALL_S1_ESCALATE"):
        create_extractor([_Doc])  # a bad value fails when the extractor is made
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "openai")
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE_EFFORT", "extreme")
    with pytest.raises(ValueError, match="reasoning_effort"):
        escalation_from_env()


def test_system_one_transport_carries_the_choice_and_refuses_typesafe() -> None:
    transport = system_one_transport(provider="cloudflare", escalate="openai")
    config = getattr(transport, "escalation", None)
    assert config is not None and config.enabled and config.transport is None
    with pytest.raises(ValueError, match="TypeSafe only"):
        system_one_transport(provider="typesafe", escalate="openai")
    with pytest.raises(ValueError, match="never switched on for you"):
        system_one_transport(provider="cloudflare", openai=OpenAIResponsesTransport())
    with pytest.raises(ValueError, match="unknown escalate"):
        system_one_transport(provider="cloudflare", escalate="gemini")  # type: ignore[arg-type]
    plain = system_one_transport(provider="cloudflare", escalate="none")
    assert getattr(plain, "escalation", None) is None
