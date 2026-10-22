"""ExperientialTransport, RoutingTransport and the provider selection, offline.

Every request goes to ``httpx.MockTransport``. The test key below is made up.

What is known about the live API comes from two probes. A request without a key got
a 401 whose body is ``{"error": {"message", "type", "param", "code"}}`` and which
carries an ``x-correlation-id`` header. A valid request got a 200 whose body is the
bare Clef response plus ``id``, ``provider`` and ``usage.cost``/``usage.is_byok``
(``docs/schemas/experiential-clef-response.example.json``, used as ``BODY``). The
``{success, result}`` / ``{data}`` envelopes are defensive and built around it.
"""

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from evals.cases import CASES
from evals.cli import main as evals_main
from evals.runner import Settings, run_suite
from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CLEF_FLASH,
    AuthError,
    BadRequestError,
    CallMeta,
    CloudflareTransport,
    ExperientialTransport,
    ModelResponseError,
    RateLimitError,
    RoutingTransport,
    Transport,
    TransportError,
    TransportResponse,
    TrustcallSystemOneError,
    UncertainOutcomeError,
    clef_provider_from_env,
    create_extractor,
    system_one_transport,
)
from trustcall_system_one._wire import ChoiceAnswer, parse_clef
from trustcall_system_one.testing import (
    RecordingTransport,
    scrub,
)

KEY = "xpl_" + "ab12cd34" * 5
OTHER_KEY = "xpl_" + "ef56ab78" * 5
URL = "https://api.experientiallabs.ai/v1/systemone"
META = CallMeta(
    kind="attribution", schema="Person", entity="Ada Lovelace", field="name"
)
PAYLOAD: dict[str, Any] = {
    "model": "clef",
    "state": {
        "task": "Review a proposed change.",
        "diff_summary": "Reject an empty project name.",
    },
    "questions": {
        "review": {
            "type": "choice",
            "instructions": "Accept or review?",
            "criteria": {"accept": "It is fine.", "review": "Evidence is missing."},
        }
    },
}
BODY: dict[str, Any] = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "docs/schemas/experiential-clef-response.example.json"
    ).read_text()
)
OBSERVED_401 = {
    "error": {
        "message": "A valid gateway Bearer key is required.",
        "type": "authentication_error",
        "param": None,
        "code": "invalid_key",
    }
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
        "EXPERIENTIAL_API_KEY",
        "TRUSTCALL_S1_CLEF_PROVIDER",
        "TRUSTCALL_S1_BASE_URL",
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


def make(
    *responses: httpx.Response | Exception, **kwargs: Any
) -> tuple[ExperientialTransport, Recorder]:
    recorder = Recorder(*responses)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    kwargs.setdefault("api_key", KEY)
    transport = ExperientialTransport(client=client, sleep=recorder.sleep, **kwargs)
    return transport, recorder


def ok(body: Any = BODY) -> httpx.Response:
    return httpx.Response(200, json=body)


async def call(transport: Transport, model: str = CLEF) -> TransportResponse:
    return await transport.run(model, PAYLOAD, meta=META)


# --------------------------------------------------------------------------- #
# Requests and responses
# --------------------------------------------------------------------------- #


async def test_one_post_with_the_bearer_key_and_the_unchanged_payload() -> None:
    transport, recorder = make(ok())
    result = await call(transport)
    (request,) = recorder.requests
    assert request.method == "POST"
    assert str(request.url) == URL
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"].startswith("trustcall-system-one/")
    assert json.loads(request.content) == PAYLOAD
    assert result.body == BODY
    assert (result.retries, result.status_code) == (0, 200)
    assert result.duration_ms >= 0
    parse_clef(result.body)  # the library's own Clef parsing accepts it


async def test_the_observed_live_response_is_returned_bare_with_its_extra_keys() -> (
    None
):
    assert set(BODY) == {"id", "model", "answers", "usage", "provider"}
    transport, _ = make(
        httpx.Response(
            200,
            json=BODY,
            headers={"x-request-id": "r-1", "x-correlation-id": "c-1"},
        )
    )
    result = await call(transport)
    assert result.body == BODY
    answer = parse_clef(result.body).answers["review"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "accept"
    assert set(answer.probabilities) == {"accept", "review"}
    usage = parse_clef(result.body).usage
    assert (usage.input_tokens, usage.output_tokens) == (221, 0)


def test_the_observed_live_response_matches_the_published_clef_schema() -> None:
    from tests.unit.schema_helpers import validator

    validator("clef-output").validate(BODY)


async def test_nothing_about_the_call_is_sent_but_the_payload() -> None:
    transport, recorder = make(ok())
    await call(transport)
    names = {name.lower() for name in recorder.requests[0].headers}
    assert not any(name.startswith("cf-aig") for name in names)
    assert "Ada Lovelace" not in recorder.requests[0].content.decode()


async def test_base_url_is_configurable() -> None:
    transport, recorder = make(ok(), base_url="https://example.test/v9/system/")
    await call(transport)
    assert str(recorder.requests[0].url) == "https://example.test/v9/system"


async def test_the_key_comes_from_the_environment_and_the_argument_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", f"  {OTHER_KEY}\n")
    transport, recorder = make(ok(), api_key=None)
    await call(transport)
    assert recorder.requests[0].headers["Authorization"] == f"Bearer {OTHER_KEY}"
    explicit, recorder = make(ok())
    await call(explicit)
    assert recorder.requests[0].headers["Authorization"] == f"Bearer {KEY}"


async def test_a_missing_key_is_an_auth_error_and_nothing_is_sent() -> None:
    transport, recorder = make(ok(), api_key=None)
    with pytest.raises(AuthError, match="EXPERIENTIAL_API_KEY"):
        await call(transport)
    assert recorder.requests == []


@pytest.mark.parametrize("model", [CLEF, CLEF_FLASH, "clef", "clef-flash"])
async def test_clef_models_are_accepted(model: str) -> None:
    transport, recorder = make(ok())
    await call(transport, model)
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("model", [CLAIRIC, "clairic", "gpt-4"])
async def test_other_models_are_refused_before_sending(model: str) -> None:
    transport, recorder = make(ok())
    with pytest.raises(BadRequestError, match="Clef only"):
        await call(transport, model)
    assert recorder.requests == []


@pytest.mark.parametrize("key", ["result", "data"])
async def test_an_envelope_is_unwrapped_into_the_bare_body(key: str) -> None:
    envelope = {"success": True, key: BODY, "errors": [], "messages": []}
    transport, _ = make(ok(envelope))
    assert (await call(transport)).body == BODY


async def test_a_bare_body_is_returned_as_it_is_with_extra_keys() -> None:
    body = {**BODY, "request_id": "r-1", "usage": {**BODY["usage"], "cached": 3}}
    transport, _ = make(ok(body))
    assert (await call(transport)).body == body


@pytest.mark.parametrize(
    "body",
    [
        {"success": False, "errors": [{"code": 1, "message": "no credit"}]},
        {"error": {"message": "no credit", "code": "payment_required"}},
    ],
)
async def test_a_failure_reported_inside_a_200_is_a_bad_request(body: Any) -> None:
    transport, _ = make(ok(body))
    with pytest.raises(BadRequestError, match="no credit"):
        await call(transport)


@pytest.mark.parametrize(
    "body",
    [
        {"model": "clef", "answers": {}},  # no usage
        {"model": "clef", "usage": {"input_tokens": 1, "output_tokens": 1}},
        {**BODY, "answers": {"review": {"type": "choice", "choice": "x"}}},
        {"something": "else"},
        ["a", "list"],
    ],
)
async def test_a_2xx_that_is_not_a_clef_response_is_a_model_response_error(
    body: Any,
) -> None:
    transport, recorder = make(ok(body))
    with pytest.raises(ModelResponseError):
        await call(transport)
    assert len(recorder.requests) == 1


async def test_a_non_json_2xx_is_a_model_response_error() -> None:
    transport, recorder = make(httpx.Response(200, text="<html>hi</html>"))
    with pytest.raises(ModelResponseError, match="non-JSON"):
        await call(transport)
    assert len(recorder.requests) == 1


async def test_a_request_the_library_builds_gets_an_answer_it_parses() -> None:
    from trustcall_system_one._wire import ClefRequest

    request = ClefRequest.model_validate(PAYLOAD)
    transport, recorder = make(ok())
    result = await transport.run(CLEF, request.to_payload(), meta=META)
    assert recorder.requests[0].content
    assert parse_clef(result.body).answers["review"].type == "choice"


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


async def test_the_observed_401_is_an_auth_error_with_the_correlation_id() -> None:
    response = httpx.Response(
        401, json=OBSERVED_401, headers={"x-correlation-id": "corr-123"}
    )
    transport, recorder = make(response)
    with pytest.raises(AuthError) as info:
        await call(transport)
    error = info.value
    assert error.status_code == 401
    assert error.body == OBSERVED_401
    assert "valid gateway Bearer key is required" in str(error)
    assert "corr-123" in str(error)
    assert len(recorder.requests) == 1  # a refusal is not retried either


async def test_403_is_an_auth_error() -> None:
    transport, recorder = make(httpx.Response(403, json={"error": "forbidden"}))
    with pytest.raises(AuthError, match="forbidden"):
        await call(transport)
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("status", [400, 402, 404, 413, 422])
async def test_other_4xx_are_bad_requests_with_the_error_object(status: int) -> None:
    body = {"error": {"message": "bad question", "type": "invalid_request_error"}}
    transport, recorder = make(httpx.Response(status, json=body))
    with pytest.raises(BadRequestError) as info:
        await call(transport)
    assert info.value.status_code == status
    assert info.value.cf_errors == [body["error"]]
    assert "bad question" in str(info.value)
    assert len(recorder.requests) == 1


async def test_a_redirect_is_a_bad_request() -> None:
    transport, recorder = make(httpx.Response(301, headers={"Location": "/x"}))
    with pytest.raises(BadRequestError):
        await call(transport)
    assert len(recorder.requests) == 1


# --------------------------------------------------------------------------- #
# No retry once the request may have arrived
# --------------------------------------------------------------------------- #

POST_SEND_ERRORS = [
    httpx.ReadTimeout("read timed out"),
    httpx.WriteTimeout("write timed out"),
    httpx.ReadError("connection reset"),
    httpx.WriteError("broken pipe"),
    httpx.RemoteProtocolError("server disconnected without a response"),
    httpx.CloseError("closed"),
]


@pytest.mark.parametrize("error", POST_SEND_ERRORS, ids=lambda e: type(e).__name__)
async def test_a_failure_after_sending_is_one_attempt_and_an_uncertain_outcome(
    error: Exception,
) -> None:
    transport, recorder = make(error, ok())
    with pytest.raises(UncertainOutcomeError) as info:
        await call(transport)
    assert len(recorder.requests) == 1
    assert recorder.delays == []
    assert info.value.__cause__ is error
    assert info.value.retries == 0
    message = str(info.value)
    assert "outcome is unknown" in message
    assert "charged" in message
    assert "not retried" in message
    assert transport.limiter.in_flight == 0


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
async def test_a_5xx_or_408_is_one_attempt_and_an_uncertain_outcome(
    status: int,
) -> None:
    response = httpx.Response(
        status,
        json={"error": {"message": "upstream exploded"}},
        headers={"x-correlation-id": "corr-9", "Retry-After": "1"},
    )
    transport, recorder = make(response, ok())
    with pytest.raises(UncertainOutcomeError) as info:
        await call(transport)
    error = info.value
    assert len(recorder.requests) == 1
    assert recorder.delays == []
    assert error.status_code == status
    assert error.body == {"error": {"message": "upstream exploded"}}
    assert "upstream exploded" in str(error)
    assert "corr-9" in str(error)
    assert "charged" in str(error)
    assert isinstance(error, TransportError)
    assert isinstance(error, TrustcallSystemOneError)
    assert transport.limiter.in_flight == 0


async def test_uncertain_outcomes_ignore_max_retries() -> None:
    transport, recorder = make(httpx.ReadTimeout("slow"), ok(), max_retries=50)
    with pytest.raises(UncertainOutcomeError):
        await call(transport)
    assert len(recorder.requests) == 1


async def test_a_read_timeout_with_a_per_call_timeout_is_uncertain() -> None:
    class Slow(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("no answer in time", request=request)

    client = httpx.AsyncClient(transport=Slow())
    transport = ExperientialTransport(api_key=KEY, client=client)
    with pytest.raises(UncertainOutcomeError):
        await transport.run(CLEF, PAYLOAD, meta=META, timeout=0.05)


# --------------------------------------------------------------------------- #
# Retries that are safe
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ConnectTimeout("handshake timed out")],
    ids=lambda e: type(e).__name__,
)
async def test_a_connect_error_before_sending_is_retried(error: Exception) -> None:
    transport, recorder = make(error, error, ok())
    result = await call(transport)
    assert result.body == BODY
    assert result.retries == 2
    assert len(recorder.requests) == 3
    assert len(recorder.delays) == 2
    assert transport.limiter.in_flight == 0


async def test_connect_retries_are_bounded_and_end_in_a_plain_transport_error() -> None:
    transport, recorder = make(httpx.ConnectError("refused"), max_retries=2)
    with pytest.raises(TransportError) as info:
        await call(transport)
    error = info.value
    assert not isinstance(error, UncertainOutcomeError)  # nothing was sent
    assert "not sent" in str(error)
    assert error.retries == 2
    assert len(recorder.requests) == 3


async def test_a_connect_error_then_a_timeout_stops_at_the_timeout() -> None:
    transport, recorder = make(
        httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), ok()
    )
    with pytest.raises(UncertainOutcomeError) as info:
        await call(transport)
    assert len(recorder.requests) == 2
    assert info.value.retries == 1


@pytest.mark.parametrize(
    "error",
    [httpx.PoolTimeout("no connection free"), httpx.ProxyError("proxy refused")],
    ids=lambda e: type(e).__name__,
)
async def test_failures_before_sending_that_are_not_connect_errors_are_not_retried(
    error: Exception,
) -> None:
    transport, recorder = make(error, ok())
    with pytest.raises(TransportError) as info:
        await call(transport)
    assert not isinstance(info.value, UncertainOutcomeError)
    assert len(recorder.requests) == 1


async def test_429_with_retry_after_is_retried_after_the_wait() -> None:
    limited = httpx.Response(
        429, json={"error": {"message": "slow down"}}, headers={"Retry-After": "3"}
    )
    transport, recorder = make(limited, ok())
    result = await call(transport)
    assert result.retries == 1
    assert recorder.delays == [3.0]
    assert len(recorder.requests) == 2
    assert transport.limiter.limit(CLEF) == 16  # halved from 32


async def test_429_without_retry_after_is_not_retried() -> None:
    transport, recorder = make(httpx.Response(429, json={"error": "slow"}), ok())
    with pytest.raises(RateLimitError) as info:
        await call(transport)
    assert len(recorder.requests) == 1
    assert info.value.retry_after is None
    assert info.value.status_code == 429


async def test_429_with_a_retry_after_beyond_the_bound_fails_at_once() -> None:
    limited = httpx.Response(429, json={"error": "slow"}, headers={"Retry-After": "90"})
    transport, recorder = make(limited, ok())
    with pytest.raises(RateLimitError) as info:
        await call(transport)
    assert info.value.retry_after == 90.0
    assert len(recorder.requests) == 1
    assert recorder.delays == []


async def test_429_retries_are_bounded() -> None:
    limited = httpx.Response(429, json={"error": "slow"}, headers={"Retry-After": "1"})
    transport, recorder = make(limited, max_retries=2)
    with pytest.raises(RateLimitError) as info:
        await call(transport)
    assert info.value.retries == 2
    assert len(recorder.requests) == 3


async def test_a_5xx_after_a_429_is_still_not_retried() -> None:
    limited = httpx.Response(429, json={"error": "slow"}, headers={"Retry-After": "1"})
    transport, recorder = make(
        limited, httpx.Response(503, json={"error": "down"}), ok()
    )
    with pytest.raises(UncertainOutcomeError) as info:
        await call(transport)
    assert len(recorder.requests) == 2
    assert info.value.retries == 1


async def test_the_call_deadline_bounds_the_waits() -> None:
    limited = httpx.Response(429, json={"error": "slow"}, headers={"Retry-After": "40"})
    transport, recorder = make(limited, call_deadline=30.0)
    with pytest.raises(TransportError, match="deadline"):
        await call(transport)
    assert len(recorder.requests) == 1


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ExperientialTransport(max_retries=-1)
    with pytest.raises(ValueError, match="backoff"):
        ExperientialTransport(backoff_base=0)
    with pytest.raises(ValueError, match="max_retry_after"):
        ExperientialTransport(max_retry_after=0)


async def test_aclose_closes_the_owned_client_and_context_manager_works() -> None:
    async with ExperientialTransport(api_key=KEY) as transport:
        transport._get_client()
        assert len(transport._clients) == 1
    assert transport._clients == {}


def test_sensitive_values_lists_an_explicit_key_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ExperientialTransport(api_key=KEY).sensitive_values() == (KEY,)
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", OTHER_KEY)
    assert ExperientialTransport().sensitive_values() == ()


def test_uncertain_outcome_error_is_a_transport_error_and_public() -> None:
    assert issubclass(UncertainOutcomeError, TransportError)
    error = UncertainOutcomeError("x", status_code=504)
    assert error.status_code == 504
    assert not issubclass(UncertainOutcomeError, (RateLimitError, AuthError))


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


class Tag:
    """A transport that answers with its name and keeps what it was given."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, Mapping[str, Any], CallMeta, float | None]] = []
        self.closed: list[str] = []

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.calls.append((model, payload, meta, timeout))
        return TransportResponse(body={"served_by": self.name}, status_code=200)

    async def aclose(self) -> None:
        self.closed.append("aclose")

    def close(self) -> None:
        self.closed.append("close")


async def test_routing_picks_the_transport_by_model_family() -> None:
    clef, clairic = Tag("experiential"), Tag("cloudflare")
    router = RoutingTransport({"clef": clef}, default=clairic)
    assert (await call(router, CLEF)).body == {"served_by": "experiential"}
    assert (await call(router, CLEF_FLASH)).body == {"served_by": "experiential"}
    assert (await call(router, "clef")).body == {"served_by": "experiential"}
    assert (await call(router, CLAIRIC)).body == {"served_by": "cloudflare"}
    assert (await call(router, "something-else")).body == {"served_by": "cloudflare"}
    assert len(clef.calls) == 3
    assert len(clairic.calls) == 2


async def test_routing_passes_model_payload_meta_and_timeout_through() -> None:
    clef, default = Tag("clef"), Tag("default")
    router = RoutingTransport({"clef": clef}, default=default)
    await router.run(CLEF, PAYLOAD, meta=META, timeout=12.5)
    ((model, payload, meta, timeout),) = clef.calls
    assert (model, timeout) == (CLEF, 12.5)
    assert payload is PAYLOAD
    assert meta is META
    assert default.calls == []


async def test_an_exact_model_route_beats_its_family() -> None:
    family, flash, default = Tag("family"), Tag("flash"), Tag("default")
    router = RoutingTransport(
        {"clef": family, "@cf/cloudflare/clef-flash": flash}, default=default
    )
    assert (await call(router, CLEF)).body == {"served_by": "family"}
    assert (await call(router, CLEF_FLASH)).body == {"served_by": "flash"}
    assert (await call(router, CLAIRIC)).body == {"served_by": "default"}


async def test_clairic_can_be_routed_too_and_clef_left_on_the_default() -> None:
    clairic, default = Tag("clairic"), Tag("default")
    router = RoutingTransport({CLAIRIC: clairic}, default=default)
    assert (await call(router, CLAIRIC)).body == {"served_by": "clairic"}
    assert (await call(router, CLEF)).body == {"served_by": "default"}


def test_routing_rejects_unknown_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="cannot route"):
        RoutingTransport({"gpt-4": Tag("x")}, default=Tag("y"))
    with pytest.raises(ValueError, match="more than one route"):
        RoutingTransport({"clef": Tag("x"), CLEF: Tag("y")}, default=Tag("z"))


def test_routing_exposes_the_sensitive_values_of_its_members() -> None:
    router = RoutingTransport(
        {"clef": ExperientialTransport(api_key=KEY)},
        default=CloudflareTransport(account_id="acct-1", api_token="tok-1"),
    )
    assert set(router.sensitive_values()) == {KEY, "acct-1", "tok-1"}
    assert RoutingTransport({}, default=Tag("x")).sensitive_values() == ()


async def test_routing_closes_each_member_once() -> None:
    shared, other = Tag("shared"), Tag("other")
    router = RoutingTransport({"clef": shared, "clairic": shared}, default=other)
    await router.aclose()
    router.close()
    assert shared.closed == ["aclose", "close"]
    assert other.closed == ["aclose", "close"]
    async with RoutingTransport({}, default=other):
        pass
    assert other.closed[-1] == "aclose"


async def test_routing_to_experiential_end_to_end_keeps_meta_local() -> None:
    cloudflare = Tag("cloudflare")
    experiential, recorder = make(ok())
    router = RoutingTransport({"clef": experiential}, default=cloudflare)
    result = await call(router, CLEF)
    assert parse_clef(result.body).model == "clef"
    assert len(recorder.requests) == 1
    assert cloudflare.calls == []


# --------------------------------------------------------------------------- #
# Selecting the provider from the environment
# --------------------------------------------------------------------------- #


def test_provider_defaults_to_cloudflare() -> None:
    assert clef_provider_from_env({}) == "cloudflare"
    assert clef_provider_from_env({"TRUSTCALL_S1_CLEF_PROVIDER": ""}) == "cloudflare"
    assert clef_provider_from_env({"EXPERIENTIAL_API_KEY": "  "}) == "cloudflare"


def test_a_key_alone_never_selects_experiential() -> None:
    # sending the text to another company is an explicit choice
    assert clef_provider_from_env({"EXPERIENTIAL_API_KEY": KEY}) == "cloudflare"
    assert (
        clef_provider_from_env(
            {"EXPERIENTIAL_API_KEY": KEY, "TRUSTCALL_S1_CLEF_PROVIDER": ""}
        )
        == "cloudflare"
    )


def test_the_variable_beats_the_key_in_both_directions() -> None:
    env = {"EXPERIENTIAL_API_KEY": KEY, "TRUSTCALL_S1_CLEF_PROVIDER": "Cloudflare"}
    assert clef_provider_from_env(env) == "cloudflare"
    assert (
        clef_provider_from_env({"TRUSTCALL_S1_CLEF_PROVIDER": " EXPERIENTIAL "})
        == "experiential"
    )


def test_an_unknown_provider_is_an_error() -> None:
    with pytest.raises(ValueError, match="TRUSTCALL_S1_CLEF_PROVIDER"):
        clef_provider_from_env({"TRUSTCALL_S1_CLEF_PROVIDER": "openai"})


def test_provider_reads_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert clef_provider_from_env() == "cloudflare"
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    assert clef_provider_from_env() == "cloudflare"
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "experiential")
    assert clef_provider_from_env() == "experiential"


def test_the_factory_builds_a_plain_cloudflare_transport_or_a_router() -> None:
    plain = system_one_transport("cloudflare")
    assert isinstance(plain, CloudflareTransport)
    router = system_one_transport("experiential")
    assert isinstance(router, RoutingTransport)
    assert isinstance(router.transport_for(CLEF), ExperientialTransport)
    assert isinstance(router.transport_for(CLEF_FLASH), ExperientialTransport)
    assert isinstance(router.transport_for(CLAIRIC), CloudflareTransport)
    with pytest.raises(ValueError, match="unknown Clef provider"):
        system_one_transport("openai")  # type: ignore[arg-type]


def test_the_factory_uses_the_environment_and_the_transports_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(system_one_transport(), CloudflareTransport)
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "experiential")
    cloudflare = CloudflareTransport(account_id="a", api_token="t")
    experiential = ExperientialTransport(api_key=KEY)
    router = system_one_transport(cloudflare=cloudflare, experiential=experiential)
    assert isinstance(router, RoutingTransport)
    assert router.transport_for(CLAIRIC) is cloudflare
    assert router.transport_for(CLEF) is experiential


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, CloudflareTransport),
        ({"TRUSTCALL_S1_CLEF_PROVIDER": "cloudflare"}, CloudflareTransport),
        ({"EXPERIENTIAL_API_KEY": KEY}, CloudflareTransport),  # a key alone: no
        (
            {"EXPERIENTIAL_API_KEY": KEY, "TRUSTCALL_S1_CLEF_PROVIDER": "cloudflare"},
            CloudflareTransport,
        ),
        ({"TRUSTCALL_S1_CLEF_PROVIDER": "experiential"}, RoutingTransport),
    ],
)
def test_the_extractors_default_transport_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: type
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    extractor = create_extractor([_Person])
    try:
        assert type(extractor.core.transport()) is expected
        assert extractor.core.transport() is extractor.core.transport()
    finally:
        extractor.close()


class _Person(BaseModel):
    name: str


# --------------------------------------------------------------------------- #
# Scrubbing and recording
# --------------------------------------------------------------------------- #


def test_scrub_removes_the_key_from_the_environment_and_its_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    text = f"failed with {KEY} in 'Authorization': 'Bearer {KEY}' and Bearer {KEY}"
    cleaned = scrub(text)
    assert KEY not in cleaned
    assert "ab12cd34" not in cleaned
    assert "Bearer <redacted>" in cleaned


def test_scrub_removes_an_explicit_key_and_anything_shaped_like_one() -> None:
    cleaned = scrub(
        f"key {OTHER_KEY} and {KEY}",
        extra=ExperientialTransport(api_key=KEY).sensitive_values(),
    )
    assert KEY not in cleaned
    assert OTHER_KEY not in cleaned  # shaped like an Experiential key
    assert scrub("xpl_short is a word") == "xpl_short is a word"


def test_scrub_leaves_ordinary_text_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", "short")  # a word, not a credential
    assert scrub("Clef said short and sweet") == "Clef said short and sweet"


async def test_a_recorded_call_that_holds_the_key_is_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    leaky = {**PAYLOAD, "state": {"note": f"my key is {KEY}"}}
    transport, _ = make(ok())
    path = tmp_path / "cassette.jsonl"
    recording = RecordingTransport(transport, path)
    with pytest.warns(RuntimeWarning, match="credential"):
        await recording.run(CLEF, leaky, meta=META)
    assert recording.written == 0
    assert not path.exists()  # nothing was written, not even the key


async def test_an_explicit_key_is_protected_through_a_router(tmp_path: Path) -> None:
    experiential, _ = make(ok())
    router = RoutingTransport({"clef": experiential}, default=Tag("x"))
    path = tmp_path / "cassette.jsonl"
    recording = RecordingTransport(router, path)
    leaky = {**PAYLOAD, "state": {"note": f"my key is {KEY}"}}
    with pytest.warns(RuntimeWarning, match="credential"):
        await recording.run(CLEF, leaky, meta=META)
    assert recording.written == 0


async def test_a_clean_call_is_recorded_and_an_error_message_is_scrubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    path = tmp_path / "cassette.jsonl"
    good, _ = make(ok())
    await RecordingTransport(good, path).run(CLEF, PAYLOAD, meta=META)
    echo = httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}})
    bad, _ = make(echo)
    with pytest.raises(AuthError):
        await RecordingTransport(bad, path, record_errors=True).run(
            CLEF, PAYLOAD, meta=META
        )
    text = path.read_text()
    assert len(text.splitlines()) == 2
    assert KEY not in text
    assert "<redacted>" in text


async def test_no_warning_for_a_clean_recording(tmp_path: Path) -> None:
    transport, _ = make(ok())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await RecordingTransport(transport, tmp_path / "c.jsonl").run(
            CLEF, PAYLOAD, meta=META
        )


# --------------------------------------------------------------------------- #
# Evals and live tests
# --------------------------------------------------------------------------- #


def test_the_evals_cli_takes_a_clef_provider(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals.cli import _parser, _settings

    args = _parser().parse_args(["run", "--live", "--clef-provider", "experiential"])
    settings = _settings(args, {})
    assert settings.clef_provider == "experiential"
    assert _settings(_parser().parse_args(["run"]), {}).clef_provider is None
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--clef-provider", "openai"])
    # live Experiential without the key stops before any request
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    status = evals_main(
        [
            "run",
            "--live",
            "--clef-provider",
            "experiential",
            "--quiet",
            "--cases",
            CASES[0].id,
        ]
    )
    assert status == 2
    assert "EXPERIENTIAL_API_KEY" in capsys.readouterr().err


async def test_run_suite_live_needs_the_experiential_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    with pytest.raises(RuntimeError, match="EXPERIENTIAL_API_KEY"):
        await run_suite(CASES[:1], Settings(live=True, clef_provider="experiential"))
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "experiential")
    with pytest.raises(RuntimeError, match="EXPERIENTIAL_API_KEY"):
        await run_suite(CASES[:1], Settings(live=True))  # the environment decides


def test_the_live_tests_need_the_experiential_key_only_when_clef_goes_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.live.conftest import credentials_present

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    assert credentials_present()
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    assert credentials_present()  # a key alone does not move Clef
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "experiential")
    assert credentials_present()
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", "  ")
    assert not credentials_present()
    monkeypatch.delenv("EXPERIENTIAL_API_KEY")
    assert not credentials_present()
    monkeypatch.delenv("TRUSTCALL_S1_CLEF_PROVIDER")
    assert credentials_present()
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    assert not credentials_present()
