"""Tests for CloudflareTransport and AdaptiveLimiter without any network."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any, cast

import httpx
import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    AuthError,
    BadRequestError,
    CallMeta,
    CloudflareTransport,
    ConcurrencyConfig,
    ModelResponseError,
    RateLimitError,
    ServerError,
    Transport,
    TransportError,
    TransportResponse,
)
from trustcall_system_one._version import __version__
from trustcall_system_one.transport import AdaptiveLimiter, _parse_retry_after

BODY = {"model": "clairic", "answers": {}, "truncated": False}
META = CallMeta(kind="extract", schema="Person", entity="e1", field="name", scope="s")
PAYLOAD = {"model": "clairic", "state": "x", "fields": {"a": {"type": "text"}}}
ENV_VARS = (
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_AUTH_TOKEN",
    "TRUSTCALL_S1_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class Recorder:
    """A MockTransport handler that records requests and sleeps."""

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
) -> tuple[CloudflareTransport, Recorder]:
    recorder = Recorder(*responses)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    kwargs.setdefault("account_id", "acct")
    kwargs.setdefault("api_token", "tok")
    transport = CloudflareTransport(client=client, sleep=recorder.sleep, **kwargs)
    return transport, recorder


def ok(body: Any = BODY) -> httpx.Response:
    return httpx.Response(200, json=body)


def err(status: int, **kwargs: Any) -> httpx.Response:
    return httpx.Response(status, **kwargs)


async def run(
    transport: CloudflareTransport, model: str = CLAIRIC
) -> TransportResponse:
    return await transport.run(model, PAYLOAD, meta=META)


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


async def test_default_url_headers_and_body() -> None:
    transport, rec = make(ok())
    assert (await run(transport)).body == BODY
    (request,) = rec.requests
    assert request.method == "POST"
    assert str(request.url) == (
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clairic"
    )
    assert request.headers["Authorization"] == "Bearer tok"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"] == f"trustcall-system-one/{__version__}"
    assert "cf-aig-metadata" not in request.headers
    assert json.loads(request.content) == PAYLOAD


async def test_result_carries_retries_duration_and_status() -> None:
    transport, _ = make(err(503), ok())
    result = await run(transport)
    assert isinstance(result, TransportResponse)
    assert result.body == BODY
    assert result.retries == 1
    assert result.status_code == 200
    assert result.duration_ms >= 0


async def test_token_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, rec = make(ok(), api_token="  tok\n")
    await run(transport)
    assert rec.requests[0].headers["Authorization"] == "Bearer tok"
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "envtok\n")
    transport, rec = make(ok(), api_token=None)
    await run(transport)
    assert rec.requests[0].headers["Authorization"] == "Bearer envtok"


async def test_gateway_url_and_metadata() -> None:
    transport, rec = make(ok(), gateway="gw")
    await run(transport, CLEF)
    (request,) = rec.requests
    assert str(request.url) == (
        "https://gateway.ai.cloudflare.com/v1/acct/gw/workers-ai/@cf/cloudflare/clef"
    )
    metadata = json.loads(request.headers["cf-aig-metadata"])
    assert metadata == {
        "kind": "extract",
        "schema": "Person",
        "entity": "e1",
        "field": "name",
        "scope": "s",
    }
    assert len(metadata) <= 5


async def test_gateway_metadata_can_be_disabled() -> None:
    transport, rec = make(ok(), gateway="gw", send_metadata=False)
    await run(transport)
    assert "cf-aig-metadata" not in rec.requests[0].headers


async def test_gateway_metadata_skips_empty_entries() -> None:
    transport, rec = make(ok(), gateway="gw")
    await transport.run(CLAIRIC, PAYLOAD, meta=CallMeta(kind="gate"))
    assert json.loads(rec.requests[0].headers["cf-aig-metadata"]) == {"kind": "gate"}


async def test_base_url_takes_precedence_and_needs_no_account() -> None:
    transport, rec = make(
        ok(), base_url="http://localhost:8787/", gateway="gw", account_id=None
    )
    await run(transport)
    (request,) = rec.requests
    assert str(request.url) == "http://localhost:8787/ai/run/@cf/cloudflare/clairic"
    assert "cf-aig-metadata" in request.headers  # gateway still set the header


async def test_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "envacct")
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "fallback")
    transport, rec = make(ok(), account_id=None, api_token=None)
    await run(transport)
    assert "envacct" in str(rec.requests[0].url)
    assert rec.requests[0].headers["Authorization"] == "Bearer fallback"

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "primary")
    monkeypatch.setenv("TRUSTCALL_S1_BASE_URL", "http://proxy.test/v1")
    transport, rec = make(ok(), account_id=None, api_token=None)
    await run(transport)
    assert (
        str(rec.requests[0].url) == "http://proxy.test/v1/ai/run/@cf/cloudflare/clairic"
    )
    assert rec.requests[0].headers["Authorization"] == "Bearer primary"


async def test_explicit_arguments_beat_the_base_url_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_BASE_URL", "http://env.test")
    transport, rec = make(ok(), gateway="gw")
    await run(transport)
    assert str(rec.requests[0].url).startswith("https://gateway.ai.cloudflare.com/")
    transport, rec = make(ok(), base_url="http://arg.test")
    await run(transport)
    assert str(rec.requests[0].url).startswith("http://arg.test/ai/run/")


async def test_extra_headers_are_sent() -> None:
    transport, rec = make(ok(), extra_headers={"cf-aig-authorization": "Bearer g"})
    await run(transport)
    assert rec.requests[0].headers["cf-aig-authorization"] == "Bearer g"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"backoff_base": 0},
        {"backoff_base": 1.0, "backoff_max": 0.5},
        {"max_retry_after": 0},
        {"max_retry_after": float("nan")},
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CloudflareTransport(**kwargs)


def test_boundary_arguments_are_accepted() -> None:
    CloudflareTransport(max_retries=0, backoff_base=0.5, backoff_max=0.5)


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #


async def test_envelope_is_unwrapped() -> None:
    envelope = {"success": True, "result": BODY, "errors": [], "messages": []}
    transport, _ = make(ok(envelope))
    assert (await run(transport)).body == BODY


async def test_bare_body_is_returned_as_is() -> None:
    transport, _ = make(ok())
    assert (await run(transport)).body == BODY


async def test_non_json_success_raises_model_response_error() -> None:
    transport, _ = make(httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(ModelResponseError) as info:
        await run(transport)
    assert info.value.raw == "<html>nope</html>"


async def test_non_object_success_raises_model_response_error() -> None:
    transport, _ = make(ok([1, 2]))
    with pytest.raises(ModelResponseError):
        await run(transport)


async def test_success_is_counted_only_after_the_body_unwraps() -> None:
    config = ConcurrencyConfig(recover_after=1)
    transport, _ = make(
        err(429), httpx.Response(200, text="not json"), concurrency=config
    )
    with pytest.raises(ModelResponseError):
        await run(transport)
    assert transport.limiter.limit(CLAIRIC) == 16  # a counted success would give 17


async def test_failure_envelope_with_400_raises_bad_request_with_cf_errors() -> None:
    body = {
        "success": False,
        "errors": [{"code": 5006, "message": "bad schema"}],
        "messages": [],
        "result": None,
    }
    transport, rec = make(err(400, json=body))
    with pytest.raises(BadRequestError) as info:
        await run(transport)
    assert info.value.status_code == 400
    assert info.value.body == body
    assert info.value.cf_errors == [{"code": 5006, "message": "bad schema"}]
    assert "bad schema" in str(info.value)
    assert info.value.retries == 0
    assert len(rec.requests) == 1
    assert rec.delays == []


async def test_singular_error_field_is_read() -> None:
    transport, _ = make(err(400, json={"error": "nope"}))
    with pytest.raises(BadRequestError) as info:
        await run(transport)
    assert info.value.cf_errors == [{"message": "nope"}]
    transport, _ = make(err(400, json={"error": {"code": 7, "message": "obj"}}))
    with pytest.raises(BadRequestError) as info:
        await run(transport)
    assert info.value.cf_errors == [{"code": 7, "message": "obj"}]


async def test_failure_envelope_with_200_status_raises_bad_request() -> None:
    body = {
        "success": False,
        "errors": [{"code": 1, "message": "nope"}],
        "result": None,
    }
    transport, _ = make(ok(body))
    with pytest.raises(BadRequestError) as info:
        await run(transport)
    assert info.value.cf_errors == [{"code": 1, "message": "nope"}]


@pytest.mark.parametrize("status", [400, 404, 422, 302, 301])
async def test_client_errors_and_redirects_are_bad_requests_and_not_retried(
    status: int,
) -> None:
    transport, rec = make(err(status, text="nope"))
    with pytest.raises(BadRequestError) as info:
        await run(transport)
    assert info.value.status_code == status
    assert info.value.body == "nope"
    assert info.value.cf_errors == []
    assert len(rec.requests) == 1
    assert rec.delays == []


# --------------------------------------------------------------------------- #
# Retries and errors
# --------------------------------------------------------------------------- #


async def test_429_with_retry_after_then_success() -> None:
    transport, rec = make(err(429, headers={"Retry-After": "2"}, json={}), ok())
    result = await run(transport)
    assert result.body == BODY
    assert rec.delays == [2.0]
    assert len(rec.requests) == 2
    assert transport.limiter.limit(CLAIRIC) == 16  # halved from 32
    assert result.retries == 1
    assert transport.limiter.in_flight == 0


async def test_retry_after_http_date() -> None:
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    header = format_datetime(when, usegmt=True)
    transport, rec = make(err(503, headers={"Retry-After": header}), ok())
    await run(transport)
    (delay,) = rec.delays
    assert 25 <= delay <= 30


def test_retry_after_parsing() -> None:
    assert _parse_retry_after("2") == 2.0
    assert _parse_retry_after("0.5") == 0.5
    assert _parse_retry_after("-3") == 0.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    for bad in ("inf", "-inf", "nan", "infinity", "soon", "Wed, 99 Foo 2026"):
        assert _parse_retry_after(bad) is None, bad
    # An HTTP-date in the past clamps to zero.
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_retry_after_dates_are_utc_even_with_a_minus_zero_offset() -> None:
    when = datetime.now(timezone.utc) + timedelta(seconds=40)
    gmt = format_datetime(when, usegmt=True)
    minus_zero = format_datetime(when.replace(microsecond=0, tzinfo=None))
    assert minus_zero.endswith("-0000")
    for header in (gmt, minus_zero):
        parsed = _parse_retry_after(header)
        assert parsed is not None
        assert 35 <= parsed <= 40


@pytest.mark.parametrize("header", ["inf", "nan", "soon"])
async def test_unusable_retry_after_falls_back_to_jitter(header: str) -> None:
    transport, rec = make(err(503, headers={"Retry-After": header}), ok())
    await run(transport)
    (delay,) = rec.delays
    assert 0.25 <= delay <= 0.75


async def test_retry_after_beyond_the_cap_fails_immediately_for_429() -> None:
    transport, rec = make(err(429, headers={"Retry-After": "61"}), ok())
    with pytest.raises(RateLimitError) as info:
        await run(transport)
    assert info.value.retry_after == 61.0
    assert info.value.retries == 0
    assert len(rec.requests) == 1
    assert rec.delays == []
    assert transport.limiter.in_flight == 0


async def test_retry_after_beyond_the_cap_fails_immediately_for_503() -> None:
    transport, rec = make(err(503, headers={"Retry-After": "3600"}), ok())
    with pytest.raises(ServerError) as info:
        await run(transport)
    assert info.value.retry_after == 3600.0
    assert rec.delays == []
    assert len(rec.requests) == 1


async def test_retry_after_at_the_cap_is_honoured() -> None:
    transport, rec = make(err(503, headers={"Retry-After": "60"}), ok())
    await run(transport)
    assert rec.delays == [60.0]
    transport, rec = make(
        err(503, headers={"Retry-After": "5"}), ok(), max_retry_after=4.0
    )
    with pytest.raises(ServerError):
        await run(transport)


async def test_persistent_503_raises_server_error_with_retry_count() -> None:
    transport, rec = make(
        err(503, json={"errors": [{"message": "down"}]}), max_retries=3
    )
    with pytest.raises(ServerError) as info:
        await run(transport)
    assert info.value.status_code == 503
    assert info.value.body == {"errors": [{"message": "down"}]}
    assert info.value.retries == 3
    assert len(rec.requests) == 4  # first try + max_retries
    assert len(rec.delays) == 3
    assert transport.limiter.in_flight == 0


async def test_default_max_retries_is_six() -> None:
    transport, rec = make(err(500))
    with pytest.raises(ServerError) as info:
        await run(transport)
    assert len(rec.requests) == 7
    assert info.value.retries == 6


async def test_zero_max_retries_never_retries() -> None:
    transport, rec = make(err(503), max_retries=0)
    with pytest.raises(ServerError):
        await run(transport)
    assert len(rec.requests) == 1
    assert rec.delays == []


async def test_persistent_429_raises_rate_limit_error() -> None:
    transport, rec = make(err(429, headers={"Retry-After": "1.5"}), max_retries=2)
    with pytest.raises(RateLimitError) as info:
        await run(transport)
    assert info.value.retry_after == 1.5
    assert info.value.status_code == 429
    assert info.value.retries == 2
    assert len(rec.requests) == 3
    assert rec.delays == [1.5, 1.5]
    assert transport.limiter.in_flight == 0


async def test_sequential_429s_each_halve_once_per_generation() -> None:
    # Each retry acquires its slot under the new generation, so a 429 that
    # persists across attempts keeps halving: 32 -> 16 -> 8 -> 4.
    transport, _ = make(err(429), max_retries=2)
    with pytest.raises(RateLimitError):
        await run(transport)
    assert transport.limiter.limit(CLAIRIC) == 4


async def test_burst_of_concurrent_429s_halves_once() -> None:
    arrived = 0
    everyone_in = asyncio.Event()
    attempts: dict[int, int] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal arrived
        call = json.loads(request.content)["n"]
        attempts[call] = attempts.get(call, 0) + 1
        if attempts[call] == 1:
            arrived += 1
            if arrived == 8:
                everyone_in.set()
            await everyone_in.wait()  # all 8 hold a slot under generation 0
            return err(429, headers={"Retry-After": "0"})
        return ok()

    async def instant(_: float) -> None:
        return None

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = CloudflareTransport(
        account_id="a", api_token="t", client=client, sleep=instant
    )
    results = await asyncio.gather(
        *(transport.run(CLAIRIC, {**PAYLOAD, "n": i}, meta=META) for i in range(8))
    )
    assert [r.retries for r in results] == [1] * 8
    assert transport.limiter.limit(CLAIRIC) == 16  # halved once, not 8 times
    assert transport.limiter.generation(CLAIRIC) == 1
    await client.aclose()


async def test_persistent_408_raises_transport_error() -> None:
    transport, rec = make(err(408), max_retries=1)
    with pytest.raises(TransportError) as info:
        await run(transport)
    assert not isinstance(info.value, (ServerError, RateLimitError))
    assert info.value.status_code == 408
    assert len(rec.requests) == 2


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_errors_are_not_retried(status: int) -> None:
    transport, rec = make(err(status, json={"errors": [{"message": "no"}]}))
    with pytest.raises(AuthError) as info:
        await run(transport)
    assert info.value.status_code == status
    assert info.value.body == {"errors": [{"message": "no"}]}
    assert info.value.retries == 0
    assert len(rec.requests) == 1
    assert rec.delays == []
    assert transport.limiter.in_flight == 0


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("boom"),
        httpx.ReadTimeout("slow"),
        httpx.ConnectTimeout("slow"),
        httpx.ReadError("reset"),
        httpx.WriteError("reset"),
        httpx.RemoteProtocolError("bad framing"),
    ],
)
async def test_transient_network_errors_are_retried(failure: Exception) -> None:
    transport, rec = make(failure, ok())
    result = await run(transport)
    assert result.body == BODY
    assert result.retries == 1
    assert len(rec.requests) == 2
    assert len(rec.delays) == 1


@pytest.mark.parametrize(
    "failure",
    [
        httpx.DecodingError("gzip"),
        httpx.TooManyRedirects("loop"),
        httpx.UnsupportedProtocol("ftp://"),
        httpx.LocalProtocolError("bad header"),
        httpx.ProxyError("407"),
    ],
)
async def test_other_httpx_errors_are_not_retried_but_wrapped(
    failure: Exception,
) -> None:
    transport, rec = make(failure, ok())
    with pytest.raises(TransportError) as info:
        await run(transport)
    assert type(info.value) is TransportError
    assert info.value.__cause__ is failure
    assert info.value.retries == 0
    assert len(rec.requests) == 1
    assert rec.delays == []
    assert transport.limiter.in_flight == 0


async def test_persistent_network_error_raises_transport_error() -> None:
    transport, rec = make(httpx.ReadTimeout("slow"), max_retries=2)
    with pytest.raises(TransportError) as info:
        await run(transport)
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)
    assert info.value.status_code is None
    assert info.value.retries == 2
    assert len(rec.requests) == 3
    assert transport.limiter.in_flight == 0


async def test_backoff_uses_decorrelated_jitter_within_bounds() -> None:
    transport, rec = make(err(500), max_retries=6, backoff_base=0.25, backoff_max=8.0)
    with pytest.raises(ServerError):
        await run(transport)
    previous = 0.25
    for delay in rec.delays:
        assert 0.25 <= delay <= min(8.0, previous * 3) + 1e-9
        previous = delay


async def test_success_only_raises_limit_after_recover_after() -> None:
    config = ConcurrencyConfig(recover_after=2)
    transport, _ = make(err(429), ok(), concurrency=config)
    await run(transport)
    assert transport.limiter.limit(CLAIRIC) == 16
    await run(transport)  # second success in a row
    assert transport.limiter.limit(CLAIRIC) == 17


async def test_per_call_timeout_is_forwarded() -> None:
    transport, rec = make(ok())
    await transport.run(CLAIRIC, PAYLOAD, meta=META, timeout=3.0)
    assert rec.requests[0].extensions["timeout"]["read"] == 3.0
    assert rec.requests[0].extensions["timeout"]["connect"] == 10.0


async def test_default_timeout_comes_from_the_client() -> None:
    transport, rec = make(ok())
    await run(transport)
    assert rec.requests[0].extensions["timeout"]["read"] == 5.0  # httpx default


# --------------------------------------------------------------------------- #
# Backoff does not hold a limiter slot
# --------------------------------------------------------------------------- #


async def test_backing_off_calls_do_not_block_another_model() -> None:
    backing_off = asyncio.Event()
    resume = asyncio.Event()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = str(request.url).rsplit("/", 1)[-1]
        seen.append(model)
        if model == "clairic" and seen.count("clairic") == 1:
            return err(503)
        return ok()

    async def gated_sleep(_: float) -> None:
        backing_off.set()
        await resume.wait()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = CloudflareTransport(
        account_id="a",
        api_token="t",
        client=client,
        sleep=gated_sleep,
        concurrency=ConcurrencyConfig(max_in_flight=1),
    )
    call_a = asyncio.create_task(run(transport, CLAIRIC))
    await asyncio.wait_for(backing_off.wait(), 1)
    assert transport.limiter.in_flight == 0  # A released its slot to back off

    # The single global slot is free, so B completes while A is still waiting.
    result_b = await asyncio.wait_for(run(transport, CLEF), 1)
    assert result_b.body == BODY
    assert not call_a.done()

    resume.set()
    assert (await call_a).retries == 1
    await client.aclose()


async def test_retries_honour_a_limit_lowered_while_backing_off() -> None:
    transport, _ = make(
        err(429), ok(), concurrency=ConcurrencyConfig(per_model={CLAIRIC: 4})
    )
    await run(transport)
    assert transport.limiter.limit(CLAIRIC) == 2


# --------------------------------------------------------------------------- #
# Credentials and lifecycle
# --------------------------------------------------------------------------- #


async def test_missing_credentials_raise_on_first_run_not_at_construction() -> None:
    transport = CloudflareTransport()  # must not raise
    with pytest.raises(AuthError, match="token"):
        await run(transport)
    assert transport.limiter.in_flight == 0
    await transport.aclose()


async def test_missing_account_id_raises_auth_error() -> None:
    transport = CloudflareTransport(api_token="tok")
    with pytest.raises(AuthError, match="account"):
        await run(transport)


async def test_missing_token_raises_even_with_base_url() -> None:
    transport = CloudflareTransport(base_url="http://x")
    with pytest.raises(AuthError, match="token"):
        await run(transport)


async def test_blank_token_is_missing() -> None:
    transport = CloudflareTransport(api_token="   ", account_id="a")
    with pytest.raises(AuthError, match="token"):
        await run(transport)


async def test_injected_client_is_not_closed() -> None:
    transport, _ = make(ok())
    client = transport._get_client()
    async with transport:
        await run(transport)
    assert not client.is_closed
    await transport.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_owned_client_is_created_lazily_per_loop_and_closed() -> None:
    transport = CloudflareTransport(account_id="a", api_token="t")
    assert len(transport._clients) == 0
    client = transport._get_client()
    assert transport._get_client() is client
    assert (
        cast(Any, client._transport)._pool._max_connections == 64
    )  # sized to max_in_flight
    await transport.aclose()
    assert client.is_closed
    await transport.aclose()  # idempotent
    assert transport._get_client() is not client  # a fresh one is made on demand
    await transport.aclose()


def test_cloudflare_transport_satisfies_protocol() -> None:
    assert isinstance(CloudflareTransport(), Transport)


def test_transport_works_from_two_event_loops_and_keeps_learned_limits() -> None:
    transport, _ = make(ok(), err(429), ok(), ok())

    async def first_loop() -> tuple[int, int]:
        await run(transport)
        await run(transport)  # 429 then ok: halves
        return transport.limiter.limit(CLAIRIC), transport.limiter.in_flight

    async def second_loop() -> tuple[int, int]:
        await run(transport)
        return transport.limiter.limit(CLAIRIC), transport.limiter.in_flight

    assert asyncio.run(first_loop()) == (16, 0)
    # A new loop starts from the limits already learned, with its own counters.
    assert asyncio.run(second_loop()) == (16, 0)


def test_sequential_asyncio_runs_with_aclose_leave_no_accumulated_state() -> None:
    transport = CloudflareTransport(account_id="a", api_token="t")

    async def use() -> None:
        async with transport.limiter.slot(CLAIRIC):
            pass
        transport._get_client()
        await transport.aclose()

    for _ in range(5):
        asyncio.run(use())
        assert transport._clients == {}
    # Only the most recent loop's counters are kept; older closed loops are
    # dropped when a new loop first uses the limiter.
    assert len(transport.limiter._states) == 1


def test_clients_and_limiter_state_of_closed_loops_are_evicted() -> None:
    transport = CloudflareTransport(account_id="a", api_token="t")

    async def use() -> None:
        async with transport.limiter.slot(CLAIRIC):
            transport._get_client()  # never closed by the caller

    for _ in range(5):
        asyncio.run(use())
    assert len(transport._clients) == 1
    assert len(transport.limiter._states) == 1
    transport.close()
    assert transport._clients == {}


def test_owned_clients_are_per_loop_and_close_works_without_a_loop() -> None:
    transport = CloudflareTransport(account_id="a", api_token="t")
    clients: list[httpx.AsyncClient] = []

    async def use() -> None:
        clients.append(transport._get_client())
        assert transport._get_client() is clients[-1]

    # Keep the loops referenced: clients are tracked weakly by loop.
    loops = [asyncio.new_event_loop(), asyncio.new_event_loop()]
    for loop in loops:
        loop.run_until_complete(use())
    assert clients[0] is not clients[1]
    transport.close()  # no loop running: closes what it can
    assert all(c.is_closed for c in clients)
    transport.close()  # idempotent
    for loop in loops:
        loop.close()


async def test_close_is_a_no_op_while_a_loop_is_running() -> None:
    transport = CloudflareTransport(account_id="a", api_token="t")
    client = transport._get_client()
    transport.close()
    assert not client.is_closed
    await transport.aclose()
    assert client.is_closed


# --------------------------------------------------------------------------- #
# AdaptiveLimiter
# --------------------------------------------------------------------------- #


def test_limiter_halves_with_floor_of_one() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(per_model={CLAIRIC: 10}))
    seen = []
    for _ in range(6):
        limiter.on_rate_limited(CLAIRIC)
        seen.append(limiter.limit(CLAIRIC))
    assert seen == [5, 2, 1, 1, 1, 1]
    assert limiter.limit(CLEF) == 32  # other models are untouched


def test_stale_generation_does_not_halve_again() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(per_model={CLAIRIC: 16}))
    assert limiter.generation(CLAIRIC) == 0
    assert limiter.on_rate_limited(CLAIRIC, 0) is True
    assert limiter.limit(CLAIRIC) == 8
    assert limiter.generation(CLAIRIC) == 1
    assert limiter.on_rate_limited(CLAIRIC, 0) is False  # same burst
    assert limiter.limit(CLAIRIC) == 8
    assert limiter.on_rate_limited(CLAIRIC, 1) is True
    assert limiter.limit(CLAIRIC) == 4
    assert limiter.on_rate_limited(CLAIRIC) is True  # no generation: always halves


def test_limiter_recovers_one_per_recover_after_up_to_cap() -> None:
    limiter = AdaptiveLimiter(
        ConcurrencyConfig(per_model={CLAIRIC: 4}, recover_after=3)
    )
    limiter.on_rate_limited(CLAIRIC)
    limiter.on_rate_limited(CLAIRIC)
    assert limiter.limit(CLAIRIC) == 1
    for _ in range(2):
        limiter.on_success(CLAIRIC)
    assert limiter.limit(CLAIRIC) == 1
    limiter.on_success(CLAIRIC)
    assert limiter.limit(CLAIRIC) == 2
    for _ in range(3 * 10):
        limiter.on_success(CLAIRIC)
    assert limiter.limit(CLAIRIC) == 4  # capped at the configured value


def test_rate_limit_resets_success_streak() -> None:
    limiter = AdaptiveLimiter(
        ConcurrencyConfig(per_model={CLAIRIC: 8}, recover_after=3)
    )
    limiter.on_rate_limited(CLAIRIC)  # 4
    limiter.on_success(CLAIRIC)
    limiter.on_success(CLAIRIC)
    limiter.on_rate_limited(CLAIRIC)  # 2, streak reset
    limiter.on_success(CLAIRIC)
    limiter.on_success(CLAIRIC)
    assert limiter.limit(CLAIRIC) == 2
    limiter.on_success(CLAIRIC)
    assert limiter.limit(CLAIRIC) == 3


def test_limiter_not_adaptive_never_changes() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(adaptive=False))
    assert limiter.on_rate_limited(CLAIRIC) is False
    assert limiter.limit(CLAIRIC) == 32


def test_limiter_unknown_model_uses_default() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(default_per_model=7))
    assert limiter.limit("@cf/other/model") == 7


def test_limiter_treats_bare_names_as_full_ids() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(per_model={"clef": 8}))
    assert limiter.limit(CLEF) == 8
    limiter.on_rate_limited("clef")
    assert limiter.limit(CLEF) == 4


def test_limiter_can_be_built_outside_a_loop_and_reused_across_loops() -> None:
    limiter = AdaptiveLimiter()

    async def use() -> None:
        async with limiter.slot(CLAIRIC):
            assert limiter.in_flight == 1

    asyncio.run(use())
    asyncio.run(use())
    assert limiter.in_flight == 0


def test_learned_limits_are_shared_across_loops_but_occupancy_is_not() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(per_model={CLAIRIC: 8}))
    seen: list[int] = []

    async def halve() -> None:
        limiter.on_rate_limited(CLAIRIC)
        async with limiter.slot(CLAIRIC):
            seen.append(limiter.in_flight)

    async def read() -> None:
        seen.append(limiter.limit(CLAIRIC))
        seen.append(limiter.in_flight)

    asyncio.run(halve())
    asyncio.run(read())
    assert seen == [1, 4, 0]
    assert limiter.limit(CLAIRIC) == 4  # also readable with no loop running
    assert limiter.in_flight == 0
    assert limiter.generation(CLAIRIC) == 1


async def _max_concurrency(
    limiter: AdaptiveLimiter,
    models: list[str],
) -> tuple[int, dict[str, int]]:
    current = 0
    peak = 0
    current_by: dict[str, int] = {}
    peak_by: dict[str, int] = {}

    async def worker(model: str) -> None:
        nonlocal current, peak
        async with limiter.slot(model):
            current += 1
            current_by[model] = current_by.get(model, 0) + 1
            peak = max(peak, current)
            peak_by[model] = max(peak_by.get(model, 0), current_by[model])
            for _ in range(3):
                await asyncio.sleep(0)
            current -= 1
            current_by[model] -= 1

    await asyncio.gather(*(worker(m) for m in models))
    return peak, peak_by


async def test_global_max_in_flight_is_respected() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(max_in_flight=5))
    peak, _ = await _max_concurrency(limiter, [CLAIRIC, CLEF] * 40)
    assert peak == 5
    assert limiter.in_flight == 0


async def test_per_model_limit_is_respected() -> None:
    limiter = AdaptiveLimiter(
        ConcurrencyConfig(max_in_flight=50, per_model={CLAIRIC: 3, CLEF: 2})
    )
    peak, peak_by = await _max_concurrency(limiter, [CLAIRIC, CLEF] * 30)
    assert peak_by == {CLAIRIC: 3, CLEF: 2}
    assert peak == 5


async def test_halved_limit_applies_to_waiting_callers() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(per_model={CLAIRIC: 4}))
    limiter.on_rate_limited(CLAIRIC)
    limiter.on_rate_limited(CLAIRIC)
    _, peak_by = await _max_concurrency(limiter, [CLAIRIC] * 20)
    assert peak_by[CLAIRIC] == 1


async def test_a_waiter_for_one_model_does_not_strand_another() -> None:
    limiter = AdaptiveLimiter(
        ConcurrencyConfig(max_in_flight=2, per_model={CLAIRIC: 1, CLEF: 2})
    )
    release_a = asyncio.Event()
    order: list[str] = []

    async def holder() -> None:
        async with limiter.slot(CLAIRIC):
            await release_a.wait()

    async def waiting_clairic() -> None:
        async with limiter.slot(CLAIRIC):  # blocked by the per-model limit
            order.append("clairic")

    async def clef_call() -> None:
        async with limiter.slot(CLEF):
            order.append("clef")

    hold = asyncio.create_task(holder())
    await asyncio.sleep(0)
    blocked = asyncio.create_task(waiting_clairic())
    await asyncio.sleep(0)
    other = asyncio.create_task(clef_call())
    await asyncio.wait_for(other, 1)
    assert order == ["clef"]
    release_a.set()
    await asyncio.gather(hold, blocked)
    assert order == ["clef", "clairic"]


async def test_limit_raised_while_waiters_exist_wakes_them() -> None:
    limiter = AdaptiveLimiter(
        ConcurrencyConfig(per_model={CLAIRIC: 2}, recover_after=1)
    )
    limiter.on_rate_limited(CLAIRIC)  # limit 1
    order: list[str] = []
    release = asyncio.Event()

    async def first() -> None:
        async with limiter.slot(CLAIRIC):
            order.append("first in")
            await release.wait()
            order.append("first out")

    async def second() -> None:
        async with limiter.slot(CLAIRIC):
            order.append("second in")

    task_a = asyncio.create_task(first())
    await asyncio.sleep(0)
    task_b = asyncio.create_task(second())
    for _ in range(3):
        await asyncio.sleep(0)
    assert order == ["first in"]  # second waits: the limit is 1

    limiter.on_success(CLAIRIC)  # recover_after=1 raises the limit to 2
    for _ in range(5):
        await asyncio.sleep(0)
    assert order == ["first in", "second in"]  # woken without any release

    release.set()
    await asyncio.gather(task_a, task_b)


async def test_slot_is_released_when_the_body_raises_or_is_cancelled() -> None:
    limiter = AdaptiveLimiter()
    with pytest.raises(RuntimeError):
        async with limiter.slot(CLAIRIC):
            raise RuntimeError("boom")
    assert limiter.in_flight == 0

    async def sleeper() -> None:
        async with limiter.slot(CLAIRIC):
            await asyncio.sleep(10)

    task = asyncio.create_task(sleeper())
    await asyncio.sleep(0)
    assert limiter.in_flight == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.in_flight == 0
    async with limiter.slot(CLAIRIC):  # still usable
        assert limiter.in_flight == 1


@pytest.mark.parametrize(
    "config",
    [
        ConcurrencyConfig(max_in_flight=1),
        ConcurrencyConfig(per_model={CLAIRIC: 1}),
    ],
    ids=["global-1", "per-model-1"],
)
async def test_waiter_is_woken_while_another_task_cycles_the_slot(
    config: ConcurrencyConfig,
) -> None:
    # Regression: the limiter used to replace its Condition when idle, which
    # lost the wake-up of a waiter when a newcomer took the freed slot first.
    limiter = AdaptiveLimiter(config)
    finished: list[str] = []

    async def task_a() -> None:
        for _ in range(5):
            async with limiter.slot(CLAIRIC):
                await asyncio.sleep(0)
        finished.append("A")

    async def task_b() -> None:
        async with limiter.slot(CLAIRIC):
            finished.append("B")

    a = asyncio.create_task(task_a())
    await asyncio.sleep(0)
    b = asyncio.create_task(task_b())
    await asyncio.wait_for(asyncio.gather(a, b), timeout=2)
    assert sorted(finished) == ["A", "B"]
    assert limiter.in_flight == 0


async def test_transport_waiter_is_not_starved_by_sequential_runs() -> None:
    transport, _ = make(ok(), concurrency=ConcurrencyConfig(max_in_flight=1))

    async def task_a() -> None:
        for _ in range(3):
            await run(transport)

    async def task_b() -> TransportResponse:
        return await run(transport)

    a = asyncio.create_task(task_a())
    await asyncio.sleep(0)
    b = asyncio.create_task(task_b())
    await asyncio.wait_for(asyncio.gather(a, b), timeout=2)
    assert transport.limiter.in_flight == 0


async def test_transport_respects_global_limit_under_load() -> None:
    inside = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inside, peak
        inside += 1
        peak = max(peak, inside)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        inside -= 1
        return ok()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = CloudflareTransport(
        account_id="a",
        api_token="t",
        client=client,
        concurrency=ConcurrencyConfig(max_in_flight=3),
    )
    results = await asyncio.gather(*(run(transport) for _ in range(30)))
    assert [r.body for r in results] == [BODY] * 30
    assert peak == 3
    assert transport.limiter.in_flight == 0
    await client.aclose()


# --------------------------------------------------------------------------- #
# call_deadline
# --------------------------------------------------------------------------- #


class FakeClock:
    """A clock that only moves when the injected sleep is awaited."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.delays: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)
        self.now += delay


def make_clocked(
    *responses: httpx.Response | Exception, **kwargs: Any
) -> tuple[CloudflareTransport, Recorder, FakeClock]:
    recorder = Recorder(*responses)
    clock = FakeClock()
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    transport = CloudflareTransport(
        client=client,
        account_id="acct",
        api_token="tok",
        sleep=clock.sleep,
        clock=clock,
        **kwargs,
    )
    return transport, recorder, clock


async def test_default_call_deadline_is_300_seconds() -> None:
    transport, rec, clock = make_clocked(
        err(503, headers={"Retry-After": "50"}), max_retries=50, max_retry_after=60.0
    )
    with pytest.raises(TransportError, match="deadline") as info:
        await run(transport)
    assert clock.delays == [50.0] * 5  # the sixth wait would end at the deadline
    assert info.value.retries == 5
    assert len(rec.requests) == 6


async def test_deadline_stops_retrying_with_a_deadline_error_that_keeps_retries() -> (
    None
):
    transport, rec, clock = make_clocked(
        err(503, headers={"Retry-After": "120"}),
        max_retries=10,
        max_retry_after=200.0,
        call_deadline=300.0,
    )
    with pytest.raises(TransportError) as info:
        await run(transport)
    error = info.value
    assert type(error) is TransportError
    assert "deadline" in str(error)
    assert "300" in str(error)
    assert error.retries == 2
    assert error.status_code == 503
    assert isinstance(error.__cause__, ServerError)
    assert clock.delays == [120.0, 120.0]
    assert len(rec.requests) == 3
    assert transport.limiter.in_flight == 0


async def test_deadline_covers_backoff_sleeps_without_retry_after() -> None:
    transport, rec, clock = make_clocked(
        err(500),
        max_retries=1000,
        backoff_base=1.0,
        backoff_max=10.0,
        call_deadline=25.0,
    )
    with pytest.raises(TransportError, match="deadline"):
        await run(transport)
    assert sum(clock.delays) < 25.0
    assert len(rec.requests) == len(clock.delays) + 1


async def test_attempt_timeout_is_capped_by_the_remaining_budget() -> None:
    transport, rec, _ = make_clocked(
        err(503, headers={"Retry-After": "4"}), ok(), call_deadline=10.0
    )
    await transport.run(CLAIRIC, PAYLOAD, meta=META, timeout=60.0)
    first, second = (r.extensions["timeout"] for r in rec.requests)
    assert first["read"] == 10.0
    assert first["connect"] == 10.0
    assert second["read"] == 6.0
    assert second["connect"] == 6.0


async def test_explicit_timeout_below_the_budget_is_left_alone() -> None:
    transport, rec, _ = make_clocked(ok(), call_deadline=100.0)
    await transport.run(CLAIRIC, PAYLOAD, meta=META, timeout=3.0)
    assert rec.requests[0].extensions["timeout"]["read"] == 3.0
    assert rec.requests[0].extensions["timeout"]["connect"] == 10.0


async def test_client_default_timeout_is_capped_only_when_the_budget_is_tighter() -> (
    None
):
    transport, rec, _ = make_clocked(ok(), call_deadline=3.0)
    await run(transport)
    assert rec.requests[0].extensions["timeout"]["read"] == 3.0  # not httpx's 5.0
    transport, rec, _ = make_clocked(ok(), call_deadline=30.0)
    await run(transport)
    assert rec.requests[0].extensions["timeout"]["read"] == 5.0


async def test_no_deadline_means_unbounded_retries_in_time() -> None:
    transport, _, clock = make_clocked(
        err(503, headers={"Retry-After": "50"}),
        max_retries=10,
        max_retry_after=60.0,
        call_deadline=None,
    )
    with pytest.raises(ServerError):
        await run(transport)
    assert clock.delays == [50.0] * 10


async def test_time_spent_queueing_for_a_slot_does_not_count_against_the_deadline() -> (
    None
):
    transport, rec, clock = make_clocked(
        ok(), call_deadline=5.0, concurrency=ConcurrencyConfig(per_model={CLAIRIC: 1})
    )
    release = asyncio.Event()
    acquired = asyncio.Event()

    async def hold() -> None:
        async with transport.limiter.slot(CLAIRIC):
            acquired.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await acquired.wait()
    call = asyncio.create_task(run(transport))
    for _ in range(5):
        await asyncio.sleep(0)  # let the call queue behind the holder
    clock.now += 10.0  # longer than the whole deadline
    release.set()
    assert (await call).body == BODY  # the request was sent and succeeded
    await holder
    assert len(rec.requests) == 1
    assert transport.limiter.in_flight == 0


async def test_the_deadline_starts_when_the_first_slot_is_acquired() -> None:
    transport, _, clock = make_clocked(
        err(503, headers={"Retry-After": "2"}),
        max_retries=10,
        call_deadline=5.0,
        concurrency=ConcurrencyConfig(per_model={CLAIRIC: 1}),
    )
    release = asyncio.Event()
    acquired = asyncio.Event()

    async def hold() -> None:
        async with transport.limiter.slot(CLAIRIC):
            acquired.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await acquired.wait()
    call = asyncio.create_task(run(transport))
    for _ in range(5):
        await asyncio.sleep(0)
    clock.now += 100.0
    release.set()
    with pytest.raises(TransportError, match="deadline") as info:
        await call
    await holder
    assert clock.delays == [2.0, 2.0]  # budget counted from the first attempt
    assert info.value.retries == 2


async def test_many_queued_calls_all_succeed_under_a_short_deadline() -> None:
    delay = 0.05

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay)
        return ok()

    client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    transport = CloudflareTransport(
        account_id="a",
        api_token="t",
        client=client,
        concurrency=ConcurrencyConfig(per_model={CLAIRIC: 1}),
        call_deadline=delay * 3,  # far less than 8 queued requests take in total
    )
    results = await asyncio.gather(*(run(transport) for _ in range(8)))
    assert [r.body for r in results] == [BODY] * 8


async def test_queue_timeout_fails_a_call_that_never_got_a_slot() -> None:
    transport, rec, _ = make_clocked(
        ok(),
        queue_timeout=0.05,
        concurrency=ConcurrencyConfig(per_model={CLAIRIC: 1}),
    )
    release = asyncio.Event()
    acquired = asyncio.Event()

    async def hold() -> None:
        async with transport.limiter.slot(CLAIRIC):
            acquired.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await acquired.wait()
    with pytest.raises(TransportError, match="waiting for a request slot") as info:
        await run(transport)
    assert info.value.retries == 0
    assert rec.requests == []
    release.set()
    await holder
    assert transport.limiter.in_flight == 0
    assert (await run(transport)).body == BODY  # the transport still works


async def test_queue_timeout_defaults_to_waiting_indefinitely() -> None:
    transport, _, _ = make_clocked(ok())
    assert transport._queue_timeout is None


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
def test_invalid_queue_timeout_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="queue_timeout"):
        CloudflareTransport(queue_timeout=bad)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
def test_invalid_call_deadline_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="call_deadline"):
        CloudflareTransport(call_deadline=bad)
