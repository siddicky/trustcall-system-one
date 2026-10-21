"""Tests for CloudflareTransport and AdaptiveLimiter without any network."""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

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
)
from trustcall_system_one._version import __version__
from trustcall_system_one.transport import AdaptiveLimiter

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


async def run(transport: CloudflareTransport, model: str = CLAIRIC) -> dict[str, Any]:
    return await transport.run(model, PAYLOAD, meta=META)


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


async def test_default_url_headers_and_body() -> None:
    transport, rec = make(ok())
    assert await run(transport) == BODY
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


async def test_extra_headers_are_sent() -> None:
    transport, rec = make(ok(), extra_headers={"cf-aig-authorization": "Bearer g"})
    await run(transport)
    assert rec.requests[0].headers["cf-aig-authorization"] == "Bearer g"


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #


async def test_envelope_is_unwrapped() -> None:
    envelope = {"success": True, "result": BODY, "errors": [], "messages": []}
    transport, _ = make(ok(envelope))
    assert await run(transport) == BODY


async def test_bare_body_is_returned_as_is() -> None:
    transport, _ = make(ok())
    assert await run(transport) == BODY


async def test_non_json_success_raises_model_response_error() -> None:
    transport, _ = make(httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(ModelResponseError) as info:
        await run(transport)
    assert info.value.raw == "<html>nope</html>"


async def test_non_object_success_raises_model_response_error() -> None:
    transport, _ = make(ok([1, 2]))
    with pytest.raises(ModelResponseError):
        await run(transport)


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
    assert len(rec.requests) == 1
    assert rec.delays == []


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


async def test_other_4xx_is_not_retried() -> None:
    transport, rec = make(err(422, text="unprocessable"))
    with pytest.raises(BadRequestError) as info:
        await run(transport)
    assert info.value.status_code == 422
    assert info.value.body == "unprocessable"
    assert info.value.cf_errors == []
    assert len(rec.requests) == 1


# --------------------------------------------------------------------------- #
# Retries and errors
# --------------------------------------------------------------------------- #


async def test_429_with_retry_after_then_success() -> None:
    transport, rec = make(err(429, headers={"Retry-After": "2"}, json={}), ok())
    assert await run(transport) == BODY
    assert rec.delays == [2.0]
    assert len(rec.requests) == 2
    assert transport.limiter.limit(CLAIRIC) == 16  # halved from 32
    assert transport.last_retries == 1


async def test_retry_after_http_date() -> None:
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    header = format_datetime(when, usegmt=True)
    transport, rec = make(err(503, headers={"Retry-After": header}), ok())
    await run(transport)
    (delay,) = rec.delays
    assert 25 <= delay <= 30


async def test_unparseable_retry_after_falls_back_to_jitter() -> None:
    transport, rec = make(err(503, headers={"Retry-After": "soon"}), ok())
    await run(
        transport,
    )
    (delay,) = rec.delays
    assert 0.25 <= delay <= 0.75


async def test_persistent_503_raises_server_error_with_retry_count() -> None:
    transport, rec = make(
        err(503, json={"errors": [{"message": "down"}]}), max_retries=3
    )
    with pytest.raises(ServerError) as info:
        await run(transport)
    assert info.value.status_code == 503
    assert info.value.body == {"errors": [{"message": "down"}]}
    assert len(rec.requests) == 4  # first try + max_retries
    assert len(rec.delays) == 3
    assert transport.last_retries == 3


async def test_default_max_retries_is_six() -> None:
    transport, rec = make(err(500))
    with pytest.raises(ServerError):
        await run(transport)
    assert len(rec.requests) == 7
    assert transport.last_retries == 6


async def test_persistent_429_raises_rate_limit_error() -> None:
    transport, rec = make(err(429, headers={"Retry-After": "1.5"}), max_retries=2)
    with pytest.raises(RateLimitError) as info:
        await run(transport)
    assert info.value.retry_after == 1.5
    assert info.value.status_code == 429
    assert len(rec.requests) == 3
    assert rec.delays == [1.5, 1.5]
    # Halved on each of the three 429s: 32 -> 16 -> 8 -> 4.
    assert transport.limiter.limit(CLAIRIC) == 4


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
    assert len(rec.requests) == 1
    assert rec.delays == []
    assert transport.last_retries == 0


async def test_connect_error_then_success() -> None:
    transport, rec = make(httpx.ConnectError("boom"), ok())
    assert await run(transport) == BODY
    assert len(rec.requests) == 2
    assert len(rec.delays) == 1
    assert transport.last_retries == 1


async def test_persistent_connect_error_raises_transport_error() -> None:
    transport, rec = make(httpx.ReadTimeout("slow"), max_retries=2)
    with pytest.raises(TransportError) as info:
        await run(transport)
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)
    assert info.value.status_code is None
    assert len(rec.requests) == 3


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


# --------------------------------------------------------------------------- #
# Credentials and lifecycle
# --------------------------------------------------------------------------- #


async def test_missing_credentials_raise_on_first_run_not_at_construction() -> None:
    transport = CloudflareTransport()  # must not raise
    with pytest.raises(AuthError, match="token"):
        await run(transport)
    await transport.aclose()


async def test_missing_account_id_raises_auth_error() -> None:
    transport = CloudflareTransport(api_token="tok")
    with pytest.raises(AuthError, match="account"):
        await run(transport)


async def test_missing_token_raises_even_with_base_url() -> None:
    transport = CloudflareTransport(base_url="http://x")
    with pytest.raises(AuthError, match="token"):
        await run(transport)


async def test_injected_client_is_not_closed() -> None:
    transport, _ = make(ok())
    client = transport._client
    assert client is not None
    async with transport:
        await run(transport)
    assert not client.is_closed
    await client.aclose()


async def test_owned_client_is_created_lazily_and_closed() -> None:
    transport = CloudflareTransport(account_id="a", api_token="t")
    assert transport._client is None
    client = transport._get_client()
    assert transport._get_client() is client
    await transport.aclose()
    assert client.is_closed
    await transport.aclose()  # idempotent


def test_cloudflare_transport_satisfies_protocol() -> None:
    assert isinstance(CloudflareTransport(), Transport)


async def test_last_retries_is_per_task() -> None:
    transport, _ = make(err(503), ok(), ok())

    async def child() -> int:
        await run(transport)
        return transport.last_retries

    # The first task consumes the 503 and sees one retry; the second sees none.
    assert await asyncio.create_task(child()) == 1
    assert await asyncio.create_task(child()) == 0


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
    limiter.on_rate_limited(CLAIRIC)
    assert limiter.limit(CLAIRIC) == 32


def test_limiter_unknown_model_uses_default() -> None:
    limiter = AdaptiveLimiter(ConcurrencyConfig(default_per_model=7))
    assert limiter.limit("@cf/other/model") == 7


def test_limiter_can_be_built_outside_a_loop() -> None:
    limiter = AdaptiveLimiter()

    async def use() -> None:
        async with limiter.slot(CLAIRIC):
            assert limiter.in_flight == 1

    # Reuse across two different event loops.
    asyncio.run(use())
    asyncio.run(use())
    assert limiter.in_flight == 0


async def _max_concurrency(
    limiter: AdaptiveLimiter,
    models: list[str],
    hold: Callable[[], Any] | None = None,
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
    assert results == [BODY] * 30
    assert peak == 3
    await client.aclose()
