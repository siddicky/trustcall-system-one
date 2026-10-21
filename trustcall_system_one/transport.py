"""Transport protocol, adaptive concurrency limiter and the Cloudflare client."""

import asyncio
import contextvars
import json
import os
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
from httpx._client import UseClientDefault

from trustcall_system_one._config import ConcurrencyConfig
from trustcall_system_one._errors import (
    AuthError,
    BadRequestError,
    ModelResponseError,
    RateLimitError,
    ServerError,
    TransportError,
)
from trustcall_system_one._version import __version__

__all__ = [
    "AdaptiveLimiter",
    "CallMeta",
    "CloudflareTransport",
    "Transport",
]

CallKind = Literal[
    "presence",
    "discover",
    "count",
    "coref",
    "qualify",
    "match",
    "delete",
    "extract",
    "crosscheck",
    "clef_fields",
    "attribution",
    "reask",
    "null_check",
    "adjudicate",
    "list_dedup",
    "gate",
    "repair",
    "raw",
]


@dataclass(frozen=True)
class CallMeta:
    """Routing and observability context for one transport call.

    Fakes route on it, and ``CloudflareTransport`` forwards it to AI Gateway as
    ``cf-aig-metadata`` when a gateway is configured.
    """

    kind: CallKind
    schema: str = ""
    entity: str | None = None
    field: str | None = None
    scope: str = ""


@runtime_checkable
class Transport(Protocol):
    """Anything that can run a System One model call."""

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Run one model call.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing and observability context.
            timeout: Per-call timeout in seconds, or ``None`` for the default.

        Returns:
            The bare response object (the REST envelope already removed).
        """
        ...


class AdaptiveLimiter:
    """A global in-flight cap plus a per-model limit that adapts to 429s.

    On a rate limit a model's limit is halved (never below 1). After
    ``recover_after`` consecutive successes it grows by one, up to the
    configured value. Waiters re-check the limits whenever a slot is released or
    a limit is raised.

    The condition variable is created per running event loop on first use, so
    the limiter can be built outside a loop and reused across loops.
    """

    def __init__(self, config: ConcurrencyConfig | None = None) -> None:
        """Create a limiter.

        Args:
            config: Concurrency settings; defaults to ``ConcurrencyConfig()``.
        """
        self._config = config or ConcurrencyConfig()
        self._limits: dict[str, int] = {}
        self._streaks: dict[str, int] = {}
        self._active: dict[str, int] = {}
        self._in_flight = 0
        self._wake_tasks: set[asyncio.Task[None]] = set()
        self._cond: asyncio.Condition | None = None
        self._cond_loop: asyncio.AbstractEventLoop | None = None

    @property
    def max_in_flight(self) -> int:
        """The global cap on concurrent requests."""
        return self._config.max_in_flight

    @property
    def in_flight(self) -> int:
        """Number of slots currently held."""
        return self._in_flight

    def limit(self, model: str) -> int:
        """Return the model's current concurrency limit."""
        return self._limits.get(model, self._config.limit_for(model))

    def on_rate_limited(self, model: str) -> None:
        """Halve the model's limit (floor 1) and reset its success streak."""
        if not self._config.adaptive:
            return
        self._limits[model] = max(1, self.limit(model) // 2)
        self._streaks[model] = 0

    def on_success(self, model: str) -> None:
        """Count a success; raise the limit by one every ``recover_after``."""
        if not self._config.adaptive:
            return
        streak = self._streaks.get(model, 0) + 1
        if streak >= self._config.recover_after:
            streak = 0
            cap = self._config.limit_for(model)
            self._limits[model] = min(cap, self.limit(model) + 1)
            self._wake_waiters()
        self._streaks[model] = streak

    def _wake_waiters(self) -> None:
        # on_success is synchronous but notifying a Condition needs its lock,
        # so hand the notification to a task on the limiter's loop.
        cond = self._cond
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if cond is None or loop is not self._cond_loop:
            return

        async def notify() -> None:
            async with cond:
                cond.notify_all()

        task = loop.create_task(notify())
        self._wake_tasks.add(task)
        task.add_done_callback(self._wake_tasks.discard)

    def _condition(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._cond is None or self._cond_loop is not loop:
            self._cond = asyncio.Condition()
            self._cond_loop = loop
        return self._cond

    @asynccontextmanager
    async def slot(self, model: str) -> AsyncIterator[None]:
        """Hold one global slot and one slot of the model's limit.

        Args:
            model: Full model id the call goes to.

        Yields:
            Once both limits have room.
        """
        cond = self._condition()
        async with cond:
            await cond.wait_for(
                lambda: (
                    self._in_flight < self.max_in_flight
                    and self._active.get(model, 0) < self.limit(model)
                )
            )
            self._in_flight += 1
            self._active[model] = self._active.get(model, 0) + 1
        try:
            yield
        finally:
            async with cond:
                self._in_flight -= 1
                self._active[model] -= 1
                cond.notify_all()


# Retry count of the most recent `run`, kept per asyncio task context so
# concurrent calls do not overwrite each other.
_last_retries: contextvars.ContextVar[int] = contextvars.ContextVar(
    "trustcall_system_one_last_retries", default=0
)

_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_RETRYABLE = frozenset({408, 429})
_METADATA_KEYS = ("kind", "schema", "entity", "field", "scope")


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (seconds or HTTP-date) into seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _cf_errors(body: Any) -> list[dict[str, object]]:
    if not isinstance(body, dict):
        return []
    errors = body.get("errors")
    if not isinstance(errors, list):
        return []
    return [e if isinstance(e, dict) else {"message": str(e)} for e in errors]


def _error_for_status(
    response: httpx.Response, *, retry_after: float | None
) -> Exception:
    """Map a non-2xx response (or an unsuccessful envelope) to an exception."""
    status = response.status_code
    body = _json_or_text(response)
    errors = _cf_errors(body)
    detail = str(errors[0].get("message", errors[0])) if errors else str(body)[:300]
    message = f"Cloudflare API returned HTTP {status}: {detail}"
    if status in (401, 403):
        return AuthError(message, status_code=status, body=body)
    if status == 429:
        return RateLimitError(
            message, status_code=status, body=body, retry_after=retry_after
        )
    if 400 <= status < 500 and status != 408:
        return BadRequestError(message, status_code=status, body=body, cf_errors=errors)
    if status >= 500:
        return ServerError(message, status_code=status, body=body)
    return TransportError(message, status_code=status, body=body)


class CloudflareTransport:
    """Calls System One models over the Cloudflare REST API or AI Gateway.

    Retries 408, 429, 5xx and ``httpx.TransportError`` up to ``max_retries``
    times, waiting for ``Retry-After`` when present and otherwise using
    decorrelated jitter. A call holds its limiter slot across retries, which
    also throttles a rate-limited model.

    Retry counts: after ``await transport.run(...)`` returns or raises, the
    ``last_retries`` property gives the number of retries that call needed. It
    is tracked per asyncio task, so read it in the same task that awaited
    ``run`` (not across ``asyncio.gather`` children).
    """

    def __init__(
        self,
        *,
        account_id: str | None = None,
        api_token: str | None = None,
        base_url: str | None = None,
        gateway: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
        max_retries: int = 6,
        backoff_base: float = 0.25,
        backoff_max: float = 8.0,
        concurrency: ConcurrencyConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Create a transport.

        Credentials are resolved on each ``run``, so missing ones raise
        ``AuthError`` there and not here.

        Args:
            account_id: Cloudflare account id; else ``CLOUDFLARE_ACCOUNT_ID``.
            api_token: API token; else ``CLOUDFLARE_API_TOKEN`` or
                ``CLOUDFLARE_AUTH_TOKEN``.
            base_url: Replaces the API root; requests go to
                ``{base_url}/ai/run/{model}``. Else ``TRUSTCALL_S1_BASE_URL``.
            gateway: AI Gateway id; routes through the gateway.
            client: An ``httpx.AsyncClient`` to use. It is never closed here.
            timeout: Timeout for a client this transport creates.
            max_retries: Retries after the first attempt.
            backoff_base: Smallest retry delay in seconds.
            backoff_max: Largest jittered retry delay in seconds.
            concurrency: Concurrency limits for the adaptive limiter.
            sleep: Awaitable sleep, injectable for tests.
            extra_headers: Headers added to (and able to override) the defaults.
        """
        self._account_id = account_id
        self._api_token = api_token
        self._base_url = base_url
        self._gateway = gateway
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._sleep = sleep
        self._extra_headers = dict(extra_headers or {})
        self.limiter = AdaptiveLimiter(concurrency)

    @property
    def last_retries(self) -> int:
        """Retries used by the most recent ``run`` in the current task."""
        return _last_retries.get()

    async def __aenter__(self) -> "CloudflareTransport":
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the owned client on exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client if this transport created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _url_and_token(self, model: str) -> tuple[str, str]:
        account_id = self._account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        token = (
            self._api_token
            or os.environ.get("CLOUDFLARE_API_TOKEN")
            or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
        )
        base_url = self._base_url or os.environ.get("TRUSTCALL_S1_BASE_URL")
        if not token:
            raise AuthError(
                "No Cloudflare API token: pass api_token or set CLOUDFLARE_API_TOKEN"
            )
        if base_url:
            return f"{base_url.rstrip('/')}/ai/run/{model}", token
        if not account_id:
            raise AuthError(
                "No Cloudflare account id: pass account_id or set CLOUDFLARE_ACCOUNT_ID"
            )
        if self._gateway:
            return (
                "https://gateway.ai.cloudflare.com/v1/"
                f"{account_id}/{self._gateway}/workers-ai/{model}",
                token,
            )
        return (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
            token,
        )

    def _headers(self, token: str, meta: CallMeta) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": f"trustcall-system-one/{__version__}",
        }
        if self._gateway:
            entries = {
                key: value for key in _METADATA_KEYS if (value := getattr(meta, key))
            }
            headers["cf-aig-metadata"] = json.dumps(entries)
        headers.update(self._extra_headers)
        return headers

    def _call_timeout(self, timeout: float | None) -> httpx.Timeout | UseClientDefault:
        if timeout is None:
            return httpx.USE_CLIENT_DEFAULT
        return httpx.Timeout(timeout, connect=self._timeout.connect)

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Run one model call with retries.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing and observability context.
            timeout: Per-call timeout in seconds; ``None`` uses the client's.

        Returns:
            The response object with the REST envelope removed.

        Raises:
            AuthError: Missing credentials, or HTTP 401/403.
            BadRequestError: HTTP 4xx other than auth, rate limit and timeout.
            RateLimitError: HTTP 429 after retries.
            ServerError: HTTP 5xx after retries.
            TransportError: Network failure or HTTP 408 after retries.
            ModelResponseError: A 2xx response that is not a JSON object.
        """
        _last_retries.set(0)
        url, token = self._url_and_token(model)
        headers = self._headers(token, meta)
        client = self._get_client()
        retries = 0
        previous_delay = self._backoff_base
        async with self.limiter.slot(model):
            while True:
                retry_after: float | None = None
                failure: Exception
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self._call_timeout(timeout),
                    )
                except httpx.TransportError as exc:
                    failure = TransportError(
                        f"Request to {url} failed: {exc!r}", body=None
                    )
                    failure.__cause__ = exc
                else:
                    if response.is_success:
                        self.limiter.on_success(model)
                        return self._unwrap(response)
                    status = response.status_code
                    retry_after = _parse_retry_after(
                        response.headers.get("Retry-After")
                    )
                    if status == 429:
                        self.limiter.on_rate_limited(model)
                    failure = _error_for_status(response, retry_after=retry_after)
                    if status not in _RETRYABLE and status < 500:
                        raise failure
                if retries >= self._max_retries:
                    raise failure
                retries += 1
                _last_retries.set(retries)
                if retry_after is not None:
                    delay = retry_after
                else:
                    delay = min(
                        self._backoff_max,
                        random.uniform(self._backoff_base, previous_delay * 3),
                    )
                    previous_delay = delay
                await self._sleep(delay)

    @staticmethod
    def _unwrap(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelResponseError(
                "Cloudflare API returned a non-JSON success response",
                raw=response.text,
            ) from exc
        if isinstance(body, dict) and body.get("success") is False:
            errors = _cf_errors(body)
            detail = str(errors[0].get("message", errors[0])) if errors else "unknown"
            raise BadRequestError(
                f"Cloudflare API reported failure: {detail}",
                status_code=response.status_code,
                body=body,
                cf_errors=errors,
            )
        if isinstance(body, dict) and "success" in body and "result" in body:
            body = body["result"]
        if not isinstance(body, dict):
            raise ModelResponseError(
                "Cloudflare API response is not a JSON object", raw=body
            )
        return body
