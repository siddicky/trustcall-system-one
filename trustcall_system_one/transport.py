"""Transport protocol, adaptive concurrency limiter and the Cloudflare client."""

import asyncio
import contextlib
import json
import math
import os
import random
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from trustcall_system_one._config import ConcurrencyConfig
from trustcall_system_one._errors import (
    AuthError,
    BadRequestError,
    ModelResponseError,
    RateLimitError,
    ServerError,
    TransportError,
)
from trustcall_system_one._types import CallKind
from trustcall_system_one._version import __version__
from trustcall_system_one._wire import normalize_model_id

__all__ = [
    "AdaptiveLimiter",
    "CallKind",
    "CallMeta",
    "CloudflareTransport",
    "Transport",
    "TransportResponse",
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


@dataclass(frozen=True)
class TransportResponse:
    """The result of one successful transport call.

    Attributes:
        body: The bare response object (the REST envelope already removed).
        retries: Retries needed before the call succeeded.
        duration_ms: Wall-clock time of the call, including retries and waits.
        status_code: HTTP status of the successful response, if there was one.
    """

    body: dict[str, Any]
    retries: int = 0
    duration_ms: float = 0.0
    status_code: int | None = None


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
    ) -> TransportResponse:
        """Run one model call.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing and observability context.
            timeout: Per-call timeout in seconds, or ``None`` for the default.

        Returns:
            The bare response body plus retry and timing information.
        """
        ...


@dataclass
class _LoopState:
    """Limiter bookkeeping for one event loop."""

    limits: dict[str, int] = field(default_factory=dict)
    streaks: dict[str, int] = field(default_factory=dict)
    active: dict[str, int] = field(default_factory=dict)
    generations: dict[str, int] = field(default_factory=dict)
    in_flight: int = 0
    cond: asyncio.Condition | None = None
    wake_tasks: set["asyncio.Task[None]"] = field(default_factory=set)


class AdaptiveLimiter:
    """A global in-flight cap plus a per-model limit that adapts to 429s.

    On a rate limit a model's limit is halved (never below 1). After
    ``recover_after`` consecutive successes it grows by one, up to the
    configured value. Waiters re-check the limits whenever a slot is released or
    a limit is raised.

    Halving happens at most once per generation: ``slot`` yields the model's
    generation at acquisition, and ``on_rate_limited`` ignores a 429 from an
    older generation, so a burst of concurrent 429s halves the limit once.

    All state is kept per running event loop (and, outside any loop, in one
    detached state), so the caps apply per loop and one limiter can be reused
    from several loops without sharing a ``Condition``.
    """

    def __init__(self, config: ConcurrencyConfig | None = None) -> None:
        """Create a limiter.

        Args:
            config: Concurrency settings; defaults to ``ConcurrencyConfig()``.
        """
        self._config = config or ConcurrencyConfig()
        self._states: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, _LoopState
        ] = weakref.WeakKeyDictionary()
        self._detached = _LoopState()

    @property
    def max_in_flight(self) -> int:
        """The global cap on concurrent requests (per event loop)."""
        return self._config.max_in_flight

    @property
    def in_flight(self) -> int:
        """Number of slots currently held on the running loop."""
        return self._state().in_flight

    def _state(self) -> _LoopState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._detached
        state = self._states.get(loop)
        if state is None:
            state = self._states[loop] = _LoopState()
        return state

    def limit(self, model: str) -> int:
        """Return the model's current concurrency limit."""
        model = normalize_model_id(model)
        return self._state().limits.get(model, self._config.limit_for(model))

    def generation(self, model: str) -> int:
        """Return how many times the model's limit has been halved."""
        return self._state().generations.get(normalize_model_id(model), 0)

    def on_rate_limited(self, model: str, generation: int | None = None) -> bool:
        """Halve the model's limit (floor 1) and reset its success streak.

        Args:
            model: The model that was rate limited.
            generation: The generation the failing request acquired its slot
                under. A request from an older generation does not halve again.

        Returns:
            Whether the limit was halved.
        """
        if not self._config.adaptive:
            return False
        model = normalize_model_id(model)
        state = self._state()
        state.streaks[model] = 0
        if generation is not None and generation != state.generations.get(model, 0):
            return False
        state.limits[model] = max(1, self.limit(model) // 2)
        state.generations[model] = state.generations.get(model, 0) + 1
        return True

    def on_success(self, model: str) -> None:
        """Count a success; raise the limit by one every ``recover_after``."""
        if not self._config.adaptive:
            return
        model = normalize_model_id(model)
        state = self._state()
        streak = state.streaks.get(model, 0) + 1
        if streak >= self._config.recover_after:
            streak = 0
            cap = self._config.limit_for(model)
            state.limits[model] = min(cap, self.limit(model) + 1)
            self._schedule_wake(state)
        state.streaks[model] = streak

    def _schedule_wake(self, state: _LoopState) -> None:
        # Notifying a Condition needs its lock, which a synchronous caller
        # cannot await, so hand the notification to a task on the loop.
        cond = state.cond
        if cond is None or state is self._detached:
            return

        async def notify() -> None:
            async with cond:
                cond.notify_all()

        task = asyncio.get_running_loop().create_task(notify())
        state.wake_tasks.add(task)
        task.add_done_callback(state.wake_tasks.discard)

    @asynccontextmanager
    async def slot(self, model: str) -> AsyncIterator[int]:
        """Hold one global slot and one slot of the model's limit.

        Args:
            model: Model id the call goes to.

        Yields:
            The model's generation at acquisition, to pass back to
            ``on_rate_limited``.
        """
        model = normalize_model_id(model)
        state = self._state()
        if state.cond is None:
            state.cond = asyncio.Condition()
        cond = state.cond
        async with cond:
            await cond.wait_for(
                lambda: (
                    state.in_flight < self.max_in_flight
                    and state.active.get(model, 0) < self.limit(model)
                )
            )
            state.in_flight += 1
            state.active[model] = state.active.get(model, 0) + 1
            generation = state.generations.get(model, 0)
        try:
            yield generation
        finally:
            # Update the counters first so they are right even if the
            # notification below is interrupted by a cancellation.
            state.in_flight -= 1
            state.active[model] -= 1
            try:
                async with cond:
                    # notify_all, not notify(n): waiters differ by model, so
                    # waking only the first n could wake ones that still
                    # cannot run and strand one that can.
                    cond.notify_all()
            except asyncio.CancelledError:
                self._schedule_wake(state)
                raise
            finally:
                if state.in_flight == 0 and state.cond is cond:
                    # Idle: drop the Condition (it pins its loop once used).
                    state.cond = None


_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_RETRYABLE_STATUS = frozenset({408, 429})
_RETRYABLE_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
_METADATA_KEYS = ("kind", "schema", "entity", "field", "scope")


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (seconds or HTTP-date) into seconds."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:  # "-0000" means UTC with no local offset known
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, when.timestamp() - time.time())


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _cf_errors(body: Any) -> list[dict[str, object]]:
    """Collect CF error entries from ``errors`` (or a singular ``error``)."""
    if not isinstance(body, dict):
        return []
    entries = body.get("errors") or body.get("error")
    if entries is None:
        return []
    if not isinstance(entries, list):
        entries = [entries]
    return [e if isinstance(e, dict) else {"message": str(e)} for e in entries]


def _error_for_status(
    response: httpx.Response, *, retry_after: float | None
) -> TransportError:
    """Map a non-2xx response to an exception."""
    status = response.status_code
    body = _json_or_text(response)
    errors = _cf_errors(body)
    detail = str(errors[0].get("message", errors[0])) if errors else str(body)[:300]
    message = f"Cloudflare API returned HTTP {status}: {detail}"
    error: TransportError
    if status in (401, 403):
        error = AuthError(message, status_code=status, body=body)
    elif status == 429:
        error = RateLimitError(
            message, status_code=status, body=body, retry_after=retry_after
        )
    elif status >= 500:
        error = ServerError(message, status_code=status, body=body)
    elif 300 <= status < 500 and status != 408:
        error = BadRequestError(
            message, status_code=status, body=body, cf_errors=errors
        )
    else:
        error = TransportError(message, status_code=status, body=body)
    error.retry_after = retry_after
    return error


class CloudflareTransport:
    """Calls System One models over the Cloudflare REST API or AI Gateway.

    Retries 408, 429, 5xx, timeouts, network errors and remote protocol errors
    up to ``max_retries`` times, waiting for ``Retry-After`` when present and
    otherwise using decorrelated jitter. The limiter slot is released while
    waiting, so retries honour a lowered limit and do not starve other models.
    If the server asks for a wait longer than ``max_retry_after`` the call
    fails at once with ``retry_after`` set on the error.

    Event loops: the httpx client this transport creates, and the limiter's
    state, are kept per running event loop, so one transport can be used from
    several loops (for example repeated ``asyncio.run`` calls) and the
    concurrency caps apply per loop. Call ``aclose()`` inside each loop that
    used the transport, or ``close()`` once no loop is running. A client
    passed in as ``client`` is used for every loop and is the caller's to
    manage.
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
        max_retry_after: float = 60.0,
        concurrency: ConcurrencyConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        extra_headers: Mapping[str, str] | None = None,
        send_metadata: bool = True,
    ) -> None:
        """Create a transport.

        Credentials are resolved on each ``run``, so missing ones raise
        ``AuthError`` there and not here.

        Args:
            account_id: Cloudflare account id; else ``CLOUDFLARE_ACCOUNT_ID``.
            api_token: API token; else ``CLOUDFLARE_API_TOKEN`` or
                ``CLOUDFLARE_AUTH_TOKEN``.
            base_url: Replaces the API root; requests go to
                ``{base_url}/ai/run/{model}``. An explicit ``base_url`` or
                ``gateway`` argument beats ``TRUSTCALL_S1_BASE_URL``.
            gateway: AI Gateway id; routes through the gateway.
            client: An ``httpx.AsyncClient`` to use. It is never closed here.
            timeout: Timeout for clients this transport creates.
            max_retries: Retries after the first attempt (at least 0).
            backoff_base: Smallest retry delay in seconds (positive).
            backoff_max: Largest jittered retry delay (at least ``backoff_base``).
            max_retry_after: Longest server-requested wait to honour; a longer
                ``Retry-After`` fails the call immediately.
            concurrency: Concurrency limits for the adaptive limiter.
            sleep: Awaitable sleep, injectable for tests.
            extra_headers: Headers added to (and able to override) the defaults.
            send_metadata: With a gateway, send ``cf-aig-metadata``. Its
                ``entity`` value is an entity label that can be a person's
                name, so turn this off when that must not leave the process
                or be logged by the gateway.

        Raises:
            ValueError: If a retry or backoff setting is out of range.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        if not 0 < backoff_base <= backoff_max:
            raise ValueError(
                "backoff_base must satisfy 0 < backoff_base <= backoff_max"
            )
        if not max_retry_after > 0:
            raise ValueError("max_retry_after must be positive")
        self._account_id = account_id
        self._api_token = api_token
        self._base_url = base_url
        self._gateway = gateway
        self._injected_client = client
        self._clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, httpx.AsyncClient
        ] = weakref.WeakKeyDictionary()
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._max_retry_after = max_retry_after
        self._sleep = sleep
        self._extra_headers = dict(extra_headers or {})
        self._send_metadata = send_metadata
        self._concurrency = concurrency or ConcurrencyConfig()
        self.limiter = AdaptiveLimiter(self._concurrency)

    async def __aenter__(self) -> "CloudflareTransport":
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close this loop's owned client on exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the client created for the running loop.

        A client passed in as ``client`` is never closed.
        """
        client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()

    def close(self) -> None:
        """Close every owned client from synchronous code, best effort.

        Safe to call when no loop is running; each client is closed on a
        temporary loop and failures are ignored (a client whose own loop has
        already been closed may not be able to release its connections).
        Clients are tracked weakly by loop, so those of loops that have been
        garbage-collected are already gone. Does nothing while a loop is
        running in this thread: use ``aclose()``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            if not client.is_closed:
                with contextlib.suppress(Exception):  # best effort
                    asyncio.run(client.aclose())

    def _get_client(self) -> httpx.AsyncClient:
        if self._injected_client is not None:
            return self._injected_client
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            cap = self._concurrency.max_in_flight
            client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=cap, max_keepalive_connections=cap),
            )
            self._clients[loop] = client
        return client

    def _url_and_token(self, model: str) -> tuple[str, str]:
        account_id = self._account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        token = (
            self._api_token
            or os.environ.get("CLOUDFLARE_API_TOKEN")
            or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
            or ""
        ).strip()
        if not token:
            raise AuthError(
                "No Cloudflare API token: pass api_token or set CLOUDFLARE_API_TOKEN"
            )
        if self._base_url:
            return f"{self._base_url.rstrip('/')}/ai/run/{model}", token
        if not self._gateway:
            env_base = os.environ.get("TRUSTCALL_S1_BASE_URL")
            if env_base:
                return f"{env_base.rstrip('/')}/ai/run/{model}", token
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
        if self._gateway and self._send_metadata:
            entries = {
                key: value for key in _METADATA_KEYS if (value := getattr(meta, key))
            }
            headers["cf-aig-metadata"] = json.dumps(entries)
        headers.update(self._extra_headers)
        return headers

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run one model call with retries.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing and observability context.
            timeout: Per-call timeout in seconds; ``None`` uses the client's.

        Returns:
            The response object with the REST envelope removed, plus the retry
            count, duration and HTTP status.

        Raises:
            AuthError: Missing credentials, or HTTP 401/403.
            BadRequestError: HTTP 3xx or 4xx other than auth, rate limit and
                timeout, or an unsuccessful envelope.
            RateLimitError: HTTP 429 after retries (or an over-long
                ``Retry-After``).
            ServerError: HTTP 5xx after retries (or an over-long
                ``Retry-After``).
            TransportError: A network failure after retries, any other httpx
                error (not retried), or HTTP 408 after retries.
            ModelResponseError: A 2xx response that is not a JSON object.

        Every ``TransportError`` raised from here carries ``retries``.
        """
        url, token = self._url_and_token(model)
        headers = self._headers(token, meta)
        client = self._get_client()
        post_kwargs: dict[str, Any] = {}
        if timeout is not None:
            post_kwargs["timeout"] = httpx.Timeout(
                timeout, connect=self._timeout.connect
            )
        started = time.perf_counter()
        retries = 0
        previous_delay = self._backoff_base
        while True:
            retry_after: float | None = None
            failure: TransportError
            async with self.limiter.slot(model) as generation:
                try:
                    response = await client.post(
                        url, json=payload, headers=headers, **post_kwargs
                    )
                except _RETRYABLE_ERRORS as exc:
                    failure = TransportError(f"Request to {url} failed: {exc!r}")
                    failure.__cause__ = exc
                except httpx.HTTPError as exc:
                    error = TransportError(f"Request to {url} failed: {exc!r}")
                    error.retries = retries
                    raise error from exc
                else:
                    status = response.status_code
                    if response.is_success:
                        try:
                            body = self._unwrap(response)
                        except TransportError as exc:
                            exc.retries = retries
                            raise
                        self.limiter.on_success(model)
                        return TransportResponse(
                            body=body,
                            retries=retries,
                            duration_ms=(time.perf_counter() - started) * 1000,
                            status_code=status,
                        )
                    retry_after = _parse_retry_after(
                        response.headers.get("Retry-After")
                    )
                    if status == 429:
                        self.limiter.on_rate_limited(model, generation)
                    failure = _error_for_status(response, retry_after=retry_after)
                    if status not in _RETRYABLE_STATUS and status < 500:
                        failure.retries = retries
                        raise failure
            # The slot is released; decide whether and how long to wait.
            failure.retries = retries
            if retries >= self._max_retries:
                raise failure
            if retry_after is not None and retry_after > self._max_retry_after:
                raise failure
            retries += 1
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
