"""Transport protocol, adaptive concurrency limiter and the Cloudflare client."""

import asyncio
import contextlib
import json
import math
import os
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import httpx

from trustcall_system_one._config import ConcurrencyConfig
from trustcall_system_one._errors import (
    AuthError,
    BadRequestError,
    ModelResponseError,
    RateLimitError,
    ServerError,
    TransportError,
    UncertainOutcomeError,
)
from trustcall_system_one._types import CallKind
from trustcall_system_one._version import __version__
from trustcall_system_one._wire import body_model, normalize_model_id, parse_clef

__all__ = [
    "AdaptiveLimiter",
    "CallKind",
    "CallMeta",
    "CloudflareTransport",
    "ExperientialTransport",
    "RoutingTransport",
    "Transport",
    "TransportResponse",
    "clef_provider_from_env",
    "system_one_transport",
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
    """Per-event-loop limiter state: occupancy and the wake-up Condition.

    The Condition is created with the state and never replaced: replacing it
    while a waiter still sleeps on the old one would lose that waiter's
    wake-ups.
    """

    in_flight: int = 0
    active: dict[str, int] = field(default_factory=dict)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
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

    What the limiter has learned (current limits, success streaks and
    generations) is shared by every event loop. Occupancy and the wake-up
    ``Condition`` are per running loop, so the caps apply per loop; state for
    loops that have since closed is dropped when a new loop first uses the
    limiter.
    """

    def __init__(self, config: ConcurrencyConfig | None = None) -> None:
        """Create a limiter.

        Args:
            config: Concurrency settings; defaults to ``ConcurrencyConfig()``.
        """
        self._config = config or ConcurrencyConfig()
        self._limits: dict[str, int] = {}
        self._streaks: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._states: dict[asyncio.AbstractEventLoop, _LoopState] = {}

    @property
    def max_in_flight(self) -> int:
        """The global cap on concurrent requests (per event loop)."""
        return self._config.max_in_flight

    @property
    def in_flight(self) -> int:
        """Number of slots currently held on the running loop (0 outside one)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return 0
        state = self._states.get(loop)
        return state.in_flight if state else 0

    def _state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        state = self._states.get(loop)
        if state is None:
            for stale in [lp for lp in list(self._states) if lp.is_closed()]:
                del self._states[stale]
            state = self._states[loop] = _LoopState()
        return state

    def limit(self, model: str) -> int:
        """Return the model's current concurrency limit."""
        model = normalize_model_id(model)
        return self._limits.get(model, self._config.limit_for(model))

    def generation(self, model: str) -> int:
        """Return how many times the model's limit has been halved."""
        return self._generations.get(normalize_model_id(model), 0)

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
        self._streaks[model] = 0
        if generation is not None and generation != self.generation(model):
            return False
        self._limits[model] = max(1, self.limit(model) // 2)
        self._generations[model] = self.generation(model) + 1
        return True

    def on_success(self, model: str) -> None:
        """Count a success; raise the limit by one every ``recover_after``."""
        if not self._config.adaptive:
            return
        model = normalize_model_id(model)
        streak = self._streaks.get(model, 0) + 1
        if streak >= self._config.recover_after:
            streak = 0
            cap = self._config.limit_for(model)
            self._limits[model] = min(cap, self.limit(model) + 1)
            self._schedule_wake()
        self._streaks[model] = streak

    def _schedule_wake(self) -> None:
        # Notifying a Condition needs its lock, which a synchronous caller
        # cannot await, so hand the notification to a task on the loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        state = self._states.get(loop)
        if state is None:
            return
        cond = state.cond

        async def notify() -> None:
            async with cond:
                cond.notify_all()

        task = loop.create_task(notify())
        state.wake_tasks.add(task)
        task.add_done_callback(state.wake_tasks.discard)

    @asynccontextmanager
    async def slot(
        self, model: str, timeout: float | None = None
    ) -> AsyncIterator[int]:
        """Hold one global slot and one slot of the model's limit.

        Args:
            model: Model id the call goes to.
            timeout: Longest time to wait for a slot, in seconds; ``None`` waits
                indefinitely.

        Raises:
            asyncio.TimeoutError: If no slot became free within ``timeout``
                (an ``_SlotTimeout``, so callers can tell the wait for a slot
                from a timeout inside the guarded block).

        Yields:
            The model's generation at acquisition, to pass back to
            ``on_rate_limited``.
        """
        model = normalize_model_id(model)
        state = self._state()
        cond = state.cond
        async with cond:
            try:
                await asyncio.wait_for(
                    cond.wait_for(
                        lambda: (
                            state.in_flight < self.max_in_flight
                            and state.active.get(model, 0) < self.limit(model)
                        )
                    ),
                    timeout,
                )
            except asyncio.TimeoutError:
                raise _SlotTimeout(timeout) from None
            state.in_flight += 1
            state.active[model] = state.active.get(model, 0) + 1
            generation = self.generation(model)
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
                self._schedule_wake()
                raise


class _SlotTimeout(asyncio.TimeoutError):
    """No request slot became free in time (as opposed to any other timeout)."""

    def __init__(self, timeout: float | None) -> None:
        super().__init__(f"no request slot became free within {timeout}s")


_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_RETRYABLE_STATUS = frozenset({408, 429})
_RETRYABLE_ERRORS = (
    asyncio.TimeoutError,
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


_T = TypeVar("_T", bound="_HttpTransport")


class _HttpTransport:
    """Shared plumbing of the HTTP transports: client per loop, limiter, deadline.

    Holds the settings that both ``CloudflareTransport`` and
    ``ExperientialTransport`` validate and use the same way: the httpx client
    that belongs to each event loop, the adaptive limiter and the per-attempt
    timeout capped by the call deadline.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None,
        timeout: httpx.Timeout,
        max_retries: int,
        backoff_base: float,
        backoff_max: float,
        max_retry_after: float,
        call_deadline: float | None,
        queue_timeout: float | None,
        concurrency: ConcurrencyConfig | None,
        sleep: Callable[[float], Awaitable[None]],
        clock: Callable[[], float],
    ) -> None:
        """Validate and store the settings (see ``CloudflareTransport``)."""
        if max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        if not 0 < backoff_base <= backoff_max:
            raise ValueError(
                "backoff_base must satisfy 0 < backoff_base <= backoff_max"
            )
        if not max_retry_after > 0:
            raise ValueError("max_retry_after must be positive")
        if call_deadline is not None and not (
            math.isfinite(call_deadline) and call_deadline > 0
        ):
            raise ValueError("call_deadline must be positive and finite, or None")
        if queue_timeout is not None and not (
            math.isfinite(queue_timeout) and queue_timeout > 0
        ):
            raise ValueError("queue_timeout must be positive and finite, or None")
        self._injected_client = client
        self._clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._max_retry_after = max_retry_after
        self._call_deadline = call_deadline
        self._queue_timeout = queue_timeout
        self._sleep = sleep
        self._clock = clock
        self._concurrency = concurrency or ConcurrencyConfig()
        self.limiter = AdaptiveLimiter(self._concurrency)

    async def __aenter__(self: _T) -> _T:
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
        """Close owned clients from synchronous code, best effort.

        Each client whose loop is not running is closed on a temporary loop
        and failures are ignored (a client whose own loop has already been
        closed may not be able to release its connections). Clients whose loop
        is running are left alone, so this does nothing from inside a running
        loop: use ``aclose()`` there. It is not thread-safe against a loop
        that starts running in another thread while it executes.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return  # asyncio.run cannot nest; the caller should await aclose()
        for loop, client in list(self._clients.items()):
            if loop.is_running():
                continue
            del self._clients[loop]
            if not client.is_closed:
                with contextlib.suppress(Exception):  # best effort
                    asyncio.run(client.aclose())

    def _get_client(self) -> httpx.AsyncClient:
        if self._injected_client is not None:
            return self._injected_client
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            # First use from this loop: forget clients of closed loops. They
            # cannot be awaited any more, so dropping the reference is all
            # that can be done.
            for stale in [lp for lp in list(self._clients) if lp.is_closed()]:
                del self._clients[stale]
            cap = self._concurrency.max_in_flight
            client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=cap, max_keepalive_connections=cap),
            )
            self._clients[loop] = client
        return client

    def _post_kwargs(
        self,
        client: httpx.AsyncClient,
        timeout: float | None,
        deadline: float | None,
    ) -> dict[str, Any] | None:
        """Build the per-attempt ``timeout`` keyword, capped by the deadline.

        Returns:
            The keyword arguments for ``client.post``, or ``None`` when no time
            is left.
        """
        remaining = None if deadline is None else deadline - self._clock()
        if remaining is not None and remaining <= 0:
            return None
        if timeout is None:
            read = client.timeout.read
            if remaining is None or (read is not None and read <= remaining):
                return {}  # the client's own timeout is the tighter one
            capped = remaining
        elif remaining is not None and remaining < timeout:
            capped = remaining
        else:
            return {"timeout": httpx.Timeout(timeout, connect=self._timeout.connect)}
        connect = self._timeout.connect
        return {
            "timeout": httpx.Timeout(
                capped, connect=capped if connect is None else min(connect, capped)
            )
        }

    def _deadline_error(
        self, retries: int, failure: TransportError | None
    ) -> TransportError:
        """Build the error raised when ``call_deadline`` is used up."""
        message = f"call deadline of {self._call_deadline}s exceeded"
        if retries:
            message += f" after {retries} retr{'y' if retries == 1 else 'ies'}"
        if failure is not None:
            message += f"; last error: {failure}"
        error = TransportError(
            message,
            status_code=failure.status_code if failure else None,
            body=failure.body if failure else None,
        )
        error.retries = retries
        if failure is not None:
            error.retry_after = failure.retry_after
            error.__cause__ = failure
        return error


class CloudflareTransport(_HttpTransport):
    """Calls System One models over the Cloudflare REST API or AI Gateway.

    Retries 408, 429, 5xx, timeouts, network errors and remote protocol errors
    up to ``max_retries`` times, waiting for ``Retry-After`` when present and
    otherwise using decorrelated jitter. The limiter slot is released while
    waiting, so retries honour a lowered limit and do not starve other models.
    If the server asks for a wait longer than ``max_retry_after`` the call
    fails at once with ``retry_after`` set on the error.

    Event loops: one transport is meant to be used from one long-lived loop
    (the extractor's sync bridge keeps a persistent loop thread for this). The
    httpx client it creates is bound to the loop that first used it, and the
    limiter's occupancy counters are per loop, so the concurrency caps apply
    per loop. Close an owned client inside its loop, with ``await
    transport.aclose()`` or ``async with CloudflareTransport() as transport:``.
    Using the transport from a further loop is supported (it gets its own
    client and starts from the limits already learned, and state belonging to
    loops that have closed is dropped), but a client left open when its loop
    closes cannot be closed afterwards. A client passed in as ``client`` is
    used for every loop and is the caller's to manage.
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
        call_deadline: float | None = 300.0,
        queue_timeout: float | None = None,
        concurrency: ConcurrencyConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
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
            call_deadline: Total wall-clock budget in seconds for one ``run()``
                across every attempt and backoff sleep; ``None`` for no budget.
                The budget starts when the first limiter slot is acquired, so
                time spent queueing for the first attempt does not count and a
                request that has not been sent never fails on the deadline. When
                the next sleep or attempt would not fit, the call fails with a
                ``TransportError`` that says the deadline was exceeded. Each
                attempt's timeout is capped by the budget that is left.
            queue_timeout: Longest wait for a limiter slot, per attempt, in
                seconds; ``None`` (the default) waits indefinitely. On expiry
                the call fails with a ``TransportError``.
            concurrency: Concurrency limits for the adaptive limiter.
            sleep: Awaitable sleep, injectable for tests.
            clock: Monotonic clock in seconds used for ``call_deadline``,
                injectable for tests.
            extra_headers: Headers added to (and able to override) the defaults.
            send_metadata: With a gateway, send ``cf-aig-metadata``. Its
                ``entity`` value is an entity label that can be a person's
                name, so turn this off when that must not leave the process
                or be logged by the gateway.

        Raises:
            ValueError: If a retry or backoff setting is out of range.
        """
        super().__init__(
            client=client,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            max_retry_after=max_retry_after,
            call_deadline=call_deadline,
            queue_timeout=queue_timeout,
            concurrency=concurrency,
            sleep=sleep,
            clock=clock,
        )
        self._account_id = account_id
        self._api_token = api_token
        self._base_url = base_url
        self._gateway = gateway
        self._extra_headers = dict(extra_headers or {})
        self._send_metadata = send_metadata

    def sensitive_values(self) -> tuple[str, ...]:
        """The account id and API token that were passed in explicitly.

        ``RecordingTransport`` redacts them from what it writes. (Those read from
        the environment are redacted by value anyway.)
        """
        return tuple(v for v in (self._account_id, self._api_token) if v)

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
        started = time.perf_counter()
        deadline: float | None = None  # set when the first slot is acquired
        retries = 0
        previous_delay = self._backoff_base
        last_failure: TransportError | None = None
        while True:
            retry_after: float | None = None
            failure: TransportError
            try:
                async with self.limiter.slot(model, self._queue_timeout) as generation:
                    if deadline is None and self._call_deadline is not None:
                        deadline = self._clock() + self._call_deadline
                    post_kwargs = self._post_kwargs(client, timeout, deadline)
                    if post_kwargs is None:
                        raise self._deadline_error(retries, last_failure)
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
            except _SlotTimeout:
                timed_out = TransportError(
                    f"timed out after {self._queue_timeout}s waiting for a request slot"
                )
                timed_out.retries = retries
                raise timed_out from None
            # The slot is released; decide whether and how long to wait.
            failure.retries = retries
            last_failure = failure
            if retries >= self._max_retries:
                raise failure
            if retry_after is not None and retry_after > self._max_retry_after:
                raise failure
            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(
                    self._backoff_max,
                    random.uniform(self._backoff_base, previous_delay * 3),
                )
            if deadline is not None and self._clock() + delay >= deadline:
                raise self._deadline_error(retries, failure)
            retries += 1
            if retry_after is None:
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


# --------------------------------------------------------------------------- #
# Experiential Labs
# --------------------------------------------------------------------------- #

EXPERIENTIAL_URL = "https://api.experientiallabs.ai/v1/systemone"
_EXPERIENTIAL_MODELS = frozenset({"clef", "clef-flash"})
_CORRELATION_HEADERS = ("x-correlation-id", "x-request-id")
# Failures that happen before a single byte of the request is sent: the server
# cannot have seen the request, so trying again cannot be charged twice.
_PRE_SEND_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout)
# Also before the request is sent, but not worth another attempt.
_PRE_SEND_FATAL = (httpx.PoolTimeout, httpx.ProxyError, httpx.UnsupportedProtocol)
_UNCERTAIN_NOTE = (
    "The outcome is unknown: the server may have run the model and may have "
    "charged for it. The request was not retried."
)


def _is_error_object(body: Any) -> bool:
    return isinstance(body, dict) and isinstance(body.get("error"), dict)


def _error_detail(body: Any) -> str:
    """The reason in an error body (``{"error": {"message": ...}}`` and kin)."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if isinstance(error, str) and error:
            return error[:300]
        errors = _cf_errors(body)
        if errors:
            return str(errors[0].get("message", errors[0]))[:300]
        for key in ("message", "detail"):
            if body.get(key):
                return str(body[key])[:300]
    return str(body)[:300]


def _correlation_id(response: httpx.Response) -> str | None:
    """The request id the server put on a response (not a secret), if it did."""
    for name in _CORRELATION_HEADERS:
        value = response.headers.get(name)
        if value:
            return str(value)[:100]
    return None


class ExperientialTransport(_HttpTransport):
    """Calls Clef through Experiential Labs (``POST /v1/systemone``).

    It takes the Clef request bodies the library already builds (``model`` is
    ``"clef"`` or ``"clef-flash"``) and returns the bare Clef response
    (``model``, ``answers``, ``usage``), validated against the Clef response
    wire model. Only Clef is served: use ``RoutingTransport`` or
    ``system_one_transport`` to keep Clairic on Cloudflare.

    **A request is never retried once it may have reached the server.** A
    timeout can leave a call that was run and charged but whose result was lost,
    and retrying it would pay again. So:

    * A timeout, a read or write error, a connection dropped after the request
      was sent, a 408 and any 5xx raise ``UncertainOutcomeError``. Exactly one
      request was sent; the error says its outcome is unknown and that it may
      have been charged. Whether to ask again is the caller's decision.
    * Only these are retried (up to ``max_retries``, within ``call_deadline``):
      a connect error or connect timeout, where no request was sent, with
      jittered backoff; and a 429 that carries a ``Retry-After`` of at most
      ``max_retry_after`` seconds (the server refused the request unprocessed).
      A 429 without ``Retry-After`` is not retried.
    * 401 and 403 raise ``AuthError``, other 4xx and 3xx ``BadRequestError``,
      429 ``RateLimitError``, and a 2xx that is not a valid Clef response
      ``ModelResponseError``.

    The ``CallMeta`` of a call is not sent anywhere (the entity label can be a
    person's name). Event loops, the adaptive limiter and ``aclose`` behave as
    for ``CloudflareTransport``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = EXPERIENTIAL_URL,
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
        max_retries: int = 6,
        backoff_base: float = 0.25,
        backoff_max: float = 8.0,
        max_retry_after: float = 60.0,
        call_deadline: float | None = 300.0,
        queue_timeout: float | None = None,
        concurrency: ConcurrencyConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Create a transport.

        The key is resolved on each ``run``, so a missing one raises ``AuthError``
        there and not here.

        Args:
            api_key: Experiential Labs API key; else ``EXPERIENTIAL_API_KEY``.
            base_url: The endpoint that receives the POST.
            client: An ``httpx.AsyncClient`` to use. It is never closed here.
            timeout: Timeout for clients this transport creates (read 60 s and
                connect 10 s by default).
            max_retries: Retries after the first attempt, for the cases that are
                safe to retry only (at least 0).
            backoff_base: Smallest delay before retrying a connect error, in
                seconds (positive).
            backoff_max: Largest jittered delay (at least ``backoff_base``).
            max_retry_after: Longest ``Retry-After`` to honour; a longer one
                fails the call at once with ``RateLimitError``.
            call_deadline: Total wall-clock budget in seconds across attempts and
                waits, or ``None`` for no budget (see ``CloudflareTransport``).
            queue_timeout: Longest wait for a limiter slot, in seconds, or
                ``None`` to wait indefinitely.
            concurrency: Concurrency limits for the adaptive limiter.
            sleep: Awaitable sleep, injectable for tests.
            clock: Monotonic clock in seconds, injectable for tests.
            extra_headers: Headers added to (and able to override) the defaults.

        Raises:
            ValueError: If a retry or backoff setting is out of range.
        """
        super().__init__(
            client=client,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            max_retry_after=max_retry_after,
            call_deadline=call_deadline,
            queue_timeout=queue_timeout,
            concurrency=concurrency,
            sleep=sleep,
            clock=clock,
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._extra_headers = dict(extra_headers or {})

    def sensitive_values(self) -> tuple[str, ...]:
        """The API key that was passed in explicitly.

        ``RecordingTransport`` redacts it from what it writes. (A key read from
        ``EXPERIENTIAL_API_KEY`` is redacted by value anyway.)
        """
        return (self._api_key,) if self._api_key else ()

    def _token(self) -> str:
        token = (self._api_key or os.environ.get("EXPERIENTIAL_API_KEY") or "").strip()
        if not token:
            raise AuthError(
                "No Experiential Labs API key: pass api_key or set EXPERIENTIAL_API_KEY"
            )
        return token

    def _headers(self, token: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": f"trustcall-system-one/{__version__}",
        }
        headers.update(self._extra_headers)
        return headers

    @staticmethod
    def _check_model(model: str) -> None:
        try:
            served = body_model(model)
        except ValueError:
            served = model
        if served not in _EXPERIENTIAL_MODELS:
            raise BadRequestError(
                f"ExperientialTransport serves Clef only, not {model!r}; route "
                "other models elsewhere (see RoutingTransport)"
            )

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run one Clef call: one request, retried only when that is safe.

        Args:
            model: ``@cf/cloudflare/clef``, ``@cf/cloudflare/clef-flash`` or the
                bare names (the request body carries the selector).
            payload: Request body.
            meta: Routing and observability context (not sent).
            timeout: Per-call timeout in seconds; ``None`` uses the client's.

        Returns:
            The bare, validated Clef response plus the retry count, duration and
            HTTP status.

        Raises:
            AuthError: No key, or HTTP 401/403.
            BadRequestError: A model other than Clef, HTTP 3xx or 4xx other than
                auth and rate limit, or an unsuccessful envelope.
            RateLimitError: HTTP 429 that was not retried or whose retries ran out.
            UncertainOutcomeError: A timeout, a failure after the request was
                sent, HTTP 408 or 5xx. Never retried; the call may have been
                charged.
            TransportError: A failure before the request was sent that outlasted
                its retries or the deadline.
            ModelResponseError: A 2xx that is not a valid Clef response.

        Every ``TransportError`` raised from here carries ``retries``.
        """
        del meta  # nothing about the call leaves the process
        self._check_model(model)
        token = self._token()
        url = self._base_url
        headers = self._headers(token)
        client = self._get_client()
        started = time.perf_counter()
        deadline: float | None = None  # set when the first slot is acquired
        retries = 0
        previous_delay = self._backoff_base
        last_failure: TransportError | None = None
        while True:
            retry_after: float | None = None
            failure: TransportError
            try:
                async with self.limiter.slot(model, self._queue_timeout) as generation:
                    if deadline is None and self._call_deadline is not None:
                        deadline = self._clock() + self._call_deadline
                    post_kwargs = self._post_kwargs(client, timeout, deadline)
                    if post_kwargs is None:
                        raise self._deadline_error(retries, last_failure)
                    try:
                        response = await client.post(
                            url, json=payload, headers=headers, **post_kwargs
                        )
                    except _PRE_SEND_RETRYABLE as exc:
                        failure = TransportError(
                            f"Could not connect to {url}; the request was not "
                            f"sent: {exc!r}"
                        )
                        failure.__cause__ = exc
                    except _PRE_SEND_FATAL as exc:
                        fatal = TransportError(
                            f"Request to {url} was not sent: {exc!r}"
                        )
                        fatal.retries = retries
                        raise fatal from exc
                    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                        uncertain = UncertainOutcomeError(
                            f"Request to {url} failed after it may have been "
                            f"sent: {exc!r}. {_UNCERTAIN_NOTE}"
                        )
                        uncertain.retries = retries
                        raise uncertain from exc
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
                        failure = self._error_for_status(
                            response, retry_after=retry_after
                        )
                        if status != 429 or retry_after is None:
                            failure.retries = retries
                            raise failure
            except _SlotTimeout:
                timed_out = TransportError(
                    f"timed out after {self._queue_timeout}s waiting for a request slot"
                )
                timed_out.retries = retries
                raise timed_out from None
            # Only a connect error or a 429 with Retry-After gets here: the
            # request was not sent, or was refused unprocessed.
            failure.retries = retries
            last_failure = failure
            if retries >= self._max_retries:
                raise failure
            if retry_after is not None and retry_after > self._max_retry_after:
                raise failure
            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(
                    self._backoff_max,
                    random.uniform(self._backoff_base, previous_delay * 3),
                )
            if deadline is not None and self._clock() + delay >= deadline:
                raise self._deadline_error(retries, failure)
            retries += 1
            if retry_after is None:
                previous_delay = delay
            await self._sleep(delay)

    @staticmethod
    def _error_for_status(
        response: httpx.Response, *, retry_after: float | None
    ) -> TransportError:
        """Map a non-2xx response to an exception."""
        status = response.status_code
        body = _json_or_text(response)
        message = f"Experiential Labs API returned HTTP {status}: {_error_detail(body)}"
        correlation = _correlation_id(response)
        if correlation:
            message += f" (correlation id {correlation})"
        error: TransportError
        if status in (401, 403):
            error = AuthError(message, status_code=status, body=body)
        elif status == 429:
            error = RateLimitError(
                message, status_code=status, body=body, retry_after=retry_after
            )
        elif status == 408 or status >= 500:
            error = UncertainOutcomeError(
                f"{message}. {_UNCERTAIN_NOTE}", status_code=status, body=body
            )
        elif 300 <= status < 500:
            entries = [body["error"]] if _is_error_object(body) else []
            error = BadRequestError(
                message, status_code=status, body=body, cf_errors=entries
            )
        else:
            error = TransportError(message, status_code=status, body=body)
        error.retry_after = retry_after
        return error

    @staticmethod
    def _unwrap(response: httpx.Response) -> dict[str, Any]:
        """Return the bare Clef response from a 2xx body, validated.

        Accepts the bare ``{model, answers, usage}`` object, and unwraps a
        ``{success, result}`` (or ``{data}``) envelope around it. An envelope that
        reports ``success: false`` or an ``error`` becomes a ``BadRequestError``.
        """
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelResponseError(
                "Experiential Labs API returned a non-JSON success response",
                raw=response.text,
            ) from exc
        if isinstance(body, dict) and "answers" not in body:
            if body.get("success") is False or _is_error_object(body):
                raise BadRequestError(
                    "Experiential Labs API reported failure: " + _error_detail(body),
                    status_code=response.status_code,
                    body=body,
                    cf_errors=[body["error"]] if _is_error_object(body) else [],
                )
            for key in ("result", "data"):
                if isinstance(body.get(key), dict):
                    body = body[key]
                    break
        if not isinstance(body, dict):
            raise ModelResponseError(
                "Experiential Labs API response is not a JSON object", raw=body
            )
        parse_clef(body)  # ModelResponseError if it is not a valid Clef response
        return body


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def _family(name: str) -> str:
    """``clef`` for ``clef`` and ``clef-flash``; the name itself otherwise."""
    return "clef" if name.startswith("clef") else name


class RoutingTransport:
    """Sends each call to a transport chosen by model.

    ``routes`` maps a model name to a transport and ``default`` serves every other
    model. A key is a bare name (``"clairic"``, ``"clef"``, ``"clef-flash"``) or a
    full ``@cf/cloudflare/...`` id. A ``"clef"`` key covers both ``clef`` and
    ``clef-flash``, unless ``"clef-flash"`` has a route of its own. The call is
    passed on unchanged, ``CallMeta`` included.

    Example:
        Clairic on Cloudflare, Clef on Experiential Labs::

            transport = RoutingTransport(
                {"clef": ExperientialTransport()}, default=CloudflareTransport()
            )
            extractor = create_extractor([Person], transport=transport)

    ``aclose`` and ``close`` close every transport it routes to, so a transport
    that is shared with other code should not be closed through it.
    """

    def __init__(self, routes: Mapping[str, Transport], default: Transport) -> None:
        """Create a router.

        Args:
            routes: Model name (or full id) to the transport that serves it.
            default: The transport for models without a route.

        Raises:
            ValueError: If a key is not a System One model, or two keys name the
                same model.
        """
        table: dict[str, Transport] = {}
        for key, transport in routes.items():
            try:
                name = body_model(key)
            except ValueError:
                raise ValueError(
                    f"cannot route {key!r}: expected clairic, clef or clef-flash"
                ) from None
            if name in table:
                raise ValueError(f"more than one route for {name!r}")
            table[name] = transport
        self._routes = table
        self._default = default

    def transport_for(self, model: str) -> Transport:
        """The transport that serves ``model`` (full id or bare name)."""
        try:
            name = body_model(model)
        except ValueError:
            return self._default
        route = self._routes.get(name) or self._routes.get(_family(name))
        return self._default if route is None else route

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run the call on the transport chosen for ``model``."""
        return await self.transport_for(model).run(
            model, payload, meta=meta, timeout=timeout
        )

    def _members(self) -> list[Transport]:
        members: list[Transport] = []
        for transport in (*self._routes.values(), self._default):
            if not any(transport is seen for seen in members):
                members.append(transport)
        return members

    def sensitive_values(self) -> tuple[str, ...]:
        """The credentials passed explicitly to the transports it routes to."""
        values: list[str] = []
        for transport in self._members():
            getter = getattr(transport, "sensitive_values", None)
            if callable(getter):
                values.extend(getter())
        return tuple(dict.fromkeys(values))

    async def __aenter__(self) -> "RoutingTransport":
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the routed transports on exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the routed transports' clients for the running loop."""
        for transport in self._members():
            closer = getattr(transport, "aclose", None)
            if callable(closer):
                await closer()

    def close(self) -> None:
        """Close the routed transports from synchronous code, best effort."""
        for transport in self._members():
            closer = getattr(transport, "close", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    closer()


ClefProvider = Literal["cloudflare", "experiential"]


def clef_provider_from_env(environ: Mapping[str, str] | None = None) -> ClefProvider:
    """Which provider serves Clef, from the environment.

    ``TRUSTCALL_S1_CLEF_PROVIDER`` (``cloudflare`` or ``experiential``, any case)
    decides. With it unset or empty, Clef goes to Experiential when
    ``EXPERIENTIAL_API_KEY`` is set and to Cloudflare otherwise.

    Args:
        environ: The environment to read; defaults to ``os.environ``.

    Returns:
        ``"cloudflare"`` or ``"experiential"``.

    Raises:
        ValueError: If the variable holds anything else.
    """
    env = os.environ if environ is None else environ
    value = (env.get("TRUSTCALL_S1_CLEF_PROVIDER") or "").strip().lower()
    if not value:
        has_key = bool((env.get("EXPERIENTIAL_API_KEY") or "").strip())
        return "experiential" if has_key else "cloudflare"
    if value == "cloudflare":
        return "cloudflare"
    if value == "experiential":
        return "experiential"
    raise ValueError(
        "TRUSTCALL_S1_CLEF_PROVIDER must be 'cloudflare' or 'experiential', "
        f"not {value!r}"
    )


def system_one_transport(
    clef: ClefProvider | None = None,
    *,
    cloudflare: CloudflareTransport | None = None,
    experiential: ExperientialTransport | None = None,
    concurrency: ConcurrencyConfig | None = None,
) -> CloudflareTransport | RoutingTransport:
    """Build the usual transport: Clairic on Cloudflare, Clef where you choose.

    Args:
        clef: ``"cloudflare"`` returns a plain ``CloudflareTransport``;
            ``"experiential"`` returns a ``RoutingTransport`` that sends Clef to
            ``ExperientialTransport`` and everything else to Cloudflare. ``None``
            decides from the environment (see ``clef_provider_from_env``).
        cloudflare: The Cloudflare transport to use; by default a new one.
        experiential: The Experiential transport to use; by default a new one.
        concurrency: Concurrency limits for the transports created here.

    Returns:
        A transport for ``create_extractor(..., transport=...)``. Close it with
        ``aclose`` or ``close`` when you are done.

    Raises:
        ValueError: If ``clef`` or the environment names an unknown provider.
    """
    provider = clef_provider_from_env() if clef is None else clef
    if provider not in ("cloudflare", "experiential"):
        raise ValueError(f"unknown Clef provider {provider!r}")
    base = cloudflare or CloudflareTransport(concurrency=concurrency)
    if provider == "cloudflare":
        return base
    clef_transport = experiential or ExperientialTransport(concurrency=concurrency)
    return RoutingTransport({"clef": clef_transport}, default=base)
