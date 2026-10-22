"""Transport protocol, adaptive limiter and the clients of the providers.

Cloudflare, Experiential Labs, the Decisions API and TypeSafe.
"""

import asyncio
import contextlib
import dataclasses
import functools
import json
import math
import os
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Protocol, TypeVar, Union, cast, runtime_checkable

import httpx

from trustcall_system_one._config import ConcurrencyConfig
from trustcall_system_one._decisions import (
    DECISIONS_MODEL,
    DEFAULT_HARD_CAP,
    _reported_cost,
    from_decisions_response,
    to_decisions_request,
)
from trustcall_system_one._errors import (
    AuthError,
    BadRequestError,
    ModelResponseError,
    RateLimitError,
    ServerError,
    TransportError,
    TrustcallSystemOneError,
    UncertainOutcomeError,
    may_have_run,
)
from trustcall_system_one._types import CallKind
from trustcall_system_one._typesafe import (
    EscalationHook,
    EscalationRequest,
    RunStats,
    TypeSafeConfig,
    estimate_tokens,
    from_jev,
    gather_all,
    run_clairic,
    to_jev_clef,
)
from trustcall_system_one._version import __version__
from trustcall_system_one._wire import (
    ClairicUsage,
    ClefUsage,
    body_model,
    normalize_model_id,
    parse_clef,
    safe_loc_parts,
)

__all__ = [
    "DEFAULT_FALLBACK_ON",
    "DEFAULT_MAX_QUESTIONS",
    "AdaptiveLimiter",
    "CallKind",
    "CallMeta",
    "CloudflareTransport",
    "DecisionsTransport",
    "EscalationHook",
    "EscalationRequest",
    "ExperientialTransport",
    "FallbackTransport",
    "RoutingTransport",
    "Transport",
    "TransportResponse",
    "TypeSafeConfig",
    "TypeSafeTransport",
    "clef_fallback_from_env",
    "clef_fallback_on_from_env",
    "clef_provider_from_env",
    "max_questions",
    "provider_from_env",
    "state_token_budget",
    "system_one_transport",
]

DEFAULT_MAX_QUESTIONS = 64
"""Questions one Clef request may carry when a transport does not say otherwise
(Cloudflare takes 64; Experiential Labs 32)."""


def max_questions(transport: object, model: str) -> int:
    """How many questions one Clef request to ``model`` may carry on ``transport``.

    A transport says it with a ``max_questions`` attribute (an int) or method
    (``max_questions(model)``): Cloudflare 64, Experiential Labs 32, the Decisions
    API 32 (not documented, so the safe number), a ``RoutingTransport`` that of the
    transport serving the model and a ``FallbackTransport`` the smaller of the two
    (the fallback answers the same request). A transport that says nothing (a fake,
    a custom one) gets ``DEFAULT_MAX_QUESTIONS``.

    Args:
        transport: The transport the requests go to.
        model: The Clef model (full id or bare name).

    Returns:
        A limit of at least 1.
    """
    said = getattr(transport, "max_questions", None)
    if callable(said):
        said = said(model)
    if isinstance(said, int) and not isinstance(said, bool) and said >= 1:
        return said
    return DEFAULT_MAX_QUESTIONS


def state_token_budget(transport: object, model: str) -> int | None:
    """The most tokens of Clef state ``transport`` takes for ``model``, or ``None``.

    A transport says it with a ``state_token_budget`` attribute (an int) or method
    (``state_token_budget(model)``), as it says its ``max_questions``. Jev reads 32k
    tokens for the state and one question, so ``TypeSafeTransport`` has one;
    Cloudflare, Experiential Labs and the Decisions API say nothing (``None``: the
    library's ``Limits.clef_state_token_budget`` alone applies). A
    ``RoutingTransport`` answers for the transport serving the model and a
    ``FallbackTransport`` with the smaller of its two.

    Args:
        transport: The transport the requests go to.
        model: The Clef model (full id or bare name).

    Returns:
        A budget of at least 1, or ``None``.
    """
    said = getattr(transport, "state_token_budget", None)
    if callable(said):
        said = said(model)
    if isinstance(said, int) and not isinstance(said, bool) and said >= 1:
        return said
    return None


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
        request_id: The id the server gave the request (a correlation or request
            id header), if it sent one.
        provider: Who answered (``cloudflare``, ``experiential`` or
            ``decisions``), if the transport says.
        fallback_from: When a ``FallbackTransport`` had to use its fallback, the
            provider that failed.
        fallback_error: ... and how it failed (error class and message).
        primary_error: When a fallback answered, the error of the primary that
            came first (its charge and request id are logged by the engine).
        served_model: The model that answered when it is not the one the request
            named (the Decisions model of a fallback), so that calls are
            accounted under the model that was used.
        rerouted_from: When a ``RoutingTransport`` handed a request it could not
            send to the routed transport (images) to its default, the provider
            it was routed to; the engine records a warning.
        warnings: What the transport changed in the request on its way (a state it
            had to narrow to fit its model's context); the engine records each as a
            warning, so nothing is cut silently.
    """

    body: dict[str, Any]
    retries: int = 0
    duration_ms: float = 0.0
    status_code: int | None = None
    request_id: str | None = None
    provider: str | None = None
    fallback_from: str | None = None
    fallback_error: str | None = None
    served_model: str | None = None
    primary_error: BaseException | None = None
    rerouted_from: str | None = None
    warnings: tuple[str, ...] = ()


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
_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])


def _tagged(run: _F) -> _F:
    """Mark every error (and cancellation) a ``run`` raises with its provider.

    The engine logs a failed call with the provider that failed, and a fallback
    needs to say which of its two providers an error came from.
    """

    @functools.wraps(run)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await run(self, *args, **kwargs)
        except (asyncio.CancelledError, TrustcallSystemOneError) as exc:
            _tag(exc, self.provider)
            if getattr(exc, "served_model", None) is None:
                setattr(exc, "served_model", self._error_model())  # noqa: B010
            raise

    return cast(_F, wrapper)


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

    def _error_model(self) -> str | None:
        """The model to file a failed call under when it is not the one asked."""
        return None

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
        self,
        retries: int,
        failure: TransportError | None,
        *,
        reason: str | None = None,
    ) -> TransportError:
        """Build the error raised when ``call_deadline`` is used up."""
        message = f"call deadline of {self._call_deadline}s exceeded"
        if retries:
            message += f" after {retries} retr{'y' if retries == 1 else 'ies'}"
        if reason:
            message += f" ({reason})"
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

    provider = "cloudflare"
    max_questions = 64
    """Questions one Clef request may carry."""

    def classify_failure(self, exc: Exception) -> str | None:
        """Say whether a failure of ``run`` means that nothing was run.

        Used by ``FallbackTransport``. This transport retries timeouts, dropped
        connections, 408 and 5xx and raises the last failure, so after its retries
        a plain ``TransportError`` can be a failed connection (``"unavailable"``:
        nothing was sent) or a request that was sent and not answered
        (``"uncertain"``: it may have run). The error and its causes are looked at:
        a ``ServerError``, a 408, an ``UncertainOutcomeError`` or an httpx error
        that is not a failure to connect is ``"uncertain"``; a ``RateLimitError``
        and a failure to connect are ``"unavailable"``.
        """
        if isinstance(exc, _NEVER_FALL_BACK):
            return None
        if isinstance(exc, ModelResponseError):
            return "bad_response"
        if getattr(exc, "maybe_sent", False):
            return "uncertain"  # an earlier attempt may have run, whatever came last
        chain: list[BaseException] = []
        link: BaseException | None = exc
        while link is not None and link not in chain:
            chain.append(link)
            link = link.__cause__
        for link in chain:
            if isinstance(link, (UncertainOutcomeError, ServerError)):
                return "uncertain"
            if isinstance(link, TransportError) and link.status_code == 408:
                return "uncertain"
            if isinstance(link, _SlotTimeout):
                continue
            if isinstance(link, (httpx.HTTPError, asyncio.TimeoutError)) and not (
                isinstance(link, (*_PRE_SEND_RETRYABLE, *_PRE_SEND_FATAL))
            ):
                return "uncertain"
        return "unavailable" if isinstance(exc, TransportError) else None

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

    @_tagged
    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run one model call with retries.

        Every ``TransportError`` that comes out says in ``maybe_sent`` whether ANY
        attempt failed after the request may have reached the server (a post-send
        httpx error, a 408 or a 5xx), not only the last one: a timeout followed by
        a refused connection is a request that may have been run.

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
        sent = [False]  # sticky: an attempt may have reached the server
        try:
            return await self._attempts(model, payload, meta, timeout, sent)
        except TransportError as exc:
            exc.maybe_sent = exc.maybe_sent or sent[0]
            raise

    async def _attempts(
        self,
        model: str,
        payload: Mapping[str, Any],
        meta: CallMeta,
        timeout: float | None,
        sent: list[bool],
    ) -> TransportResponse:
        """The attempts of ``run``; ``sent[0]`` becomes true once one may have run."""
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
                        if not isinstance(
                            exc, (*_PRE_SEND_RETRYABLE, httpx.PoolTimeout)
                        ):
                            sent[0] = True
                        failure = TransportError(f"Request to {url} failed: {exc!r}")
                        failure.__cause__ = exc
                    except httpx.HTTPError as exc:
                        if not isinstance(exc, _PRE_SEND_FATAL):
                            sent[0] = True
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
                                provider="cloudflare",
                            )
                        retry_after = _parse_retry_after(
                            response.headers.get("Retry-After")
                        )
                        if status == 429:
                            self.limiter.on_rate_limited(model, generation)
                        if status == 408 or status >= 500:
                            sent[0] = True  # it may have been run
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
_CORRELATION_HEADERS = ("x-correlation-id", "x-request-id", "x-typesafe-request-id")
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


def _starved(left: float, need: float, *, waiting: bool = False) -> str:
    """Why a re-send is not made: too little of the deadline would be left."""
    when = "after the wait" if waiting else "now"
    return (
        f"{max(left, 0.0):.1f}s left {when}, less than the {need:.1f}s an attempt "
        "needs to be answered, so the request was not sent"
    )


def _tag_request_id(
    exc: TransportError | ModelResponseError, request_id: str | None
) -> None:
    """Put the request id on an error and in its message."""
    if not request_id:
        return
    exc.request_id = request_id
    if exc.args:
        exc.args = (f"{exc.args[0]} (correlation id {request_id})", *exc.args[1:])


def _tag_charge(
    exc: TrustcallSystemOneError, response: httpx.Response, model: str | None
) -> None:
    """Put the cost a response reported on the error it caused (it was charged)."""
    if exc.reported_cost is None:
        with contextlib.suppress(ValueError):
            exc.reported_cost = _reported_cost(response.json())
    if exc.served_model is None:
        exc.served_model = model


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

    **The key is inside the exceptions httpx raises.** The ``UncertainOutcomeError``
    of a transport failure chains (``__cause__``) the httpx error, and that error's
    ``request`` holds the ``Authorization`` header. The library's own messages
    never contain the key, but an error tracker that captures chained exceptions
    or local variables can send it. Scrub ``Authorization`` headers and ``xpl_``
    keys in its ``before_send`` hook (``trustcall_system_one.testing.scrub`` does
    this for text).
    """

    max_questions = 32
    """Questions one Clef request may carry: the API answers a 400 ("questions must
    map 1 through 32 ids to typed questions") above 32."""

    _unprocessed_statuses: frozenset[int] = frozenset({429})
    _overload_statuses: frozenset[int] = frozenset()
    """Statuses that tell the limiter to slow down although the request may have
    been run (so it is not retried): TypeSafe's 529."""
    """Statuses that mean the request was refused unprocessed (so, can be resent)."""
    _retry_needs_hint = True
    """Whether such a refusal is retried only when it carries a ``Retry-After``."""

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """Seconds the server asked to wait, from the response headers."""
        return _parse_retry_after(response.headers.get("Retry-After"))

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
        min_attempt_budget: float | None = None,
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
            min_attempt_budget: The least time that must be left of
                ``call_deadline`` for a *re*-send (after a connect error or a 429
                wait) to go out; otherwise the call fails with a deadline
                ``TransportError`` before sending. ``None`` (the default) means the
                read timeout the attempt would have (the call's ``timeout``, else
                the client's), so a request is never sent with less time to answer
                than it normally gets: one that is cut off early is likely to have
                been run, and charged, with its result lost. ``0`` turns the check
                off. The first attempt always goes out.
            concurrency: Concurrency limits for the adaptive limiter.
            sleep: Awaitable sleep, injectable for tests.
            clock: Monotonic clock in seconds, injectable for tests.
            extra_headers: Headers added to (and able to override) the defaults.

        Raises:
            ValueError: If a retry, backoff or budget setting is out of range.
        """
        if min_attempt_budget is not None and not (
            math.isfinite(min_attempt_budget) and min_attempt_budget >= 0
        ):
            raise ValueError(
                "min_attempt_budget must be finite and at least 0, or None"
            )
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
        self._min_attempt_budget = min_attempt_budget
        self._extra_headers = dict(extra_headers or {})

    def _attempt_budget(
        self, client: httpx.AsyncClient, timeout: float | None
    ) -> float:
        """The time a re-send needs: ``min_attempt_budget``, else its read timeout."""
        if self._min_attempt_budget is not None:
            return self._min_attempt_budget
        if timeout is not None:
            return timeout
        read = client.timeout.read
        return 0.0 if read is None else float(read)

    provider = "experiential"
    _api_name = "Experiential Labs API"

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        """Whether this transport can answer ``model`` (Clef only)."""
        try:
            return body_model(model) in _EXPERIENTIAL_MODELS
        except ValueError:
            return False

    def _served_model(self, body: Mapping[str, Any]) -> str | None:
        """The model that answered, when it is not the one asked (a fallback)."""
        return None

    def _prepare(self, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        """The body to send and what is needed to read the answer back."""
        return payload, None

    def _read_success(self, response: httpx.Response, context: Any) -> dict[str, Any]:
        """The Clef response in a 2xx."""
        return self._unwrap(response)

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

    @_tagged
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
        outgoing, context = self._prepare(payload)
        return await self._post(model, outgoing, context, timeout=timeout)

    async def _post(
        self,
        model: str,
        outgoing: Mapping[str, Any],
        context: Any,
        *,
        timeout: float | None,
    ) -> TransportResponse:
        """Send one request body with the retry policy of the class.

        ``model`` is the key of the adaptive limiter; ``context`` is handed to
        ``_read_success`` to read the answer.
        """
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
                    if retries and deadline is not None:
                        left = deadline - self._clock()
                        need = self._attempt_budget(client, timeout)
                        if left < need:
                            raise self._deadline_error(
                                retries, last_failure, reason=_starved(left, need)
                            )
                    # Built outside the try: a payload that cannot be encoded
                    # fails here, before anything is sent, and is not uncertain.
                    request = client.build_request(
                        "POST", url, json=outgoing, headers=headers, **post_kwargs
                    )
                    try:
                        response = await client.send(request)
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
                    except Exception as exc:  # anything else: it may have been sent
                        uncertain = UncertainOutcomeError(
                            f"Request to {url} failed after it may have been "
                            f"sent: {exc!r}. {_UNCERTAIN_NOTE}"
                        )
                        uncertain.retries = retries
                        raise uncertain from exc
                    else:
                        status = response.status_code
                        if response.is_success:
                            request_id = _correlation_id(response)
                            try:
                                body = self._read_success(response, context)
                            except (TransportError, ModelResponseError) as exc:
                                if isinstance(exc, TransportError):
                                    exc.retries = retries
                                _tag_request_id(exc, request_id)
                                _tag_charge(exc, response, self._served_model({}))
                                raise
                            self.limiter.on_success(model)
                            return TransportResponse(
                                body=body,
                                retries=retries,
                                duration_ms=(time.perf_counter() - started) * 1000,
                                status_code=status,
                                request_id=request_id,
                                provider=self.provider,
                                served_model=self._served_model(body),
                            )
                        retry_after = self._retry_after(response)
                        if (
                            status in self._unprocessed_statuses
                            or status in self._overload_statuses
                        ):
                            self.limiter.on_rate_limited(model, generation)
                        failure = self._error_for_status(
                            response, retry_after=retry_after
                        )
                        if status not in self._unprocessed_statuses or (
                            retry_after is None and self._retry_needs_hint
                        ):
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
            if deadline is not None:
                # Do not wait for a send that would be starved of time anyway.
                left = deadline - self._clock() - delay
                need = self._attempt_budget(client, timeout)
                if left < need:
                    raise self._deadline_error(
                        retries, failure, reason=_starved(left, need, waiting=True)
                    )
            retries += 1
            if retry_after is None:
                previous_delay = delay
            await self._sleep(delay)

    @classmethod
    def _error_for_status(
        cls, response: httpx.Response, *, retry_after: float | None
    ) -> TransportError:
        """Map a non-2xx response to an exception."""
        status = response.status_code
        body = _json_or_text(response)
        message = f"{cls._api_name} returned HTTP {status}: {_error_detail(body)}"
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
        error.request_id = correlation
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
# Decisions API
# --------------------------------------------------------------------------- #

DECISIONS_URL = "https://api.experientiallabs.ai/v1/decisions"


class DecisionsTransport(ExperientialTransport):
    """Answers Clef requests with Experiential Labs' Decisions API.

    A different model (``gpt-6-luna-decisions`` by default) behind another endpoint
    (``POST /v1/decisions``), meant as a *fallback* for Clef (see
    ``FallbackTransport``). It takes the Clef request bodies the library builds,
    translates them into a Decisions request, and translates the answer back into
    a Clef response that ``parse_clef`` accepts, so nothing above the transport
    changes. The translation is in ``trustcall_system_one._decisions``:

    * ``state`` becomes ``input``: a string as it is, an object as one
      ``## key`` section per key (strings verbatim, other values as indented JSON).
    * ``questions`` become a list in the same order, ``name`` being the question id.
    * ``choice``: its options are the choices, ids and descriptions kept.
      ``noul``: two choices, ``yes`` and ``no``, and ``noul`` is the probability of
      ``yes``. ``score``: one choice per level (``"0"``, ``"1"``, ...) and the score
      is the probability-weighted level.
    * Only the ``choice`` question type of the Decisions API has been observed, so
      every question is sent as one. Probabilities come from the API; an answer
      without any is *hard* (all the probability on the chosen option) and tagged
      ``"hard": true``. Probabilities seen so far were often 0 and 1, so thresholds
      written for Clef's graded answers behave more like yes/no here.
    * ``images`` are refused (``BadRequestError``) before anything is sent.

    The retry policy is that of ``ExperientialTransport``: one request, never
    resent once it may have arrived (``UncertainOutcomeError``), retried only
    before it was sent and after a 429 with ``Retry-After``. The key is the same
    (``EXPERIENTIAL_API_KEY``).
    """

    provider = "decisions"
    _api_name = "Experiential Labs Decisions API"

    max_questions = 32
    """Questions one request may carry. The Decisions API documents no limit, so the
    Experiential one is used, the safe number."""

    def __init__(
        self,
        *,
        decisions_model: str = DECISIONS_MODEL,
        base_url: str = DECISIONS_URL,
        decisions_hard_cap: float = DEFAULT_HARD_CAP,
        **kwargs: Any,
    ) -> None:
        """Create a transport.

        Args:
            decisions_model: The Decisions model to ask.
            base_url: The endpoint that receives the POST.
            decisions_hard_cap: The most probability (and confidence) a hard pick is
                given, in ``(0.5, 1]``. A hard pick is an answer without
                probabilities or with only (nearly: within 0.001) 0 and 1 (the shape
                the API sends), tagged ``"hard"``. The default 0.8 passes the add
                (0.55) and replace (0.6) gates and the 0.75 attribution keep, but
                sits below the 0.85 remove / set-null gate, so a bare yes cannot
                authorise a removal by itself (only while
                ``gates.remove_threshold`` stays above the cap: a lower threshold
                loses that protection); 1.0 gives it full weight.
            **kwargs: The other settings of ``ExperientialTransport`` (``api_key``,
                ``client``, ``timeout``, ``max_retries``, ``call_deadline``,
                ``min_attempt_budget``, ...).

        Raises:
            ValueError: If ``decisions_model`` is empty or a setting is out of range.
        """
        super().__init__(base_url=base_url, **kwargs)
        if not decisions_model.strip():
            raise ValueError("decisions_model must not be empty")
        if not (0.5 < decisions_hard_cap <= 1.0):
            raise ValueError("decisions_hard_cap must be in (0.5, 1]")
        self._decisions_model = decisions_model
        self._hard_cap = decisions_hard_cap

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        """Whether the request can be answered.

        That is Clef only, and nothing that has no Decisions equivalent (images)
        or would not translate.
        """
        if not super().serves(model):
            return False
        if payload is None:
            return True
        try:
            to_decisions_request(payload, model=self._decisions_model)
        except BadRequestError:
            return False
        return True

    def _served_model(self, body: Mapping[str, Any]) -> str | None:
        return str(body.get("decisions_model") or self._decisions_model)

    def _error_model(self) -> str | None:
        return self._decisions_model

    def _prepare(self, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        body, plans = to_decisions_request(payload, model=self._decisions_model)
        return body, (plans, str(payload.get("model", "clef")).strip())

    def _read_success(self, response: httpx.Response, context: Any) -> dict[str, Any]:
        plans, clef_model = context
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelResponseError(
                f"{self._api_name} returned a non-JSON success response",
                raw=response.text,
            ) from exc
        failed = isinstance(body, dict) and "answers" not in body
        if failed and (body.get("success") is False or _is_error_object(body)):
            raise BadRequestError(
                f"{self._api_name} reported failure: " + _error_detail(body),
                status_code=response.status_code,
                body=body,
                cf_errors=[body["error"]] if _is_error_object(body) else [],
            )
        return from_decisions_response(
            body, plans, clef_model=clef_model, hard_cap=self._hard_cap
        )


# --------------------------------------------------------------------------- #
# TypeSafe (Jev)
# --------------------------------------------------------------------------- #

TYPESAFE_URL = "https://api.typesafe.ai"
TYPESAFE_MODEL = "jev-latest"
_TYPESAFE_LIMITER_KEY = "typesafe"
_JEV_STATE_TOKENS = 24_000
"""Estimated tokens (``estimate_request_tokens``) of a whole Clef request the engine
may build for Jev: below the 30k and 60k that the transport itself enforces, which
leaves room for the widest question and for the estimate's error."""
_JEV_QUESTION_TOKENS = 30_000
"""Jev takes 32k tokens for the state and one question (estimated as above)."""
_JEV_REQUEST_TOKENS = 60_000
"""... and 64k for the state and all the questions of a request."""
_JEV_LONG_TOKENS = 250_000  # 262k with the long-context header


def _clean_detail(body: Any) -> Any:
    """A 422 body without the values that failed.

    ``input`` and ``ctx`` echo the offending value, which can be the user's text, and
    a ``loc`` past ``criteria`` is an option id, which can be too: the body that is
    kept on the error has ``type``, ``loc`` (cut at ``criteria``) and ``msg`` only.
    """
    if not (isinstance(body, dict) and isinstance(body.get("detail"), list)):
        return body
    cleaned: list[Any] = []
    for entry in body["detail"]:
        if isinstance(entry, dict):
            cleaned.append(
                {
                    "type": entry.get("type"),
                    "loc": safe_loc_parts(entry.get("loc", ())),
                    "msg": str(entry.get("msg"))[:200] if entry.get("msg") else None,
                }
            )
        else:
            cleaned.append(str(entry)[:100])
    return {"detail": cleaned}


def _typesafe_detail(body: Any) -> str:
    """The reason in a TypeSafe error body, without echoing the request."""
    cleaned = _clean_detail(body)
    if isinstance(cleaned, dict) and isinstance(cleaned.get("detail"), list):
        parts = []
        for entry in cleaned["detail"][:5]:
            if isinstance(entry, dict):
                parts.append(
                    f"{'.'.join(entry['loc'])}: "
                    f"{entry.get('msg') or entry.get('type') or ''}"
                )
            else:
                parts.append(str(entry))
        return "; ".join(parts)[:400]
    return _error_detail(body)


class TypeSafeTransport(ExperientialTransport):
    """Serves BOTH System One models with TypeSafe's Jev (``POST /v1/systemone``).

    Clef requests go through almost unchanged (Jev's ``noul``, ``choice`` and
    ``score`` are Clef's). Clairic requests are served by *selection* (code finds
    candidates, Jev picks), and dates, times and durations by Jev reading their
    parts: see ``docs/DESIGN.md`` ("TypeSafe-only provider") and
    ``trustcall_system_one._typesafe``. No Cloudflare credential is needed.

    The retry policy is that of ``ExperientialTransport`` (a request that may have
    reached the server is never resent: a timeout, a dropped connection, a 408 or a
    5xx, **529 Overloaded included**, raises ``UncertainOutcomeError``; a connect
    error is retried), with TypeSafe's rule for ``429 Too Many Requests``, which the
    API says to back off and retry: it is retried (honouring ``Retry-After`` and
    ``retry-after-ms``) up to ``max_retries``, within ``call_deadline``, with or
    without a hint. A 529 may have been processed (overload can strike after the
    work), so it is not resent. A Clairic call is several Jev requests: an error is
    that of the request that failed, carries the tokens of the requests that had
    answered (``partial_usage``) and counts as ``uncertain`` once any request went
    out; ``call_deadline`` bounds the whole call, not each request.

    A call reports the Jev model that answered (``TransportResponse.served_model``);
    ``usage.passes`` of a Clairic response is its number of requests and ``jev.stages``
    says which. TypeSafe reports no cost.
    """

    provider = "typesafe"
    _api_name = "TypeSafe API"
    max_questions = 64
    """Questions of a Clef request the engine may put in one request. The docs state
    no limit on the number of questions (only the context: 64k tokens per request,
    32k per question); 64 is the bound of the Clef request schema the library builds.
    The transport itself splits a request by tokens only."""
    _retry_needs_hint = False
    _overload_statuses = frozenset({529})  # slows the limiter; never retried

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        config: TypeSafeConfig | None = None,
        escalate: EscalationHook | None = None,
        long_context: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a transport.

        The key is resolved on each ``run``, so a missing one raises ``AuthError``
        there and not here.

        Args:
            api_key: TypeSafe API key; else ``TYPESAFE_API_KEY``.
            base_url: The API root; else ``TYPESAFE_BASE_URL``, else
                ``https://api.typesafe.ai`` (requests go to ``/v1/systemone``).
            model: The Jev model or alias; else ``TYPESAFE_DEFAULT_MODEL``, else
                ``jev-latest``. A versioned id (``jev-1.14.0``) pins the model, which
                matters because the library's thresholds depend on its calibration.
            config: Thresholds and limits of the Clairic stand-in (the cookbooks'
                numbers by default).
            escalate: A callable that answers a field the verification doubts with a
                stronger model (see ``EscalationHook``); ``None`` (the default)
                returns such a field as a low-confidence or null answer.
            long_context: Send ``TypeSafe-Danger-Zone-Long-Context: true`` (262k
                tokens per request and question instead of 64k and 32k; slower and
                less accurate) and let a request hold more.
            **kwargs: The other settings of ``ExperientialTransport`` (``client``,
                ``timeout``, ``max_retries``, ``call_deadline``, ``concurrency``...).

        Raises:
            ValueError: If a setting is out of range.
        """
        root = (base_url or os.environ.get("TYPESAFE_BASE_URL") or TYPESAFE_URL).strip()
        super().__init__(
            api_key=api_key, base_url=f"{root.rstrip('/')}/v1/systemone", **kwargs
        )
        self._model = (
            model or os.environ.get("TYPESAFE_DEFAULT_MODEL") or TYPESAFE_MODEL
        ).strip()
        if not self._model:
            raise ValueError("model must not be empty")
        base = config or TypeSafeConfig()
        if long_context:
            base = dataclasses.replace(
                base,
                request_token_budget=max(base.request_token_budget, 200_000),
                max_input_chars=max(base.max_input_chars, 300_000),
            )
        self._config = base
        self._escalate = escalate
        self._long_context = long_context
        self._question_tokens = (
            _JEV_LONG_TOKENS if long_context else _JEV_QUESTION_TOKENS
        )
        self._request_tokens = _JEV_LONG_TOKENS if long_context else _JEV_REQUEST_TOKENS
        self.state_token_budget: int = (
            _JEV_LONG_TOKENS - 30_000 if long_context else _JEV_STATE_TOKENS
        )
        """Estimated tokens (``estimate_request_tokens``, the engine's own unit) of a
        Clef request this transport takes, state and questions together: the engine
        fits every request to ``min(limits.clef_state_token_budget, this)``
        (``Run.clef_budget``). Jev reads 32k tokens for the state and one question
        and 64k for the state and all of them, and this is below both, so a request
        the engine fitted passes the transport's own checks untouched."""
        self._stitch_memo: dict[str, asyncio.Future[dict[str, float]]] = {}

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        """Whether Jev can answer the request: Clairic or Clef, text only."""
        try:
            name = body_model(model)
        except ValueError:
            return False
        if payload is None:
            return True
        if payload.get("images"):
            return False
        fields = payload.get("fields") if name == "clairic" else None
        if isinstance(fields, Mapping):
            return all(
                isinstance(f, Mapping) and f.get("type") != "forecast"
                for f in fields.values()
            )
        return True

    def classify_failure(self, exc: Exception) -> str | None:
        """Say whether a failure of ``run`` means that nothing was run.

        A Clairic call that failed after an earlier request of it had answered (or
        went out) is ``"uncertain"``: it was charged, whatever the last error was.
        """
        if isinstance(exc, _NEVER_FALL_BACK):
            return None
        if getattr(exc, "maybe_sent", False) or getattr(exc, "partial_usage", None):
            return "uncertain"
        return classify_failure(exc)

    def _token(self) -> str:
        token = (self._api_key or os.environ.get("TYPESAFE_API_KEY") or "").strip()
        if not token:
            raise AuthError("No TypeSafe API key: pass api_key or set TYPESAFE_API_KEY")
        return token

    def _headers(self, token: str) -> dict[str, str]:
        headers = super()._headers(token)
        if self._long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        return headers

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        millis = response.headers.get("retry-after-ms")
        if millis:
            try:
                value = float(millis) / 1000.0
            except ValueError:
                value = None
            if value is not None and math.isfinite(value):
                return max(0.0, value)
        return _parse_retry_after(response.headers.get("Retry-After"))

    def _served_model(self, body: Mapping[str, Any]) -> str | None:
        return str(body.get("model") or self._model)

    def _error_model(self) -> str | None:
        return self._model

    def _read_success(self, response: httpx.Response, context: Any) -> dict[str, Any]:
        body = self._unwrap(response)
        if context is not None:  # the questions: check the answers against them
            from_jev(body, context)
        return body

    @staticmethod
    def _unwrap(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelResponseError(
                "TypeSafe API returned a non-JSON success response",
                raw=response.text,
            ) from exc
        if not isinstance(body, dict):
            raise ModelResponseError(
                "TypeSafe API response is not a JSON object", raw=body
            )
        parse_clef(body)  # ModelResponseError if it is not a valid response
        return body

    @classmethod
    def _error_for_status(
        cls, response: httpx.Response, *, retry_after: float | None
    ) -> TransportError:
        """Map a non-2xx response: 429 is retried, 408, 529 and 5xx are uncertain."""
        status = response.status_code
        body = _clean_detail(_json_or_text(response))
        message = f"{cls._api_name} returned HTTP {status}: {_typesafe_detail(body)}"
        request_id = _correlation_id(response)
        if request_id:
            message += f" (request id {request_id})"
        error: TransportError
        if status in (401, 403):
            error = AuthError(message, status_code=status, body=body)
        elif status == 429:
            error = RateLimitError(
                message, status_code=status, body=body, retry_after=retry_after
            )
        elif status == 408 or status >= 500:  # 529 Overloaded too: it may have run
            error = UncertainOutcomeError(
                f"{message}. {_UNCERTAIN_NOTE}", status_code=status, body=body
            )
        elif 300 <= status < 500:
            entries = (
                [e for e in body["detail"] if isinstance(e, dict)]
                if isinstance(body, dict) and isinstance(body.get("detail"), list)
                else []
            )
            error = BadRequestError(
                message, status_code=status, body=body, cf_errors=entries
            )
        else:
            error = TransportError(message, status_code=status, body=body)
        error.retry_after = retry_after
        error.request_id = request_id
        return error

    @_tagged
    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run one Clairic or Clef call on Jev.

        Args:
            model: The Clairic or Clef model (full id or bare name).
            payload: The Clairic or Clef request body.
            meta: Routing context (not sent).
            timeout: Per-request timeout in seconds; ``None`` uses the client's.

        Returns:
            A bare Clef response, or a Clairic response assembled from the Jev
            requests (``usage.passes`` counts them), with the retry count, the total
            duration and the Jev model that answered as ``served_model``.

        Raises:
            AuthError: No key, or HTTP 401/403.
            BadRequestError: A model that is not Clairic or Clef, a request Jev
                cannot serve (images, ``forecast``), or HTTP 3xx/4xx (a 422's
                ``detail`` is in the message, without the user's text).
            RateLimitError: HTTP 429 that outlasted the retries.
            UncertainOutcomeError: A timeout, a failure after the request was sent,
                HTTP 408, 529 or 5xx. Never retried; the call may have been charged.
            TransportError: A failure before the request was sent.
            ModelResponseError: A 2xx that is not a valid answer to the questions.
        """
        del meta  # nothing about the call leaves the process
        try:
            name = body_model(model)
        except ValueError:
            raise BadRequestError(
                f"TypeSafeTransport serves Clairic and Clef, not {model!r}"
            ) from None
        if name != "clairic":
            return await self._clef(payload, timeout)
        return await self._clairic(payload, timeout)

    async def _clef(
        self, payload: Mapping[str, Any], timeout: float | None
    ) -> TransportResponse:
        """A Clef request: one Jev request, or several when Jev's context needs it."""
        notes: list[str] = []
        requests = to_jev_clef(
            payload,
            self._model,
            question_budget=self._question_tokens,
            request_budget=self._request_tokens,
            notes=notes,
        )
        if len(requests) == 1:
            (request,) = requests
            single = await self._post(
                _TYPESAFE_LIMITER_KEY, request, request["questions"], timeout=timeout
            )
            return (
                dataclasses.replace(single, warnings=tuple(notes)) if notes else single
            )
        started = time.perf_counter()
        answered: list[TransportResponse] = []

        async def one(r: Mapping[str, Any]) -> TransportResponse:
            done = await self._post(
                _TYPESAFE_LIMITER_KEY, r, r["questions"], timeout=timeout
            )
            answered.append(done)  # counted as it completes, whatever its siblings do
            return done

        try:
            sent = await gather_all(*(one(r) for r in requests))
        except BaseException as exc:
            error = _blame(exc)
            siblings = int(getattr(error, "cancelled_siblings", 0))
            if answered or siblings:
                error.maybe_sent = True  # type: ignore[attr-defined]
            if answered:  # those parts were run, and charged
                error.partial_usage = ClefUsage(  # type: ignore[attr-defined]
                    input_tokens=sum(
                        int(p.body["usage"]["input_tokens"]) for p in answered
                    ),
                    output_tokens=sum(
                        int(p.body["usage"]["output_tokens"]) for p in answered
                    ),
                )
                error.served_model = answered[-1].served_model  # type: ignore[attr-defined]
            _note_siblings(error, siblings)
            if error is exc:
                raise
            raise error from exc
        answers: dict[str, Any] = {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        for part in sent:
            answers.update(part.body["answers"])
            for key in usage:
                usage[key] += int(part.body["usage"][key])
        order = list(payload["questions"])
        body = {
            "model": sent[-1].body["model"],
            "answers": {q: answers[q] for q in order if q in answers},
            "usage": usage,
        }
        merged_response: TransportResponse = dataclasses.replace(
            sent[0],
            body=body,
            retries=sum(p.retries for p in sent),
            duration_ms=(time.perf_counter() - started) * 1000,
            warnings=tuple(notes),
        )
        return merged_response

    def _batches(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> list[dict[str, Mapping[str, Any]]]:
        """Split questions into requests by the context budget alone.

        There is no question-count cap here: TypeSafe states none, so the stitch,
        selection and verification requests hold as many questions as fit.
        """
        budget = self._config.request_token_budget
        base = estimate_tokens(state)
        batches: list[dict[str, Mapping[str, Any]]] = [{}]
        used = base
        for qid, question in questions.items():
            cost = estimate_tokens(question)
            current = batches[-1]
            if current and used + cost > budget:
                batches.append({})
                current = batches[-1]
                used = base
            current[qid] = question
            used += cost
        return [b for b in batches if b]

    def _annotate(self, exc: BaseException, stats: RunStats) -> BaseException:
        """Put what the earlier requests of a failed Clairic call cost on its error.

        Returns the error to raise: ``exc`` itself, or, when a request that failed
        at the same time may have been charged (``_blame``), that one.
        """
        error = _blame(exc)
        error.served_model = stats.model or self._model  # type: ignore[attr-defined]
        siblings = int(getattr(error, "cancelled_siblings", 0))
        if stats.requests or siblings:
            error.maybe_sent = True  # type: ignore[attr-defined]
        if stats.requests:
            error.partial_usage = ClairicUsage(  # type: ignore[attr-defined]
                input_tokens=stats.input_tokens,
                output_tokens=stats.output_tokens,
                passes=stats.requests,
            )
        _note_siblings(error, siblings)
        return error

    async def _clairic(
        self, payload: Mapping[str, Any], timeout: float | None
    ) -> TransportResponse:
        started = time.perf_counter()
        stats = RunStats()

        async def send(
            state: Any, questions: Mapping[str, Mapping[str, Any]], stage: str
        ) -> Mapping[str, Any]:
            request = {
                "model": self._model,
                "state": state,
                "questions": dict(questions),
            }
            sent = await self._post(
                _TYPESAFE_LIMITER_KEY, request, dict(questions), timeout=timeout
            )
            response = parse_clef(sent.body)
            # Counted the moment this request has answered: when a sibling fails
            # later, what this one cost is already in the stats (and on the error).
            stats.requests += 1
            stats.stages[stage] += 1
            stats.input_tokens += response.usage.input_tokens
            stats.output_tokens += response.usage.output_tokens
            stats.retries += sent.retries
            stats.model = response.model
            if sent.request_id:
                stats.request_ids.append(sent.request_id)
            return response.answers

        async def ask(
            state: Any, questions: Mapping[str, Mapping[str, Any]], stage: str
        ) -> Mapping[str, Any]:
            merged: dict[str, Any] = {}
            batches = self._batches(state, questions)
            for answers in await gather_all(*(send(state, b, stage) for b in batches)):
                merged.update(answers)
            return merged

        try:
            body = await asyncio.wait_for(
                run_clairic(
                    payload,
                    ask=ask,
                    stats=stats,
                    config=self._config,
                    memo=self._stitch_memo,
                    escalate=self._escalate,
                    model_name=self._model,
                ),
                self._call_deadline,
            )
        except asyncio.TimeoutError:
            note = (
                f"the Clairic call used up its {self._call_deadline}s deadline after "
                f"{stats.requests} answered request(s)"
            )
            # something may have been sent (and charged) when time ran out, so it
            # is never "nothing ran", whether or not a request had answered
            error = UncertainOutcomeError(f"{note}. {_UNCERTAIN_NOTE}")
            error.maybe_sent = True
            self._annotate(error, stats)
            raise error from None
        except BaseException as exc:
            blamed = self._annotate(exc, stats)
            if blamed is exc:
                raise
            raise blamed from exc
        return TransportResponse(
            body=body,
            retries=stats.retries,
            duration_ms=(time.perf_counter() - started) * 1000,
            status_code=200,
            request_id=stats.request_ids[0] if stats.request_ids else None,
            provider=self.provider,
            served_model=stats.model or self._model,
        )


# --------------------------------------------------------------------------- #
# Fallback
# --------------------------------------------------------------------------- #

FALLBACK_CATEGORIES: tuple[str, ...] = ("unavailable", "uncertain", "bad_response")
"""What a failure of the primary can be, for ``FallbackTransport(on=...)``."""
DEFAULT_FALLBACK_ON: frozenset[str] = frozenset({"unavailable"})
"""The categories that trigger the fallback by default: only a primary that
certainly ran nothing. ``uncertain`` (the request may have been run and charged) and
``bad_response`` (it certainly was) are opt-ins: the fallback is a second request to
the same vendor, which is what "do not auto-retry" is about."""
FallbackTrigger = Union[str, "type[Exception]"]
_NEVER_FALL_BACK = (AuthError, BadRequestError)


def _blame(exc: BaseException) -> BaseException:
    """The error to raise for a call of several requests that failed.

    ``gather_all`` leaves on ``exc`` what became of the other requests: the ones it
    cancelled and the errors of the ones that failed with it. Whenever there are
    any, the call is NOT a plain refusal of one request: a sibling may have been sent
    and charged (it timed out, or was cancelled mid-flight), and the engine must not
    contain the error (a size refusal, say) and send the siblings' questions again.

    * If a sibling's error says the request may have run (``may_have_run``: an
      uncertain outcome, a server failure, ``maybe_sent``, ``partial_usage``) and
      ``exc`` does not, THAT error is the one raised (the original is chained from
      it, and logged by the engine as a sibling).
    * Otherwise ``exc`` is raised, marked ``maybe_sent``.

    The choice is made once per error: a failure shared by several calls (the
    stitch memo) is annotated by each of them.
    """
    siblings: list[BaseException] = list(getattr(exc, "sibling_errors", ()))
    cancelled = int(getattr(exc, "cancelled_siblings", 0))
    if not siblings and not cancelled:
        return exc
    chosen: BaseException | None = getattr(exc, "_blamed", None)
    if chosen is not None:
        return chosen
    heavy = None
    if not may_have_run(exc):
        heavy = next((s for s in siblings if may_have_run(s)), None)
    if heavy is None:
        chosen = exc
    else:
        chosen = heavy
        chosen.sibling_errors = [  # type: ignore[attr-defined]
            exc,
            *(s for s in siblings if s is not heavy),
        ]
        chosen.cancelled_siblings = cancelled  # type: ignore[attr-defined]
    chosen.maybe_sent = True  # type: ignore[attr-defined]
    exc._blamed = chosen  # type: ignore[attr-defined]
    return chosen


def _note_siblings(exc: BaseException, siblings: int) -> None:
    """Say once, on a failed call's error, that other requests were cancelled.

    The text is built from the message the error had before any note, so an error
    that reaches this twice (it is annotated by every call that shares it) is not
    annotated twice.
    """
    if not siblings or not isinstance(exc, TrustcallSystemOneError) or not exc.args:
        return
    base: tuple[Any, ...] = getattr(exc, "_args_before_note", exc.args)
    exc._args_before_note = base  # type: ignore[attr-defined]
    exc.args = (
        f"{base[0]} ({siblings} other request(s) of this call were cancelled after "
        "they may have been sent; their outcome is unknown)",
        *base[1:],
    )


def classify_failure(exc: Exception) -> str | None:
    """The category of a transport failure, as far as the type says.

    * ``"uncertain"``: ``UncertainOutcomeError`` and ``ServerError``: the request
      may have been run (and charged).
    * ``"bad_response"``: ``ModelResponseError``: it was run, and answered badly.
    * ``"unavailable"``: ``RateLimitError`` and a plain ``TransportError``: a
      request that was refused, never sent or never connected.
    * ``None``: anything else (``AuthError``, ``BadRequestError``, a bug).

    An error that says a request may have gone out and been charged
    (``maybe_sent``, or ``partial_usage``: a call of several requests that failed
    part-way) is ``"uncertain"`` whatever its type, so wrappers that do not know the
    transport (``RecordingTransport``, ``RoutingTransport``) do not fall back on it.

    A transport that knows more says so with its own ``classify_failure(exc)``
    (``CloudflareTransport`` does: its failures after retries include timeouts and
    dropped connections that are not "nothing ran").
    """
    if isinstance(exc, _NEVER_FALL_BACK):
        return None
    if getattr(exc, "maybe_sent", False) or getattr(exc, "partial_usage", None):
        return "uncertain"  # a call of several requests: some went out and were charged
    if isinstance(exc, (UncertainOutcomeError, ServerError)):
        return "uncertain"
    if isinstance(exc, ModelResponseError):
        return "bad_response"
    if isinstance(exc, TransportError):  # RateLimitError, or nothing was sent
        return "unavailable"
    return None


def _provider_name(transport: object) -> str:
    return str(getattr(transport, "provider", None) or type(transport).__name__.lower())


def _tag(exc: BaseException, provider: str) -> None:
    """Say who raised an error (the first to say it wins)."""
    if getattr(exc, "provider", None) is None:
        exc.provider = provider  # type: ignore[attr-defined]


class FallbackTransport:
    """Sends a request to a fallback once if the primary could not answer it.

    ``run`` tries ``primary``. If it raises an error that ``on`` selects, the same
    request goes to ``fallback`` exactly once and its answer is returned, with
    ``provider`` and ``fallback_from`` / ``fallback_error`` set on the response so
    the call log shows who answered. Neither transport is resent by this class,
    and the built-in ones never resend a request that may have arrived.

    ``on`` is a set of categories and exception classes. The categories, which the
    primary decides for its own errors (``classify_failure``):

    * ``"unavailable"`` (**the default**): nothing was run: a refused request
      (``RateLimitError`` after the bounded retries), a failed connection, a request
      that was not sent. Falling back cannot cost a second charge.
    * ``"uncertain"`` (opt-in): the request may have been run and charged: a
      timeout, a dropped connection after sending, a 408 or 5xx
      (``UncertainOutcomeError``, ``ServerError``, and for ``CloudflareTransport`` a
      failure after the request was sent). **The same question then goes to the
      fallback, a vendor that may be the same one, so the user may pay twice and
      "do not auto-retry after a timeout" no longer holds.**
    * ``"bad_response"`` (opt-in): ``ModelResponseError``: the primary certainly ran
      and charged, and its answer was unusable.

    ``AuthError`` and ``BadRequestError`` never fall back (a rejected key or a
    request we built wrongly would fail the same way), and listing them is a
    ``ValueError``. So is ``FallbackTransport(p, p)``. A cancellation is never
    caught. If the fallback fails too, its error is raised with the primary's in
    ``primary_error`` (the engine logs both calls). A fallback that cannot serve the
    request (``serves``: not Clef, or images the Decisions API cannot take) is not
    tried, and it is the primary's error that is raised.
    """

    def __init__(
        self,
        primary: Transport,
        fallback: Transport,
        *,
        on: Iterable[FallbackTrigger] | str = DEFAULT_FALLBACK_ON,
    ) -> None:
        """Create a fallback pair.

        Args:
            primary: The transport tried first.
            fallback: The transport that gets the request if the primary fails.
            on: Categories (``"unavailable"``, ``"uncertain"``, ``"bad_response"``)
                and/or exception classes that trigger the fallback; by default only
                ``"unavailable"``.

        Raises:
            ValueError: If ``on`` names an unknown category or ``AuthError`` /
                ``BadRequestError`` (or a subclass), selects nothing, or if
                ``primary`` and ``fallback`` are the same transport.
        """
        if primary is fallback:
            raise ValueError(
                "primary and fallback are the same transport: a fallback is a "
                "different provider, not a resend"
            )
        categories: set[str] = set()
        classes: list[type[Exception]] = []
        for item in (on,) if isinstance(on, (str, type)) else on:
            if isinstance(item, str):
                if item not in FALLBACK_CATEGORIES:
                    raise ValueError(
                        f"unknown fallback trigger {item!r}; expected one of "
                        f"{list(FALLBACK_CATEGORIES)} or an exception class"
                    )
                categories.add(item)
            elif isinstance(item, type) and issubclass(item, Exception):
                if issubclass(item, _NEVER_FALL_BACK) or any(
                    issubclass(never, item) for never in _NEVER_FALL_BACK
                ):
                    raise ValueError(
                        f"{item.__name__} never falls back: it would fail the same way"
                    )
                classes.append(item)
            else:
                raise ValueError(f"not a fallback trigger: {item!r}")
        if not categories and not classes:
            raise ValueError("on selects no error")
        self._categories = frozenset(categories)
        self._classes = tuple(classes)
        self._primary = primary
        self._fallback = fallback

    def max_questions(self, model: str) -> int:
        """Questions one request may carry: the smaller limit of the two transports.

        The fallback receives the same request, so it has to take it too; a
        fallback that does not serve the model is not counted.
        """
        limit = max_questions(self._primary, model)
        serves = getattr(self._fallback, "serves", None)
        if callable(serves) and not serves(model):
            return limit
        return min(limit, max_questions(self._fallback, model))

    def state_token_budget(self, model: str) -> int | None:
        """The smaller budget of the two transports (``None``: none says one).

        Like ``max_questions``: the fallback receives the same request, so it has to
        take it too, but a fallback that does not serve the model is not counted
        (a Decisions fallback behind a Jev primary does not narrow the request for a
        model it would never be asked).
        """
        found = [state_token_budget(self._primary, model)]
        serves = getattr(self._fallback, "serves", None)
        if not (callable(serves) and not serves(model)):
            found.append(state_token_budget(self._fallback, model))
        known = [b for b in found if b is not None]
        return min(known) if known else None

    def _falls_back_on(self, exc: Exception) -> bool:
        if isinstance(exc, _NEVER_FALL_BACK):
            return False
        classify = getattr(self._primary, "classify_failure", None)
        category = classify(exc) if callable(classify) else classify_failure(exc)
        if category is not None and category in self._categories:
            return True
        return isinstance(exc, self._classes)

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run the call on the primary, else once on the fallback."""
        try:
            return await self._primary.run(model, payload, meta=meta, timeout=timeout)
        except Exception as primary_error:
            _tag(primary_error, _provider_name(self._primary))
            if not self._falls_back_on(primary_error):
                raise
            serves = getattr(self._fallback, "serves", None)
            if callable(serves) and not serves(model, payload):
                raise
            try:
                answered = await self._fallback.run(
                    model, payload, meta=meta, timeout=timeout
                )
            except asyncio.CancelledError as cancelled:
                _tag(cancelled, _provider_name(self._fallback))
                cancelled.primary_error = primary_error  # type: ignore[attr-defined]
                raise
            except Exception as fallback_error:
                _tag(fallback_error, _provider_name(self._fallback))
                fallback_error.primary_error = primary_error  # type: ignore[attr-defined]
                raise
            failure = f"{type(primary_error).__name__}: {primary_error}"
            first_error: BaseException = primary_error  # kept past the except block
        return dataclasses.replace(
            answered,
            provider=answered.provider or _provider_name(self._fallback),
            fallback_from=_provider_name(self._primary),
            fallback_error=failure[:600],
            primary_error=first_error,
        )

    def sensitive_values(self) -> tuple[str, ...]:
        """The credentials passed explicitly to the two transports."""
        values: list[str] = []
        for transport in (self._primary, self._fallback):
            getter = getattr(transport, "sensitive_values", None)
            if callable(getter):
                values.extend(getter())
        return tuple(dict.fromkeys(values))

    async def __aenter__(self) -> "FallbackTransport":
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close both transports on exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close both transports; the first error is raised after both ran."""
        first: Exception | None = None
        for transport in (self._primary, self._fallback):
            closer = getattr(transport, "aclose", None)
            if callable(closer):
                try:
                    await closer()
                except Exception as exc:
                    first = first or exc
        if first is not None:
            raise first

    def close(self) -> None:
        """Close both transports from synchronous code, best effort."""
        for transport in (self._primary, self._fallback):
            closer = getattr(transport, "close", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    closer()


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

    def max_questions(self, model: str) -> int:
        """Questions one request may carry: the limit of the transport serving it."""
        return max_questions(self.transport_for(model), model)

    def state_token_budget(self, model: str) -> int | None:
        """The state budget of the transport serving the model (``None``: none)."""
        return state_token_budget(self.transport_for(model), model)

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run the call on the transport chosen for ``model``.

        A routed transport that says it cannot serve this request because it
        carries images (the Decisions API takes text only) hands it to the default
        transport, when the default is another one that can; the response says so
        (``rerouted_from``) and the engine records a warning. Any other request the
        routed transport cannot serve (a ``forecast`` field on Jev, an invalid
        Decisions payload) is not moved: the routed transport refuses it locally,
        rather than the text going to another company without being asked.
        """
        route = self.transport_for(model)
        serves = getattr(route, "serves", None)
        rerouted: str | None = None
        if (
            route is not self._default
            and payload.get("images")
            and callable(serves)
            and not serves(model, payload)
        ):
            other = getattr(self._default, "serves", None)
            if not callable(other) or other(model, payload):
                rerouted = _provider_name(route)
                route = self._default
        reply = await route.run(model, payload, meta=meta, timeout=timeout)
        if rerouted is not None:
            reply = dataclasses.replace(reply, rerouted_from=rerouted)
        return reply

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
        """Close the routed transports' clients for the running loop.

        Every member is closed even if one fails; the first error is raised
        afterwards.
        """
        first: Exception | None = None
        for transport in self._members():
            closer = getattr(transport, "aclose", None)
            if callable(closer):
                try:
                    await closer()
                except Exception as exc:
                    first = first or exc
        if first is not None:
            raise first

    def close(self) -> None:
        """Close the routed transports from synchronous code, best effort."""
        for transport in self._members():
            closer = getattr(transport, "close", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    closer()


ClefProvider = Literal["cloudflare", "experiential", "decisions"]
Provider = Literal["cloudflare", "typesafe"]


def provider_from_env(environ: Mapping[str, str] | None = None) -> Provider:
    """Which provider serves BOTH models (Clairic and Clef), from the environment.

    ``TRUSTCALL_S1_PROVIDER`` is ``cloudflare`` (the default: Clairic and Clef on
    Cloudflare, Clef optionally elsewhere, see ``clef_provider_from_env``) or
    ``typesafe`` (TypeSafe's Jev replaces both, see ``TypeSafeTransport``). Having a
    ``TYPESAFE_API_KEY`` in the environment does not switch anything: sending your
    text to another company is an explicit choice, made with this variable or by
    passing a transport.

    Args:
        environ: The environment to read; defaults to ``os.environ``.

    Returns:
        ``"cloudflare"`` or ``"typesafe"``.

    Raises:
        ValueError: If the variable holds anything else.
    """
    env = os.environ if environ is None else environ
    value = (env.get("TRUSTCALL_S1_PROVIDER") or "").strip().lower()
    if not value or value == "cloudflare":
        return "cloudflare"
    if value == "typesafe":
        return "typesafe"
    raise ValueError(
        f"TRUSTCALL_S1_PROVIDER must be 'cloudflare' or 'typesafe', not {value!r}"
    )


def clef_provider_from_env(environ: Mapping[str, str] | None = None) -> ClefProvider:
    """Which provider serves Clef, from the environment.

    ``TRUSTCALL_S1_CLEF_PROVIDER`` (``cloudflare``, ``experiential`` or ``decisions``,
    any case) decides. With it unset or empty, Clef stays on Cloudflare: having an
    ``EXPERIENTIAL_API_KEY`` in the environment does not move it. Sending your
    text to another company is an explicit choice, made with this variable or by
    passing a transport. ``decisions`` makes the Decisions API (``DecisionsTransport``,
    a GPT model behind Experiential Labs) answer every Clef request as the primary,
    not as a fallback.

    Args:
        environ: The environment to read; defaults to ``os.environ``.

    Returns:
        ``"cloudflare"``, ``"experiential"`` or ``"decisions"``.

    Raises:
        ValueError: If the variable holds anything else.
    """
    env = os.environ if environ is None else environ
    value = (env.get("TRUSTCALL_S1_CLEF_PROVIDER") or "").strip().lower()
    if not value or value == "cloudflare":
        return "cloudflare"
    if value in ("experiential", "decisions"):
        return value  # type: ignore[return-value]
    raise ValueError(
        "TRUSTCALL_S1_CLEF_PROVIDER must be 'cloudflare', 'experiential' or "
        f"'decisions', not {value!r}"
    )


ClefFallback = Literal["decisions", "none"]


def clef_fallback_from_env(environ: Mapping[str, str] | None = None) -> ClefFallback:
    """Whether Clef has a fallback, from the environment.

    ``TRUSTCALL_S1_CLEF_FALLBACK=decisions`` adds the Decisions API as the fallback
    of Clef (see ``FallbackTransport``); unset, empty or ``none`` means no fallback.
    It is an explicit opt-in, like the provider: an ``EXPERIENTIAL_API_KEY`` alone
    never sends text to a second provider.

    Args:
        environ: The environment to read; defaults to ``os.environ``.

    Returns:
        ``"decisions"`` or ``"none"``.

    Raises:
        ValueError: If the variable holds anything else.
    """
    env = os.environ if environ is None else environ
    value = (env.get("TRUSTCALL_S1_CLEF_FALLBACK") or "").strip().lower()
    if value in ("", "none"):
        return "none"
    if value == "decisions":
        return "decisions"
    raise ValueError(
        f"TRUSTCALL_S1_CLEF_FALLBACK must be 'decisions' or 'none', not {value!r}"
    )


def clef_fallback_on_from_env(
    environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Which failures of Clef trigger the fallback, from the environment.

    ``TRUSTCALL_S1_CLEF_FALLBACK_ON`` is a comma-separated list of
    ``unavailable``, ``uncertain`` and ``bad_response`` (see ``FallbackTransport``).
    Unset or empty means ``unavailable`` only: the fallback is used when the
    primary certainly ran nothing. ``uncertain`` (a timeout or 5xx: the primary may
    have run and charged) and ``bad_response`` (it certainly did) send the same
    question to a second request, so they are opt-ins.

    Args:
        environ: The environment to read; defaults to ``os.environ``.

    Returns:
        The categories.

    Raises:
        ValueError: If the variable names anything else.
    """
    env = os.environ if environ is None else environ
    raw = env.get("TRUSTCALL_S1_CLEF_FALLBACK_ON") or ""
    names = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not names:
        return DEFAULT_FALLBACK_ON
    unknown = sorted(names - set(FALLBACK_CATEGORIES))
    if unknown:
        raise ValueError(
            f"TRUSTCALL_S1_CLEF_FALLBACK_ON must list {list(FALLBACK_CATEGORIES)}, "
            f"not {unknown}"
        )
    return frozenset(names)


def system_one_transport(
    clef: ClefProvider | None = None,
    *,
    provider: Provider | None = None,
    clef_fallback: ClefFallback | None = None,
    clef_fallback_on: Iterable[FallbackTrigger] | str | None = None,
    cloudflare: CloudflareTransport | None = None,
    experiential: ExperientialTransport | None = None,
    decisions: DecisionsTransport | None = None,
    typesafe: TypeSafeTransport | None = None,
    concurrency: ConcurrencyConfig | None = None,
) -> CloudflareTransport | RoutingTransport | TypeSafeTransport:
    """Build the usual transport: Clairic on Cloudflare, Clef where you choose.

    Three combinations are selectable (each is an explicit opt-in):

    * Cloudflare for both: the default;
    * Cloudflare Clairic with the Decisions API answering every Clef request:
      ``clef="decisions"``;
    * TypeSafe's Jev for both Clairic and Clef: ``provider="typesafe"``.

    Args:
        clef: ``"cloudflare"`` serves Clef from Cloudflare; ``"experiential"`` from
            ``ExperientialTransport``; ``"decisions"`` from ``DecisionsTransport``
            (the Decisions API with its 0.8 hard-pick cap and 32-question limit, as
            the primary). ``None`` decides from the environment (see
            ``clef_provider_from_env``).
        provider: ``"cloudflare"`` (the default) or ``"typesafe"``, which serves
            Clairic AND Clef with Jev and needs no Cloudflare credential. ``None``
            decides from the environment (see ``provider_from_env``). With
            ``"typesafe"`` a ``clef`` other than Cloudflare's and a Clef fallback are
            contradictions (Jev already answers Clef; a fallback would send the text
            elsewhere) and raise ``ValueError``.
        clef_fallback: ``"decisions"`` wraps Clef in a ``FallbackTransport`` whose
            fallback is ``DecisionsTransport`` (the Decisions API, same key); this
            sends text to Experiential Labs when the primary fails, also if Clef
            itself is on Cloudflare. ``"none"`` adds nothing. ``None`` decides from
            the environment (see ``clef_fallback_from_env``).
        clef_fallback_on: Which failures of Clef trigger the fallback: categories
            and/or exception classes (see ``FallbackTransport``). ``None`` reads
            ``TRUSTCALL_S1_CLEF_FALLBACK_ON`` (see ``clef_fallback_on_from_env``);
            the default is ``unavailable`` only.
        cloudflare: The Cloudflare transport to use; by default a new one.
        experiential: The Experiential transport to use; by default a new one.
        decisions: The Decisions transport to use; by default a new one.
        typesafe: The TypeSafe transport to use; by default a new one.
        concurrency: Concurrency limits for the transports created here.

    Returns:
        A plain ``CloudflareTransport`` when Clef is on Cloudflare without a
        fallback; a ``TypeSafeTransport`` for ``provider="typesafe"``; otherwise a
        ``RoutingTransport`` that sends Clef to the chosen transport (wrapped in the
        fallback, if any) and everything else to Cloudflare. For
        ``create_extractor(..., transport=...)``; close it with ``aclose`` or
        ``close`` when you are done.

    Raises:
        ValueError: If an argument or the environment names an unknown provider or
            fallback, or contradicts the TypeSafe-only provider or itself.
    """
    chosen = provider_from_env() if provider is None else provider
    if chosen not in ("cloudflare", "typesafe"):
        raise ValueError(f"unknown provider {chosen!r}")
    if typesafe is not None and chosen != "typesafe":
        raise ValueError(
            "a TypeSafe transport was passed but the provider is "
            f"{chosen!r}: the text would go to Cloudflare. Say it: "
            "provider='typesafe' (or TRUSTCALL_S1_PROVIDER=typesafe); the provider "
            "is never switched for you"
        )
    if chosen == "typesafe":
        others = [
            name
            for name, given in (
                ("cloudflare", cloudflare),
                ("experiential", experiential),
                ("decisions", decisions),
            )
            if given is not None
        ]
        if others:
            raise ValueError(
                f"the TypeSafe provider uses none of the {others} transports "
                "you passed: TypeSafe answers both models (pass typesafe=)"
            )
        explicit_clef = clef_provider_from_env() if clef is None else clef
        if explicit_clef != "cloudflare":
            raise ValueError(
                "the TypeSafe provider answers Clef with Jev itself: do not also "
                f"choose Clef provider {explicit_clef!r} (unset "
                "TRUSTCALL_S1_CLEF_PROVIDER)"
            )
        if (
            clef_fallback_from_env() if clef_fallback is None else clef_fallback
        ) != "none":
            raise ValueError(
                "the TypeSafe provider is TypeSafe only: a Clef fallback would send "
                "the text to another company (unset TRUSTCALL_S1_CLEF_FALLBACK)"
            )
        return typesafe or TypeSafeTransport(concurrency=concurrency)
    clef_choice = clef_provider_from_env() if clef is None else clef
    if clef_choice not in ("cloudflare", "experiential", "decisions"):
        raise ValueError(f"unknown Clef provider {clef_choice!r}")
    fallback = clef_fallback_from_env() if clef_fallback is None else clef_fallback
    if fallback not in ("decisions", "none"):
        raise ValueError(f"unknown Clef fallback {fallback!r}")
    if clef_choice == "decisions" and fallback == "decisions":
        raise ValueError(
            "Clef already is the Decisions API: a Decisions fallback would send the "
            "same request to it again (unset TRUSTCALL_S1_CLEF_FALLBACK)"
        )
    base = cloudflare or CloudflareTransport(concurrency=concurrency)
    if clef_choice == "cloudflare" and fallback == "none":
        return base
    clef_transport: Transport
    if clef_choice == "cloudflare":
        clef_transport = base
    elif clef_choice == "decisions":
        clef_transport = decisions or DecisionsTransport(concurrency=concurrency)
    else:
        clef_transport = experiential or ExperientialTransport(concurrency=concurrency)
    if fallback == "decisions":
        triggers = (
            clef_fallback_on_from_env()
            if clef_fallback_on is None
            else clef_fallback_on
        )
        clef_transport = FallbackTransport(
            clef_transport,
            decisions or DecisionsTransport(concurrency=concurrency),
            on=triggers,
        )
    return RoutingTransport({"clef": clef_transport}, default=base)
