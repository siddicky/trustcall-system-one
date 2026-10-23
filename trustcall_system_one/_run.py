"""Per-invoke context: transport calls, usage, call log and warnings."""

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union

from trustcall_system_one._config import (
    CorefConfig,
    DiscoveryConfig,
    EscalationConfig,
    GateConfig,
    Limits,
    MatchConfig,
    PassesConfig,
    PresenceConfig,
    RepairConfig,
    VerifyConfig,
)
from trustcall_system_one._errors import (
    AuthError,
    BadRequestError,
    RequestTooLargeError,
    TrustcallSystemOneError,
    exception_chain,
    is_request_rejection,
    is_size_rejection,
)
from trustcall_system_one._render import (
    RenderedState,
    clef_state,
    estimate_request_tokens,
    fit_clef_state,
    reanchor,
    state_spans,
)
from trustcall_system_one._types import CallRecord, Usage, _reported_cost
from trustcall_system_one._wire import (
    CLAIRIC,
    CLEF,
    ChoiceAnswer,
    ChoiceQuestion,
    ClairicField,
    ClairicOptions,
    ClairicRequest,
    ClairicResponse,
    ClairicUsage,
    ClefRequest,
    ClefUsage,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    body_model,
    parse_clairic,
    parse_clef,
)
from trustcall_system_one.openai_transport import OPENAI_DEFAULT_MODEL
from trustcall_system_one.transport import (
    CallMeta,
    Transport,
    TransportResponse,
    max_questions,
    prefers_keep_uncertain,
    state_token_budget,
)

__all__ = [
    "CANCELLED_AFTER_SEND",
    "FALLBACK_WARNING",
    "HARD_PICKS_WARNING",
    "Answer",
    "Question",
    "Run",
    "gather",
]

_FALLBACK_FIELDS_SHOWN = 8
FALLBACK_WARNING = "clef_fallback_used"
"""Start of the warning that says a fallback answered."""
HARD_PICKS_WARNING = "clef_hard_picks"
"""Start of the warning of a run whose PRIMARY Clef provider (Decisions, say) gave hard
picks: answers without probabilities, or with only 0 and 1, whose probability and
confidence are capped (``DecisionsTransport(decisions_hard_cap=...)``). Not a
fallback."""

CANCELLED_AFTER_SEND = "cancelled after send; outcome unknown"
"""``CallRecord.error`` of a call cancelled while it may have been on the wire."""

_CLEF_CONTEXT_TOKENS = 65_536
_TRUNCATION_RATIO = 0.97
_STATE_SHARE = 0.5
"""Share of the request budget a state may keep when its questions have to share it."""
_MAX_SPLIT = 3
"""Most requests a Clef request is split into before its state is narrowed instead."""
_REJECTION_SHOWN = 160
_MAX_SAME_REFUSAL = 8
"""How often one contained refusal may repeat in a run before it is raised: a refusal
that every request meets is not a one-off, and the run would only be a string of
warnings."""

Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]  # noqa: UP007
"""Any Clef answer."""
Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]  # noqa: UP007
"""Any Clef question."""

T = TypeVar("T")


def _shown(exc: BaseException) -> str:
    """An error as a warning shows it: its class and a clipped message."""
    text = " ".join(str(exc).split())
    if len(text) > _REJECTION_SHOWN:
        text = text[: _REJECTION_SHOWN - 1] + "\u2026"
    return f"{type(exc).__name__}: {text}"


_REQUEST_ID = re.compile(
    r"\s*\((?:request|correlation|trace)\s+id\b[^)]*\)", re.IGNORECASE
)


def _refusal_signature(exc: BadRequestError) -> str:
    """What makes two refusals "the same" for the circuit breaker.

    The status and the message with the figures taken out (``Request requires 69639
    tokens``) and WITHOUT the id the server gave the request: transports put it in
    the message (``(request id 3f9a...)``, ``(correlation id ...)``), it is different
    every time, and a breaker that counted it would never trip.
    """
    text = str(exc)
    request_id = getattr(exc, "request_id", None)
    if request_id:
        text = text.replace(str(request_id), "")
    text = _REQUEST_ID.sub("", text)
    return re.sub(r"\d+", "N", f"{exc.status_code} {text}")


_CONTEXT_CODES = frozenset(
    {"context_length_exceeded", "string_above_max_length", "max_tokens_exceeded"}
)


def _is_context_refusal(exc: BadRequestError) -> bool:
    """Whether a 400 says the request was too large (the escalation can go without)."""
    return any(entry.get("code") in _CONTEXT_CODES for entry in exc.cf_errors)


def _pack(
    chunk: Sequence[tuple[str, Question]], costs: Sequence[float], room: float
) -> list[list[tuple[str, Question]]]:
    """Group questions, in order, so that each group's cost is at most ``room``."""
    groups: list[list[tuple[str, Question]]] = []
    used = 0.0
    for item, cost in zip(chunk, costs, strict=True):
        if not groups or used + cost > room:
            groups.append([])
            used = 0.0
        groups[-1].append(item)
        used += cost
    return groups


def _narrowable(state: Any) -> bool:
    """Whether a state is the structured kind whose ``input`` can be narrowed."""
    return isinstance(state, dict) and isinstance(state.get("input"), str)


def _being_cancelled() -> bool:
    """Whether the current task has a cancellation request that is not delivered yet.

    Python 3.11+ counts the requests (``Task.cancelling``). 3.10 has no such
    counter; there ``gather`` does not need one, because it waits with
    ``asyncio.wait`` and so a ``CancelledError`` raised by that wait is by
    construction the caller's own cancellation (a child's cancellation only shows
    up as a finished task).
    """
    task = asyncio.current_task()
    cancelling = getattr(task, "cancelling", None)
    return cancelling is not None and cancelling() > 0


async def _stop(tasks: Sequence["asyncio.Future[Any]"]) -> None:
    """Cancel whatever is still running and wait until all of it has stopped."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _error(task: "asyncio.Future[Any]") -> BaseException | None:
    """A finished task's own error (``None`` for a result or a cancellation)."""
    return None if task.cancelled() else task.exception()


async def gather(*awaitables: Awaitable[T]) -> list[T]:
    """Run awaitables concurrently; if one raises, cancel the rest and re-raise.

    Unlike a bare ``asyncio.gather`` this does not leave the siblings of a failed
    call running in the background (and so still spending requests) after the
    error has propagated, and the error that propagates is the one that was
    raised: a sibling that ends up cancelled (by this very cleanup, or because
    something it awaited was cancelled) never hides it behind a
    ``CancelledError``.

    The caller's own cancellation always wins. A real error is substituted for a
    cancelled sibling only while the calling task is not being cancelled itself:
    the wait below is ``asyncio.wait``, which raises ``CancelledError`` in the
    caller only when the caller was cancelled (so on Python 3.10, which lacks
    ``Task.cancelling``, that is the signal), and on 3.11+ a cancellation request
    that has not been delivered yet (``cancelling() > 0``) also stops the
    substitution.
    """
    tasks = [asyncio.ensure_future(item) for item in awaitables]
    pending: set[asyncio.Future[Any]] = set(tasks)
    finished: set[asyncio.Future[Any]] = set()
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            if any(t.cancelled() or t.exception() is not None for t in done):
                finished = done
                break
    except BaseException:  # the caller was cancelled (or interrupted)
        await _stop(tasks)
        raise
    if pending:  # a child failed or was cancelled: stop the others
        await _stop(tasks)
        failure = next((e for t in tasks if t in finished and (e := _error(t))), None)
        if failure is None and not _being_cancelled():
            # a sibling's cancellation got here first: the real error is behind it
            failure = next((e for t in tasks if (e := _error(t))), None)
        if failure is not None:
            raise failure from None
        raise asyncio.CancelledError
    # every task has finished: read EVERY error, not only the first, so that none is
    # left "never retrieved" (asyncio logs that when the task is collected)
    errors = [_error(t) for t in tasks]
    failure = next((e for e in errors if e is not None), None)
    if failure is not None:
        raise failure from None
    if any(t.cancelled() for t in tasks):
        raise asyncio.CancelledError
    return [t.result() for t in tasks]


def _fallback_provider(answer: Any) -> str | None:
    """Who answered, when it was a fallback (``provider`` is an extra key)."""
    extra = getattr(answer, "model_extra", None) or {}
    provider = extra.get("provider")
    return str(provider) if provider else None


def _partitions(response: ClairicResponse) -> list[dict[str, Any]]:
    """What each provider answered, for a response that was merged from several.

    A ``FieldTypeRoutingTransport`` puts ``routing.partitions`` in the body it
    returns (so a recording and its replay carry it too). Anything that does not look
    like that is ignored: the response is then logged as one call.
    """
    routing = (response.model_extra or {}).get("routing")
    found = routing.get("partitions") if isinstance(routing, dict) else None
    if not isinstance(found, list) or len(found) < 2:
        return []
    parts: list[dict[str, Any]] = []
    for item in found:
        if not isinstance(item, dict) or not isinstance(item.get("usage"), dict):
            return []
        usage = item["usage"]
        if not all(
            isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)
            for k in ("input_tokens", "output_tokens")
        ):
            return []
        parts.append(
            {
                "provider": item.get("provider"),
                "served_model": item.get("served_model"),
                "fields": item["fields"] if isinstance(item.get("fields"), int) else 0,
                "usage": usage,
                "duration_ms": float(item.get("duration_ms") or 0.0),
                "retries": int(item.get("retries") or 0),
                "status_code": item.get("status_code"),
                "request_id": item.get("request_id"),
                "fallback_from": item.get("fallback_from"),
            }
        )
    return parts


def _is_hard(answer: Any) -> bool:
    """Whether the answer came without probabilities (a fallback's bare pick)."""
    return bool((getattr(answer, "model_extra", None) or {}).get("hard"))


@dataclass(frozen=True)
class _Sent:
    """A successful transport response with its timing."""

    body: dict[str, Any]
    retries: int
    duration_ms: float
    status_code: int | None
    request_id: str | None = None
    provider: str | None = None
    fallback_from: str | None = None
    fallback_error: str | None = None
    served_model: str | None = None


def memo_key(model: str, payload: Mapping[str, Any]) -> str:
    """The key of a Clef request in ``Run.clef_memo``: a hash of its canonical JSON.

    The keys of every object are sorted, so two requests that differ only in the
    order of their keys are one request; the QUESTIONS are kept in the order asked
    (Clef's answer to a question depends on the ones around it, so a different order
    is a different request). A hash, not the request, is what the run holds: a
    request carries the whole state and the images, and the memo outlives them.

    Args:
        model: The model the request goes to.
        payload: The request body (``ClefRequest.to_payload()``).

    Returns:
        A hex SHA-256 digest.
    """
    body = {k: v for k, v in payload.items() if k != "questions"}
    questions = [[qid, q] for qid, q in (payload.get("questions") or {}).items()]
    canonical = json.dumps(
        {"model": model, "request": body, "questions": questions},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Run:
    """Everything one ``invoke`` shares: configuration, transport and accounting.

    Attributes:
        transport: Where requests go.
        clairic_model: Full id of the extraction model.
        clef_model: Full id of the model that answers decisions and verification.
        gate_model: Full id of the model that gates changes (updates).
        verify: Verification settings.
        presence: Presence-gate thresholds.
        gates: Answer-level thresholds.
        passes: How Clairic fields are grouped into requests.
        repair: Settings of the validation repair loop.
        limits: Structural limits (including the Clef state budget).
        discovery: Settings of entity discovery (``Many``).
        coref: Settings of mention clustering (``Many``).
        match: Settings of matching entities to existing records (``Many``).
        images: Images sent with every request that carries the user's input.
            Each request model validates the documented size limits before it is
            sent.
        reference_date: ISO date Clairic resolves relative dates against.
        strict: Raise instead of returning an invalid record or a truncated state.
        enable_inserts: Whether an item the input mentions that matches none of the
            existing items of a nested list is added to it (a nested list the
            record holds no items of takes them regardless).
        enable_item_removals: Whether the items of a nested list that the input
            says to remove are removed.
        usage: Token and request totals so far.
        calls: One record per transport call, failed calls included.
        warnings: Notes about anything that was degraded, without duplicates.
        list_kinds: What kind of items a ``list[str]`` field holds (``"items"`` or
            ``"statements"``), asked of Clef once per schema and field in this run.
        clef_memo: The answers to the Clef requests sent so far, by the hash of the
            request (``memo_key``): a byte-identical request is answered from here,
            not sent again.
        rejected: The hashes of the requests (Clef and Clairic) a provider refused
            as a request (HTTP 400, 413 or 422): the same request is not sent again,
            it is treated as refused again.
        refusals: How often each contained refusal happened (by its message with the
            figures taken out); past ``_MAX_SAME_REFUSAL`` the next one is raised.
        largest_sent: The largest estimated size (``estimate_request_tokens``) of a
            Clef request put on the wire, for tests and diagnostics.
        fallback_used: How many requests a fallback transport answered after
            its primary failed (also reported as the ``clef_fallback_used`` warning).
        fallback_hard: How many of those answers had no probabilities.
        hard_picks: How many answers of the PRIMARY Clef provider were hard picks
            (also reported as the ``clef_hard_picks`` warning, which a run with a
            fallback never mixes up with ``clef_fallback_used``).
        hard_kinds: Those hard picks by call kind.
        fallback_fields: What the fallback's answers were used for (the call
            kind, schema and field), also when an answer is replayed from
            ``clef_memo``.
        fallback_route: ``(fallback provider, primary provider)`` of the warning.
        escalation: The opt-in escalation tier (``EscalationConfig``); off by default.
        escalation_calls: Escalation calls made so far (all kinds), against
            ``escalation.max_calls_per_invoke``.
        escalation_by_kind: ... and per kind, against ``escalation.limits``.
        escalation_memo: Single-flight results of the escalation calls that several
            entities share (the rows of one nested list, the entities of one schema).
    """

    transport: Transport
    clairic_model: str = CLAIRIC
    clef_model: str = CLEF
    gate_model: str = CLEF
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    passes: PassesConfig = field(default_factory=PassesConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    limits: Limits = field(default_factory=Limits)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    coref: CorefConfig = field(default_factory=CorefConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    images: Sequence[Any] | None = None
    reference_date: str | None = None
    strict: bool = False
    enable_inserts: bool = False
    enable_item_removals: bool = False
    usage: Usage = field(default_factory=Usage)
    calls: list[CallRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    list_kinds: "dict[tuple[type, str], asyncio.Future[str]]" = field(
        default_factory=dict
    )
    clef_memo: "dict[str, dict[str, Answer]]" = field(default_factory=dict)
    rejected: set[str] = field(default_factory=set)
    refusals: dict[str, int] = field(default_factory=dict)
    largest_sent: float = 0.0
    fallback_used: int = 0
    fallback_hard: int = 0
    hard_picks: int = 0
    hard_kinds: dict[str, int] = field(default_factory=dict)
    fallback_fields: dict[str, None] = field(default_factory=dict)
    fallback_route: tuple[str, str] | None = None
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    escalation_calls: int = 0
    escalation_by_kind: dict[str, int] = field(default_factory=dict)
    escalation_memo: "dict[Any, asyncio.Future[Any]]" = field(default_factory=dict)

    def warn(self, message: str) -> None:
        """Record a warning once."""
        if message not in self.warnings:
            self.warnings.append(message)

    # ------------------------------------------------------------------ #
    # Calls
    # ------------------------------------------------------------------ #

    def _record(
        self,
        model: str,
        meta: CallMeta,
        n_items: int,
        *,
        usage: Any = None,
        duration_ms: float = 0.0,
        retries: int = 0,
        status_code: int | None = None,
        error: str | None = None,
        request_id: str | None = None,
        provider: str | None = None,
        fallback_from: str | None = None,
        cost: float | None = None,
    ) -> None:
        """Append one ``CallRecord`` (and the usage, for a successful call)."""
        counted: dict[str, int] = {}
        cost_usd: float | None = cost
        if usage is not None:
            self.usage.add_call(model, meta.kind, usage)
            counted = {
                key: value
                for key, value in usage.model_dump().items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
            cost_usd = _reported_cost(usage)
        self.calls.append(
            CallRecord(
                model=model,
                kind=meta.kind,
                schema=meta.schema,
                entity=meta.entity,
                field=meta.field,
                scope=meta.scope,
                n_items=n_items,
                usage=counted,
                duration_ms=duration_ms,
                retries=retries,
                error=error,
                status_code=status_code,
                request_id=request_id,
                cost_usd=cost_usd,
                provider=provider,
                fallback_from=fallback_from,
            )
        )

    def _record_primary(
        self, model: str, meta: CallMeta, n_items: int, exc: BaseException
    ) -> None:
        """Log the primary's failure that came before a failed fallback.

        A ``FallbackTransport`` whose fallback fails too attaches the primary's
        error to it (``primary_error``): both calls reached a provider, and either
        may have been charged, so both are logged, each with its provider.
        """
        primary = getattr(exc, "primary_error", None)
        if primary is not None:
            self._log_primary(model, meta, n_items, primary)

    def _record_siblings(
        self, model: str, meta: CallMeta, n_items: int, exc: BaseException
    ) -> None:
        """Log the other requests of a failed call of several (TypeSafe's split).

        Those that failed with it (a timeout is an uncertain outcome: it may have
        been charged) are logged with their own error, and those that were cancelled
        in flight as ``CANCELLED_AFTER_SEND``, so a charge is never invisible just
        because another request's error is the one that was raised.
        """
        for sibling in getattr(exc, "sibling_errors", ()):
            self._log_primary(model, meta, n_items, sibling)
        for _ in range(int(getattr(exc, "cancelled_siblings", 0))):
            self._record(
                getattr(exc, "served_model", None) or model,
                meta,
                n_items,
                error=CANCELLED_AFTER_SEND,
                provider=getattr(exc, "provider", None),
            )

    def _log_primary(
        self, model: str, meta: CallMeta, n_items: int, primary: BaseException
    ) -> None:
        """Log a primary's failed call, with what it may have cost.

        Its status, retries, request id and provider are recorded, and a cost its
        response reported (a bad but charged answer) is the call's ``cost_usd`` and
        is added to the spend.
        """
        charged_model = getattr(primary, "served_model", None) or model
        spent = getattr(primary, "reported_cost", None)
        self._record(
            charged_model,
            meta,
            n_items,
            retries=getattr(primary, "retries", 0),
            status_code=getattr(primary, "status_code", None),
            error=f"{type(primary).__name__}: {primary}",
            request_id=getattr(primary, "request_id", None),
            provider=getattr(primary, "provider", None),
            cost=spent,
            # a primary whose call of several requests failed part-way still used tokens
            usage=getattr(primary, "partial_usage", None),
        )
        if spent is not None:
            self.usage.add_spend(charged_model, meta.kind, spent)

    async def _send(
        self,
        model: str,
        payload: dict[str, Any],
        meta: CallMeta,
        n_items: int,
        transport: Transport | None = None,
    ) -> _Sent:
        """Send one request; a failure is logged first and then re-raised.

        A request that is cancelled (a sibling failed, or the caller gave up) may
        already have been sent, and charged, so it is logged too, as
        ``CANCELLED_AFTER_SEND``, before the cancellation goes on. When a fallback
        was in use, the primary's failure is logged as well.
        """
        started = time.perf_counter()
        try:
            response = await (transport or self.transport).run(
                model, payload, meta=meta
            )
        except asyncio.CancelledError as cancelled:
            self._record_primary(model, meta, n_items, cancelled)
            self._record_siblings(model, meta, n_items, cancelled)
            self._record(
                getattr(cancelled, "served_model", None) or model,
                meta,
                n_items,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=CANCELLED_AFTER_SEND,
                provider=getattr(cancelled, "provider", None),
                usage=getattr(cancelled, "partial_usage", None),
            )
            raise
        except Exception as exc:
            self._record_primary(model, meta, n_items, exc)
            self._record_siblings(model, meta, n_items, exc)
            charged_model = getattr(exc, "served_model", None) or model
            spent = getattr(exc, "reported_cost", None)
            self._record(
                charged_model,
                meta,
                n_items,
                duration_ms=(time.perf_counter() - started) * 1000,
                retries=getattr(exc, "retries", 0),
                status_code=getattr(exc, "status_code", None),
                error=f"{type(exc).__name__}: {exc}",
                request_id=getattr(exc, "request_id", None),
                provider=getattr(exc, "provider", None),
                cost=spent,
                # a call of several requests that failed part-way still used tokens
                usage=getattr(exc, "partial_usage", None),
            )
            if spent is not None:  # charged for a response that could not be used
                self.usage.add_spend(charged_model, meta.kind, spent)
            raise
        if response.fallback_from is not None:
            self._note_fallback(model, meta, n_items, response, started)
        for note in response.warnings:  # what the transport changed on the way
            self.warn(note)
        if response.rerouted_from is not None:
            self.warn(
                f"request_rerouted: {response.rerouted_from} cannot serve a request "
                f"with images, so it went to {response.provider or 'the default'}"
            )
        return _Sent(
            body=response.body,
            retries=response.retries,
            duration_ms=response.duration_ms or (time.perf_counter() - started) * 1000,
            status_code=response.status_code,
            request_id=response.request_id,
            provider=response.provider,
            fallback_from=response.fallback_from,
            fallback_error=response.fallback_error,
            served_model=response.served_model,
        )

    def _note_fallback(
        self,
        model: str,
        meta: CallMeta,
        n_items: int,
        response: TransportResponse,
        started: float,
    ) -> None:
        """Log the primary's failed attempt and count the fallback in a warning.

        The primary may have run the request (an uncertain outcome), so its
        failure is a ``CallRecord`` of its own, with the error it raised and the
        provider that raised it; ``clef_fallback_used`` says how many requests the
        fallback answered.
        """
        self.fallback_used += 1
        self.fallback_route = (str(response.provider), str(response.fallback_from))
        if response.primary_error is not None:
            self._log_primary(model, meta, n_items, response.primary_error)
        else:
            self._record(
                model,
                meta,
                n_items,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=response.fallback_error,
                provider=response.fallback_from,
            )
        self._note_fallback_use(meta)

    def _note_fallback_use(self, meta: CallMeta) -> None:
        """Remember what a fallback answer was used for and refresh the warning."""
        where = ".".join(part for part in (meta.schema, meta.field) if part)
        self.fallback_fields[f"{meta.kind}:{where}" if where else meta.kind] = None
        self._refresh_fallback_warning()

    def _refresh_fallback_warning(self) -> None:
        provider, origin = self.fallback_route or ("the fallback", "the primary")
        message = (
            f"{FALLBACK_WARNING}: {self.fallback_used} Clef request(s) were answered "
            f"by the fallback ({provider}) because {origin} could not answer them"
        )
        if self.fallback_hard:
            message += (
                f"; {self.fallback_hard} hard pick(s) (no or saturated probabilities, "
                "confidence capped)"
            )
        fields = list(self.fallback_fields)
        if fields:
            shown = ", ".join(fields[:_FALLBACK_FIELDS_SHOWN])
            more = len(fields) - _FALLBACK_FIELDS_SHOWN
            message += f"; used for: {shown}" + (f", +{more} more" if more > 0 else "")
        self.warnings[:] = [
            w for w in self.warnings if not w.startswith(FALLBACK_WARNING)
        ]
        self.warnings.append(message)

    def _refresh_hard_warning(self, provider: str) -> None:
        kinds = ", ".join(
            f"{kind} {count}"
            for kind, count in sorted(
                self.hard_kinds.items(), key=lambda i: (-i[1], i[0])
            )[:_FALLBACK_FIELDS_SHOWN]
        )
        message = (
            f"{HARD_PICKS_WARNING}: {self.hard_picks} Clef answer(s) from {provider} "
            "were hard picks (no or saturated probabilities, confidence capped below "
            f"the removal gate); by call kind: {kinds}"
        )
        self.warnings[:] = [
            w for w in self.warnings if not w.startswith(HARD_PICKS_WARNING)
        ]
        self.warnings.append(message)

    def _unparsable(
        self, model: str, meta: CallMeta, n_items: int, sent: _Sent, exc: Exception
    ) -> None:
        self._record(
            model,
            meta,
            n_items,
            duration_ms=sent.duration_ms,
            retries=sent.retries,
            status_code=sent.status_code,
            error=f"{type(exc).__name__}: {exc}",
            request_id=sent.request_id,
            provider=sent.provider,
            fallback_from=sent.fallback_from,
        )

    async def call_clairic(
        self,
        fields: Mapping[str, ClairicField],
        state: RenderedState | str,
        *,
        meta: CallMeta,
    ) -> ClairicResponse:
        """Send one Clairic request and return the parsed response.

        The request always asks for ``state_text`` back (use ``anchor`` to
        re-locate the input in it), carries ``reference_date`` and the images,
        and is logged as a ``CallRecord``.

        Args:
            fields: Clairic fields keyed by wire name.
            state: The rendered state, or a plain string.
            meta: Routing context; ``meta.kind`` is the call kind.

        Returns:
            The parsed response.

        A request the provider refuses as such (HTTP 400, 413 or 422: too large,
        malformed) does not raise: it is logged, a ``request_rejected`` warning is
        recorded, and the response has no answers (every field then resolves as
        ``missing_answer``). The same request is never sent again. In ``strict`` mode
        the error is raised.

        Raises:
            TransportError: If the call failed in any other way (after it was
                logged): a rejected key, a rate limit, a server failure, an
                uncertain outcome.
            ModelResponseError: If the response does not match the schema (after
                it was logged).
        """
        text = state.text if isinstance(state, RenderedState) else state
        request = ClairicRequest(
            model=body_model(self.clairic_model),
            state=text,
            fields=dict(fields),
            images=list(self.images) if self.images else None,
            reference_date=self.reference_date,
            options=ClairicOptions(return_state_text=True),
        )
        payload = request.to_payload()
        key = memo_key(self.clairic_model, payload)
        if key in self.rejected:  # refused once: it would be refused again
            return self._unanswered_clairic(text)
        try:
            sent = await self._send(self.clairic_model, payload, meta, len(fields))
        except BadRequestError as exc:
            if not self._contains(exc):
                raise
            # this request is refused (too large): its fields come back
            # without an answer (`missing_answer`) and the rest of the run goes on;
            # the same request is never sent a second time
            self.rejected.add(key)
            self._warn_rejected(meta, len(fields), exc)
            return self._unanswered_clairic(text)
        try:
            response = parse_clairic(sent.body)
        except Exception as exc:
            self._unparsable(self.clairic_model, meta, len(fields), sent, exc)
            raise
        partitions = _partitions(response)
        if partitions:
            # several providers answered (a split by field type): each one's tokens
            # are filed under its own model and provider, and they add up to the
            # merged response's usage
            for part in partitions:
                self._record(
                    part["served_model"] or self.clairic_model,
                    meta,
                    part["fields"],
                    usage=ClairicUsage.model_validate(part["usage"]),
                    duration_ms=part["duration_ms"],
                    retries=part["retries"],
                    status_code=part["status_code"],
                    request_id=part["request_id"],
                    provider=part["provider"],
                    fallback_from=part["fallback_from"],
                )
            return response
        # a stand-in provider's calls are filed under the model that answered
        self._record(
            sent.served_model or self.clairic_model,
            meta,
            len(fields),
            usage=response.usage,
            duration_ms=sent.duration_ms,
            retries=sent.retries,
            status_code=sent.status_code,
            request_id=sent.request_id,
            provider=sent.provider,
            fallback_from=sent.fallback_from,
        )
        return response

    def _unanswered_clairic(self, text: str) -> ClairicResponse:
        """The stand-in for a Clairic request that was refused: no answers at all."""
        return ClairicResponse(
            model=self.clairic_model,
            answers={},
            truncated=False,
            state_text=text,
            usage=ClairicUsage(input_tokens=0, output_tokens=0),
        )

    def anchor(
        self, rendered: RenderedState, response: ClairicResponse
    ) -> RenderedState | None:
        """Re-locate the input in the ``state_text`` a response reports.

        Args:
            rendered: The state that was sent.
            response: Clairic's response to it.

        Returns:
            The state to map the response's spans with, or ``None`` (after a
            warning) when the returned ``state_text`` does not contain the input,
            in which case spans are unknown.
        """
        anchored = reanchor(rendered, response.state_text)
        if anchored is None:
            self.warn(
                "state_text_mismatch: Clairic returned a state without the input "
                "block; span offsets are unknown"
            )
        return anchored

    def clef_budget(self, model: str | None = None) -> int:
        """The effective Clef request budget, in estimated tokens.

        The ONE place that combines ``limits.clef_state_token_budget`` with what the
        transport says it takes (``transport.state_token_budget``: an int attribute or
        a ``(model)`` method, read defensively like ``max_questions``; absent or
        ``None`` means no limit of its own). Jev reads 32k tokens, so
        ``TypeSafeTransport`` advertises 24k and the effective budget is
        ``min(56k, 24k)``. A routing or fallback transport answers for the model
        behind it, so the budget depends on the model the request goes to. Everything
        that sizes a Clef request asks this.

        Args:
            model: The Clef model the request goes to. Without one (a state built
                before it is known which model will read it) the smallest budget of
                the models this run uses (``clef_model`` and ``gate_model``): a state
                that fits that fits whichever reads it.
        """
        knob = self.limits.clef_state_token_budget
        models = (
            [model]
            if model
            else list(dict.fromkeys((self.clef_model, self.gate_model)))
        )
        found = [state_token_budget(self.transport, m) for m in models]
        return min([knob, *(b for b in found if b is not None)])

    @property
    def keeps_uncertain(self) -> bool:
        """Whether an uncertain attribution may stand after the repair (an insert).

        ``repair.keep_uncertain`` when it was set (``True`` or ``False`` always
        wins); otherwise on exactly when the provider that answers the attribution
        question (``clef_model``, kind ``attribution``) and the one that answers the
        repair and the isolated re-ask (``clairic_model``, kinds ``repair`` and
        ``reask``) all say they prefer it (``prefers_keep_uncertain``: TypeSafe
        does, the rest do not).
        """
        chosen = self.repair.keep_uncertain
        if chosen is not None:
            return chosen
        asked = (
            (self.clef_model, "attribution"),
            (self.clairic_model, "repair"),
            (self.clairic_model, "reask"),
        )
        return all(prefers_keep_uncertain(self.transport, m, k) for m, k in asked)

    def clef_batch(self, model: str | None = None) -> int:
        """Most questions one Clef request to ``model`` may carry on this transport.

        64 on Cloudflare, 32 on Experiential Labs and the Decisions API, and the
        limit of the transport that would serve the model behind a routing or a
        fallback (``transport.max_questions``). Everything that groups Clef
        questions asks this instead of assuming 64.
        """
        return max_questions(self.transport, model or self.clef_model)

    async def call_clef(
        self,
        questions: Mapping[str, Question],
        state: Any,
        *,
        meta: CallMeta,
        model: str | None = None,
        decisive: bool = False,
    ) -> dict[str, Answer]:
        """Send Clef questions in requests the transport accepts; merge the answers.

        More questions than one request may carry (``clef_batch``: 64 on Cloudflare,
        32 on Experiential Labs) are split into several requests that are issued
        concurrently; each is logged as its own ``CallRecord``.

        Args:
            questions: Questions keyed by id.
            state: The structured (or string) state, shared by every question.
            meta: Routing context; ``meta.kind`` is the call kind.
            model: Model id; defaults to ``clef_model``.
            decisive: The answers decide whether a value is kept (attribution, the
                novelty check, the record-presence check). With
                ``verify.isolate_decisions`` each such question gets a request of
                its own, because Clef's answer depends on the other questions in
                its request.

        Returns:
            Answers keyed by question id.

        Every request is estimated first (``estimate_request_tokens``, state and
        questions) and kept within ``limits.clef_state_token_budget``: split by
        questions, or its ``input`` narrowed to windows around the spans the state
        cites. A request the provider refuses as such (HTTP 400, 413, 422) is
        contained like Clairic's: warning, no answers for its questions, never sent
        again (a size rejection of several questions is first sent once more as
        two smaller requests). ``strict`` raises instead.

        Raises:
            TransportError: If a call failed in any other way (after it was
                logged): a rejected key, a rate limit, a server failure, an
                uncertain outcome (never retried here).
            ModelResponseError: If a response does not match the schema.
        """
        model = model or self.clef_model
        items = list(questions.items())
        if not items:
            return {}
        width = (
            1 if decisive and self.verify.isolate_decisions else self.clef_batch(model)
        )
        chunks = [items[i : i + width] for i in range(0, len(items), width)]
        parts = await gather(
            *(self._clef_chunk(model, chunk, state, meta) for chunk in chunks)
        )
        merged: dict[str, Answer] = {}
        for part in parts:
            merged.update(part)
        return merged

    async def _clef_chunk(
        self,
        model: str,
        chunk: list[tuple[str, Question]],
        state: Any,
        meta: CallMeta,
    ) -> dict[str, Answer]:
        """Fit one chunk of questions to the request budget, then send it.

        The request is estimated (``estimate_request_tokens``: the state AND its
        questions) before anything is sent. One that is over
        ``limits.clef_state_token_budget`` is split into several requests when its
        questions are what is large, and has its ``input`` narrowed to windows
        around the spans the state cites when the state is; if it still does not fit
        it is dropped as a rejected request (its questions go unanswered, with a
        warning) and nothing is sent.
        """
        try:
            plans = self._fit_request(chunk, state, meta, model)
        except RequestTooLargeError as exc:
            if self.strict:
                raise
            isolated = exc.question_id
            if len(chunk) > 1 and any(q == isolated for q, _ in chunk):
                # ONE question cannot fit even alone: refuse it, and fit the others
                # again without it (they are asked together, not in halves)
                self._warn_rejected(meta, 1, exc)
                rest = [item for item in chunk if item[0] != isolated]
                self.warn(self._isolated(meta, str(isolated), len(rest)))
                return await self._clef_chunk(model, rest, state, meta)
            self._warn_rejected(meta, len(chunk), exc)
            return {}
        if len(plans) == 1:
            return await self._clef_once(model, plans[0][0], plans[0][1], meta)
        parts = await gather(
            *(self._clef_once(model, group, shown, meta) for group, shown in plans)
        )
        merged: dict[str, Answer] = {}
        for part in parts:
            merged.update(part)
        return merged

    @staticmethod
    def _isolated(meta: CallMeta, question_id: str, siblings: int) -> str:
        return (
            f"clef_request_isolated: question {question_id!r} of a {meta.kind} "
            f"request was too large to send and was refused alone; its "
            f"{siblings} sibling question(s) were sent without it"
        )

    def _fit_request(
        self,
        chunk: list[tuple[str, Question]],
        state: Any,
        meta: CallMeta,
        model: str | None = None,
    ) -> list[tuple[list[tuple[str, Question]], Any]]:
        """Plan the requests of a chunk so that each is within the budget.

        Returns:
            ``[(questions, state), ...]``: one entry when the chunk fits as it is.

        Raises:
            RequestTooLargeError: If no split and no narrowing makes it fit.
        """
        budget = self.clef_budget(model)
        if estimate_request_tokens(state, dict(chunk)) <= budget:
            return [(chunk, state)]
        empty = estimate_request_tokens("", {})
        costs = [
            estimate_request_tokens("", {qid: question}) - empty
            for qid, question in chunk
        ]
        where = ".".join(p for p in (meta.schema, meta.field) if p)
        label = f"{meta.kind} ({where})" if where else meta.kind
        widest = max(range(len(chunk)), key=costs.__getitem__)
        if empty + costs[widest] > budget:
            # no state, however narrow, makes this question fit: it alone is refused
            refusal = RequestTooLargeError(
                f"a {label} question ({chunk[widest][0]!r}) is estimated at "
                f"{empty + costs[widest]:.0f} tokens on its own, over the budget "
                f"of {budget}"
            )
            refusal.question_id = chunk[widest][0]
            raise refusal
        base = estimate_request_tokens(state)
        room = budget - base
        if room >= max(costs):
            groups = _pack(chunk, costs, room)
            if len(groups) <= _MAX_SPLIT or not _narrowable(state):
                self.warn(
                    f"clef_request_split: a {label} request of {len(chunk)} "
                    f"question(s) was estimated over the budget of {budget} "
                    f"tokens and was sent as {len(groups)} requests"
                )
                return [(group, state) for group in groups]
        if _narrowable(state):
            fitted, warning = self._narrow(state, budget * _STATE_SHARE)
            need = max(costs)
            if estimate_request_tokens(fitted) + need > budget:
                fitted, warning = self._narrow(state, budget - need)
            if warning:
                self.warn(f"{warning} ({label})")
            room = budget - estimate_request_tokens(fitted)
            if room >= need:
                groups = _pack(chunk, costs, room)
                if len(groups) > 1:
                    self.warn(
                        f"clef_request_split: a {label} request of {len(chunk)} "
                        f"question(s) was sent as {len(groups)} requests"
                    )
                return [(group, fitted) for group in groups]
        raise RequestTooLargeError(
            f"a {label} request is estimated at "
            f"{estimate_request_tokens(state, dict(chunk)):.0f} tokens, over the "
            f"budget of {budget}, and could not be narrowed or split enough"
        )

    def _narrow(self, state: dict[str, Any], budget: float) -> tuple[Any, str | None]:
        """Narrow a state's input to the spans it cites (or its start) to ``budget``."""
        return fit_clef_state(
            state,
            int(budget),
            spans=state_spans(state) or None,
            window_chars=self.verify.window_chars,
            estimate=estimate_request_tokens,
            rebase=True,
        )

    async def _clef_once(
        self,
        model: str,
        chunk: list[tuple[str, Question]],
        state: Any,
        meta: CallMeta,
        *,
        resplit: bool = True,
    ) -> dict[str, Answer]:
        request = ClefRequest(
            model=body_model(model),
            state=state,
            questions=dict(chunk),
            images=list(self.images) if self.images else None,
        )
        payload = request.to_payload()
        # Clef answers a byte-identical request identically, so a second one in the
        # same run (a re-check that asks what was just asked) is not sent again.
        # Only finished answers are kept: nothing is shared while a request runs,
        # so a cancelled call is cancelled.
        key = memo_key(model, payload)
        known = self.clef_memo.get(key)
        if known is not None:
            # an answer keeps where it came from (its ``provider``), also here; it is
            # a FALLBACK's use only when a fallback answered something in this run
            # (a primary Decisions run has providers on its answers but no fallback)
            if self.fallback_used and any(
                _fallback_provider(a) for a in known.values()
            ):
                self._note_fallback_use(meta)
            return dict(known)
        if key in self.rejected:  # refused once: it would be refused again
            return {}
        self.largest_sent = max(
            self.largest_sent, estimate_request_tokens(state, dict(chunk))
        )
        try:
            answers = await self._clef_send(model, chunk, payload, meta)
        except BadRequestError as exc:
            if not self._contains(exc, count=resplit):
                raise
            self.rejected.add(key)
            isolated = (
                getattr(exc, "question_id", None)
                if isinstance(exc, RequestTooLargeError)
                else None
            )
            if resplit and len(chunk) > 1 and any(q == isolated for q, _ in chunk):
                # A refusal made locally, before anything was sent (it can never have
                # been charged), that names ONE question that cannot fit: refuse
                # that one alone and ask its siblings without it, rather than
                # halving the request over and over.
                self._warn_rejected(meta, 1, exc)
                rest = [item for item in chunk if item[0] != isolated]
                self.warn(self._isolated(meta, str(isolated), len(rest)))
                return await self._clef_once(model, rest, state, meta)
            if resplit and len(chunk) > 1 and is_size_rejection(exc):
                # two SMALLER requests, never the one that was refused again
                middle = len(chunk) // 2
                self.warn(
                    f"clef_request_resplit: a {meta.kind} request of {len(chunk)} "
                    f"question(s) was refused as too large ({_shown(exc)}) and "
                    "was sent again as two smaller requests"
                )
                halves = await gather(
                    self._clef_once(model, chunk[:middle], state, meta, resplit=False),
                    self._clef_once(model, chunk[middle:], state, meta, resplit=False),
                )
                return {**halves[0], **halves[1]}
            self._warn_rejected(meta, len(chunk), exc)
            return {}
        self.clef_memo[key] = answers
        return dict(answers)

    def _contains(self, exc: BadRequestError, *, count: bool = True) -> bool:
        """Whether a refused request is contained (the call goes on without answers).

        Only a recognised size refusal is, never in ``strict`` mode, never when the
        request may have been run and charged (the error, its ``primary_error`` or
        its causes say uncertain, ``maybe_sent`` or ``partial_usage``; the
        transport's own ``classify_failure`` is asked too), and not when the same
        refusal has come ``_MAX_SAME_REFUSAL`` times already: that one is raised.
        """
        if self.strict or not is_request_rejection(exc):
            return False
        classify = getattr(self.transport, "classify_failure", None)
        if callable(classify):
            for link in exception_chain(exc):
                if isinstance(link, Exception) and classify(link) == "uncertain":
                    return False
        if not count:  # the halves of a request that was counted already
            return True
        signature = _refusal_signature(exc)
        self.refusals[signature] = self.refusals.get(signature, 0) + 1
        return self.refusals[signature] <= _MAX_SAME_REFUSAL

    def _warn_rejected(self, meta: CallMeta, n_items: int, exc: BaseException) -> None:
        """Warn that a request was refused and its questions or fields go unanswered."""
        where = ".".join(
            p for p in (meta.scope, meta.schema, meta.entity, meta.field) if p
        )
        self.warn(
            f"request_rejected: the {meta.kind} request"
            f"{f' for {where}' if where else ''} ({n_items} item(s)) was refused "
            f"({_shown(exc)}); its answers are missing and the rest of the run goes "
            "on"
        )

    async def _clef_send(
        self,
        model: str,
        chunk: list[tuple[str, Question]],
        payload: dict[str, Any],
        meta: CallMeta,
    ) -> dict[str, Answer]:
        sent = await self._send(model, payload, meta, len(chunk))
        # a fallback's calls are accounted under the model that answered
        accounted = sent.served_model or model
        try:
            response = parse_clef(sent.body)
        except Exception as exc:
            self._unparsable(accounted, meta, len(chunk), sent, exc)
            raise
        self._record(
            accounted,
            meta,
            len(chunk),
            usage=response.usage,
            duration_ms=sent.duration_ms,
            retries=sent.retries,
            status_code=sent.status_code,
            request_id=sent.request_id,
            provider=sent.provider,
            fallback_from=sent.fallback_from,
        )
        for note in (response.model_extra or {}).get("decisions_notes", ()):
            self.warn(str(note))
        wanted = [qid for qid, _ in chunk if qid in response.answers]
        hard = sum(1 for qid in wanted if _is_hard(response.answers[qid]))
        if hard and sent.fallback_from is not None:
            self.fallback_hard += hard
            self._refresh_fallback_warning()
        elif hard:  # the primary's own hard picks: not a fallback, so not that warning
            self.hard_picks += hard
            self.hard_kinds[meta.kind] = self.hard_kinds.get(meta.kind, 0) + hard
            self._refresh_hard_warning(sent.provider or accounted)
        if sent.served_model is not None:
            pass  # another model's tokens say nothing about Clef's context
        elif response.usage.input_tokens >= _TRUNCATION_RATIO * _CLEF_CONTEXT_TOKENS:
            self.warn(
                "clef_state_possibly_truncated: a Clef request used "
                f"{response.usage.input_tokens} input tokens, close to the "
                f"{_CLEF_CONTEXT_TOKENS} limit; Clef truncates silently"
            )
        return {
            qid: response.answers[qid] for qid, _ in chunk if qid in response.answers
        }

    # ------------------------------------------------------------------ #
    # The escalation tier (opt-in, ``_escalate``)
    # ------------------------------------------------------------------ #

    def escalation_budget(self, sub: str) -> bool:
        """Whether one more escalation call of kind ``sub`` may be made.

        Counts against ``escalation.max_calls_per_invoke`` and the kind's own limit,
        and warns once per limit that was reached.
        """
        cfg = self.escalation
        if not cfg.runs(sub):
            return False
        limit = cfg.limit_for(sub)
        if self.escalation_calls >= cfg.max_calls_per_invoke:
            self.warn(
                f"escalation_limit: max_calls_per_invoke={cfg.max_calls_per_invoke} "
                f"was reached; further escalations ({sub} and the rest) were skipped"
            )
            return False
        if limit is not None and self.escalation_by_kind.get(sub, 0) >= limit:
            self.warn(
                f"escalation_limit: the limit of {limit} {sub} call(s) per invoke "
                "was reached; further ones were skipped"
            )
            return False
        return True

    async def call_escalate(
        self, sub: str, payload: dict[str, Any], meta: CallMeta
    ) -> dict[str, Any] | None:
        """Ask the escalation model one question and return its parsed JSON answer.

        The call is logged as a ``CallRecord`` of kind ``escalate`` whose ``field``
        starts with the sub-kind. It returns ``None`` (the escalation did not
        happen) when the kind is off, a limit was reached, the model gave no answer
        (a refusal, an incomplete response) or the call failed in a way that says
        nothing about the request (a timeout, a rate limit): the default setup's
        result then stands, with a warning. A rejected key, a malformed request and
        every failure in ``strict`` mode are raised.
        """
        if not self.escalation_budget(sub):
            return None
        cfg = self.escalation
        transport = cfg.transport
        if transport is None:
            raise RuntimeError(
                "escalation is enabled but has no transport (the extractor creates "
                "one from the configuration)"
            )
        self.escalation_calls += 1
        self.escalation_by_kind[sub] = self.escalation_by_kind.get(sub, 0) + 1
        model = str(getattr(transport, "model", None) or OPENAI_DEFAULT_MODEL)
        body = dict(payload)
        if cfg.reasoning_effort is not None:
            body["reasoning_effort"] = cfg.reasoning_effort
        try:
            sent = await self._send(model, body, meta, 1, transport=transport)
        except AuthError:
            raise
        except BadRequestError as exc:
            if self.strict or not _is_context_refusal(exc):
                raise
            self.warn(f"escalation_failed: {sub}: {_shown(exc)}")
            return None
        except TrustcallSystemOneError as exc:
            if self.strict:
                raise
            self.warn(f"escalation_failed: {sub}: {_shown(exc)}")
            return None
        reply = sent.body
        raw_usage = reply.get("usage")
        usage_in: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        usage = ClefUsage(
            input_tokens=int(usage_in.get("input_tokens") or 0),
            output_tokens=int(usage_in.get("output_tokens") or 0),
            reasoning_tokens=int(usage_in.get("reasoning_tokens") or 0),
        )
        self._record(
            sent.served_model or model,
            meta,
            1,
            usage=usage,
            duration_ms=sent.duration_ms,
            retries=sent.retries,
            status_code=sent.status_code,
            request_id=sent.request_id,
            provider=sent.provider,
        )
        parsed = reply.get("parsed")
        if not isinstance(parsed, dict):
            self.warn(
                f"escalation_no_answer: {sub}: the model gave no answer "
                f"({reply.get('no_answer') or 'empty'})"
            )
            return None
        return parsed

    async def once(self, key: Any, make: "Callable[[], Awaitable[T]]") -> T:
        """Run ``make`` once for ``key`` in this run; every caller gets its result.

        The entities of one nested list share one escalation call for their rows.
        """
        future = self.escalation_memo.get(key)
        if future is None:
            future = asyncio.ensure_future(make())
            self.escalation_memo[key] = future
        result: T = await asyncio.shield(future)
        return result

    def clef_state(
        self,
        text: str,
        *,
        entity: Mapping[str, Any] | None = None,
        record: Any = None,
        proposed_change: Any = None,
        spans: Sequence[tuple[int, int]] | None = None,
        extra: Mapping[str, Any] | None = None,
        budget: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Build a Clef state and keep it under ``limits.clef_state_token_budget``.

        The size is ``estimate_request_tokens`` (calibrated on recorded requests;
        table-heavy text is denser than the old characters-over-3.5 rule). Spans
        that ``extra`` cites are re-based into the narrowed input, or dropped when
        they fall outside every window.

        Args:
            text: The user's input.
            entity: What the question is about.
            record: An existing record (updates).
            proposed_change: A change under review (updates).
            spans: Spans in ``text`` to keep if the input has to be narrowed.
            extra: Further keys of the state (see ``_render.clef_state``).
            budget: A smaller budget than ``limits.clef_state_token_budget`` (the
                question that goes with the state takes up the rest). That limit is
                itself capped by what the transport says it takes
                (``transport.state_token_budget``: Jev reads 32k tokens).
            model: The Clef model that will read the state, when it is known (its
                transport's budget applies); else the smallest budget of the models
                this run uses (``clef_budget``).

        Returns:
            The state; when it was too large the input is narrowed and a warning
            is recorded.
        """
        state = clef_state(
            text,
            entity=entity,
            record=record,
            proposed_change=proposed_change,
            extra=extra,
        )
        cap = self.clef_budget(model)
        state, warning = fit_clef_state(
            state,
            cap if budget is None else min(budget, cap),
            spans=spans,
            window_chars=self.verify.window_chars,
            estimate=estimate_request_tokens,
            rebase=True,
        )
        if warning:
            self.warn(warning)
        return state
