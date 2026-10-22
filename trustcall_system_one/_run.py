"""Per-invoke context: transport calls, usage, call log and warnings."""

import asyncio
import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union

from trustcall_system_one._config import (
    CorefConfig,
    DiscoveryConfig,
    GateConfig,
    Limits,
    PassesConfig,
    PresenceConfig,
    RepairConfig,
    VerifyConfig,
)
from trustcall_system_one._render import (
    RenderedState,
    clef_state,
    fit_clef_state,
    reanchor,
)
from trustcall_system_one._types import CallRecord, Usage
from trustcall_system_one._wire import (
    CLAIRIC,
    CLEF,
    ChoiceAnswer,
    ChoiceQuestion,
    ClairicField,
    ClairicOptions,
    ClairicRequest,
    ClairicResponse,
    ClefRequest,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    body_model,
    parse_clairic,
    parse_clef,
)
from trustcall_system_one.transport import CallMeta, Transport

__all__ = ["CLEF_BATCH", "Answer", "Question", "Run", "gather"]

CLEF_BATCH = 64
"""Most questions a single Clef request may carry."""
_CLEF_CONTEXT_TOKENS = 65_536
_TRUNCATION_RATIO = 0.97

Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]  # noqa: UP007
"""Any Clef answer."""
Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]  # noqa: UP007
"""Any Clef question."""

T = TypeVar("T")


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
    failure = next((e for t in tasks if (e := _error(t))), None)
    if failure is not None:
        raise failure from None
    if any(t.cancelled() for t in tasks):
        raise asyncio.CancelledError
    return [t.result() for t in tasks]


@dataclass(frozen=True)
class _Sent:
    """A successful transport response with its timing."""

    body: dict[str, Any]
    retries: int
    duration_ms: float
    status_code: int | None


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
        images: Images sent with every request that carries the user's input.
            Each request model validates the documented size limits before it is
            sent.
        reference_date: ISO date Clairic resolves relative dates against.
        strict: Raise instead of returning an invalid record or a truncated state.
        usage: Token and request totals so far.
        calls: One record per transport call, failed calls included.
        warnings: Notes about anything that was degraded, without duplicates.
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
    images: Sequence[Any] | None = None
    reference_date: str | None = None
    strict: bool = False
    usage: Usage = field(default_factory=Usage)
    calls: list[CallRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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
    ) -> None:
        """Append one ``CallRecord`` (and the usage, for a successful call)."""
        counted: dict[str, int] = {}
        if usage is not None:
            self.usage.add_call(model, meta.kind, usage)
            counted = {
                key: value
                for key, value in usage.model_dump().items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
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
            )
        )

    async def _send(
        self,
        model: str,
        payload: dict[str, Any],
        meta: CallMeta,
        n_items: int,
    ) -> _Sent:
        """Send one request; a failure is logged first and then re-raised."""
        started = time.perf_counter()
        try:
            response = await self.transport.run(model, payload, meta=meta)
        except Exception as exc:
            self._record(
                model,
                meta,
                n_items,
                duration_ms=(time.perf_counter() - started) * 1000,
                retries=getattr(exc, "retries", 0),
                status_code=getattr(exc, "status_code", None),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        return _Sent(
            body=response.body,
            retries=response.retries,
            duration_ms=response.duration_ms or (time.perf_counter() - started) * 1000,
            status_code=response.status_code,
        )

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

        Raises:
            TransportError: If the call failed (after it was logged).
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
        sent = await self._send(
            self.clairic_model, request.to_payload(), meta, len(fields)
        )
        try:
            response = parse_clairic(sent.body)
        except Exception as exc:
            self._unparsable(self.clairic_model, meta, len(fields), sent, exc)
            raise
        self._record(
            self.clairic_model,
            meta,
            len(fields),
            usage=response.usage,
            duration_ms=sent.duration_ms,
            retries=sent.retries,
            status_code=sent.status_code,
        )
        return response

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

    async def call_clef(
        self,
        questions: Mapping[str, Question],
        state: Any,
        *,
        meta: CallMeta,
        model: str | None = None,
    ) -> dict[str, Answer]:
        """Send Clef questions, in requests of at most 64, and merge the answers.

        More than 64 questions are split into several requests that are issued
        concurrently; each is logged as its own ``CallRecord``.

        Args:
            questions: Questions keyed by id.
            state: The structured (or string) state, shared by every question.
            meta: Routing context; ``meta.kind`` is the call kind.
            model: Model id; defaults to ``clef_model``.

        Returns:
            Answers keyed by question id.

        Raises:
            TransportError: If a call failed (after it was logged).
            ModelResponseError: If a response does not match the schema.
        """
        model = model or self.clef_model
        items = list(questions.items())
        if not items:
            return {}
        chunks = [items[i : i + CLEF_BATCH] for i in range(0, len(items), CLEF_BATCH)]
        parts = await gather(
            *(self._clef_once(model, chunk, state, meta) for chunk in chunks)
        )
        merged: dict[str, Answer] = {}
        for part in parts:
            merged.update(part)
        return merged

    async def _clef_once(
        self,
        model: str,
        chunk: list[tuple[str, Question]],
        state: Any,
        meta: CallMeta,
    ) -> dict[str, Answer]:
        request = ClefRequest(
            model=body_model(model),
            state=state,
            questions=dict(chunk),
            images=list(self.images) if self.images else None,
        )
        sent = await self._send(model, request.to_payload(), meta, len(chunk))
        try:
            response = parse_clef(sent.body)
        except Exception as exc:
            self._unparsable(model, meta, len(chunk), sent, exc)
            raise
        self._record(
            model,
            meta,
            len(chunk),
            usage=response.usage,
            duration_ms=sent.duration_ms,
            retries=sent.retries,
            status_code=sent.status_code,
        )
        if response.usage.input_tokens >= _TRUNCATION_RATIO * _CLEF_CONTEXT_TOKENS:
            self.warn(
                "clef_state_possibly_truncated: a Clef request used "
                f"{response.usage.input_tokens} input tokens, close to the "
                f"{_CLEF_CONTEXT_TOKENS} limit; Clef truncates silently"
            )
        return {
            qid: response.answers[qid] for qid, _ in chunk if qid in response.answers
        }

    def clef_state(
        self,
        text: str,
        *,
        entity: Mapping[str, Any] | None = None,
        record: Any = None,
        proposed_change: Any = None,
        spans: Sequence[tuple[int, int]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a Clef state and keep it under ``limits.clef_state_token_budget``.

        Args:
            text: The user's input.
            entity: What the question is about.
            record: An existing record (updates).
            proposed_change: A change under review (updates).
            spans: Spans in ``text`` to keep if the input has to be narrowed.
            extra: Further keys of the state (see ``_render.clef_state``).

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
        state, warning = fit_clef_state(
            state,
            self.limits.clef_state_token_budget,
            spans=spans,
            window_chars=self.verify.window_chars,
        )
        if warning:
            self.warn(warning)
        return state
