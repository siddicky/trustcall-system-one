"""The public entry points: ``create_extractor`` and ``Extractor``.

``Extractor.invoke`` is the sync bridge: every extractor owns ONE persistent
private event loop on a daemon thread (started on first use) and runs
``ainvoke`` on it with ``asyncio.run_coroutine_threadsafe``. That makes it work
from plain code and from inside a running loop (Jupyter, ``asyncio.run``)
without nesting loops, and it keeps one httpx client and one limiter state for
every sync call.
"""

import asyncio
import atexit
import contextlib
import dataclasses
import functools
import threading
import weakref
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, TypeVar

from typing_extensions import Required, TypedDict

from trustcall_system_one._cardinality import Many, One, resolve_cardinality
from trustcall_system_one._compile import SchemaPlan, compile_schema
from trustcall_system_one._config import (
    ConcurrencyConfig,
    GateConfig,
    Limits,
    PassesConfig,
    PresenceConfig,
    Prices,
    VerifyConfig,
)
from trustcall_system_one._entity import insert_one
from trustcall_system_one._run import Run, gather
from trustcall_system_one._schemas import SchemaLike, coerce_schema
from trustcall_system_one._types import ExtractionResult, ResultRecord, Usage
from trustcall_system_one._wire import CLAIRIC, CLEF
from trustcall_system_one.transport import CloudflareTransport, Transport

__all__ = ["ExtractionInput", "Extractor", "create_extractor"]

T = TypeVar("T")
_JOIN_TIMEOUT = 10.0


class ExtractionInput(TypedDict, total=False):
    """One input for ``invoke`` and ``batch``.

    Attributes:
        text: The text to extract from.
        existing: Existing records to update (arrives in M5).
        images: Images sent along with the text.
        reference_date: Date that relative dates in the text are resolved against.
    """

    text: Required[str]
    existing: Any
    images: Sequence[Any]
    reference_date: date | str


# --------------------------------------------------------------------------- #
# Transport ownership and the sync bridge
# --------------------------------------------------------------------------- #


def _close_at_exit(ref: "weakref.ReferenceType[_Core]") -> None:
    core = ref()
    if core is not None:
        with contextlib.suppress(Exception):  # best effort at interpreter exit
            core.close()


class _Core:
    """Shared by an extractor and its ``with_options`` copies.

    Holds the transport (created lazily when none was given, and then owned),
    and the private event loop thread behind the sync bridge.
    """

    def __init__(
        self, transport: Transport | None, concurrency: ConcurrencyConfig | None
    ) -> None:
        self._given = transport
        self._concurrency = concurrency
        self._owned: CloudflareTransport | None = None
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._exit_hook: functools.partial[None] | None = None

    # -- transport -------------------------------------------------------- #

    def transport(self) -> Transport:
        """Return the transport, creating the default one on first use."""
        if self._given is not None:
            return self._given
        with self._lock:
            if self._owned is None:
                self._owned = CloudflareTransport(concurrency=self._concurrency)
            return self._owned

    # -- the bridge loop --------------------------------------------------- #

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def target() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                try:
                    loop.run_forever()
                finally:
                    with contextlib.suppress(Exception):
                        for task in asyncio.all_tasks(loop):
                            task.cancel()
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

            thread = threading.Thread(
                target=target, name="trustcall-system-one-loop", daemon=True
            )
            thread.start()
            ready.wait()
            self._loop, self._thread = loop, thread
            self._exit_hook = functools.partial(_close_at_exit, weakref.ref(self))
            atexit.register(self._exit_hook)
            return loop

    def run_sync(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the bridge loop and wait for its result."""
        loop = self._ensure_loop()
        if threading.current_thread() is self._thread:
            coroutine.close()
            raise RuntimeError(
                "invoke() was called from the extractor's own event loop thread, "
                "which would deadlock; await ainvoke() instead"
            )
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def _take_loop(
        self,
    ) -> tuple[asyncio.AbstractEventLoop | None, threading.Thread | None]:
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
            hook, self._exit_hook = self._exit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        return loop, thread

    # -- closing ----------------------------------------------------------- #

    def close(self) -> None:
        """Close an owned transport and stop the bridge thread (idempotent)."""
        loop, thread = self._take_loop()
        owned, self._owned = self._owned, None
        if loop is not None and thread is not None and not loop.is_closed():
            if threading.current_thread() is thread:
                loop.call_soon(loop.stop)  # cannot wait for ourselves
                return
            if owned is not None:
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(owned.aclose(), loop).result(
                        timeout=_JOIN_TIMEOUT
                    )
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=_JOIN_TIMEOUT)
        if owned is not None:
            with contextlib.suppress(Exception):
                owned.close()  # clients of other, finished loops

    async def aclose(self) -> None:
        """Close an owned transport and stop the bridge thread (idempotent)."""
        running = asyncio.get_running_loop()
        loop, thread = self._take_loop()
        owned, self._owned = self._owned, None
        if owned is not None:
            with contextlib.suppress(Exception):
                await owned.aclose()
        if loop is None or thread is None or loop.is_closed():
            return
        if loop is running:
            loop.call_soon(loop.stop)  # we are on the bridge: stop after this task
            return
        if owned is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.wrap_future(
                        asyncio.run_coroutine_threadsafe(owned.aclose(), loop)
                    ),
                    _JOIN_TIMEOUT,
                )
        loop.call_soon_threadsafe(loop.stop)
        await running.run_in_executor(None, thread.join, _JOIN_TIMEOUT)


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


def _iso_date(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise ValueError(
            f"reference_date must be a date or an ISO date string, got {value!r}"
        ) from None


def _inner(item: Any) -> SchemaLike:
    schema = item.schema if isinstance(item, (One, Many)) else item
    if schema is None:
        raise ValueError(f"{type(item).__name__}(...) needs a schema")
    return schema


@dataclass(frozen=True, eq=False)
class Extractor:
    """Extracts structured records from text with Clairic and Clef.

    Build one with ``create_extractor``. It is immutable; ``with_options`` returns
    a changed copy that shares the transport and the sync bridge. Closing any of
    the copies (``close``, ``aclose`` or a ``with`` block) closes the shared
    owned transport and bridge; a closed extractor starts them again on next use.

    Attributes:
        schemas: The schemas (or ``One`` / ``Many`` wrappers) it extracts.
        transport: The transport given at creation, or ``None`` for the default
            ``CloudflareTransport`` (created on first use and owned).
        clairic_model: Model id of the extraction model.
        clef_model: Model id of the decision and verification model.
        gate_model: Model id of the change-gate model (updates).
        enable_inserts: Allow inserting new records next to existing ones.
        enable_updates: Allow updating existing records.
        enable_deletes: Allow deleting existing records.
        enable_item_removals: Allow removing items from nested lists.
        existing_schema_policy: How ``existing`` records of unknown schemas are
            treated.
        presence: Presence-gate thresholds.
        verify: Verification-ladder settings.
        gates: Answer-level thresholds.
        passes: How Clairic fields are grouped into requests.
        limits: Structural limits.
        concurrency: Concurrency limits of the default transport (no effect on
            a transport you pass in).
        prices: Input-token prices used for ``ExtractionResult.cost``.
        reference_date: Default date relative dates are resolved against.
        strict: Raise on a terminal validation failure.
    """

    schemas: tuple[Any, ...]
    transport: Transport | None = None
    clairic_model: str = CLAIRIC
    clef_model: str = CLEF
    gate_model: str = CLEF
    enable_inserts: bool = False
    enable_updates: bool = True
    enable_deletes: bool = False
    enable_item_removals: bool = False
    existing_schema_policy: bool | Literal["ignore"] = True
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    passes: PassesConfig = field(default_factory=PassesConfig)
    limits: Limits = field(default_factory=Limits)
    concurrency: ConcurrencyConfig | None = None
    prices: Prices = field(default_factory=Prices)
    reference_date: date | str | None = None
    strict: bool = False
    _core: _Core | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate the settings and compile every schema."""
        if not (self.enable_inserts or self.enable_updates or self.enable_deletes):
            raise ValueError(
                "At least one of enable_inserts, enable_updates, or "
                "enable_deletes must be True."
            )
        if not self.schemas:
            raise ValueError("create_extractor needs at least one schema")
        if self.existing_schema_policy not in (True, False, "ignore"):
            raise ValueError(
                "existing_schema_policy must be True, False or 'ignore', got "
                f"{self.existing_schema_policy!r}"
            )
        _iso_date(self.reference_date)
        names: list[str] = []
        for item in self.schemas:
            names.append(self.plan_for(_inner(item)).name)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"schema names must be unique, got {duplicates} twice")
        if self._core is None:
            object.__setattr__(self, "_core", _Core(self.transport, self.concurrency))

    # -- helpers ------------------------------------------------------------ #

    @property
    def core(self) -> _Core:
        """The shared transport and bridge holder."""
        assert self._core is not None
        return self._core

    def plan_for(self, schema: SchemaLike | One | Many) -> SchemaPlan:
        """Return the compiled plan for a schema (cached).

        Args:
            schema: A schema or a ``One`` / ``Many`` wrapper.

        Returns:
            The plan compiled with this extractor's gates and limits.
        """
        return compile_schema(_inner(schema), gates=self.gates, limits=self.limits)

    def _select(self, schema: Any) -> list[Any]:
        if schema is None:
            return list(self.schemas)
        if isinstance(schema, str):
            wanted = schema
        else:
            wanted = coerce_schema(_inner(schema)).__name__
        chosen = [i for i in self.schemas if self.plan_for(i).name == wanted]
        if not chosen:
            known = [self.plan_for(i).name for i in self.schemas]
            raise ValueError(f"schema {wanted!r} is not one of {known}")
        return chosen

    def with_options(self, **overrides: Any) -> "Extractor":
        """Return a copy with some settings changed.

        The copy shares the transport and the sync bridge. Passing ``transport``
        or ``concurrency`` gives it a transport of its own.

        Args:
            **overrides: Any ``create_extractor`` setting.

        Raises:
            TypeError: For an unknown setting.
        """
        known = {f.name for f in dataclasses.fields(self) if f.init} - {"_core"}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(
                f"unknown option(s) {sorted(unknown)}; known: {sorted(known)}"
            )
        if "transport" in overrides or "concurrency" in overrides:
            overrides["_core"] = None
        return dataclasses.replace(self, **overrides)

    # -- async API ---------------------------------------------------------- #

    async def ainvoke(
        self,
        input: str | ExtractionInput,
        *,
        existing: Any = None,
        schema: Any = None,
        images: Sequence[Any] | None = None,
        reference_date: date | str | None = None,
    ) -> ExtractionResult:
        """Extract records from text.

        Without ``existing`` every schema is extracted as an insert (whatever
        ``enable_inserts`` says). Several schemas run in parallel and each is
        gated by a presence check.

        Args:
            input: The text, or an ``ExtractionInput`` carrying it. Explicit
                keyword arguments override the keys of an ``ExtractionInput``.
            existing: Existing records to update. Not supported yet (M5).
            schema: Force one of the extractor's schemas (a name, a schema or a
                wrapper): only it runs and its presence gate is skipped, like
                trustcall's ``tool_choice``.
            images: Images sent with every request that carries the text
                (data URLs or ``{"content_type", "base64"}`` objects).
            reference_date: Date relative dates are resolved against; defaults to
                the extractor's.

        Returns:
            The records, models, usage, cost, call log and warnings.

        Raises:
            NotImplementedError: For ``existing`` (M5) and ``Many`` schemas (M7).
            ValueError: If ``schema`` is not one of the extractor's schemas.
            ExtractionValidationError: With ``strict``, on a validation failure.
            StateTruncatedError: With ``strict``, if Clairic keeps truncating.
            TransportError: If a request failed.
        """
        if isinstance(input, Mapping):
            text = input["text"]
            if existing is None:
                existing = input.get("existing")
            if images is None:
                images = input.get("images")
            if reference_date is None:
                reference_date = input.get("reference_date")
        else:
            text = input
        if existing is not None:
            raise NotImplementedError(
                "existing / update extraction arrives in M5; "
                "call without existing= to insert"
            )
        selected = self._select(schema)
        cards = [resolve_cardinality(item) for item in selected]
        for card in cards:
            if isinstance(card, Many):
                raise NotImplementedError("Many(...) arrives in M7")
        if not text.strip() and not images:
            return ExtractionResult(
                cost=self.prices.cost(Usage()),
                warnings=("the input is empty; nothing was extracted",),
            )
        run = Run(
            transport=self.core.transport(),
            clairic_model=self.clairic_model,
            clef_model=self.clef_model,
            gate_model=self.gate_model,
            verify=self.verify,
            presence=self.presence,
            gates=self.gates,
            passes=self.passes,
            limits=self.limits,
            images=list(images) if images else None,
            reference_date=_iso_date(
                reference_date if reference_date is not None else self.reference_date
            ),
            strict=self.strict,
        )
        jobs = []
        for card in cards:
            assert isinstance(card, One)
            plan = self.plan_for(card)
            label = card.label or f"the {plan.title}"
            jobs.append(
                insert_one(
                    run, plan, text, label=label, skip_presence=schema is not None
                )
            )
        outcomes = await gather(*jobs)
        records: list[ResultRecord] = [
            o.record for o in outcomes if o.record is not None
        ]
        return ExtractionResult(
            records=tuple(records),
            responses=tuple(
                r.value for r in records if r.valid and r.value is not None
            ),
            attempts=1 if any(o.extracted for o in outcomes) else 0,
            usage=run.usage,
            cost=self.prices.cost(run.usage),
            calls=tuple(run.calls),
            warnings=tuple(run.warnings),
        )

    async def abatch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = None,
        schema: Any = None,
        images: Sequence[Any] | None = None,
        reference_date: date | str | None = None,
    ) -> list[ExtractionResult]:
        """Run ``ainvoke`` on every input concurrently.

        The keyword arguments apply to every input (an ``ExtractionInput``'s own
        keys are used when the argument is ``None``). If one input fails the
        others are cancelled and the error propagates.

        Returns:
            One result per input, in order.
        """
        return await gather(
            *(
                self.ainvoke(
                    item,
                    existing=existing,
                    schema=schema,
                    images=images,
                    reference_date=reference_date,
                )
                for item in inputs
            )
        )

    # -- sync API ----------------------------------------------------------- #

    def invoke(
        self,
        input: str | ExtractionInput,
        *,
        existing: Any = None,
        schema: Any = None,
        images: Sequence[Any] | None = None,
        reference_date: date | str | None = None,
    ) -> ExtractionResult:
        """Run ``ainvoke`` on the extractor's private event loop thread.

        Works from plain code and from inside a running event loop (Jupyter,
        ``asyncio.run``): it blocks the calling thread, never nests a loop. It
        must not be called from the bridge thread itself.
        """
        return self.core.run_sync(
            self.ainvoke(
                input,
                existing=existing,
                schema=schema,
                images=images,
                reference_date=reference_date,
            )
        )

    def batch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = None,
        schema: Any = None,
        images: Sequence[Any] | None = None,
        reference_date: date | str | None = None,
    ) -> list[ExtractionResult]:
        """Run ``abatch`` on the extractor's private event loop thread."""
        return self.core.run_sync(
            self.abatch(
                inputs,
                existing=existing,
                schema=schema,
                images=images,
                reference_date=reference_date,
            )
        )

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        """Close an owned default transport and stop the bridge thread.

        A transport passed to ``create_extractor`` is not closed. Safe to call
        twice; the extractor can be used again afterwards.
        """
        self.core.close()

    async def aclose(self) -> None:
        """Async ``close``: closes owned clients on their own event loops."""
        await self.core.aclose()

    def __enter__(self) -> "Extractor":
        """Enter a ``with`` block; ``close`` runs on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close on leaving the ``with`` block."""
        self.close()

    async def __aenter__(self) -> "Extractor":
        """Enter an ``async with`` block; ``aclose`` runs on exit."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close on leaving the ``async with`` block."""
        await self.aclose()


def create_extractor(
    schemas: Sequence[SchemaLike | Many | One],
    *,
    transport: Transport | None = None,
    clairic_model: str = CLAIRIC,
    clef_model: str = CLEF,
    gate_model: str = CLEF,
    enable_inserts: bool = False,
    enable_updates: bool = True,
    enable_deletes: bool = False,
    enable_item_removals: bool = False,
    existing_schema_policy: bool | Literal["ignore"] = True,
    presence: PresenceConfig | None = None,
    verify: VerifyConfig | None = None,
    gates: GateConfig | None = None,
    passes: PassesConfig | None = None,
    limits: Limits | None = None,
    concurrency: ConcurrencyConfig | None = None,
    prices: Prices | None = None,
    reference_date: date | str | None = None,
    strict: bool = False,
) -> Extractor:
    """Create an extractor for one or more schemas.

    Args:
        schemas: Pydantic models, TypedDicts, JSON Schemas, annotated functions
            or ``One`` / ``Many`` wrappers around them.
        transport: Where requests go. By default a ``CloudflareTransport`` is
            created on first use (credentials are only needed then) and closed
            by ``close``.
        clairic_model: Model id of the extraction model.
        clef_model: Model id of the decision and verification model.
        gate_model: Model id of the change-gate model (updates, M5).
        enable_inserts: Allow inserting new records next to existing ones. With no
            ``existing`` records the extractor always inserts.
        enable_updates: Allow updating existing records (M5).
        enable_deletes: Allow deleting existing records (M8).
        enable_item_removals: Allow removing items from nested lists (M9).
        existing_schema_policy: How ``existing`` records of unknown schemas are
            treated (M5).
        presence: Presence-gate thresholds.
        verify: Verification-ladder settings.
        gates: Answer-level thresholds.
        passes: How Clairic fields are grouped into requests.
        limits: Structural limits.
        concurrency: Concurrency limits of the default transport.
        prices: Input-token prices used for the cost report.
        reference_date: Default date relative dates are resolved against.
        strict: Raise ``ExtractionValidationError`` (or ``StateTruncatedError``)
            instead of returning an invalid record (or a warning).

    Returns:
        The extractor.

    Raises:
        ValueError: If none of ``enable_inserts``, ``enable_updates`` and
            ``enable_deletes`` is true, no schema is given, schema names repeat,
            or a schema cannot be compiled.
    """
    return Extractor(
        schemas=tuple(schemas),
        transport=transport,
        clairic_model=clairic_model,
        clef_model=clef_model,
        gate_model=gate_model,
        enable_inserts=enable_inserts,
        enable_updates=enable_updates,
        enable_deletes=enable_deletes,
        enable_item_removals=enable_item_removals,
        existing_schema_policy=existing_schema_policy,
        presence=presence or PresenceConfig(),
        verify=verify or VerifyConfig(),
        gates=gates or GateConfig(),
        passes=passes or PassesConfig(),
        limits=limits or Limits(),
        concurrency=concurrency,
        prices=prices or Prices(),
        reference_date=reference_date,
        strict=strict,
    )
