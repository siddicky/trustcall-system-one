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
import concurrent.futures
import contextlib
import dataclasses
import functools
import threading
import weakref
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, TypeVar, overload

from typing_extensions import Required, TypedDict

from trustcall_system_one._cardinality import Many, One, resolve_cardinality
from trustcall_system_one._collection import Scope, apply_collection, identity_fields
from trustcall_system_one._compile import SchemaPlan, compile_schema
from trustcall_system_one._config import (
    ConcurrencyConfig,
    CorefConfig,
    DiscoveryConfig,
    GateConfig,
    Limits,
    MatchConfig,
    PassesConfig,
    PresenceConfig,
    Prices,
    RepairConfig,
    VerifyConfig,
)
from trustcall_system_one._entity import ask_presence, insert_one, update_one
from trustcall_system_one._errors import TrustcallSystemOneError
from trustcall_system_one._existing import (
    ANY,
    ExistingRecord,
    ExistingSet,
    coerce_existing,
    fits,
    target_for,
)
from trustcall_system_one._run import Run, gather
from trustcall_system_one._schemas import SchemaLike, coerce_schema
from trustcall_system_one._types import (
    ExtractionResult,
    ResultRecord,
    SkippedEntity,
    Usage,
)
from trustcall_system_one._wire import CLAIRIC, CLEF
from trustcall_system_one.transport import CloudflareTransport, Transport

__all__ = ["ExtractionInput", "Extractor", "create_extractor"]

T = TypeVar("T")
_JOIN_TIMEOUT = 10.0
_CANCEL_GRACE = 2.0
"""Seconds the bridge thread waits for cancelled calls to unwind before it cancels
what is still running."""


class ExtractionInput(TypedDict, total=False):
    """One input for ``invoke`` and ``batch``.

    Attributes:
        text: The text to extract from.
        existing: Existing records to update (any of the five shapes, see
            ``Extractor.ainvoke``).
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
    and the private event loop thread behind the sync bridge. Calls that are in
    flight on the bridge are tracked so that closing cancels them: their callers
    get a ``TrustcallSystemOneError`` at once instead of waiting for a loop that
    is going away.
    """

    def __init__(
        self, transport: Transport | None, concurrency: ConcurrencyConfig | None
    ) -> None:
        self._given = transport
        self._concurrency = concurrency
        self._owned: CloudflareTransport | None = None
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._exit_hook: functools.partial[None] | None = None
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._closed_futures: set[concurrent.futures.Future[Any]] = set()

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

    def _start_loop(self) -> asyncio.AbstractEventLoop:
        """Start the bridge thread. The caller holds the lock."""
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def target() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            try:
                loop.run_forever()
            finally:
                with contextlib.suppress(Exception):
                    # Let what is left finish, so that nothing is destroyed
                    # pending and every future gets resolved. The calls that
                    # close() cancelled are unwinding already (their cleanup
                    # may need the loop), so they are waited for first: a
                    # second cancel() that lands after the first one was
                    # delivered interrupts that cleanup. Only what is still
                    # running after the grace period is cancelled.
                    tasks = list(asyncio.all_tasks(loop))
                    if tasks:
                        _, stuck = loop.run_until_complete(
                            asyncio.wait(tasks, timeout=_CANCEL_GRACE)
                        )
                        for task in stuck:
                            task.cancel()
                        if stuck:
                            loop.run_until_complete(
                                asyncio.gather(*stuck, return_exceptions=True)
                            )
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
        """Run a coroutine on the bridge loop and wait for its result.

        Raises:
            RuntimeError: If called from the bridge thread itself.
            TrustcallSystemOneError: If the extractor was closed while the
                coroutine was running.
        """
        with self._lock:
            if self._thread is threading.current_thread():
                coroutine.close()
                raise RuntimeError(
                    "invoke() was called from the extractor's own event loop "
                    "thread, which would deadlock; await ainvoke() instead"
                )
            loop = self._loop
            if loop is None or loop.is_closed():
                loop = self._start_loop()
            # Submitted under the lock, so a concurrent close() either sees this
            # future and cancels it or has already replaced the loop.
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            self._pending.add(future)
        future.add_done_callback(self._forget)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            if future in self._closed_futures:
                raise TrustcallSystemOneError(
                    "the extractor was closed while the call was running"
                ) from None
            raise
        except BaseException:
            future.cancel()
            raise
        finally:
            self._closed_futures.discard(future)

    def _forget(self, future: "concurrent.futures.Future[Any]") -> None:
        with self._lock:
            self._pending.discard(future)

    def _take_loop(
        self,
    ) -> tuple[asyncio.AbstractEventLoop | None, threading.Thread | None]:
        """Detach the bridge and cancel the calls running on it."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
            hook, self._exit_hook = self._exit_hook, None
            pending = list(self._pending)
            self._closed_futures.update(pending)
        for future in pending:
            future.cancel()
        if hook is not None:
            atexit.unregister(hook)
        return loop, thread

    # -- closing ----------------------------------------------------------- #

    def close(self) -> None:
        """Close an owned transport and stop the bridge thread (idempotent).

        Calls running on the bridge are cancelled. Called from the bridge thread
        itself, the transport is closed on the loop and the loop stops once that
        is done.
        """
        loop, thread = self._take_loop()
        with self._lock:
            owned, self._owned = self._owned, None
        if loop is not None and thread is not None and not loop.is_closed():
            if threading.current_thread() is thread:
                if owned is None:
                    loop.call_soon(loop.stop)
                else:
                    closing = loop.create_task(owned.aclose())
                    closing.add_done_callback(lambda _: loop.stop())
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
        with self._lock:
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


def _note_failure(exc: Exception, run: Run) -> None:
    """Add a one-line summary of the spent calls to a foreign exception."""
    add_note = getattr(exc, "add_note", None)  # Python 3.11+
    if add_note is None:
        return
    kinds = ", ".join(call.kind for call in run.calls) or "none"
    add_note(
        f"trustcall-system-one: {len(run.calls)} call(s) were made before this "
        f"failure ({kinds}); {run.usage.input_tokens} input tokens"
    )


@dataclass(frozen=True)
class _Entry:
    """A schema that runs: its plan and how prompts name it.

    ``many`` is the wrapper of a schema that has any number of entities (run by
    the collection engine); ``None`` for a ``One``.
    """

    plan: SchemaPlan
    label: str
    many: Many | None = None


@dataclass(frozen=True)
class _Done:
    """The outcome of one schema: its records (inserted, updated, deleted) and skips."""

    records: tuple[ResultRecord, ...]
    skipped: tuple[SkippedEntity, ...]
    handled: tuple[tuple[str, str], ...]
    extracted: bool
    attempts: int = 0
    deleted: tuple[str, ...] = ()


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
        repair: Validation repair loop settings.
        discovery: Settings of entity discovery for ``Many`` schemas.
        coref: Settings of mention clustering for ``Many`` schemas.
        match: Settings of matching a ``Many`` schema's entities to its existing
            records.
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
    repair: RepairConfig = field(default_factory=RepairConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    coref: CorefConfig = field(default_factory=CorefConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
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
            plan = self.plan_for(_inner(item))
            names.append(plan.name)
            if isinstance(item, Many):
                identity_fields(plan, item)  # an unknown identity field fails here
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

        The copy shares the transport and the sync bridge with the original, so
        closing either (``close``, ``aclose`` or a ``with`` block) closes the
        shared owned transport and stops the bridge thread for both; each starts
        again lazily on its next use. Passing ``transport`` or ``concurrency``
        gives the copy a transport and bridge of its own.

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

        With ``existing`` a ``One`` schema that has a record is updated: the
        record is shown to the model next to the text, the answers are compared
        with it, and what differs becomes JSON Patch operations that a Clef gate
        checks one by one (see ``ResultRecord.patches`` and ``.rejected``). A
        schema without a record is extracted as an insert. Records are taken in
        five shapes: ``{schema_name: record}``; ``list[SchemaInstance]``;
        ``list[(record_id, schema_name, record)]``; ``list[(record_id,
        BaseModel)]``; and a bare ``list[BaseModel]`` (ids are minted). A record
        is a dict or a pydantic model; ``"__any__"`` as schema name applies to
        any schema. Several list-form records of a schema imply ``Many``; a ``One``
        takes exactly one.

        A ``Many`` schema without existing records is a multi-entity insert: every
        distinct entity of the schema the input mentions gets a record (discovery,
        coreference and qualification, then the full single-entity pipeline per
        entity, concurrently). With existing records of the schema each entity is
        also matched to one of them (or to none), one-to-one: a matched record is
        updated (``enable_updates``), an entity that matches none is inserted
        (``enable_inserts``), a record the input says should be removed is deleted
        (``enable_deletes``) and records the input does not mention are listed in
        ``untouched``; see ``docs/DESIGN.md``, "Collection engine".

        Args:
            input: The text, or an ``ExtractionInput`` carrying it. Explicit
                keyword arguments override the keys of an ``ExtractionInput``.
            existing: Existing records to update (see above). Records of a schema
                the extractor does not know are handled by
                ``existing_schema_policy``.
            schema: Force one of the extractor's schemas (a name, a schema or a
                wrapper): only it runs and its presence gate is skipped, like
                trustcall's ``tool_choice``.
            images: Images sent with every request that carries the text
                (data URLs or ``{"content_type", "base64"}`` objects).
            reference_date: Date relative dates are resolved against; defaults to
                the extractor's.

        Returns:
            The records, models, usage, cost, call log and warnings;
            ``untouched`` lists the ids of existing records the input did not
            touch (including opaque ones kept by ``existing_schema_policy=False``)
            and ``deleted`` those that were deleted.

        Raises:
            AmbiguousMatchError: With ``match.on_ambiguous="raise"``, for an
                entity that cannot be matched with enough certainty.
            ValueError: If ``schema`` is not one of the extractor's schemas, or
                ``existing`` is malformed, names an unknown schema under
                ``existing_schema_policy=True``, or gives a ``One`` several
                records.
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
        selected = self._select(schema)
        records = coerce_existing(
            existing,
            [self.plan_for(item).name for item in self.schemas],
            self.existing_schema_policy,
        )
        cards = [resolve_cardinality(item, existing) for item in selected]
        if not text.strip() and not images:
            return ExtractionResult(
                cost=self.prices.cost(Usage()),
                untouched=records.ids,
                warnings=(
                    *records.warnings,
                    "the input is empty; nothing was extracted",
                ),
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
            repair=self.repair,
            limits=self.limits,
            discovery=self.discovery,
            coref=self.coref,
            match=self.match,
            images=list(images) if images else None,
            reference_date=_iso_date(
                reference_date if reference_date is not None else self.reference_date
            ),
            strict=self.strict,
        )
        for warning in records.warnings:
            run.warn(warning)
        entries: list[_Entry] = []
        for card in cards:
            plan = self.plan_for(card)
            if isinstance(card, Many):
                entries.append(_Entry(plan, card.label or plan.title, many=card))
                continue
            entries.append(_Entry(plan, card.label or f"the {plan.title}"))

        async def everything() -> list[_Done]:
            claims, presences = await self._claim_any(
                run, records, entries, text, forced=schema is not None
            )
            jobs = []
            for entry in entries:
                name = entry.plan.name
                view = records.without_unclaimed_any(name, claims)
                if entry.many is not None:
                    jobs.append(
                        self._collect(
                            run,
                            entry,
                            text,
                            view.for_schema(name),
                            forced=schema is not None,
                            enable_inserts=self.enable_inserts,
                            enable_updates=self.enable_updates,
                            enable_deletes=self.enable_deletes,
                        )
                    )
                    continue
                target = target_for(name, view)
                known = presences.get(name)
                if target is None:
                    jobs.append(
                        self._insert(
                            run,
                            entry.plan,
                            text,
                            entry.label,
                            skip=schema is not None,
                            presence=known,
                        )
                    )
                else:
                    jobs.append(
                        self._update(
                            run,
                            entry.plan,
                            text,
                            target,
                            entry.label,
                            skip=schema is not None,
                            presence=known,
                        )
                    )
            return await gather(*jobs)

        try:
            outcomes = await everything()
        except TrustcallSystemOneError as exc:
            exc.calls = tuple(run.calls)
            exc.usage = run.usage
            exc.cost = self.prices.cost(run.usage)
            raise
        except Exception as exc:
            _note_failure(exc, run)
            raise
        results: list[ResultRecord] = [r for o in outcomes for r in o.records]
        handled = {h for o in outcomes for h in o.handled}
        return ExtractionResult(
            records=tuple(results),
            responses=tuple(
                r.value for r in results if r.valid and r.value is not None
            ),
            skipped=tuple(s for o in outcomes for s in o.skipped),
            untouched=records.untouched(handled),
            deleted=tuple(d for o in outcomes for d in o.deleted),
            attempts=max((o.attempts for o in outcomes), default=0),
            usage=run.usage,
            cost=self.prices.cost(run.usage),
            calls=tuple(run.calls),
            warnings=tuple(run.warnings),
        )

    async def _claim_any(
        self,
        run: Run,
        records: ExistingSet,
        entries: list["_Entry"],
        text: str,
        *,
        forced: bool,
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Decide which schema each ``"__any__"`` record belongs to.

        A record that may be of any schema is updated by at most ONE schema, or
        two schemas would write conflicting patches for it, and a schema takes at
        most ONE such record (a ``Many`` too: it then has that record to match its
        entities against). Claims are one-to-one: once a
        schema has claimed a record it leaves the candidates, and a record that
        no schema is left to claim stays untouched (with a warning), so this
        never raises after calls were spent.

        Among the schemas that have no record of their own and no claim yet,
        only those whose presence in the input passes the presence threshold
        can claim: a schema the input is not about cannot take a record just
        because its model happens to validate it (an all-optional schema
        validates almost anything). Of the present ones, the only one whose
        model validates the record claims it; when several or none validate, the
        most present does. The presence of every candidate is asked first, in
        parallel, and not asked again. A schema that loses the claim takes the
        insert path. A warning names the claimant.

        Returns:
            The claimant's schema name by record id, and the presence
            probabilities that were asked (so they are not asked again).
        """
        claims: dict[str, str] = {}
        presences: dict[str, float] = {}
        candidates = [e for e in entries if not records.named(e.plan.name)]
        open_ = list(candidates)  # schemas that have not claimed a record yet
        floor = run.presence.threshold_absent
        for record in records.named(ANY):
            if not open_:
                run.warn(
                    f"the {ANY!r} record {record.record_id!r} was claimed by no "
                    "schema ("
                    + (
                        "each candidate took another such record"
                        if candidates
                        else "every schema has records of its own"
                    )
                    + "); it is left untouched"
                )
                continue
            if len(candidates) <= 1:
                claims[record.record_id] = open_.pop().plan.name
                continue
            missing = [e for e in open_ if e.plan.name not in presences]
            if missing and not forced:
                asked = await gather(
                    *(ask_presence(run, e.plan, text, e.label) for e in missing)
                )
                presences.update(
                    {e.plan.name: p for e, p in zip(missing, asked, strict=True)}
                )
            present = [
                e for e in open_ if forced or presences.get(e.plan.name, 0.0) >= floor
            ]
            if not present:
                run.warn(
                    f"the {ANY!r} record {record.record_id!r} was claimed by no "
                    "schema (none of the candidates is present in the input); it "
                    "is left untouched"
                )
                continue
            fitting = [e for e in present if fits(e.plan, record)]
            if len(fitting) == 1:
                claim, how = (
                    fitting[0],
                    "the only present schema whose model validates it",
                )
            else:
                claim = max(
                    fitting or present, key=lambda e: presences.get(e.plan.name, 0.0)
                )
                how = (
                    "the schema most present in the input "
                    f"(p={presences.get(claim.plan.name, 0.0):.2f})"
                )
            open_.remove(claim)
            claims[record.record_id] = claim.plan.name
            others = [e.plan.name for e in candidates if e is not claim]
            run.warn(
                f"the {ANY!r} record {record.record_id!r} was given to "
                f"{claim.plan.name} ({how}); {', '.join(others)} take the insert "
                "path unless they claim another such record"
            )
        return claims, presences

    @staticmethod
    async def _insert(
        run: Run,
        plan: SchemaPlan,
        text: str,
        label: str,
        *,
        skip: bool,
        presence: float | None = None,
    ) -> _Done:
        outcome = await insert_one(
            run, plan, text, label=label, skip_presence=skip, presence=presence
        )
        return _Done(
            (outcome.record,) if outcome.record is not None else (),
            (outcome.skipped,) if outcome.skipped is not None else (),
            (),
            outcome.extracted,
            outcome.attempts,
        )

    @staticmethod
    async def _collect(
        run: Run,
        entry: _Entry,
        text: str,
        existing: Sequence[ExistingRecord],
        *,
        forced: bool,
        enable_inserts: bool,
        enable_updates: bool,
        enable_deletes: bool,
    ) -> _Done:
        """Deal with every entity of a ``Many`` schema the input mentions.

        Without existing records of the schema every entity is inserted; with them
        each entity is matched to a record (updated) or to none (inserted if
        inserts are enabled), and records the input says to remove are deleted.
        """
        assert entry.many is not None
        scope = Scope(
            plan=entry.plan,
            source=text,
            cardinality=entry.many,
            existing=tuple(existing),
        )
        outcome = await apply_collection(
            run,
            scope,
            skip_presence=forced,
            enable_inserts=enable_inserts,
            enable_updates=enable_updates,
            enable_deletes=enable_deletes,
        )
        return _Done(
            outcome.records,
            outcome.skipped,
            outcome.handled,
            outcome.extracted,
            outcome.attempts,
            outcome.deleted,
        )

    async def _update(
        self,
        run: Run,
        plan: SchemaPlan,
        text: str,
        target: ExistingRecord,
        label: str,
        *,
        skip: bool,
        presence: float | None = None,
    ) -> _Done:
        outcome = await update_one(
            run,
            plan,
            text,
            target,
            label=label,
            enable_updates=self.enable_updates,
            skip_presence=skip,
            presence=presence,
        )
        return _Done(
            (outcome.record,) if outcome.record is not None else (),
            (outcome.skipped,) if outcome.skipped is not None else (),
            (outcome.handled,) if outcome.handled is not None else (),
            outcome.extracted,
            outcome.attempts,
        )

    @overload
    async def abatch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = ...,
        schema: Any = ...,
        images: Sequence[Any] | None = ...,
        reference_date: date | str | None = ...,
        return_exceptions: Literal[False] = ...,
    ) -> list[ExtractionResult]: ...

    @overload
    async def abatch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = ...,
        schema: Any = ...,
        images: Sequence[Any] | None = ...,
        reference_date: date | str | None = ...,
        return_exceptions: bool,
    ) -> list[ExtractionResult | BaseException]: ...

    async def abatch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = None,
        schema: Any = None,
        images: Sequence[Any] | None = None,
        reference_date: date | str | None = None,
        return_exceptions: bool = False,
    ) -> list[ExtractionResult] | list[ExtractionResult | BaseException]:
        """Run ``ainvoke`` on every input concurrently.

        The keyword arguments apply to every input (an ``ExtractionInput``'s own
        keys are used when the argument is ``None``).

        Args:
            inputs: The inputs.
            existing: See ``ainvoke``.
            schema: See ``ainvoke``.
            images: See ``ainvoke``.
            reference_date: See ``ainvoke``.
            return_exceptions: If false (the default) the first failure cancels
                the other inputs and propagates. If true a failing input yields
                its exception at its position and the others still complete.

        Returns:
            One result per input, in order.
        """
        calls = [
            self.ainvoke(
                item,
                existing=existing,
                schema=schema,
                images=images,
                reference_date=reference_date,
            )
            for item in inputs
        ]
        if return_exceptions:
            outcomes: list[ExtractionResult | BaseException] = list(
                await asyncio.gather(*calls, return_exceptions=True)
            )
            return outcomes
        return await gather(*calls)

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

    @overload
    def batch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = ...,
        schema: Any = ...,
        images: Sequence[Any] | None = ...,
        reference_date: date | str | None = ...,
        return_exceptions: Literal[False] = ...,
    ) -> list[ExtractionResult]: ...

    @overload
    def batch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = ...,
        schema: Any = ...,
        images: Sequence[Any] | None = ...,
        reference_date: date | str | None = ...,
        return_exceptions: bool,
    ) -> list[ExtractionResult | BaseException]: ...

    def batch(
        self,
        inputs: Sequence[str | ExtractionInput],
        *,
        existing: Any = None,
        schema: Any = None,
        images: Sequence[Any] | None = None,
        reference_date: date | str | None = None,
        return_exceptions: bool = False,
    ) -> list[ExtractionResult] | list[ExtractionResult | BaseException]:
        """Run ``abatch`` on the extractor's private event loop thread."""
        return self.core.run_sync(
            self.abatch(
                inputs,
                existing=existing,
                schema=schema,
                images=images,
                reference_date=reference_date,
                return_exceptions=return_exceptions,
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
    repair: RepairConfig | None = None,
    discovery: DiscoveryConfig | None = None,
    coref: CorefConfig | None = None,
    match: MatchConfig | None = None,
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
        gate_model: Model id of the change-gate model (updates).
        enable_inserts: Allow inserting new records next to existing ones. With no
            ``existing`` records of a schema the extractor always inserts.
        enable_updates: Allow updating existing records.
        enable_deletes: Allow deleting existing records of a ``Many`` schema that
            the input says should be removed.
        enable_item_removals: Allow removing items from nested lists (M9).
        existing_schema_policy: How ``existing`` records of unknown schemas are
            treated: ``True`` raises ``ValueError``, ``False`` keeps them as
            context only (never patched, listed in ``untouched``, with a
            warning) and ``"ignore"`` drops them.
        presence: Presence-gate thresholds.
        verify: Verification-ladder settings.
        gates: Answer-level thresholds.
        repair: Validation repair loop settings (on by default): a record that
            fails validation gets one narrow re-ask per failing field.
        discovery: Settings of entity discovery for ``Many`` schemas (presence
            thresholds, how far the count of entities is trusted).
        coref: Settings of mention clustering for ``Many`` schemas (strategy,
            merge threshold, the size at which clustering becomes incremental).
        match: Settings of matching a ``Many`` schema's entities to existing
            records (acceptance thresholds, collisions, ambiguity, deletes).
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
            a schema cannot be compiled or ``Many(identity=...)`` names a field
            the schema lacks.
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
        repair=repair or RepairConfig(),
        discovery=discovery or DiscoveryConfig(),
        coref=coref or CorefConfig(),
        match=match or MatchConfig(),
        passes=passes or PassesConfig(),
        limits=limits or Limits(),
        concurrency=concurrency,
        prices=prices or Prices(),
        reference_date=reference_date,
        strict=strict,
    )
