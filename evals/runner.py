"""Running cases against an extractor configuration and aggregating the scores."""

import asyncio
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from evals.configs import build_options
from evals.model import Case, CaseResult
from evals.scoring import score
from trustcall_system_one import (
    Transport,
    clef_provider_from_env,
    create_extractor,
    system_one_transport,
)
from trustcall_system_one.testing import CassetteTransport, RecordingTransport, scrub

__all__ = ["Settings", "SuiteResult", "aggregate", "run_suite"]

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Settings:
    """How a suite runs.

    Attributes:
        overrides: ``group.field`` overrides of the extractor configuration.
        live: Send requests to Cloudflare (credentials from the environment).
            Offline, each case is answered by its scripted ``FakeWorld``.
        record: Directory to write cassettes to (``<case id>.jsonl``). With a
            replay, only the requests that were not replayed are recorded.
        replay: Directory (or file) of cassettes to answer from. With ``live``,
            a request that is not recorded goes to the live API; without it,
            such a request fails the case (``CassetteMissError``).
        concurrency: Cases run at the same time.
        clef_provider: Who serves Clef when requests go live: ``cloudflare`` or
            ``experiential``. ``None`` decides from ``TRUSTCALL_S1_CLEF_PROVIDER``
            and ``EXPERIENTIAL_API_KEY`` (Cloudflare when neither is set).
            Clairic always goes to Cloudflare.
    """

    overrides: Mapping[str, Any] = field(default_factory=dict)
    live: bool = False
    record: Path | None = None
    replay: Path | None = None
    concurrency: int = 4
    clef_provider: Literal["cloudflare", "experiential"] | None = None


@dataclass
class SuiteResult:
    """The outcome of a suite.

    Attributes:
        config: The configuration's name.
        overrides: Its overrides.
        results: One score per case that ran.
        skipped: Ids of cases that could not run (no offline world).
        mode: ``offline``, ``live``, ``replay`` or ``replay+live``.
        clef_provider: Who served the live Clef requests (empty without ``live``).
    """

    config: str
    overrides: Mapping[str, Any]
    results: list[CaseResult]
    skipped: list[str]
    mode: str
    clef_provider: str = ""


def _have_credentials(clef_provider: str = "cloudflare") -> bool:
    """Cloudflare credentials (Clairic) and, for Experiential Clef, its key."""
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get(
        "CLOUDFLARE_AUTH_TOKEN"
    )
    if clef_provider == "experiential" and not os.environ.get("EXPERIENTIAL_API_KEY"):
        return False
    return bool(account and token)


def _provider(settings: Settings) -> Literal["cloudflare", "experiential"]:
    """The provider that serves Clef: the setting, else the environment."""
    return settings.clef_provider or clef_provider_from_env()


def mode_of(settings: Settings) -> str:
    """A label for how requests are answered."""
    if settings.replay is not None:
        return "replay+live" if settings.live else "replay"
    return "live" if settings.live else "offline"


async def _run_case(
    case: Case,
    settings: Settings,
    live: Transport | None,
    snapshot: CassetteTransport | None = None,
) -> CaseResult:
    options = build_options(settings.overrides, case.options)
    base: Transport
    if settings.live:
        assert live is not None
        base = live
    else:
        assert case.world is not None
        base = case.world().transport()
    name = _SAFE.sub("_", case.id)
    cassette: CassetteTransport | None = None
    if settings.record is not None:
        base = RecordingTransport(base, settings.record / f"{name}.jsonl")
    transport: Transport = base
    if settings.replay is not None:
        # one snapshot of the directory per suite: cases record into it while
        # others replay from it
        source = snapshot or CassetteTransport(settings.replay)
        cassette = source.fork(
            strict=not settings.live, inner=base if settings.live else None
        )
        transport = cassette
    started = time.perf_counter()
    secrets: tuple[str, ...] = ()
    sensitive = getattr(live, "sensitive_values", None)
    if callable(sensitive):
        secrets = tuple(sensitive())
    try:
        extractor = create_extractor(case.schemas(), transport=transport, **options)
        result = await extractor.ainvoke(case.text, existing=case.existing)
    except Exception as exc:
        failed = CaseResult(case=case.id)
        # an error message can carry the request URL, and with it the account id
        failed.error = scrub(f"{type(exc).__name__}: {str(exc)[:300]}", secrets)
        failed.wall_s = time.perf_counter() - started
        failed.n_expected = len(case.expect_ops)
        failed.n_values = len(case.expect_values)
        if cassette is not None:
            failed.forwarded = cassette.forwarded
        return failed
    scored = score(case, result)
    scored.warnings = [scrub(w, secrets) for w in scored.warnings]
    scored.wall_s = time.perf_counter() - started
    if cassette is not None:
        scored.forwarded = cassette.forwarded
    return scored


async def run_suite(
    cases: Sequence[Case], settings: Settings, *, config: str = "default"
) -> SuiteResult:
    """Run every case under one configuration.

    Args:
        cases: The cases.
        settings: How to run.
        config: The configuration's name (for reports).

    Returns:
        The scores, in case order.

    Raises:
        RuntimeError: If ``live`` is set without Cloudflare credentials.
    """
    runnable = [c for c in cases if settings.live or c.world is not None]
    skipped = [c.id for c in cases if c not in runnable]
    provider = _provider(settings) if settings.live else ""
    if settings.live and not _have_credentials(provider):
        raise RuntimeError(
            "--live needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the "
            "environment"
            + (
                ", and EXPERIENTIAL_API_KEY for --clef-provider experiential"
                if provider == "experiential"
                else ""
            )
        )
    if settings.record is not None:
        settings.record.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(max(1, settings.concurrency))
    live = system_one_transport(provider or None) if settings.live else None
    snapshot = CassetteTransport(settings.replay) if settings.replay else None

    async def one(case: Case) -> CaseResult:
        async with gate:
            return await _run_case(case, settings, live, snapshot)

    try:
        results = list(await asyncio.gather(*(one(c) for c in runnable)))
    finally:
        if live is not None:
            await live.aclose()
    return SuiteResult(
        config,
        dict(settings.overrides),
        results,
        skipped,
        mode_of(settings),
        provider,
    )


def aggregate(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Totals and rates over a suite (micro-averaged over operations and values).

    Returns:
        ``op_precision``, ``op_recall``, ``value_accuracy`` (``None`` without
        expected values), ``notes_recall`` (the expected notes that were
        produced), ``bleed`` and ``wrong_deletes`` (violations by kind),
        ``violations``, ``valid_rate``, ``requests``, ``input_tokens``,
        ``wall_s``, ``cases``, ``passed``, ``errors`` and ``forwarded``.
    """

    def ratio(num: int, den: int) -> float | None:
        return num / den if den else None

    produced = sum(r.n_actual for r in results)
    expected = sum(r.n_expected for r in results)
    values = sum(r.n_values for r in results)
    records = sum(r.n_records for r in results)
    categories = [c for r in results for c, _ in r.violations]
    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "errors": sum(1 for r in results if r.error),
        "op_precision": ratio(sum(r.matched_actual for r in results), produced)
        if produced
        else 1.0,
        "op_recall": ratio(sum(r.matched_expected for r in results), expected)
        if expected
        else 1.0,
        "value_accuracy": ratio(sum(r.values_correct for r in results), values),
        "notes_recall": ratio(
            sum(r.notes_found for r in results), sum(r.n_notes for r in results)
        ),
        "bleed": categories.count("bleed"),
        "wrong_deletes": categories.count("wrong_delete"),
        "violations": len(categories),
        "valid_rate": ratio(sum(r.valid for r in results), records),
        "requests": sum(r.requests for r in results),
        "input_tokens": sum(r.input_tokens for r in results),
        "wall_s": round(sum(r.wall_s for r in results), 2),
        "forwarded": sum(r.forwarded for r in results),
    }
