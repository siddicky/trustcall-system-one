"""Running cases against an extractor configuration and aggregating the scores."""

import asyncio
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from evals.configs import build_options
from evals.model import Case, CaseResult
from evals.scoring import score
from trustcall_system_one import (
    Transport,
    clef_fallback_from_env,
    clef_fallback_on_from_env,
    clef_provider_from_env,
    create_extractor,
    provider_from_env,
    system_one_transport,
)
from trustcall_system_one.combos import (
    ComboSpec,
    build_combo_transport,
    calibrate_transport,
    create_provider_transports,
    provider_label,
)
from trustcall_system_one.testing import CassetteTransport, RecordingTransport, scrub

__all__ = [
    "Settings",
    "SuiteResult",
    "aggregate",
    "combo_labels",
    "missing_credentials",
    "run_suite",
    "usage_by_provider",
]

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
        provider: Who serves BOTH models when requests go live: ``cloudflare``
            (the default) or ``typesafe`` (Jev for Clairic and Clef; needs
            ``TYPESAFE_API_KEY``). ``None`` decides from ``TRUSTCALL_S1_PROVIDER``.
        clef_provider: Who serves Clef when requests go live with the Cloudflare
            provider: ``cloudflare``, ``experiential`` or ``decisions`` (the Decisions
            API answers every Clef request as the primary). ``None`` decides from
            ``TRUSTCALL_S1_CLEF_PROVIDER`` (Cloudflare when it is unset; an
            ``EXPERIENTIAL_API_KEY`` alone never selects another). Clairic goes to
            Cloudflare unless ``provider`` is ``typesafe``.
        clef_fallback: ``decisions`` adds the Decisions API as the fallback of
            Clef on live requests; ``None`` reads ``TRUSTCALL_S1_CLEF_FALLBACK``
            (no fallback when it is unset).
        clef_fallback_on: Which failures of Clef trigger the fallback (a subset of
            ``unavailable``, ``uncertain`` and ``bad_response``); ``None`` reads
            ``TRUSTCALL_S1_CLEF_FALLBACK_ON``, by default ``unavailable`` only.
            ``uncertain`` can mean two charges for one question.
        combo: A combination of providers (``trustcall_system_one.combos``) that
            answers instead of the provider settings above (which must then be
            left unset). Live, the providers it names are created (and need their
            credentials); offline each provider is a scripted world.
        members: Offline, with ``combo``: builds the transport of each provider the
            combination names for a case (default: the case's scripted world for
            every provider). Lets a test make one provider better at one kind.
        suite: The name of the case suite, recorded in the results.
        tag: A label of the run, recorded in the results.
    """

    overrides: Mapping[str, Any] = field(default_factory=dict)
    live: bool = False
    record: Path | None = None
    replay: Path | None = None
    concurrency: int = 4
    provider: Literal["cloudflare", "typesafe"] | None = None
    clef_provider: Literal["cloudflare", "experiential", "decisions"] | None = None
    clef_fallback: Literal["decisions", "none"] | None = None
    clef_fallback_on: frozenset[str] | None = None
    combo: ComboSpec | None = None
    members: Callable[[Case], Mapping[str, Transport]] | None = None
    suite: str = "default"
    tag: str = ""


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
        clef_fallback: ``decisions (on ...)`` when the Decisions API was the
            fallback, with the failures that trigger it.
        provider: ``cloudflare`` or ``typesafe``: who served both models (empty
            without ``live``).
        extraction: A label of what answered extraction (Clairic), for tabulating
            combinations: ``cloudflare clairic``, ``typesafe jev`` or, offline,
            ``scripted world``.
        decisions: A label of what answered the decisions (Clef): ``cloudflare
            clef``, ``experiential clef``, ``experiential decisions
            (gpt-6-luna-decisions, hard cap 0.8)`` (plus ``+ fallback decisions``),
            ``typesafe jev`` or ``scripted world``.
        combo: The combination of providers that answered (``None`` for the usual
            provider settings); it is written, with its labels, to ``results.json``.
        suite: The case suite (``default`` or ``holdout``).
        tag: The label of the run.
        replay: The cassette directory the run replayed from (empty: none). With
            ``mode`` ``replay+live`` the requests that were not recorded went live.
    """

    config: str
    overrides: Mapping[str, Any]
    results: list[CaseResult]
    skipped: list[str]
    mode: str
    clef_provider: str = ""
    clef_fallback: str = ""
    provider: str = ""
    extraction: str = ""
    decisions: str = ""
    combo: ComboSpec | None = None
    suite: str = "default"
    tag: str = ""
    replay: str = ""


def combination(
    provider: str, clef_provider: str, clef_fallback: str
) -> tuple[str, str]:
    """The labels of what answers extraction and what answers decisions."""
    if provider == "typesafe":
        return "typesafe jev", "typesafe jev"
    decisions = {
        "cloudflare": "cloudflare clef",
        "experiential": "experiential clef",
        "decisions": "experiential decisions (gpt-6-luna-decisions, hard cap 0.8)",
    }.get(clef_provider, clef_provider)
    if clef_fallback == "decisions":
        decisions += " + fallback decisions"
    return "cloudflare clairic", decisions


_PROVIDER_LABELS = {
    "cloudflare": ("cloudflare clairic", "cloudflare clef"),
    "typesafe": ("typesafe jev", "typesafe jev"),
    "experiential-clef": ("", "experiential clef"),
    "decisions": ("", "experiential decisions (gpt-6-luna-decisions, hard cap 0.8)"),
}


def combo_labels(spec: ComboSpec) -> tuple[str, str]:
    """The labels of what answers extraction and what answers decisions in a combo.

    For example ``cloudflare clairic (date, datetime: typesafe jev)`` and
    ``cloudflare clef (attribution: experiential decisions ...)``.
    """
    spec = spec.canonical()
    extraction = _PROVIDER_LABELS[spec.extraction][0]
    types = sorted(spec.extraction_by_type.items())
    if types:
        extraction += (
            " (" + "; ".join(f"{t}: {_PROVIDER_LABELS[p][0]}" for t, p in types) + ")"
        )
    decisions = _PROVIDER_LABELS[spec.decision][1]
    kinds = sorted(spec.by_kind.items())
    if kinds:
        decisions += (
            " (" + "; ".join(f"{k}: {_PROVIDER_LABELS[p][1]}" for k, p in kinds) + ")"
        )
    return extraction, decisions


def missing_credentials(spec: ComboSpec) -> list[str]:
    """The environment variables a combination's live run needs and does not have.

    Names only, never values.
    """
    named = spec.providers()
    missing: list[str] = []
    if "cloudflare" in named:
        if not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
            missing.append("CLOUDFLARE_ACCOUNT_ID")
        if not (
            os.environ.get("CLOUDFLARE_API_TOKEN")
            or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
        ):
            missing.append("CLOUDFLARE_API_TOKEN")
    if ("experiential-clef" in named or "decisions" in named) and not (
        os.environ.get("EXPERIENTIAL_API_KEY") or ""
    ).strip():
        missing.append("EXPERIENTIAL_API_KEY")
    if "typesafe" in named and not (os.environ.get("TYPESAFE_API_KEY") or "").strip():
        missing.append("TYPESAFE_API_KEY")
    return missing


def usage_by_provider(
    calls: Sequence[Any], replayed: Mapping[str, Mapping[str, float]] | None = None
) -> dict[str, dict[str, float]]:
    """Requests, tokens and reported cost per provider, from a call log.

    Failed calls are included. ``replayed`` (``_replayed``) is what each provider's
    calls answered from a cassette used: it is added as ``replayed_requests``,
    ``replayed_input_tokens``, ``replayed_output_tokens`` and ``replayed_cost_usd``,
    so that a cost estimate can leave it out (a replay spends nothing).
    """
    out: dict[str, dict[str, float]] = {}
    for call in calls:
        row = out.setdefault(
            str(call.provider or "unknown"),
            {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        row["requests"] += 1
        row["input_tokens"] += int(call.usage.get("input_tokens", 0))
        row["output_tokens"] += int(call.usage.get("output_tokens", 0))
        row["cost_usd"] += float(call.cost_usd or 0.0)
    for label, used in (replayed or {}).items():
        row = out.setdefault(
            label,
            {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        row["replayed_requests"] = used["requests"]
        row["replayed_input_tokens"] = used["input_tokens"]
        row["replayed_output_tokens"] = used["output_tokens"]
        row["replayed_cost_usd"] = used["cost"]
    return out


def _replayed(cassettes: Sequence[CassetteTransport]) -> dict[str, dict[str, float]]:
    """What the calls answered from the cassettes used, by provider."""
    out: dict[str, dict[str, float]] = {}
    for cassette in cassettes:
        label = cassette.provider
        if label is None or not cassette.replayed["requests"]:
            continue
        row = out.setdefault(label, dict.fromkeys(cassette.replayed, 0.0))
        for key, value in cassette.replayed.items():
            row[key] += value
    return out


def _combo_stack(
    case: Case,
    settings: Settings,
    spec: ComboSpec,
    shared: Mapping[str, Transport] | None,
    snapshot: CassetteTransport | None,
    cassettes: list[CassetteTransport],
    name: str,
) -> Transport:
    """The transport of a combination for one case.

    Each provider's transport (shared real ones live, the case's scripted ones
    offline) is wrapped on its own: a recorder that writes what THAT provider
    answered, then a cassette that replays only what that provider recorded
    (``CassetteTransport(provider=...)``), so a spec that routes a call to another
    provider never gets the answer of the one that was recorded; a call with no
    recording is a miss, and goes live only when ``--live`` was given (the cassette's
    ``forwarded`` says how many did). The calibration is outermost, over the raw
    answers.
    """
    if shared is not None:
        providers: Mapping[str, Transport] = shared
    else:
        assert case.world is not None
        scripted = settings.members(case) if settings.members is not None else None
        providers = scripted or {p: case.world().transport() for p in spec.providers()}

    def wrap(provider: str, member: Transport) -> Transport:
        if settings.record is not None:
            member = RecordingTransport(member, settings.record / f"{name}.jsonl")
        if settings.replay is not None:
            source = snapshot or CassetteTransport(settings.replay)
            cassette = source.fork(
                strict=not settings.live,
                inner=member if settings.live else None,
                provider=provider_label(provider),
            )
            cassettes.append(cassette)
            return cassette
        return member

    stack = build_combo_transport(spec, providers=providers, calibrate=False, wrap=wrap)
    return calibrate_transport(spec, stack)


_RECORDED_ON = {
    "cloudflare": ("cloudflare clairic", "cloudflare clef"),
    "typesafe": ("typesafe jev", "typesafe jev"),
    "experiential": ("experiential clef", "experiential clef"),
    "decisions": (
        "experiential decisions",
        "experiential decisions (gpt-6-luna-decisions, hard cap 0.8)",
    ),
}


def recorded_labels(snapshot: CassetteTransport) -> tuple[str, str]:
    """The labels of a pure replay: what the cassettes were recorded on.

    Each recording says who answered it (``provider``); a cassette from before that
    was recorded says nothing, and its label is ``unknown``.
    """
    found = snapshot.providers()

    def label(family: str, index: int) -> str:
        names = [_RECORDED_ON.get(p, (p, p))[index] for p in found[family]]
        return f"replay (recorded on {', '.join(names) or 'unknown'})"

    return label("clairic", 0), label("clef", 1)


def _have_credentials(
    clef_provider: str = "cloudflare",
    clef_fallback: str = "none",
    provider: str = "cloudflare",
) -> bool:
    """Whether the credentials of the providers in use are set.

    Cloudflare (Clairic), or TypeSafe alone, and the Experiential key for Experiential
    or Decisions Clef.
    """
    if provider == "typesafe":
        return bool((os.environ.get("TYPESAFE_API_KEY") or "").strip())
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get(
        "CLOUDFLARE_AUTH_TOKEN"
    )
    needs_key = clef_provider in ("experiential", "decisions") or (
        clef_fallback == "decisions"
    )
    if needs_key and not (os.environ.get("EXPERIENTIAL_API_KEY") or "").strip():
        return False
    return bool(account and token)


def _fallback_on(settings: Settings) -> frozenset[str]:
    """The failures that trigger the fallback: the setting, else the environment."""
    return settings.clef_fallback_on or clef_fallback_on_from_env()


def _fallback(settings: Settings) -> Literal["decisions", "none"]:
    """The Clef fallback: the setting, else the environment."""
    return settings.clef_fallback or clef_fallback_from_env()


def _provider(settings: Settings) -> Literal["cloudflare", "experiential", "decisions"]:
    """The provider that serves Clef: the setting, else the environment."""
    return settings.clef_provider or clef_provider_from_env()


def _main_provider(settings: Settings) -> Literal["cloudflare", "typesafe"]:
    """The provider that serves both models: the setting, else the environment."""
    return settings.provider or provider_from_env()


def count_unknown_outcomes(calls: Sequence[Any]) -> int:
    """Calls that failed after they may have been run (and charged).

    These are the ``UncertainOutcomeError`` calls and the calls that were
    cancelled after they may have been sent (``CallRecord.error`` starts with
    ``cancelled after send``).
    """
    return sum(
        1
        for call in calls
        if call.error
        and (
            call.error.startswith("UncertainOutcomeError")
            or call.error.startswith("cancelled after send")
        )
    )


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
    shared: Mapping[str, Transport] | None = None,
) -> CaseResult:
    options = build_options(settings.overrides, case.options)
    spec = settings.combo.canonical() if settings.combo is not None else None
    name = _SAFE.sub("_", case.id)
    cassettes: list[CassetteTransport] = []
    transport: Transport
    if spec is not None:
        transport = _combo_stack(
            case, settings, spec, shared, snapshot, cassettes, name
        )
    else:
        base: Transport
        if settings.live:
            assert live is not None
            base = live
        else:
            assert case.world is not None
            base = case.world().transport()
        if settings.record is not None:
            base = RecordingTransport(base, settings.record / f"{name}.jsonl")
        transport = base
        if settings.replay is not None:
            # one snapshot of the directory per suite: cases record into it while
            # others replay from it
            source = snapshot or CassetteTransport(settings.replay)
            cassettes.append(
                source.fork(
                    strict=not settings.live, inner=base if settings.live else None
                )
            )
            transport = cassettes[0]
    started = time.perf_counter()
    secrets: tuple[str, ...] = ()
    for owner in (live, *(shared or {}).values()):
        sensitive = getattr(owner, "sensitive_values", None)
        if callable(sensitive):
            secrets = tuple(dict.fromkeys([*secrets, *sensitive()]))
    try:
        extractor = create_extractor(case.schemas(), transport=transport, **options)
        result = await extractor.ainvoke(case.text, existing=case.existing)
    except Exception as exc:
        failed = CaseResult(case=case.id)
        # the failed invoke's log: requests that were sent, answered or not
        failed.requests = len(getattr(exc, "calls", ()))
        failed.unknown_outcomes = count_unknown_outcomes(getattr(exc, "calls", ()))
        spent = getattr(exc, "usage", None)
        failed.reported_cost_usd = getattr(spent, "reported_cost_usd", 0.0)
        failed.output_tokens = getattr(spent, "output_tokens", 0)
        failed.by_provider = usage_by_provider(
            getattr(exc, "calls", ()), _replayed(cassettes)
        )
        # an error message can carry the request URL, and with it the account id
        failed.error = scrub(f"{type(exc).__name__}: {str(exc)[:300]}", secrets)
        failed.wall_s = time.perf_counter() - started
        failed.n_expected = len(case.expect_ops)
        failed.n_values = len(case.expect_values)
        failed.forwarded = sum(c.forwarded for c in cassettes)
        return failed
    scored = score(case, result)
    scored.unknown_outcomes = count_unknown_outcomes(result.calls)
    scored.reported_cost_usd = result.usage.reported_cost_usd
    scored.output_tokens = result.usage.output_tokens
    scored.by_provider = usage_by_provider(result.calls, _replayed(cassettes))
    scored.warnings = [scrub(w, secrets) for w in scored.warnings]
    scored.wall_s = time.perf_counter() - started
    scored.forwarded = sum(c.forwarded for c in cassettes)
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
        RuntimeError: If ``live`` is set without the credentials of the providers.
        ValueError: If a combination is given together with the provider settings,
            or the provider settings contradict each other.
    """
    if settings.combo is not None:
        return await _run_combo(cases, settings, config)
    if _main_provider(settings) == "typesafe" and (
        _provider(settings) != "cloudflare" or _fallback(settings) != "none"
    ):  # Jev answers Clef itself: nothing else may be asked for
        raise ValueError(
            "--provider typesafe answers Clef with Jev itself: do not combine "
            "it with --clef-provider or --clef-fallback (or the "
            "TRUSTCALL_S1_CLEF_PROVIDER / TRUSTCALL_S1_CLEF_FALLBACK variables)"
        )
    runnable = [c for c in cases if settings.live or c.world is not None]
    skipped = [c.id for c in cases if c not in runnable]
    # Shown in the report whenever a live request is possible: live, or a replay
    # that sends the requests it has no recording for.
    possible = settings.live or settings.replay is not None
    main = _main_provider(settings) if possible else ""
    provider = "typesafe" if main == "typesafe" else _provider(settings) if main else ""
    fallback = _fallback(settings) if main == "cloudflare" else "none"
    if settings.live and not _have_credentials(
        provider or "cloudflare", fallback, main or "cloudflare"
    ):
        if main == "typesafe":
            raise RuntimeError(
                "--live with --provider typesafe needs TYPESAFE_API_KEY in the "
                "environment"
            )
        raise RuntimeError(
            "--live needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the "
            "environment"
            + (
                ", and EXPERIENTIAL_API_KEY for Experiential, the Decisions API or "
                "the Decisions fallback"
                if provider in ("experiential", "decisions") or fallback == "decisions"
                else ""
            )
        )
    if settings.record is not None:
        settings.record.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(max(1, settings.concurrency))
    live = None
    if settings.live and main == "typesafe":
        live = system_one_transport(
            settings.clef_provider,
            provider="typesafe",
            clef_fallback=settings.clef_fallback,
        )
    elif settings.live and provider:
        live = system_one_transport(
            provider,  # type: ignore[arg-type]
            provider="cloudflare",
            clef_fallback=fallback,
            clef_fallback_on=_fallback_on(settings),
        )
    snapshot = CassetteTransport(settings.replay) if settings.replay else None
    if snapshot is not None and not settings.live:  # nothing is sent: what was recorded
        extraction, decisions = recorded_labels(snapshot)
    elif possible:
        extraction, decisions = combination(main, provider, fallback)
    else:
        extraction = decisions = "scripted world"

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
        f"{fallback} (on {','.join(sorted(_fallback_on(settings)))})"
        if provider and fallback != "none"
        else "",
        main,
        extraction,
        decisions,
        suite=settings.suite,
        tag=settings.tag,
    )


async def _run_combo(
    cases: Sequence[Case], settings: Settings, config: str
) -> SuiteResult:
    """Run the cases on a combination of providers (``Settings.combo``).

    The spec is the opt-in: it names every provider, and nothing in the environment
    (``TRUSTCALL_S1_PROVIDER`` and the like) is read or changes it.
    """
    assert settings.combo is not None
    if (
        settings.provider
        or settings.clef_provider
        or settings.clef_fallback
        or settings.clef_fallback_on
    ):
        raise ValueError(
            "a combination names its own providers: do not combine it with "
            "--provider, --clef-provider, --clef-fallback or --clef-fallback-on"
        )
    spec = settings.combo.canonical()
    runnable = [c for c in cases if settings.live or c.world is not None]
    skipped = [c.id for c in cases if c not in runnable]
    if settings.live:
        missing = missing_credentials(spec)
        if missing:
            raise RuntimeError(
                f"--live with this combination ({', '.join(spec.providers())}) needs "
                f"{', '.join(missing)} in the environment"
            )
    if settings.record is not None:
        settings.record.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(max(1, settings.concurrency))
    # one transport per provider, shared by the cases (each case stacks its own
    # recorder and cassette on top of them)
    shared = create_provider_transports(spec) if settings.live else None
    snapshot = CassetteTransport(settings.replay) if settings.replay else None
    if snapshot is not None and not settings.live:
        extraction, decisions = recorded_labels(snapshot)
    else:
        extraction, decisions = combo_labels(spec)
        if not settings.live and settings.replay is None:
            extraction, decisions = (
                f"{extraction} (scripted)",
                f"{decisions} (scripted)",
            )

    async def one(case: Case) -> CaseResult:
        async with gate:
            return await _run_case(case, settings, None, snapshot, shared)

    try:
        results = list(await asyncio.gather(*(one(c) for c in runnable)))
    finally:
        for member in (shared or {}).values():
            closer = getattr(member, "aclose", None)
            if callable(closer):
                await closer()
    return SuiteResult(
        config,
        dict(settings.overrides),
        results,
        skipped,
        mode_of(settings),
        extraction=extraction,
        decisions=decisions,
        combo=settings.combo,
        suite=settings.suite,
        tag=settings.tag,
        replay=str(settings.replay) if settings.replay is not None else "",
    )


def aggregate(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Totals and rates over a suite (micro-averaged over operations and values).

    Returns:
        ``op_precision``, ``op_recall``, ``op_f1`` (their harmonic mean),
        ``value_accuracy`` (``None`` without expected values), ``notes_recall`` (the
        expected notes that were produced), ``bleed`` and ``wrong_deletes``
        (violations by kind), ``violations``, ``valid_rate``, ``requests``,
        ``input_tokens``, ``output_tokens``, ``wall_s``, ``cases``, ``passed``,
        ``errors`` and ``forwarded``.
    """

    def ratio(num: int, den: int) -> float | None:
        return num / den if den else None

    produced = sum(r.n_actual for r in results)
    expected = sum(r.n_expected for r in results)
    values = sum(r.n_values for r in results)
    records = sum(r.n_records for r in results)
    categories = [c for r in results for c, _ in r.violations]
    precision = sum(r.matched_actual for r in results) / produced if produced else 1.0
    recall = sum(r.matched_expected for r in results) / expected if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
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
        "op_f1": f1,
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "wall_s": round(sum(r.wall_s for r in results), 2),
        "forwarded": sum(r.forwarded for r in results),
        "unknown_outcomes": sum(r.unknown_outcomes for r in results),
        "reported_cost_usd": round(sum(r.reported_cost_usd for r in results), 6),
    }
