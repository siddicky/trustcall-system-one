"""The ``python -m evals`` command line.

Commands: ``run``, ``sweep``, ``combo``, ``search`` and ``list``.
"""

import argparse
import asyncio
import datetime as dt
import itertools
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.configs import (
    PRESETS,
    build_options,
    classify,
    parse_overrides,
    parse_value,
)
from evals.model import Case
from evals.report import suite_markdown, sweep_markdown, write_suite, write_sweep
from evals.runner import Settings, SuiteResult, aggregate, run_suite
from evals.search import (
    DEFAULT_ORDER,
    GRID,
    SearchResult,
    Trial,
    leaderboard_markdown,
    load_previous,
    run_search,
    write_search,
)
from evals.suites import DEFAULT_SUITE, SUITES, load_suite
from trustcall_system_one.combos import PROVIDERS, ComboSpec, companies, load_combo

__all__ = ["main", "select_cases"]


def select_cases(
    ids: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    suite: str = DEFAULT_SUITE,
) -> list[Case]:
    """The cases picked by ``--cases`` (ids) and ``--tag`` (any tag), in order.

    Args:
        ids: Case ids (all of the suite when empty).
        tags: Keep the cases with any of these tags.
        suite: ``default`` (``evals/cases.py``) or ``holdout``
            (``evals/holdout_cases.py``).

    Raises:
        ValueError: If an id is unknown, the suite is unknown or missing, or
            nothing is selected.
    """
    cases = load_suite(suite)
    by_id = {c.id: c for c in cases}
    chosen = list(cases)
    if ids:
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(
                f"unknown case(s) {unknown} in the {suite} suite; see "
                f"`python -m evals list --suite {suite}`"
            )
        chosen = [by_id[i] for i in ids]
    if tags:
        chosen = [c for c in chosen if set(tags) & set(c.tags)]
    if not chosen:
        raise ValueError("no case selected")
    return chosen


def _triggers(text: str) -> frozenset[str]:
    """Parse ``--clef-fallback-on``."""
    names = frozenset(part.strip().lower() for part in text.split(",") if part.strip())
    allowed = {"unavailable", "uncertain", "bad_response"}
    if not names or not names <= allowed:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of {sorted(allowed)}, got {text!r}"
        )
    return names


def _split(values: Sequence[str] | None) -> list[str]:
    return [p for v in values or () for p in v.split(",") if p]


def _suite_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default=DEFAULT_SUITE,
        help="the case suite: default (the dev cases of evals/cases.py) or holdout "
        "(evals/holdout_cases.py: cases a search never sees)",
    )


def _output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record", type=Path, help="write cassettes to this directory")
    parser.add_argument(
        "--replay",
        type=Path,
        help="answer from the cassettes in this directory; with --live, requests "
        "that are not recorded go to the API",
    )
    parser.add_argument(
        "--out", type=Path, help="where to write results.json/summary.md"
    )
    parser.add_argument("--concurrency", type=int, default=4, help="cases at a time")
    parser.add_argument("--quiet", action="store_true", help="do not print the table")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cases", action="append", help="case ids (comma separated)")
    parser.add_argument("--tag", action="append", help="only cases with this tag")
    _suite_arg(parser)
    parser.add_argument(
        "--live",
        action="store_true",
        help="send requests to Cloudflare (needs CLOUDFLARE_ACCOUNT_ID and "
        "CLOUDFLARE_API_TOKEN), or to TypeSafe with --provider typesafe; the "
        "default answers from each case's scripted world",
    )
    parser.add_argument(
        "--clef-fallback",
        choices=("decisions", "none"),
        default=None,
        help="on live requests, answer a Clef request that the provider could "
        "not answer with the Decisions API (same key; may be charged twice "
        "after an uncertain outcome); default: TRUSTCALL_S1_CLEF_FALLBACK, "
        "else none",
    )
    parser.add_argument(
        "--clef-fallback-on",
        type=_triggers,
        default=None,
        metavar="LIST",
        help="which failures of Clef use the fallback: a comma-separated list of "
        "unavailable (nothing ran), uncertain (a timeout or 5xx: the primary may "
        "have run and charged, so the question can be paid for twice) and "
        "bad_response (the primary ran and charged, and answered badly); default: "
        "TRUSTCALL_S1_CLEF_FALLBACK_ON, else unavailable",
    )
    parser.add_argument(
        "--provider",
        choices=("cloudflare", "typesafe"),
        default=None,
        help="who serves BOTH models on live requests: cloudflare, or typesafe "
        "(Jev answers Clairic and Clef; needs TYPESAFE_API_KEY, no Cloudflare "
        "credential; excludes --clef-provider other than cloudflare and "
        "--clef-fallback); default: TRUSTCALL_S1_PROVIDER, else cloudflare (a "
        "TYPESAFE_API_KEY alone never switches). It only matters with --live",
    )
    parser.add_argument(
        "--clef-provider",
        choices=("cloudflare", "experiential", "decisions"),
        default=None,
        help="who serves Clef on live requests (Clairic stays on Cloudflare); "
        "default: TRUSTCALL_S1_CLEF_PROVIDER, else cloudflare (an "
        "EXPERIENTIAL_API_KEY alone never moves Clef). experiential and decisions "
        "need EXPERIENTIAL_API_KEY and never retry an uncertain request; decisions "
        "is the Decisions API (gpt-6-luna-decisions) answering ALL Clef requests, "
        "with its hard-pick cap of 0.8 and 32-question limit; it only matters "
        "with --live",
    )
    _output_args(parser)


def _combo_common(parser: argparse.ArgumentParser) -> None:
    """The arguments of ``combo`` and ``search``: a spec decides the providers."""
    parser.add_argument("--cases", action="append", help="case ids (comma separated)")
    parser.add_argument("--case-tag", action="append", help="only cases with this tag")
    _suite_arg(parser)
    parser.add_argument(
        "--live",
        action="store_true",
        help="send requests to the providers the combination names (credentials "
        "from the environment: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN, "
        "EXPERIENTIAL_API_KEY, TYPESAFE_API_KEY); a key alone never selects a "
        "provider, the spec does. The default answers from each case's scripted "
        "world",
    )
    parser.add_argument(
        "--tag",
        default="",
        metavar="NAME",
        help="a name for the run: recorded in results.json, and the directory under "
        "--record where its cassettes go (default: the spec's name)",
    )
    _output_args(parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the cases under one configuration")
    _common(run)
    run.add_argument(
        "--config",
        default="default",
        help=f"a preset ({', '.join(PRESETS)}) or `custom` with --set",
    )
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="GROUP.FIELD=VALUE",
        help="override one setting (repeatable)",
    )
    sweep = sub.add_parser("sweep", help="run the grid of parameter values")
    _common(sweep)
    sweep.add_argument(
        "--param",
        action="append",
        required=True,
        metavar="GROUP.FIELD=V1,V2,...",
        help="a parameter and its values (repeatable: the grid is the product)",
    )
    sweep.add_argument(
        "--config", default="default", help="the preset the grid starts from"
    )
    sweep.add_argument(
        "--set", action="append", default=[], metavar="GROUP.FIELD=VALUE"
    )
    combo = sub.add_parser(
        "combo",
        help="run the cases on a combination of providers (a spec file)",
        description="Run the cases on the combination of providers a spec names: "
        "the extraction provider per field type, the decision provider per call "
        "kind, calibration shifts and extractor knobs. Naming a provider in the "
        "spec is the opt-in to send text to it.",
    )
    _combo_common(combo)
    combo.add_argument(
        "--spec", type=Path, required=True, help="the spec (JSON, or YAML with PyYAML)"
    )
    combo.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="GROUP.FIELD=VALUE",
        help="override one extractor knob of the spec (repeatable)",
    )
    search = sub.add_parser(
        "search",
        help="greedy search for the best provider combination on the dev suite",
        description="Start from a base spec and, dimension by dimension (decision "
        "kinds in groups, then the extraction provider of each field type), try "
        "each candidate provider and keep it if the objective improves: more cases "
        "passed, then op F1, then value accuracy, then fewer bleed and wrong "
        "deletes (a rise in either is a veto). Dev suite only; check the winner "
        "with `combo --suite holdout`. A failed case is reported, never retried.",
    )
    _combo_common(search)
    search.add_argument(
        "--base", type=Path, help="the spec to start from (default: all Cloudflare)"
    )
    search.add_argument(
        "--providers",
        help="the candidate providers, comma separated, of "
        f"{', '.join(PROVIDERS)}; required with --live (naming them is the opt-in "
        "to send text to them); default offline: all",
    )
    search.add_argument(
        "--order",
        help="the dimensions to search, in order, comma separated (default: "
        f"{','.join(DEFAULT_ORDER)}; also: extract, other)",
    )
    search.add_argument("--max-runs", type=int, help="stop after this many evaluations")
    search.add_argument(
        "--budget-usd",
        type=float,
        help="stop before a trial that could take the estimated cost past this; "
        "needs a --price for every provider that can be used",
    )
    search.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="PROVIDER=IN:OUT",
        help="USD per million input:output tokens of a provider (repeatable)",
    )
    search.add_argument(
        "--resume",
        type=Path,
        help="continue the search in this results directory (search.json): "
        "combinations already evaluated are not run again",
    )
    search.add_argument(
        "--calibrate",
        action="store_true",
        help="after the routing, grid-search a logit shift per (provider, kind)",
    )
    search.add_argument(
        "--grid",
        default=None,
        metavar="LOW:HIGH:STEP",
        help="the shifts to try (default -1.5:1.5:0.5)",
    )
    search.add_argument(
        "--calibrate-kinds",
        help="the call kinds to calibrate, comma separated (default: every Clef kind)",
    )
    list_ = sub.add_parser("list", help="list the cases")
    _suite_arg(list_)
    return parser


def _out_dir(args: argparse.Namespace, name: str) -> Path:
    if args.out is not None:
        return Path(args.out)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("evals") / "results" / f"{stamp}-{name}"


def _settings(args: argparse.Namespace, overrides: dict[str, Any]) -> Settings:
    given = (
        args.provider
        or args.clef_provider
        or args.clef_fallback
        or args.clef_fallback_on
    )
    if given and not args.live:
        print(
            "note: --provider, --clef-provider, --clef-fallback and "
            "--clef-fallback-on have no effect without "
            "--live: nothing is sent, requests are answered offline or from the "
            "cassettes",
            file=sys.stderr,
        )
    return Settings(
        overrides=overrides,
        live=args.live,
        record=args.record,
        replay=args.replay,
        concurrency=args.concurrency,
        provider=args.provider,
        clef_provider=args.clef_provider,
        clef_fallback=args.clef_fallback,
        clef_fallback_on=args.clef_fallback_on,
        suite=args.suite,
    )


def _base_overrides(args: argparse.Namespace) -> dict[str, Any]:
    if args.config not in PRESETS:
        raise ValueError(f"unknown config {args.config!r}; presets: {sorted(PRESETS)}")
    return {**PRESETS[args.config], **parse_overrides(args.set)}


def _run(args: argparse.Namespace) -> int:
    cases = select_cases(_split(args.cases), _split(args.tag), args.suite)
    overrides = _base_overrides(args)
    suite = asyncio.run(
        run_suite(cases, _settings(args, overrides), config=args.config)
    )
    js, md = write_suite(suite, _out_dir(args, args.config))
    if not args.quiet:
        print(suite_markdown(suite))
    print(f"wrote {js} and {md}")
    return 0 if all(r.error is None for r in suite.results) else 1


def _sweep(args: argparse.Namespace) -> int:
    cases = select_cases(_split(args.cases), _split(args.tag), args.suite)
    base = _base_overrides(args)
    grid: dict[str, list[Any]] = {}
    for item in args.param:
        key, sep, values = item.partition("=")
        if not sep:
            raise ValueError(f"expected GROUP.FIELD=V1,V2,... got {item!r}")
        grid[key.strip()] = [parse_value(v) for v in values.split(",") if v.strip()]
    for key in grid:
        kind = classify(key)
        if kind == "changes requests":
            print(
                f"note: {key} changes the requests that are made; a replay sends "
                "every point's new requests live",
                file=sys.stderr,
            )
        elif kind == "unknown":
            print(
                f"note: {key} is not in the knob tables of evals/configs.py",
                file=sys.stderr,
            )
    params = list(grid)
    points: list[tuple[dict[str, Any], SuiteResult]] = []
    for combo in itertools.product(*grid.values()):
        values = dict(zip(params, combo, strict=True))
        try:
            build_options({**base, **values})
        except ValueError as exc:  # e.g. attribution_drop above attribution_keep
            print(f"skipping {values}: {exc}", file=sys.stderr)
            continue
        suite = asyncio.run(
            run_suite(cases, _settings(args, {**base, **values}), config=args.config)
        )
        points.append((values, suite))
        agg = aggregate(suite.results)
        print(f"{values}: passed {agg['passed']}/{agg['cases']}", file=sys.stderr)
    out = _out_dir(args, "sweep")
    js, md = write_sweep(params, points, out)
    if not args.quiet:
        print(sweep_markdown(params, points))
    print(f"wrote {js} and {md}")
    return 0


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def _disclose(spec: ComboSpec, live: bool) -> None:
    """Say who will see the text before a live run starts."""
    if live:
        print(
            f"live: the text goes to {', '.join(spec.companies())} "
            f"(providers: {', '.join(spec.providers())}); "
            "no request that may have been charged is ever retried",
            file=sys.stderr,
        )


def _combo(args: argparse.Namespace) -> int:
    spec = load_combo(args.spec)
    cases = select_cases(_split(args.cases), _split(args.case_tag), args.suite)
    tag = args.tag or spec.name or "combo"
    overrides = {**spec.options, **parse_overrides(args.set)}
    spec = spec.replace(options=overrides)
    _disclose(spec, args.live)
    settings = Settings(
        overrides=overrides,
        live=args.live,
        record=None if args.record is None else Path(args.record) / _slug(tag),
        replay=args.replay,
        concurrency=args.concurrency,
        combo=spec,
        suite=args.suite,
        tag=tag,
    )
    suite = asyncio.run(run_suite(cases, settings, config=tag))
    js, md = write_suite(suite, _out_dir(args, f"combo-{_slug(tag)}"))
    if not args.quiet:
        print(suite_markdown(suite))
    print(f"wrote {js} and {md}")
    return 0 if all(r.error is None for r in suite.results) else 1


def _prices(items: Sequence[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for item in items:
        name, sep, rest = item.partition("=")
        low, colon, high = rest.partition(":")
        try:
            if name not in PROVIDERS or not sep or not colon:
                raise ValueError
            out[name] = (float(low), float(high))
        except ValueError:
            raise ValueError(
                f"expected PROVIDER=IN:OUT (USD per million tokens) with a provider "
                f"of {list(PROVIDERS)}, got {item!r}"
            ) from None
    return out


def _grid(text: str | None) -> tuple[float, ...]:
    if text is None:
        return GRID
    try:
        low, high, step = (float(p) for p in text.split(":"))
        if step <= 0 or high < low:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"--grid is LOW:HIGH:STEP with STEP > 0, got {text!r}"
        ) from None
    count = round((high - low) / step) + 1
    return tuple(round(low + i * step, 6) for i in range(count))


def _search(args: argparse.Namespace) -> int:
    if args.suite != DEFAULT_SUITE:
        raise ValueError(
            "a search runs on the dev suite only (--suite default): choosing on the "
            "holdout would make it a dev set. Check the winner with "
            "`python -m evals combo --spec <out>/best.json --suite holdout`"
        )
    base = load_combo(args.base) if args.base else ComboSpec(name="base")
    providers = _split([args.providers]) if args.providers else []
    if not providers:
        if args.live:
            raise ValueError(
                "--live needs --providers: naming the candidates is the opt-in to "
                f"send text to them (of {', '.join(PROVIDERS)})"
            )
        providers = list(PROVIDERS)
    order = _split([args.order]) if args.order else list(DEFAULT_ORDER)
    prices = _prices(args.price)
    if args.budget_usd is not None and args.live:
        unpriced = sorted((set(providers) | set(base.providers())) - set(prices))
        if unpriced:
            raise ValueError(
                f"--budget-usd needs a price for every provider that can be used: "
                f"add --price {unpriced[0]}=IN:OUT (USD per million tokens)"
            )
    cases = select_cases(_split(args.cases), _split(args.case_tag), args.suite)
    out = Path(args.resume or args.out or _out_dir(args, "search"))
    header = {
        "suite": args.suite,
        "cases": [c.id for c in cases],
        "providers": providers,
        "order": order,
        "live": bool(args.live),
        "calibrate": bool(args.calibrate),
        "grid": list(_grid(args.grid)),
    }
    previous: list[Trial] = []
    if args.resume:
        old, previous = load_previous(Path(args.resume))
        for name in ("suite", "cases"):
            if old.get(name) != header[name]:
                raise ValueError(
                    f"cannot resume: the earlier search ran another {name} "
                    f"({old.get(name)!r}); a result is only reused on the same cases"
                )
    if args.live:
        every = list(dict.fromkeys([*base.providers(), *providers]))
        print(
            f"live: the text may go to {', '.join(companies(every))} (providers: "
            f"{', '.join(every)}); no request that may have been charged is ever "
            "retried, a failed case is reported and the search goes on",
            file=sys.stderr,
        )

    async def evaluate(spec: ComboSpec, label: str) -> SuiteResult:
        settings = Settings(
            overrides={**spec.options},
            live=args.live,
            record=None if args.record is None else Path(args.record) / label,
            replay=args.replay,
            concurrency=args.concurrency,
            combo=spec,
            suite=args.suite,
            tag=label,
        )
        return await run_suite(cases, settings, config=label)

    def on_trial(trial: Trial, suite: SuiteResult | None, so_far: SearchResult) -> None:
        if suite is not None:
            where = out / "trials" / f"{trial.n:02d}-{_slug(trial.change)}"
            write_suite(suite, where)
            trial.path = str(where)
        write_search(out, so_far, header)

    result = asyncio.run(
        run_search(
            base,
            evaluate,
            providers=providers,
            order=order,
            prices=prices,
            max_runs=args.max_runs,
            budget_usd=args.budget_usd,
            previous=previous,
            calibrate=args.calibrate,
            grid=_grid(args.grid),
            calibrate_kinds=_split(
                [args.calibrate_kinds] if args.calibrate_kinds else None
            )
            or None,
            on_trial=on_trial,
            log=lambda line: print(line, file=sys.stderr),
        )
    )
    js, md, best = write_search(out, result, header)
    if not args.quiet:
        print(leaderboard_markdown(result))
    print(f"wrote {js}, {md} and {best}")
    return 0


def _list(suite: str = DEFAULT_SUITE) -> int:
    for case in load_suite(suite):
        offline = "" if case.world is not None else " (live only)"
        print(f"{case.id:34} [{', '.join(case.tags)}]{offline}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m evals``.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).

    Returns:
        The exit status: 0, or 1 when a case could not run.
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            return _list(args.suite)
        if args.command == "combo":
            return _combo(args)
        if args.command == "search":
            return _search(args)
        return _run(args) if args.command == "run" else _sweep(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
