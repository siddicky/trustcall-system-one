"""The ``python -m evals`` command line: ``run``, ``sweep`` and ``list``."""

import argparse
import asyncio
import datetime as dt
import itertools
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.cases import CASES
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

__all__ = ["main", "select_cases"]


def select_cases(
    ids: Sequence[str] | None = None, tags: Sequence[str] | None = None
) -> list[Case]:
    """The cases picked by ``--cases`` (ids) and ``--tag`` (any tag), in order.

    Raises:
        ValueError: If an id is unknown or nothing is selected.
    """
    by_id = {c.id: c for c in CASES}
    chosen = list(CASES)
    if ids:
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"unknown case(s) {unknown}; see `python -m evals list`")
        chosen = [by_id[i] for i in ids]
    if tags:
        chosen = [c for c in chosen if set(tags) & set(c.tags)]
    if not chosen:
        raise ValueError("no case selected")
    return chosen


def _split(values: Sequence[str] | None) -> list[str]:
    return [p for v in values or () for p in v.split(",") if p]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cases", action="append", help="case ids (comma separated)")
    parser.add_argument("--tag", action="append", help="only cases with this tag")
    parser.add_argument(
        "--live",
        action="store_true",
        help="send requests to Cloudflare (needs CLOUDFLARE_ACCOUNT_ID and "
        "CLOUDFLARE_API_TOKEN); the default answers from each case's scripted world",
    )
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
    sub.add_parser("list", help="list the cases")
    return parser


def _out_dir(args: argparse.Namespace, name: str) -> Path:
    if args.out is not None:
        return Path(args.out)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("evals") / "results" / f"{stamp}-{name}"


def _settings(args: argparse.Namespace, overrides: dict[str, Any]) -> Settings:
    return Settings(
        overrides=overrides,
        live=args.live,
        record=args.record,
        replay=args.replay,
        concurrency=args.concurrency,
    )


def _base_overrides(args: argparse.Namespace) -> dict[str, Any]:
    if args.config not in PRESETS:
        raise ValueError(f"unknown config {args.config!r}; presets: {sorted(PRESETS)}")
    return {**PRESETS[args.config], **parse_overrides(args.set)}


def _run(args: argparse.Namespace) -> int:
    cases = select_cases(_split(args.cases), _split(args.tag))
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
    cases = select_cases(_split(args.cases), _split(args.tag))
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


def _list() -> int:
    for case in CASES:
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
            return _list()
        return _run(args) if args.command == "run" else _sweep(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
