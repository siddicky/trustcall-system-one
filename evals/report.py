"""JSON and Markdown reports of suites and sweeps."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.runner import SuiteResult, aggregate
from trustcall_system_one.testing import scrub

__all__ = ["suite_markdown", "sweep_markdown", "write_suite", "write_sweep"]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _row(label: str, agg: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {agg['passed']}/{agg['cases']} | {_pct(agg['op_precision'])} "
        f"| {_pct(agg['op_recall'])} | {_pct(agg['notes_recall'])} "
        f"| {_pct(agg['value_accuracy'])} "
        f"| {agg['bleed']} | {agg['wrong_deletes']} | {_pct(agg['valid_rate'])} "
        f"| {agg['requests']} | {agg['input_tokens']} | {agg['wall_s']:.1f} |"
    )


_HEADER = (
    "| {first} | passed | op precision | op recall | notes recall "
    "| value accuracy | bleed "
    "| wrong deletes | valid | requests | input tokens | wall s |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def suite_markdown(suite: SuiteResult) -> str:
    """A Markdown summary: the totals, then one row per case."""
    agg = aggregate(suite.results)
    lines = [
        f"# Evals: config `{suite.config}` ({suite.mode})",
        "",
        f"Overrides: `{dict(suite.overrides)}`"
        if suite.overrides
        else "Overrides: none",
        "",
        _HEADER.format(first="config"),
        _row(suite.config, agg),
        "",
        "## Cases",
        "",
        "| case | result | precision | recall | values | violations "
        "| requests | tokens | s | notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in suite.results:
        notes = r.error or "; ".join(
            [*(f"missing {m}" for m in r.missing), *(f"extra {e}" for e in r.extra)]
        )
        violations = "; ".join(f"{c}: {label}" for c, label in r.violations) or "-"
        values = f"{r.values_correct}/{r.n_values}" if r.n_values else "-"
        lines.append(
            f"| {r.case} | {'ok' if r.passed else 'FAIL'} | {_pct(r.precision)} "
            f"| {_pct(r.recall)} | {values} | {violations} | {r.requests} "
            f"| {r.input_tokens} | {r.wall_s:.1f} | {notes[:200]} |"
        )
    if suite.skipped:
        lines += ["", f"Skipped (no offline world): {', '.join(suite.skipped)}"]
    if agg["forwarded"]:
        lines += ["", f"Requests sent live during the replay: {agg['forwarded']}"]
    if suite.mode == "offline":
        lines += [
            "",
            "Offline: the models are scripted worlds, so these numbers check the "
            "harness and the library, not the models (token counts are the fake "
            "transport's constants).",
        ]
    return "\n".join(lines) + "\n"


def write_suite(suite: SuiteResult, out: Path) -> tuple[Path, Path]:
    """Write ``results.json`` and ``summary.md`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": suite.config,
        "mode": suite.mode,
        "overrides": dict(suite.overrides),
        "aggregate": aggregate(suite.results),
        "skipped": suite.skipped,
        "cases": [r.as_dict() for r in suite.results],
    }
    js, md = out / "results.json", out / "summary.md"
    js.write_text(scrub(json.dumps(payload, indent=2, default=str)))
    md.write_text(scrub(suite_markdown(suite)))
    return js, md


def sweep_markdown(
    params: Sequence[str], points: Sequence[tuple[dict[str, Any], SuiteResult]]
) -> str:
    """A table with one row per grid point."""
    head = _HEADER.format(first=" / ".join(params))
    lines = ["# Sweep", "", head]
    for values, suite in points:
        label = " / ".join(str(values[p]) for p in params)
        lines.append(_row(label, aggregate(suite.results)))
    forwarded = [aggregate(s.results)["forwarded"] for _, s in points]
    if any(forwarded):
        lines += [
            "",
            "Requests sent live per point (the rest were replayed): "
            + ", ".join(str(f) for f in forwarded),
        ]
    return "\n".join(lines) + "\n"


def write_sweep(
    params: Sequence[str],
    points: Sequence[tuple[dict[str, Any], SuiteResult]],
    out: Path,
) -> tuple[Path, Path]:
    """Write ``sweep.json`` and ``sweep.md`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "params": values,
            "mode": suite.mode,
            "aggregate": aggregate(suite.results),
            "cases": [r.as_dict() for r in suite.results],
        }
        for values, suite in points
    ]
    js, md = out / "sweep.json", out / "sweep.md"
    js.write_text(scrub(json.dumps(payload, indent=2, default=str)))
    md.write_text(scrub(sweep_markdown(params, points)))
    return js, md
