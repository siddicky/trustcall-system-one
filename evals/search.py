"""Searching for the best provider combination: a greedy coordinate search.

``python -m evals search`` starts from a base ``ComboSpec`` and walks the dimensions
of a combination in a given order (the decision kinds in groups, then the extraction
provider of each field type). For each dimension it tries every candidate provider
and keeps the one that improves the objective on the dev suite, then moves on. With
``--calibrate`` it then grid-searches a logit shift per (provider, call kind).

**The objective** is lexicographic: more cases passed, then a higher operation F1,
then a higher value accuracy, then fewer bleed and wrong deletes. Bleed and wrong
deletes going UP is a veto, whatever else improved: a combination that leaks a value
onto another record or deletes a record it should not is never kept for a gain
elsewhere.

**Nothing is retried.** A case that fails is a failed case in the table (its error is
reported), never a reason to send it again; a trial that raises is recorded as an
error and the search goes on (the base trial is the exception: without it there is
nothing to compare with). A trial is not run twice either: the same combination
(``ComboSpec.key``) is looked up, also in the trials of a results directory that is
resumed.

**Cost** is estimated from usage (``estimate_cost``): the tokens each provider used
times its price (``--price provider=in:out`` USD per million tokens), plus what the
providers reported for themselves. ``--max-runs`` and ``--budget-usd`` stop the search
between trials, never in the middle of one.
"""

import dataclasses
import json
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.model import CaseResult
from evals.runner import SuiteResult, aggregate
from trustcall_system_one.combos import (
    CLAIRIC_PROVIDERS,
    PROVIDERS,
    REMOVAL_KINDS,
    ComboSpec,
)
from trustcall_system_one.testing import scrub
from trustcall_system_one.transport import CLAIRIC_KINDS

__all__ = [
    "DEFAULT_ORDER",
    "DIMENSIONS",
    "EFFORT_CANDIDATES",
    "EFFORT_DIMENSION",
    "GRID",
    "Margin",
    "Objective",
    "SearchResult",
    "Trial",
    "compare",
    "estimate_cost",
    "leaderboard_markdown",
    "load_previous",
    "run_search",
    "write_search",
]

DIMENSIONS: Mapping[str, tuple[str, ...]] = {
    "presence": ("presence", "qualify", "count"),
    "coref": ("coref",),
    "match": ("match", "match_confirm"),
    "attribution": ("attribution", "stated", "null_check"),
    "gates": ("gate", "novelty", "list_dedup", "relative_edit"),
    "clef_fields": ("clef_fields",),
    "verify": ("crosscheck", "reask", "adjudicate"),
    "other": ("record_presence", "delete", "sentences", "list_kind"),
}
"""The call kinds of each decision dimension. A candidate provider is applied to the
kinds it can serve (``count``, ``crosscheck`` and ``reask`` are Clairic calls: only
``cloudflare`` and ``typesafe`` can take them)."""
EFFORT_DIMENSION = "escalate.effort"
"""The dimension that sweeps ``reasoning.effort`` of the spec's escalation tier."""
EFFORT_CANDIDATES: tuple[str, ...] = ("none", "low", "medium", "high")
"""The efforts the search tries (a spec can name ``xhigh`` and ``max`` itself)."""
EXTRACTION_TYPES: tuple[str, ...] = (
    "text",
    "number",
    "date",
    "time",
    "datetime",
    "duration",
    "list",
)
DEFAULT_ORDER: tuple[str, ...] = (
    "presence",
    "coref",
    "match",
    "attribution",
    "gates",
    "clef_fields",
    "verify",
    *(f"extract:{t}" for t in EXTRACTION_TYPES),
)
"""The order of the dimensions. ``extract`` (the default provider of every field type)
and ``other`` are dimensions too; name them in ``--order`` to search them."""
GRID: tuple[float, ...] = tuple(round(-1.5 + 0.5 * i, 1) for i in range(7))
"""The default grid of logit shifts: -1.5 to 1.5 in steps of 0.5."""

_LABEL_TO_PROVIDER = {
    "cloudflare": "cloudflare",
    "experiential": "experiential-clef",
    "decisions": "decisions",
    "typesafe": "typesafe",
}
_EPS = 1e-9


@dataclass(frozen=True)
class Objective:
    """What a combination is judged by (see the module docstring)."""

    passed: int
    f1: float
    value_accuracy: float
    bleed: int
    wrong_deletes: int

    @classmethod
    def of(cls, agg: Mapping[str, Any]) -> "Objective":
        """The objective of a suite's ``aggregate``."""
        accuracy = agg.get("value_accuracy")
        return cls(
            passed=int(agg["passed"]),
            f1=float(agg.get("op_f1") or 0.0),
            value_accuracy=1.0 if accuracy is None else float(accuracy),
            bleed=int(agg["bleed"]),
            wrong_deletes=int(agg["wrong_deletes"]),
        )

    @property
    def rank(self) -> tuple[int, float, float, int]:
        """The tuple that orders objectives (greater is better)."""
        return (
            self.passed,
            round(self.f1, 6),
            round(self.value_accuracy, 6),
            -(self.bleed + self.wrong_deletes),
        )


@dataclass(frozen=True)
class Margin:
    """How much better than the incumbent a candidate must be to be kept.

    A difference of one case, or of a few hundredths of F1, is within what two runs of
    the same combination can differ by live; a search that keeps such a "win" walks
    towards noise. ``passed`` is a number of cases (at least 1), ``f1`` the op F1
    (and the value accuracy) by which a candidate has to beat the incumbent when
    the passed cases are equal.
    """

    passed: int = 1
    f1: float = 0.02


def compare(
    candidate: Objective, best: Objective, margin: Margin | None = None
) -> tuple[bool, str]:
    """Whether ``candidate`` replaces ``best``, and why or why not.

    Bleed or wrong deletes going up is a veto. Otherwise the candidate has to pass at
    least ``margin.passed`` more cases; or the same or more and an op F1, or a value
    accuracy, higher by at least ``margin.f1``; or lose nothing and have fewer bleed
    and wrong deletes (a safety gain needs no margin).

    Args:
        candidate: The objective of the candidate.
        best: The incumbent's.
        margin: The margin (default: one case, or 0.02 of F1).
    """
    margin = margin or Margin()
    if candidate.bleed > best.bleed:
        return False, f"veto: bleed {best.bleed} -> {candidate.bleed}"
    if candidate.wrong_deletes > best.wrong_deletes:
        return (
            False,
            f"veto: wrong deletes {best.wrong_deletes} -> {candidate.wrong_deletes}",
        )
    need = max(margin.passed, 1)
    if candidate.passed >= best.passed + need:
        return True, "improves the objective"
    if candidate.passed >= best.passed:
        step = max(margin.f1, 1e-6)
        if candidate.f1 - best.f1 >= step - _EPS:
            return True, "improves the objective"
        no_worse = candidate.f1 >= best.f1 - _EPS
        if no_worse and candidate.value_accuracy - best.value_accuracy >= step - _EPS:
            return True, "improves the objective"
        safer = candidate.bleed + candidate.wrong_deletes < (
            best.bleed + best.wrong_deletes
        )
        if (
            no_worse
            and safer
            and candidate.value_accuracy >= best.value_accuracy - _EPS
        ):
            return True, "improves the objective"
    if candidate.rank > best.rank:
        return False, "better, but by less than the margin"
    return False, "no gain"


def estimate_cost(
    results: Sequence[CaseResult], prices: Mapping[str, tuple[float, float]]
) -> tuple[float, bool]:
    """The cost of a suite in USD, estimated from its usage, and whether it is partial.

    Per provider (``CaseResult.by_provider``): when it has a price (``(input,
    output)`` USD per million tokens) its tokens are charged at it; when it has none
    but reported a cost (Experiential Labs reports one per call) that cost is added.
    Calls answered from a cassette are not counted (a replay spends nothing). The
    estimate is ``partial`` whenever a provider that spent tokens has neither a
    price nor a reported cost.

    Args:
        results: The case results.
        prices: Provider name (``cloudflare``, ``typesafe``, ``decisions``,
            ``experiential-clef``) to ``(input, output)`` price per million tokens.

    Returns:
        ``(usd, partial)``.
    """
    totals: dict[str, list[float]] = {}
    for r in results:
        for label, row in r.by_provider.items():
            name = _LABEL_TO_PROVIDER.get(label, label)
            used = totals.setdefault(name, [0.0, 0.0, 0.0])
            used[0] += row["input_tokens"] - row.get("replayed_input_tokens", 0)
            used[1] += row["output_tokens"] - row.get("replayed_output_tokens", 0)
            used[2] += row.get("cost_usd", 0.0) - row.get("replayed_cost_usd", 0.0)
    usd = 0.0
    partial = False
    for name, (inp, out, reported) in totals.items():
        if name in prices:
            low, high = prices[name]
            usd += inp * low / 1e6 + out * high / 1e6
        elif reported > 0:
            usd += reported
        elif inp > 0 or out > 0:
            partial = True
    return usd, partial


@dataclass
class Trial:
    """One combination that was evaluated (or is a cached evaluation of one).

    Attributes:
        n: Its number in the search.
        spec: The combination.
        dimension: What was varied (``presence``, ``extract:date``,
            ``calibrate:typesafe.gate``, or ``base``).
        change: How, for example ``presence,qualify,count=decisions``.
        outcome: ``base``, ``kept`` (it became the best so far), ``rejected``,
            ``veto`` (bleed or wrong deletes rose) or ``error``.
        note: Why.
        aggregate: The suite's ``aggregate``.
        cost_usd: The estimated cost, and ``cost_partial`` whether it misses a
            provider that has no price.
        failures: The cases that did not pass, with what was wrong (no retry).
        cached: Whether it came from a resumed results directory.
        error: What stopped the trial, if it did.
        path: The directory of its ``results.json``.
        mode: How it ran (``offline``, ``live``, ``replay:<dir>``,
            ``live+replay:<dir>``): a trial is only reused by a search in the same
            mode, so an offline result never stands in for a live one.
    """

    n: int
    spec: ComboSpec
    dimension: str
    change: str
    outcome: str = "rejected"
    note: str = ""
    aggregate: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    cost_partial: bool = False
    failures: list[dict[str, Any]] = field(default_factory=list)
    cached: bool = False
    error: str | None = None
    path: str | None = None
    mode: str = ""

    @property
    def objective(self) -> Objective:
        """The objective of the trial (it must have an aggregate)."""
        return Objective.of(self.aggregate)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready form."""
        data = dataclasses.asdict(self)
        data["spec"] = self.spec.model_dump()
        data["key"] = self.spec.key()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trial":
        """Read a trial back from ``as_dict``."""
        names = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in names}
        values["spec"] = ComboSpec.model_validate(data["spec"])
        return cls(**values)


@dataclass
class SearchResult:
    """What a search found.

    Attributes:
        trials: Every combination tried, in order (cached ones included).
        best: The best trial.
        stopped: Why the search ended early (``--max-runs``, ``--budget-usd``), or
            empty when it ran every dimension.
        spent_usd: The estimated cost of all the trials.
        not_run: Candidates that were not tried and why (an invalid spec, say).
    """

    trials: list[Trial]
    best: Trial
    stopped: str = ""
    spent_usd: float = 0.0
    not_run: list[str] = field(default_factory=list)


Evaluate = Callable[[ComboSpec, str], Awaitable[SuiteResult]]
"""Runs the dev suite on a combination (the second argument is the trial's label)."""


class _Stop(Exception):
    """The budget says no more trials."""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", text).strip("_")[:60] or "trial"


def _failures(results: Sequence[CaseResult]) -> list[dict[str, Any]]:
    return [
        {
            "case": r.case,
            "error": r.error,
            "missing": r.missing[:5],
            "extra": r.extra[:5],
            "violations": [f"{c}: {label}" for c, label in r.violations],
        }
        for r in results
        if not r.passed
    ]


def _apply(spec: ComboSpec, dimension: str, provider: str) -> ComboSpec | None:
    """``spec`` with ``provider`` on ``dimension``, or ``None`` if it cannot be."""
    if dimension == "extract":
        return spec.replace(extraction=provider)
    if dimension.startswith("extract:"):
        if provider not in CLAIRIC_PROVIDERS:
            return None
        kind = dimension.partition(":")[2]
        return spec.replace(
            extraction_by_type={**spec.extraction_by_type, kind: provider}
        )
    kinds = DIMENSIONS[dimension]
    chosen = {
        k: provider
        for k in kinds
        if not (k in CLAIRIC_KINDS and provider not in CLAIRIC_PROVIDERS)
    }
    return spec.replace(by_kind={**spec.by_kind, **chosen}) if chosen else None


def _why(exc: ValueError) -> str:
    """The first line of a validation error."""
    return " ".join(str(exc).split("\n")[:2])[:200]


def _check_order(order: Sequence[str]) -> None:
    for name in order:
        if name in DIMENSIONS or name in ("extract", EFFORT_DIMENSION):
            continue
        if name.startswith("extract:") and name.partition(":")[2]:
            continue
        raise ValueError(
            f"unknown dimension {name!r}; dimensions: "
            f"{[*DIMENSIONS, 'extract', 'extract:<field type>', EFFORT_DIMENSION]}"
        )


def _effective_provider(spec: ComboSpec, kind: str) -> str:
    """The provider that answers a Clef call kind in a spec."""
    return spec.by_kind.get(kind) or spec.by_kind.get(f"{kind}@clef") or spec.decision


async def run_search(
    base: ComboSpec,
    evaluate: Evaluate,
    *,
    providers: Sequence[str],
    order: Sequence[str] = DEFAULT_ORDER,
    prices: Mapping[str, tuple[float, float]] | None = None,
    max_runs: int | None = None,
    budget_usd: float | None = None,
    previous: Sequence[Trial] = (),
    calibrate: bool = False,
    grid: Sequence[float] = GRID,
    calibrate_kinds: Sequence[str] | None = None,
    allow_positive_removal_shift: bool = False,
    mode: str = "offline",
    retry_errors: bool = False,
    margin: Margin | None = None,
    repeats: int = 1,
    on_trial: Callable[[Trial, SuiteResult | None, SearchResult], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> SearchResult:
    """Run the greedy search.

    Args:
        base: The combination to start from.
        evaluate: Runs the dev suite on a combination.
        providers: The candidate providers of every dimension.
        order: The dimensions, in the order they are searched.
        prices: USD per million (input, output) tokens by provider.
        max_runs: Stop after this many new evaluations (``None``: no limit).
        budget_usd: Stop before a trial whose estimated cost would take the total
            past this (the estimate is the dearest trial so far). The total counts
            the trials of ``previous``.
        previous: Trials of an earlier search (``load_previous``): a combination
            that was evaluated, IN THE SAME ``mode``, is taken from here instead of
            being run again; so is one that errored, unless ``retry_errors``.
        calibrate: After the routing, grid-search a logit shift per (provider, kind).
        grid: The shifts to try.
        calibrate_kinds: The call kinds to calibrate. The default is every Clef kind
            of ``DIMENSIONS`` EXCEPT those that can authorise a removal
            (``REMOVAL_KINDS``: ``gate``, ``null_check``, ``delete``), which must be
            listed here, and then only shifts of 0 or less are tried unless
            ``allow_positive_removal_shift``.
        allow_positive_removal_shift: Try positive shifts of the removal kinds too.
        mode: How the evaluations run (``offline``, ``live``, ``replay:<dir>``...),
            recorded on each trial and matched when ``previous`` is reused.
        retry_errors: Run again the trials of ``previous`` that errored.
        margin: How much better than the incumbent a candidate must be to be kept
            (see ``Margin``; default one case or 0.02 of F1).
        repeats: A candidate that wins narrowly (less than two cases and 0.05 of F1
            ahead) is evaluated this many times in all and kept only if every run
            beats the incumbent by the margin (each run counts for ``max_runs`` and
            the budget).
        on_trial: Called once a trial's outcome is known, with the trial, its suite
            (``None`` for a cached or failed one) and the search so far, to write it
            down as the search goes.
        log: Called with a line of progress.

    Returns:
        The trials and the best of them.

    Raises:
        ValueError: If a provider or dimension is unknown.
    """
    _check_order(order)
    unknown = sorted(set(providers) - set(PROVIDERS))
    if unknown:
        raise ValueError(f"unknown provider(s) {unknown}; expected one of {PROVIDERS}")
    price = dict(prices or {})
    note = log or (lambda line: None)
    margin = margin or Margin()
    cache = {
        t.spec.key(): t
        for t in previous
        if t.mode == mode and (t.aggregate or (t.error and not retry_errors))
    }
    trials: list[Trial] = []
    seen: set[str] = set()
    not_run: list[str] = []
    state = {"runs": 0, "spent": sum(t.cost_usd or 0.0 for t in previous)}
    dearest = [max((t.cost_usd or 0.0 for t in previous), default=0.0)]

    def refuse() -> str:
        if max_runs is not None and state["runs"] >= max_runs:
            return f"--max-runs {max_runs} reached"
        if budget_usd is not None and state["spent"] + dearest[0] > budget_usd:
            return (
                f"--budget-usd {budget_usd:g}: ${state['spent']:.4f} spent and a "
                f"trial costs up to ${dearest[0]:.4f}"
            )
        return ""

    def record(trial: Trial, suite: SuiteResult | None, best: Trial) -> None:
        if on_trial is not None:
            on_trial(
                trial,
                suite,
                SearchResult(list(trials), best, "", state["spent"], list(not_run)),
            )

    async def attempt(
        spec: ComboSpec, dimension: str, change: str, *, first: bool = False
    ) -> tuple[Trial, SuiteResult | None] | None:
        key = spec.key()
        if key in seen:
            return None
        seen.add(key)
        n = len(trials)
        old = cache.get(key)
        if old is not None:
            trial = dataclasses.replace(
                old,
                n=n,
                spec=spec,
                dimension=dimension,
                change=change,
                outcome="error" if old.error else "rejected",
                note="",
                cached=True,
            )
            trials.append(trial)
            note(
                f"[{n}] {change}: cached ("
                + (f"error {trial.error}" if trial.error else _summary(trial))
                + ")"
            )
            return trial, None
        stop = refuse()
        if stop and not first:  # the base always runs: there is nothing to compare with
            raise _Stop(stop)
        label = f"{n:02d}-{_slug(change)}"
        state["runs"] += 1
        try:
            suite = await evaluate(spec, label)
        except Exception as exc:
            if first:
                raise
            trial = Trial(
                n,
                spec,
                dimension,
                change,
                outcome="error",
                note="the trial raised; it was not retried",
                error=scrub(f"{type(exc).__name__}: {str(exc)[:200]}"),
                mode=mode,
            )
            trials.append(trial)
            note(f"[{n}] {change}: error {trial.error}")
            return trial, None
        agg = aggregate(suite.results)
        usd, partial = estimate_cost(suite.results, price)
        trial = Trial(
            n,
            spec,
            dimension,
            change,
            aggregate=agg,
            cost_usd=usd,
            cost_partial=partial,
            failures=_failures(suite.results),
            mode=mode,
        )
        state["spent"] += usd
        dearest[0] = max(dearest[0], usd)
        trials.append(trial)
        note(f"[{n}] {change}: {_summary(trial)}")
        return trial, suite

    best: Trial | None = None
    stopped = ""

    async def confirm(spec: ComboSpec, trial: Trial, label: str) -> str:
        """Evaluate a narrow winner again; "" if every repeat beat the incumbent."""
        assert best is not None
        for again in range(2, repeats + 1):
            stop = refuse()
            if stop:
                return f"not confirmed: {stop}"
            state["runs"] += 1
            try:
                suite = await evaluate(spec, f"{label}-repeat{again}")
            except Exception as exc:
                return f"not confirmed: repeat {again} raised {type(exc).__name__}"
            usd, _ = estimate_cost(suite.results, price)
            state["spent"] += usd
            dearest[0] = max(dearest[0], usd)
            ok, why = compare(
                Objective.of(aggregate(suite.results)), best.objective, margin
            )
            if not ok:
                return f"won once but not on repeat {again} ({why})"
        return ""

    async def consider(spec: ComboSpec, dimension: str, change: str) -> None:
        nonlocal best
        assert best is not None
        found = await attempt(spec, dimension, change)
        if found is None:
            return
        trial, suite = found
        if trial.error is None:
            ok, why = compare(trial.objective, best.objective, margin)
            trial.note = why
            if ok and repeats > 1 and _narrow(trial.objective, best.objective):
                refused = await confirm(spec, trial, f"{trial.n:02d}-{_slug(change)}")
                if refused:
                    ok, trial.note = False, refused
            if ok:
                trial.outcome = "kept"
                best = trial
            else:
                trial.outcome = "veto" if why.startswith("veto") else "rejected"
        record(trial, suite, best)

    try:
        found = await attempt(base.canonical(), "base", "base", first=True)
        assert found is not None
        first, first_suite = found
        first.outcome, first.note = "base", "the combination searched from"
        best = first
        record(first, first_suite, best)
        for dimension in order:
            if dimension == EFFORT_DIMENSION:
                # the tier's reasoning effort: same margin and veto as any dimension,
                # on the dev suite (a search never runs on the holdout)
                tier = best.spec.escalate
                if tier is None:
                    not_run.append(
                        f"{EFFORT_DIMENSION}: the base spec has no escalate to sweep"
                    )
                    continue
                for effort in EFFORT_CANDIDATES:
                    if effort == best.spec.escalate.reasoning_effort:  # type: ignore[union-attr]
                        continue
                    swept = best.spec.replace(
                        escalate=tier.model_copy(update={"reasoning_effort": effort})
                    )
                    await consider(swept, dimension, f"{dimension}={effort}")
                continue
            for provider in providers:
                try:
                    spec = _apply(best.spec, dimension, provider)
                except ValueError as exc:
                    not_run.append(f"{dimension}={provider}: {_why(exc)}")
                    continue
                if spec is None or spec.key() == best.spec.key():
                    continue
                await consider(spec, dimension, f"{dimension}={provider}")
        if calibrate:
            for kind, provider in _calibration_pairs(best.spec, calibrate_kinds):
                for shift in grid:
                    cal: dict[str, dict[str, float]] = {
                        p: dict(t) for p, t in best.spec.calibration.items()
                    }
                    current = cal.get(provider, {}).get(kind, 0.0)
                    if abs(shift - current) < _EPS:
                        continue
                    change = f"cal {provider}.{kind}{shift:+g}"
                    if (
                        kind in REMOVAL_KINDS
                        and shift > 0
                        and not allow_positive_removal_shift
                    ):
                        not_run.append(
                            f"{change}: {kind} can authorise a removal; a positive "
                            "shift needs --allow-positive-removal-shift"
                        )
                        continue
                    cal[provider] = {**cal.get(provider, {}), kind: shift}
                    try:
                        spec = best.spec.replace(calibration=cal)
                    except ValueError as exc:
                        not_run.append(f"{change}: {_why(exc)}")
                        continue
                    await consider(spec, f"calibrate:{provider}.{kind}", change)
    except _Stop as stop:
        stopped = str(stop)
        note(f"stopped: {stopped}")
    assert best is not None
    return SearchResult(trials, best, stopped, state["spent"], not_run)


def _narrow(candidate: Objective, best: Objective) -> bool:
    """Whether a candidate is ahead by little: under two cases and 0.05 of F1."""
    return candidate.passed - best.passed < 2 and candidate.f1 - best.f1 < 0.05


def _calibration_pairs(
    spec: ComboSpec, kinds: Sequence[str] | None
) -> Iterator[tuple[str, str]]:
    """``(kind, provider)`` of the Clef kinds to calibrate, as the spec routes them.

    Without ``kinds``, the kinds that can authorise a removal are left out.
    """
    wanted = kinds or [
        k
        for group in DIMENSIONS.values()
        for k in group
        if k not in CLAIRIC_KINDS and k not in REMOVAL_KINDS
    ]
    for kind in dict.fromkeys(wanted):
        if kind in CLAIRIC_KINDS:
            raise ValueError(
                f"{kind!r} is a Clairic call: only Clef probabilities move"
            )
        yield kind, _effective_provider(spec, kind)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _summary(trial: Trial) -> str:
    agg = trial.aggregate
    return (
        f"passed {agg['passed']}/{agg['cases']}, F1 {_pct(agg.get('op_f1'))}, "
        f"values {_pct(agg.get('value_accuracy'))}, bleed {agg['bleed']}, "
        f"wrong deletes {agg['wrong_deletes']}"
    )


def _cost(trial: Trial) -> str:
    if trial.cost_usd is None:
        return "-"
    return f"{'~' if trial.cost_partial else ''}${trial.cost_usd:.4f}"


def _flag(trial: Trial) -> str:
    """A marker for a combination that calibrates a removal kind upwards."""
    shifts = trial.spec.removal_shifts()
    return f" **[removal shift: {', '.join(shifts)}]**" if shifts else ""


def leaderboard_markdown(result: SearchResult) -> str:
    """A Markdown leaderboard of every combination tried, best first.

    Columns: the combination, how it was reached and what became of it, passed
    cases, op F1, value accuracy, bleed, wrong deletes, requests, tokens (in/out),
    estimated cost (``~`` when a provider had no price) and seconds per case.
    """
    tiered = any(
        t.spec.escalate is not None or (t.aggregate or {}).get("escalation_calls")
        for t in result.trials
    )
    lines = [
        "# Combination search",
        "",
        f"Best: `{result.best.spec.describe()}`"
        + (f" (trial {result.best.n})" if result.best.n else " (the base)"),
        "",
        f"Trials: {len(result.trials)}"
        + f", estimated cost ${result.spent_usd:.4f}"
        + (f"; stopped: {result.stopped}" if result.stopped else ""),
        "",
        "| # | combination | change | result | pass | op F1 | value acc | bleed "
        "| wrong del | requests | tokens in/out | cost | s/case |"
        + (" esc calls | esc reasoning tok | esc ms/call |" if tiered else ""),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        + ("---|---|---|" if tiered else ""),
    ]
    done = [t for t in result.trials if t.aggregate]
    ranked = sorted(done, key=lambda t: t.objective.rank, reverse=True)
    for t in [*ranked, *(t for t in result.trials if not t.aggregate)]:
        agg = t.aggregate
        star = "best " if t is result.best else ""
        outcome = f"{star}{t.outcome}" + (" (cached)" if t.cached else "")
        if not agg:
            lines.append(
                f"| {t.n} | `{t.spec.describe()}`{_flag(t)} | {t.change} | {outcome}: "
                f"{t.error or t.note} | - | - | - | - | - | - | - | - | - |"
                + (" - | - | - |" if tiered else "")
            )
            continue
        per_case = agg["wall_s"] / agg["cases"] if agg["cases"] else 0.0
        lines.append(
            f"| {t.n} | `{t.spec.describe()}`{_flag(t)} | {t.change} | {outcome} "
            f"| {agg['passed']}/{agg['cases']} | {_pct(agg.get('op_f1'))} "
            f"| {_pct(agg.get('value_accuracy'))} | {agg['bleed']} "
            f"| {agg['wrong_deletes']} | {agg['requests']} "
            f"| {agg['input_tokens']}/{agg.get('output_tokens', 0)} | {_cost(t)} "
            f"| {per_case:.2f} |"
            + (
                f" {agg.get('escalation_calls', 0)} "
                f"| {agg.get('escalation_reasoning_tokens', 0)} "
                f"| {agg.get('escalation_mean_latency_ms', 0):.0f} |"
                if tiered
                else ""
            )
        )
    failing = [t for t in result.trials if t.failures and t is result.best]
    for t in failing:
        lines += ["", f"## Cases the best combination did not pass (trial {t.n})", ""]
        lines += [
            f"- {f['case']}: "
            + (f["error"] or "; ".join([*f["missing"], *f["extra"], *f["violations"]]))[
                :200
            ]
            for f in t.failures
        ]
    risky = [t for t in result.trials if t.spec.removal_shifts()]
    if risky:
        lines += [
            "",
            "## Removal shifts",
            "",
            "These combinations raise the probability that decides a removal, a "
            "set-null or a delete (`gate`, `null_check`, `delete`): they make the "
            "engine remove more data. Check them before trusting the numbers.",
            "",
            *(f"- trial {t.n}: {', '.join(t.spec.removal_shifts())}" for t in risky),
        ]
    if result.not_run:
        lines += ["", "## Not run", ""] + [f"- {line}" for line in result.not_run]
    lines += [
        "",
        "A failed case is reported, never retried. Offline the models are scripted "
        "worlds, so an offline table checks the harness, not the providers.",
    ]
    return "\n".join(lines) + "\n"


def write_search(
    out: Path, result: SearchResult, header: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    """Write ``search.json`` (resumable), ``leaderboard.md`` and ``best.json``.

    Args:
        out: The results directory.
        result: The search so far.
        header: What identifies the run (suite, case ids, overrides...): a resume
            checks it matches.

    Returns:
        The three paths.
    """
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        **header,
        "stopped": result.stopped,
        "spent_usd": result.spent_usd,
        "best": result.best.n,
        "not_run": result.not_run,
        "trials": [t.as_dict() for t in result.trials],
    }
    js, md, best = out / "search.json", out / "leaderboard.md", out / "best.json"
    # an error message can carry the request URL, and with it the account id
    js.write_text(scrub(json.dumps(payload, indent=2, default=str)))
    md.write_text(scrub(leaderboard_markdown(result)))
    best.write_text(json.dumps(result.best.spec.model_dump(), indent=2, default=str))
    return js, md, best


def load_previous(out: Path) -> tuple[dict[str, Any], list[Trial]]:
    """Read the header and the trials of an earlier search in ``out``.

    Raises:
        ValueError: If ``out`` holds no ``search.json`` or it is not readable.
    """
    file = out / "search.json"
    try:
        data = json.loads(file.read_text())
        trials = [Trial.from_dict(t) for t in data["trials"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot resume from {out}: no readable search.json") from exc
    header = {k: v for k, v in data.items() if k not in ("trials",)}
    return header, trials
