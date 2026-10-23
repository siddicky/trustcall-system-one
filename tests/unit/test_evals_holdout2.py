"""The second held-out suite: clean, independent of the others, and passing offline.

``holdout`` was used for diagnosis, so ``holdout2`` is the clean held-out set. These
tests pin its size, its coverage and its independence of the dev suite and of the
first holdout (case ids, schema names and text), that it is registered for ``run``
and ``combo`` and refused for ``search``, ``sweep`` and calibration.
"""

import asyncio
import importlib
import json
import re
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from evals import cli, suites
from evals.cases import CASES
from evals.holdout2_cases import HOLDOUT2_CASES, SUITE_NAME, holdout2_cases
from evals.holdout_cases import HOLDOUT_CASES
from evals.model import Case
from evals.runner import Settings, aggregate, run_suite
from trustcall_system_one import Many, One


def test_holdout2_is_named_and_sized() -> None:
    assert SUITE_NAME == "holdout2"
    assert holdout2_cases() == HOLDOUT2_CASES
    assert len(HOLDOUT2_CASES) == 30


def _model_names(case: Case) -> set[str]:
    """The names of every model a case's schemas reach, nested ones included."""
    names: set[str] = set()
    stack: list[Any] = []
    for schema in case.schemas():
        stack.append(schema.schema if isinstance(schema, (Many, One)) else schema)
    while stack:
        found = stack.pop()
        if isinstance(found, type) and issubclass(found, BaseModel):
            if found.__name__ in names:
                continue
            names.add(found.__name__)
            stack.extend(f.annotation for f in found.model_fields.values())
        else:
            stack.extend(typing.get_args(found))
    return names


def _sixgrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[i : i + 6]) for i in range(len(words) - 5)}


@pytest.mark.parametrize(
    ("label", "others"), [("dev", CASES), ("holdout", HOLDOUT_CASES)]
)
def test_holdout2_is_disjoint_from_the_other_suites(
    label: str, others: list[Case]
) -> None:
    ids = [c.id for c in HOLDOUT2_CASES]
    assert len(ids) == len(set(ids))
    assert not set(ids) & {c.id for c in others}, f"case ids shared with {label}"
    mine = {n for c in HOLDOUT2_CASES for n in _model_names(c)}
    theirs = {n for c in others for n in _model_names(c)}
    assert not mine & theirs, f"schema names shared with {label}: {mine & theirs}"
    # no run of six words of a holdout2 text occurs in a text of the other suite
    other_grams: set[tuple[str, ...]] = set()
    for case in others:
        other_grams |= _sixgrams(case.text)
    for case in HOLDOUT2_CASES:
        shared = _sixgrams(case.text) & other_grams
        assert not shared, f"{case.id} shares text with {label}: {sorted(shared)[:3]}"
    assert not {c.text for c in HOLDOUT2_CASES} & {c.text for c in others}


def test_every_holdout2_case_has_gold_a_world_and_tags() -> None:
    for case in HOLDOUT2_CASES:
        assert case.world is not None, case.id
        assert case.tags, case.id
        assert case.note, case.id
        assert (
            case.expect_ops
            or case.expect_values
            or case.must_not
            or case.expect_skipped
        ), case.id


def test_holdout2_covers_the_requested_ground() -> None:
    tags = {t for c in HOLDOUT2_CASES for t in c.tags}
    wanted = {
        "scalar", "decimal", "date", "relative-date", "month-year", "time", "duration",
        "bool", "literal", "enum", "optional", "nested", "table", "many", "coref",
        "negative", "similar-names", "update", "correction", "notes", "rename",
        "relative", "delete", "no-longer", "conflict", "bleed", "dotted-identity",
        "email", "chat", "ocr", "abbreviations", "placeholder",
    }  # fmt: skip
    assert wanted <= tags, wanted - tags
    domains = {
        "insurance", "school", "fleet", "supplier", "library", "assets", "speakers",
        "soil", "charter", "meters",
    }  # fmt: skip
    assert domains <= tags
    for domain in domains:  # three cases each
        assert sum(domain in c.tags for c in HOLDOUT2_CASES) == 3, domain
    # a dotted identity is the identity of at least two cases
    dotted = [c for c in HOLDOUT2_CASES if "dotted-identity" in c.tags]
    assert len(dotted) >= 2
    for case in dotted:
        marker = next(s for s in case.schemas() if isinstance(s, Many))
        assert marker.identity is not None and "." in str(marker.identity)
    # an update/delete conflict must say so, or doing nothing would pass it
    conflicts = [c for c in HOLDOUT2_CASES if "conflict" in c.tags]
    assert conflicts
    assert all(c.expect_skipped for c in conflicts)


def test_every_holdout2_case_passes_on_its_scripted_world() -> None:
    suite = asyncio.run(run_suite(list(HOLDOUT2_CASES), Settings()))
    failing = [
        (r.case, r.error, r.missing, r.extra, r.violations)
        for r in suite.results
        if not r.passed
    ]
    assert failing == []
    agg = aggregate(suite.results)
    assert agg["cases"] == agg["passed"] == len(HOLDOUT2_CASES)
    assert agg["op_precision"] == 1.0 and agg["op_recall"] == 1.0
    assert agg["value_accuracy"] == 1.0 and agg["valid_rate"] == 1.0
    assert agg["bleed"] == 0 and agg["wrong_deletes"] == 0
    assert suite.mode == "offline" and suite.skipped == []


def test_the_bleed_traps_catch_a_run_without_verification() -> None:
    traps = [c for c in HOLDOUT2_CASES if "bleed" in c.tags and c.world is not None]
    assert len(traps) >= 6
    guarded = asyncio.run(run_suite(traps, Settings()))
    assert all(r.passed for r in guarded.results)
    unguarded = asyncio.run(
        run_suite(
            traps,
            Settings(
                overrides={
                    "verify.attribution": False,
                    "verify.input_only_crosscheck": False,
                }
            ),
        )
    )
    # every fact offered for the wrong entity lands on it and its trap notices
    hit = [
        r.case for r in unguarded.results if any(c == "bleed" for c, _ in r.violations)
    ]
    assert sorted(hit) == sorted(c.id for c in traps)


# --------------------------------------------------------------------------- #
# the registration: run and combo accept it, search, sweep and calibration refuse it
# --------------------------------------------------------------------------- #


def test_holdout2_is_registered_as_a_suite() -> None:
    assert SUITE_NAME in suites.SUITES and suites.SUITES[0] == "default"
    loaded = suites.load_suite("holdout2")
    assert [c.id for c in loaded] == [c.id for c in HOLDOUT2_CASES]
    assert [c.id for c in cli.select_cases(None, None, "holdout2")] == [
        c.id for c in HOLDOUT2_CASES
    ]
    # the first holdout is still its own suite
    assert [c.id for c in suites.load_suite("holdout")] == [c.id for c in HOLDOUT_CASES]
    picked = cli.select_cases(["claim_scalars"], None, "holdout2")
    assert [c.id for c in picked] == ["claim_scalars"]
    with pytest.raises(ValueError, match="unknown case"):
        cli.select_cases(["rename"], None, "holdout2")  # a dev case id


def test_a_missing_holdout2_file_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "evals.holdout2_cases":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(
        ValueError, match=r"evals/holdout2_cases\.py, which does not exist"
    ):
        suites.load_suite("holdout2")
    assert cli.main(["list", "--suite", "holdout2"]) == 2
    assert "holdout2_cases.py" in capsys.readouterr().err
    # the other suites do not depend on it
    assert len(suites.load_suite("holdout")) == len(HOLDOUT_CASES)


def test_run_and_combo_take_holdout2(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert cli.main(["run", "--suite", "holdout2", "--quiet", "--out", str(out)]) == 0
    data = json.loads((out / "results.json").read_text())
    assert data["suite"] == "holdout2"
    assert data["aggregate"]["passed"] == data["aggregate"]["cases"] == 30
    # a tag and a case selection work inside the suite
    only = tmp_path / "only"
    tagged = ["run", "--suite", "holdout2", "--tag", "dotted-identity", "--quiet"]
    code = cli.main([*tagged, "--out", str(only)])
    assert code == 0
    got = json.loads((only / "results.json").read_text())
    assert got["aggregate"]["cases"] == 2

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"name": "mix", "extraction": "cloudflare"}))
    combo = tmp_path / "combo"
    given = ["combo", "--spec", str(spec), "--suite", "holdout2", "--quiet"]
    code = cli.main([*given, "--out", str(combo)])
    assert code == 0
    data = json.loads((combo / "results.json").read_text())
    assert data["suite"] == "holdout2"
    assert data["aggregate"]["passed"] == data["aggregate"]["cases"] == 30


def test_holdout2_results_never_land_among_the_dev_ones() -> None:
    import argparse

    where = cli._out_dir(argparse.Namespace(out=None, suite="holdout2"), "default")
    assert where.name.endswith("-holdout2-default")


def test_search_sweep_and_calibration_refuse_holdout2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["search", "--suite", "holdout2", "--quiet"]) == 2
    assert "dev suite only" in capsys.readouterr().err
    assert cli.main(["search", "--calibrate", "--suite", "holdout2", "--quiet"]) == 2
    assert "dev suite only" in capsys.readouterr().err
    code = cli.main(["search", "--calibrate-only", "--suite", "holdout2", "--quiet"])
    assert code == 2
    assert "dev suite only" in capsys.readouterr().err
    code = cli.main(
        ["sweep", "--suite", "holdout2", "--param", "gates.add_threshold=0.5,0.6"]
    )
    assert code == 2
    assert "dev set" in capsys.readouterr().err
