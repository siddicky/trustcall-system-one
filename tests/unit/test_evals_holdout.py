"""The held-out eval suite: independent of the dev suite, and passing offline."""

import asyncio

from evals.cases import CASES
from evals.holdout_cases import HOLDOUT_CASES, SUITE_NAME, holdout_cases
from evals.model import Case
from evals.runner import Settings, aggregate, run_suite
from trustcall_system_one import Many, One


def test_the_holdout_is_named_and_sized() -> None:
    assert SUITE_NAME == "holdout"
    assert holdout_cases() == HOLDOUT_CASES
    assert len(HOLDOUT_CASES) in range(28, 33)


def test_the_holdout_is_independent_of_the_dev_suite() -> None:
    dev_ids = {c.id for c in CASES}
    dev_texts = {c.text for c in CASES}
    ids = [c.id for c in HOLDOUT_CASES]
    assert len(ids) == len(set(ids))
    assert not set(ids) & dev_ids
    assert not {c.text for c in HOLDOUT_CASES} & dev_texts
    dev_schemas = {n for c in CASES for n in _models(c)}
    assert not {n for c in HOLDOUT_CASES for n in _models(c)} & dev_schemas


def _models(case: Case) -> list[str]:
    """The names of a case's top-level schemas."""
    names: list[str] = []
    for schema in case.schemas():
        model = schema.schema if isinstance(schema, (Many, One)) else schema
        names.append(str(getattr(model, "__name__", model)))
    return names


def test_every_holdout_case_has_gold_a_world_and_tags() -> None:
    for case in HOLDOUT_CASES:
        assert case.world is not None, case.id
        assert case.tags, case.id
        assert case.note, case.id
        assert case.expect_ops or case.expect_values or case.must_not, case.id


def test_the_holdout_covers_the_requested_ground() -> None:
    tags = {t for c in HOLDOUT_CASES for t in c.tags}
    wanted = {
        "scalar", "decimal", "date", "relative-date", "time", "duration", "bool",
        "literal", "enum", "optional", "nested", "table", "many", "coref",
        "negative", "similar-names", "update", "correction", "notes", "relative",
        "delete", "conflict", "bleed", "dotted-identity", "email", "transcript",
        "ocr", "abbreviations",
    }  # fmt: skip
    assert wanted <= tags, wanted - tags
    domains = {"logistics", "healthcare", "realestate", "aviation", "saas", "events"}
    assert domains <= tags
    assert tags & {"recipe", "nutrition"}


def test_every_holdout_case_passes_on_its_scripted_world() -> None:
    suite = asyncio.run(run_suite(list(HOLDOUT_CASES), Settings()))
    failing = [
        (r.case, r.error, r.missing, r.extra, r.violations)
        for r in suite.results
        if not r.passed
    ]
    assert failing == []
    agg = aggregate(suite.results)
    assert agg["cases"] == agg["passed"] == len(HOLDOUT_CASES)
    assert agg["op_precision"] == 1.0 and agg["op_recall"] == 1.0
    assert agg["value_accuracy"] == 1.0 and agg["valid_rate"] == 1.0
    assert agg["bleed"] == 0 and agg["wrong_deletes"] == 0
    assert suite.mode == "offline" and suite.skipped == []


def test_the_bleed_traps_catch_a_run_without_verification() -> None:
    traps = [c for c in HOLDOUT_CASES if "bleed" in c.tags and c.world is not None]
    assert len(traps) >= 8
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
    # the facts offered for the wrong entity land on it and the traps notice
    assert aggregate(unguarded.results)["bleed"] >= 1
