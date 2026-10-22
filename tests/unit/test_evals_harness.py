"""The evals harness, run offline against the scripted worlds of its own dataset."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from evals import cli
from evals.cases import CASES
from evals.configs import (
    PRESETS,
    REQUEST_CHANGING,
    REQUEST_LIGHT,
    build_options,
    classify,
    parse_overrides,
    parse_value,
)
from evals.model import ANY, Case, CaseResult, MustNot, Op, Val, normalise
from evals.runner import Settings, _run_case, aggregate, run_suite
from evals.scoring import actual_ops, score
from trustcall_system_one import (
    ExtractionResult,
    Limits,
    PatchOp,
    ResultRecord,
    VerifyConfig,
)
from trustcall_system_one._types import EntityInfo
from trustcall_system_one.testing import FakeTransport

# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #

REQUIRED = {
    "readme_people",  # the trustcall README scenario
    "colleague_bleed",  # Tom must not get Priya's city
    "computed_total",  # a computed total that must be accepted
    "other_invoice_total",  # a total that belongs to another invoice
    "rename",  # Sarah Johnson -> Sarah Miller
    "short_form",  # "Emma" for Emma Thompson
    "order_line_items",  # the order line items
    "invoice_contacts",  # a contacts list
    "notes_update_contact",  # open-ended notes
    "relative_edit",  # "two more"
    "delete_request",
    "change_not_delete",  # a change request on a record
    "no_entity",
    "pet_not_person",
}


def test_the_dataset_has_the_required_cases() -> None:
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))
    assert len(ids) in range(20, 31)
    assert set(ids) >= REQUIRED
    for case in CASES:
        assert case.world is not None, case.id
        assert case.tags, case.id
        assert case.expect_ops or case.expect_values or case.must_not, case.id
    notes = [c.id for c in CASES if "notes" in c.tags]
    assert len(notes) >= 4


def run(cases: list[Case], **kwargs: Any) -> Any:
    settings = Settings(**kwargs)
    return asyncio.run(run_suite(cases, settings))


def test_every_case_passes_on_its_scripted_world() -> None:
    suite = run(list(CASES))
    failing = [
        (r.case, r.error, r.missing, r.extra, r.violations)
        for r in suite.results
        if not r.passed
    ]
    assert failing == []
    agg = aggregate(suite.results)
    assert agg["cases"] == len(CASES) and agg["passed"] == len(CASES)
    assert agg["op_precision"] == 1.0 and agg["op_recall"] == 1.0
    assert agg["value_accuracy"] == 1.0 and agg["valid_rate"] == 1.0
    assert agg["bleed"] == 0 and agg["wrong_deletes"] == 0
    assert agg["requests"] > 100 and agg["input_tokens"] > 0
    assert suite.mode == "offline" and suite.skipped == []


def test_the_notes_strategies_can_be_compared_offline() -> None:
    cases = [c for c in CASES if "notes" in c.tags]
    outcome = {}
    for name in ("list", "sentences", "auto"):
        overrides = dict(PRESETS[name])
        suite = run(cases, overrides=overrides)
        outcome[name] = {r.case for r in suite.results if r.passed}
    everything = {c.id for c in cases}
    assert outcome["auto"] == everything  # Clairic first, sentences when it is unsure
    assert outcome["sentences"] == everything
    # the world's Clairic list is unsure about an open-ended field, as live: the
    # list-only strategy loses the notes
    assert outcome["list"] < everything
    assert "notes_update_contact" not in outcome["list"]
    assert "notes_insert_contact" not in outcome["list"]


def test_a_fault_the_guards_catch_is_not_a_violation_but_one_they_miss_is() -> None:
    cases = [c for c in CASES if c.id in ("colleague_bleed", "bleed_update")]
    guarded = run(cases)
    assert all(r.passed for r in guarded.results)
    unguarded = run(
        cases,
        overrides={"verify.attribution": False, "verify.input_only_crosscheck": False},
    )
    # without the attribution check the facts that were offered for the wrong
    # person land on it: the must_not checks count them as bleed
    agg = aggregate(unguarded.results)
    assert agg["bleed"] >= 1, [r.violations for r in unguarded.results]
    assert agg["passed"] < len(cases)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def record(
    kind: str, who: str | None, patches: list[PatchOp] | None = None, **kw: Any
) -> ResultRecord:
    entity = EntityInfo(key="e0", label=who or "x") if kind == "insert" else None
    return ResultRecord(
        schema_name="Thing",
        json_doc_id=None if kind == "insert" else who,
        kind=kind,  # type: ignore[arg-type]
        value=None,
        before={"city": "Austin"} if kind == "update" else None,
        after=kw.pop("after", {"city": "Berlin"} if kind != "delete" else None),
        patches=tuple(patches or ()),
        entity=entity,
        **kw,
    )


def case(**kw: Any) -> Case:
    return Case(id="c", text="t", schemas=list, **kw)


def test_precision_and_recall_count_matched_operations_once() -> None:
    result = ExtractionResult(
        records=(
            record(
                "update",
                "0",
                [
                    PatchOp("replace", "/city", "Berlin"),
                    PatchOp("add", "/notes/-", "She moved to Berlin."),
                ],
            ),
            record("insert", "Olivia Davis"),
        )
    )
    scored = score(
        case(
            expect_ops=[
                Op("replace", "/city", "berlin", who="0"),
                Op("add", "/notes/-", contains="moved to berlin", who="0"),
                Op("replace", "/phone", "1", who="0"),
                Op("insert", who="olivia"),
            ]
        ),
        result,
    )
    assert (scored.matched_expected, scored.n_expected) == (3, 4)
    assert (scored.matched_actual, scored.n_actual) == (3, 3)
    assert scored.precision == 1.0 and scored.recall == 0.75
    assert scored.missing == ['0: replace /phone "1"'] and not scored.passed
    # an expected operation is used once: two identical produced ones are not both right
    twice = ExtractionResult(
        records=(record("update", "0", [PatchOp("replace", "/city", "Berlin")] * 2),)
    )
    scored = score(case(expect_ops=[Op("replace", "/city", "Berlin")]), twice)
    assert (scored.matched_actual, scored.n_actual) == (1, 2)
    assert scored.extra == ['0: replace /city "Berlin"']


def test_an_operation_on_the_wrong_record_does_not_match() -> None:
    result = ExtractionResult(
        records=(record("update", "1", [PatchOp("replace", "/city", "Berlin")]),)
    )
    scored = score(case(expect_ops=[Op("replace", "/city", "Berlin", who="0")]), result)
    assert scored.matched_expected == 0 and scored.matched_actual == 0


def test_must_not_counts_bleed_and_wrong_deletes_apart() -> None:
    result = ExtractionResult(
        records=(
            record(
                "update",
                "1",
                [PatchOp("replace", "/city", "Lisbon")],
                after={"city": "Lisbon"},
            ),
            record("delete", "7"),
            record("insert", "Sunny"),
        )
    )
    scored = score(
        case(
            must_not=[
                MustNot.op("replace", "/city", who="1"),
                MustNot.value_at("1", "/city", contains="lisbon"),
                MustNot.delete("7"),
                MustNot.delete("8"),
                MustNot.insert("Sunny"),
                MustNot.op(path="/phone"),
            ]
        ),
        result,
    )
    assert sorted(c for c, _ in scored.violations) == [
        "bleed",
        "wrong_delete",
        "wrong_insert",
        "wrong_op",
    ]
    assert not scored.passed
    agg = aggregate([scored])
    assert (agg["bleed"], agg["wrong_deletes"], agg["violations"]) == (1, 1, 4)


def test_path_prefixes_match_whole_segments() -> None:
    result = ExtractionResult(
        records=(record("update", "o", [PatchOp("replace", "/items/10/quantity", 1)]),)
    )
    assert not score(case(must_not=[MustNot.op(path="/items/1")]), result).violations
    assert score(case(must_not=[MustNot.op(path="/items/10")]), result).violations


def test_values_are_read_from_the_final_documents() -> None:
    doc = {
        "name": "Olivia",
        "notes": ["Graphic designer.", "Likes art"],
        "age": 27,
        "x": None,
    }
    result = ExtractionResult(records=(record("insert", "Olivia Davis", after=doc),))
    scored = score(
        case(
            expect_values=[
                Val("olivia", "/name", "olivia"),
                Val("Olivia", "/notes", contains="graphic designer"),
                Val("Olivia", "/age", 27.0),
                Val("Olivia", "/x", None),
                Val("Olivia", "/missing", None),
                Val("Olivia", "/age", 28),
                Val("Nobody", "/name", "x"),
            ]
        ),
        result,
    )
    assert (scored.values_correct, scored.n_values) == (5, 7)


def test_valid_rate_and_totals() -> None:
    result = ExtractionResult(
        records=(
            record("update", "0"),
            record("insert", "A", valid=False),
            record("delete", "3"),
        )
    )
    scored = score(case(), result)
    assert (scored.valid, scored.n_records) == (1, 2)
    assert aggregate([scored])["valid_rate"] == 0.5
    assert [op.kind for op in actual_ops(result)] == ["insert", "delete"]


def test_normalise_and_op_labels() -> None:
    assert normalise(" Berlin. ") == "berlin"
    assert normalise("1,080.00") == normalise(1080) == 1080.0
    assert normalise(True) is True and normalise(None) is None
    assert normalise({"b": " X ", "a": [1]}) == {"a": [1.0], "b": "x"}
    assert Op("replace", "/a", "x", who="0").label() == '0: replace /a "x"'
    assert Op("add", "/n/-", contains="hi").label() == "add /n/- ~'hi'"
    assert Op("insert").value is ANY


def test_a_case_result_reports_an_error_as_a_failure() -> None:
    failed = CaseResult(case="x", error="boom")
    assert not failed.passed and failed.precision == 1.0 and failed.recall == 1.0
    assert failed.as_dict()["passed"] is False


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #


def test_values_are_read_from_the_command_line() -> None:
    assert parse_value("0.75") == 0.75
    assert parse_value("12") == 12
    assert parse_value("true") is True
    assert parse_value("None") is None
    assert parse_value("sentences") == "sentences"
    assert parse_value("0.2:0.8") == (0.2, 0.8)
    assert parse_overrides(["a.b=1", "c=false"]) == {"a.b": 1, "c": False}
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["nope"])


def test_overrides_build_config_objects() -> None:
    options = build_options(
        {
            "verify.attribution_keep": 0.8,
            "limits.list_strategy": "sentences",
            "enable_inserts": True,
        },
        {"enable_deletes": True},
    )
    assert options["verify"] == VerifyConfig(attribution_keep=0.8)
    assert options["limits"] == Limits(list_strategy="sentences")
    assert options["enable_inserts"] is True and options["enable_deletes"] is True
    # a case's own config object is the starting point
    start = {"verify": VerifyConfig(attribution_drop=0.3)}
    merged = build_options({"verify.attribution_keep": 0.8}, start)
    assert merged["verify"] == VerifyConfig(attribution_keep=0.8, attribution_drop=0.3)
    for bad, text in [
        ({"nope.x": 1}, "unknown group"),
        ({"verify.nope": 1}, "no field"),
        ({"nope": 1}, "unknown option"),
    ]:
        with pytest.raises(ValueError, match=text):
            build_options(bad)
    with pytest.raises(ValueError, match="attribution_keep"):
        build_options({"verify.attribution_keep": 7})


def test_every_preset_builds_and_the_knob_tables_name_real_fields() -> None:
    for overrides in PRESETS.values():
        build_options(overrides)
    for knob in (*REQUEST_LIGHT, *REQUEST_CHANGING):
        group, _, name = knob.partition(".")
        build_options(
            {
                knob: getattr(
                    __import__("evals.configs", fromlist=["GROUPS"]).GROUPS[group](),
                    name,
                )
            }
        )
    assert classify("verify.attribution_keep") == "light"
    assert classify("limits.list_strategy") == "changes requests"
    assert classify("verify.nothing") == "unknown"
    assert not set(REQUEST_LIGHT) & set(REQUEST_CHANGING)


# --------------------------------------------------------------------------- #
# Recording, replay and sweeps
# --------------------------------------------------------------------------- #


def pick(*ids: str) -> list[Case]:
    return [c for c in CASES if c.id in ids]


def test_a_recorded_run_replays_with_the_same_scores_and_no_live_request(
    tmp_path: Path,
) -> None:
    cases = pick("notes_update_contact", "colleague_bleed", "order_line_items")
    recorded = run(cases, record=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "colleague_bleed.jsonl",
        "notes_update_contact.jsonl",
        "order_line_items.jsonl",
    ]
    replayed = run(cases, replay=tmp_path)
    assert replayed.mode == "replay"
    assert [r.passed for r in replayed.results] == [r.passed for r in recorded.results]
    assert (
        aggregate(replayed.results)["requests"]
        == aggregate(recorded.results)["requests"]
    )
    assert all(r.error is None and r.forwarded == 0 for r in replayed.results)


def test_a_threshold_that_changes_no_request_replays_from_the_cassettes(
    tmp_path: Path,
) -> None:
    cases = pick("bleed_update", "other_invoice_total", "readme_people")
    run(cases, record=tmp_path)
    for keep in (0.7, 0.8, 0.9):
        suite = run(
            cases,
            replay=tmp_path,
            overrides={"verify.attribution_keep": keep, "gates.replace_threshold": 0.7},
        )
        assert all(r.error is None for r in suite.results), [
            r.error for r in suite.results
        ]
        assert all(r.passed for r in suite.results)


def test_a_setting_that_changes_the_requests_misses_the_cassette(
    tmp_path: Path,
) -> None:
    cases = pick("notes_update_contact")
    run(cases, record=tmp_path, overrides=PRESETS["list"])
    suite = run(cases, replay=tmp_path, overrides=PRESETS["sentences"])
    (result,) = suite.results
    assert result.error is not None and "CassetteMissError" in result.error
    assert not result.passed


def test_a_replay_with_a_live_fallback_sends_only_the_new_requests(
    tmp_path: Path,
) -> None:
    (notes,) = pick("notes_update_contact")
    run([notes], record=tmp_path, overrides=PRESETS["list"])  # no sentences requests
    assert notes.world is not None
    live = cast(FakeTransport, notes.world().transport())
    settings = Settings(live=True, replay=tmp_path, overrides=PRESETS["auto"])
    result = asyncio.run(_run_case(notes, settings, live))
    assert result.error is None and result.passed
    # the Clairic and attribution requests were replayed; the sentence requests
    # (not recorded under "list") were the only ones to go out
    assert result.forwarded >= 1
    assert {meta.kind for _, _, meta in live.calls} <= {
        "sentences",
        "list_kind",
        "attribution",
        "gate",
        "list_dedup",
        "crosscheck",
        "extract",
        "stated",
        "null_check",
        "reask",
        "adjudicate",
    }
    assert "sentences" in {meta.kind for _, _, meta in live.calls}
    assert len(live.calls) == result.forwarded


def test_live_without_credentials_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        asyncio.run(run_suite(pick("no_entity"), Settings(live=True)))


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def test_run_writes_the_json_and_markdown_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = cli.main(["run", "--cases", "rename,short_form", "--out", str(tmp_path)])
    assert status == 0
    shown = capsys.readouterr().out
    assert "| default |" in shown and "rename" in shown
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["config"] == "default" and data["mode"] == "offline"
    assert data["aggregate"]["cases"] == 2
    assert {c["case"] for c in data["cases"]} == {"rename", "short_form"}
    markdown = (tmp_path / "summary.md").read_text()
    assert "op precision" in markdown and "| rename | ok |" in markdown


def test_cases_and_tags_are_selected(tmp_path: Path) -> None:
    assert [c.id for c in cli.select_cases(["rename", "no_entity"])] == [
        "rename",
        "no_entity",
    ]
    assert {c.id for c in cli.select_cases(tags=["bleed"])} >= {
        "colleague_bleed",
        "bleed_update",
    }
    assert all("notes" in c.tags for c in cli.select_cases(tags=["notes"]))
    with pytest.raises(ValueError, match="unknown case"):
        cli.select_cases(["nope"])
    with pytest.raises(ValueError, match="no case selected"):
        cli.select_cases(tags=["nonexistent"])


def test_a_sweep_runs_the_grid_and_tabulates_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = cli.main(
        [
            "sweep",
            "--tag",
            "bleed",
            "--param",
            "verify.attribution_keep=0.7,0.8",
            "--param",
            "limits.list_strategy=list,auto",
            "--out",
            str(tmp_path),
        ]
    )
    assert status == 0
    captured = capsys.readouterr()
    assert "limits.list_strategy changes the requests" in captured.err
    assert "note: verify.attribution_keep" not in captured.err
    rows = [line for line in captured.out.splitlines() if line.startswith("| 0.")]
    assert len(rows) == 4
    assert {line.split("|")[1].strip() for line in rows} == {
        "0.7 / list",
        "0.7 / auto",
        "0.8 / list",
        "0.8 / auto",
    }
    data = json.loads((tmp_path / "sweep.json").read_text())
    assert [d["params"] for d in data] == [
        {"verify.attribution_keep": 0.7, "limits.list_strategy": "list"},
        {"verify.attribution_keep": 0.7, "limits.list_strategy": "auto"},
        {"verify.attribution_keep": 0.8, "limits.list_strategy": "list"},
        {"verify.attribution_keep": 0.8, "limits.list_strategy": "auto"},
    ]
    assert (tmp_path / "sweep.md").read_text().startswith("# Sweep")


def test_bad_arguments_exit_with_status_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--cases", "nope"]) == 2
    assert cli.main(["run", "--config", "nope"]) == 2
    assert cli.main(["run", "--set", "verify.nope=1"]) == 2
    assert cli.main(["run", "--live", "--cases", "no_entity"]) == 2
    assert "error:" in capsys.readouterr().err


def test_list_names_every_case(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert all(c.id in out for c in CASES)


def test_invalid_grid_points_are_skipped_not_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = cli.main(
        [
            "sweep",
            "--cases",
            "no_entity",
            "--param",
            "verify.attribution_keep=0.5,0.9",
            "--param",
            "verify.attribution_drop=0.4,0.7",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert status == 0
    assert "skipping" in capsys.readouterr().err
    assert len(json.loads((tmp_path / "sweep.json").read_text())) == 3


def test_the_probabilities_behind_a_result_are_kept_for_threshold_work() -> None:
    suite = run([c for c in CASES if c.id == "colleague_bleed"])
    (result,) = suite.results
    keys = result.probabilities
    assert any(k.endswith("city attribution") for k in keys), keys
    refused = [
        p
        for k, p in keys.items()
        if k.startswith("Priya") and k.endswith("role attribution")
    ]
    assert refused and all(isinstance(p, float) and p < 0.4 for p in refused)


def test_the_live_tests_need_the_flag_and_the_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.live.conftest import live_enabled

    for name in (
        "TRUSTCALL_S1_LIVE",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    assert not live_enabled()
    monkeypatch.setenv("TRUSTCALL_S1_LIVE", "1")
    assert not live_enabled()  # no credentials
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "tok")
    assert live_enabled()
    monkeypatch.setenv("TRUSTCALL_S1_LIVE", "0")
    assert not live_enabled()


def test_make_test_runs_only_the_unit_tests_without_live_ones() -> None:
    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text()
    test_rule = makefile.split("\ntest:\n", 1)[1].split("\n\n", 1)[0]
    assert "--disable-socket" in test_rule
    assert '-m "not live and not eval"' in test_rule
    assert test_rule.strip().endswith("tests/unit")


@pytest.mark.parametrize(
    ("scenario", "case_id"),
    [
        ("people", "readme_people"),
        ("invoice_insert", "invoice_contacts"),
        ("contact_update", "notes_update_contact"),
        ("order_update", "order_line_items"),
    ],
)
def test_the_smoke_invariants_hold_on_the_scripted_worlds(
    scenario: str, case_id: str
) -> None:
    """The live smoke checks run here, so they are known to work before a live run."""
    from evals.smoke import SCENARIOS
    from trustcall_system_one import create_extractor

    chosen = next(s for s in SCENARIOS if s.name == scenario)
    case = next(c for c in CASES if c.id == case_id)
    assert case.world is not None
    assert (chosen.text, chosen.existing) == (case.text, case.existing)
    extractor = create_extractor(
        chosen.schemas(), transport=case.world().transport(), **chosen.options
    )
    result = extractor.invoke(chosen.text, existing=chosen.existing)
    chosen.check(result)


@pytest.mark.parametrize("gateway", [None, "my-gw"])
def test_an_account_id_in_an_error_never_reaches_the_results_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: str | None
) -> None:
    import httpx

    from evals.report import write_suite
    from evals.runner import SuiteResult, _run_case
    from trustcall_system_one import CloudflareTransport

    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    account, token = "0123456789abcdef0123456789abcdef", "secret-token-123456"

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    live = CloudflareTransport(
        account_id=account,
        api_token=token,
        gateway=gateway,
        client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
        max_retries=0,
    )
    (case,) = pick("readme_people")
    cassettes = tmp_path / "cassettes"
    settings = Settings(live=True, record=cassettes)
    cassettes.mkdir()
    result = asyncio.run(_run_case(case, settings, live))
    assert result.error is not None and "TransportError" in result.error
    assert account not in result.error and "<redacted>" in result.error
    suite = SuiteResult("default", {}, [result], [], "live")
    write_suite(suite, tmp_path / "out")
    for folder in (tmp_path / "out", cassettes):
        for path in folder.rglob("*"):
            if path.is_file():
                text = path.read_text()
                assert account not in text and token not in text, path


# --------------------------------------------------------------------------- #
# Allowed extras, notes recall, rejections, comparison of values
# --------------------------------------------------------------------------- #


def test_an_allowed_extra_neither_fails_the_case_nor_lowers_precision() -> None:
    result = ExtractionResult(
        records=(
            record(
                "update",
                "0",
                [
                    PatchOp("add", "/notes/-", "She got promoted."),
                    PatchOp("add", "/notes/-", "She took up rock climbing."),
                    PatchOp("add", "/notes/-", "Her cat is called Max."),
                ],
            ),
        )
    )
    wanted = [Op("add", "/notes/-", contains="promoted", who="0")]
    strict = score(case(expect_ops=wanted), result)
    assert strict.precision == pytest.approx(1 / 3) and not strict.passed
    lenient = score(
        case(
            expect_ops=wanted,
            allow_ops=[Op("add", "/notes/-", contains="climbing", who="0")],
        ),
        result,
    )
    assert lenient.allowed == 1 and (lenient.matched_actual, lenient.n_actual) == (1, 2)
    assert lenient.extra == ['0: add /notes/- "Her cat is called Max."']
    both = score(
        case(
            expect_ops=wanted,
            allow_ops=[
                Op("add", "/notes/-", contains="climbing", who="0"),
                Op("add", "/notes/-", contains="cat", who="0"),
            ],
        ),
        result,
    )
    assert both.precision == 1.0 and both.passed
    # an allowed operation on another record is still wrong
    other = score(
        case(
            expect_ops=wanted,
            allow_ops=[Op("add", "/notes/-", contains="climbing", who="1")],
        ),
        result,
    )
    assert other.extra and not other.passed


def test_notes_recall_is_counted_apart() -> None:
    result = ExtractionResult(
        records=(
            record("update", "0", [PatchOp("add", "/notes/-", "She got promoted.")]),
            record("insert", "Olivia Davis", after={"notes": ["Graphic designer."]}),
        )
    )
    scored = score(
        case(
            expect_ops=[
                Op("add", "/notes/-", contains="promoted", who="0"),
                Op("add", "/notes/-", contains="vegan", who="0"),
                Op("replace", "/city", "Berlin", who="0"),
            ],
            expect_values=[
                Val("Olivia", "/notes", contains="designer"),
                Val("Olivia", "/notes", contains="sketching"),
                Val("Olivia", "/city", None),
            ],
        ),
        result,
    )
    assert (scored.notes_found, scored.n_notes) == (2, 4)
    summary = aggregate([scored])
    assert summary["notes_recall"] == 0.5 and summary["op_recall"] == pytest.approx(
        1 / 3
    )
    assert aggregate([score(case(), result)])["notes_recall"] is None


def test_digit_strings_are_not_numbers_when_both_sides_are_strings() -> None:
    from evals.model import values_equal

    assert not values_equal("02139", "2139")
    assert values_equal("02139", "02139.")
    assert values_equal("2139", 2139) and values_equal(1080, "1,080.00")
    result = ExtractionResult(
        records=(record("update", "0", [PatchOp("replace", "/zip", "2139")]),)
    )
    assert (
        score(case(expect_ops=[Op("replace", "/zip", "02139", who="0")]), result).recall
        == 0
    )
    assert (
        score(case(expect_ops=[Op("replace", "/zip", 2139, who="0")]), result).recall
        == 1
    )


def test_a_value_must_hold_in_every_record_it_is_about() -> None:
    result = ExtractionResult(
        records=(
            record("insert", "Olivia Davis", after={"city": "Rome"}),
            record("insert", "Olivia Davis", after={"city": "Paris"}),
        )
    )
    assert (
        score(
            case(expect_values=[Val("Olivia", "/city", "Rome")]), result
        ).values_correct
        == 0
    )
    same = ExtractionResult(
        records=(
            record("insert", "Olivia Davis", after={"city": "Rome"}),
            record("insert", "Olivia Davis", after={"city": "Rome"}),
        )
    )
    assert (
        score(case(expect_values=[Val("Olivia", "/city", "Rome")]), same).values_correct
        == 1
    )


def test_an_expected_rejection_must_be_reported() -> None:
    from evals.model import Rejected
    from trustcall_system_one import RejectedOp

    rejected = RejectedOp("add", "/notes/-", "duplicate", "same fact", 0.9)
    result = ExtractionResult(records=(record("update", "c", rejected=(rejected,)),))
    assert score(
        case(expect_rejected=[Rejected("duplicate", "/notes/-", "c")]), result
    ).passed
    missing = score(case(expect_rejected=[Rejected("attribution")]), result)
    assert not missing.passed and missing.missing == ["rejected attribution"]
    # the shipped near-duplicate case asserts the rejection, not just the absence
    (near,) = pick("notes_near_duplicate")
    assert near.expect_rejected and near.expect_rejected[0].reason == "duplicate"
    assert next(r for r in run([near]).results).passed


def test_the_summary_has_a_notes_recall_column() -> None:
    suite = run(pick("readme_people", "notes_update_contact"))
    from evals.report import suite_markdown

    markdown = suite_markdown(suite)
    assert "notes recall" in markdown
    assert aggregate(suite.results)["notes_recall"] == 1.0
    readme = next(r for r in suite.results if r.case == "readme_people")
    assert readme.allowed >= 1  # the legitimate extra notes were not held against it
