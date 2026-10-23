"""``EscalationConfig(mode="route")``: three classes go to the tier first.

Relative dates, the rows of a nested list and the discovery of a dotted identity are
asked of the escalation model BEFORE the default setup answers (``on_failure`` asks only
where it failed). The same verification, ladder and caps apply, and a call that fails or
an answer that cannot be verified leaves the default path as it was. Every text is
synthetic; the scripted worlds and helpers are those of ``test_escalation``.
"""

import datetime as dt
from collections import Counter
from typing import Any

import pytest

from tests.unit.test_escalation import (
    GOOD,
    SHEET,
    THREE_DAYS,
    Aircraft,
    Engine,
    Plane,
    _Failing,
    all_rows,
    blind_discovery,
    degrade,
    engine_world,
    escalations,
    insert_aircraft,
    run_task,
    sheet_world,
    tier,
    update_aircraft,
)
from trustcall_system_one import (
    EscalationConfig,
    Many,
    TransportError,
    create_extractor,
)
from trustcall_system_one._config import (
    ESCALATION_KINDS,
    ESCALATION_MODE_PRESETS,
    ESCALATION_MODES,
    ROUTED_KINDS,
    describe_escalation_mode,
    parse_escalation_mode,
)
from trustcall_system_one.combos import ComboSpec, EscalateSpec
from trustcall_system_one.testing import FakeEscalation, FakeTransport, FakeWorld

DAY = dt.date(2026, 10, 20)


def route(fake: Any, **kwargs: Any) -> EscalationConfig:
    return tier(fake, mode="route", **kwargs)


def three_days(*_: Any) -> dict[str, Any]:
    return {"phrase": "three days ago", "iso_date": "2026-10-20", "time": None}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_mode_defaults_to_on_failure_and_is_validated() -> None:
    assert EscalationConfig().mode_for("list_rows") == "on_failure"
    assert ESCALATION_MODES == ("on_failure", "route")
    with pytest.raises(ValueError, match="mode must be one of"):
        EscalationConfig(mode="always")


def test_only_the_three_classes_are_routed() -> None:
    on = EscalationConfig(enabled=True, mode="route")
    assert {
        k for k in ("relative_dates", "list_rows", "dotted_discovery") if on.routes(k)
    } == set(ROUTED_KINDS)
    assert not on.routes("attribution_tiebreak")  # asked after the second look, always
    assert on.runs("attribution_tiebreak")
    # a kind that is switched off is not routed, and on_failure routes nothing
    assert not EscalationConfig(
        enabled=True, mode="route", kinds=frozenset({"list_rows"})
    ).routes("relative_dates")
    assert not EscalationConfig(enabled=True).routes("list_rows")
    assert not EscalationConfig(mode="route").routes("list_rows")  # not enabled


def test_a_combo_spec_names_the_mode() -> None:
    spec = ComboSpec.model_validate(
        {"name": "x", "escalate": {"provider": "openai", "mode": "route"}}
    )
    assert spec.escalate is not None and spec.escalate.mode == "route"
    assert spec.escalate.config().mode_for("list_rows") == "route"
    assert spec.escalate.describe().endswith("route")
    assert "route" not in EscalateSpec().describe()
    assert EscalateSpec().config().mode_for("list_rows") == "on_failure"
    with pytest.raises(ValueError, match=r"escalate\.mode"):
        EscalateSpec(mode="sometimes")


def test_the_environment_names_the_mode() -> None:
    from trustcall_system_one import escalation_from_env

    config = escalation_from_env(
        {"TRUSTCALL_S1_ESCALATE": "openai", "TRUSTCALL_S1_ESCALATE_MODE": "route"}
    )
    assert config is not None and config.mode_for("list_rows") == "route"
    plain = escalation_from_env({"TRUSTCALL_S1_ESCALATE": "openai"})
    assert plain is not None and plain.mode_for("list_rows") == "on_failure"
    with pytest.raises(ValueError, match="mode"):
        escalation_from_env(
            {"TRUSTCALL_S1_ESCALATE": "openai", "TRUSTCALL_S1_ESCALATE_MODE": "x"}
        )


def test_the_cli_takes_a_plain_mode() -> None:
    from evals.cli import _parser as build_parser

    args = build_parser().parse_args(
        ["run", "--escalate", "openai", "--escalate-mode", "route"]
    )
    assert args.escalate_mode == "route"
    assert build_parser().parse_args(["run"]).escalate_mode is None


# --------------------------------------------------------------------------- #
# The mode by kind
# --------------------------------------------------------------------------- #

RECOMMENDED = {
    "relative_dates": "route",
    "list_rows": "route",
    "dotted_discovery": "on_failure",
    "attribution_tiebreak": "on_failure",
}


def modes(config: EscalationConfig) -> dict[str, str]:
    return {kind: config.mode_for(kind) for kind in ESCALATION_KINDS}


def test_a_mapping_sets_the_mode_of_each_kind() -> None:
    config = EscalationConfig(
        enabled=True,
        mode={
            "relative_dates": "route",
            "list_rows": "route",
            "dotted_discovery": "on_failure",
            "attribution_tiebreak": "on_failure",
        },
    )
    assert modes(config) == RECOMMENDED
    assert config.routes("relative_dates") and config.routes("list_rows")
    assert not config.routes("dotted_discovery")
    assert not config.routes("attribution_tiebreak")
    # a kind the mapping omits is on_failure; the mapping is frozen and hashable
    only = EscalationConfig(mode={"dotted_discovery": "route"})
    assert modes(only) == {
        **dict.fromkeys(ESCALATION_KINDS, "on_failure"),
        **{"dotted_discovery": "route"},
    }
    assert hash(only) == hash(EscalationConfig(mode={"dotted_discovery": "route"}))
    with pytest.raises(TypeError):
        only.mode["list_rows"] = "route"  # type: ignore[index]


def test_a_plain_string_means_the_same_mode_for_every_kind() -> None:
    off = EscalationConfig(mode="on_failure")
    assert set(modes(off).values()) == {"on_failure"}
    route_all = EscalationConfig(mode="route")
    assert modes(route_all) == {
        **dict.fromkeys(ROUTED_KINDS, "route"),
        "attribution_tiebreak": "on_failure",  # never routed
    }
    # the string and the equal mapping are the same configuration
    assert route_all == EscalationConfig(mode=modes(route_all))


def test_the_recommended_preset_is_exactly_the_mapping() -> None:
    preset = EscalationConfig(mode="recommended")
    assert modes(preset) == RECOMMENDED
    assert dict(ESCALATION_MODE_PRESETS["recommended"]) == RECOMMENDED
    assert preset == EscalationConfig(mode=RECOMMENDED)
    assert describe_escalation_mode("recommended") == "recommended"
    assert describe_escalation_mode({"list_rows": "route"}) == "route(list_rows)"


def test_route_is_refused_for_the_attribution_tie_break() -> None:
    with pytest.raises(ValueError, match=r"attribution_tiebreak.*on_failure"):
        EscalationConfig(mode={"attribution_tiebreak": "route"})
    with pytest.raises(ValueError, match="attribution_tiebreak"):
        EscalateSpec(mode={"attribution_tiebreak": "route"})
    with pytest.raises(ValueError, match="attribution_tiebreak"):
        parse_escalation_mode("attribution_tiebreak=route")
    # on_failure for it is the only (and the default) value
    assert (
        EscalationConfig(mode={"attribution_tiebreak": "on_failure"}).enabled is False
    )


@pytest.mark.parametrize(
    "mode",
    [
        {"nope": "route"},
        {"list_rows": "sometimes"},
        {"list_rows": "recommended"},
        "recommended,route",
        42,
    ],
)
def test_a_bad_mode_is_refused(mode: Any) -> None:
    with pytest.raises(ValueError):
        EscalationConfig(mode=mode)


def test_a_combo_spec_takes_the_mode_as_a_string_or_a_mapping() -> None:
    by_name = ComboSpec.model_validate(
        {"name": "x", "escalate": {"provider": "openai", "mode": "recommended"}}
    )
    assert by_name.escalate is not None and by_name.escalate.mode == "recommended"
    assert modes(by_name.escalate.config()) == RECOMMENDED
    assert by_name.escalate.describe().endswith("recommended")
    by_map = ComboSpec.model_validate(
        {
            "name": "x",
            "escalate": {"mode": {"relative_dates": "route", "list_rows": "route"}},
        }
    )
    assert by_map.escalate is not None
    assert modes(by_map.escalate.config()) == RECOMMENDED
    # the same mapping as the preset is labelled as the preset
    assert by_map.escalate.describe().endswith("recommended")
    partial = EscalateSpec(mode={"list_rows": "route", "dotted_discovery": "route"})
    assert partial.describe().endswith("route(list_rows+dotted_discovery)")
    # the spec round-trips through JSON
    again = ComboSpec.model_validate_json(by_map.model_dump_json())
    assert again == by_map
    with pytest.raises(ValueError, match=r"escalate\.mode"):
        ComboSpec.model_validate({"name": "x", "escalate": {"mode": {"zzz": "route"}}})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("on_failure", "on_failure"),
        ("route", "route"),
        (" Recommended ", "recommended"),
        (
            "relative_dates=route,list_rows=route",
            {"relative_dates": "route", "list_rows": "route"},
        ),
        (
            " Relative_Dates = ROUTE , dotted_discovery=on_failure ",
            {"relative_dates": "route", "dotted_discovery": "on_failure"},
        ),
    ],
)
def test_the_text_syntax_is_parsed(text: str, expected: Any) -> None:
    assert parse_escalation_mode(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "always",
        "list_rows",
        "list_rows=",
        "=route",
        "list_rows=route,list_rows=route",
    ],
)
def test_a_malformed_text_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        parse_escalation_mode(text)


def test_the_cli_parses_every_form_of_the_mode() -> None:
    from evals.cli import _parser as build_parser

    def parse(*mode: str) -> Any:
        return build_parser().parse_args(
            [
                "run",
                "--escalate",
                "openai",
                *(["--escalate-mode", *mode] if mode else []),
            ]
        )

    assert parse().escalate_mode is None
    assert parse("recommended").escalate_mode == "recommended"
    assert parse("on_failure").escalate_mode == "on_failure"
    assert parse("relative_dates=route,list_rows=route").escalate_mode == {
        "relative_dates": "route",
        "list_rows": "route",
    }
    for bad in ("attribution_tiebreak=route", "sometimes", "list_rows=maybe"):
        with pytest.raises(SystemExit):
            parse(bad)


def test_the_environment_parses_every_form_of_the_mode() -> None:
    from trustcall_system_one import escalation_from_env

    def env(mode: str) -> EscalationConfig:
        config = escalation_from_env(
            {"TRUSTCALL_S1_ESCALATE": "openai", "TRUSTCALL_S1_ESCALATE_MODE": mode}
        )
        assert config is not None
        return config

    assert modes(env("recommended")) == RECOMMENDED
    assert modes(env("relative_dates=route, list_rows=route")) == RECOMMENDED
    assert set(modes(env("on_failure")).values()) == {"on_failure"}
    with pytest.raises(ValueError, match="TRUSTCALL_S1_ESCALATE_MODE"):
        env("attribution_tiebreak=route")
    with pytest.raises(ValueError, match="TRUSTCALL_S1_ESCALATE_MODE"):
        env("list_rows=maybe")
    # the mode alone never turns the tier on
    assert escalation_from_env({"TRUSTCALL_S1_ESCALATE_MODE": "recommended"}) is None


def test_the_eval_runner_carries_the_mode_into_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evals.runner import Settings, escalation_plan

    monkeypatch.delenv("TRUSTCALL_S1_ESCALATE", raising=False)
    monkeypatch.delenv("TRUSTCALL_S1_ESCALATE_MODE", raising=False)
    assert escalation_plan(Settings(), None) is None
    plan = escalation_plan(
        Settings(escalate="openai", escalate_mode="recommended"), None
    )
    assert plan is not None and plan.mode == "recommended"
    mapped = escalation_plan(
        Settings(escalate="openai", escalate_mode={"list_rows": "route"}), None
    )
    assert mapped is not None and mapped.mode == {"list_rows": "route"}
    default = escalation_plan(Settings(escalate="openai"), None)
    assert default is not None and default.mode == "on_failure"
    # --escalate openai with the mode in the environment
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE_MODE", "recommended")
    from_env = escalation_plan(Settings(escalate="openai"), None)
    assert from_env is not None and from_env.mode == "recommended"


async def test_recommended_asks_dates_and_rows_first_and_discovery_on_failure() -> None:
    # dates: asked although Clairic is confident
    dates = FakeEscalation(three_days)
    await run_task(
        THREE_DAYS, DAY, answer="2026-01-01", escalation=tier(dates, mode="recommended")
    )
    assert len(dates.calls) == 1
    # rows: asked first, Clairic does not discover them
    rows = FakeEscalation(all_rows)
    result = await insert_aircraft(
        engine_world().handler, tier(rows, mode="recommended")
    )
    assert len(rows.calls) == 1
    kinds = Counter(c.kind for c in result.calls if c.schema == "Engine")
    assert kinds["discover"] == 0 and kinds["count"] == 0
    # dotted discovery: Clairic works, so the tier is never asked
    discovery = FakeEscalation(GOOD)
    found = await run_sheet(sheet_world().handler, tier(discovery, mode="recommended"))
    assert not discovery.calls
    assert len(found.records) == 2
    assert Counter(c.kind for c in found.calls)["discover"] >= 1
    # ... and it is asked where Clairic finds nothing
    rescue = FakeEscalation(GOOD)
    rescued = await run_sheet(
        blind_discovery(sheet_world().handler), tier(rescue, mode="recommended")
    )
    assert len(rescue.calls) == 1 and len(rescued.records) == 2


async def test_a_mapping_routes_only_the_kinds_it_names() -> None:
    fake = FakeEscalation(all_rows)
    mode = {"relative_dates": "route"}  # rows stay on_failure
    result = await insert_aircraft(engine_world().handler, tier(fake, mode=mode))
    assert not fake.calls  # nothing failed, rows were not routed
    kinds = Counter(c.kind for c in result.calls if c.schema == "Engine")
    assert kinds["discover"] >= 1


# --------------------------------------------------------------------------- #
# relative_dates
# --------------------------------------------------------------------------- #


async def test_a_routed_date_is_asked_even_when_clairic_is_confident() -> None:
    fake = FakeEscalation(three_days)
    # Clairic confidently read a wrong day; on_failure never asks about it
    off = FakeEscalation(three_days)
    await run_task(THREE_DAYS, DAY, answer="2026-01-01", escalation=tier(off))
    assert not off.calls
    result = await run_task(
        THREE_DAYS, DAY, answer="2026-01-01", escalation=route(fake)
    )
    (record,) = result.records
    assert len(fake.calls) == 1
    assert record.valid and record.value.done_on == DAY  # type: ignore[union-attr]
    assert "escalated:relative_dates" in record.fields["done_on"].warnings
    # judged like any value: Clef was asked whether the text states it
    assert any(c.kind == "attribution" for c in result.calls)


async def test_a_routed_date_that_agrees_with_clairic_changes_nothing() -> None:
    fake = FakeEscalation(three_days)
    result = await run_task(
        THREE_DAYS, DAY, answer="2026-10-20", escalation=route(fake)
    )
    (record,) = result.records
    assert len(fake.calls) == 1 and record.valid
    assert "escalated:relative_dates" not in record.fields["done_on"].warnings
    assert any("routed, Clairic agrees" in w for w in result.warnings)


@pytest.mark.parametrize(
    "answer",
    [
        {
            "phrase": "a fortnight hence",
            "iso_date": "2026-11-06",
            "time": None,
        },  # not in the input
        {
            "phrase": "three days ago",
            "iso_date": "2026-01-01",
            "time": None,
        },  # not what it says
        {"phrase": None, "iso_date": None, "time": None},  # no date
    ],
)
async def test_an_unverifiable_routed_date_leaves_clairics_answer(
    answer: dict[str, Any],
) -> None:
    fake = FakeEscalation(lambda p, m: answer)
    result = await run_task(
        THREE_DAYS, DAY, answer="2026-10-20", escalation=route(fake)
    )
    (record,) = result.records
    assert len(fake.calls) == 1
    assert record.valid and record.value.done_on == DAY  # type: ignore[union-attr]
    assert "escalated:relative_dates" not in record.fields["done_on"].warnings


async def test_a_failed_routed_date_call_leaves_clairics_answer() -> None:
    failing = _Failing(TransportError("down"))
    result = await run_task(
        THREE_DAYS, DAY, answer="2026-10-20", escalation=route(failing)
    )
    (record,) = result.records
    assert record.valid and record.value.done_on == DAY  # type: ignore[union-attr]
    assert any(
        w.startswith("escalation_failed: relative_dates") for w in result.warnings
    )
    assert len(failing.calls) == 1  # never sent again


async def test_a_routed_date_still_needs_a_reference_date_and_a_phrase() -> None:
    fake = FakeEscalation(three_days)
    await run_task(
        THREE_DAYS, DAY, answer="2026-10-20", escalation=route(fake), reference=None
    )
    await run_task(
        "Task: Replace the pump. Completed on 2026-10-20.",
        DAY,
        answer="2026-10-20",
        escalation=route(fake),
    )
    assert not fake.calls


# --------------------------------------------------------------------------- #
# list_rows
# --------------------------------------------------------------------------- #


async def test_routed_rows_replace_clairics_discovery_and_values() -> None:
    fake = FakeEscalation(all_rows)
    result = await insert_aircraft(engine_world().handler, route(fake))
    (record,) = result.records
    assert [e["serial"] for e in record.after["engines"]] == [  # type: ignore[index]
        "SN-1001",
        "SN-1002",
        "SN-1003",
    ]
    assert len(fake.calls) == 1  # discovery and the fields share one call
    kinds = Counter(c.kind for c in result.calls if c.schema == "Engine")
    assert kinds["discover"] == 0 and kinds["count"] == 0  # Clairic was not asked
    assert any("escalation_routed: Engine" in w for w in result.warnings)
    assert Counter(c.kind for c in result.calls)["attribution"] >= 3  # judged by Clef


async def test_without_the_route_clairic_discovers_the_rows() -> None:
    fake = FakeEscalation(all_rows)
    result = await insert_aircraft(engine_world().handler, tier(fake))
    kinds = Counter(c.kind for c in result.calls if c.schema == "Engine")
    assert kinds["discover"] >= 1 and not fake.calls  # nothing failed, nothing asked


async def test_a_routed_row_value_replaces_a_wrong_clairic_value() -> None:
    # Clairic reads a serial for every row, and a wrong one
    handler = degrade(
        engine_world().handler, schema="Engine", field="serial", value="SN-0000"
    )
    fake = FakeEscalation(all_rows)
    result = await insert_aircraft(handler, route(fake))
    engines = result.records[0].after["engines"]  # type: ignore[index]
    assert [e["serial"] for e in engines] == ["SN-1001", "SN-1002", "SN-1003"]


async def test_rows_the_model_invents_fall_back_to_clairic_discovery() -> None:
    fake = FakeEscalation(
        lambda p, m: {
            "rows": [
                {"identity": "ENG-7", "fields": {"position": "ENG-7", "serial": "SN-7"}}
            ]
        }
    )
    result = await insert_aircraft(engine_world().handler, route(fake))
    assert [e["position"] for e in result.records[0].after["engines"]] == [  # type: ignore[index]
        "ENG-1",
        "ENG-2",
        "ENG-3",
    ]
    assert (
        Counter(c.kind for c in result.calls if c.schema == "Engine")["discover"] >= 1
    )
    assert "SN-7" not in str(result.records[0].after)


async def test_a_failed_routed_rows_call_falls_back_to_clairic() -> None:
    failing = _Failing(TransportError("down"))
    result = await insert_aircraft(engine_world().handler, route(failing))
    assert len(result.records[0].after["engines"]) == 3  # type: ignore[index]
    assert any(w.startswith("escalation_failed: list_rows") for w in result.warnings)
    assert len(failing.calls) == 1  # one call for the list, never sent again


async def test_routed_rows_keep_short_codes_that_the_code_screen_would_drop() -> None:
    text = (
        "Aircraft N512QX\n\n| Pos | Serial |\n|---|---|\n"
        "| A1 | SN-1001 |\n| B2 | SN-1002 |\n"
    )
    world = FakeWorld(text)
    world.entity(
        "the Aircraft", schema=Aircraft, mentions=[], fields={"registration": "N512QX"}
    )
    for pos, serial in (("A1", "SN-1001"), ("B2", "SN-1002")):
        world.entity(
            pos,
            schema=Engine,
            parent="the Aircraft",
            mentions=[pos],
            fields={"position": pos, "serial": serial},
        )
    fake = FakeEscalation(
        lambda p, m: {
            "rows": [
                {"identity": "A1", "fields": {"position": "A1", "serial": "SN-1001"}},
                {"identity": "B2", "fields": {"position": "B2", "serial": "SN-1002"}},
            ]
        }
    )
    result = await insert_aircraft(world.handler, route(fake), text=text)
    assert [e["position"] for e in result.records[0].after["engines"]] == ["A1", "B2"]  # type: ignore[index]


async def test_a_routed_update_never_removes_a_row() -> None:
    fake = FakeEscalation(lambda p, m: {"rows": []})
    result = await update_aircraft(
        route(fake), enable_item_removals=True, enable_deletes=True
    )
    assert not [p for r in result.records for p in r.patches if p.op == "remove"]
    assert not result.deleted


# --------------------------------------------------------------------------- #
# dotted_discovery
# --------------------------------------------------------------------------- #


async def run_sheet(handler: Any, escalation: EscalationConfig | None) -> Any:
    extractor = create_extractor(
        [Many(Plane, identity="identification.msn")],
        transport=FakeTransport(handler),
        escalation=escalation,
    )
    return await extractor.ainvoke(SHEET)


async def test_routed_discovery_asks_the_tier_before_clairic() -> None:
    fake = FakeEscalation(GOOD)
    result = await run_sheet(sheet_world().handler, route(fake))  # Clairic works too
    assert sorted(r.after["identification"]["msn"] for r in result.records) == [
        "12345",
        "8547",
    ]
    assert len(fake.calls) == 1
    kinds = Counter(c.kind for c in result.calls)
    assert kinds["discover"] == 0 and kinds["count"] == 0
    assert kinds["qualify"] >= 1 and kinds["extract"] >= 2  # the normal path went on
    (call,) = escalations(result)
    assert call.field == "dotted_discovery"


async def test_unverifiable_routed_entities_fall_back_to_clairic() -> None:
    fake = FakeEscalation(
        lambda p, m: {"entities": [{"label": "Embraer E190", "identity": "99999"}]}
    )
    result = await run_sheet(sheet_world().handler, route(fake))
    assert sorted(r.after["identification"]["msn"] for r in result.records) == [
        "12345",
        "8547",
    ]
    assert Counter(c.kind for c in result.calls)["discover"] >= 1


async def test_a_failed_routed_discovery_falls_back_and_on_failure_is_unchanged() -> (
    None
):
    failing = _Failing(TransportError("down"))
    result = await run_sheet(sheet_world().handler, route(failing))
    assert len(result.records) == 2
    # on_failure with blind Clairic still rescues, as before
    rescued = await run_sheet(
        blind_discovery(sheet_world().handler), tier(FakeEscalation(GOOD))
    )
    assert len(rescued.records) == 2


async def test_the_tie_break_still_runs_in_route_mode() -> None:
    # a route-mode config keeps every kind: the tie-break is asked after the second look
    config = route(FakeEscalation(GOOD))
    assert config.runs("attribution_tiebreak") and not config.routes(
        "attribution_tiebreak"
    )
