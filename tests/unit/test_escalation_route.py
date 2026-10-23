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
from trustcall_system_one._config import ESCALATION_MODES, ROUTED_KINDS
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
    assert EscalationConfig().mode == "on_failure"
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
    assert spec.escalate.config().mode == "route"
    assert spec.escalate.describe().endswith("route")
    assert "route" not in EscalateSpec().describe()
    assert EscalateSpec().config().mode == "on_failure"
    with pytest.raises(ValueError, match=r"escalate\.mode"):
        EscalateSpec(mode="sometimes")


def test_the_environment_names_the_mode() -> None:
    from trustcall_system_one import escalation_from_env

    config = escalation_from_env(
        {"TRUSTCALL_S1_ESCALATE": "openai", "TRUSTCALL_S1_ESCALATE_MODE": "route"}
    )
    assert config is not None and config.mode == "route"
    plain = escalation_from_env({"TRUSTCALL_S1_ESCALATE": "openai"})
    assert plain is not None and plain.mode == "on_failure"
    with pytest.raises(ValueError, match="mode"):
        escalation_from_env(
            {"TRUSTCALL_S1_ESCALATE": "openai", "TRUSTCALL_S1_ESCALATE_MODE": "x"}
        )


def test_the_cli_takes_the_mode() -> None:
    from evals.cli import _parser as build_parser

    args = build_parser().parse_args(
        ["run", "--escalate", "openai", "--escalate-mode", "route"]
    )
    assert args.escalate_mode == "route"
    assert build_parser().parse_args(["run"]).escalate_mode is None


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
