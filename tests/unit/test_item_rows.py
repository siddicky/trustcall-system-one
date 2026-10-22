"""Lists of items that have no text identity (rows of a spec-sheet table).

``list[EngineInfo]`` where an engine has a position, a type and serial numbers but
no name: discovery used to propose every word of the table as an entity, and each
one was clustered and qualified (live: 40 to 70 junk entities, 134 to 208 requests
for a one-aircraft sheet). The items are now asked for by what an item is, the
mentions are screened in code and by one cheap question each before coreference,
and weakly qualified extras are cut to the count.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

import enum
from typing import Any, Literal, Optional

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import CallMeta, One, create_extractor
from trustcall_system_one._cardinality import Many
from trustcall_system_one._collection import (
    Scope,
    _plausible_item,
    _rank,
    _screens,
)
from trustcall_system_one._compile import compile_schema
from trustcall_system_one.testing import FakeTransport, FakeWorld


class EngineInfo(BaseModel):
    """One engine of the aircraft, a row of the engine table."""

    position: Literal["1", "2"]
    engine_type: Optional[str] = None
    esn: Optional[str] = Field(None, description="engine serial number")
    tsn_hours: Optional[int] = None


class GearPosition(str, enum.Enum):
    NOSE = "Nose"
    LEFT = "Left main"
    RIGHT = "Right main"


class LandingGear(BaseModel):
    position: GearPosition
    cycles: Optional[int] = None


class Aircraft(BaseModel):
    """An aircraft."""

    msn: Optional[str] = None
    engines: list[EngineInfo] = Field(default_factory=list, description="engines")
    landing_gear: list[LandingGear] = Field(default_factory=list)


COMPONENTS = ["LPC", "HPC", "HPT", "LPT", "FAN"]

SHEET = """\
| Aircraft Model | Airbus A320-232 |
| --- | --- |
| MSN | 8547 |
| Engines | 2 x V2527-A5 |

| Engine | Position | Type | ESN | TSN | CSN |
| --- | --- | --- | --- | --- | --- |
| Engine 1 | 1 | V2527-A5 | V12345 | 18542 | 9876 |
| Engine 2 | 2 | V2527-A5 | V12346 | 17000 | 9000 |

| Landing gear | Position | Cycles |
| --- | --- | --- |
| NLG | Nose | 5000 |
| LH MLG | Left main | 5100 |
| RH MLG | Right main | 5200 |

| No | Component | Part number | TSN | CSN |
| --- | --- | --- | --- | --- |
""" + "\n".join(
    f"| {n} | {COMPONENTS[n % len(COMPONENTS)]} {n} DISK | 3A{2400 + n * 7} "
    f"| {30000 + n * 113} | {10000 + n * 37} |"
    for n in range(1, 41)
)


JUNK = [
    "Model", "Position", "Type", "ESN", "TSN", "CSN", "Engines", "Engine", "Cycles",
    "Landing gear", "x", "1", "2", "5000", "9000", "Nose", "V2527-A5", "Aircraft",
    "A320-232", "MSN",
]  # fmt: skip


TABLE_WORDS = [
    *(f"{COMPONENTS[n % len(COMPONENTS)]} {n} DISK" for n in range(1, 41)),
    *(f"3A{2400 + n * 7}" for n in range(1, 41)),
    *(str(30000 + n * 113) for n in range(1, 41)),
]
"""The cells of the component table: the junk a model lists for an engine."""


def world() -> FakeWorld:
    sheet = FakeWorld(SHEET)
    sheet.entity(
        "Aircraft",
        schema=Aircraft,
        mentions=["Airbus A320-232"],
        fields={"msn": "8547"},
    )
    for position, esn, tsn in (("1", "V12345", 18542), ("2", "V12346", 17000)):
        sheet.entity(
            f"Engine {position}",
            schema=EngineInfo,
            parent="Aircraft",
            mentions=[f"Engine {position}"],
            fields={
                "position": position,
                "engine_type": "V2527-A5",
                "esn": esn,
                "tsn_hours": tsn,
            },
        )
    for word in [*JUNK, *TABLE_WORDS]:  # every word is a different non-entity
        sheet.non_entity(word, schema=EngineInfo, parent="Aircraft", mentions=[word])
    for label, position, cycles in (
        ("NLG", "Nose", 5000),
        ("LH MLG", "Left main", 5100),
        ("RH MLG", "Right main", 5200),
    ):
        sheet.entity(
            label,
            schema=LandingGear,
            parent="Aircraft",
            mentions=[label],
            fields={"position": position, "cycles": cycles},
        )
    for word in [*JUNK, *TABLE_WORDS]:
        sheet.non_entity(word, schema=LandingGear, parent="Aircraft", mentions=[word])
    return sheet


def test_the_code_screen_drops_what_no_row_is_called() -> None:
    plan = compile_schema(EngineInfo)
    scope = Scope(plan=plan, source="", cardinality=Many(EngineInfo))
    screen = _screens(scope)
    assert screen.numbers is False  # no required number: a bare figure is no engine
    for junk in ("5", "A", "MA", "**", "TSN:**<br", "Engine Type:**", "ESN", "42"):
        assert not _plausible_item(junk, screen), junk
    for item in (
        "Engine 1",
        "NLG",
        "LH MLG",
        "6Y",
        "CFM56-7B26 engines",
        "APU",
        "889101",  # a serial number
        "1",  # an option of `position`
    ):
        assert _plausible_item(item, screen), item


def test_the_options_of_a_choice_field_name_a_row() -> None:
    scope = Scope(
        plan=compile_schema(LandingGear), source="", cardinality=Many(LandingGear)
    )
    screen = _screens(scope)
    assert {"nose", "left main", "right main"} <= screen.options
    assert _plausible_item("Nose", screen)
    assert _rank("L", screen) == 2 and _rank("LH", screen) == 2  # a code: second chance


def test_a_required_number_makes_a_bare_figure_a_candidate() -> None:
    class Reading(BaseModel):
        """A reading."""

        value: float

    scope = Scope(plan=compile_schema(Reading), source="", cardinality=Many(Reading))
    screen = _screens(scope)
    assert screen.numbers is True
    assert _plausible_item("21.5", screen)
    assert not _plausible_item("**", screen)


async def test_rows_without_a_name_are_found_without_the_junk() -> None:
    sheet = world()
    transport = sheet.transport()
    extractor = create_extractor([One(Aircraft)], transport=transport)
    result = await extractor.ainvoke(SHEET)
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    plane = record.value
    assert isinstance(plane, Aircraft)
    assert [e.position for e in plane.engines] == ["1", "2"]
    assert [e.esn for e in plane.engines] == ["V12345", "V12346"]
    assert [g.position for g in plane.landing_gear] == [
        GearPosition.NOSE,
        GearPosition.LEFT,
        GearPosition.RIGHT,
    ]
    assert any("junk_mentions_dropped" in w for w in result.warnings)

    # nothing but the rows became an entity, and the junk cost little: the same
    # sheet used to take 61 requests (29 of them coreference) and to report 139
    # table cells as skipped entities
    by_kind: dict[str, int] = {}
    asked = 0
    for _, body, meta in transport.calls:
        by_kind[meta.kind] = by_kind.get(meta.kind, 0) + 1
        if meta.kind == "qualify":
            asked += len(body["questions"])
    assert by_kind.get("coref", 0) <= 2
    assert asked <= 150  # one question per screened mention, 64 to a request
    assert by_kind.get("qualify", 0) <= 4
    assert len(transport.calls) <= 40
    assert any("mentions_screened_out" in w for w in result.warnings)
    junk = [s for s in result.skipped if s.schema_name in ("EngineInfo", "LandingGear")]
    assert len(junk) <= 5


async def test_the_discovery_request_says_what_an_item_is() -> None:
    sheet = world()
    transport = sheet.transport()
    await create_extractor([One(Aircraft)], transport=transport).ainvoke(SHEET)
    asks = {
        meta.schema: body["fields"]["mentions"]["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover" and meta.schema in ("EngineInfo", "LandingGear")
    }
    engines = asks["EngineInfo"]
    assert "Give one entry per engine" in engines
    assert "position, engine type, esn" in engines and "tsn hours" in engines
    assert "Do not list headings or units" in engines
    assert "A number or a code is an entry only when it is what labels" in engines
    assert "belonging to" in engines


async def test_more_items_than_the_count_keeps_the_best_qualified() -> None:
    sheet = world()
    # the world believes there are 2 engines; a model that qualifies a third word
    # ("V2527-A5" as an engine) must not add a third row
    sheet.non_entity(
        "type value", schema=EngineInfo, parent="Aircraft", mentions=["V2527-A5"]
    )
    result = await create_extractor(
        [One(Aircraft)], transport=sheet.transport()
    ).ainvoke(SHEET)
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    assert isinstance(record.value, Aircraft)
    assert len(record.value.engines) == 2


@pytest.mark.parametrize("identity", ["position", None])
async def test_a_collection_with_an_identity_is_discovered_as_before(
    identity: Optional[str],
) -> None:
    class Item(BaseModel):
        """A listed item."""

        label: str

    class Box(BaseModel):
        """A box."""

        items: list[Item] = Field(default_factory=list)

    text = "The box holds a Spoon and a Whisk."
    sheet = FakeWorld(text)
    sheet.entity("Box", schema=Box, mentions=["box"], fields={})
    for name in ("Spoon", "Whisk"):
        sheet.entity(
            name, schema=Item, parent="Box", mentions=[name], fields={"label": name}
        )
    transport = sheet.transport()
    result = await create_extractor([One(Box)], transport=transport).ainvoke(text)
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    assert isinstance(record.value, Box)
    assert [i.label for i in record.value.items] == ["Spoon", "Whisk"]
    ask = next(
        body["fields"]["mentions"]["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover" and meta.schema == "Item"
    )
    assert "by their label" in ask and "Give one entry per" not in ask


# --------------------------------------------------------------------------- #
# Rows told apart only by numbers or codes (a 737 sheet)
# --------------------------------------------------------------------------- #


class Engine737(BaseModel):
    position: Literal["1", "2", "3", "4"]
    engine_type: Optional[str] = None
    esn: Optional[str] = None


class Gear737(BaseModel):
    position: Optional[str] = None
    cycles: Optional[int] = None


class Plane737(BaseModel):
    """An aircraft."""

    msn: Optional[str] = None
    engines: list[Engine737] = Field(default_factory=list)
    landing_gear: list[Gear737] = Field(default_factory=list)


SHEET_737 = """\
Boeing 737-800 MSN 12345.

## Powerplant

| Position | Engine Type | ESN | CSN |
|---|---|---|---|
| 1 | CFM56-7B26 | 889101 | 24310 |
| 2 | CFM56-7B26 | 889102 | 24102 |

## Landing gear

| Position | Cycles |
|---|---|
| L | 5000 |
| R | 5100 |
| LH | 5200 |
"""


def world_737() -> FakeWorld:
    sheet = FakeWorld(SHEET_737)
    sheet.entity(
        "Aircraft",
        schema=Plane737,
        mentions=["Boeing 737-800"],
        fields={"msn": "12345"},
    )
    for pos, esn in (("1", "889101"), ("2", "889102")):
        sheet.entity(
            f"engine {pos}",
            schema=Engine737,
            parent="Plane737",
            mentions=[esn],  # the row is named by its serial number
            fields={"position": pos, "engine_type": "CFM56-7B26", "esn": esn},
        )
    for label, cycles in (("L", 5000), ("R", 5100), ("LH", 5200)):
        sheet.entity(
            label,
            schema=Gear737,
            parent="Plane737",
            mentions=[label],
            fields={"position": label, "cycles": cycles},
        )
    return sheet


async def test_rows_named_by_numbers_and_codes_are_not_lost() -> None:
    sheet = world_737()
    result = await create_extractor(
        [One(Plane737)], transport=sheet.transport()
    ).ainvoke(SHEET_737)
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    plane = record.value
    assert isinstance(plane, Plane737)
    assert [e.esn for e in plane.engines] == ["889101", "889102"]
    assert [e.position for e in plane.engines] == ["1", "2"]
    assert [g.position for g in plane.landing_gear] == ["L", "R", "LH"]
    # the codes were dropped by the screen and came back because of the count
    assert any("came back because Clairic counted 3" in w for w in result.warnings)


async def test_the_discovery_wording_lets_numbers_label_a_row() -> None:
    sheet = world_737()
    transport = sheet.transport()
    await create_extractor([One(Plane737)], transport=transport).ainvoke(SHEET_737)
    ask = next(
        body["fields"]["mentions"]["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover" and meta.schema == "Engine737"
    )
    assert "bare numbers" not in ask
    assert "A number or a code is an entry only when it is what labels the row" in ask


async def test_a_clairic_undercount_does_not_drop_confident_items() -> None:
    sheet = world_737()
    sheet.count(Engine737, 1)  # Clairic miscounts: there are two engines
    result = await create_extractor(
        [One(Plane737)], transport=sheet.transport()
    ).ainvoke(SHEET_737)
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    assert isinstance(record.value, Plane737)
    assert [e.esn for e in record.value.engines] == ["889101", "889102"]
    assert any("count_below_discovered" in w for w in result.warnings)
    assert not any(s.reason == "capped_to_count" for s in result.skipped)
    assert not any("items_capped_to_count" in w for w in result.warnings)


async def test_a_weakly_qualified_extra_is_capped_with_its_own_reason() -> None:
    sheet = world_737()
    # the model half-believes that 4 is an engine of its own: p 0.5 to 0.8
    sheet.non_entity("maybe", schema=Engine737, parent="Plane737", mentions=["24310"])
    sheet.count(Engine737, 2)
    transport = sheet.transport()
    handler = transport._handler

    def lukewarm(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        body = handler(model, payload, meta)
        assert isinstance(body, dict)
        for qid, answer in body["answers"].items():
            if qid.startswith("qualify.") and answer["noul"] < 0.5:
                label = payload["state"]["entities"][qid.split(".", 1)[1]]["label"]
                if label == "24310":
                    answer["noul"] = 0.6
        return body

    result = await create_extractor(
        [One(Plane737)], transport=FakeTransport(lukewarm)
    ).ainvoke(SHEET_737)
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    assert isinstance(record.value, Plane737)
    assert [e.esn for e in record.value.engines] == ["889101", "889102"]
    capped = [s for s in result.skipped if s.reason == "capped_to_count"]
    assert [s.entity.label for s in capped] == ["24310"]
    assert any("items_capped_to_count" in w for w in result.warnings)
