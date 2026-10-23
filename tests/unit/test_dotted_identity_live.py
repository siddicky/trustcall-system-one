"""What the live aircraft-sheet runs found about a dotted identity, as small fixtures.

``Many(Aircraft, identity="identification.msn")`` on a two-aircraft sheet (the live
run kept the A320 and lost the B737), and on a 591-line file (1,705 refused entities,
97 per cent of them maintenance checks, a table's worth of cells). The fixtures are
synthetic; the shapes are the live ones: a mention ``12345 |`` (the table cell with its
pipe) that a model refuses to call an aircraft, 40 table cells proposed as checks.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Any, Optional, cast

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    CallMeta,
    DiscoveryConfig,
    GateConfig,
    Many,
    One,
    create_extractor,
)
from trustcall_system_one._coerce import is_placeholder
from trustcall_system_one._collection import clean_mention
from trustcall_system_one.testing import FakeTransport, FakeWorld, noul

# --------------------------------------------------------------------------- #
# Markup around a mention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("8547 |", "8547"),
        ("| 12345 |", "12345"),
        ("AIRCRAFTS**", "AIRCRAFTS"),
        ("**Airbus A320CEO**", "Airbus A320CEO"),
        ("NEO) \N{EN DASH}", "NEO"),
        ("_4`677_", "4`677"),
    ],
)
def test_table_and_emphasis_markup_is_stripped_from_a_mention(
    raw: str, clean: str
) -> None:
    mention = clean_mention(raw, (100, 100 + len(raw)))
    assert mention is not None and mention.text == clean
    assert mention.span is not None
    start, end = mention.span
    assert (end - start) == len(clean)  # the span shrank with the text
    assert start - 100 == raw.index(clean)


def test_a_mention_that_is_only_markup_is_dropped() -> None:
    for raw in ("|", "**", "*\\", "\N{EN DASH}", "| |"):
        assert clean_mention(raw, (0, len(raw))) is None


# --------------------------------------------------------------------------- #
# The two-aircraft sheet
# --------------------------------------------------------------------------- #


class Ident(BaseModel):
    """Who the aircraft is."""

    manufacturer: Optional[str] = None
    msn: Optional[str] = Field(None, description="Manufacturer Serial Number")
    registration: Optional[str] = None


class Weights(BaseModel):
    """Weights in kilograms."""

    mtow_kg: Optional[int] = None


class Aircraft(BaseModel):
    """An aircraft described on a specification sheet."""

    identification: Ident = Field(default_factory=Ident)
    weights: Weights = Field(default_factory=Weights)
    remarks: Optional[str] = None


MULTI = """\
# AIRCRAFT SPECIFICATION SHEET (1 of 2)

Airbus A320-200 in operational condition, available for lease or sale.

## 1. Aircraft Identification

| Item | Detail |
|---|---|
| Manufacturer | Airbus |
| Aircraft Model | A320-214 |
| Manufacturer Serial Number (MSN) | 8547 |
| Registration | AP-BLD |

## 3. Operational Weights

| Weight | Value |
|---|---|
| Maximum Take-Off Weight (MTOW) | 77,000 kg |

Aircraft in good condition.

---

# AIRCRAFT SPECIFICATION SHEET (2 of 2)

Boeing 737-800, also offered by the asset management team.

## 1. Aircraft Identification

| Item | Detail |
|---|---|
| Manufacturer | Boeing |
| Aircraft Model | 737-8AS |
| Manufacturer Serial Number (MSN) | 12345 |
| Registration | AP-BMX |

## 3. Operational Weights

| Weight | Value |
|---|---|
| Maximum Take-Off Weight (MTOW) | 79,016 kg |
"""


def sheet_world() -> FakeWorld:
    """Both aircraft are identified by MSN; each is also called by its model."""
    world = FakeWorld(MULTI)
    world.entity(
        "Airbus A320-200",
        schema=Aircraft,
        mentions=["Airbus A320-200", "8547 |"],
        fields={
            "identification.manufacturer": "Airbus",
            "identification.msn": "8547",
            "identification.registration": "AP-BLD",
            "weights.mtow_kg": 77000,
            "remarks": "Aircraft in good condition.",
        },
    )
    world.entity(
        "Boeing 737-800",
        schema=Aircraft,
        mentions=["Boeing 737-800", "12345 |"],
        fields={
            "identification.manufacturer": "Boeing",
            "identification.msn": "12345",
            "identification.registration": "AP-BMX",
            "weights.mtow_kg": 79016,
        },
    )
    return world


def like_the_live_model(world: FakeWorld) -> FakeTransport:
    """The world, with two things the live model did.

    * Discovery follows its instruction: asked for mentions "by their msn" it lists the
      table cells that hold one (``8547 |``, with the pipe); asked for how the input
      names the aircraft it lists those words.
    * Qualification refuses a bare ``12345`` (live: 0.14) unless the question says an
      identifier counts ("or the msn of an aircraft").
    """
    base = world.handler

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        body = cast("dict[str, Any]", base(model, payload, meta))
        if meta.kind == "discover":
            asked = str(payload["fields"]["mentions"]["instructions"])
            cells = "by their msn" in asked
            answer = body["answers"]["mentions"]
            keep = [
                i
                for i, text in enumerate(answer["value"])
                if text.rstrip(" |").isdigit() == cells
            ]
            answer["value"] = [answer["value"][i] for i in keep]
            answer["spans"] = [answer["spans"][i] for i in keep]
        if meta.kind == "qualify":
            for qid, question in payload["questions"].items():
                entry = payload["state"]["entities"][qid.removeprefix("qualify.")]
                bare = entry["label"].rstrip(" |").isdigit()
                in_context = "or the msn of" in str(question["instructions"])
                if bare and not in_context and "12345" in entry["label"]:
                    body["answers"][qid] = noul(0.14)
        return body

    return FakeTransport(handler)


async def test_both_aircraft_of_a_sheet_identified_by_msn_are_found() -> None:
    transport = like_the_live_model(sheet_world())
    extractor = create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    )
    result = await extractor.ainvoke(MULTI)
    planes = {
        r.value.identification.msn: r.value
        for r in result.records
        if r.valid and isinstance(r.value, Aircraft)
    }
    assert sorted(str(msn) for msn in planes) == ["12345", "8547"]  # B737 kept
    assert planes["8547"].identification.registration == "AP-BLD"
    assert planes["12345"].identification.registration == "AP-BMX"  # no bleed
    assert planes["8547"].weights.mtow_kg == 77000
    assert planes["8547"].remarks == "Aircraft in good condition."
    assert planes["12345"].remarks is None
    assert not [s for s in result.skipped if s.reason == "not_qualified"]


async def test_the_entities_are_called_by_what_the_input_calls_them() -> None:
    transport = like_the_live_model(sheet_world())
    extractor = create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    )
    result = await extractor.ainvoke(MULTI)
    labels = sorted(str(r.entity.label) for r in result.records if r.entity)
    assert labels == ["Airbus A320-200", "Boeing 737-800"]  # not "8547 |"
    asked = [
        field["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover"
        for field in body["fields"].values()
    ]
    assert asked and not any("by their msn" in text for text in asked)
    assert all("name or describe the aircraft" in text for text in asked)
    assert all("not the msn or another number or code" in text for text in asked)
    # the entity's own extraction is focused on those words, not on a table cell
    focuses = [
        str(body["state"]).split("</focus>")[0]
        for _, body, meta in transport.calls
        if meta.kind == "extract"
    ]
    assert focuses and all("|" not in focus for focus in focuses)
    assert any('entity="Airbus A320-200"' in focus for focus in focuses)


async def test_qualification_judges_an_identifier_in_context() -> None:
    transport = like_the_live_model(sheet_world())
    extractor = create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    )
    await extractor.ainvoke(MULTI)
    questions = [
        question
        for _, body, meta in transport.calls
        if meta.kind == "qualify"
        for question in body["questions"].values()
    ]
    assert questions
    for question in questions:
        assert "or the msn of an aircraft?" in question["instructions"]
        assert "or the msn that identifies one" in question["criteria"]["true"]


async def test_a_plain_identity_is_asked_as_before() -> None:
    class Account(BaseModel):
        """An account."""

        company: str
        seats: Optional[int] = None

    world = FakeWorld("Orbit Freight has 25 seats.")
    world.entity(
        "Orbit Freight",
        schema=Account,
        mentions=["Orbit Freight"],
        fields={"company": "Orbit Freight", "seats": 25},
    )
    transport = world.transport()
    await create_extractor(
        [Many(Account, identity="company")], transport=transport
    ).ainvoke("Orbit Freight has 25 seats.")
    asks = [
        f["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover"
        for f in body["fields"].values()
    ]
    assert asks and all("by their company" in a for a in asks)
    questions = [
        q["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "qualify"
        for q in body["questions"].values()
    ]
    assert questions and not any(" or the " in q for q in questions)


# --------------------------------------------------------------------------- #
# A table's worth of cells
# --------------------------------------------------------------------------- #


class Check(BaseModel):
    """One maintenance check."""

    check_type: str = Field(..., description="e.g. A-check, C-check, 6Y")
    remarks: Optional[str] = None


class Plane(BaseModel):
    """An aircraft."""

    identification: Ident = Field(default_factory=Ident)
    checks: list[Check] = Field(default_factory=list)


NOUNS = ["LPC", "HPC", "HPT", "LPT", "FAN"]
CELLS = [
    *(str(n) for n in range(1, 41)),  # the item numbers: bare figures
    *(f"{NOUNS[n % 5]} {n} DISK" for n in range(1, 41)),  # words, no check
    *(f"3A{2400 + n * 7}" for n in range(1, 41)),  # part numbers
]
TABLE = """\
| Aircraft Model | Airbus A320-232 |
| MSN | 8547 |

| Check | Last Completed | Remarks |
| --- | --- | --- |
| A Check | March 2024 | none |
| C Check | June 2022 | none |

| No | Component | Part number |
| --- | --- | --- |
""" + "\n".join(
    f"| {n} | {NOUNS[n % 5]} {n} DISK | 3A{2400 + n * 7} |" for n in range(1, 41)
)


def table_world() -> FakeWorld:
    world = FakeWorld(TABLE)
    world.entity(
        "Plane",
        schema=Plane,
        mentions=["Airbus A320-232"],
        fields={"identification.msn": "8547"},
    )
    for check in ("A Check", "C Check"):
        world.entity(
            check,
            schema=Check,
            parent="Plane",
            mentions=[check],
            fields={"check_type": check},
        )
    for cell in CELLS:
        world.non_entity(cell, schema=Check, parent="Plane", mentions=[cell])
    return world


def count_kinds(transport: FakeTransport) -> dict[str, int]:
    seen: dict[str, int] = {}
    for _, _, meta in transport.calls:
        seen[meta.kind] = seen.get(meta.kind, 0) + 1
    return seen


async def test_the_cells_of_a_table_are_not_clustered_as_checks() -> None:
    transport = table_world().transport()
    result = await create_extractor([One(Plane)], transport=transport).ainvoke(TABLE)
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    assert isinstance(record.value, Plane)
    assert [c.check_type for c in record.value.checks] == ["A Check", "C Check"]
    kinds = count_kinds(transport)
    assert kinds.get("coref", 0) <= 2  # not one request per batch of 16 cells
    refused = [s for s in result.skipped if s.schema_name == "Check"]
    assert len(refused) <= 90  # the 80 words and part numbers: figures never got there
    assert any("junk_mentions_dropped" in w for w in result.warnings)
    assert any("mentions_screened_out" in w for w in result.warnings)


async def test_switching_the_screen_off_gives_the_old_behaviour() -> None:
    on, off = table_world().transport(), table_world().transport()
    await create_extractor([One(Plane)], transport=on).ainvoke(TABLE)
    unscreened = create_extractor(
        [One(Plane)], transport=off, discovery=DiscoveryConfig(screen_above=0)
    )
    result = await unscreened.ainvoke(TABLE)
    assert count_kinds(off).get("coref", 0) > 4 * max(
        1, count_kinds(on).get("coref", 0)
    )
    assert len(result.skipped) > 100
    assert not any("junk_mentions_dropped" in w for w in result.warnings)


async def test_a_short_list_of_checks_is_never_screened() -> None:
    text = "| Check | Remarks |\n| A Check | none |\n| C Check | none |\n| MSN | 8547 |"
    world = FakeWorld(text)
    world.entity(
        "Plane", schema=Plane, mentions=["MSN"], fields={"identification.msn": "8547"}
    )
    for check in ("A Check", "C Check"):
        world.entity(
            check,
            schema=Check,
            parent="Plane",
            mentions=[check],
            fields={"check_type": check},
        )
    result = await create_extractor([One(Plane)], transport=world.transport()).ainvoke(
        text
    )
    (record,) = [r for r in result.records if r.valid and r.value is not None]
    assert isinstance(record.value, Plane)
    assert [c.check_type for c in record.value.checks] == ["A Check", "C Check"]
    assert not any("junk_mentions_dropped" in w for w in result.warnings)


LEGEND = " ".join(
    ["| A | B | C | AD | AT | 77 | CF | E | V |"]
    + [f"| CN{n}.{n + 20} | {n} |" for n in range(21, 31)]
)


async def test_a_top_level_dotted_identity_screens_a_table_of_candidates() -> None:
    text = MULTI + "\n" + LEGEND
    world = FakeWorld(text)
    for label, msn, name in (
        ("Airbus A320-200", "8547", "Airbus A320-200"),
        ("Boeing 737-800", "12345", "Boeing 737-800"),
    ):
        world.entity(
            label,
            schema=Aircraft,
            mentions=[name],
            fields={"identification.msn": msn},
        )
    junk = ["A", "B", "C", "AD", "AT", "77", "CF", "E", "V"] + [
        f"CN{n}.{n + 20}" for n in range(21, 31)
    ]
    for cell in junk:
        world.non_entity(cell, schema=Aircraft, mentions=[cell])
    transport = world.transport()
    result = await create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    ).ainvoke(text)
    found = sorted(str(r.entity.label) for r in result.records if r.entity)
    assert found == ["Airbus A320-200", "Boeing 737-800"]
    assert any("junk_mentions_dropped" in w for w in result.warnings)
    assert count_kinds(transport).get("coref", 0) <= 2


async def test_a_top_level_plain_identity_is_never_screened() -> None:
    class Person(BaseModel):
        """A person."""

        name: str

    names = [f"Pat{chr(65 + n)} Lee" for n in range(14)]
    text = ". ".join(f"{n} came" for n in names) + "."
    world = FakeWorld(text)
    for n in names:
        world.entity(n, schema=Person, mentions=[n], fields={"name": n})
    result = await create_extractor(
        [Many(Person, identity="name")], transport=world.transport()
    ).ainvoke(text)
    assert len([r for r in result.records if r.valid]) == len(names)
    assert not any("junk_mentions_dropped" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Placeholders are no values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    ["TBA", "tba", " TBA ", "**TBA**", "N/A", "n/a.", "n.a.",
     "TBD", "—", "-", "To be confirmed", "Not available"],
)  # fmt: skip
def test_placeholders_are_recognised(text: str) -> None:
    assert is_placeholder(text, GateConfig().placeholders)


@pytest.mark.parametrize(
    "text",
    ["Honeywell", "GTCP 131-9A", "None", "Unknown", "NA", "TBA-1", "", "0", "A-"],
)
def test_real_values_are_not_placeholders(text: str) -> None:
    assert not is_placeholder(text, GateConfig().placeholders)


class Apu(BaseModel):
    """An auxiliary power unit."""

    manufacturer: Optional[str] = None
    status: Optional[str] = None
    serial_number: Optional[str] = None


APU_TEXT = "APU: Honeywell, status TBA, serial number N/A."


def apu_world() -> FakeWorld:
    world = FakeWorld(APU_TEXT)
    world.entity(
        "the Apu",
        schema=Apu,
        mentions=["APU"],
        fields={
            "manufacturer": "Honeywell",
            "status": "TBA",
            "serial_number": "N/A",
        },
    )
    return world


async def test_a_placeholder_is_null_not_a_value() -> None:
    result = await create_extractor(
        [One(Apu)], transport=apu_world().transport()
    ).ainvoke(APU_TEXT)
    (record,) = result.records
    apu = record.value
    assert isinstance(apu, Apu)
    assert apu.manufacturer == "Honeywell"
    assert apu.status is None and apu.serial_number is None
    assert "placeholder" in record.fields["status"].warnings
    assert "placeholder" in record.fields["serial_number"].warnings
    assert record.valid


async def test_placeholders_can_be_kept_as_values() -> None:
    result = await create_extractor(
        [One(Apu)],
        transport=apu_world().transport(),
        gates=GateConfig(placeholders=()),
    ).ainvoke(APU_TEXT)
    (record,) = result.records
    assert isinstance(record.value, Apu)
    assert (record.value.status, record.value.serial_number) == ("TBA", "N/A")


def test_the_placeholders_setting_is_validated() -> None:
    with pytest.raises(ValueError, match="placeholders"):
        GateConfig(placeholders=["TBA"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="placeholders"):
        GateConfig(placeholders=("TBA", " "))
    assert GateConfig(placeholders=("tba",)).placeholders == ("tba",)
    assert hash(GateConfig()) == hash(GateConfig())


def test_a_screen_setting_is_validated() -> None:
    with pytest.raises(ValueError, match="screen_above"):
        DiscoveryConfig(screen_above=-1)
    assert DiscoveryConfig().screen_above == 12
    assert DiscoveryConfig(screen_above=0).screen_above == 0
