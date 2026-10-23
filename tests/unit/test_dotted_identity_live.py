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
    s1_field,
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


def _in_a_cell(source: str, span: Any) -> bool:
    """Whether a mention sits in a table cell (a pipe before it on its line)."""
    line_start = source.rfind("\n", 0, span[0]) + 1
    return "|" in source[line_start : span[0]]


def like_the_live_model(
    world: FakeWorld, *, obeys_the_hint: bool = True, literal: bool = True
) -> FakeTransport:
    """The world, with what the live model did.

    * Discovery follows its instruction: asked for mentions "by their msn" it lists the
      table cells that hold one (``8547 |``, with the pipe); asked for how the input
      names the aircraft it lists those words.
    * ``obeys_the_hint=False``: discovery IGNORES the wording and lists the cells of the
      msn whatever it was asked (the recall risk of asking for words: the model that
      does not read "name or describe").
    * ``literal=True``: a reader that takes the instruction at its word: told "do not
      list table cells" it lists none, so a sheet that names its aircraft in a table
      cell only (``| Aircraft | Airbus A320-214 |``) loses them, unless the wording
      allows the cells that name or describe the aircraft.
    * Qualification refuses a bare ``12345`` (live: 0.14) unless the question says an
      identifier counts ("or the msn of an aircraft").
    """
    base = world.handler

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        body = cast("dict[str, Any]", base(model, payload, meta))
        if meta.kind == "discover":
            asked = str(payload["fields"]["mentions"]["instructions"])
            cells = "by their msn" in asked or not obeys_the_hint
            answer = body["answers"]["mentions"]
            source = str(payload.get("state_text") or world.text)
            forbids_cells = (
                "Do not list" in asked
                and "table cells" in asked
                and "table cell is a mention" not in asked
            )
            keep = [
                i
                for i, text in enumerate(answer["value"])
                if text.rstrip(" |").isdigit() == cells
                and not (
                    literal
                    and forbids_cells
                    and not cells
                    and _in_a_cell(source, answer["spans"][i])
                )
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
     "TBD", "—", "--", "To be confirmed", "Not available"],
)  # fmt: skip
def test_placeholders_are_recognised(text: str) -> None:
    assert is_placeholder(text, GateConfig().placeholders)


@pytest.mark.parametrize(
    "text",
    [
        "Honeywell",
        "GTCP 131-9A",
        "None",
        "Unknown",
        "NA",
        "TBA-1",
        "",
        "0",
        "A-",
        "-",
        "?",
        "TBC",
    ],
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


class Inspection(BaseModel):
    """An inspection."""

    result: Optional[str] = Field(None, json_schema_extra=s1_field(placeholders=False))
    status: Optional[str] = None


async def test_one_field_can_keep_its_placeholder() -> None:
    text = "Check: result N/A, status TBA."
    world = FakeWorld(text)
    world.entity(
        "the Check",
        schema=Inspection,
        mentions=["Check"],
        fields={"result": "N/A", "status": "TBA"},
    )
    (record,) = (
        await create_extractor([One(Inspection)], transport=world.transport()).ainvoke(
            text
        )
    ).records
    assert isinstance(record.value, Inspection)
    assert record.value.result == "N/A"  # this field keeps it
    assert record.value.status is None  # the run's setting applies elsewhere
    with pytest.raises(ValueError, match="placeholders must be a bool"):
        s1_field(placeholders="no")  # type: ignore[arg-type]


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


# --------------------------------------------------------------------------- #
# The review: recall of the words, figures that are no aircraft, cheap matching
# --------------------------------------------------------------------------- #

TABLE_ONLY = """\
| Aircraft | Airbus A320-214 |
| MSN | 8547 |
| Registration | AP-ABC |
| MTOW | 78000 kg |
Remarks: leased

| Aircraft | Boeing 737-800 |
| MSN | 12345 |
| Registration | AP-XYZ |
| MTOW | 79016 kg |
"""


def table_only_world() -> FakeWorld:
    """Both aircraft are named only by a table cell (``| Aircraft | Airbus A320 |``)."""
    world = FakeWorld(TABLE_ONLY)
    world.entity(
        "Airbus A320-214",
        schema=Aircraft,
        mentions=["Airbus A320-214", "8547"],
        fields={
            "identification.msn": "8547",
            "identification.registration": "AP-ABC",
            "weights.mtow_kg": 78000,
            "remarks": "leased",
        },
    )
    world.entity(
        "Boeing 737-800",
        schema=Aircraft,
        mentions=["Boeing 737-800", "12345"],
        fields={
            "identification.msn": "12345",
            "identification.registration": "AP-XYZ",
            "weights.mtow_kg": 79016,
        },
    )
    return world


def _msns(result: Any) -> list[str]:
    return sorted(
        str(r.value.identification.msn)
        for r in result.records
        if r.valid and isinstance(r.value, Aircraft)
    )


async def test_a_table_only_sheet_still_finds_its_aircraft() -> None:
    # a reader that takes "do not list table cells" literally would find nothing: the
    # wording allows the cells that name or describe the aircraft
    transport = like_the_live_model(table_only_world())
    result = await create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    ).ainvoke(TABLE_ONLY)
    assert _msns(result) == ["12345", "8547"]
    asked = [
        field["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover"
        for field in body["fields"].values()
    ]
    assert asked and not any(
        "Do not list pronouns, headings or table cells" in a for a in asked
    )
    assert all(
        "A table cell is a mention when it names or describes" in a for a in asked
    )


async def test_a_model_that_ignores_the_wording_still_finds_them_by_msn() -> None:
    # the recall risk the other way: discovery lists the msn cells whatever it is asked
    transport = like_the_live_model(sheet_world(), obeys_the_hint=False)
    result = await create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    ).ainvoke(MULTI)
    assert _msns(result) == ["12345", "8547"]
    assert not [s for s in result.skipped if s.reason == "not_qualified"]


async def test_the_broader_second_pass_asks_by_the_identity() -> None:
    world = sheet_world()
    base = like_the_live_model(world)

    def only_the_first(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        body = cast("dict[str, Any]", base._handler(model, payload, meta))
        if meta.kind == "discover":
            asked = str(payload["fields"]["mentions"]["instructions"])
            if "by their msn" not in asked:  # the words find one aircraft only
                answer = body["answers"]["mentions"]
                keep = [i for i, t in enumerate(answer["value"]) if "Airbus" in t]
                answer["value"] = [answer["value"][i] for i in keep]
                answer["spans"] = [answer["spans"][i] for i in keep]
        return body

    transport = FakeTransport(only_the_first)
    result = await create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    ).ainvoke(MULTI)
    asked = [
        field["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover"
        for field in body["fields"].values()
    ]
    assert any("by their msn" in a for a in asked), "the fallback never asked by msn"
    assert _msns(result) == ["12345", "8547"]


FLEET = """\
| Model | MSN | Registration |
| --- | --- | --- |
| A320 | 8547 | AP-BLD |
| A320 | 8601 | AP-BMX |
"""


async def test_two_a320_rows_are_two_records_told_apart_by_msn() -> None:
    world = FakeWorld(FLEET)
    for index, (msn, reg) in enumerate((("8547", "AP-BLD"), ("8601", "AP-BMX"))):
        world.entity(
            f"A320 {msn}",
            schema=Aircraft,
            mentions=[("A320", index)],
            fields={
                "identification.msn": msn,
                "identification.registration": reg,
            },
        )
    result = await create_extractor(
        [Many(Aircraft, identity="identification.msn")],
        transport=like_the_live_model(world),
    ).ainvoke(FLEET)
    planes = {
        r.value.identification.msn: r.value.identification.registration
        for r in result.records
        if r.valid and isinstance(r.value, Aircraft)
    }
    assert planes == {"8547": "AP-BLD", "8601": "AP-BMX"}


async def test_a_figure_of_the_table_is_not_an_aircraft() -> None:
    # discovery lists the MTOW 78000, a gullible qualification says yes, and the
    # extraction finds no msn equal to it: the post-check drops the entity, and the
    # warning names the kind, not the figure
    text = TABLE_ONLY
    world = table_only_world()
    world.non_entity("78000", schema=Aircraft, mentions=["78000"])
    inner = like_the_live_model(world, obeys_the_hint=True)

    def gullible(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        body = cast("dict[str, Any]", inner._handler(model, payload, meta))
        if meta.kind == "discover":
            answer = body["answers"]["mentions"]
            start = text.index("78000")
            answer["value"].append("78000")
            answer["spans"].append([start, start + 5])
        if meta.kind == "qualify":
            for qid in payload["questions"]:
                entry = payload["state"]["entities"][qid.removeprefix("qualify.")]
                if entry["label"] == "78000":
                    body["answers"][qid] = noul(0.95)
        return body

    result = await create_extractor(
        [Many(Aircraft, identity="identification.msn")],
        transport=FakeTransport(gullible),
    ).ainvoke(text)
    assert _msns(result) == ["12345", "8547"]
    dropped = [w for w in result.warnings if "bare_number_entity_dropped" in w]
    assert dropped and "78000" not in " ".join(dropped)
    assert "aircraft" in dropped[0]
    assert any(s.reason == "not_qualified" for s in result.skipped)


class Shaped(BaseModel):
    """An aircraft whose msn has a shape."""

    identification: "ShapedIdent"
    remarks: Optional[str] = None


class ShapedIdent(BaseModel):
    """Who the aircraft is."""

    msn: Optional[str] = Field(None, pattern=r"^\d{4,5}$")


Shaped.model_rebuild()


def test_a_figure_that_is_not_the_shape_of_the_identity_is_not_offered() -> None:
    from types import SimpleNamespace

    from trustcall_system_one._collection import _identity_shape, _rank, _Screen
    from trustcall_system_one._compile import compile_schema

    scope: Any = SimpleNamespace(
        plan=compile_schema(Shaped),
        cardinality=Many(Shaped, identity="identification.msn"),
    )
    shape = _identity_shape(scope)
    assert shape is not None and shape.fullmatch("8547")
    screen = _Screen(frozenset(), False, frozenset(), shape)
    assert _rank("8547", screen) >= 3  # the shape of an msn
    assert _rank("178000", screen) == 1  # a weight, not an msn
    assert _rank("2024", screen) >= 3  # still a candidate: the post-check decides
    plain = _Screen(frozenset(), False, frozenset(), None)
    assert _rank("178000", plain) >= 3  # no shape known: as before


PARTS = (
    "Inspection update: the fuel pump S/N FP-0193 is now unserviceable, seal leak "
    "found. Starter generator SG-5521 is unchanged."
)


async def test_a_stated_identifier_needs_no_presence_request() -> None:
    from evals.holdout_cases import Component

    world = FakeWorld(PARTS)
    world.entity(
        "fuel pump",
        schema=Component,
        mentions=["fuel pump"],
        fields={
            "identification": {"serial": "FP-0193"},
            "condition": "unserviceable",
            "remarks": "seal leak found",
        },
        matches="FP-0193",
    )
    world.entity(
        "Starter generator",
        schema=Component,
        mentions=["Starter generator"],
        fields={"identification": {"serial": "SG-5521"}},
        matches="SG-5521",
    )
    existing: list[tuple[str, str, dict[str, Any]]] = [
        (
            "part-1",
            "Component",
            {
                "identification": {"serial": "SG-5521", "part_number": "23080-14"},
                "condition": "serviceable",
                "remarks": None,
            },
        ),
        (
            "part-2",
            "Component",
            {
                "identification": {"serial": "FP-0193", "part_number": "FP-770"},
                "condition": "serviceable",
                "remarks": None,
            },
        ),
    ]
    transport = world.transport()
    result = await create_extractor(
        [Many(Component, identity="identification.serial")],
        transport=transport,
        enable_inserts=True,
    ).ainvoke(PARTS, existing=existing)
    kinds = [meta.kind for _, _, meta in transport.calls]
    assert "record_presence" not in kinds
    updated = {r.json_doc_id: r for r in result.records if r.valid}
    assert set(updated) == {"part-1", "part-2"}
    assert len(result.records) == 2  # no entity made twice
    pump = updated["part-2"].value
    assert pump is not None and pump.condition == "unserviceable"  # type: ignore[attr-defined]


def test_an_identifier_no_entity_covers_adopts_its_own_words() -> None:
    from types import SimpleNamespace

    from trustcall_system_one._collection import (
        _adopt_by_identifier,
        _distinctive,
        _Group,
    )
    from trustcall_system_one._types import Mention

    text = "Fuel pump overhauled.\nSG-5521 | unserviceable\nStarter noted."
    scope: Any = SimpleNamespace(source=text)
    pump = _Group((Mention("Fuel pump", (0, 9)),), 0.9)
    starter = _Group(
        (Mention("Starter", (text.index("Starter"), text.index("Starter") + 7)),), 0.2
    )
    groups, live = [pump, starter], [pump]
    assert _adopt_by_identifier(scope, ["SG-5521"], groups, live)
    # no live entity in its row: its own words become the entity
    assert len(groups) == 3 and live[-1].mentions[0].text == "SG-5521"
    assert live[-1].adopted
    # one in the row covers it: nothing is added
    row = _Group(
        (
            Mention(
                "unserviceable",
                (text.index("unserviceable"), text.index("unserviceable") + 13),
            ),
        ),
        0.9,
    )
    groups, live = [row], [row]
    assert _adopt_by_identifier(scope, ["SG-5521"], groups, live)
    assert len(groups) == 1 and live == [row]
    # a refused group in the row comes back (from REVIVE_MIN)
    refused = _Group(row.mentions, 0.5)
    groups, live = [refused], []
    assert _adopt_by_identifier(scope, ["SG-5521"], groups, live)
    assert live == [refused] and refused.adopted
    # a word that is not an identifier is left to the presence question
    assert not _adopt_by_identifier(scope, ["Fuel pump"], [pump], [pump])
    assert not _adopt_by_identifier(scope, ["ZZ-9999"], [pump], [pump])
    assert _distinctive("8547") and _distinctive("FP-0193")
    assert not _distinctive("Pump") and not _distinctive("A32")
