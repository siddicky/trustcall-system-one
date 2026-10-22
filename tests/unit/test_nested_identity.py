"""``Many(identity="identification.msn")``: an identity inside a sub-model.

The aircraft spec sheet that exposed it: a record has no top-level name, its serial
number sits one level down.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Annotated, Any, Optional

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import Many, compile_schema, create_extractor
from trustcall_system_one._collection import _identity_phrase, identity_fields
from trustcall_system_one._match import (
    compact_record,
    identity_keys,
    identity_values,
    record_value,
)
from trustcall_system_one.testing import FakeWorld


class Ident(BaseModel):
    """Who the aircraft is."""

    msn: Optional[str] = Field(None, description="manufacturer serial number")
    registration: Optional[str] = None


class Weights(BaseModel):
    """Weights in kilograms."""

    mtow_kg: Optional[int] = None


class Aircraft(BaseModel):
    """An aircraft described on a spec sheet."""

    identification: Ident = Field(default_factory=Ident)
    weights: Weights = Field(default_factory=Weights)
    remarks: Optional[str] = None


SHEET = """\
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


def world() -> FakeWorld:
    sheet = FakeWorld(SHEET)
    sheet.entity(
        "8547",
        schema=Aircraft,
        mentions=["8547"],
        fields={
            "identification.msn": "8547",
            "identification.registration": "AP-ABC",
            "weights.mtow_kg": 78000,
            "remarks": "leased",
        },
        matches="8547",
    )
    sheet.entity(
        "12345",
        schema=Aircraft,
        mentions=["12345"],
        fields={
            "identification.msn": "12345",
            "identification.registration": "AP-XYZ",
            "weights.mtow_kg": 79016,
        },
    )
    return sheet


def test_a_dotted_identity_resolves_to_the_leaf_below_the_sub_model() -> None:
    plan = compile_schema(Aircraft)
    assert identity_fields(plan, Many(Aircraft, identity="identification.msn")) == (
        "identification.msn",
    )
    assert identity_fields(
        plan, Many(Aircraft, identity=["identification.msn", "remarks"])
    ) == ("identification.msn", "remarks")
    phrase = _identity_phrase(plan, ["identification.msn"])
    assert phrase == "msn (manufacturer serial number)"


@pytest.mark.parametrize(
    ("identity", "fragment"),
    [
        ("identification.serial", "Ident has no field of"),
        ("identity.msn", "Aircraft has no field of"),
        ("remarks.msn", "'remarks' is not a sub-model"),
        ("identification", "sub-model or a list of models"),
        ("identification.msn.x", "'msn' is not a sub-model"),
    ],
)
def test_a_bad_path_is_a_clear_error(identity: str, fragment: str) -> None:
    plan = compile_schema(Aircraft)
    with pytest.raises(ValueError, match=fragment):
        identity_fields(plan, Many(Aircraft, identity=identity))


def test_a_bad_path_in_an_annotation_fails_when_the_schema_is_compiled() -> None:
    class Fleet(BaseModel):
        owner: str
        planes: Annotated[list[Aircraft], Many(identity="identification.nope")] = Field(
            default_factory=list
        )

    with pytest.raises(ValueError, match=r"planes.*Ident has no field of"):
        compile_schema(Fleet)

    class Fleet2(BaseModel):
        owner: str
        planes: Annotated[list[Aircraft], Many(identity="identification.msn")] = Field(
            default_factory=list
        )

    assert compile_schema(Fleet2).spec(("planes",)).many is not None


def test_a_record_is_read_at_the_resolved_aliased_path() -> None:
    class Ident2(BaseModel):
        msn: Optional[str] = Field(None, alias="serial")

    class Plane(BaseModel):
        ident: Ident2 = Field(default_factory=Ident2, alias="identification")

    plan = compile_schema(Plane)
    keys = identity_keys(plan, ["ident.msn"])
    assert keys == ("identification.serial",)
    record = {"identification": {"serial": "8547"}, "other": 1}
    assert record_value(record, keys[0]) == "8547"
    assert identity_values(record, keys) == ["8547"]
    assert identity_values({"identification": {}}, keys) == []  # nothing to say
    assert record_value({"identification": "flat"}, keys[0]) is None


def test_a_compact_record_keeps_the_nested_identity_and_little_else() -> None:
    record = {
        "identification": {"msn": "8547", "registration": "AP-ABC"},
        "weights": {"mtow_kg": 78000},
        "remarks": "leased",
        "year": 2018,
    }
    compact = compact_record(record, ["identification.msn"], 1)
    assert compact["identification"] == {"msn": "8547"}
    assert "weights" not in compact
    assert compact["remarks"] == "leased"  # the first of the other scalars


async def test_two_aircraft_are_found_by_their_serial_numbers() -> None:
    sheet = world()
    extractor = create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=sheet.transport()
    )
    result = await extractor.ainvoke(SHEET)
    got = {
        r.value.identification.msn: r.value
        for r in result.records
        if isinstance(r.value, Aircraft)
    }
    assert set(got) == {"8547", "12345"}
    assert got["8547"].weights.mtow_kg == 78000
    assert got["12345"].identification.registration == "AP-XYZ"
    assert not any("no_identity_field" in w for w in result.warnings)


async def test_discovery_is_asked_by_the_nested_identity() -> None:
    sheet = world()
    transport = sheet.transport()
    extractor = create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=transport
    )
    await extractor.ainvoke(SHEET)
    asks = [
        field["instructions"]
        for _, body, meta in transport.calls
        if meta.kind == "discover"
        for field in body["fields"].values()
    ]
    assert asks and all("by their msn (manufacturer serial number)" in a for a in asks)


async def test_existing_records_are_matched_by_the_nested_identity() -> None:
    sheet = world()
    existing: list[tuple[str, str, dict[str, Any]]] = [
        (
            "rec-a",
            "Aircraft",
            {
                "identification": {"msn": "8547", "registration": "AP-OLD"},
                "weights": {"mtow_kg": 77000},
                "remarks": None,
            },
        ),
        (
            "rec-b",
            "Aircraft",
            {"identification": {"msn": "99999"}, "weights": {"mtow_kg": 1}},
        ),
    ]
    extractor = create_extractor(
        [Many(Aircraft, identity="identification.msn")],
        transport=sheet.transport(),
        enable_inserts=True,
    )
    result = await extractor.ainvoke(SHEET, existing=existing)
    updated = {r.json_doc_id: r for r in result.records if r.valid}
    assert "rec-a" in updated  # the aircraft with MSN 8547 is that record
    assert "rec-b" not in updated  # no other aircraft is MSN 99999
    inserted = [r for r in result.records if r.json_doc_id not in ("rec-a", "rec-b")]
    assert any(
        isinstance(r.value, Aircraft) and r.value.identification.msn == "12345"
        for r in inserted
    )


async def test_a_nested_identity_is_never_replaced_by_a_short_form_of_itself() -> None:
    sheet = world()
    existing = [
        (
            "rec-a",
            "Aircraft",
            {"identification": {"msn": "MSN 8547 (ex-PK)"}, "weights": {"mtow_kg": 1}},
        )
    ]
    extractor = create_extractor(
        [Many(Aircraft, identity="identification.msn")], transport=sheet.transport()
    )
    result = await extractor.ainvoke(SHEET, existing=existing)
    record = next(r for r in result.records if r.json_doc_id == "rec-a")
    assert record.value is not None
    assert isinstance(record.value, Aircraft)
    assert record.value.identification.msn == "MSN 8547 (ex-PK)"
    assert any(
        r.reason == "identity_subset" and r.path == "/identification/msn"
        for r in record.rejected
    )
