"""The second held-out eval suite: 30 cases written blind, after the first holdout.

``evals/holdout_cases.py`` was used for diagnosis, so it no longer measures
generalisation cleanly. This suite replaces it as the clean held-out set. It was
written from the case format, the scorer and ``docs/DESIGN.md`` alone: no live
result, report, cassette or case of the first holdout was read, and no case was
ever run live before it was frozen. The rule is in ``evals/README.md``: never tune
on it, never use it for diagnosis. A case is only changed when its gold is wrong
from the text alone (in a commit of its own that says so).

The format is the dev suite's (``evals.model.Case``): the same scoring and a scripted
``FakeWorld`` per case, so every case passes offline. The domains are new: insurance
claims, school field-trip permission forms, fleet vehicle service logs, restaurant
supplier orders, library book loans, IT asset inventory, conference speaker bios,
soil sample tracking (lab work, non-medical), boat charter bookings and utility
meter readings; three cases each. Every name, number and identifier is made up.

Where the library documents a limit, the gold follows the documented behaviour and
the case says so in a comment and in its ``note``:

* a date with a month and a year but no day ("June 2025") is not resolved to a
  ``date`` (``absolute date incomplete``: never guessed), so the field is null;
* a relative edit ("three more guests") is refused (``relative_edit_unsupported``),
  never computed and never written as the stated number;
* an input that both changes and removes a record is a ``conflict_update_delete``:
  neither is applied;
* a text value is a verbatim span of the input (a value broken over two lines is
  quoted with its break; scoring folds whitespace);
* a dotted identity is not asked for by value: such a record is named by the words
  the input uses for it (its model, type or heading), so the gold names inserts by
  those words.
"""

# ruff: noqa: E501

from datetime import date, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from evals.model import Case, MustNot, Op, Rejected, Val
from trustcall_system_one import Many, One, s1_field
from trustcall_system_one.testing import FakeWorld, clairic_answer

__all__ = ["HOLDOUT2_CASES", "SUITE_NAME", "holdout2_cases"]

SUITE_NAME = "holdout2"


def _tags(words: str) -> tuple[str, ...]:
    """The tags of a case, written as one string of words."""
    return tuple(words.split())


# Relative dates in these texts are resolved against Wednesday 2026-09-16:
# "yesterday" is 2026-09-15, "three days ago" 2026-09-13 and "in two weeks"
# 2026-09-30.
REFERENCE_DATE = "2026-09-16"
ON_THE_DAY = {"reference_date": REFERENCE_DATE}

# --------------------------------------------------------------------------- #
# Scripted worlds: a FakeWorld that also knows durations
# --------------------------------------------------------------------------- #


def _iso_duration(value: timedelta) -> str:
    seconds = int(value.total_seconds())
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return (
        "PT"
        + (f"{hours}H" if hours else "")
        + (f"{minutes}M" if minutes else "")
        + (f"{secs}S" if secs or not seconds else "")
    )


class DurationWorld(FakeWorld):
    """A ``FakeWorld`` whose Clairic also answers ``duration`` fields.

    The shipped world answers text, list, number, date, time and datetime fields; a
    duration is a null there. Here a ``timedelta`` of an entity is answered as the
    ISO 8601 duration the library parses.
    """

    def _clairic_answer(
        self,
        spec: Any,
        element: Any,
        wire: str,
        view: Any,
        *,
        primary: bool,
    ) -> dict[str, Any]:
        if spec["type"] == "duration" and element is not None:
            present, value = self._value(element, wire, bleed=primary)
            if present and isinstance(value, timedelta):
                return clairic_answer("duration", _iso_duration(value))
        return super()._clairic_answer(spec, element, wire, view, primary=primary)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class Coverage(str, Enum):
    """The kind of insurance policy."""

    AUTO = "auto"
    HOME = "home"
    TRAVEL = "travel"


class InsuranceClaim(BaseModel):
    """An insurance claim file."""

    claim_number: str = Field(description="the claim number")
    claimant: str = Field(description="the claimant's full name")
    loss_date: date = Field(description="the date of the loss")
    claimed_amount: Decimal = Field(description="the amount claimed, in dollars")
    coverage: Coverage = Field(description="the kind of policy the claim is under")
    status: Literal["open", "approved", "denied"] = Field(
        description="the current status of the claim"
    )
    police_report_filed: bool = Field(description="whether a police report was filed")
    adjuster: str | None = Field(None, description="the adjuster handling the claim")
    deductible: Decimal | None = Field(
        None, description="the policy deductible, in dollars"
    )


class ClaimedItem(BaseModel):
    """One item listed on a contents claim."""

    code: str = Field(description="the item code")
    description: str = Field(description="what the item is")
    replacement_cost: Decimal = Field(description="the replacement cost in dollars")


class ContentsClaim(BaseModel):
    """A contents claim after a fire."""

    claim_number: str = Field(description="the claim number")
    items: Annotated[list[ClaimedItem], Many(identity="code")] = Field(
        description="the items claimed"
    )


class Claimant(BaseModel):
    """A person named in an accident report."""

    name: str = Field(description="the person's full name")
    vehicle: str | None = Field(None, description="the vehicle the person was in")
    injury: str | None = Field(None, description="the injury the person reports")


class FieldTripForm(BaseModel):
    """A school field-trip permission form."""

    student: str = Field(description="the student's full name")
    destination: str = Field(description="where the class is going")
    signed_on: date = Field(description="the date the parent signed the form")
    trip_date: date = Field(description="the date of the trip")
    departure_time: time = Field(description="the time the bus leaves school")
    swimming_permitted: bool = Field(
        description="whether the student may take part in swimming"
    )
    lunch: Literal["packed", "school", "none"] = Field(
        description="how the student's lunch is provided"
    )
    medical_form_filed_on: date | None = Field(
        None, description="the exact day the medical consent form was filed"
    )
    allergies: str | None = Field(None, description="the student's allergies")


class Pupil(BaseModel):
    """A pupil on a school trip."""

    name: str = Field(description="the pupil's full name")
    allergy: str | None = Field(None, description="what the pupil is allergic to")
    bus_group: str | None = Field(
        None, description="the colour of the pupil's group on the bus"
    )


class ServiceVisit(BaseModel):
    """A fleet vehicle's visit to the workshop."""

    plate: str = Field(description="the vehicle's number plate")
    dropped_off_on: date = Field(description="the date the vehicle came in")
    drop_off_time: time = Field(description="the time the vehicle came in")
    odometer_km: int = Field(description="the odometer reading in kilometres")
    labour_time: timedelta = Field(
        description="how long the labour took",
        json_schema_extra=s1_field(unit="minutes"),
    )
    invoice_total: Decimal = Field(description="the invoice total in dollars")
    tyres_rotated: bool = Field(description="whether the tyres were rotated")
    warranty_claim: bool | None = Field(
        None, description="whether the work is a warranty claim"
    )
    next_service_km: int | None = Field(
        None, description="the odometer reading of the next service"
    )


class FleetVehicle(BaseModel):
    """A vehicle of the company fleet."""

    plate: str = Field(description="the vehicle's number plate")
    make_model: str | None = Field(None, description="the make and model")
    odometer_km: int | None = Field(None, description="the odometer reading in km")
    status: Literal["in_service", "workshop", "retired"] = Field(
        "in_service", description="the vehicle's status"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="things worth remembering about the vehicle",
    )


class ServiceEntry(BaseModel):
    """One vehicle's entry in a workshop shorthand log."""

    plate: str = Field(description="the vehicle's number plate")
    make_model: str | None = Field(None, description="the make and model")
    labour: timedelta = Field(
        description="how long the labour took",
        json_schema_extra=s1_field(unit="minutes"),
    )
    invoice: Decimal = Field(description="the invoice amount in dollars")
    paid: bool | None = Field(None, description="whether the invoice is paid")


class DeliverySlot(BaseModel):
    """When and where the supplier delivers."""

    day: date = Field(description="the delivery date")
    dock: str = Field(description="the loading dock")


class OrderLine(BaseModel):
    """One product line of a supplier order."""

    name: str = Field(description="the product name")
    quantity: int = Field(description="how many units are ordered")
    unit_price: Decimal = Field(description="the price per unit in dollars")


class SupplierOrder(BaseModel):
    """A restaurant order placed with a supplier."""

    po_number: str = Field(description="the order number")
    supplier: str = Field(description="the supplier's name")
    delivery: DeliverySlot
    lines: Annotated[list[OrderLine], Many(identity="name")] = Field(
        description="the product lines ordered"
    )


class SupplierConfirmation(BaseModel):
    """A supplier's confirmation of a restaurant order."""

    reference: str = Field(description="the order reference")
    supplier: str = Field(description="the supplier's name")
    delivery_date: date = Field(description="the confirmed delivery date")
    total: Decimal = Field(description="the confirmed total in dollars")
    sea_bass_available: bool = Field(
        description="whether the supplier can deliver the sea bass"
    )


class Vendor(BaseModel):
    """A food supplier a restaurant buys ingredients from."""

    name: str = Field(description="the supplier's name")
    supplies: str | None = Field(None, description="what the supplier provides")


class BookLoan(BaseModel):
    """One book on loan."""

    barcode: str = Field(description="the book's barcode")
    title: str = Field(description="the book's title")
    due_date: date = Field(description="the date the book is due back")
    late_fee: Decimal | None = Field(None, description="the late fee in dollars")


class PatronLoans(BaseModel):
    """A library patron's loans."""

    patron: str = Field(description="the patron's full name")
    card_number: str = Field(description="the library card number")
    loans: Annotated[list[BookLoan], Many(identity="barcode")] = Field(
        description="the books on loan"
    )


class Patron(BaseModel):
    """A library patron."""

    name: str = Field(description="the patron's full name")
    email: str | None = Field(None, description="the patron's email address")
    branch: str | None = Field(None, description="the patron's home branch")


class AssetId(BaseModel):
    """How a piece of equipment is identified."""

    tag: str | None = Field(None, description="the asset tag")
    serial: str | None = Field(None, description="the manufacturer's serial number")


class ItAsset(BaseModel):
    """A piece of IT equipment on the inventory."""

    asset: AssetId = Field(default_factory=AssetId)
    make_model: str | None = Field(None, description="the make and model")
    assigned_to: str | None = Field(
        None, description="the person or place the asset is assigned to"
    )
    purchase_cost: Decimal | None = Field(
        None, description="the purchase cost in dollars"
    )


class SoftwareLicence(BaseModel):
    """A software licence on the register."""

    product: str = Field(description="the product name")
    vendor: str | None = Field(None, description="the vendor's name")
    seats: int | None = Field(None, description="the number of seats")


class Speaker(BaseModel):
    """A person giving a talk at a conference."""

    name: str = Field(description="the speaker's full name")
    affiliation: str | None = Field(
        None, description="the organisation the speaker works for"
    )
    topic: str | None = Field(None, description="what the speaker's talk is about")


class SessionEntry(BaseModel):
    """A speaker's entry in a conference timetable."""

    name: str = Field(description="the speaker's full name")
    affiliation: str | None = Field(
        None, description="the organisation the speaker works for"
    )
    topic: str | None = Field(None, description="what the speaker's talk is about")
    hall: str | None = Field(None, description="the hall the talk is in")


class SoilSample(BaseModel):
    """A soil sample logged by the lab."""

    sample_id: str = Field(description="the sample id")
    site: str = Field(description="the name of the sampling site")
    collected_on: date = Field(description="the date the sample was collected")
    ph: float = Field(description="the soil pH")
    moisture_percent: float = Field(description="the moisture content in percent")
    depth_class: Literal["topsoil", "subsoil", "bedrock"] = Field(
        description="the layer the sample came from"
    )
    contaminated: bool = Field(description="whether contamination was found")
    analyst: str | None = Field(None, description="the analyst who ran the sample")
    lead_ppm: float | None = Field(None, description="the lead content in ppm")


class CoreLayer(BaseModel):
    """One layer of a soil core."""

    code: str = Field(description="the layer code")
    texture: str = Field(description="the soil texture")
    thickness_cm: float = Field(description="the thickness in centimetres")


class CoreLog(BaseModel):
    """A soil core log."""

    core_id: str = Field(description="the core id")
    layers: Annotated[list[CoreLayer], Many(identity="code")] = Field(
        description="the layers of the core"
    )


class FieldSample(BaseModel):
    """A soil sample noted in the field."""

    sample_id: str = Field(description="the sample id")
    location: str = Field(description="where the sample was taken")
    ph: float = Field(description="the soil pH")


class CharterBooking(BaseModel):
    """A boat charter booking."""

    booking_ref: str = Field(description="the booking reference")
    skipper: str = Field(description="the skipper's name")
    vessel: str = Field(description="the name of the boat")
    start_date: date = Field(description="the first day of the charter")
    guests: int = Field(description="how many guests to book for")


class CharterRequest(BaseModel):
    """A request to charter a boat, taken over chat."""

    customer_name: str = Field(description="the customer's full name")
    vessel: str = Field(description="the name of the boat")
    start_date: date = Field(description="the first day of the charter")
    party_size: int = Field(description="how many people are in the party")
    deposit: Decimal | None = Field(None, description="the deposit in dollars")
    fuel_included: bool = Field(description="whether fuel is included in the rate")


class Yacht(BaseModel):
    """A charter yacht of a fleet."""

    name: str = Field(description="the yacht's name")
    length_m: float | None = Field(None, description="the length in metres")
    cabins: int | None = Field(None, description="the number of cabins")
    berth: int | None = Field(None, description="the berth number")


class MeterRead(BaseModel):
    """One electricity meter reading."""

    meter_number: str = Field(description="the meter number")
    read_on: date = Field(description="the date the meter was read")
    read_time: time = Field(description="the time of day the meter was read")
    reading_kwh: float = Field(description="the reading in kWh")
    billing_period: str = Field(description="the billing period the reading covers")
    estimated: bool = Field(description="whether the reading is an estimate")
    tariff: Literal["standard", "night", "solar"] = Field(
        description="the tariff the household is on"
    )
    rate_per_kwh: Decimal | None = Field(
        None, description="the price per kWh in dollars"
    )
    peak_demand_kw: float | None = Field(None, description="the peak demand in kW")


class WaterMeterCard(BaseModel):
    """A water meter card."""

    account_holder: str = Field(description="the account holder's full name")
    service_address: str = Field(description="the service address")
    meter_number: str = Field(description="the meter number")
    last_read: date = Field(description="the date of the last reading")
    reading_m3: float = Field(description="the last reading in cubic metres")
    next_read_due: date | None = Field(
        None, description="the date the next reading is due"
    )


class MeterRow(BaseModel):
    """One meter of a reading round."""

    meter_no: str = Field(description="the meter number")
    reading_kwh: float = Field(description="the reading in kWh")
    note: str | None = Field(None, description="a note about access or condition")


class ReadingRound(BaseModel):
    """A meter-reading round on one route."""

    route: str = Field(description="the route code")
    meters: Annotated[list[MeterRow], Many(identity="meter_no")] = Field(
        description="the meters read"
    )


# --------------------------------------------------------------------------- #
# Insurance claims
# --------------------------------------------------------------------------- #


def claim_scalars() -> Case:
    text = (
        "Claim CL-77310 was opened for Marisol Quenby after a hailstorm damaged her "
        "car on 14 March 2026. She is claiming $4,820.50 for the repairs and has not "
        "filed a police report. The claim is under her auto policy and is still open."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the InsuranceClaim",
            schema=InsuranceClaim,
            mentions=["CL-77310"],
            fields={
                "claim_number": "CL-77310",
                "claimant": "Marisol Quenby",
                "loss_date": date(2026, 3, 14),
                "claimed_amount": 4820.5,
                "coverage": "auto",
                "status": "open",
                "police_report_filed": False,
            },
        )
        return w

    return Case(
        id="claim_scalars",
        text=text,
        schemas=lambda: [One(InsuranceClaim)],
        expect_ops=[Op("insert", who="InsuranceClaim")],
        expect_values=[
            Val("InsuranceClaim", "/claim_number", "CL-77310"),
            Val("InsuranceClaim", "/claimant", "Marisol Quenby"),
            Val("InsuranceClaim", "/loss_date", "2026-03-14"),
            Val("InsuranceClaim", "/claimed_amount", Decimal("4820.50")),
            Val("InsuranceClaim", "/coverage", "auto"),
            Val("InsuranceClaim", "/status", "open"),
            Val("InsuranceClaim", "/police_report_filed", False),
            # never mentioned: stays empty (an unstated optional is not invented)
            Val("InsuranceClaim", "/adjuster", None),
            Val("InsuranceClaim", "/deductible", None),
        ],
        tags=_tags("insurance insert scalar decimal date bool literal enum optional"),
        world=world,
        note="One insert with an absolute date, Decimal money, an Enum, a Literal, a "
        "negated bool and two optionals the text never mentions.",
    )


CLAIM_TABLE = """\
Claim CL-80214: contents lost in the fire at the Aldermere flat.

| Code | Item | Replacement cost |
| --- | --- | --- |
| K-101 | Espresso machine | $389.00 |
| K-102 | Stand mixer | $449.99 |
| L-230 | Leather armchair | $1,250.00 |
"""


def claim_items_table() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(CLAIM_TABLE)
        w.entity(
            "the ContentsClaim",
            schema=ContentsClaim,
            mentions=["CL-80214"],
            fields={"claim_number": "CL-80214"},
        )
        for code, item, cost in (
            ("K-101", "Espresso machine", 389),
            ("K-102", "Stand mixer", 449.99),
            ("L-230", "Leather armchair", 1250),
        ):
            w.entity(
                code,
                schema=ClaimedItem,
                parent="ContentsClaim",
                mentions=[code],
                fields={"code": code, "description": item, "replacement_cost": cost},
            )
        return w

    return Case(
        id="claim_items_table",
        text=CLAIM_TABLE,
        schemas=lambda: [One(ContentsClaim)],
        expect_ops=[Op("insert", who="ContentsClaim")],
        expect_values=[
            Val("ContentsClaim", "/claim_number", "CL-80214"),
            Val("ContentsClaim", "/items/0/code", "K-101"),
            Val("ContentsClaim", "/items/0/description", "Espresso machine"),
            Val("ContentsClaim", "/items/0/replacement_cost", Decimal("389.00")),
            Val("ContentsClaim", "/items/1/code", "K-102"),
            Val("ContentsClaim", "/items/1/description", "Stand mixer"),
            Val("ContentsClaim", "/items/1/replacement_cost", Decimal("449.99")),
            Val("ContentsClaim", "/items/2/code", "L-230"),
            Val("ContentsClaim", "/items/2/description", "Leather armchair"),
            Val("ContentsClaim", "/items/2/replacement_cost", Decimal("1250.00")),
        ],
        must_not=[MustNot.value_at("ContentsClaim", "/items/3")],
        tags=_tags("insurance insert nested table decimal"),
        world=world,
        note="A markdown table: one row per item, each item named by its code; "
        "three rows in, exactly three items out, each with its own cost.",
    )


CLAIMANTS_TEXT = (
    "Report on the collision at Fenwick Street. Imogen Raskov was driving a green "
    "Kestrel hatchback and says her wrist hurts. Dmitri Alvarez was riding a courier "
    "bike and was not hurt."
)


def claimants_bleed() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(CLAIMANTS_TEXT)
        w.entity(
            "Imogen Raskov",
            schema=Claimant,
            mentions=["Imogen Raskov"],
            fields={
                "name": "Imogen Raskov",
                "vehicle": "green Kestrel hatchback",
                "injury": "wrist hurts",
            },
        )
        w.entity(
            "Dmitri Alvarez",
            schema=Claimant,
            mentions=["Dmitri Alvarez"],
            fields={"name": "Dmitri Alvarez", "vehicle": "courier bike"},
        )
        # Imogen's injury is offered for Dmitri, who was not hurt
        w.bleed("Dmitri Alvarez", "injury", "wrist hurts")
        return w

    return Case(
        id="claimants_bleed",
        text=CLAIMANTS_TEXT,
        schemas=lambda: [Many(Claimant, identity="name")],
        expect_ops=[Op("insert", who="Imogen"), Op("insert", who="Dmitri")],
        expect_values=[
            Val("Imogen", "/vehicle", contains="Kestrel"),
            Val("Imogen", "/injury", contains="wrist"),
            Val("Dmitri", "/vehicle", contains="courier bike"),
            Val("Dmitri", "/injury", None),
        ],
        must_not=[
            MustNot.value_at("Dmitri", "/injury", contains="wrist"),
            MustNot.value_at("Dmitri", "/vehicle", contains="Kestrel"),
        ],
        tags=_tags("insurance insert many bleed"),
        world=world,
        note="Two people in consecutive sentences: the injury of the first is not "
        "the second's, who was not hurt (null).",
    )


# --------------------------------------------------------------------------- #
# School field-trip permission forms
# --------------------------------------------------------------------------- #


def trip_form_dates() -> Case:
    text = (
        "Permission form for Theo Nakamura-Bell, Year 5. The class is visiting the "
        "Harbour Science Centre. The form was signed yesterday and the trip is in two "
        "weeks. The bus leaves school at 8:15 am. Theo may take part in the swimming "
        "session. He will bring a packed lunch. His medical consent form has been on "
        "file since June 2025."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the FieldTripForm",
            schema=FieldTripForm,
            mentions=["Theo Nakamura-Bell"],
            fields={
                "student": "Theo Nakamura-Bell",
                "destination": "Harbour Science Centre",
                "signed_on": date(2026, 9, 15),
                "trip_date": date(2026, 9, 30),
                "departure_time": time(8, 15),
                "swimming_permitted": True,
                "lunch": "packed",
                # "June 2025" has no day: the library never guesses one, so the
                # exact day of filing is null (docs/DESIGN.md, Dates)
            },
        )
        return w

    return Case(
        id="trip_form_dates",
        text=text,
        schemas=lambda: [One(FieldTripForm)],
        options=dict(ON_THE_DAY),
        expect_ops=[Op("insert", who="FieldTripForm")],
        expect_values=[
            Val("FieldTripForm", "/student", "Theo Nakamura-Bell"),
            Val("FieldTripForm", "/destination", contains="Harbour Science Centre"),
            Val("FieldTripForm", "/signed_on", "2026-09-15"),
            Val("FieldTripForm", "/trip_date", "2026-09-30"),
            Val("FieldTripForm", "/departure_time", "08:15:00"),
            Val("FieldTripForm", "/swimming_permitted", True),
            Val("FieldTripForm", "/lunch", "packed"),
            Val("FieldTripForm", "/medical_form_filed_on", None),
            Val("FieldTripForm", "/allergies", None),
        ],
        tags=_tags(
            "school insert scalar date relative-date month-year time bool literal optional"
        ),
        world=world,
        note="Relative dates against 2026-09-16 (yesterday, in two weeks), a clock "
        "time, a Literal and a bool. Documented limit: 'June 2025' is a month and a "
        "year with no day, which the library does not resolve to a date, so the "
        "exact-day field is null.",
    )


PUPILS_TEXT = (
    "Bus 2 list for the Harbour Science Centre trip. Keanu Fairweather has returned "
    "his form and is allergic to peanuts. Ravi Bhatt's form is also in, and he sits "
    "in group Blue. Mrs Okonkwo is the chaperone for group Blue. Keanu is in group Red."
)


def trip_pupils_many() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(PUPILS_TEXT)
        w.entity(
            "Keanu Fairweather",
            schema=Pupil,
            mentions=["Keanu Fairweather", "Keanu"],
            fields={
                "name": "Keanu Fairweather",
                "allergy": "peanuts",
                "bus_group": "Red",
            },
        )
        w.entity(
            "Ravi Bhatt",
            schema=Pupil,
            mentions=["Ravi Bhatt"],
            fields={"name": "Ravi Bhatt", "bus_group": "Blue"},
        )
        w.non_entity("Mrs Okonkwo", schema=Pupil, mentions=["Mrs Okonkwo"])
        # Keanu's allergy is offered for Ravi, whose sentence follows
        w.bleed("Ravi Bhatt", "allergy", "peanuts")
        return w

    return Case(
        id="trip_pupils_many",
        text=PUPILS_TEXT,
        schemas=lambda: [Many(Pupil, identity="name")],
        expect_ops=[Op("insert", who="Keanu"), Op("insert", who="Ravi")],
        expect_values=[
            Val("Keanu", "/name", contains="Keanu Fairweather"),
            Val("Keanu", "/allergy", contains="peanut"),
            Val("Keanu", "/bus_group", "Red"),
            Val("Ravi", "/name", "Ravi Bhatt"),
            Val("Ravi", "/bus_group", "Blue"),
            Val("Ravi", "/allergy", None),
        ],
        must_not=[
            MustNot.insert("Okonkwo"),
            MustNot.value_at("Ravi", "/allergy", contains="peanut"),
            MustNot.value_at("Keanu", "/bus_group", contains="Blue"),
        ],
        tags=_tags("school insert many coref negative bleed"),
        world=world,
        note="Short form (Keanu), a pronoun, a chaperone who is not a pupil and a "
        "fact (the allergy) that sits one sentence before the next pupil.",
    )


def trip_time_correction() -> Case:
    text = (
        "Correction to the Millbrook Farm notice for Hana Lindqvist: the coach now "
        "leaves at 8:45 am, not 8:30 am as first printed. Everything else on the form "
        "stands."
    )
    existing = [
        (
            "form-31",
            "FieldTripForm",
            {
                "student": "Hana Lindqvist",
                "destination": "Millbrook Farm",
                "signed_on": "2026-09-01",
                "trip_date": "2026-10-08",
                "departure_time": "08:30:00",
                "swimming_permitted": False,
                "lunch": "school",
                "medical_form_filed_on": None,
                "allergies": None,
            },
        )
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the FieldTripForm",
            schema=FieldTripForm,
            mentions=["Hana Lindqvist"],
            fields={
                "student": "Hana Lindqvist",
                "destination": "Millbrook Farm",
                "departure_time": time(8, 45),
            },
        )
        return w

    return Case(
        id="trip_time_correction",
        text=text,
        schemas=lambda: [One(FieldTripForm)],
        existing=existing,
        expect_ops=[Op("replace", "/departure_time", "08:45:00", who="form-31")],
        expect_values=[
            Val("form-31", "/departure_time", "08:45:00"),
            Val("form-31", "/destination", "Millbrook Farm"),
            Val("form-31", "/trip_date", "2026-10-08"),
            Val("form-31", "/lunch", "school"),
        ],
        must_not=[
            MustNot.op("replace", "/destination"),
            MustNot.op("replace", "/trip_date"),
            MustNot.op("replace", "/student"),
            MustNot.op("remove"),
        ],
        tags=_tags("school update correction time"),
        world=world,
        note="A correction that names the old time too ('not 8:30 am'): the new "
        "time replaces it and nothing else on the form moves.",
    )


# --------------------------------------------------------------------------- #
# Fleet vehicle service logs
# --------------------------------------------------------------------------- #


def service_duration() -> Case:
    text = (
        "Service log, plate KDR-4471. The van came in three days ago at 9:20 am with "
        "84,210 km on the clock. Labour took 1 hour 45 minutes, the tyres were "
        "rotated, and the invoice came to $612.40."
    )

    def world() -> FakeWorld:
        w = DurationWorld(text)
        w.entity(
            "the ServiceVisit",
            schema=ServiceVisit,
            mentions=["KDR-4471"],
            fields={
                "plate": "KDR-4471",
                "dropped_off_on": date(2026, 9, 13),
                "drop_off_time": time(9, 20),
                "odometer_km": 84210,
                "labour_time": timedelta(minutes=105),
                "invoice_total": 612.4,
                "tyres_rotated": True,
            },
        )
        return w

    return Case(
        id="service_duration",
        text=text,
        schemas=lambda: [One(ServiceVisit)],
        options=dict(ON_THE_DAY),
        expect_ops=[Op("insert", who="ServiceVisit")],
        expect_values=[
            Val("ServiceVisit", "/plate", "KDR-4471"),
            Val("ServiceVisit", "/dropped_off_on", "2026-09-13"),
            Val("ServiceVisit", "/drop_off_time", "09:20:00"),
            Val("ServiceVisit", "/odometer_km", 84210),
            Val("ServiceVisit", "/labour_time", timedelta(minutes=105)),
            Val("ServiceVisit", "/invoice_total", Decimal("612.40")),
            Val("ServiceVisit", "/tyres_rotated", True),
            Val("ServiceVisit", "/warranty_claim", None),
            Val("ServiceVisit", "/next_service_km", None),
        ],
        tags=_tags(
            "fleet insert scalar duration relative-date time decimal bool optional"
        ),
        world=world,
        note="A duration in words (1 hour 45 minutes), a relative date (three days "
        "ago), a clock time, a thousands-separated integer and an Optional[bool] "
        "that the log never mentions.",
    )


FLEET_NOTE = (
    "Van KDR-4471 now shows 91,300 km on the clock. The driver says the rear brakes "
    "squeal when it is wet."
)
SQUEAL = "The driver says the rear brakes squeal when it is wet."


def vehicle_notes_update() -> Case:
    existing = [
        (
            "v-1",
            "FleetVehicle",
            {
                "plate": "KDR-4471",
                "make_model": "Ford Transit",
                "odometer_km": 84210,
                "status": "in_service",
                "notes": ["Fitted new wipers in July."],
            },
        ),
        (
            "v-2",
            "FleetVehicle",
            {
                "plate": "KDR-5120",
                "make_model": "VW Caddy",
                "odometer_km": 40115,
                "status": "workshop",
                "notes": [],
            },
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(FLEET_NOTE)
        w.entity(
            "KDR-4471",
            schema=FleetVehicle,
            mentions=["KDR-4471"],
            fields={"plate": "KDR-4471", "odometer_km": 91300, "notes": [SQUEAL]},
            matches={"plate": "KDR-4471"},
        )
        return w

    return Case(
        id="vehicle_notes_update",
        text=FLEET_NOTE,
        schemas=lambda: [Many(FleetVehicle, identity="plate")],
        existing=existing,
        options={"enable_inserts": True},
        expect_ops=[
            Op("replace", "/odometer_km", 91300, who="v-1"),
            Op("add", "/notes/-", contains="rear brakes squeal", who="v-1"),
        ],
        expect_values=[
            Val("v-1", "/odometer_km", 91300),
            Val("v-1", "/notes", contains="Fitted new wipers in July"),
            Val("v-1", "/notes", contains="rear brakes squeal"),
            Val("v-1", "/make_model", "Ford Transit"),
            Val("v-1", "/status", "in_service"),
        ],
        must_not=[
            MustNot.op(who="v-2"),
            MustNot.op("replace", "/notes"),
            MustNot.op("replace", "/plate"),
            MustNot.op("remove"),
            MustNot.insert(),
        ],
        tags=_tags("fleet update notes many"),
        world=world,
        note="Only what the text states changes: the odometer is replaced, the "
        "driver's remark is appended to the notes (the old note stays), the make, "
        "the status and the other van are left alone.",
    )


SHORTHAND = (
    "Veh KDR-4471 (Ford Transit): oil+filter chgd, R/F tyre rpl'd, approx 2h labour, "
    "inv $389.00. Veh KDR-5120: brake pads F&R, pls note 45 min labour, "
    "inv $540.25 - paid."
)


def service_shorthand() -> Case:
    def world() -> FakeWorld:
        w = DurationWorld(SHORTHAND)
        w.entity(
            "KDR-4471",
            schema=ServiceEntry,
            mentions=["KDR-4471"],
            fields={
                "plate": "KDR-4471",
                "make_model": "Ford Transit",
                "labour": timedelta(hours=2),
                "invoice": 389,
            },
        )
        w.entity(
            "KDR-5120",
            schema=ServiceEntry,
            mentions=["KDR-5120"],
            fields={
                "plate": "KDR-5120",
                "labour": timedelta(minutes=45),
                "invoice": 540.25,
                "paid": True,
            },
        )
        # the first van's model is offered for the second one, which has none
        w.bleed("KDR-5120", "make_model", "Ford Transit")
        return w

    return Case(
        id="service_shorthand",
        text=SHORTHAND,
        schemas=lambda: [Many(ServiceEntry, identity="plate")],
        expect_ops=[Op("insert", who="KDR-4471"), Op("insert", who="KDR-5120")],
        expect_values=[
            Val("KDR-4471", "/make_model", "Ford Transit"),
            Val("KDR-4471", "/labour", timedelta(hours=2)),
            Val("KDR-4471", "/invoice", Decimal("389.00")),
            Val("KDR-4471", "/paid", None),
            Val("KDR-5120", "/make_model", None),
            Val("KDR-5120", "/labour", timedelta(minutes=45)),
            Val("KDR-5120", "/invoice", Decimal("540.25")),
            Val("KDR-5120", "/paid", True),
        ],
        must_not=[MustNot.value_at("KDR-5120", "/make_model", contains="Transit")],
        tags=_tags("fleet insert many abbreviations duration bleed"),
        world=world,
        note="Workshop shorthand (chgd, rpl'd, inv, 2h, 45 min). Only the second van "
        "says it is paid (the first one's flag is unstated, null) and only the first "
        "names its model.",
    )


# --------------------------------------------------------------------------- #
# Restaurant supplier orders
# --------------------------------------------------------------------------- #

ORDER_TEXT = (
    "Order PO-6120 to Greenmarket Produce: deliver on 24 September 2026 at the "
    "North Dock. We need 12 crates of lemons at $18.50 per crate and 5 sacks of "
    "basmati rice at $42.00 per sack."
)


def supplier_order_lines() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(ORDER_TEXT)
        w.entity(
            "the SupplierOrder",
            schema=SupplierOrder,
            mentions=["PO-6120"],
            fields={
                "po_number": "PO-6120",
                "supplier": "Greenmarket Produce",
                "delivery": {"day": date(2026, 9, 24), "dock": "North Dock"},
            },
        )
        for name, quantity, price in (
            ("lemons", 12, 18.5),
            ("basmati rice", 5, 42),
        ):
            w.entity(
                name,
                schema=OrderLine,
                parent="SupplierOrder",
                mentions=[name],
                fields={"name": name, "quantity": quantity, "unit_price": price},
            )
        return w

    return Case(
        id="supplier_order_lines",
        text=ORDER_TEXT,
        schemas=lambda: [One(SupplierOrder)],
        expect_ops=[Op("insert", who="SupplierOrder")],
        expect_values=[
            Val("SupplierOrder", "/po_number", "PO-6120"),
            Val("SupplierOrder", "/supplier", "Greenmarket Produce"),
            Val("SupplierOrder", "/delivery/day", "2026-09-24"),
            Val("SupplierOrder", "/delivery/dock", contains="North Dock"),
            Val("SupplierOrder", "/lines/0/name", "lemons"),
            Val("SupplierOrder", "/lines/0/quantity", 12),
            Val("SupplierOrder", "/lines/0/unit_price", Decimal("18.50")),
            Val("SupplierOrder", "/lines/1/name", "basmati rice"),
            Val("SupplierOrder", "/lines/1/quantity", 5),
            Val("SupplierOrder", "/lines/1/unit_price", Decimal("42.00")),
        ],
        must_not=[MustNot.value_at("SupplierOrder", "/lines/2")],
        tags=_tags("supplier insert nested decimal date"),
        world=world,
        note="A required sub-model (the delivery slot) and a list of product lines in "
        "prose; each line keeps its own quantity and price.",
    )


THREAD_TEXT = """\
From: Tamsin Whitby <tamsin@marlowfish.example>
To: Kitchen Orders <orders@lanternhouse.example>
Subject: Re: Re: Friday fish order FO-3308

Hi Chef,

Confirmed for Friday 25 September 2026. The total comes to $1,184.60 with the extra halibut.
I'm afraid the sea bass is out, so I have put in more halibut instead.

Best,
Tamsin
Marlow Fish Merchants

> On 14 Sep, Kitchen Orders wrote:
> Please send 20 kg of halibut and 15 kg of sea bass for Friday.
> Order ref FO-3308. Thanks, Chef
"""


def supplier_email_thread() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(THREAD_TEXT)
        w.entity(
            "the SupplierConfirmation",
            schema=SupplierConfirmation,
            mentions=["FO-3308"],
            fields={
                "reference": "FO-3308",
                "supplier": "Marlow Fish Merchants",
                "delivery_date": date(2026, 9, 25),
                "total": 1184.6,
                "sea_bass_available": False,
            },
        )
        return w

    return Case(
        id="supplier_email_thread",
        text=THREAD_TEXT,
        schemas=lambda: [One(SupplierConfirmation)],
        expect_ops=[Op("insert", who="SupplierConfirmation")],
        expect_values=[
            Val("SupplierConfirmation", "/reference", "FO-3308"),
            Val("SupplierConfirmation", "/supplier", "Marlow Fish Merchants"),
            Val("SupplierConfirmation", "/delivery_date", "2026-09-25"),
            Val("SupplierConfirmation", "/total", Decimal("1184.60")),
            Val("SupplierConfirmation", "/sea_bass_available", False),
        ],
        tags=_tags("supplier insert email scalar decimal date bool"),
        world=world,
        note="An email reply with headers, a signature and a quoted original: the "
        "confirmation's own figures count, the quoted request ('15 kg of sea bass', "
        "'14 Sep') does not, and 'the sea bass is out' makes the bool false.",
    )


VENDORS_TEXT = (
    "Our regular suppliers are Marlow Fish Merchants for seafood and Greenmarket "
    "Produce for vegetables. Chef Antonia Reyes calls Greenmarket every morning. The "
    "walk-in fridge is serviced by CoolRight, a repair firm and not a food supplier."
)


def vendors_not_people() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(VENDORS_TEXT)
        w.entity(
            "Marlow Fish Merchants",
            schema=Vendor,
            mentions=["Marlow Fish Merchants"],
            fields={"name": "Marlow Fish Merchants", "supplies": "seafood"},
        )
        w.entity(
            "Greenmarket Produce",
            schema=Vendor,
            mentions=["Greenmarket Produce", "Greenmarket"],
            fields={"name": "Greenmarket Produce", "supplies": "vegetables"},
        )
        w.non_entity("Antonia Reyes", schema=Vendor, mentions=["Antonia Reyes"])
        w.non_entity("CoolRight", schema=Vendor, mentions=["CoolRight"])
        return w

    return Case(
        id="vendors_not_people",
        text=VENDORS_TEXT,
        schemas=lambda: [Many(Vendor, identity="name")],
        expect_ops=[Op("insert", who="Marlow"), Op("insert", who="Greenmarket")],
        expect_values=[
            Val("Marlow", "/supplies", "seafood"),
            Val("Greenmarket", "/name", contains="Greenmarket"),
            Val("Greenmarket", "/supplies", "vegetables"),
        ],
        must_not=[MustNot.insert("Antonia"), MustNot.insert("CoolRight")],
        tags=_tags("supplier insert many negative coref"),
        world=world,
        note="A chef (a person) and a repair firm (a company, but not a food "
        "supplier) look like entities and are not; 'Greenmarket' is a short form.",
    )


# --------------------------------------------------------------------------- #
# Library book loans
# --------------------------------------------------------------------------- #

LOANS_TEXT = """\
Loans for patron Ottilie Brandvold, card 20481.

| Barcode | Title | Due | Late fee |
| --- | --- | --- | --- |
| 3100422 | The Salt Road | Oct 2, 2026 | |
| 3100987 | Field Guide to Lichens | Sep 12, 2026 | $1.40 |
"""


def loans_table() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(LOANS_TEXT)
        w.entity(
            "the PatronLoans",
            schema=PatronLoans,
            mentions=["Ottilie Brandvold"],
            fields={"patron": "Ottilie Brandvold", "card_number": "20481"},
        )
        w.entity(
            "3100422",
            schema=BookLoan,
            parent="PatronLoans",
            mentions=["3100422"],
            fields={
                "barcode": "3100422",
                "title": "The Salt Road",
                "due_date": date(2026, 10, 2),
            },
        )
        w.entity(
            "3100987",
            schema=BookLoan,
            parent="PatronLoans",
            mentions=["3100987"],
            fields={
                "barcode": "3100987",
                "title": "Field Guide to Lichens",
                "due_date": date(2026, 9, 12),
                "late_fee": 1.4,
            },
        )
        return w

    return Case(
        id="loans_table",
        text=LOANS_TEXT,
        schemas=lambda: [One(PatronLoans)],
        expect_ops=[Op("insert", who="PatronLoans")],
        expect_values=[
            Val("PatronLoans", "/patron", "Ottilie Brandvold"),
            Val("PatronLoans", "/card_number", "20481"),
            Val("PatronLoans", "/loans/0/barcode", "3100422"),
            Val("PatronLoans", "/loans/0/title", "The Salt Road"),
            Val("PatronLoans", "/loans/0/due_date", "2026-10-02"),
            Val("PatronLoans", "/loans/0/late_fee", None),
            Val("PatronLoans", "/loans/1/barcode", "3100987"),
            Val("PatronLoans", "/loans/1/title", "Field Guide to Lichens"),
            Val("PatronLoans", "/loans/1/due_date", "2026-09-12"),
            Val("PatronLoans", "/loans/1/late_fee", Decimal("1.40")),
        ],
        must_not=[MustNot.value_at("PatronLoans", "/loans/2")],
        tags=_tags("library insert nested table date decimal"),
        world=world,
        note="Rows identified by numeric barcodes; an empty late-fee cell is null, "
        "not a value borrowed from the next row.",
    )


def patron_rename() -> Case:
    text = (
        "Lena Voss was married last month, so please change the name on her library "
        "account to Lena Brandt."
    )
    existing = [
        (
            "p-14",
            "Patron",
            {
                "name": "Lena Voss",
                "email": "lena.voss@mailbox.example",
                "branch": "Eastside",
            },
        ),
        (
            "p-15",
            "Patron",
            {"name": "Jorge Salcedo", "email": None, "branch": "Central"},
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Lena Voss",
            schema=Patron,
            mentions=["Lena Voss", "Lena Brandt"],
            fields={"name": "Lena Brandt"},
            matches={"name": "Lena Voss"},
        )
        return w

    return Case(
        id="patron_rename",
        text=text,
        schemas=lambda: [Many(Patron, identity="name")],
        existing=existing,
        options={"enable_inserts": True, "enable_deletes": True},
        expect_ops=[Op("replace", "/name", "Lena Brandt", who="p-14")],
        expect_values=[
            Val("p-14", "/name", "Lena Brandt"),
            Val("p-14", "/email", "lena.voss@mailbox.example"),
            Val("p-14", "/branch", "Eastside"),
        ],
        must_not=[
            MustNot.insert(),
            MustNot.delete(),
            MustNot.op(who="p-15"),
            MustNot.op(path="/email"),
        ],
        tags=_tags("library update many rename"),
        world=world,
        note="A married-name change: the patron is the same record (the old surname "
        "is named too), the name is replaced, and no second patron is inserted.",
    )


REMOVE_AND_CHANGE = (
    "Please remove Tomas Eklund from the patron list, and change his email address "
    "to tomas.e@mailbox.example."
)


def patron_remove_and_change() -> Case:
    existing = [
        ("p-21", "Patron", {"name": "Maren Koh", "email": None, "branch": "Central"}),
        (
            "p-22",
            "Patron",
            {
                "name": "Tomas Eklund",
                "email": "tomas.eklund@oldmail.example",
                "branch": "Eastside",
            },
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(REMOVE_AND_CHANGE)
        w.entity(
            "Tomas Eklund",
            schema=Patron,
            mentions=["Tomas Eklund"],
            fields={"name": "Tomas Eklund", "email": "tomas.e@mailbox.example"},
            matches={"name": "Tomas Eklund"},
        )
        w.delete("p-22", 0.97)
        return w

    return Case(
        id="patron_remove_and_change",
        text=REMOVE_AND_CHANGE,
        schemas=lambda: [Many(Patron, identity="name")],
        existing=existing,
        options={"enable_inserts": True, "enable_deletes": True},
        # documented: an input that both changes and removes a record is a
        # conflict, and neither is applied (docs/DESIGN.md, step 9)
        expect_ops=[],
        expect_skipped=[("conflict_update_delete", "p-22")],
        must_not=[
            MustNot.delete(),
            MustNot.op("replace", "/email"),
            MustNot.op(who="p-21"),
            MustNot.insert(),
        ],
        tags=_tags("library update many delete conflict"),
        world=world,
        note="Documented behaviour: asking to remove a record and to change it in "
        "the same input is a conflict_update_delete; the record is neither deleted "
        "nor updated, and the result says why.",
    )


# --------------------------------------------------------------------------- #
# IT asset inventory
# --------------------------------------------------------------------------- #

ASSET_SHEET = """\
Asset record: Lumen X14 laptop
Tag: AT-30417
Serial: 9F2K-7731
Assigned to: Bea Kowalczyk
Cost: $1,180.00

Asset record: Harbor 27 monitor
Tag: AT-30422
Serial: HM27-0058
Assigned to: Stock room
"""


def asset_dotted_insert() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(ASSET_SHEET)
        w.entity(
            "Lumen X14 laptop",
            schema=ItAsset,
            mentions=["Lumen X14 laptop"],
            fields={
                "asset.tag": "AT-30417",
                "asset.serial": "9F2K-7731",
                "make_model": "Lumen X14 laptop",
                "assigned_to": "Bea Kowalczyk",
                "purchase_cost": 1180,
            },
        )
        w.entity(
            "Harbor 27 monitor",
            schema=ItAsset,
            mentions=["Harbor 27 monitor"],
            fields={
                "asset.tag": "AT-30422",
                "asset.serial": "HM27-0058",
                "make_model": "Harbor 27 monitor",
                "assigned_to": "Stock room",
            },
        )
        return w

    return Case(
        id="asset_dotted_insert",
        text=ASSET_SHEET,
        schemas=lambda: [Many(ItAsset, identity="asset.tag")],
        # documented: a dotted identity is not asked for by value, so the records
        # are named by the words the sheet uses for them (their model)
        expect_ops=[Op("insert", who="Lumen X14"), Op("insert", who="Harbor 27")],
        expect_values=[
            Val("Lumen X14", "/asset/tag", "AT-30417"),
            Val("Lumen X14", "/asset/serial", "9F2K-7731"),
            Val("Lumen X14", "/make_model", contains="Lumen X14"),
            Val("Lumen X14", "/assigned_to", "Bea Kowalczyk"),
            Val("Lumen X14", "/purchase_cost", Decimal("1180.00")),
            Val("Harbor 27", "/asset/tag", "AT-30422"),
            Val("Harbor 27", "/asset/serial", "HM27-0058"),
            Val("Harbor 27", "/assigned_to", "Stock room"),
            Val("Harbor 27", "/purchase_cost", None),
        ],
        must_not=[
            MustNot.value_at("Harbor 27", "/asset/tag", "AT-30417"),
            MustNot.value_at("Lumen X14", "/asset/tag", "AT-30422"),
            MustNot.value_at("Harbor 27", "/assigned_to", contains="Bea"),
        ],
        tags=_tags("assets insert many dotted-identity nested decimal"),
        world=world,
        note="Many(ItAsset, identity='asset.tag'): the tag sits one level down in a "
        "sub-model. Documented: such records are found by the words that name them "
        "(the model), not by the tag itself.",
    )


ASSET_CHANGE = (
    "Inventory change: asset AT-30417 has been reassigned to Cora Wan. Asset "
    "AT-30422 is unchanged."
)


def asset_dotted_update() -> Case:
    existing = [
        (
            "a-417",
            "ItAsset",
            {
                "asset": {"tag": "AT-30417", "serial": "9F2K-7731"},
                "make_model": "Lumen X14 laptop",
                "assigned_to": "Bea Kowalczyk",
                "purchase_cost": "1180.00",
            },
        ),
        (
            "a-422",
            "ItAsset",
            {
                "asset": {"tag": "AT-30422", "serial": "HM27-0058"},
                "make_model": "Harbor 27 monitor",
                "assigned_to": "Stock room",
                "purchase_cost": None,
            },
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(ASSET_CHANGE)
        w.entity(
            "AT-30417",
            schema=ItAsset,
            mentions=["AT-30417"],
            fields={"asset.tag": "AT-30417", "assigned_to": "Cora Wan"},
            matches="AT-30417",
        )
        w.entity(
            "AT-30422",
            schema=ItAsset,
            mentions=["AT-30422"],
            fields={"asset.tag": "AT-30422"},
            matches="AT-30422",
        )
        return w

    return Case(
        id="asset_dotted_update",
        text=ASSET_CHANGE,
        schemas=lambda: [Many(ItAsset, identity="asset.tag")],
        existing=existing,
        options={"enable_inserts": True},
        expect_ops=[Op("replace", "/assigned_to", "Cora Wan", who="a-417")],
        expect_values=[
            Val("a-417", "/assigned_to", "Cora Wan"),
            Val("a-417", "/asset/tag", "AT-30417"),
            Val("a-417", "/asset/serial", "9F2K-7731"),
            Val("a-417", "/make_model", "Lumen X14 laptop"),
            Val("a-422", "/assigned_to", "Stock room"),
        ],
        must_not=[
            MustNot.op(who="a-422"),
            MustNot.op(path="/asset"),
            MustNot.op("replace", "/make_model"),
            MustNot.op("replace", "/purchase_cost"),
            MustNot.insert(),
        ],
        tags=_tags("assets update many dotted-identity"),
        world=world,
        note="Records matched by their dotted identity (asset.tag): one asset is "
        "reassigned, the tag, serial, model and cost of it and everything of the "
        "other asset stay untouched.",
    )


LICENCE_TEXT = (
    "Please remove the Quillpad Pro licence from the register; we cancelled it at "
    "the end of August. The Nimbus Backup licence stays as it is."
)


def licence_delete() -> Case:
    existing = [
        (
            "lic-1",
            "SoftwareLicence",
            {"product": "Quillpad Pro", "vendor": "Quill Works", "seats": 25},
        ),
        (
            "lic-2",
            "SoftwareLicence",
            {"product": "Nimbus Backup", "vendor": "Nimbus", "seats": 40},
        ),
        (
            "lic-3",
            "SoftwareLicence",
            {"product": "Orbit CAD", "vendor": None, "seats": 6},
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(LICENCE_TEXT)
        w.entity(
            "Quillpad Pro",
            schema=SoftwareLicence,
            mentions=["Quillpad Pro"],
            fields={"product": "Quillpad Pro"},
            matches={"product": "Quillpad Pro"},
        )
        w.entity(
            "Nimbus Backup",
            schema=SoftwareLicence,
            mentions=["Nimbus Backup"],
            fields={"product": "Nimbus Backup"},
            matches={"product": "Nimbus Backup"},
        )
        w.delete("lic-1", 0.97)
        return w

    return Case(
        id="licence_delete",
        text=LICENCE_TEXT,
        schemas=lambda: [Many(SoftwareLicence, identity="product")],
        existing=existing,
        options={"enable_deletes": True},
        expect_ops=[Op("delete", who="lic-1")],
        must_not=[
            MustNot.delete("lic-2"),
            MustNot.delete("lic-3"),
            MustNot.op("replace"),
            MustNot.insert(),
        ],
        tags=_tags("assets delete many"),
        world=world,
        note="An explicit removal request deletes that record only; the licence the "
        "text says stays is mentioned and kept.",
    )


# --------------------------------------------------------------------------- #
# Conference speaker bios
# --------------------------------------------------------------------------- #

BIOS_TEXT = (
    "Programme notes. Dr. Priya Natarajan is a materials scientist at Brightwater "
    "Labs and will speak on solid-state batteries. Priya Nair is a policy analyst at "
    "the Calloway Institute; she will speak on grid regulation. Tomasz Wrobel opens "
    "the afternoon with a talk on repair cafes. Tomasz is also bringing a stall to "
    "the exhibition. Tickets are sold at the "
    "Henrietta Cole Auditorium box office."
)


def speaker_bios_many() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(BIOS_TEXT)
        w.entity(
            "Priya Natarajan",
            schema=Speaker,
            mentions=["Priya Natarajan"],
            fields={
                "name": "Priya Natarajan",
                "affiliation": "Brightwater Labs",
                "topic": "solid-state batteries",
            },
        )
        w.entity(
            "Priya Nair",
            schema=Speaker,
            mentions=["Priya Nair"],
            fields={
                "name": "Priya Nair",
                "affiliation": "Calloway Institute",
                "topic": "grid regulation",
            },
        )
        w.entity(
            "Tomasz Wrobel",
            schema=Speaker,
            mentions=["Tomasz Wrobel", "Tomasz"],
            fields={
                "name": "Tomasz Wrobel",
                "topic": "repair cafes",
            },
        )
        w.non_entity(
            "Henrietta Cole Auditorium",
            schema=Speaker,
            mentions=["Henrietta Cole Auditorium"],
        )
        # the Institute of the sentence before is offered for Tomasz, who has none
        w.bleed("Tomasz Wrobel", "affiliation", "Calloway Institute")
        return w

    return Case(
        id="speaker_bios_many",
        text=BIOS_TEXT,
        schemas=lambda: [Many(Speaker, identity="name")],
        expect_ops=[
            Op("insert", who="Natarajan"),
            Op("insert", who="Nair"),
            Op("insert", who="Tomasz"),
        ],
        expect_values=[
            Val("Natarajan", "/name", contains="Priya Natarajan"),
            Val("Natarajan", "/affiliation", "Brightwater Labs"),
            Val("Natarajan", "/topic", "solid-state batteries"),
            Val("Nair", "/name", "Priya Nair"),
            Val("Nair", "/affiliation", contains="Calloway Institute"),
            Val("Nair", "/topic", "grid regulation"),
            Val("Tomasz", "/name", contains="Tomasz Wrobel"),
            Val("Tomasz", "/affiliation", None),
            Val("Tomasz", "/topic", "repair cafes"),
        ],
        must_not=[
            MustNot.insert("Auditorium"),
            MustNot.insert("Henrietta"),
            MustNot.value_at("Tomasz", "/affiliation", contains="Calloway"),
            MustNot.value_at("Nair", "/topic", contains="batter"),
            MustNot.value_at("Natarajan", "/affiliation", contains="Calloway"),
        ],
        tags=_tags("speakers insert many coref similar-names negative bleed"),
        world=world,
        note="Two speakers called Priya (full names only), a title (Dr.), a pronoun "
        "(she), a short form (Tomasz) and a venue named after a person.",
    )


def speaker_no_longer() -> Case:
    text = (
        "Amara Osei no longer works at Brightwater Labs; she is now a research "
        "director at the Calloway Institute."
    )
    existing = [
        (
            "s-1",
            "SessionEntry",
            {
                "name": "Amara Osei",
                "affiliation": "Brightwater Labs",
                "topic": "lithium recycling",
                "hall": "Hall B",
            },
        ),
        (
            "s-2",
            "SessionEntry",
            {
                "name": "Kofi Lindgren",
                "affiliation": "Aldgate Makerspace",
                "topic": None,
                "hall": None,
            },
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Amara Osei",
            schema=SessionEntry,
            mentions=["Amara Osei"],
            fields={"name": "Amara Osei", "affiliation": "Calloway Institute"},
            matches={"name": "Amara Osei"},
        )
        return w

    return Case(
        id="speaker_no_longer",
        text=text,
        schemas=lambda: [Many(SessionEntry, identity="name")],
        existing=existing,
        options={"enable_inserts": True, "enable_deletes": True},
        expect_ops=[
            Op("replace", "/affiliation", contains="Calloway Institute", who="s-1")
        ],
        expect_values=[
            Val("s-1", "/affiliation", contains="Calloway Institute"),
            Val("s-1", "/topic", "lithium recycling"),
            Val("s-1", "/hall", "Hall B"),
        ],
        must_not=[
            MustNot.delete(),
            MustNot.insert(),
            MustNot.op("remove"),
            MustNot.op(who="s-2"),
        ],
        tags=_tags("speakers update many delete no-longer"),
        world=world,
        note="'No longer works at X' is a change of affiliation, not a request to "
        "delete the speaker: the record stays and the new employer replaces the old.",
    )


TIMETABLE_TEXT = (
    "Naomi Achebe has moved her session on microplastics to the main hall. Elias "
    "Brandt is still speaking on tidal power, but he now works at Harbourline Energy."
)


def speaker_update_bleed() -> Case:
    existing = [
        (
            "s-1",
            "SessionEntry",
            {
                "name": "Naomi Achebe",
                "affiliation": "Tidewater University",
                "topic": "microplastics",
                "hall": None,
            },
        ),
        (
            "s-2",
            "SessionEntry",
            {
                "name": "Elias Brandt",
                "affiliation": "Gannet Marine",
                "topic": "tidal power",
                "hall": None,
            },
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(TIMETABLE_TEXT)
        w.entity(
            "Naomi Achebe",
            schema=SessionEntry,
            mentions=["Naomi Achebe"],
            fields={
                "name": "Naomi Achebe",
                "topic": "microplastics",
                "hall": "main hall",
            },
            matches={"name": "Naomi Achebe"},
        )
        w.entity(
            "Elias Brandt",
            schema=SessionEntry,
            mentions=["Elias Brandt"],
            fields={
                "name": "Elias Brandt",
                "topic": "tidal power",
                "affiliation": "Harbourline Energy",
            },
            matches={"name": "Elias Brandt"},
        )
        # Elias's new employer is offered for Naomi, whose sentence comes first
        w.bleed("Naomi Achebe", "affiliation", "Harbourline Energy")
        return w

    return Case(
        id="speaker_update_bleed",
        text=TIMETABLE_TEXT,
        schemas=lambda: [Many(SessionEntry, identity="name")],
        existing=existing,
        options={"enable_inserts": True},
        expect_ops=[
            Op("add", "/hall", "main hall", who="s-1"),
            Op("replace", "/affiliation", contains="Harbourline Energy", who="s-2"),
        ],
        expect_values=[
            Val("s-1", "/hall", "main hall"),
            Val("s-1", "/affiliation", "Tidewater University"),
            Val("s-2", "/affiliation", contains="Harbourline Energy"),
            Val("s-2", "/hall", None),
        ],
        must_not=[
            MustNot.op("replace", "/affiliation", who="s-1"),
            MustNot.value_at("s-1", "/affiliation", contains="Harbourline"),
            MustNot.value_at("s-2", "/hall", contains="main hall"),
            MustNot.op("replace", "/topic"),
            MustNot.insert(),
        ],
        tags=_tags("speakers update many bleed"),
        world=world,
        note="Two timetable changes in two sentences: the new employer belongs to "
        "Elias only, the new hall to Naomi only; the topics they repeat are no "
        "changes.",
    )


# --------------------------------------------------------------------------- #
# Lab sample tracking (soil)
# --------------------------------------------------------------------------- #


def soil_scalars() -> Case:
    text = (
        "Sample SO-5521 from the Harlow Meadow plot was collected on September 2, "
        "2026 from the topsoil. pH 6.4, moisture 23.5 percent. No contamination was "
        "observed."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the SoilSample",
            schema=SoilSample,
            mentions=["SO-5521"],
            fields={
                "sample_id": "SO-5521",
                "site": "Harlow Meadow",
                "collected_on": date(2026, 9, 2),
                "ph": 6.4,
                "moisture_percent": 23.5,
                "depth_class": "topsoil",
                "contaminated": False,
            },
        )
        return w

    return Case(
        id="soil_scalars",
        text=text,
        schemas=lambda: [One(SoilSample)],
        expect_ops=[Op("insert", who="SoilSample")],
        expect_values=[
            Val("SoilSample", "/sample_id", "SO-5521"),
            Val("SoilSample", "/site", contains="Harlow Meadow"),
            Val("SoilSample", "/collected_on", "2026-09-02"),
            Val("SoilSample", "/ph", 6.4),
            Val("SoilSample", "/moisture_percent", 23.5),
            Val("SoilSample", "/depth_class", "topsoil"),
            Val("SoilSample", "/contaminated", False),
            Val("SoilSample", "/analyst", None),
            Val("SoilSample", "/lead_ppm", None),
        ],
        tags=_tags("soil insert scalar date bool literal optional"),
        world=world,
        note="Floats, a month-name date, a Literal layer and a negated bool; the "
        "analyst and the lead content are never mentioned.",
    )


CORE_TEXT = """\
Core log C-19, Harlow Meadow, drilled with a hand auger.

| Layer | Texture | Thickness (cm) |
| --- | --- | --- |
| L1 | sandy loam | 12 |
| L2 | clay | 30 |
| L3 | gravel | 8.5 |
"""


def soil_layers_table() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(CORE_TEXT)
        w.entity(
            "the CoreLog",
            schema=CoreLog,
            mentions=["C-19"],
            fields={"core_id": "C-19"},
        )
        for code, texture, thickness in (
            ("L1", "sandy loam", 12),
            ("L2", "clay", 30),
            ("L3", "gravel", 8.5),
        ):
            w.entity(
                code,
                schema=CoreLayer,
                parent="CoreLog",
                mentions=[code],
                fields={"code": code, "texture": texture, "thickness_cm": thickness},
            )
        return w

    return Case(
        id="soil_layers_table",
        text=CORE_TEXT,
        schemas=lambda: [One(CoreLog)],
        expect_ops=[Op("insert", who="CoreLog")],
        expect_values=[
            Val("CoreLog", "/core_id", "C-19"),
            Val("CoreLog", "/layers/0/code", "L1"),
            Val("CoreLog", "/layers/0/texture", "sandy loam"),
            Val("CoreLog", "/layers/0/thickness_cm", 12),
            Val("CoreLog", "/layers/1/code", "L2"),
            Val("CoreLog", "/layers/1/texture", "clay"),
            Val("CoreLog", "/layers/1/thickness_cm", 30),
            Val("CoreLog", "/layers/2/code", "L3"),
            Val("CoreLog", "/layers/2/texture", "gravel"),
            Val("CoreLog", "/layers/2/thickness_cm", 8.5),
        ],
        must_not=[MustNot.value_at("CoreLog", "/layers/3")],
        tags=_tags("soil insert nested table"),
        world=world,
        note="A core log with one table row per layer, each identified by a short "
        "code (L1 to L3) that is easy to mistake for a unit or a heading.",
    )


FIELD_NOTES = (
    "Field notes. Sample SO-7001 came from the north terrace and has a pH of 5.8. "
    "Sample SO-7002, taken near the creek, has a pH of 7.2."
)


def soil_samples_bleed() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(FIELD_NOTES)
        w.entity(
            "SO-7001",
            schema=FieldSample,
            mentions=["SO-7001"],
            fields={"sample_id": "SO-7001", "location": "north terrace", "ph": 5.8},
        )
        w.entity(
            "SO-7002",
            schema=FieldSample,
            mentions=["SO-7002"],
            fields={"sample_id": "SO-7002", "location": "creek", "ph": 7.2},
        )
        # the second sample's pH is offered for the first, whose sentence precedes it
        w.bleed("SO-7001", "ph", 7.2)
        w.bleed("SO-7002", "location", "north terrace")
        return w

    return Case(
        id="soil_samples_bleed",
        text=FIELD_NOTES,
        schemas=lambda: [Many(FieldSample, identity="sample_id")],
        expect_ops=[Op("insert", who="SO-7001"), Op("insert", who="SO-7002")],
        expect_values=[
            Val("SO-7001", "/location", contains="north terrace"),
            Val("SO-7001", "/ph", 5.8),
            Val("SO-7002", "/location", contains="creek"),
            Val("SO-7002", "/ph", 7.2),
        ],
        must_not=[
            MustNot.value_at("SO-7001", "/ph", 7.2),
            MustNot.value_at("SO-7002", "/location", contains="terrace"),
        ],
        tags=_tags("soil insert many bleed"),
        world=world,
        note="Two samples, two pH values: each reading belongs to the sample whose "
        "sentence states it.",
    )


# --------------------------------------------------------------------------- #
# Boat charter bookings
# --------------------------------------------------------------------------- #

GUESTS_TEXT = (
    "Booking BK-310 update: we are bringing three more guests, so please add them "
    "to the headcount."
)


def charter_guests_relative() -> Case:
    existing = [
        (
            "bk-310",
            "CharterBooking",
            {
                "booking_ref": "BK-310",
                "skipper": "Oskar Lund",
                "vessel": "Sea Osprey",
                "start_date": "2026-10-03",
                "guests": 4,
            },
        )
    ]

    def world() -> FakeWorld:
        w = FakeWorld(GUESTS_TEXT)
        w.entity(
            "the CharterBooking",
            schema=CharterBooking,
            mentions=["BK-310"],
            fields={"booking_ref": "BK-310", "guests": 3},
        )
        return w

    return Case(
        id="charter_guests_relative",
        text=GUESTS_TEXT,
        schemas=lambda: [One(CharterBooking)],
        existing=existing,
        # documented: "three more" is relative to the record's 4; the library never
        # computes it (7) and never writes the stated 3 over the 4: it refuses
        expect_ops=[],
        expect_rejected=[
            Rejected("relative_edit_unsupported", "/guests", "bk-310"),
        ],
        must_not=[
            MustNot.op("replace", "/guests", who="bk-310"),
            MustNot.value_at("bk-310", "/guests", 3),
            MustNot.value_at("bk-310", "/guests", 7),
        ],
        tags=_tags("charter update relative"),
        world=world,
        note="Documented limit: a relative edit ('three more guests') is refused as "
        "relative_edit_unsupported; the headcount stays 4 (not 3, not 7).",
    )


CHAT_TEXT = """\
[09:12] Nadia: hi, it's Nadia Petrenko, I'd like to book the Sea Osprey for a week
[09:13] Harbour desk: sure! what start date?
[09:14] Nadia: 3 October 2026, six of us
[09:15] Harbour desk: ok. deposit is TBD until the skipper is confirmed
[09:15] Harbour desk: fuel is included in the rate
[09:16] Nadia: perfect, thanks
"""


def charter_chat_log() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(CHAT_TEXT)
        w.entity(
            "the CharterRequest",
            schema=CharterRequest,
            mentions=["Nadia Petrenko"],
            fields={
                "customer_name": "Nadia Petrenko",
                "vessel": "Sea Osprey",
                "start_date": date(2026, 10, 3),
                "party_size": 6,
                "fuel_included": True,
                # the deposit is "TBD": a placeholder, not a value
            },
        )
        return w

    return Case(
        id="charter_chat_log",
        text=CHAT_TEXT,
        schemas=lambda: [One(CharterRequest)],
        expect_ops=[Op("insert", who="CharterRequest")],
        expect_values=[
            Val("CharterRequest", "/customer_name", "Nadia Petrenko"),
            Val("CharterRequest", "/vessel", "Sea Osprey"),
            Val("CharterRequest", "/start_date", "2026-10-03"),
            Val("CharterRequest", "/party_size", 6),
            Val("CharterRequest", "/fuel_included", True),
            Val("CharterRequest", "/deposit", None),
        ],
        tags=_tags("charter insert chat placeholder date bool decimal"),
        world=world,
        note="A chat log with timestamps and two speakers; 'six of us' is 6, and a "
        "deposit of 'TBD' is a placeholder that must come out null.",
    )


FLEET_TEXT = (
    "Fleet list for Port Merrin. The Sea Heron is a 12 m sloop with three cabins, "
    "moored at berth 14. The Sea Hawk is a 9 m cutter moored at berth 7. Heron Bay "
    "Marina handles all moorings."
)


def charter_fleet_similar() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(FLEET_TEXT)
        w.entity(
            "Sea Heron",
            schema=Yacht,
            mentions=["Sea Heron"],
            fields={"name": "Sea Heron", "length_m": 12, "cabins": 3, "berth": 14},
        )
        w.entity(
            "Sea Hawk",
            schema=Yacht,
            mentions=["Sea Hawk"],
            fields={"name": "Sea Hawk", "length_m": 9, "berth": 7},
        )
        w.non_entity("Heron Bay Marina", schema=Yacht, mentions=["Heron Bay Marina"])
        # the Heron's cabins are offered for the Hawk, which follows it and has none
        w.bleed("Sea Hawk", "cabins", 3)
        return w

    return Case(
        id="charter_fleet_similar",
        text=FLEET_TEXT,
        schemas=lambda: [Many(Yacht, identity="name")],
        expect_ops=[Op("insert", who="Sea Heron"), Op("insert", who="Sea Hawk")],
        expect_values=[
            Val("Sea Heron", "/length_m", 12),
            Val("Sea Heron", "/cabins", 3),
            Val("Sea Heron", "/berth", 14),
            Val("Sea Hawk", "/length_m", 9),
            Val("Sea Hawk", "/cabins", None),
            Val("Sea Hawk", "/berth", 7),
        ],
        must_not=[
            MustNot.insert("Marina"),
            MustNot.value_at("Sea Hawk", "/cabins", 3),
            MustNot.value_at("Sea Hawk", "/berth", 14),
        ],
        tags=_tags("charter insert many similar-names negative bleed"),
        world=world,
        note="Boats with near-identical names (Sea Heron, Sea Hawk) and a marina "
        "named after one of them; the Hawk's text gives no cabins, so the Heron's three "
        "must not appear on it.",
    )


# --------------------------------------------------------------------------- #
# Utility meter readings
# --------------------------------------------------------------------------- #


def meter_scalars() -> Case:
    text = (
        "Electricity meter 55-20193 was read on 11 September 2026 at 7:45 am: "
        "48,210.5 kWh. The reading covers the billing period August 2026 and is an "
        "estimate, not an actual reading. The household is on the night tariff."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the MeterRead",
            schema=MeterRead,
            mentions=["55-20193"],
            fields={
                "meter_number": "55-20193",
                "read_on": date(2026, 9, 11),
                "read_time": time(7, 45),
                "reading_kwh": 48210.5,
                "billing_period": "August 2026",
                "estimated": True,
                "tariff": "night",
            },
        )
        return w

    return Case(
        id="meter_scalars",
        text=text,
        schemas=lambda: [One(MeterRead)],
        expect_ops=[Op("insert", who="MeterRead")],
        expect_values=[
            Val("MeterRead", "/meter_number", "55-20193"),
            Val("MeterRead", "/read_on", "2026-09-11"),
            Val("MeterRead", "/read_time", "07:45:00"),
            Val("MeterRead", "/reading_kwh", 48210.5),
            Val("MeterRead", "/billing_period", "August 2026"),
            Val("MeterRead", "/estimated", True),
            Val("MeterRead", "/tariff", "night"),
            Val("MeterRead", "/rate_per_kwh", None),
            Val("MeterRead", "/peak_demand_kw", None),
        ],
        tags=_tags("meters insert scalar date time month-year bool literal optional"),
        world=world,
        note="A month-year kept as the text the bill prints (a string field), a "
        "decimal-separated float, a clock time, a bool that is true because the "
        "text says 'estimate', and two optionals that are never mentioned.",
    )


CARD_TEXT = (
    "WATER METER CARD -- SOUTHFIELD DISTRICT\n"
    "Account holder:  Bronwyn\n"
    "Tillerman\n"
    "Service address:  14 Willow\n"
    "Crescent, Ashby\n"
    "Meter no.  WM-77331920\n"
    "Last read:  09 Sep\n"
    "2026\n"
    "Reading (m3):  1,204.6\n"
    "Next read due:  [illegible]\n"
)


def meter_ocr_card() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(CARD_TEXT)
        w.entity(
            "the WaterMeterCard",
            schema=WaterMeterCard,
            mentions=["WM-77331920"],
            fields={
                "account_holder": "Bronwyn\nTillerman",
                "service_address": "14 Willow\nCrescent, Ashby",
                "meter_number": "WM-77331920",
                "last_read": date(2026, 9, 9),
                "reading_m3": 1204.6,
                # "[illegible]" is not a date
            },
        )
        return w

    return Case(
        id="meter_ocr_card",
        text=CARD_TEXT,
        schemas=lambda: [One(WaterMeterCard)],
        expect_ops=[Op("insert", who="WaterMeterCard")],
        expect_values=[
            Val("WaterMeterCard", "/account_holder", "Bronwyn Tillerman"),
            Val("WaterMeterCard", "/service_address", "14 Willow Crescent, Ashby"),
            Val("WaterMeterCard", "/meter_number", "WM-77331920"),
            Val("WaterMeterCard", "/last_read", "2026-09-09"),
            Val("WaterMeterCard", "/reading_m3", 1204.6),
            Val("WaterMeterCard", "/next_read_due", None),
        ],
        tags=_tags("meters insert ocr placeholder date optional"),
        world=world,
        note="OCR output: a name, an address and a date each broken over two lines, "
        "doubled spaces, and a next-read date that is '[illegible]' (null). Text "
        "values are quoted with their line break; scoring folds whitespace.",
    )


ROUND_TEXT = """\
Route R-14, read by the field team.

| Meter | Reading (kWh) | Note |
| --- | --- | --- |
| 4471093 | 12,480 | |
| 4471107 | 9,315 | dog in yard |
| 4471111 | 20,044 | meter box locked |
"""


def meter_rows_table() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(ROUND_TEXT)
        w.entity(
            "the ReadingRound",
            schema=ReadingRound,
            mentions=["R-14"],
            fields={"route": "R-14"},
        )
        for number, reading, note in (
            ("4471093", 12480, None),
            ("4471107", 9315, "dog in yard"),
            ("4471111", 20044, "meter box locked"),
        ):
            fields: dict[str, Any] = {"meter_no": number, "reading_kwh": reading}
            if note is not None:
                fields["note"] = note
            w.entity(
                number,
                schema=MeterRow,
                parent="ReadingRound",
                mentions=[number],
                fields=fields,
            )
        return w

    return Case(
        id="meter_rows_table",
        text=ROUND_TEXT,
        schemas=lambda: [One(ReadingRound)],
        expect_ops=[Op("insert", who="ReadingRound")],
        expect_values=[
            Val("ReadingRound", "/route", "R-14"),
            Val("ReadingRound", "/meters/0/meter_no", "4471093"),
            Val("ReadingRound", "/meters/0/reading_kwh", 12480),
            Val("ReadingRound", "/meters/0/note", None),
            Val("ReadingRound", "/meters/1/meter_no", "4471107"),
            Val("ReadingRound", "/meters/1/reading_kwh", 9315),
            Val("ReadingRound", "/meters/1/note", "dog in yard"),
            Val("ReadingRound", "/meters/2/meter_no", "4471111"),
            Val("ReadingRound", "/meters/2/reading_kwh", 20044),
            Val("ReadingRound", "/meters/2/note", "meter box locked"),
        ],
        must_not=[
            MustNot.value_at("ReadingRound", "/meters/0/note", contains="dog"),
            MustNot.value_at("ReadingRound", "/meters/3"),
        ],
        tags=_tags("meters insert nested table"),
        world=world,
        note="Rows named by seven-digit meter numbers with thousands-separated "
        "readings; a row with an empty note cell has no note, whatever the next "
        "row says.",
    )


HOLDOUT2_CASES: list[Case] = [
    claim_scalars(),
    claim_items_table(),
    claimants_bleed(),
    trip_form_dates(),
    trip_pupils_many(),
    trip_time_correction(),
    service_duration(),
    vehicle_notes_update(),
    service_shorthand(),
    supplier_order_lines(),
    supplier_email_thread(),
    vendors_not_people(),
    loans_table(),
    patron_rename(),
    patron_remove_and_change(),
    asset_dotted_insert(),
    asset_dotted_update(),
    licence_delete(),
    speaker_bios_many(),
    speaker_no_longer(),
    speaker_update_bleed(),
    soil_scalars(),
    soil_layers_table(),
    soil_samples_bleed(),
    charter_guests_relative(),
    charter_chat_log(),
    charter_fleet_similar(),
    meter_scalars(),
    meter_ocr_card(),
    meter_rows_table(),
]
"""Every holdout2 case, in the order reports list them."""

assert len({c.id for c in HOLDOUT2_CASES}) == len(HOLDOUT2_CASES), (
    "case ids must be unique"
)


def holdout2_cases() -> list[Case]:
    """The holdout2 suite (a copy of the list; the cases are shared)."""
    return list(HOLDOUT2_CASES)


if __name__ == "__main__":  # offline: `uv run python -m evals.holdout2_cases`
    import asyncio

    from evals.report import suite_markdown
    from evals.runner import Settings, aggregate, run_suite

    outcome = asyncio.run(run_suite(HOLDOUT2_CASES, Settings(), config="holdout2"))
    print(suite_markdown(outcome))
    totals = aggregate(outcome.results)
    raise SystemExit(0 if totals["passed"] == totals["cases"] else 1)
