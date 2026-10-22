"""The held-out eval suite: 30 cases that no search or tuning has ever seen.

It exists to MEASURE the provider and configuration combinations that a search over
the dev suite (``evals/cases.py``) picked, so it must stay independent of that suite
and of the search. The rule is in ``evals/README.md``: the holdout is never used
for tuning. Do not change a case, a threshold or a prompt
after looking at how a provider scores on it; a case is only fixed when its gold is
wrong from the text alone (and then the change is a new commit that says so).

The format is the dev suite's (``evals.model.Case``): the same scoring, the same
offline scripted ``FakeWorld`` per case. The schemas, domains and texts are new:
logistics, healthcare scheduling (fictional, non-sensitive), real estate listings,
aviation maintenance, SaaS support, a nutrition label and a recipe, event
registrations. Every name, number and identifier is made up.

Where the library has a documented limit, the gold follows the documented
behaviour and the case says so in its ``note``: a relative edit ("two more") is
refused, never computed; a text value is a verbatim span of the input (a value
broken over two lines is quoted with its break, which scoring folds); an input that
both changes and removes a record is neither (``conflict_update_delete``).
"""

from datetime import date, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from evals.model import Case, MustNot, Op, Rejected, Val
from trustcall_system_one import Many, One, s1_field
from trustcall_system_one.testing import FakeWorld, clairic_answer

__all__ = ["HOLDOUT_CASES", "SUITE_NAME", "holdout_cases"]

SUITE_NAME = "holdout"

# Relative dates in these texts are resolved against Monday 2026-03-23: "yesterday"
# is 2026-03-22, "tomorrow" 2026-03-24, "in two weeks" 2026-04-06, "three days ago"
# 2026-03-20, and "last Tuesday" is 2026-03-17 (the most recent Tuesday and the
# Tuesday of last week are the same day when today is a Monday).
REFERENCE_DATE = "2026-03-23"
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


class Shipment(BaseModel):
    """A freight consignment."""

    reference: str = Field(description="consignment reference")
    origin: str = Field(description="the city of the depot the consignment left")
    pallets: int = Field(description="number of pallets")
    gross_weight_kg: float = Field(description="gross weight in kilograms")
    declared_value_eur: Decimal = Field(description="declared value in euros")
    carrier: str = Field(description="the carrier's name")
    service: Literal["economy", "standard", "express"] = Field(
        description="service level"
    )
    signature_required: bool = Field(
        description="whether a signature is required on delivery"
    )
    insurance_policy: str | None = Field(
        None, description="the insurance policy number"
    )


class Booking(BaseModel):
    """A freight booking with its key dates."""

    reference: str = Field(description="booking reference")
    booked_on: date = Field(description="the date the booking was confirmed")
    pickup_on: date = Field(description="the date of the pickup")
    window_opens: date = Field(description="the date the delivery window opens")
    customs_filed_on: date = Field(description="the date customs paperwork was filed")


class Appointment(BaseModel):
    """A clinic appointment reminder."""

    patient_name: str = Field(description="the patient's full name")
    clinic: str = Field(description="the clinic's name")
    visit_date: date = Field(description="the date of the visit")
    start_time: time = Field(description="the time the visit starts")
    duration: timedelta = Field(
        description="how long the visit takes",
        json_schema_extra=s1_field(unit="minutes"),
    )
    visit_type: Literal["checkup", "follow_up", "vaccination", "lab_draw"] = Field(
        description="the kind of visit"
    )
    fasting_required: bool = Field(description="whether the patient must fast")
    interpreter_needed: bool | None = Field(
        None, description="whether an interpreter is needed"
    )
    wheelchair_access: bool | None = Field(
        None, description="whether wheelchair access is needed"
    )


class Listing(BaseModel):
    """A property listing."""

    address: str = Field(description="street address, with the unit if any")
    property_type: Literal["house", "condo", "townhouse", "land"] = Field(
        description="the kind of property"
    )
    bedrooms: int = Field(description="number of bedrooms")
    bathrooms: float = Field(description="number of bathrooms (a half bath is 0.5)")
    area_sqft: int = Field(description="floor area in square feet")
    asking_price: Decimal = Field(description="the asking price in dollars")
    pets_allowed: bool = Field(description="whether pets are allowed")
    hoa_fee_monthly: Decimal | None = Field(
        None, description="the monthly HOA fee in dollars"
    )
    listed_on: date = Field(description="the date the listing went live")


class Area(str, Enum):
    """The product area a support ticket is about."""

    BILLING = "billing"
    LOGIN = "login"
    API = "api"
    EXPORT = "export"


class Ticket(BaseModel):
    """A customer support ticket."""

    ticket_id: str = Field(description="ticket id")
    product_area: Area = Field(description="the product area")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="severity of the issue"
    )
    affected_users: int = Field(description="number of users affected")
    reported_on: date = Field(description="the date the issue was reported")
    customer_waiting: bool = Field(description="whether the customer awaits a reply")
    assignee: str | None = Field(None, description="who the ticket is assigned to")


class NutritionLabel(BaseModel):
    """A food product's nutrition label."""

    product_name: str = Field(description="the product's name")
    serving_size_g: int = Field(description="serving size in grams")
    servings_per_container: int = Field(description="servings per container")
    calories: int = Field(description="calories per serving")
    fat_g: float = Field(description="fat per serving in grams")
    sodium_mg: int = Field(description="sodium per serving in milligrams")
    contains_nuts: bool = Field(description="whether the product contains nuts")
    best_before: str = Field(description="the best-before month and year as printed")
    vegan: bool | None = Field(None, description="whether the product is vegan")


class Registration(BaseModel):
    """A conference registration."""

    attendee_name: str = Field(description="the attendee's full name")
    email: str = Field(description="the attendee's email address")
    event_name: str = Field(description="the event's name")
    event_date: date = Field(description="the date of the event")
    ticket_type: Literal["general", "student", "speaker", "vip"] = Field(
        description="the ticket type"
    )
    bringing_guest: bool = Field(description="whether the attendee brings a guest")
    dietary: str | None = Field(None, description="dietary requirements")
    tshirt_size: Literal["S", "M", "L", "XL"] | None = Field(
        None, description="t-shirt size"
    )


class WorkOrder(BaseModel):
    """An aircraft maintenance work order."""

    wo_number: str = Field(description="work order number")
    tail_number: str = Field(description="the aircraft's registration")
    task: str = Field(description="the work that was done")
    completed_on: date = Field(description="the date the work was completed")
    labour: timedelta = Field(
        description="labour time spent", json_schema_extra=s1_field(unit="hours")
    )
    deferred: bool = Field(description="whether the item was deferred")
    inspector: str | None = Field(None, description="the inspector who signed off")


class POLine(BaseModel):
    """One line of a purchase order."""

    description: str = Field(description="the item's description")
    quantity: int = Field(description="number of units ordered")
    unit_price: Decimal = Field(description="price of one unit")


class PurchaseOrder(BaseModel):
    """A purchase order."""

    po_number: str = Field(description="purchase order number")
    supplier: str = Field(description="the supplier's name")
    lines: Annotated[list[POLine], Many(identity="description")]


class Engine(BaseModel):
    """An engine fitted to an aircraft."""

    position: str = Field(description="the engine position code")
    serial_number: str = Field(description="the engine serial number")
    hours_since_overhaul: int = Field(description="hours since overhaul (TSO)")


class GearLeg(BaseModel):
    """One landing gear leg."""

    position: str = Field(description="the gear position code")
    serial_number: str = Field(description="the gear leg serial number")
    remarks: str | None = Field(None, description="remarks about the leg")


class Aircraft(BaseModel):
    """An aircraft record card."""

    registration: str = Field(description="the aircraft's registration")
    type_designation: str = Field(description="the aircraft type")
    engines: Annotated[list[Engine], Many(identity="position")]
    gear: Annotated[list[GearLeg], Many(identity="position")]


class Ingredient(BaseModel):
    """One ingredient of a recipe."""

    name: str = Field(description="the ingredient's name")
    amount: float = Field(description="the amount, as a number")
    unit: str | None = Field(None, description="the unit of the amount")


class Recipe(BaseModel):
    """A recipe."""

    title: str = Field(description="the recipe's title")
    servings: int = Field(description="how many servings it makes")
    ingredients: Annotated[list[Ingredient], Many(identity="name")]


class Slot(BaseModel):
    """A booked slot on a clinic day sheet."""

    patient_code: str = Field(description="the patient code")
    start_time: time = Field(description="the time the slot starts")
    room: str = Field(description="the room")
    visit_type: Literal["checkup", "follow_up", "vaccination", "lab_draw"] = Field(
        description="the kind of visit"
    )


class Participant(BaseModel):
    """A person taking part in a conversation."""

    name: str = Field(description="full name")
    role: str | None = Field(None, description="the person's job role")
    company: str | None = Field(None, description="the company the person is at")


class Booked(BaseModel):
    """A person with an appointment."""

    patient_name: str = Field(description="the patient's full name")
    start_time: time = Field(description="the time the appointment starts")
    room: str | None = Field(None, description="the room")


class Agent(BaseModel):
    """A real estate agent."""

    name: str = Field(description="the agent's full name")
    phone: str | None = Field(None, description="the agent's phone number")


class Attendee(BaseModel):
    """A person at a meeting."""

    name: str = Field(description="full name")
    commitment: str | None = Field(
        None, description="what the person said they would do"
    )


class Lead(BaseModel):
    """A buyer lead in a real estate CRM."""

    name: str = Field(description="the lead's name as written")
    budget: Decimal | None = Field(None, description="budget in dollars")
    min_bedrooms: int | None = Field(None, description="fewest bedrooms wanted")
    pre_approved: bool | None = Field(
        None, description="whether the lead has a mortgage pre-approval"
    )
    area: str | None = Field(None, description="the preferred part of town")


class Identification(BaseModel):
    """How a part is identified."""

    serial: str = Field(description="the part's serial number")
    part_number: str = Field(description="the part number")


class Component(BaseModel):
    """An aircraft component."""

    identification: Identification
    condition: Literal["serviceable", "unserviceable", "scrap"] = Field(
        description="the component's condition"
    )
    remarks: str | None = Field(None, description="remarks about the component")


class Visit(BaseModel):
    """A scheduled clinic visit."""

    patient_name: str = Field(description="the patient's full name")
    visit_date: date = Field(description="the date of the visit")
    start_time: time = Field(description="the time the visit starts")
    room: str | None = Field(None, description="the room")


class Account(BaseModel):
    """A customer account of a SaaS product."""

    company_name: str = Field(description="the customer company's name")
    plan: Literal["starter", "business", "enterprise"] = Field(
        description="the subscription plan"
    )
    seats: int = Field(description="number of seats")


class SupportCase(BaseModel):
    """A support case being worked."""

    ticket_id: str = Field(description="ticket id")
    status: Literal["open", "pending_customer", "resolved"] = Field(
        description="the status of the case"
    )
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering about the case"
    )


class Reservation(BaseModel):
    """A table reservation for an event."""

    booking_ref: str = Field(description="booking reference")
    team_name: str = Field(description="the team's name")
    seats: int = Field(description="number of seats booked")


class Property(BaseModel):
    """A property in a listing roundup."""

    address: str = Field(description="street address")
    asking_price: Decimal = Field(description="the asking price in dollars")
    garage: str | None = Field(None, description="the garage, as described")


# --------------------------------------------------------------------------- #
# Existing records
# --------------------------------------------------------------------------- #

SHIPMENT_RECORD: dict[str, Any] = {
    "reference": "CN-88213",
    "origin": "Rotterdam",
    "pallets": 14,
    "gross_weight_kg": 6350.5,
    "declared_value_eur": "48200.00",
    "carrier": "Baltic Freight Lines",
    "service": "express",
    "signature_required": True,
    "insurance_policy": None,
}


def listing_record(address: str, **changes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "address": address,
        "property_type": "townhouse",
        "bedrooms": 3,
        "bathrooms": 2.5,
        "area_sqft": 1780,
        "asking_price": "529900",
        "pets_allowed": True,
        "hoa_fee_monthly": None,
        "listed_on": "2026-03-09",
    }
    record.update(changes)
    return record


ACCOUNTS = [
    (
        "acct-1",
        "Account",
        {"company_name": "Orbit Freight", "plan": "business", "seats": 25},
    ),
    (
        "acct-2",
        "Account",
        {"company_name": "Quillon Foods", "plan": "starter", "seats": 5},
    ),
]

VISITS = [
    (
        "visit-1",
        "Visit",
        {
            "patient_name": "Rosa Bellamy",
            "visit_date": "2026-04-02",
            "start_time": "14:00:00",
            "room": "Room 1",
        },
    ),
    (
        "visit-2",
        "Visit",
        {
            "patient_name": "Ivan Szabo",
            "visit_date": "2026-04-02",
            "start_time": "14:30:00",
            "room": "Room 2",
        },
    ),
    (
        "visit-3",
        "Visit",
        {
            "patient_name": "Keiko Tanaka",
            "visit_date": "2026-04-02",
            "start_time": "15:00:00",
            "room": "Room 3",
        },
    ),
]


# --------------------------------------------------------------------------- #
# One-insert extraction
# --------------------------------------------------------------------------- #


def ship_scalars() -> Case:
    text = (
        "Consignment CN-88213 left the Rotterdam depot carrying 14 pallets, gross "
        "weight 6,350.5 kg. Declared value EUR 48,200.00. Carrier: Baltic Freight "
        "Lines. Service level: express. A signature on delivery is required."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Shipment",
            schema=Shipment,
            mentions=["CN-88213"],
            fields={
                "reference": "CN-88213",
                "origin": "Rotterdam",
                "pallets": 14,
                "gross_weight_kg": 6350.5,
                "declared_value_eur": 48200,
                "carrier": "Baltic Freight Lines",
                "service": "express",
                "signature_required": True,
            },
        )
        return w

    return Case(
        id="ship_scalars",
        text=text,
        schemas=lambda: [One(Shipment)],
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Shipment", "/reference", "CN-88213"),
            Val("Shipment", "/origin", "Rotterdam"),
            Val("Shipment", "/pallets", 14),
            Val("Shipment", "/gross_weight_kg", 6350.5),
            Val("Shipment", "/declared_value_eur", 48200),
            Val("Shipment", "/carrier", "Baltic Freight Lines"),
            Val("Shipment", "/service", "express"),
            Val("Shipment", "/signature_required", True),
            Val("Shipment", "/insurance_policy", None),
        ],
        tags=(
            "logistics",
            "insert",
            "scalar",
            "decimal",
            "literal",
            "bool",
            "optional",
        ),
        world=world,
        note="One insert of every scalar kind: text, int, float, a Decimal amount "
        "with a thousands separator, a Literal, a bool, and an optional field the "
        "text never mentions (insurance_policy stays null).",
    )


def booking_dates() -> Case:
    text = (
        "Booking confirmed yesterday for BK-90417. Pickup is tomorrow at the "
        "Antwerp depot and the delivery window opens in two weeks. Customs "
        "paperwork was filed on 2026-03-12."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Booking",
            schema=Booking,
            mentions=["BK-90417"],
            fields={
                "reference": "BK-90417",
                "booked_on": date(2026, 3, 22),
                "pickup_on": date(2026, 3, 24),
                "window_opens": date(2026, 4, 6),
                "customs_filed_on": date(2026, 3, 12),
            },
        )
        return w

    return Case(
        id="booking_dates",
        text=text,
        schemas=lambda: [One(Booking)],
        options=ON_THE_DAY,
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Booking", "/reference", "BK-90417"),
            Val("Booking", "/booked_on", "2026-03-22"),
            Val("Booking", "/pickup_on", "2026-03-24"),
            Val("Booking", "/window_opens", "2026-04-06"),
            Val("Booking", "/customs_filed_on", "2026-03-12"),
        ],
        tags=("logistics", "insert", "date", "relative-date"),
        world=world,
        note="Dates against the reference date 2026-03-23 (a Monday): 'yesterday', "
        "'tomorrow', 'in two weeks' and one absolute ISO date, four distinct fields.",
    )


def appointment_time_duration() -> Case:
    text = (
        "Reminder for Tomas Greer: follow-up visit at Lakeside Family Clinic on "
        "2 April 2026 at 2:45 pm. Expect it to take about 45 minutes. Please "
        "arrive fasting. No interpreter is needed."
    )

    def world() -> FakeWorld:
        w = DurationWorld(text)
        w.entity(
            "the Appointment",
            schema=Appointment,
            mentions=["Tomas Greer"],
            fields={
                "patient_name": "Tomas Greer",
                "clinic": "Lakeside Family Clinic",
                "visit_date": date(2026, 4, 2),
                "start_time": time(14, 45),
                "duration": timedelta(minutes=45),
                "visit_type": "follow_up",
                "fasting_required": True,
                "interpreter_needed": False,
            },
        )
        return w

    return Case(
        id="appointment_time_duration",
        text=text,
        schemas=lambda: [One(Appointment)],
        options=ON_THE_DAY,
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Appointment", "/patient_name", "Tomas Greer"),
            Val("Appointment", "/clinic", "Lakeside Family Clinic"),
            Val("Appointment", "/visit_date", "2026-04-02"),
            Val("Appointment", "/start_time", "14:45:00"),
            # a timedelta is held as Python's str(timedelta) in the record
            Val("Appointment", "/duration", "0:45:00"),
            Val("Appointment", "/visit_type", "follow_up"),
            Val("Appointment", "/fasting_required", True),
            Val("Appointment", "/interpreter_needed", False),
            Val("Appointment", "/wheelchair_access", None),
        ],
        tags=("healthcare", "insert", "time", "duration", "date", "bool", "optional"),
        world=world,
        note="A 12-hour clock time (2:45 pm is 14:45), a duration in minutes, a date "
        "written 'D Month YYYY', a Literal, a required bool, an Optional[bool] that "
        "is stated false (interpreter) and one the text never mentions (null).",
    )


def listing_basic() -> Case:
    text = (
        "NEW TO MARKET: 48 Alder Row, Unit 7. Townhouse, 3 bd / 2.5 ba, 1,780 sq ft. "
        "Asking $529,900. Pets welcome. Listed March 9, 2026."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Listing",
            schema=Listing,
            mentions=["48 Alder Row, Unit 7"],
            fields={
                "address": "48 Alder Row, Unit 7",
                "property_type": "townhouse",
                "bedrooms": 3,
                "bathrooms": 2.5,
                "area_sqft": 1780,
                "asking_price": Decimal("529900"),
                "pets_allowed": True,
                "listed_on": date(2026, 3, 9),
            },
        )
        return w

    return Case(
        id="listing_basic",
        text=text,
        schemas=lambda: [One(Listing)],
        options=ON_THE_DAY,
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Listing", "/address", "48 Alder Row, Unit 7"),
            Val("Listing", "/property_type", "townhouse"),
            Val("Listing", "/bedrooms", 3),
            Val("Listing", "/bathrooms", 2.5),
            Val("Listing", "/area_sqft", 1780),
            Val("Listing", "/asking_price", 529900),
            Val("Listing", "/pets_allowed", True),
            Val("Listing", "/listed_on", "2026-03-09"),
            Val("Listing", "/hoa_fee_monthly", None),
        ],
        tags=("realestate", "insert", "scalar", "decimal", "date", "optional"),
        world=world,
        note="A listing headline with real-estate shorthand ('3 bd / 2.5 ba', "
        "'sq ft'), a dollar amount, a month-name date and an unmentioned optional "
        "HOA fee.",
    )


def ticket_enum() -> Case:
    text = (
        "Ticket TK-4471, area: login. Customers cannot sign in through SSO since the "
        "09:10 deploy. Roughly 120 users are affected. Severity: high. The customer "
        "is waiting for a reply. Reported three days ago."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Ticket",
            schema=Ticket,
            mentions=["TK-4471"],
            fields={
                "ticket_id": "TK-4471",
                "product_area": "LOGIN",  # an Enum is asked by member name
                "severity": "high",
                "affected_users": 120,
                "reported_on": date(2026, 3, 20),
                "customer_waiting": True,
            },
        )
        return w

    return Case(
        id="ticket_enum",
        text=text,
        schemas=lambda: [One(Ticket)],
        options=ON_THE_DAY,
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Ticket", "/ticket_id", "TK-4471"),
            Val("Ticket", "/product_area", "login"),
            Val("Ticket", "/severity", "high"),
            Val("Ticket", "/affected_users", 120),
            Val("Ticket", "/reported_on", "2026-03-20"),
            Val("Ticket", "/customer_waiting", True),
            Val("Ticket", "/assignee", None),
        ],
        tags=("saas", "insert", "enum", "literal", "bool", "relative-date", "optional"),
        world=world,
        note="An Enum and a Literal each stated in words, a rounded count ('roughly "
        "120'), a count-of-days-ago date ('three days ago' is 2026-03-20) and an "
        "unassigned ticket (assignee null).",
    )


def nutrition_label() -> Case:
    text = (
        "Hearthside Maple Granola\n"
        "Nutrition Facts\n"
        "Serving size 45 g\n"
        "Servings per container 8\n"
        "Calories 190\n"
        "Total Fat 7 g\n"
        "Sodium 85 mg\n"
        "Contains tree nuts (almonds, pecans).\n"
        "Best before: NOV 2026"
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the NutritionLabel",
            schema=NutritionLabel,
            mentions=["Hearthside Maple Granola"],
            fields={
                "product_name": "Hearthside Maple Granola",
                "serving_size_g": 45,
                "servings_per_container": 8,
                "calories": 190,
                "fat_g": 7,
                "sodium_mg": 85,
                "contains_nuts": True,
                "best_before": "NOV 2026",
            },
        )
        return w

    return Case(
        id="nutrition_label",
        text=text,
        schemas=lambda: [One(NutritionLabel)],
        expect_ops=[Op("insert")],
        expect_values=[
            Val("NutritionLabel", "/product_name", "Hearthside Maple Granola"),
            Val("NutritionLabel", "/serving_size_g", 45),
            Val("NutritionLabel", "/servings_per_container", 8),
            Val("NutritionLabel", "/calories", 190),
            Val("NutritionLabel", "/fat_g", 7),
            Val("NutritionLabel", "/sodium_mg", 85),
            Val("NutritionLabel", "/contains_nuts", True),
            # a month and year as printed, kept as text (the day is not stated)
            Val("NutritionLabel", "/best_before", "NOV 2026"),
            Val("NutritionLabel", "/vegan", None),
        ],
        tags=("nutrition", "insert", "scalar", "bool", "optional", "label"),
        world=world,
        note="A label laid out one value per line: six numbers with units, a bool "
        "read from a sentence ('Contains tree nuts'), a month-year kept as text, and "
        "an unmentioned Optional[bool] (vegan stays null, not false).",
    )


def event_registration() -> Case:
    text = (
        "Registration for Harbor DevDay on 12 June 2026: Lena Kowalski "
        "(lena.kowalski@example.org), student ticket. No guest. Dietary: vegetarian."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Registration",
            schema=Registration,
            mentions=["Lena Kowalski"],
            fields={
                "attendee_name": "Lena Kowalski",
                "email": "lena.kowalski@example.org",
                "event_name": "Harbor DevDay",
                "event_date": date(2026, 6, 12),
                "ticket_type": "student",
                "bringing_guest": False,
                "dietary": "vegetarian",
            },
        )
        return w

    return Case(
        id="event_registration",
        text=text,
        schemas=lambda: [One(Registration)],
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Registration", "/attendee_name", "Lena Kowalski"),
            Val("Registration", "/email", "lena.kowalski@example.org"),
            Val("Registration", "/event_name", "Harbor DevDay"),
            Val("Registration", "/event_date", "2026-06-12"),
            Val("Registration", "/ticket_type", "student"),
            Val("Registration", "/bringing_guest", False),
            Val("Registration", "/dietary", "vegetarian"),
            Val("Registration", "/tshirt_size", None),
        ],
        tags=("events", "insert", "literal", "bool", "optional"),
        world=world,
        note="A negated bool ('No guest' is false), an email, and an optional "
        "Literal (t-shirt size) that the text never mentions: it stays null and is "
        "not forced to one of S/M/L/XL.",
    )


def work_order() -> Case:
    text = (
        "WO 55-1182, tail ZZ-KFA: replaced the left main wheel brake pad set. "
        "Completed last Tuesday after 2 h 30 min of labour. Item deferred: no."
    )

    def world() -> FakeWorld:
        w = DurationWorld(text)
        w.entity(
            "the WorkOrder",
            schema=WorkOrder,
            mentions=["WO 55-1182"],
            fields={
                "wo_number": "55-1182",
                "tail_number": "ZZ-KFA",
                "task": "replaced the left main wheel brake pad set",
                "completed_on": date(2026, 3, 17),
                "labour": timedelta(hours=2, minutes=30),
                "deferred": False,
            },
        )
        return w

    return Case(
        id="work_order",
        text=text,
        schemas=lambda: [One(WorkOrder)],
        options=ON_THE_DAY,
        expect_ops=[Op("insert")],
        expect_values=[
            Val("WorkOrder", "/wo_number", contains="55-1182"),
            Val("WorkOrder", "/tail_number", "ZZ-KFA"),
            Val("WorkOrder", "/task", contains="brake pad set"),
            Val("WorkOrder", "/completed_on", "2026-03-17"),
            # 2 h 30 min, held as Python's str(timedelta)
            Val("WorkOrder", "/labour", "2:30:00"),
            Val("WorkOrder", "/deferred", False),
            Val("WorkOrder", "/inspector", None),
        ],
        tags=("aviation", "insert", "duration", "date", "relative-date", "bool"),
        world=world,
        note="A weekday-relative past date ('last Tuesday', today being Monday "
        "2026-03-23, is 2026-03-17), a compound duration ('2 h 30 min'), a negated "
        "bool and an unmentioned inspector.",
    )


# --------------------------------------------------------------------------- #
# Nested sub-models and list[Model] items
# --------------------------------------------------------------------------- #


def po_table_lines() -> Case:
    text = (
        "PO-7731 to Delta Fasteners\n"
        "\n"
        "| Item | Qty | Unit price |\n"
        "|---|---|---|\n"
        "| M8 hex bolt | 500 | 0.12 |\n"
        "| M8 washer | 1000 | 0.03 |\n"
        "| Torque wrench 40Nm | 2 | 89.50 |\n"
    )
    rows = [
        ("M8 hex bolt", 500, 0.12),
        ("M8 washer", 1000, 0.03),
        ("Torque wrench 40Nm", 2, 89.5),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the PurchaseOrder",
            schema=PurchaseOrder,
            mentions=["PO-7731"],
            fields={"po_number": "PO-7731", "supplier": "Delta Fasteners"},
        )
        for name, quantity, price in rows:
            w.entity(
                name,
                schema=POLine,
                parent="PurchaseOrder",
                mentions=[name],
                fields={
                    "description": name,
                    "quantity": quantity,
                    "unit_price": price,
                },
            )
        return w

    values = [
        Val("PurchaseOrder", "/po_number", "PO-7731"),
        Val("PurchaseOrder", "/supplier", "Delta Fasteners"),
    ]
    for i, (name, quantity, price) in enumerate(rows):
        values += [
            Val("PurchaseOrder", f"/lines/{i}/description", name),
            Val("PurchaseOrder", f"/lines/{i}/quantity", quantity),
            Val("PurchaseOrder", f"/lines/{i}/unit_price", price),
        ]
    values.append(Val("PurchaseOrder", "/lines/3", None))  # exactly three lines
    return Case(
        id="po_table_lines",
        text=text,
        schemas=lambda: [One(PurchaseOrder)],
        expect_ops=[Op("insert")],
        expect_values=values,
        tags=("logistics", "insert", "nested", "table", "decimal"),
        world=world,
        note="Order lines in a markdown table: quantity and unit price sit in "
        "neighbouring cells (500 vs 0.12), each line takes its own row's numbers, "
        "and there are exactly three lines (no fourth, no header row as a line).",
    )


AIRCRAFT_CARD = (
    "AIRCRAFT RECORD CARD\n"
    "Reg: ZZ-KFA     Type: Marlin T40\n"
    "-- ENGINES --\n"
    "ENG-1  S/N PCE-40817   TSO 1,204 hrs\n"
    "ENG-2  S/N PCE-40822   TSO 1,187 hrs\n"
    "-- LANDING GEAR --\n"
    "NLG   S/N NG-3301   remarks: shimmy damper\n"
    " weeping at lower seal\n"
    "LMG   S/N MG-7120\n"
    "RMG   S/N MG-7135   remarks: brake wear\n"
    " within limits\n"
)


def aircraft_spec_insert() -> Case:
    text = AIRCRAFT_CARD
    wrapped_nlg = "shimmy damper\n weeping at lower seal"
    wrapped_rmg = "brake wear\n within limits"

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Aircraft",
            schema=Aircraft,
            mentions=["ZZ-KFA"],
            fields={"registration": "ZZ-KFA", "type_designation": "Marlin T40"},
        )
        for position, serial, hours in (
            ("ENG-1", "PCE-40817", 1204),
            ("ENG-2", "PCE-40822", 1187),
        ):
            w.entity(
                position,
                schema=Engine,
                parent="Aircraft",
                mentions=[position],
                fields={
                    "position": position,
                    "serial_number": serial,
                    "hours_since_overhaul": hours,
                },
            )
        for position, serial, remarks in (
            ("NLG", "NG-3301", wrapped_nlg),
            ("LMG", "MG-7120", None),
            ("RMG", "MG-7135", wrapped_rmg),
        ):
            fields: dict[str, Any] = {"position": position, "serial_number": serial}
            if remarks is not None:
                fields["remarks"] = remarks
            w.entity(
                position,
                schema=GearLeg,
                parent="Aircraft",
                mentions=[position],
                fields=fields,
            )
        return w

    return Case(
        id="aircraft_spec_insert",
        text=text,
        schemas=lambda: [One(Aircraft)],
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Aircraft", "/registration", "ZZ-KFA"),
            Val("Aircraft", "/type_designation", "Marlin T40"),
            Val("Aircraft", "/engines/0/position", "ENG-1"),
            Val("Aircraft", "/engines/0/serial_number", "PCE-40817"),
            Val("Aircraft", "/engines/0/hours_since_overhaul", 1204),
            Val("Aircraft", "/engines/1/position", "ENG-2"),
            Val("Aircraft", "/engines/1/serial_number", "PCE-40822"),
            Val("Aircraft", "/engines/1/hours_since_overhaul", 1187),
            Val("Aircraft", "/engines/2", None),
            Val("Aircraft", "/gear/0/position", "NLG"),
            Val("Aircraft", "/gear/0/serial_number", "NG-3301"),
            # a value broken over two lines by the sheet is quoted with its break
            # (verbatim span); scoring folds whitespace, so the break does not matter
            Val("Aircraft", "/gear/0/remarks", contains="weeping at lower seal"),
            Val("Aircraft", "/gear/1/position", "LMG"),
            Val("Aircraft", "/gear/1/serial_number", "MG-7120"),
            Val("Aircraft", "/gear/1/remarks", None),
            Val("Aircraft", "/gear/2/position", "RMG"),
            Val("Aircraft", "/gear/2/serial_number", "MG-7135"),
            Val("Aircraft", "/gear/2/remarks", contains="within limits"),
            Val("Aircraft", "/gear/3", None),
        ],
        must_not=[
            MustNot.value_at("Aircraft", "/gear/1/remarks", contains="weeping"),
            MustNot.value_at("Aircraft", "/gear/1/remarks", contains="brake"),
        ],
        tags=("aviation", "insert", "nested", "ocr", "identity-code"),
        world=world,
        note="An OCR-style record card: two engines and three gear legs identified "
        "by position codes (ENG-1, NLG, ...), hours written 'TSO 1,204 hrs', and "
        "remarks wrapped over a line break. The middle leg has no remarks and must "
        "stay null (neither neighbour's remarks may land on it).",
    )


def recipe_ingredients() -> Case:
    text = (
        "Lemon Herb Chickpea Salad (serves 4). Toss 2 cans chickpeas with 1 lemon, "
        "3 tbsp olive oil and 1/4 cup chopped parsley. Season to taste."
    )
    rows: list[tuple[str, float, str | None]] = [
        ("chickpeas", 2, "cans"),
        ("lemon", 1, None),
        ("olive oil", 3, "tbsp"),
        ("parsley", 0.25, "cup"),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Recipe",
            schema=Recipe,
            mentions=["Lemon Herb Chickpea Salad"],
            fields={"title": "Lemon Herb Chickpea Salad", "servings": 4},
        )
        for name, amount, unit in rows:
            fields: dict[str, Any] = {"name": name, "amount": amount}
            if unit is not None:
                fields["unit"] = unit
            w.entity(
                name,
                schema=Ingredient,
                parent="Recipe",
                mentions=[name],
                fields=fields,
            )
        return w

    values = [
        Val("Recipe", "/title", "Lemon Herb Chickpea Salad"),
        Val("Recipe", "/servings", 4),
    ]
    for i, (name, amount, unit) in enumerate(rows):
        values += [
            Val("Recipe", f"/ingredients/{i}/name", contains=name),
            Val("Recipe", f"/ingredients/{i}/amount", amount),
            Val("Recipe", f"/ingredients/{i}/unit", unit),
        ]
    values.append(Val("Recipe", "/ingredients/4", None))  # nothing else is invented
    return Case(
        id="recipe_ingredients",
        text=text,
        schemas=lambda: [One(Recipe)],
        expect_ops=[Op("insert")],
        expect_values=values,
        tags=("recipe", "insert", "nested", "optional", "fraction"),
        world=world,
        note="Ingredients in running prose with a fraction ('1/4 cup' is 0.25), a "
        "line with no unit ('1 lemon': unit stays null) and a closing 'Season to "
        "taste' that names no ingredient (no fifth item).",
    )


def aircraft_engine_swap_update() -> Case:
    text = (
        "Engine swap on ZZ-KFA: ENG-2 was replaced with a unit that has S/N "
        "PCE-51207 and 0 hours since overhaul."
    )
    existing = [
        (
            "ac-1",
            "Aircraft",
            {
                "registration": "ZZ-KFA",
                "type_designation": "Marlin T40",
                "engines": [
                    {
                        "position": "ENG-1",
                        "serial_number": "PCE-40817",
                        "hours_since_overhaul": 1204,
                    },
                    {
                        "position": "ENG-2",
                        "serial_number": "PCE-40822",
                        "hours_since_overhaul": 1187,
                    },
                ],
                "gear": [
                    {"position": "NLG", "serial_number": "NG-3301", "remarks": None}
                ],
            },
        )
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Aircraft",
            schema=Aircraft,
            mentions=["ZZ-KFA"],
            fields={"registration": "ZZ-KFA"},
        )
        w.entity(
            "ENG-2",
            schema=Engine,
            parent="Aircraft",
            mentions=["ENG-2"],
            fields={
                "position": "ENG-2",
                "serial_number": "PCE-51207",
                "hours_since_overhaul": 0,
            },
            matches={"position": "ENG-2"},
        )
        return w

    return Case(
        id="aircraft_engine_swap_update",
        text=text,
        schemas=lambda: [One(Aircraft)],
        existing=existing,
        options={"enable_inserts": True},
        expect_ops=[
            Op("replace", "/engines/1/serial_number", "PCE-51207", who="ac-1"),
            Op("replace", "/engines/1/hours_since_overhaul", 0, who="ac-1"),
        ],
        must_not=[
            MustNot.op(path="/engines/0"),
            MustNot.op(path="/gear"),
            MustNot.op("remove"),
            MustNot.op("add", "/engines/-"),
            MustNot.op("replace", "/registration"),
        ],
        tags=("aviation", "update", "nested", "identity-code"),
        world=world,
        note="A nested item found by its position code (ENG-2 is the second element) "
        "gets two changed fields; the other engine and the gear are untouched, and "
        "no engine is added. Hours going to 0 is a stated value, not a relative edit.",
    )


def clinic_schedule_table() -> Case:
    text = (
        "Clinic day sheet, Lakeside Family Clinic\n"
        "| Code | Time | Room | Visit |\n"
        "|------|------|------|-------|\n"
        "| PT-1041 | 09:00 | Room 2 | checkup |\n"
        "| PT-1042 | 09:30 | Room 3 | vaccination |\n"
        "| PT-1043 | 10:15 | Room 2 | lab draw |\n"
        "| PT-1044 | 11:00 | Room 1 | follow-up |\n"
    )
    rows = [
        ("PT-1041", time(9, 0), "Room 2", "checkup"),
        ("PT-1042", time(9, 30), "Room 3", "vaccination"),
        ("PT-1043", time(10, 15), "Room 2", "lab_draw"),
        ("PT-1044", time(11, 0), "Room 1", "follow_up"),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        for code, start, room, kind in rows:
            w.entity(
                code,
                schema=Slot,
                mentions=[code],
                fields={
                    "patient_code": code,
                    "start_time": start,
                    "room": room,
                    "visit_type": kind,
                },
            )
        return w

    values: list[Val] = []
    for code, start, room, kind in rows:
        values += [
            Val(code, "/start_time", start.isoformat()),
            Val(code, "/room", room),
            Val(code, "/visit_type", kind),
        ]
    return Case(
        id="clinic_schedule_table",
        text=text,
        schemas=lambda: [Many(Slot, identity="patient_code")],
        options={"enable_inserts": True},
        expect_ops=[Op("insert", who=code) for code, *_ in rows],
        expect_values=values,
        must_not=[MustNot.insert("Code"), MustNot.insert("Room")],
        tags=("healthcare", "insert", "many", "table", "literal", "time"),
        world=world,
        note="One record per table row (four, none for the header or the clinic), "
        "each with its own time, room and visit type; 'lab draw' and 'follow-up' "
        "are written differently from the Literal values lab_draw and follow_up.",
    )


# --------------------------------------------------------------------------- #
# Many entities: coreference, non-entities, similar names, bleed
# --------------------------------------------------------------------------- #


def support_thread_participants() -> Case:
    text = (
        "From: Marta Veld\n"
        "Subject: Re: SSO outage\n"
        "\n"
        "Hi team, we are still seeing the failures. I'm the IT administrator at "
        "Northgate Logistics, so I can test any fix today. Our office dog Biscuit "
        "unplugged a router this morning, which did not help.\n"
        "\n"
        "On Tue, Joel Park wrote:\n"
        "> Hi Marta, I'm the support engineer on your ticket. I have asked Priya "
        "Nair to look at the logs. She is our SSO specialist.\n"
        "> Regards, Joel\n"
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Marta Veld",
            schema=Participant,
            mentions=["Marta Veld", "Marta"],
            fields={
                "name": "Marta Veld",
                "role": "IT administrator",
                "company": "Northgate Logistics",
            },
        )
        w.entity(
            "Joel Park",
            schema=Participant,
            mentions=["Joel Park", "Joel"],
            fields={"name": "Joel Park", "role": "support engineer"},
        )
        w.entity(
            "Priya Nair",
            schema=Participant,
            mentions=["Priya Nair"],
            fields={"name": "Priya Nair", "role": "SSO specialist"},
        )
        w.non_entity("Biscuit", schema=Participant, mentions=["Biscuit"])
        w.non_entity(
            "Northgate Logistics", schema=Participant, mentions=["Northgate Logistics"]
        )
        w.bleed("Joel Park", "company", "Northgate Logistics")
        w.bleed("Priya Nair", "company", "Northgate Logistics")
        return w

    return Case(
        id="support_thread_participants",
        text=text,
        schemas=lambda: [Many(Participant, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[
            Op("insert", who="Marta Veld"),
            Op("insert", who="Joel Park"),
            Op("insert", who="Priya Nair"),
        ],
        expect_values=[
            Val("Marta Veld", "/role", contains="IT administrator"),
            Val("Marta Veld", "/company", "Northgate Logistics"),
            Val("Joel Park", "/role", contains="support engineer"),
            Val("Joel Park", "/company", None),
            Val("Priya Nair", "/role", contains="SSO specialist"),
            Val("Priya Nair", "/company", None),
        ],
        must_not=[
            MustNot.insert("Biscuit"),
            MustNot.insert("Northgate"),
            MustNot.value_at("Joel Park", "/company", contains="Northgate"),
            MustNot.value_at("Priya Nair", "/role", contains="support engineer"),
            MustNot.value_at("Marta Veld", "/role", contains="support engineer"),
            MustNot.value_at("Joel Park", "/role", contains="SSO"),
        ],
        tags=("saas", "insert", "many", "email", "coref", "negative", "bleed"),
        world=world,
        note="An email thread with a quoted reply: 'Marta' / 'Joel' are short forms "
        "of the signed names (three people, not five), the company and the office "
        "dog are not people, and each role belongs to the person who states it "
        "(Priya's through the pronoun 'She'). Joel's company is not stated (the "
        "quoted text says only 'our'), so null.",
    )


def similar_names_patients() -> Case:
    text = (
        "Priya Menon is booked at 10:15 in Room 4. Her sister Priyanka Menon is "
        "booked at 11:00 in Room 2. Priya will bring her referral letter."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Priya Menon",
            schema=Booked,
            mentions=["Priya Menon", "Priya"],
            fields={
                "patient_name": "Priya Menon",
                "start_time": time(10, 15),
                "room": "Room 4",
            },
        )
        w.entity(
            "Priyanka Menon",
            schema=Booked,
            mentions=["Priyanka Menon"],
            fields={
                "patient_name": "Priyanka Menon",
                "start_time": time(11, 0),
                "room": "Room 2",
            },
        )
        return w

    return Case(
        id="similar_names_patients",
        text=text,
        schemas=lambda: [Many(Booked, identity="patient_name")],
        options={"enable_inserts": True},
        expect_ops=[
            Op("insert", who="Priya Menon"),
            Op("insert", who="Priyanka Menon"),
        ],
        expect_values=[
            Val("Priya Menon", "/start_time", "10:15:00"),
            Val("Priya Menon", "/room", "Room 4"),
            Val("Priyanka Menon", "/start_time", "11:00:00"),
            Val("Priyanka Menon", "/room", "Room 2"),
        ],
        must_not=[
            MustNot.value_at("Priyanka Menon", "/room", contains="Room 4"),
            MustNot.value_at("Priya Menon", "/room", contains="Room 2"),
            MustNot.value_at("Priyanka Menon", "/start_time", contains="10:15"),
        ],
        tags=("healthcare", "insert", "many", "coref", "similar-names", "bleed"),
        world=world,
        note="Two people whose names share a prefix (Priya, Priyanka), each with her "
        "own time and room in neighbouring sentences; the lone 'Priya' at the end "
        "is the first one, never a third person or the sister.",
    )


def agents_vs_companies() -> Case:
    text = (
        "Listing shared by Tomás Reyes (tel. 555-0142) of Hartley & Moss Realty. "
        "Co-listed with Hannah Birch. Showings are booked through Hartley & Moss "
        "Realty's front desk. The seller's cat, Pepper, will be at home but is shy."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Tomás Reyes",
            schema=Agent,
            mentions=["Tomás Reyes"],
            fields={"name": "Tomás Reyes", "phone": "555-0142"},
        )
        w.entity(
            "Hannah Birch",
            schema=Agent,
            mentions=["Hannah Birch"],
            fields={"name": "Hannah Birch"},
        )
        w.non_entity(
            "Hartley & Moss Realty", schema=Agent, mentions=["Hartley & Moss Realty"]
        )
        w.non_entity("Pepper", schema=Agent, mentions=["Pepper"])
        w.bleed("Hannah Birch", "phone", "555-0142")
        return w

    return Case(
        id="agents_vs_companies",
        text=text,
        schemas=lambda: [Many(Agent, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[Op("insert", who="Tomás Reyes"), Op("insert", who="Hannah Birch")],
        expect_values=[
            Val("Tomás Reyes", "/phone", "555-0142"),
            Val("Hannah Birch", "/phone", None),
        ],
        must_not=[
            MustNot.insert("Hartley"),
            MustNot.insert("Moss"),
            MustNot.insert("Pepper"),
            MustNot.value_at("Hannah Birch", "/phone", contains="0142"),
        ],
        tags=("realestate", "insert", "many", "negative", "bleed"),
        world=world,
        note="A brokerage named like two people ('Hartley & Moss Realty') and a cat "
        "are not agents; the phone number is Tomás's (it follows his name), and "
        "the co-listing agent has none, though her name comes next.",
    )


def meeting_transcript() -> Case:
    text = (
        "Ana Ruiz: Let's start. Ben, where are we on the invoice export bug?\n"
        "Ben Okoye: I reproduced it. It only fails for accounts with more than 10k "
        "rows. I'll ship a fix by Thursday.\n"
        "Ana Ruiz: Good. Chen, can you check the docs?\n"
        "Chen Wei: Sure, I'll update the export docs this week.\n"
        "Ana Ruiz: Great, that's everything. Thanks both.\n"
    )
    ben = "I'll ship a fix by Thursday."
    chen = "I'll update the export docs this week."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Ana Ruiz",
            schema=Attendee,
            mentions=["Ana Ruiz"],
            fields={"name": "Ana Ruiz"},
        )
        w.entity(
            "Ben Okoye",
            schema=Attendee,
            mentions=["Ben Okoye", "Ben"],
            fields={"name": "Ben Okoye", "commitment": ben},
        )
        w.entity(
            "Chen Wei",
            schema=Attendee,
            mentions=["Chen Wei", "Chen"],
            fields={"name": "Chen Wei", "commitment": chen},
        )
        w.bleed("Ana Ruiz", "commitment", chen)
        return w

    return Case(
        id="meeting_transcript",
        text=text,
        schemas=lambda: [Many(Attendee, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[
            Op("insert", who="Ana Ruiz"),
            Op("insert", who="Ben Okoye"),
            Op("insert", who="Chen Wei"),
        ],
        expect_values=[
            Val("Ben Okoye", "/commitment", contains="ship a fix by Thursday"),
            Val("Chen Wei", "/commitment", contains="update the export docs"),
            Val("Ana Ruiz", "/commitment", None),
        ],
        must_not=[
            MustNot.value_at("Ana Ruiz", "/commitment", contains="docs"),
            MustNot.value_at("Ben Okoye", "/commitment", contains="docs"),
            MustNot.value_at("Chen Wei", "/commitment", contains="Thursday"),
        ],
        tags=("saas", "insert", "many", "transcript", "coref", "bleed"),
        world=world,
        note="A meeting transcript with speaker labels. 'Ben' and 'Chen' are short "
        "forms of the speakers; each promise belongs to the speaker who makes it, "
        "and Ana (who only asks the others and closes) has none.",
    )


def crm_two_leads() -> Case:
    text = (
        "Leads 3/12: J. Okafor - buyer, pre-appr. $450k, wants 3BR+, N. side pref. "
        "M. Brandt - buyer, no pre-appr yet, budget ~$620k, 2BR ok. VM left for "
        "both; f/u Fri."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "J. Okafor",
            schema=Lead,
            mentions=["J. Okafor"],
            fields={
                "name": "J. Okafor",
                "budget": Decimal("450000"),
                "min_bedrooms": 3,
                "pre_approved": True,
                "area": "N. side",
            },
        )
        w.entity(
            "M. Brandt",
            schema=Lead,
            mentions=["M. Brandt"],
            fields={
                "name": "M. Brandt",
                "budget": Decimal("620000"),
                "min_bedrooms": 2,
                "pre_approved": False,
            },
        )
        return w

    return Case(
        id="crm_two_leads",
        text=text,
        schemas=lambda: [Many(Lead, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[Op("insert", who="Okafor"), Op("insert", who="Brandt")],
        expect_values=[
            Val("Okafor", "/name", "J. Okafor"),
            Val("Okafor", "/budget", 450000),
            Val("Okafor", "/min_bedrooms", 3),
            Val("Okafor", "/pre_approved", True),
            Val("Okafor", "/area", contains="N. side"),
            Val("Brandt", "/name", "M. Brandt"),
            Val("Brandt", "/budget", 620000),
            Val("Brandt", "/min_bedrooms", 2),
            Val("Brandt", "/pre_approved", False),
            Val("Brandt", "/area", None),
        ],
        must_not=[
            MustNot.value_at("Okafor", "/budget", 620000),
            MustNot.value_at("Brandt", "/budget", 450000),
            MustNot.value_at("Brandt", "/area", contains="N. side"),
            MustNot.value_at("Brandt", "/min_bedrooms", 3),
        ],
        tags=("realestate", "insert", "many", "abbreviations", "bleed", "crm"),
        world=world,
        note="A CRM note in shorthand: '$450k' is 450000, '3BR+' is at least 3 "
        "bedrooms, 'pre-appr.' / 'no pre-appr yet' are true / false, names with "
        "initials. The numbers and the area of one lead never land on the other; "
        "'VM left', '3/12' and 'f/u Fri' are not facts about either field.",
    )


# --------------------------------------------------------------------------- #
# Dotted identity
# --------------------------------------------------------------------------- #

STORES_LOG = (
    "Stores receiving log: starter generator P/N 23080-14, S/N SG-5521, "
    "serviceable. Fuel pump P/N FP-770, S/N FP-0193, unserviceable (leaking at the "
    "shaft seal)."
)


def dotted_insert_components() -> Case:
    text = STORES_LOG

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "SG-5521",
            schema=Component,
            mentions=["SG-5521"],
            fields={
                "identification": {"serial": "SG-5521", "part_number": "23080-14"},
                "condition": "serviceable",
            },
        )
        w.entity(
            "FP-0193",
            schema=Component,
            mentions=["FP-0193"],
            fields={
                "identification": {"serial": "FP-0193", "part_number": "FP-770"},
                "condition": "unserviceable",
                "remarks": "leaking at the shaft seal",
            },
        )
        return w

    return Case(
        id="dotted_insert_components",
        text=text,
        schemas=lambda: [Many(Component, identity="identification.serial")],
        options={"enable_inserts": True},
        expect_ops=[Op("insert", who="SG-5521"), Op("insert", who="FP-0193")],
        expect_values=[
            Val("SG-5521", "/identification/serial", "SG-5521"),
            Val("SG-5521", "/identification/part_number", "23080-14"),
            Val("SG-5521", "/condition", "serviceable"),
            Val("SG-5521", "/remarks", None),
            Val("FP-0193", "/identification/serial", "FP-0193"),
            Val("FP-0193", "/identification/part_number", "FP-770"),
            Val("FP-0193", "/condition", "unserviceable"),
            Val("FP-0193", "/remarks", contains="leaking at the shaft seal"),
        ],
        must_not=[
            MustNot.value_at("SG-5521", "/condition", "unserviceable"),
            MustNot.value_at("SG-5521", "/remarks", contains="leaking"),
            MustNot.value_at("FP-0193", "/identification/part_number", "23080-14"),
        ],
        tags=("aviation", "insert", "many", "dotted-identity", "bleed", "literal"),
        world=world,
        note="Many(Component, identity='identification.serial'): the identity is a "
        "field inside a sub-model, two parts told apart by serial, each with its "
        "own part number and condition (a Literal), and a remark only on the "
        "second part.",
    )


def dotted_update_components() -> Case:
    text = (
        "Inspection update: the fuel pump S/N FP-0193 is now unserviceable, seal leak "
        "found. Starter generator SG-5521 is unchanged."
    )
    existing = [
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
        (
            "part-3",
            "Component",
            {
                "identification": {"serial": "HY-3302", "part_number": "HY-90"},
                "condition": "serviceable",
                "remarks": None,
            },
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "FP-0193",
            schema=Component,
            mentions=["FP-0193"],
            fields={
                "identification": {"serial": "FP-0193"},
                "condition": "unserviceable",
                "remarks": "seal leak found",
            },
            matches="FP-0193",
        )
        w.entity(
            "SG-5521",
            schema=Component,
            mentions=["SG-5521"],
            fields={"identification": {"serial": "SG-5521"}},
            matches="SG-5521",
        )
        w.bleed("SG-5521", "condition", "unserviceable")
        return w

    return Case(
        id="dotted_update_components",
        text=text,
        schemas=lambda: [Many(Component, identity="identification.serial")],
        existing=existing,
        options={"enable_inserts": True},
        expect_ops=[Op("replace", "/condition", "unserviceable", who="part-2")],
        # 'seal leak found' is a grounded remark about the pump
        allow_ops=[Op("add", "/remarks", contains="seal leak", who="part-2")],
        must_not=[
            MustNot.op(who="part-1"),
            MustNot.op(who="part-3"),
            MustNot.value_at("part-1", "/condition", "unserviceable"),
            MustNot.insert(),
            MustNot.delete(),
            MustNot.op("replace", "/identification"),
        ],
        tags=("aviation", "update", "many", "dotted-identity", "bleed"),
        world=world,
        note="Existing records matched through identification.serial: the pump "
        "(FP-0193) becomes unserviceable, the generator named next to it stays "
        "as it is (the word 'unserviceable' is not its), the third record is "
        "never mentioned, and nothing is inserted or deleted.",
    )


# --------------------------------------------------------------------------- #
# Updates
# --------------------------------------------------------------------------- #


def visit_reschedule_time_only() -> Case:
    text = "Tomas Greer's check-up moves to 3:30 pm. Same day, same room."
    existing = [
        (
            "visit-204",
            "Visit",
            {
                "patient_name": "Tomas Greer",
                "visit_date": "2026-04-02",
                "start_time": "10:00:00",
                "room": "Room 3",
            },
        )
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Visit",
            schema=Visit,
            mentions=["Tomas Greer"],
            fields={"start_time": time(15, 30)},
        )
        return w

    return Case(
        id="visit_reschedule_time_only",
        text=text,
        schemas=lambda: [One(Visit)],
        existing=existing,
        expect_ops=[Op("replace", "/start_time", "15:30:00", who="visit-204")],
        expect_values=[
            Val("visit-204", "/visit_date", "2026-04-02"),
            Val("visit-204", "/room", "Room 3"),
        ],
        must_not=[
            MustNot.op("replace", "/visit_date"),
            MustNot.op("replace", "/room"),
            MustNot.op("replace", "/patient_name"),
            MustNot.op("remove"),
        ],
        tags=("healthcare", "update", "time"),
        world=world,
        note="A time is rescheduled; 'same day, same room' restate the record and "
        "change nothing else. One operation, on the time only.",
    )


def account_rename() -> Case:
    text = (
        "Orbit Freight has changed its name to Orbit Logistics. Quillon Foods is "
        "unchanged."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Orbit Logistics",
            schema=Account,
            mentions=["Orbit Freight", "Orbit Logistics"],
            fields={"company_name": "Orbit Logistics"},
            matches="Orbit Freight",
        )
        w.entity(
            "Quillon Foods",
            schema=Account,
            mentions=["Quillon Foods"],
            fields={"company_name": "Quillon Foods"},
            matches="Quillon Foods",
        )
        return w

    return Case(
        id="account_rename",
        text=text,
        schemas=lambda: [Many(Account, identity="company_name")],
        existing=ACCOUNTS,
        options={"enable_inserts": True},
        expect_ops=[Op("replace", "/company_name", "Orbit Logistics", who="acct-1")],
        must_not=[
            MustNot.insert(),
            MustNot.delete(),
            MustNot.op(who="acct-2"),
            MustNot.op("replace", "/plan"),
            MustNot.op("replace", "/seats"),
        ],
        tags=("saas", "update", "many", "identity"),
        world=world,
        note="A company renamed: the old and the new name are one account, so the "
        "identity field is replaced (not a new account, not a delete); the account "
        "named in the next sentence as unchanged gets no operation.",
    )


def pallet_correction() -> Case:
    text = (
        "Actually, CN-88213 is 12 pallets, not 14. Everything else on the manifest "
        "stands."
    )
    existing = [("CN-88213", "Shipment", SHIPMENT_RECORD)]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Shipment",
            schema=Shipment,
            mentions=["CN-88213"],
            fields={"reference": "CN-88213", "pallets": 12},
        )
        return w

    return Case(
        id="pallet_correction",
        text=text,
        schemas=lambda: [One(Shipment)],
        existing=existing,
        expect_ops=[Op("replace", "/pallets", 12, who="CN-88213")],
        must_not=[
            MustNot.op("remove"),
            MustNot.op("replace", "/gross_weight_kg"),
            MustNot.op("replace", "/declared_value_eur"),
            MustNot.op("replace", "/reference"),
            MustNot.op("replace", "/pallets", value=14),
        ],
        tags=("logistics", "update", "correction"),
        world=world,
        note="'Actually it's 12, not 14': an explicit correction states the new "
        "number outright, so it replaces the old one (this is not a relative edit); "
        "nothing else on the record changes.",
    )


def ticket_status_and_note() -> Case:
    text = (
        "Update on TK-4471: the customer says it only fails in Safari. I've set the "
        "ticket to pending customer."
    )
    existing = [
        (
            "TK-4471",
            "SupportCase",
            {
                "ticket_id": "TK-4471",
                "status": "open",
                "notes": ["Customer reports SSO failures after the 09:10 deploy."],
            },
        )
    ]
    note = "Update on TK-4471: the customer says it only fails in Safari."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the SupportCase",
            schema=SupportCase,
            mentions=["TK-4471"],
            fields={"status": "pending_customer", "notes": [note]},
        )
        return w

    return Case(
        id="ticket_status_and_note",
        text=text,
        schemas=lambda: [One(SupportCase)],
        existing=existing,
        expect_ops=[
            Op("replace", "/status", "pending_customer", who="TK-4471"),
            Op("add", "/notes/-", contains="only fails in Safari", who="TK-4471"),
        ],
        # the second sentence, as a note, repeats the status change: grounded
        allow_ops=[Op("add", "/notes/-", contains="pending customer", who="TK-4471")],
        must_not=[
            MustNot.op("remove"),
            MustNot.op("replace", "/notes"),
            MustNot.op("replace", "/ticket_id"),
        ],
        tags=("saas", "update", "notes", "literal"),
        world=world,
        note="A Literal changed in words ('pending customer' is pending_customer) "
        "and one new fact appended to the notes; the existing note stays and is "
        "never rewritten.",
    )


def seats_relative_refused() -> Case:
    text = "Team Quokka asked to add two more seats to RES-9."
    existing = [
        (
            "res-9",
            "Reservation",
            {"booking_ref": "RES-9", "team_name": "Team Quokka", "seats": 4},
        )
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Reservation",
            schema=Reservation,
            mentions=["RES-9"],
            fields={"seats": 2},
        )
        return w

    return Case(
        id="seats_relative_refused",
        text=text,
        schemas=lambda: [One(Reservation)],
        existing=existing,
        # documented limit (docs/DESIGN.md, "Relative edits are not supported"): "two
        # more" is a change relative to the record's 4. The library neither writes the
        # stated 2 (data loss) nor computes 6, so the gold is NO operation and one
        # rejection, relative_edit_unsupported, on the seats.
        expect_ops=[],
        expect_rejected=[Rejected("relative_edit_unsupported", "/seats", "res-9")],
        must_not=[
            MustNot.op("replace", "/seats", who="res-9"),
            MustNot.value_at("res-9", "/seats", 2),
            MustNot.value_at("res-9", "/seats", 6),
        ],
        tags=("events", "update", "relative"),
        world=world,
        note="'Two more seats' on a booking of 4: relative, so refused as "
        "relative_edit_unsupported; neither 2 (the stated amount) nor 6 (the sum) "
        "may be written. Documented behaviour, not a model failure to be rewarded.",
    )


def visit_cancel_one() -> Case:
    text = (
        "Please cancel Rosa Bellamy's 2:00 pm slot; she has moved to another clinic. "
        "Everyone else stays as booked."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Rosa Bellamy",
            schema=Visit,
            mentions=["Rosa Bellamy"],
            fields={"patient_name": "Rosa Bellamy", "start_time": time(14, 0)},
            matches="Rosa Bellamy",
        )
        w.delete("visit-1", 0.97)
        return w

    return Case(
        id="visit_cancel_one",
        text=text,
        schemas=lambda: [Many(Visit, identity="patient_name")],
        existing=VISITS,
        options={"enable_deletes": True},
        expect_ops=[Op("delete", who="visit-1")],
        must_not=[
            MustNot.delete("visit-2"),
            MustNot.delete("visit-3"),
            MustNot.insert(),
            MustNot.op("replace"),
        ],
        tags=("healthcare", "delete", "many"),
        world=world,
        note="A cancellation names one person's slot: that record is deleted and "
        "the other two are not (the closing sentence does not name them). The "
        "'2:00 pm' repeats the record's own time and is no change.",
    )


def listing_no_longer_pets() -> Case:
    text = (
        "48 Alder Row, Unit 7 is no longer pet-friendly (the HOA changed its rules), "
        "but it is still on the market."
    )
    existing = [
        ("L-1", "Listing", listing_record("48 Alder Row, Unit 7")),
        (
            "L-2",
            "Listing",
            listing_record(
                "9 Birch Lane", property_type="house", asking_price="415000"
            ),
        ),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "48 Alder Row, Unit 7",
            schema=Listing,
            mentions=["48 Alder Row, Unit 7"],
            fields={"address": "48 Alder Row, Unit 7", "pets_allowed": False},
            matches="48 Alder Row",
        )
        return w

    return Case(
        id="listing_no_longer_pets",
        text=text,
        schemas=lambda: [Many(Listing, identity="address")],
        existing=existing,
        options={"enable_deletes": True},
        expect_ops=[Op("replace", "/pets_allowed", False, who="L-1")],
        must_not=[
            MustNot.delete(),
            MustNot.insert(),
            MustNot.op(who="L-2"),
            MustNot.op("replace", "/address"),
        ],
        tags=("realestate", "update", "delete", "negative", "bool"),
        world=world,
        note="'No longer pet-friendly ... still on the market': a bool turns false "
        "and the listing is NOT deleted (a 'no longer' about one attribute is a "
        "change, not a removal).",
    )


def update_and_delete_conflict() -> Case:
    text = (
        "Please remove Quillon Foods from the account list, and note that they are "
        "now at 40 seats."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Quillon Foods",
            schema=Account,
            mentions=["Quillon Foods"],
            fields={"company_name": "Quillon Foods", "seats": 40},
            matches="Quillon Foods",
        )
        w.delete("acct-2", 0.97)
        return w

    return Case(
        id="update_and_delete_conflict",
        text=text,
        schemas=lambda: [Many(Account, identity="company_name")],
        existing=ACCOUNTS,
        options={"enable_deletes": True},
        # documented behaviour (docs/DESIGN.md, collection engine step 9): an input
        # that both removes and changes the same record is a conflict
        # (SkippedEntity conflict_update_delete): neither the delete nor the update is
        # applied. A careful reader agrees: the message contradicts itself.
        expect_ops=[],
        must_not=[
            MustNot.delete(),
            MustNot.op("replace"),
            MustNot.insert(),
        ],
        tags=("saas", "update", "delete", "conflict"),
        world=world,
        note="The text removes an account and updates its seats in one breath. "
        "Documented behaviour: a conflict, so nothing is applied (the record is "
        "reported as skipped, conflict_update_delete), not a delete and not a "
        "change. The other account is not mentioned.",
    )


def account_update_bleed() -> Case:
    text = (
        "Orbit Freight moved to the Enterprise plan this week. Quillon Foods stays "
        "on Starter, with 5 seats as before."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Orbit Freight",
            schema=Account,
            mentions=["Orbit Freight"],
            fields={"company_name": "Orbit Freight", "plan": "enterprise"},
            matches="Orbit Freight",
        )
        w.entity(
            "Quillon Foods",
            schema=Account,
            mentions=["Quillon Foods"],
            fields={"company_name": "Quillon Foods", "plan": "starter", "seats": 5},
            matches="Quillon Foods",
        )
        w.bleed("Quillon Foods", "plan", "enterprise")
        return w

    return Case(
        id="account_update_bleed",
        text=text,
        schemas=lambda: [Many(Account, identity="company_name")],
        existing=ACCOUNTS,
        expect_ops=[Op("replace", "/plan", "enterprise", who="acct-1")],
        must_not=[
            MustNot.op(who="acct-2"),
            MustNot.value_at("acct-2", "/plan", "enterprise"),
            MustNot.op("replace", "/seats", who="acct-1"),
        ],
        tags=("saas", "update", "many", "bleed", "literal"),
        world=world,
        note="One account upgrades, the next one is stated to be unchanged: the "
        "'Enterprise' plan belongs to the first and must not reach the second; "
        "the second's restated plan and seats equal its record (no operation).",
    )


def property_insert_bleed() -> Case:
    text = (
        "Two new listings: 42 Alder Row asks $529,900. 9 Birch Lane asks $415,000 "
        "and has a two-car garage."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "42 Alder Row",
            schema=Property,
            mentions=["42 Alder Row"],
            fields={"address": "42 Alder Row", "asking_price": Decimal("529900")},
        )
        w.entity(
            "9 Birch Lane",
            schema=Property,
            mentions=["9 Birch Lane"],
            fields={
                "address": "9 Birch Lane",
                "asking_price": Decimal("415000"),
                "garage": "two-car garage",
            },
        )
        w.bleed("42 Alder Row", "garage", "two-car garage")
        return w

    return Case(
        id="property_insert_bleed",
        text=text,
        schemas=lambda: [Many(Property, identity="address")],
        options={"enable_inserts": True},
        expect_ops=[Op("insert", who="Alder Row"), Op("insert", who="Birch Lane")],
        expect_values=[
            Val("Alder Row", "/asking_price", 529900),
            Val("Alder Row", "/garage", None),
            Val("Birch Lane", "/asking_price", 415000),
            Val("Birch Lane", "/garage", contains="two-car garage"),
        ],
        must_not=[
            MustNot.value_at("Alder Row", "/garage", contains="garage"),
            MustNot.value_at("Alder Row", "/asking_price", 415000),
            MustNot.value_at("Birch Lane", "/asking_price", 529900),
        ],
        tags=("realestate", "insert", "many", "bleed", "decimal"),
        world=world,
        note="The garage is mentioned only for the second property, right after "
        "its price; the first property has none and must stay null.",
    )


HOLDOUT_CASES: list[Case] = [
    ship_scalars(),
    booking_dates(),
    appointment_time_duration(),
    listing_basic(),
    ticket_enum(),
    nutrition_label(),
    event_registration(),
    work_order(),
    po_table_lines(),
    aircraft_spec_insert(),
    recipe_ingredients(),
    aircraft_engine_swap_update(),
    clinic_schedule_table(),
    support_thread_participants(),
    similar_names_patients(),
    agents_vs_companies(),
    meeting_transcript(),
    crm_two_leads(),
    dotted_insert_components(),
    dotted_update_components(),
    visit_reschedule_time_only(),
    account_rename(),
    pallet_correction(),
    ticket_status_and_note(),
    seats_relative_refused(),
    visit_cancel_one(),
    listing_no_longer_pets(),
    update_and_delete_conflict(),
    account_update_bleed(),
    property_insert_bleed(),
]
"""Every holdout case, in the order reports list them."""

assert len({c.id for c in HOLDOUT_CASES}) == len(HOLDOUT_CASES), (
    "case ids must be unique"
)


def holdout_cases() -> list[Case]:
    """The holdout suite (a copy of the list; the cases are shared)."""
    return list(HOLDOUT_CASES)


if __name__ == "__main__":  # offline: `uv run python -m evals.holdout_cases`
    import asyncio

    from evals.report import suite_markdown
    from evals.runner import Settings, aggregate, run_suite

    outcome = asyncio.run(run_suite(HOLDOUT_CASES, Settings(), config="holdout"))
    print(suite_markdown(outcome))
    totals = aggregate(outcome.results)
    raise SystemExit(0 if totals["passed"] == totals["cases"] else 1)
