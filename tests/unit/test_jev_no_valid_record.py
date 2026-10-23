"""Regression tests for the TypeSafe (Jev) path: ``no_valid_record`` and a wrong span.

Every test here is built from the SHAPE of a live recording (the text, the options
Jev was offered, the probabilities it answered, the attribution it gave), never from
the recordings themselves. The recordings are in ``docs/DESIGN.md`` ("Known limits of
selection") by case and cause.

* a pick whose value is offered in several shapes (span, sentence, block) is judged by
  the odds of the value, not of one shape;
* a table cell is an option;
* a date written as a distance from today ("three days ago", "in two weeks") is read;
* the attribution of a date says what day it is;
* the repair keeps a required text value that the attribution could neither confirm
  nor refute;
* an initial ("J. Okafor") is one name, and the period of an initial ends no sentence;
* the span of an identity taken from its mention is the value's own words.
"""

# ruff: noqa: UP045 - tests deliberately spell Optional and Union

from datetime import date
from typing import Any, Optional

import pytest
from pydantic import BaseModel

from evals.invariants import spans_quote_the_input
from tests.unit.extract_helpers import Script, ca, make_extractor
from tests.unit.test_typesafe_transport import (
    Jev,
    choice,
    clairic,
    clairic_payload,
    make,
)
from trustcall_system_one import Many, RepairConfig, TypeSafeConfig, create_extractor
from trustcall_system_one._candidates import structure_pieces
from trustcall_system_one._collection import parse_mentions
from trustcall_system_one._dateparts import Part, assemble_date, date_questions
from trustcall_system_one._entity import _own_span, _proper_noun, _sentence_initial
from trustcall_system_one._structure import build_structure, to_lines
from trustcall_system_one._typesafe import _apposition
from trustcall_system_one._wire import ClairicAnswer
from trustcall_system_one.testing import FakeWorld

__all__: list[str] = []

# --------------------------------------------------------------------------- #
# A value offered in several shapes (nutrition_label, recipe, a name of a record)
# --------------------------------------------------------------------------- #

NUTRITION = (
    "Hearthside Maple Granola\nNutrition Facts\nServing size 45 g\n"
    "Servings per container 8\nCalories 190\nBest before: NOV 2026\n"
)
PRODUCT = {
    "product_name": {
        "type": "text",
        "instructions": "the NutritionLabel's product's name",
        "null_threshold": 0.7,
    }
}


def _spread(
    shares: dict[str, float], p_none: float = 0.02, confidence: float = 0.4
) -> Any:
    """A script: ``shares`` is the odds of every option that SAYS a value.

    An option says a value when its id is the value or its description quotes it
    (``sentence: 'Hearthside Maple Granola'``).
    """

    def script(qid: str, question: dict[str, Any], body: dict[str, Any]) -> Any:
        if not qid.endswith("::pick"):
            return None
        probabilities = {o: 0.001 for o in question["criteria"]}
        for option, description in question["criteria"].items():
            for value, share in shares.items():
                if option == value or f"'{value}'" in str(description):
                    probabilities[option] = share
        probabilities["none"] = p_none
        top = max(probabilities, key=lambda o: probabilities[o])
        return {
            "type": "choice",
            "choice": top,
            "confidence": confidence,  # Jev's own, low: the options share the odds
            "probabilities": probabilities,
        }

    return script


def test_the_shapes_of_one_value_pool_their_odds_and_their_confidence() -> None:
    # live (nutrition_label): span, sentence and block of the product name share the
    # odds, so Jev's confidence in the chosen one is 0.42: below `min_confidence`
    jev = Jev(_spread({"Hearthside Maple Granola": 0.3}))
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(PRODUCT, NUTRITION),
    )
    answer = sent.body["answers"]["product_name"]
    assert answer["value"] == "Hearthside Maple Granola"
    assert answer["confidence"] >= 0.85  # three shapes of 0.3 against none at 0.02
    assert answer["p_null"] == pytest.approx(0.02)
    start, end = answer["span"]
    assert sent.body["state_text"][start:end] == "Hearthside Maple Granola"


def test_the_pooled_value_beats_a_single_rival_option() -> None:
    # the product name is three shapes of 0.2 (0.6); the number 190 is one option at
    # 0.3: the best single option is the rival, the best VALUE is the product name
    jev = Jev(_spread({"Hearthside Maple Granola": 0.2, "190": 0.3}))
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(PRODUCT, NUTRITION),
    )
    answer = sent.body["answers"]["product_name"]
    assert answer["value"] == "Hearthside Maple Granola"
    assert answer["guess"] == "Hearthside Maple Granola"
    assert answer["confidence"] >= 0.4  # never lower than Jev's own


def test_an_option_that_stands_alone_keeps_jevs_confidence() -> None:
    email = {"email": {"type": "text", "instructions": "the email address"}}
    jev = Jev(
        lambda qid, q, body: (
            choice(q, "dana.whit@acme-corp.com", 0.93) if qid == "email::pick" else None
        )
    )
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(email),
    )
    assert sent.body["answers"]["email"]["confidence"] == 0.93


def test_shapes_that_outweigh_none_are_a_choice_of_the_value() -> None:
    # `none` is Jev's single likeliest option (0.40) against three shapes of 0.2
    # (0.60 for the value): the value was chosen, it is not "not stated"
    jev = Jev(_spread({"Hearthside Maple Granola": 0.2}, p_none=0.4))
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(PRODUCT, NUTRITION),
    )
    answer = sent.body["answers"]["product_name"]
    assert answer["value"] == "Hearthside Maple Granola"
    assert answer["p_null"] == pytest.approx(0.4)


def test_none_still_wins_when_the_value_is_not_the_likelier() -> None:
    jev = Jev(_spread({"Hearthside Maple Granola": 0.1}, p_none=0.6))
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(PRODUCT, NUTRITION),
    )
    answer = sent.body["answers"]["product_name"]
    assert answer["value"] is None and answer["p_null"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# The cells of a table (clinic_schedule_table, po_table_lines)
# --------------------------------------------------------------------------- #

CLINIC = (
    "Clinic day sheet, Lakeside Family Clinic\n"
    "| Code | Time | Room | Visit |\n"
    "|------|------|------|-------|\n"
    "| PT-1041 | 09:00 | Room 2 | checkup |\n"
    "| PT-1042 | 09:30 | Room 3 | vaccination |\n"
)


def test_a_cell_of_a_table_row_is_a_candidate_of_its_own() -> None:
    structure = build_structure(CLINIC, to_lines(CLINIC), {})
    cells = {c.text: c for c in structure.candidates if "." in c.id}
    for cell in ("PT-1042", "09:30", "Room 3", "vaccination", "Room", "Visit"):
        assert cell in cells, cell
        assert CLINIC[cells[cell].start : cells[cell].end] == cell
    # the separator row holds no cell, and a cell is not the row around it
    assert not [c for c in cells if set(c) <= set("-|")]
    assert "| PT-1042 | 09:30 | Room 3 | vaccination |" not in cells


def test_a_tab_separated_field_is_a_clause_too() -> None:
    found = [
        c.text
        for c in structure_pieces("PT-1042\t09:30\tRoom 3\tvaccination")["clause"]
    ]
    assert found == ["PT-1042", "09:30", "Room 3", "vaccination"]


def test_a_comma_inside_a_time_or_a_number_is_still_no_cell_edge() -> None:
    found = [
        c.text for c in structure_pieces("| 09:30 | $1,200.00 | Room 3 |")["clause"]
    ]
    assert found == ["09:30", "$1,200.00", "Room 3"]


def test_jev_can_pick_the_room_cell_of_the_row_it_was_asked_about() -> None:
    def script(qid: str, question: dict[str, Any], body: dict[str, Any]) -> Any:
        if qid != "room::pick":
            return None
        # the cell of the second data row: the option whose description names it
        option = next(
            o for o, d in question["criteria"].items() if d == "part of S004: 'Room 3'"
        )
        return choice(question, option)

    fields = {"room": {"type": "text", "instructions": "PT-1042's room"}}
    jev = Jev(script)
    sent = clairic(
        make(jev, config=TypeSafeConfig(verify_selection=False)),
        clairic_payload(fields, CLINIC),
    )
    answer = sent.body["answers"]["room"]
    assert answer["value"] == "Room 3"
    start, end = answer["span"]
    assert sent.body["state_text"][start:end] == "Room 3"


# --------------------------------------------------------------------------- #
# A date given as a distance from today (booking_dates, ticket_enum)
# --------------------------------------------------------------------------- #

TODAY = date(2026, 3, 23)  # a Monday


def _offset(count: str, unit: str, *, p: float = 0.97) -> dict[str, Part]:
    def part(choice_: str) -> Part:
        return Part(choice_, p, {choice_: p, "none": 1 - p})

    return {
        "mode": part("relative"),
        "day_anchor": part("offset"),
        "offset_count": part(count),
        "offset_unit": part(unit),
    }


@pytest.mark.parametrize(
    ("count", "unit", "expected"),
    [
        ("3", "days_ago", "2026-03-20"),  # "Reported three days ago"
        ("14", "days_ahead", "2026-04-06"),
        ("2", "weeks_ahead", "2026-04-06"),  # "the delivery window opens in two weeks"
        ("1", "weeks_ago", "2026-03-16"),
        ("1", "months_ago", "2026-02-23"),
        ("2", "months_ahead", "2026-05-23"),
        ("1", "years_ago", "2025-03-23"),
    ],
)
def test_a_distance_from_today_is_resolved_in_code(
    count: str, unit: str, expected: str
) -> None:
    result = assemble_date(_offset(count, unit), TODAY)
    assert result.value == expected
    assert result.confidence == pytest.approx(0.97)
    assert not result.inferred and not result.needs_review


def test_a_month_that_is_shorter_stops_on_its_last_day() -> None:
    assert assemble_date(_offset("1", "months_ago"), date(2026, 3, 31)).value == (
        "2026-02-28"
    )
    assert assemble_date(_offset("1", "years_ahead"), date(2028, 2, 29)).value == (
        "2029-02-28"
    )
    assert assemble_date(_offset("1", "months_ahead"), date(2026, 12, 31)).value == (
        "2027-01-31"
    )


def test_an_offset_whose_parts_were_not_read_is_no_date() -> None:
    result = assemble_date(_offset("none", "days_ago"), TODAY)
    assert result.value is None and result.note == "relative offset not read"
    result = assemble_date(_offset("3", "none"), TODAY)
    assert result.value is None and result.needs_review
    # an answer without the offset parts at all (an older response) is no date either
    bare = {
        k: v for k, v in _offset("3", "days_ago").items() if k in ("mode", "day_anchor")
    }
    assert assemble_date(bare, TODAY).value is None


def test_the_date_questions_name_the_offset() -> None:
    questions = date_questions("the date of the report")
    anchors = questions["day_anchor"]["criteria"]
    assert "offset" in anchors and "none" in anchors
    assert "three days ago" in questions["day_anchor"]["instructions"]
    assert "60" in questions["offset_count"]["criteria"]
    units = list(questions["offset_unit"]["criteria"])
    assert units[:2] == ["days_ago", "days_ahead"] and units[-1] == "none"


def test_jev_reads_three_days_ago_through_the_whole_adapter() -> None:
    parts = {
        "mode": "relative",
        "day_anchor": "offset",
        "offset_count": "3",
        "offset_unit": "days_ago",
    }

    def script(qid: str, question: dict[str, Any], body: dict[str, Any]) -> Any:
        _, _, part = qid.partition("::")
        return choice(question, parts[part], 0.97) if part in parts else None

    fields = {
        "reported_on": {
            "type": "date",
            "instructions": "the date the issue was reported",
        }
    }
    payload = clairic_payload(
        fields,
        "Customers cannot sign in. Reported three days ago.",
        reference_date="2026-03-23",
    )
    answer = clairic(make(Jev(script)), payload).body["answers"]["reported_on"]
    assert answer["value"] == "2026-03-20"
    assert answer["needs_review"] is False


# --------------------------------------------------------------------------- #
# The attribution of a date says what day it is (work_order, booking_dates)
# --------------------------------------------------------------------------- #


class WorkOrder(BaseModel):
    """An aircraft maintenance work order."""

    wo_number: str
    completed_on: date


WORK_ORDER = "WO 55-1182: replaced the brake pad set. Completed last Tuesday."


def _work_order_world() -> Script:
    world = Script()
    world.clairic = {
        "wo_number": ca("text", "WO 55-1182", find="WO 55-1182"),
        "completed_on": ca("date", "2026-03-17"),
    }
    return world


async def test_the_attribution_of_a_date_says_what_day_it_is() -> None:
    world = _work_order_world()
    extractor = make_extractor([WorkOrder], world, reference_date="2026-03-23")
    await extractor.ainvoke(WORK_ORDER)
    asked = {
        qid: q["instructions"]
        for _, payload, _ in world.calls("attribution")
        for qid, q in payload["questions"].items()
    }
    assert asked["attr.completed_on"].startswith(
        "Today is Monday, 2026-03-23. Does `input` state that"
    )
    # the number of a record is not a date: it is asked as it always was
    assert asked["attr.wo_number"].startswith("Does `input` state that")


async def test_without_a_reference_date_the_attribution_is_as_it_was() -> None:
    world = _work_order_world()
    await make_extractor([WorkOrder], world).ainvoke(WORK_ORDER)
    asked = [
        q["instructions"]
        for _, payload, _ in world.calls("attribution")
        for q in payload["questions"].values()
    ]
    assert asked and not [a for a in asked if "Today is" in a]


# --------------------------------------------------------------------------- #
# The repair keeps a required text value that is neither confirmed nor refuted
# (appointment_time_duration, nutrition_label, notes_insert_contact, bill_insert)
# --------------------------------------------------------------------------- #


class Appointment(BaseModel):
    """A clinic appointment reminder."""

    patient_name: str
    clinic: str
    visit_date: Optional[date] = None


REMINDER = "Reminder for Tomas Greer: follow-up visit at Lakeside Family Clinic."


def _appointment_world(attribution: list[float]) -> Script:
    world = Script()
    world.clairic = {
        "patient_name": ca("text", "Tomas Greer", find="Tomas Greer", confidence=0.63),
        "clinic": ca("text", "Lakeside Family Clinic", find="Lakeside Family Clinic"),
        "visit_date": ca("date", None, p_null=0.9),
    }
    world.reask = {"patient_name": ca("text", "Tomas Greer", find="Tomas Greer")}
    # what Jev answered (appointment_time_duration): 0.44, then 0.46 on the second look
    world.attribution = {"attr.patient_name": attribution}
    return world


async def test_an_uncertain_attribution_does_not_lose_the_whole_record() -> None:
    world = _appointment_world([0.44, 0.46])
    result = await make_extractor([Appointment], world).ainvoke(REMINDER)
    (record,) = result.records
    assert record.valid and record.after is not None
    assert record.after["patient_name"] == "Tomas Greer"
    assert record.after["clinic"] == "Lakeside Family Clinic"
    warnings = record.fields["patient_name"].warnings
    assert any("attribution_uncertain: p=0.46" in w for w in warnings)
    assert any("repair[1]" in w for w in warnings)
    # the first pass was as strict as ever: it threw the value away before the repair
    assert any("before the repair" in w for w in warnings)


async def test_the_knob_turns_it_back_off() -> None:
    world = _appointment_world([0.44, 0.46])
    extractor = make_extractor(
        [Appointment], world, repair=RepairConfig(keep_uncertain=False)
    )
    (record,) = (await extractor.ainvoke(REMINDER)).records
    assert not record.valid


async def test_a_value_the_attribution_refutes_never_stands() -> None:
    world = _appointment_world([0.30, 0.30])  # below `attribution_drop` (0.4)
    (record,) = (await make_extractor([Appointment], world).ainvoke(REMINDER)).records
    assert not record.valid


async def test_a_confirmed_value_is_not_noted_as_uncertain() -> None:
    world = _appointment_world([0.95])
    (record,) = (await make_extractor([Appointment], world).ainvoke(REMINDER)).records
    assert record.valid
    assert not [w for w in record.fields["patient_name"].warnings if "uncertain" in w]


class Reading(BaseModel):
    """A reading of a meter."""

    meter: str
    value: float


async def test_a_required_number_is_not_kept_on_an_uncertain_attribution() -> None:
    # only the text of the input is kept: a number may be derived or asked for
    world = Script()
    world.clairic = {
        "meter": ca("text", "M-12", find="M-12"),
        "value": ca("number", 12.5, find="12.5"),
    }
    world.reask = {"value": ca("number", 12.5, find="12.5")}
    world.attribution = {"attr.value": [0.5]}
    result = await make_extractor([Reading], world).ainvoke("Meter M-12 reads 12.5.")
    (record,) = result.records
    assert not record.valid


# --------------------------------------------------------------------------- #
# An initial is one name (crm_two_leads)
# --------------------------------------------------------------------------- #

LEADS = (
    "Leads 3/12: J. Okafor - buyer, budget $450k, pre-appr., wants 3BR+, N. side "
    "pref. M. Brandt - buyer, no pre-appr yet, budget ~$620k, min 2BR."
)


def _mentions(text: str, values: list[str]) -> list[tuple[str, tuple[int, int] | None]]:
    answer = ClairicAnswer.model_validate({"type": "list", "value": values})
    return [(m.text, m.span) for m in parse_mentions(text, answer, None)]


def test_an_initial_and_a_surname_are_one_mention_not_two() -> None:
    # live: "J." and "Okafor" were kept and "J. Okafor" dropped, because the period
    # that the cleaning took off the initial was left in the gap between the parts
    found = _mentions(LEADS, ["J.", "J. Okafor", "Okafor", "M.", "M. Brandt", "Brandt"])
    assert found == [("J. Okafor", (12, 21)), ("M. Brandt", (82, 91))]


def test_two_names_across_a_sentence_end_stay_two() -> None:
    text = "I met Emma. Tom came later."
    found = _mentions(text, ["Emma", "Tom", "Emma. Tom"])
    assert [t for t, _ in found] == ["Emma", "Tom"]


def test_the_period_of_an_initial_or_a_title_ends_no_sentence() -> None:
    assert not _sentence_initial(LEADS, LEADS.index("Okafor"))  # "J. Okafor"
    ask = "Ask Dr. Wu about it."
    assert not _sentence_initial(ask, ask.index("Wu"))
    rained = "It rained. Okafor left."
    assert _sentence_initial(rained, rained.index("Okafor"))  # a real sentence end


def test_a_surname_after_an_initial_is_a_name() -> None:
    at = LEADS.index("Okafor")
    assert _proper_noun("Okafor", LEADS, (at, at + 6))
    assert _proper_noun("J. Okafor", LEADS, (12, 21))


def test_a_clause_that_continues_a_name_after_a_dash_is_an_apposition() -> None:
    from trustcall_system_one._candidates import Candidate

    def candidate(text: str, start: int = 0) -> Candidate:
        return Candidate(text, start, start + len(text), text, "clause")

    name = candidate("J. Okafor")
    assert _apposition(candidate("J. Okafor - buyer"), name)
    assert _apposition(candidate("J. Okafor, buyer"), name)
    assert _apposition(candidate("J. Okafor: buyer"), name)
    assert _apposition(candidate("J. Okafor (buyer)"), name)
    assert not _apposition(candidate("J. Okafor-Smith"), name)  # a hyphenated name
    assert not _apposition(candidate("J. Okafor's brother"), name)


# --------------------------------------------------------------------------- #
# Defect 2: the span of an identity is its own words (smoke ``people``)
# --------------------------------------------------------------------------- #

FRIEND = (
    "Friend: I wanted to introduce you to my cousin. Her name is Olivia Davis, "
    "she's a 27-year-old graphic designer. She's looking to meet new people."
)


class Person(BaseModel):
    """A person."""

    name: str
    occupation: Optional[str] = None


async def test_the_span_of_an_identity_from_a_clause_mention_is_the_name() -> None:
    # live (smoke people, intermittent): Jev listed the clause "Her name is Olivia
    # Davis" among the mentions, the longer mention replaced "Olivia Davis", the name
    # field was unsure and took the entity's label with the CLAUSE's span
    world = FakeWorld(FRIEND)
    olivia = world.entity(
        "Olivia Davis",
        schema=Person,
        mentions=["Her name is Olivia Davis", "Olivia Davis"],
        fields={"name": "Olivia Davis", "occupation": "graphic designer"},
    )
    world.low_confidence(olivia, "name", 0.2, persistent=True)
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=world.transport(),
        enable_inserts=True,
    )
    result = await extractor.ainvoke(FRIEND)
    (record,) = [r for r in result.records if r.valid]
    answer = record.fields["name"]
    assert answer.value == "Olivia Davis"
    assert any("identity_from_mention" in w for w in answer.warnings)  # the path
    assert answer.span is not None
    assert FRIEND[answer.span[0] : answer.span[1]] == "Olivia Davis"
    spans_quote_the_input(result, FRIEND)  # evals/invariants.py: the smoke's check


def test_the_own_span_of_a_value_is_its_words_inside_the_mention() -> None:
    clause = (FRIEND.index("Her name"), FRIEND.index("Olivia Davis") + 12)
    start = FRIEND.index("Olivia Davis")
    assert _own_span(FRIEND, "Olivia Davis", clause) == (start, start + 12)
    # a mention that already is the value keeps its span; no mention at all: its words
    assert _own_span(FRIEND, "Olivia Davis", (start, start + 12)) == (start, start + 12)
    # case and spacing do not matter, and the closest occurrence wins
    text = "Dana  Ruiz called. Later dana ruiz wrote."
    near = (text.index("Later"), len(text))
    found = _own_span(text, "Dana Ruiz", near)
    assert found is not None and text[found[0] : found[1]] == "dana ruiz"
    # the input does not hold the words: no span, never a wrong one
    assert _own_span(FRIEND, "Olivia Smith", clause) is None
    assert _own_span(FRIEND, "", clause) is None
    assert _own_span("Ines wrote.", "Ines", None) == (0, 4)
