"""Regressions of the review of the escalation tier (route mode and its verification).

Every text is synthetic. The scripted worlds and helpers are those of
``test_escalation``.
"""

import datetime as dt
from collections import Counter
from types import SimpleNamespace
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import EscalationConfig, Many, One, create_extractor
from trustcall_system_one._coerce import coerce
from trustcall_system_one._compile import compile_schema
from trustcall_system_one._escalate import (
    _ask_rows,
    _check_date,
    _overlap,
    _pair_entities,
    _row_of,
    _row_specs,
    _verify_rows,
    locate,
    tiebreak,
)
from trustcall_system_one._relative import find_phrases, plausible, resolve_phrase
from trustcall_system_one._run import Run
from trustcall_system_one._types import EntityInfo, Mention
from trustcall_system_one._verify import (
    Ctx,
    FieldState,
    Resolved,
    _attribution_question,
    _clause_of,
)
from trustcall_system_one._wire import ClairicAnswer
from trustcall_system_one.testing import FakeEscalation, FakeTransport, FakeWorld

TODAY = dt.date(2026, 10, 23)


# --------------------------------------------------------------------------- #
# S5: "N days later" counts from another date; "in N weeks" can be a duration
# --------------------------------------------------------------------------- #

ANCHORED = (
    "The aircraft arrived on 2026-03-01 and was inspected two days later. "
    "The overhaul took place in three weeks."
)


@pytest.mark.parametrize("phrase", ["two days later", "a week earlier", "3 days later"])
def test_a_phrase_counted_from_another_date_is_not_a_relative_date(phrase: str) -> None:
    assert resolve_phrase(phrase, TODAY) is None
    assert resolve_phrase(phrase, TODAY, context=f"inspected {phrase}") is None
    # nothing to judge: it is neither found nor refused or accepted by code
    assert plausible(phrase, dt.date(2026, 3, 3), TODAY) is True
    assert plausible(phrase, dt.date(2026, 10, 25), TODAY) is True
    assert find_phrases(f"It was inspected {phrase}.") == []


def test_in_n_weeks_needs_a_future_sentence() -> None:
    assert find_phrases(ANCHORED) == []  # a duration, and a count from another date
    ahead = "The overhaul will take place in three weeks."
    assert [p.text for p in find_phrases(ahead)] == ["in three weeks"]
    assert [p.text for p in find_phrases("Delivery is due in 2 weeks.")] == [
        "in 2 weeks"
    ]
    assert [p.text for p in find_phrases("The window opens in two weeks.")] == [
        "in two weeks"
    ]
    # the sentence is the unit: a future in another sentence does not count
    assert find_phrases("It will rain. The repair was done in two weeks.") == []
    # "from now" and "ago" always count
    assert [p.text for p in find_phrases("Ready two weeks from now.")] == [
        "two weeks from now"
    ]
    assert resolve_phrase("in three weeks", TODAY, context=ahead) == dt.date(
        2026, 11, 13
    )
    assert resolve_phrase("in three weeks", TODAY) is None
    assert plausible("in three weeks", dt.date(2026, 11, 13), TODAY, context=ahead)
    assert not plausible("in three weeks", dt.date(2026, 3, 22), TODAY, context=ahead)


def test_a_coerced_answer_that_is_a_later_phrase_is_not_today_plus_n() -> None:
    class Report(BaseModel):
        inspected_on: dt.date = Field(description="the inspection date")

    spec = next(
        s for s in compile_schema(Report).all_specs() if s.pretty == "inspected_on"
    )
    for phrase in ("two days later", "in three weeks"):
        coerced = coerce(spec, ClairicAnswer(type="date", value=phrase), today=TODAY)
        assert not coerced.ok  # not read as a date; nothing invented
    done = coerce(spec, ClairicAnswer(type="date", value="two days ago"), today=TODAY)
    assert done.ok and done.value == dt.date(2026, 10, 21)


# --------------------------------------------------------------------------- #
# S6: "N more X" is the whole amount only when no held item is X
# --------------------------------------------------------------------------- #


class Line(BaseModel):
    """One line of an order."""

    name: str = Field(description="the product name")
    quantity: int = Field(description="number of units ordered")


class Order(BaseModel):
    """An order."""

    ref: str = Field(description="the order reference")
    lines: Annotated[list[Line], Many(identity="name")] = Field(
        default_factory=list, description="its lines"
    )


MUGS = "Order 7: please add two more mugs."


async def add_line(
    text: str, label: str, quantity: int, held: list[dict[str, Any]]
) -> tuple[Any, list[str]]:
    world = FakeWorld(text)
    world.entity(
        "the Order", schema=Order, mentions=[], fields={"ref": "7"}, matches="7"
    )
    mention = label
    world.entity(
        mention,
        schema=Line,
        parent="the Order",
        mentions=[mention],
        fields={"name": label, "quantity": quantity},
        matches=None,  # the match says: a new line
    )
    seen: list[str] = []
    inner = world.handler

    def handler(model: str, payload: Any, meta: Any) -> Any:
        if meta.kind == "attribution":
            for qid, q in payload.get("questions", {}).items():
                if "quantity" in qid:
                    seen.append(str(q.get("instructions")))
        return inner(model, payload, meta)

    extractor = create_extractor(
        [One(Order)], transport=FakeTransport(handler), enable_inserts=True
    )
    result = await extractor.ainvoke(
        text, existing={"Order": {"ref": "7", "lines": held}}
    )
    return result, seen


def patches(result: Any) -> list[tuple[str, str, Any]]:
    return [(p.op, p.path, p.value) for r in result.records for p in r.patches]


async def test_two_more_of_a_held_item_is_not_added_as_a_new_line() -> None:
    held = [{"name": "blue mug", "quantity": 3}]
    result, seen = await add_line(MUGS, "mugs", 2, held)
    assert patches(result) == []  # not {"mugs", 2} next to the 3 blue mugs
    assert any(w.startswith("relative_edit_unsupported") for w in result.warnings)
    assert [s.reason for s in result.skipped] == ["validation_failed"]
    # the "new, nothing held before" sentence is not told to the model
    assert seen and all("is new" not in q for q in seen)
    assert any(c.kind == "relative_edit" for c in result.calls)


async def test_two_more_of_a_stated_amount_is_still_written_for_a_held_item() -> None:
    # "5 mugs" is an amount, not a change of the held 3: nothing is refused
    held = [{"name": "blue mug", "quantity": 3}]
    result, _ = await add_line("Order 7: please add 5 mugs.", "mugs", 5, held)
    assert patches(result) == [("add", "/lines/-", {"name": "mugs", "quantity": 5})]


async def test_a_new_item_that_matches_nothing_held_is_the_whole_amount() -> None:
    held = [{"name": "blue mug", "quantity": 3}]
    text = "Order 7: please add two more notebooks."
    result, seen = await add_line(text, "notebooks", 2, held)
    assert patches(result) == [
        ("add", "/lines/-", {"name": "notebooks", "quantity": 2})
    ]
    assert any("is new" in q for q in seen)  # the item is fresh: told so
    assert not any(c.kind == "relative_edit" for c in result.calls)


async def test_a_new_item_in_an_empty_list_is_the_whole_amount() -> None:
    result, seen = await add_line(MUGS, "mugs", 2, [])
    assert patches(result) == [("add", "/lines/-", {"name": "mugs", "quantity": 2})]
    assert any("is new" in q for q in seen)


async def test_a_variant_of_a_held_item_is_a_new_item() -> None:
    # "red mugs" is not "blue mug": neither name holds the other's words
    held = [{"name": "blue mug", "quantity": 3}]
    result, _ = await add_line(
        "Order 7: please add two more red mugs.", "red mugs", 2, held
    )
    assert patches(result) == [("add", "/lines/-", {"name": "red mugs", "quantity": 2})]


# --------------------------------------------------------------------------- #
# S1: quotes match whole words, and a value belongs to its own row
# --------------------------------------------------------------------------- #


class EngineRow(BaseModel):
    """One engine."""

    position: str = Field(description="the engine position code")
    serial: str = Field(description="the engine serial number")
    cycles: int = Field(description="cycles since new")


def row_specs() -> Any:
    return _row_specs(compile_schema(EngineRow))


def test_a_quote_matches_whole_words_only() -> None:
    text = "| ENG-10 | SN-1002 | B2 |"
    assert locate(text, "ENG-1") is None  # not inside ENG-10
    assert locate(text, "ENG-10") == (2, 8)
    assert locate(text, "2") is None  # not inside B2 or SN-1002
    assert locate(text, "0") is None
    assert locate(text, "B2") is not None and locate(text, "SN-1002") is not None
    assert locate("Serial  SN-1\n001 x", "Serial SN-1 001") == (0, 16)  # white space
    # punctuation at an end does not need a boundary
    assert locate("Value (A-1), then B.", "(A-1),") is not None
    # an occurrence that overlaps a taken span, or lies outside the ranges, is skipped
    assert locate("a ENG-1 b ENG-1 c", "ENG-1", taken=[(2, 7)]) == (10, 15)
    assert locate("a ENG-1 b ENG-1 c", "ENG-1", within=[(8, 17)]) == (10, 15)
    assert locate("a ENG-1 b ENG-1 c", "ENG-1", avoid=[(10, 15)]) == (2, 7)
    # of several occurrences the nearest to one of the anchors
    assert locate("ENG-1 x ENG-1 y ENG-1", "ENG-1", near=[(12, 13), (20, 21)]) == (
        8,
        13,
    ) or locate("ENG-1 x ENG-1 y ENG-1", "ENG-1", near=(20, 21)) == (17, 22)


TABLE10 = (
    "| Pos | Serial | Cycles |\n| ENG-10 | SN-9010 | 5 |\n| ENG-1 | SN-9001 | 7 |\n"
)


def test_an_identity_that_is_the_start_of_another_is_a_row_of_its_own() -> None:
    answer = {
        "rows": [
            {
                "identity": "ENG-1",
                "fields": {"position": "ENG-1", "serial": "SN-9001", "cycles": "7"},
            },
            {
                "identity": "ENG-10",
                "fields": {"position": "ENG-10", "serial": "SN-9010", "cycles": "5"},
            },
        ]
    }
    rows = _verify_rows(answer, TABLE10, row_specs())
    assert [r.identity for r in rows] == ["ENG-1", "ENG-10"]
    first, second = rows
    assert TABLE10[first.span[0] : first.span[1]] == "ENG-1"
    assert first.span == (
        TABLE10.index("| ENG-1 |") + 2,
        TABLE10.index("| ENG-1 |") + 7,
    )
    assert second.span[1] - second.span[0] == 6 and not _overlap(
        first.span, second.span
    )
    # every value is the one of its own line
    assert {k: v[0] for k, v in first.values.items()} == {
        "position": "ENG-1",
        "serial": "SN-9001",
        "cycles": 7,
    }
    assert {k: v[0] for k, v in second.values.items()} == {
        "position": "ENG-10",
        "serial": "SN-9010",
        "cycles": 5,
    }
    line = TABLE10.index("| ENG-1 |")
    assert all(line <= v[1][0] for v in first.values.values())  # type: ignore[index]


def entity_of(*mentions: tuple[str, tuple[int, int]]) -> Any:
    return SimpleNamespace(
        info=SimpleNamespace(mentions=[Mention(t, s) for t, s in mentions])
    )


def test_a_mention_picks_the_row_with_its_own_span() -> None:
    answer = {
        "rows": [
            {"identity": "ENG-1", "fields": {}},
            {"identity": "ENG-10", "fields": {}},
        ]
    }
    rows = _verify_rows(answer, TABLE10, row_specs())
    e10 = TABLE10.index("ENG-10")
    e1 = TABLE10.index("| ENG-1 |") + 2
    assert _row_of(entity_of(("ENG-10", (e10, e10 + 6))), rows).identity == "ENG-10"  # type: ignore[union-attr]
    assert _row_of(entity_of(("ENG-1", (e1, e1 + 5))), rows).identity == "ENG-1"  # type: ignore[union-attr]
    # an overlap that is not equal: the longest overlap wins
    assert _row_of(entity_of(("ENG-10 |", (e10, e10 + 8))), rows).identity == "ENG-10"  # type: ignore[union-attr]


TABLE2 = "| Pos | Serial | Cycles |\n| A1 | SN-1002 | 2 |\n| B2 | SN-2000 | 5 |\n"


def test_a_value_quoted_from_another_row_or_inside_a_token_is_not_taken() -> None:
    # B2's serial is quoted from the A1 line; its cycles "2" is only part of "B2"
    answer = {
        "rows": [
            {
                "identity": "B2",
                "fields": {"position": "B2", "serial": "SN-1002", "cycles": "2"},
            }
        ]
    }
    (row,) = _verify_rows(answer, TABLE2, row_specs())
    assert set(row.values) == {"position"}
    # "0" occurs only inside SN-2000 and SN-1002: no value for the A1 row
    answer = {"rows": [{"identity": "A1", "fields": {"position": "A1", "cycles": "0"}}]}
    (row,) = _verify_rows(answer, TABLE2, row_specs())
    assert set(row.values) == {"position"}


def test_a_value_on_the_next_line_of_a_label_is_still_this_rows() -> None:
    text = "ENG-1\nserial SN-5001\ncycles 12\n\nENG-2\nserial SN-5002\ncycles 30\n"
    answer = {
        "rows": [
            {"identity": "ENG-1", "fields": {"serial": "SN-5001", "cycles": "12"}},
            {"identity": "ENG-2", "fields": {"serial": "SN-5002", "cycles": "30"}},
        ]
    }
    first, second = _verify_rows(answer, text, row_specs(), window=400)
    assert (first.values["serial"][0], first.values["cycles"][0]) == ("SN-5001", 12)
    assert (second.values["serial"][0], second.values["cycles"][0]) == ("SN-5002", 30)
    # ... but not another row's, even within the window
    swapped = {"rows": [{"identity": "ENG-1", "fields": {"serial": "SN-5002"}}]}
    (row,) = _verify_rows(swapped, text, row_specs(), window=400)
    assert row.values == {}


def test_the_identity_is_the_occurrence_whose_line_holds_the_values() -> None:
    text = "ENG-1 was replaced last year.\nPos ENG-1 serial SN-7001 cycles 9\n"
    answer = {
        "rows": [{"identity": "ENG-1", "fields": {"serial": "SN-7001", "cycles": "9"}}]
    }
    (row,) = _verify_rows(answer, text, row_specs())
    assert row.span[0] == text.index("Pos ENG-1") + 4  # the table line, not the prose
    assert set(row.values) == {"serial", "cycles"}


ENGINES = (
    "Aircraft N512QX\n\n| Pos | Serial |\n|---|---|\n"
    "| ENG-10 | SN-9010 |\n| ENG-1 | SN-9001 |\n"
)


class Engine(BaseModel):
    """One engine of an aircraft."""

    position: str = Field(description="the engine position code, such as ENG-1")
    serial: str = Field(description="the engine serial number")


class Aircraft(BaseModel):
    """An aircraft."""

    registration: str = Field(description="the registration")
    engines: Annotated[list[Engine], Many(identity="position")] = Field(
        default_factory=list, description="its engines"
    )


def engines_world() -> FakeWorld:
    world = FakeWorld(ENGINES)
    world.entity(
        "the Aircraft", schema=Aircraft, mentions=[], fields={"registration": "N512QX"}
    )
    world.entity(
        "ENG-10",
        schema=Engine,
        parent="the Aircraft",
        mentions=["ENG-10"],
        fields={"position": "ENG-10", "serial": "SN-9010"},
    )
    world.entity(
        "ENG-1",
        schema=Engine,
        parent="the Aircraft",
        mentions=["ENG-1"],
        fields={"position": "ENG-1", "serial": "SN-9001"},
    )
    return world


async def run_engines(mode: str, rows: Any) -> Any:
    fake = FakeEscalation(lambda p, m: rows)
    extractor = create_extractor(
        [One(Aircraft)],
        transport=FakeTransport(engines_world().handler),
        escalation=EscalationConfig(enabled=True, transport=fake, mode=mode),
        enable_inserts=True,
    )
    return await extractor.ainvoke(ENGINES)


async def test_two_engines_one_code_the_start_of_the_other_are_both_read_routed() -> (
    None
):
    rows = {
        "rows": [
            {"identity": "ENG-1", "fields": {"position": "ENG-1", "serial": "SN-9001"}},
            {
                "identity": "ENG-10",
                "fields": {"position": "ENG-10", "serial": "SN-9010"},
            },
        ]
    }
    for mode in ("route", "on_failure"):
        result = await run_engines(mode, rows)
        assert sorted(
            (e["position"], e["serial"]) for e in result.records[0].after["engines"]
        ) == [("ENG-1", "SN-9001"), ("ENG-10", "SN-9010")]
        assert not result.skipped


async def test_a_routed_serial_of_another_row_replaces_nothing() -> None:
    # the tier quotes ENG-10's serial for ENG-1: not in ENG-1's line, so not taken
    rows = {
        "rows": [
            {"identity": "ENG-1", "fields": {"position": "ENG-1", "serial": "SN-9010"}},
            {
                "identity": "ENG-10",
                "fields": {"position": "ENG-10", "serial": "SN-9010"},
            },
        ]
    }
    result = await run_engines("route", rows)
    by_position = {
        e["position"]: e["serial"] for e in result.records[0].after["engines"]
    }
    assert by_position == {"ENG-1": "SN-9001", "ENG-10": "SN-9010"}


# --------------------------------------------------------------------------- #
# S2: dotted discovery keeps a label with its own identity
# --------------------------------------------------------------------------- #

TWO_SHEETS = """\
Aircraft fleet update for the lessor.

# Sheet 1
Aircraft: Airbus A320-200
| Manufacturer Serial Number (MSN) | 8547 |

# Sheet 2
Aircraft: Boeing 737-800
| Manufacturer Serial Number (MSN) | 12345 |
"""


def pairs(
    *items: tuple[str, str], window: int = 600
) -> list[tuple[str, tuple[int, int]]]:
    entities = [{"label": label, "identity": ident} for label, ident in items]
    return _pair_entities(entities, TWO_SHEETS, window)


def line_of(span: tuple[int, int]) -> str:
    start = TWO_SHEETS.rfind("\n", 0, span[0]) + 1
    return TWO_SHEETS[start : TWO_SHEETS.find("\n", span[1])]


def test_a_generic_label_is_the_one_next_to_its_own_identity() -> None:
    found = pairs(("Aircraft", "8547"), ("Aircraft", "12345"))
    assert [line_of(span) for _, span in found] == [
        "Aircraft: Airbus A320-200",
        "Aircraft: Boeing 737-800",
    ]  # not the first line of the text, and not the same occurrence twice
    assert len({span for _, span in found}) == 2


def test_a_label_paired_with_the_other_entitys_identity_is_refused() -> None:
    assert pairs(("Airbus A320-200", "12345"), ("Boeing 737-800", "8547")) == []
    # the right pairing stands
    found = pairs(("Airbus A320-200", "8547"), ("Boeing 737-800", "12345"))
    assert [label for label, _ in found] == ["Airbus A320-200", "Boeing 737-800"]


def test_one_entity_per_identity_and_both_quotes_have_to_be_there() -> None:
    found = pairs(
        ("Airbus A320-200", "8547"),
        ("Airbus A320-200 again", "8547"),  # the identity is taken
        ("Boeing 737-800", "99999"),  # no such identity
        ("Embraer E190", "12345"),  # no such label
    )
    assert [label for label, _ in found] == ["Airbus A320-200"]
    # a label is no occurrence of another identity's line: 8547 is not in 85470
    assert (
        _pair_entities([{"label": "A", "identity": "8547"}], "A 85470 and A", 600) == []
    )


def test_a_label_far_from_its_identity_is_not_taken() -> None:
    text = "Airbus A320-200\n" + "filler words " * 100 + "\nMSN 8547\n"
    entity = [{"label": "Airbus A320-200", "identity": "8547"}]
    assert _pair_entities(entity, text, 600) == []
    assert len(_pair_entities(entity, text, 2000)) == 1


async def test_routed_generic_labels_are_screened_and_clairic_discovers() -> None:
    # "Aircraft" names no aircraft: the routed mentions are dropped as junk, and
    # Clairic's discovery runs as it would without the tier
    fake = FakeEscalation(
        lambda p, m: {
            "entities": [
                {"label": "Aircraft", "identity": "8547"},
                {"label": "Aircraft", "identity": "12345"},
            ]
        }
    )
    extractor = create_extractor(
        [Many(Plane, identity="identification.msn")],
        transport=FakeTransport(plane_world().handler),
        escalation=EscalationConfig(enabled=True, transport=fake, mode="route"),
    )
    result = await extractor.ainvoke(TWO_SHEETS)
    assert len(fake.calls) == 1
    assert sorted(r.after["identification"]["msn"] for r in result.records) == [  # type: ignore[index]
        "12345",
        "8547",
    ]
    assert any(w.startswith("junk_mentions_dropped") for w in result.warnings)
    assert any(c.kind == "discover" for c in result.calls)


class Ident(BaseModel):
    """The identification of an aircraft."""

    msn: str = Field(description="the manufacturer serial number")


class Plane(BaseModel):
    """An aircraft on a specification sheet."""

    identification: Ident = Field(description="its identification")


def plane_world() -> FakeWorld:
    world = FakeWorld(TWO_SHEETS)
    for label, msn in (("Airbus A320-200", "8547"), ("Boeing 737-800", "12345")):
        world.entity(
            label,
            schema=Plane,
            mentions=[label, msn],
            fields={"identification.msn": msn},
        )
    return world


# --------------------------------------------------------------------------- #
# S3: a routed date never costs a date Clairic read
# --------------------------------------------------------------------------- #


class Task(BaseModel):
    """A task report."""

    title: str = Field(description="the task title")
    done_on: dt.date | None = Field(None, description="the day the task was completed")


TASK_TEXT = (
    "Task: Replace the pump. Completed on 2026-03-01; the part was ordered three "
    "days ago."
)


async def run_task(mode: str, answer: dict[str, Any]) -> tuple[Any, FakeEscalation]:
    world = FakeWorld(TASK_TEXT)
    world.entity(
        "the Task",
        schema=Task,
        mentions=[],
        fields={"title": "Replace the pump", "done_on": dt.date(2026, 3, 1)},
    )
    fake = FakeEscalation(lambda p, m: answer)
    extractor = create_extractor(
        [One(Task)],
        transport=FakeTransport(world.handler),
        escalation=EscalationConfig(enabled=True, transport=fake, mode=mode),
        enable_inserts=True,
        reference_date="2026-10-23",
    )
    return await extractor.ainvoke(TASK_TEXT), fake


WRONG_DATE = {"phrase": "three days ago", "iso_date": "2026-10-20", "time": None}


async def test_a_refused_routed_date_gives_the_clairic_date_back() -> None:
    result, fake = await run_task("route", WRONG_DATE)
    (record,) = result.records
    assert len(fake.calls) == 1  # the tier was asked first
    assert record.after["done_on"] == dt.date(2026, 3, 1)  # not null
    assert "escalation_refused" in " ".join(record.fields["done_on"].warnings)
    # the restored answer was judged like any other: Clef was asked about it
    assert sum(c.kind == "attribution" for c in result.calls) >= 2
    off, _ = await run_task("on_failure", WRONG_DATE)
    assert off.records[0].after["done_on"] == dt.date(2026, 3, 1)


async def test_a_routed_date_clef_accepts_still_replaces_it() -> None:
    world = FakeWorld(TASK_TEXT)
    world.entity(
        "the Task",
        schema=Task,
        mentions=[],
        fields={"title": "Replace the pump", "done_on": dt.date(2026, 10, 20)},
    )
    fake = FakeEscalation(lambda p, m: WRONG_DATE)
    extractor = create_extractor(
        [One(Task)],
        transport=FakeTransport(world.handler),
        escalation=EscalationConfig(enabled=True, transport=fake, mode="route"),
        enable_inserts=True,
        reference_date="2026-10-23",
    )
    result = await extractor.ainvoke(TASK_TEXT)
    # Clef's world states 2026-10-20: the proposal is judged and stands
    assert result.records[0].after["done_on"] == dt.date(2026, 10, 20)  # type: ignore[index]


def test_the_phrase_nearest_the_entity_is_the_one_checked() -> None:
    text = "Ordered three days ago. " + "x " * 600 + "Fitted three days ago."
    far = text.index("Fitted")
    window = [(far - 100, len(text))]
    phrases = find_phrases(text, window)
    today = dt.date(2026, 10, 23)
    answer = {"phrase": "three days ago", "iso_date": "2026-10-20", "time": None}
    verdict = _check_date(answer, text, window, phrases, today, False, [(far, far + 6)])
    assert not isinstance(verdict, str)
    assert verdict[1][0] > far  # the second occurrence: the first lies outside
    # with both inside the window, the one nearest the mention
    both = [(0, len(text))]
    verdict = _check_date(
        answer, text, both, find_phrases(text), today, False, [(far, far + 6)]
    )
    assert not isinstance(verdict, str) and verdict[1][0] > far
    verdict = _check_date(
        answer, text, both, find_phrases(text), today, False, [(0, 7)]
    )
    assert not isinstance(verdict, str) and verdict[1][0] < 20
    # a phrase that is nowhere in a window is said to be outside it
    assert (
        _check_date(answer, text, [(30, 60)], find_phrases(text), today, False)
        == "the phrase is outside the entity's mention window"
    )


# --------------------------------------------------------------------------- #
# S4: a partial routed answer does not suppress Clairic's count and discovery
# --------------------------------------------------------------------------- #


async def test_a_partial_routed_row_list_is_completed_by_clairic() -> None:
    from tests.unit.test_escalation import engine_world, insert_aircraft

    one_row = {
        "rows": [
            {"identity": "ENG-1", "fields": {"position": "ENG-1", "serial": "SN-1001"}}
        ]
    }
    fake = FakeEscalation(lambda p, m: one_row)
    result = await insert_aircraft(
        engine_world().handler,
        EscalationConfig(enabled=True, transport=fake, mode="route"),
    )
    assert [e["position"] for e in result.records[0].after["engines"]] == [  # type: ignore[index]
        "ENG-1",
        "ENG-2",
        "ENG-3",
    ]
    kinds = Counter(c.kind for c in result.calls if c.schema == "Engine")
    assert kinds["count"] == 1 and kinds["discover"] >= 1
    assert any("escalation_routed_partial" in w for w in result.warnings)
    assert len(fake.calls) == 1


async def test_a_complete_routed_row_list_costs_no_discovery() -> None:
    from tests.unit.test_escalation import all_rows, engine_world, insert_aircraft

    fake = FakeEscalation(all_rows)
    result = await insert_aircraft(
        engine_world().handler,
        EscalationConfig(enabled=True, transport=fake, mode="route"),
    )
    kinds = Counter(c.kind for c in result.calls if c.schema == "Engine")
    assert kinds["count"] == 1 and kinds["discover"] == 0
    assert not any("escalation_routed_partial" in w for w in result.warnings)


async def test_a_partial_routed_dotted_discovery_is_completed_by_clairic() -> None:
    from tests.unit.test_escalation import sheet_world

    only_first = {"entities": [{"label": "Airbus A320-200", "identity": "8547"}]}
    fake = FakeEscalation(lambda p, m: only_first)
    from tests.unit.test_escalation import SHEET

    extractor = create_extractor(
        [Many(Plane2, identity="identification.msn")],
        transport=FakeTransport(sheet_world().handler),
        escalation=EscalationConfig(enabled=True, transport=fake, mode="route"),
    )
    result = await extractor.ainvoke(SHEET)
    assert sorted(r.after["identification"]["msn"] for r in result.records) == [  # type: ignore[index]
        "12345",
        "8547",
    ]
    assert any(c.kind == "discover" for c in result.calls)
    assert len(fake.calls) == 1


from tests.unit.test_escalation import Plane as Plane2  # noqa: E402

# --------------------------------------------------------------------------- #
# Nits: the clause description, the tie-break's windows, a routed list's part
# --------------------------------------------------------------------------- #


class Addr(BaseModel):
    """An address."""

    city: str = Field(description="which city the person lives in")


class Person(BaseModel):
    """A person."""

    name: str
    relation: str = Field(description="How the person relates to the speaker")
    nick: str = Field(description="what friends call them")
    addr: Addr
    which_way: str = Field(description="Which-way sign text")
    whoever: str = Field(description="Who's Who entry title")


def question(pretty: str, *, element: bool = False, ask_for: bool = False) -> str:
    spec = next(
        s for s in compile_schema(Person).all_specs() if s.pretty.startswith(pretty)
    )
    q = _attribution_question(spec.scoped, "x", element=element, ask_for=ask_for)
    return str(q.instructions)


def test_only_a_clause_word_with_more_words_makes_a_clause() -> None:
    assert _clause_of("How the person relates to the speaker") == (
        "how the person relates to the speaker"
    )
    assert _clause_of("whether it was paid") == "whether it was paid"
    for noun_phrase in ("Which-way sign text", "Who's Who entry title", "what", "name"):
        assert _clause_of(noun_phrase) is None
    # the scalar question lowercases the clause and keeps the entity off its last noun
    assert (
        'state that for `entity.label`, how the person relates to the speaker is "x"'
        in question("relation")
    )
    assert 'state that Which-way sign text of `entity.label` is "x"' in question(
        "which_way"
    )
    assert "state that Who's Who entry title of `entity.label`" in question("whoever")


def test_the_element_and_ask_for_questions_take_a_clause_too() -> None:
    element = question("relation", element=True)
    assert 'an answer, for `entity.label`, to "how the person relates' in element
    assert " a how " not in element and " an how " not in element
    ask = question("relation", ask_for=True)
    assert "(as how the person relates to the speaker)" in ask
    assert "(as the how" not in ask
    # a noun phrase is as before
    assert "(as the Which-way sign text)" in question("which_way", ask_for=True)
    assert 'is "x" a name of `entity.label`' in question("name", element=True)


async def test_the_tie_break_reads_the_windows_around_the_entity_not_the_input() -> (
    None
):
    plan = compile_schema(Person)
    spec = next(s for s in plan.all_specs() if s.pretty == "nick")
    head = "Dana Reyes joined the board. Friends call her Dee. "
    text = head + "unrelated filler. " * 200 + "SECOND-SECTION-MARKER."
    fake = FakeEscalation(lambda p, m: {"answer": "yes"})
    run = Run(
        transport=FakeTransport(lambda *a: {}),
        escalation=EscalationConfig(enabled=True, transport=fake),
    )
    info = EntityInfo(
        key="e0", label="Dana", mentions=(Mention("Dana Reyes", (0, 10)),)
    )
    ctx = Ctx(run=run, plan=plan, text=text, label="Dana", key="e0", info=info)
    st = FieldState(spec=spec, resolved=Resolved(value="Dee", raw="Dee"))
    p = await tiebreak(ctx, st, "Is it stated?", span=None)
    assert p == 0.8
    shown = fake.calls[0][1]["input"]
    assert "Friends call her Dee" in shown
    assert "SECOND-SECTION-MARKER" not in shown  # the end of the input is not sent
    # an entity without mentions has the whole input
    bare = Ctx(run=run, plan=plan, text=text, label="Dana", key="e1")
    assert await tiebreak(bare, st, "Is it stated?", span=None) == 0.8
    assert "SECOND-SECTION-MARKER" in fake.calls[1][1]["input"]


async def test_a_routed_list_is_asked_in_its_parents_part_of_the_input() -> None:
    two = (
        "Aircraft N111AA\n\n| Pos | Serial |\n|---|---|\n| ENG-1 | SN-1001 |\n"
        + "filler line about something else\n" * 80
        + "Aircraft N222BB\n\n| Pos | Serial |\n|---|---|\n| ENG-1 | SN-2001 |\n"
    )
    first = (0, len("Aircraft N111AA"))
    parent = EntityInfo(key="e0", label="N111AA", mentions=(Mention("N111AA", first),))
    fake = FakeEscalation(
        lambda p, m: {
            "rows": [
                {
                    "identity": "ENG-1",
                    "fields": {"position": "ENG-1", "serial": "SN-1001"},
                },
                {
                    "identity": "ENG-1",
                    "fields": {"position": "ENG-1", "serial": "SN-2001"},
                },
            ]
        }
    )
    run = Run(
        transport=FakeTransport(lambda *a: {}),
        escalation=EscalationConfig(enabled=True, transport=fake, mode="route"),
    )
    plan = compile_schema(Engine)
    rows = await _ask_rows(
        run, plan, parent, two, SimpleNamespace(plan=plan, key=None, scope=""), "engine"
    )
    shown = fake.calls[0][1]["input"]
    assert "SN-1001" in shown and "N222BB" not in shown and "SN-2001" not in shown
    # only the rows of the parent's part come back (the second one is outside it)
    assert [r.identity for r in rows] == ["ENG-1"]
    assert rows[0].values["serial"][0] == "SN-1001"
