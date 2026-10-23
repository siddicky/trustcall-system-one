"""The opt-in escalation tier: one failing case per kind, then fixed by the tier.

Every text here is synthetic. Each kind is reproduced first with the tier OFF (the
default setup fails), then fixed with the tier ON, and the tier's own safety is checked:
a quote the model invented is rejected, an answer is capped below the removal gates,
the call limits hold, a failure of the escalation model degrades to the default result.
The default setup is a ``FakeWorld``; "Clairic fails" is the world's answer degraded
by a wrapper, and the escalation model is a scripted ``FakeEscalation``.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

import datetime as dt
from collections import Counter
from collections.abc import Callable
from typing import Annotated, Any, Optional

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    CallMeta,
    EscalationConfig,
    ExtractionResult,
    GateConfig,
    Many,
    MatchConfig,
    One,
    TransportError,
    UncertainOutcomeError,
    create_extractor,
)
from trustcall_system_one._errors import AuthError
from trustcall_system_one._relative import find_phrases, plausible, resolve_phrase
from trustcall_system_one.testing import (
    FakeEscalation,
    FakeTransport,
    FakeWorld,
    clairic_answer,
    clef_response,
    noul,
)

Handler = Callable[[str, dict[str, Any], CallMeta], Any]
TODAY = "2026-10-23"  # a Friday


def escalations(result: ExtractionResult) -> list[Any]:
    return [c for c in result.calls if c.kind == "escalate"]


def tier(fake: FakeEscalation, **kwargs: Any) -> EscalationConfig:
    return EscalationConfig(enabled=True, transport=fake, **kwargs)


def degrade(
    inner: Handler,
    *,
    schema: str,
    kinds: tuple[str, ...] = ("extract", "crosscheck", "reask", "repair"),
    field: str | None = None,
    value: Any = None,
    kind: str = "text",
) -> Handler:
    """The world, with Clairic answering ``value`` (null by default) for a field."""

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        out = inner(model, payload, meta)
        if meta.kind in kinds and meta.schema == schema and "answers" in out:
            for wire, answer in out["answers"].items():
                if field is None or field in wire:
                    out["answers"][wire] = clairic_answer(
                        answer["type"] if value is None else kind,
                        value,
                        p_null=0.99 if value is None else 1e-6,
                        confidence=0.9,
                    )
        return out

    return handler


# --------------------------------------------------------------------------- #
# relative_dates
# --------------------------------------------------------------------------- #


class Task(BaseModel):
    """A task report."""

    title: str = Field(description="the task title")
    done_on: dt.date = Field(description="the day the task was completed")


class Visit(BaseModel):
    """A site visit."""

    site: str = Field(description="the site")
    visited: dt.date = Field(description="the day of the visit")
    returns: Optional[dt.date] = Field(None, description="the day of the next visit")


def task_world(text: str, day: dt.date) -> FakeWorld:
    world = FakeWorld(text)
    world.entity(
        "the Task",
        schema=Task,
        mentions=[],
        fields={"title": "Replace the pump", "done_on": day},
    )
    return world


async def run_task(
    text: str,
    day: dt.date,
    *,
    answer: Any = None,
    escalation: EscalationConfig | None = None,
    reference: str | None = TODAY,
    transport_answer: Handler | None = None,
) -> ExtractionResult:
    """Clairic gives ``answer`` for the date (None: null); the world knows ``day``."""
    world = task_world(text, day)
    handler = degrade(
        world.handler, schema="Task", field="done_on", value=answer, kind="date"
    )
    extractor = create_extractor(
        [One(Task)],
        transport=FakeTransport(transport_answer or handler),
        reference_date=reference,
        escalation=escalation,
        enable_inserts=True,
    )
    return await extractor.ainvoke(text)


THREE_DAYS = "Task: Replace the pump. Completed three days ago by the night crew."
LAST_TUESDAY = "Task: Replace the pump. It was completed last Tuesday."


@pytest.mark.parametrize(
    ("text", "answer", "expected"),
    [
        (THREE_DAYS, "three days ago", dt.date(2026, 10, 20)),
        ("Task: Replace the pump. Done yesterday.", "yesterday", dt.date(2026, 10, 22)),
        (
            "Task: Replace the pump. Done two weeks from now.",
            "two weeks from now",
            dt.date(2026, 11, 6),
        ),
        (
            "Task: Replace the pump. Done a month ago.",
            "a month ago",
            dt.date(2026, 9, 23),
        ),
    ],
)
async def test_a_phrase_that_clairic_hands_back_is_resolved_in_code(
    text: str, answer: str, expected: dt.date
) -> None:
    """The coercion fix: no escalation involved, nothing sent anywhere."""
    result = await run_task(text, expected, answer=answer)
    (record,) = result.records
    assert record.valid and record.value.done_on == expected  # type: ignore[union-attr]
    assert "relative_date: resolved in code" in " ".join(
        record.fields["done_on"].warnings
    )
    assert not escalations(result)


async def test_without_a_reference_date_a_phrase_stays_invalid() -> None:
    result = await run_task(
        THREE_DAYS, dt.date(2026, 10, 20), answer="three days ago", reference=None
    )
    (record,) = result.records
    assert not record.valid  # the code never guesses today's date


async def test_a_weekday_phrase_is_not_picked_by_the_code() -> None:
    result = await run_task(LAST_TUESDAY, dt.date(2026, 10, 20), answer="last Tuesday")
    (record,) = result.records
    assert not record.valid  # "last Tuesday" depends on a convention: left to the tier


async def test_relative_dates_failure_is_reproduced_with_the_tier_off() -> None:
    result = await run_task(THREE_DAYS, dt.date(2026, 10, 20))  # Clairic says null
    (record,) = result.records
    assert not record.valid and "done_on" in record.errors[0]
    assert result.skipped[0].reason == "validation_failed"
    assert not escalations(result)


async def test_the_tier_disabled_by_default_never_calls_even_when_configured_off() -> (
    None
):
    fake = FakeEscalation(lambda p, m: {"phrase": None, "iso_date": None, "time": None})
    result = await run_task(
        THREE_DAYS, dt.date(2026, 10, 20), escalation=EscalationConfig(transport=fake)
    )
    assert not fake.calls and not escalations(result)
    only_others = EscalationConfig(
        enabled=True, transport=fake, kinds=frozenset({"list_rows"})
    )
    await run_task(THREE_DAYS, dt.date(2026, 10, 20), escalation=only_others)
    assert not fake.calls  # a kind that is not listed does not run


async def test_the_tier_fixes_a_relative_date_and_the_ladder_still_judges_it() -> None:
    fake = FakeEscalation(
        lambda p, m: {
            "phrase": "three days ago",
            "iso_date": "2026-10-20",
            "time": None,
        }
    )
    result = await run_task(THREE_DAYS, dt.date(2026, 10, 20), escalation=tier(fake))
    (record,) = result.records
    assert record.valid and record.value.done_on == dt.date(2026, 10, 20)  # type: ignore[union-attr]
    answer = record.fields["done_on"]
    assert "escalated:relative_dates" in answer.warnings
    # the proposal went through the attribution check (a Clef call) like any value
    assert any(c.kind == "attribution" for c in result.calls)
    (call,) = escalations(result)
    assert call.field == "relative_dates:done_on" and call.provider == "openai"
    assert call.model == "gpt-6-luna" and call.usage["input_tokens"] > 0
    assert any(w.startswith("escalation:relative_dates") for w in result.warnings)
    # the prompt carries the reference date and the text, nothing else of the run
    prompt = fake.calls[0][1]["input"]
    assert THREE_DAYS in prompt and "2026-10-23" in fake.calls[0][1]["instructions"]
    assert result.usage.by_kind["escalate"].requests == 1


async def test_a_weekday_phrase_is_fixed_by_the_tier_when_the_date_fits() -> None:
    fake = FakeEscalation(
        lambda p, m: {"phrase": "last Tuesday", "iso_date": "2026-10-20", "time": None}
    )
    result = await run_task(LAST_TUESDAY, dt.date(2026, 10, 20), escalation=tier(fake))
    (record,) = result.records
    assert record.valid and record.value.done_on == dt.date(2026, 10, 20)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        (
            {"phrase": "five days ago", "iso_date": "2026-10-18", "time": None},
            "the phrase is not in the input",
        ),
        (
            {"phrase": "last Tuesday", "iso_date": "2026-10-21", "time": None},
            "the date cannot be what the phrase says",  # a Wednesday
        ),
        (
            {"phrase": "last Tuesday", "iso_date": "2026-10-27", "time": None},
            "the date cannot be what the phrase says",  # a Tuesday after today
        ),
        (
            {"phrase": "last Tuesday", "iso_date": "20 October", "time": None},
            "iso_date is not YYYY-MM-DD",
        ),
        (
            {"phrase": "last Tuesday", "iso_date": "2026-02-30", "time": None},
            "iso_date is not a real date",
        ),
        (
            {"phrase": "last Tuesday", "iso_date": None, "time": None},
            "phrase and iso_date must both be given",
        ),
        (
            {"phrase": "Replace the pump", "iso_date": "2026-10-20", "time": None},
            "the quote holds no relative phrase",
        ),
    ],
)
async def test_a_date_the_code_cannot_verify_is_rejected(
    answer: dict[str, Any], why: str
) -> None:
    fake = FakeEscalation(lambda p, m: answer)
    result = await run_task(LAST_TUESDAY, dt.date(2026, 10, 20), escalation=tier(fake))
    (record,) = result.records
    assert not record.valid  # nothing was written on the model's word
    assert any(why in w for w in result.warnings), result.warnings
    assert len(fake.calls) == 1


async def test_a_model_that_says_there_is_no_date_changes_nothing() -> None:
    fake = FakeEscalation(lambda p, m: {"phrase": None, "iso_date": None, "time": None})
    result = await run_task(THREE_DAYS, dt.date(2026, 10, 20), escalation=tier(fake))
    assert not result.records[0].valid
    assert any("found no relative date" in w for w in result.warnings)


async def test_no_phrase_near_the_entity_means_no_call() -> None:
    text = "Task: Replace the pump. Completed on the shift of the night crew."
    fake = FakeEscalation(lambda p, m: {"phrase": None, "iso_date": None, "time": None})
    await run_task(text, dt.date(2026, 10, 20), escalation=tier(fake))
    assert not fake.calls  # nothing relative to ask about, nothing sent


async def test_no_reference_date_means_no_call() -> None:
    fake = FakeEscalation(lambda p, m: {"phrase": None, "iso_date": None, "time": None})
    await run_task(
        THREE_DAYS, dt.date(2026, 10, 20), escalation=tier(fake), reference=None
    )
    assert not fake.calls  # "three days ago" needs a day to count from


async def test_a_confident_clairic_date_is_never_asked_about() -> None:
    fake = FakeEscalation(
        lambda p, m: {
            "phrase": "three days ago",
            "iso_date": "2026-01-01",
            "time": None,
        }
    )
    result = await run_task(
        THREE_DAYS, dt.date(2026, 10, 20), answer="2026-10-20", escalation=tier(fake)
    )
    assert result.records[0].valid and not fake.calls  # no override of a good answer


async def test_a_datetime_needs_its_clock_time() -> None:
    class Call(BaseModel):
        """A call."""

        topic: str = Field(description="the topic")
        at: dt.datetime = Field(description="when the call took place")

    text = "Call about the budget, yesterday at 15:30."
    world = FakeWorld(text)
    world.entity(
        "the Call",
        schema=Call,
        mentions=[],
        fields={"topic": "the budget", "at": "2026-10-22T15:30:00"},
    )
    seen = [
        {"phrase": "yesterday", "iso_date": "2026-10-22", "time": None},
        {"phrase": "yesterday", "iso_date": "2026-10-22", "time": "15:30"},
    ]
    outcomes = []
    for answer in seen:
        fake = FakeEscalation(lambda p, m, a=answer: a)  # type: ignore[misc]
        extractor = create_extractor(
            [One(Call)],
            transport=FakeTransport(degrade(world.handler, schema="Call", field="at")),
            reference_date=TODAY,
            escalation=tier(fake),
        )
        result = await extractor.ainvoke(text)
        outcomes.append(result)
    assert not outcomes[0].records[0].valid
    assert any("needs a clock time" in w for w in outcomes[0].warnings)
    assert outcomes[1].records[0].valid
    assert outcomes[1].records[0].value.at == dt.datetime(2026, 10, 22, 15, 30)  # type: ignore[union-attr]


def test_the_relative_phrases_the_code_knows() -> None:
    today = dt.date(2026, 10, 23)
    assert resolve_phrase("yesterday", today) == dt.date(2026, 10, 22)
    assert resolve_phrase("the day after tomorrow", today) == dt.date(2026, 10, 25)
    assert resolve_phrase("Three days ago", today) == dt.date(2026, 10, 20)
    ahead = "It is due in 2 weeks"
    assert resolve_phrase("in 2 weeks", today, context=ahead) == dt.date(2026, 11, 6)
    assert resolve_phrase("in 2 weeks", today) is None  # a duration as often as a date
    assert resolve_phrase("in 2 weeks", today, context="Done in 2 weeks") is None
    assert resolve_phrase("two days later", today) is None  # counts from another date
    assert resolve_phrase("a week earlier", today) is None
    assert resolve_phrase("a fortnight ago", today) == dt.date(2026, 10, 9)
    assert resolve_phrase("2 fortnights ago", today) == dt.date(2026, 9, 25)
    assert resolve_phrase("a month from now", today) == dt.date(2026, 11, 23)
    assert resolve_phrase("last Tuesday", today) is None  # a convention, not a fact
    assert resolve_phrase("whenever", today) is None
    text = "Seen yesterday, will be seen again in two weeks, and last Tuesday; no more."
    assert [p.text for p in find_phrases(text)] == [
        "yesterday",
        "in two weeks",
        "last Tuesday",
    ]
    assert plausible("last Tuesday", dt.date(2026, 10, 20), today)
    assert not plausible("last Tuesday", dt.date(2026, 10, 21), today)
    assert not plausible("yesterday", dt.date(2026, 10, 21), today)
    assert plausible("next Friday", dt.date(2026, 10, 30), today)
    assert not plausible("next Friday", dt.date(2026, 10, 16), today)


# --------------------------------------------------------------------------- #
# list_rows
# --------------------------------------------------------------------------- #


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


TABLE = (
    "Aircraft N512QX\n\n| Pos | Serial |\n|---|---|\n"
    "| ENG-1 | SN-1001 |\n| ENG-2 | SN-1002 |\n| ENG-3 | SN-1003 |\n"
)


def engine_world(text: str = TABLE, rows: tuple[int, ...] = (1, 2, 3)) -> FakeWorld:
    world = FakeWorld(text)
    world.entity(
        "the Aircraft", schema=Aircraft, mentions=[], fields={"registration": "N512QX"}
    )
    for i in rows:
        world.entity(
            f"ENG-{i}",
            schema=Engine,
            parent="the Aircraft",
            mentions=[f"ENG-{i}"],
            fields={"position": f"ENG-{i}", "serial": f"SN-{1000 + i}"},
            matches=f"ENG-{i}" if text != TABLE else None,
        )
    return world


def all_rows(*_: Any) -> dict[str, Any]:
    return {
        "rows": [
            {
                "identity": f"ENG-{i}",
                "fields": {"position": f"ENG-{i}", "serial": f"SN-{1000 + i}"},
            }
            for i in (1, 2, 3)
        ]
    }


async def insert_aircraft(
    handler: Handler, escalation: EscalationConfig | None = None, text: str = TABLE
) -> ExtractionResult:
    extractor = create_extractor(
        [One(Aircraft)],
        transport=FakeTransport(handler),
        escalation=escalation,
        enable_inserts=True,
    )
    return await extractor.ainvoke(text)


async def test_rows_that_come_back_empty_are_reproduced_with_the_tier_off() -> None:
    handler = degrade(engine_world().handler, schema="Engine")
    result = await insert_aircraft(handler)
    assert result.records[0].after == {"registration": "N512QX", "engines": []}
    assert [s.reason for s in result.skipped] == ["validation_failed"] * 3
    assert not escalations(result)


async def test_the_tier_fills_rows_identified_by_position_codes() -> None:
    handler = degrade(engine_world().handler, schema="Engine")
    fake = FakeEscalation(all_rows)
    result = await insert_aircraft(handler, tier(fake))
    (record,) = result.records
    assert record.after == {
        "registration": "N512QX",
        "engines": [
            {"position": "ENG-1", "serial": "SN-1001"},
            {"position": "ENG-2", "serial": "SN-1002"},
            {"position": "ENG-3", "serial": "SN-1003"},
        ],
    }
    assert not result.skipped
    assert len(fake.calls) == 1  # the three rows share one call
    (call,) = escalations(result)
    assert call.field == "list_rows" and call.scope == "e0/engines"
    answer = record.fields["engines.0.serial"]
    assert "escalated:list_rows" in answer.warnings
    # every row went through attribution like any value
    assert Counter(c.kind for c in result.calls)["attribution"] >= 3


def short_discovery(inner: Handler, keep: int = 1) -> Handler:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        out = inner(model, payload, meta)
        if meta.kind == "discover" and meta.schema == "Engine":
            answer = out["answers"]["mentions"]
            answer["value"] = answer["value"][:keep]
            if answer.get("spans"):
                answer["spans"] = answer["spans"][:keep]
        return out

    return handler


async def test_a_count_above_the_rows_found_is_reproduced_then_fixed() -> None:
    handler = short_discovery(engine_world().handler)
    off = await insert_aircraft(handler)
    assert len(off.records[0].after["engines"]) == 1  # type: ignore[index]
    assert any("count_exceeds_discovered" in w for w in off.warnings)
    fake = FakeEscalation(all_rows)
    on = await insert_aircraft(handler, tier(fake))
    assert [e["position"] for e in on.records[0].after["engines"]] == [  # type: ignore[index]
        "ENG-1",
        "ENG-2",
        "ENG-3",
    ]
    assert len(fake.calls) == 1


async def test_discovery_and_the_empty_rows_share_one_call() -> None:
    handler = degrade(short_discovery(engine_world().handler), schema="Engine")
    fake = FakeEscalation(all_rows)
    result = await insert_aircraft(handler, tier(fake))
    assert [e["serial"] for e in result.records[0].after["engines"]] == [  # type: ignore[index]
        "SN-1001",
        "SN-1002",
        "SN-1003",
    ]
    assert len(fake.calls) == 1  # the rows of one list are asked once per invoke


async def test_a_row_the_model_invents_is_rejected() -> None:
    handler = degrade(engine_world().handler, schema="Engine")
    rows = {
        "rows": [
            {"identity": "ENG-1", "fields": {"position": "ENG-1", "serial": "SN-9999"}},
            {"identity": "ENG-7", "fields": {"position": "ENG-7", "serial": "SN-1002"}},
            {"identity": "ENG-2", "fields": {"position": "ENG-2", "serial": "SN-1002"}},
        ]
    }
    fake = FakeEscalation(lambda p, m: rows)
    result = await insert_aircraft(handler, tier(fake))
    engines = result.records[0].after["engines"]  # type: ignore[index]
    # ENG-7 is not in the input; SN-9999 is not in the input, so ENG-1 has no serial
    assert engines == [{"position": "ENG-2", "serial": "SN-1002"}]
    assert any("dropped (not in the input)" in w for w in result.warnings)
    assert "SN-9999" not in str(result.records[0].after)


def update_world() -> tuple[str, FakeWorld, dict[str, Any]]:
    text = "Update for N512QX: ENG-2 serial number is now SN-2222."
    world = FakeWorld(text)
    world.entity(
        "the Aircraft", schema=Aircraft, mentions=[], fields={"registration": "N512QX"}
    )
    world.entity(
        "ENG-2",
        schema=Engine,
        parent="the Aircraft",
        mentions=["ENG-2"],
        fields={"position": "ENG-2", "serial": "SN-2222"},
        matches="ENG-2",
    )
    existing = {
        "Aircraft": {
            "registration": "N512QX",
            "engines": [
                {"position": "ENG-1", "serial": "SN-1001"},
                {"position": "ENG-2", "serial": "SN-1002"},
            ],
        }
    }
    return text, world, existing


async def update_aircraft(
    escalation: EscalationConfig | None, **options: Any
) -> ExtractionResult:
    text, world, existing = update_world()
    extractor = create_extractor(
        [One(Aircraft)],
        transport=FakeTransport(degrade(world.handler, schema="Engine")),
        escalation=escalation,
        **options,
    )
    return await extractor.ainvoke(text, existing=existing)


async def test_an_update_to_one_row_that_was_not_written_is_reproduced_then_fixed() -> (
    None
):
    off = await update_aircraft(None)
    assert off.records[0].patches == ()  # the mentioned row's change was not proposed
    fake = FakeEscalation(
        lambda p, m: {
            "rows": [
                {
                    "identity": "ENG-2",
                    "fields": {"position": None, "serial": "SN-2222"},
                }
            ]
        }
    )
    on = await update_aircraft(tier(fake))
    ops = [(p.op, p.path, p.value) for p in on.records[0].patches]
    assert ops == [("replace", "/engines/1/serial", "SN-2222")]
    # the change gate (Clef) judged it like any other change
    assert any(c.kind == "gate" for c in on.calls)
    assert len(fake.calls) == 1


async def test_the_tier_never_deletes_a_row_on_the_models_word() -> None:
    fake = FakeEscalation(lambda p, m: {"rows": []})  # "there are no rows"
    result = await update_aircraft(
        tier(fake), enable_item_removals=True, enable_deletes=True
    )
    assert not [p for r in result.records for p in r.patches if p.op == "remove"]
    assert not result.deleted
    ops = [(p.op, p.path) for r in result.records for p in r.patches]
    assert ops == []  # the rows list stays as it was


async def test_a_value_that_is_not_in_the_text_is_not_taken() -> None:
    fake = FakeEscalation(
        lambda p, m: {
            "rows": [
                {"identity": "ENG-2", "fields": {"position": None, "serial": "SN-4242"}}
            ]
        }
    )
    result = await update_aircraft(tier(fake))
    assert result.records[0].patches == ()  # SN-4242 is not in the text


# --------------------------------------------------------------------------- #
# attribution_tiebreak
# --------------------------------------------------------------------------- #


class Contact(BaseModel):
    """A contact."""

    name: str = Field(description="the name")
    phone: Optional[str] = Field(None, description="the phone number")


class Member(BaseModel):
    """A team member."""

    name: str = Field(description="the name")
    commitment: Optional[str] = Field(None, description="what they committed to do")


CONTACT_TEXT = "Dana Reyes joined the board. Call her on 555-0142."
TRANSCRIPT = (
    "Priya: I will send the revised budget on Monday.\n"
    "Marcus: Sounds good, thank you.\n"
    "Priya: Please also review the draft."
)


def mid_band(inner: Handler, field: str, p: float = 0.6) -> Handler:
    """Clef's attribution of ``field`` lands in [0.4, 0.75), twice."""

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        if meta.kind == "attribution":
            answers = {q: noul(p if field in q else 0.95) for q in payload["questions"]}
            return clef_response(answers, model=payload["model"])
        return inner(model, payload, meta)

    return handler


async def run_contact(
    text: str,
    schema: type[BaseModel],
    fields: dict[str, Any],
    field: str,
    escalation: EscalationConfig | None,
    *,
    p: float = 0.6,
    **options: Any,
) -> ExtractionResult:
    world = FakeWorld(text)
    world.entity("the " + schema.__name__, schema=schema, mentions=[], fields=fields)
    extractor = create_extractor(
        [One(schema)],
        transport=FakeTransport(mid_band(world.handler, field, p)),
        escalation=escalation,
        enable_inserts=True,
        **options,
    )
    return await extractor.ainvoke(text)


CASES = [
    (CONTACT_TEXT, Contact, {"name": "Dana Reyes", "phone": "555-0142"}, "phone"),
    (
        TRANSCRIPT,
        Member,
        {"name": "Priya", "commitment": "send the revised budget on Monday"},
        "commitment",
    ),
]


@pytest.mark.parametrize(("text", "schema", "fields", "field"), CASES)
async def test_a_mid_band_attribution_drops_a_correct_value_with_the_tier_off(
    text: str, schema: type[BaseModel], fields: dict[str, Any], field: str
) -> None:
    result = await run_contact(text, schema, fields, field, None)
    (record,) = result.records
    assert record.after[field] is None  # type: ignore[index]  # a correct value lost
    assert not escalations(result)


@pytest.mark.parametrize(("text", "schema", "fields", "field"), CASES)
async def test_the_tier_breaks_the_tie_with_yes_as_prob_cap(
    text: str, schema: type[BaseModel], fields: dict[str, Any], field: str
) -> None:
    fake = FakeEscalation(lambda p, m: {"answer": "yes"})
    result = await run_contact(text, schema, fields, field, tier(fake))
    (record,) = result.records
    assert record.after[field] == fields[field]  # type: ignore[index]
    answer = record.fields[field]
    assert answer.attribution_p == pytest.approx(0.8)
    assert any(
        "attribution_tiebreak: clef 0.60 -> escalation 0.80" in w
        for w in answer.warnings
    )
    (call,) = escalations(result)
    assert call.field.startswith("attribution_tiebreak:")
    # the question the model got is the one Clef was asked, about THIS entity
    asked = fake.calls[0][1]["input"]
    assert fields[field] in asked and "Question:" in asked
    assert fake.calls[0][1]["schema"]["properties"]["answer"]["enum"] == ["yes", "no"]


@pytest.mark.parametrize(("text", "schema", "fields", "field"), CASES)
async def test_a_no_from_the_tier_keeps_the_value_out(
    text: str, schema: type[BaseModel], fields: dict[str, Any], field: str
) -> None:
    fake = FakeEscalation(lambda p, m: {"answer": "no"})
    result = await run_contact(text, schema, fields, field, tier(fake))
    assert result.records[0].after[field] is None  # type: ignore[index]
    assert len(fake.calls) == 1


async def test_a_confident_clef_answer_is_never_overridden() -> None:
    fake = FakeEscalation(lambda p, m: {"answer": "no"})
    result = await run_contact(
        CONTACT_TEXT, Contact, CASES[0][2], "phone", tier(fake), p=0.95
    )
    assert result.records[0].after["phone"] == "555-0142"  # type: ignore[index]
    assert not fake.calls  # outside the band the tier is not asked
    refuted = await run_contact(
        CONTACT_TEXT, Contact, CASES[0][2], "phone", tier(fake), p=0.1
    )
    assert refuted.records[0].after["phone"] is None  # type: ignore[index]
    assert not fake.calls  # a clear no from Clef stands too


async def test_prob_cap_below_the_keep_bar_cannot_keep_a_value() -> None:
    fake = FakeEscalation(lambda p, m: {"answer": "yes"})
    result = await run_contact(
        CONTACT_TEXT, Contact, CASES[0][2], "phone", tier(fake, prob_cap=0.7)
    )
    # yes counts as 0.7, under attribution_keep (0.75): a tie-break can be refused
    assert result.records[0].after["phone"] is None  # type: ignore[index]
    assert len(fake.calls) == 1


def test_prob_cap_must_stay_below_the_removal_and_delete_gates() -> None:
    for bad in (0.85, 0.9, 1.0, 0.5, 0.0, True, "0.8"):
        with pytest.raises(ValueError, match="prob_cap"):
            EscalationConfig(enabled=True, prob_cap=bad)  # type: ignore[arg-type]
    assert EscalationConfig(enabled=True).prob_cap == 0.8
    on = EscalationConfig(enabled=True, transport=FakeEscalation())
    with pytest.raises(ValueError, match=r"gates\.remove_threshold"):
        create_extractor(
            [Contact], escalation=on, gates=GateConfig(remove_threshold=0.8)
        )
    with pytest.raises(ValueError, match=r"match\.delete_threshold"):
        create_extractor(
            [Contact], escalation=on, match=MatchConfig(delete_threshold=0.75)
        )
    create_extractor([Contact], escalation=on)  # the defaults are fine
    # a disabled tier is not checked: it will never run
    create_extractor(
        [Contact],
        escalation=EscalationConfig(prob_cap=0.8),
        gates=GateConfig(remove_threshold=0.8),
    )


class Note(BaseModel):
    """A record that can be deleted."""

    name: str = Field(description="the name")


async def test_the_tier_is_never_asked_about_a_delete_or_a_removal() -> None:
    text = "Please remove the record for Ada Lovelace."
    world = FakeWorld(text)
    world.entity(
        "Ada Lovelace",
        schema=Note,
        mentions=["Ada Lovelace"],
        fields={"name": "Ada Lovelace"},
        matches="Ada",
    )
    world.delete("rec-1", 0.6)  # Clef's delete answer lands in the middle
    fake = FakeEscalation(lambda p, m: {"answer": "yes"})
    extractor = create_extractor(
        [Many(Note)],
        transport=FakeTransport(world.handler),
        escalation=tier(fake),
        enable_deletes=True,
    )
    result = await extractor.ainvoke(
        text,
        existing=[("rec-1", "Note", {"name": "Ada"})],
    )
    assert not result.deleted  # 0.6 is under match.delete_threshold, and stays so
    assert not [c for c in escalations(result) if "delete" in (c.field or "")]
    assert all(
        (c.field or "").startswith("attribution_tiebreak") for c in escalations(result)
    )


# --------------------------------------------------------------------------- #
# dotted_discovery
# --------------------------------------------------------------------------- #


class Ident(BaseModel):
    """Who the aircraft is."""

    msn: Optional[str] = Field(None, description="Manufacturer Serial Number")
    registration: Optional[str] = None


class Plane(BaseModel):
    """An aircraft described on a specification sheet."""

    identification: Ident = Field(default_factory=Ident)


SHEET = """\
# SPECIFICATION SHEET (1 of 2)

Airbus A320-200 available for lease.

| Item | Detail |
|---|---|
| Manufacturer Serial Number (MSN) | 8547 |
| Registration | AP-BLD |

# SPECIFICATION SHEET (2 of 2)

Boeing 737-800, also offered.

| Item | Detail |
|---|---|
| Manufacturer Serial Number (MSN) | 12345 |
| Registration | AP-BMX |
"""


def sheet_world() -> FakeWorld:
    world = FakeWorld(SHEET)
    for label, msn, reg in (
        ("Airbus A320-200", "8547", "AP-BLD"),
        ("Boeing 737-800", "12345", "AP-BMX"),
    ):
        world.entity(
            label,
            schema=Plane,
            mentions=[label, msn],
            fields={"identification.msn": msn, "identification.registration": reg},
        )
    return world


def blind_discovery(inner: Handler) -> Handler:
    """Discovery and the count find nothing; presence still says there are some."""

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        out = inner(model, payload, meta)
        if meta.kind == "discover":
            out["answers"]["mentions"]["value"] = []
            out["answers"]["mentions"]["spans"] = []
        if meta.kind == "count":
            out["answers"]["count"]["value"] = None
        return out

    return handler


def entities(*pairs: tuple[str, str]) -> Callable[..., dict[str, Any]]:
    return lambda *_: {
        "entities": [{"label": label, "identity": ident} for label, ident in pairs]
    }


async def run_sheet(escalation: EscalationConfig | None) -> ExtractionResult:
    extractor = create_extractor(
        [Many(Plane, identity="identification.msn")],
        transport=FakeTransport(blind_discovery(sheet_world().handler)),
        escalation=escalation,
    )
    return await extractor.ainvoke(SHEET)


GOOD = entities(("Airbus A320-200", "8547"), ("Boeing 737-800", "12345"))


async def test_a_dotted_identity_that_inserts_nothing_is_reproduced_then_fixed() -> (
    None
):
    off = await run_sheet(None)
    assert off.records == () and any("no_mentions_found" in w for w in off.warnings)
    fake = FakeEscalation(GOOD)
    on = await run_sheet(tier(fake))
    assert sorted(r.after["identification"]["msn"] for r in on.records) == [  # type: ignore[index]
        "12345",
        "8547",
    ]
    assert all(r.valid for r in on.records)
    assert len(fake.calls) == 1
    (call,) = escalations(on)
    assert call.field == "dotted_discovery"
    # the normal path ran: qualify, extraction, attribution
    kinds = Counter(c.kind for c in on.calls)
    assert kinds["qualify"] >= 1 and kinds["extract"] >= 2


async def test_an_entity_whose_quotes_are_not_in_the_input_is_rejected() -> None:
    fake = FakeEscalation(
        entities(
            ("Airbus A320-200", "8547"),
            ("Boeing 737-800", "99999"),  # the identity is not in the input
            ("Embraer E190", "12345"),  # the label is not in the input
        )
    )
    on = await run_sheet(tier(fake))
    assert [r.after["identification"]["msn"] for r in on.records] == ["8547"]  # type: ignore[index]
    assert any("1 of 3 entit(ies) had both quotes" in w for w in on.warnings)


async def test_dotted_discovery_only_runs_where_discovery_found_nothing() -> None:
    fake = FakeEscalation(GOOD)
    extractor = create_extractor(
        [Many(Plane, identity="identification.msn")],
        transport=FakeTransport(sheet_world().handler),  # discovery works
        escalation=tier(fake),
    )
    result = await extractor.ainvoke(SHEET)
    assert len(result.records) == 2 and not fake.calls


# --------------------------------------------------------------------------- #
# Limits, failures, accounting
# --------------------------------------------------------------------------- #


class Trip(BaseModel):
    """A trip."""

    name: str = Field(description="the trip name")
    left: dt.date = Field(description="the day the trip started")
    back: dt.date = Field(description="the day the trip ended")
    paid: dt.date = Field(description="the day the trip was paid")


TRIP_TEXT = (
    "Trip: Oslo. It started three days ago and ended yesterday, payment is due in "
    "two weeks."
)


def trip_answers(payload: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
    pairs = {
        "left": ("three days ago", "2026-10-20"),
        "back": ("yesterday", "2026-10-22"),
        "paid": ("in two weeks", "2026-11-06"),
    }
    field = (meta.field or "").split(":")[1]
    phrase, iso = pairs[field]
    return {"phrase": phrase, "iso_date": iso, "time": None}


def all_dates_null(world: FakeWorld) -> Handler:
    inner: Handler = world.handler

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        out = inner(model, payload, meta)
        if (
            meta.kind in ("extract", "crosscheck", "reask", "repair")
            and "answers" in out
        ):
            for wire, answer in out["answers"].items():
                if answer["type"] == "date":
                    out["answers"][wire] = clairic_answer(
                        "date", None, p_null=0.99, confidence=0.9
                    )
        return out

    return handler


async def run_three_dates(config: EscalationConfig) -> ExtractionResult:
    world = FakeWorld(TRIP_TEXT)
    world.entity(
        "the Trip",
        schema=Trip,
        mentions=[],
        fields={
            "name": "Oslo",
            "left": dt.date(2026, 10, 20),
            "back": dt.date(2026, 10, 22),
            "paid": dt.date(2026, 11, 6),
        },
    )
    extractor = create_extractor(
        [One(Trip)],
        transport=FakeTransport(all_dates_null(world)),
        reference_date=TODAY,
        escalation=config,
    )
    return await extractor.ainvoke(TRIP_TEXT)


async def test_max_calls_per_invoke_is_respected() -> None:
    fake = FakeEscalation(trip_answers)
    result = await run_three_dates(tier(fake, max_calls_per_invoke=2))
    assert len(fake.calls) == 2 and len(escalations(result)) == 2
    assert any("max_calls_per_invoke=2" in w for w in result.warnings)
    assert not result.records[0].valid  # the third date stayed null


async def test_a_per_kind_limit_is_respected_and_zero_switches_a_kind_off() -> None:
    fake = FakeEscalation(trip_answers)
    result = await run_three_dates(tier(fake, limits={"relative_dates": 1}))
    assert len(fake.calls) == 1
    assert any("limit of 1 relative_dates call" in w for w in result.warnings)
    none = FakeEscalation(trip_answers)
    await run_three_dates(tier(none, limits={"relative_dates": 0}))
    assert not none.calls


async def test_with_enough_calls_all_three_dates_are_fixed() -> None:
    fake = FakeEscalation(trip_answers)
    result = await run_three_dates(tier(fake))
    (record,) = result.records
    assert record.valid and len(fake.calls) == 3
    assert record.after == {
        "name": "Oslo",
        "left": dt.date(2026, 10, 20),
        "back": dt.date(2026, 10, 22),
        "paid": dt.date(2026, 11, 6),
    }
    assert Counter(c.field.split(":")[0] for c in escalations(result)) == {
        "relative_dates": 3
    }


class _Failing(FakeEscalation):
    def __init__(self, failure: Exception) -> None:
        super().__init__()
        self.failure = failure

    async def run(
        self, model: str, payload: Any, *, meta: Any, timeout: Any = None
    ) -> Any:
        self.calls.append((model, dict(payload), meta))
        raise self.failure


async def test_an_escalation_that_fails_leaves_the_default_result_and_says_so() -> None:
    failing = _Failing(
        UncertainOutcomeError("the outcome is unknown; it may have charged")
    )
    result = await run_task(THREE_DAYS, dt.date(2026, 10, 20), escalation=tier(failing))
    (record,) = result.records
    assert not record.valid  # exactly what the default setup gave
    assert any(
        w.startswith("escalation_failed: relative_dates") for w in result.warnings
    )
    (call,) = escalations(result)
    assert call.error and call.error.startswith("UncertainOutcomeError")  # logged
    assert len(failing.calls) == 1  # never sent again


async def test_a_refused_key_is_raised_not_swallowed() -> None:
    with pytest.raises(AuthError):
        await run_task(
            THREE_DAYS,
            dt.date(2026, 10, 20),
            escalation=tier(_Failing(AuthError("rejected key"))),
        )


async def test_strict_mode_raises_an_escalation_failure() -> None:
    world = task_world(THREE_DAYS, dt.date(2026, 10, 20))
    extractor = create_extractor(
        [One(Task)],
        transport=FakeTransport(degrade(world.handler, schema="Task", field="done_on")),
        reference_date=TODAY,
        escalation=tier(_Failing(TransportError("down"))),
        strict=True,
    )
    with pytest.raises(TransportError):
        await extractor.ainvoke(THREE_DAYS)


async def test_a_refusal_is_a_warning_not_an_error() -> None:
    class Refusing(FakeEscalation):
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: Any = None
        ) -> Any:
            reply = await super().run(model, payload, meta=meta, timeout=timeout)
            reply.body["no_answer"] = "refusal"
            reply.body["parsed"] = None
            return reply

    result = await run_task(
        THREE_DAYS, dt.date(2026, 10, 20), escalation=tier(Refusing())
    )
    assert not result.records[0].valid
    assert any("escalation_no_answer: relative_dates" in w for w in result.warnings)


async def test_reasoning_tokens_and_effort_reach_the_log_and_the_request() -> None:
    fake = FakeEscalation(
        lambda p, m: {
            "phrase": "three days ago",
            "iso_date": "2026-10-20",
            "time": None,
        },
        reasoning_tokens=40,
    )
    result = await run_task(
        THREE_DAYS,
        dt.date(2026, 10, 20),
        escalation=tier(fake, reasoning_effort="medium"),
    )
    assert fake.calls[0][1]["reasoning_effort"] == "medium"
    (call,) = escalations(result)
    assert call.usage["reasoning_tokens"] == 40
    assert call.usage["output_tokens"] == 45  # reasoning tokens are output tokens
    assert call.duration_ms > 0
    total = result.usage.by_kind["escalate"]
    assert total.reasoning_tokens == 40 and total.output_tokens == 45
    assert result.usage.reasoning_tokens == 40


def test_the_config_validates_and_a_disabled_one_runs_nothing() -> None:
    assert not EscalationConfig().enabled and not EscalationConfig().runs("list_rows")
    assert EscalationConfig(enabled=True).runs("list_rows")
    assert not EscalationConfig(enabled=True, kinds=frozenset()).runs("list_rows")
    with pytest.raises(ValueError, match="unknown escalation kind"):
        EscalationConfig(kinds=frozenset({"magic"}))
    with pytest.raises(ValueError, match="reasoning_effort"):
        EscalationConfig(reasoning_effort="extreme")
    with pytest.raises(ValueError, match="limits"):
        EscalationConfig(limits={"magic": 1})
    with pytest.raises(ValueError, match="max_calls_per_invoke"):
        EscalationConfig(max_calls_per_invoke=-1)
    for effort in ("none", "low", "medium", "high", "xhigh", "max", None):
        EscalationConfig(reasoning_effort=effort)
    assert hash(EscalationConfig(enabled=True)) == hash(EscalationConfig(enabled=True))


async def test_the_default_extractor_has_no_tier_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "k" * 40)  # a key alone routes nothing
    monkeypatch.delenv("TRUSTCALL_S1_ESCALATE", raising=False)
    world = task_world(THREE_DAYS, dt.date(2026, 10, 20))
    extractor = create_extractor(
        [One(Task)],
        transport=FakeTransport(degrade(world.handler, schema="Task", field="done_on")),
        reference_date=TODAY,
    )
    result = await extractor.ainvoke(THREE_DAYS)
    assert not escalations(result)
    assert all(not w.startswith("escalation") for w in result.warnings)


async def test_an_owned_escalation_transport_is_closed_with_the_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trustcall_system_one import OpenAIResponsesTransport

    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "openai")
    extractor = create_extractor([One(Task)])
    chosen = extractor.core.escalation(None)
    assert chosen is not None and isinstance(chosen.transport, OpenAIResponsesTransport)
    again = extractor.core.escalation(None)
    assert again is not None and again.transport is chosen.transport  # one, owned
    closed: list[str] = []

    async def aclose() -> None:
        closed.append("closed")

    chosen.transport.aclose = aclose
    await extractor.aclose()
    assert closed == ["closed"]
