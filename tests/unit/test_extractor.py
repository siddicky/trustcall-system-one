"""End-to-end tests of the single-entity insert pipeline on a fake transport."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import asyncio
import contextlib
import dataclasses
import datetime as dt
import enum
import itertools
import threading
from decimal import Decimal
from typing import Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import (
    Script,
    ca,
    kind_counts,
    kinds,
    make_extractor,
    pick,
)
from trustcall_system_one import (
    AuthError,
    ExtractionResult,
    ExtractionValidationError,
    Many,
    One,
    Prices,
    ResultRecord,
    ServerError,
    StateTruncatedError,
    TransportResponse,
    TrustcallSystemOneError,
    VerifyConfig,
    create_extractor,
    s1_field,
)
from trustcall_system_one.testing import FakeTransport, clef_response, noul, route

PNG = "data:image/png;base64,iVBORw0KGgo="


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Level(enum.Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class Everything(BaseModel):
    """A record with one field of every flat type."""

    name: str
    age: int
    height: float
    price: Decimal
    born: dt.date
    seen: dt.datetime
    opens: dt.time
    lasts: dt.timedelta
    tags: list[str]
    labels: set[str]
    kinds: list[Literal["bug", "feature", "docs"]]
    active: bool
    verified: Optional[bool] = None
    status: Literal["open", "closed"]
    color: Color
    priority: Level = Field(default=Level.LOW, json_schema_extra=s1_field(clef="score"))
    ident: Union[int, str]
    kind: Literal["person"]


TEXT = (
    "Ada Lovelace, aged 36, is 1.65 m tall and paid $12.50 on 1815-12-10. "
    "Tags: alpha, beta. Labels: x, y. She filed a bug and a feature. "
    "The ticket is open, the colour is blue, the priority is high, "
    "and the code is 42."
)


def everything_script() -> Script:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace"),
        "age": ca("number", 36),
        "height": ca("number", 1.65),
        "price": ca("number", "12.50"),
        "born": ca("date", "1815-12-10"),
        "seen": ca("datetime", "1815-12-10T09:30:00"),
        "opens": ca("time", "09:30:00"),
        "lasts": ca("duration", 2.5),
        "tags": ca("list", ["alpha", "beta"], find_all=["alpha", "beta"]),
        "labels": ca("list", ["x", "y", "x"], find_all=["x", "y", "x"]),
        "kinds": ca("list", ["bug", "feature"], find_all=["bug", "feature"]),
        "ident": ca("text", "42", find="42"),
    }
    script.clef = {
        "active": {"type": "noul", "noul": 0.97},
        "verified": pick("unstated", ["yes", "no", "unstated"]),
        "status": pick("open", ["open", "closed"]),
        "color": pick("BLUE", ["RED", "BLUE"]),
        "priority": {
            "type": "score",
            "score": 2.0,
            "legend": {},
            "probabilities": {"0": 0.02, "1": 0.03, "2": 0.95},
            "confidence": 0.95,
        },
        "ident.__type": pick("int", ["int", "str"]),
        "kinds.el0": pick("bug", ["bug", "feature", "docs", "none_of_these"]),
        "kinds.el1": pick("feature", ["bug", "feature", "docs", "none_of_these"]),
    }
    return script


async def test_every_flat_type_end_to_end() -> None:
    script = everything_script()
    extractor = make_extractor([Everything], script)
    result = await extractor.ainvoke(TEXT)

    (record,) = result.records
    assert isinstance(record, ResultRecord)
    assert record.valid, record.errors
    assert record.kind == "insert"
    assert record.json_doc_id is None
    value = record.value
    assert isinstance(value, Everything)
    assert value == Everything(
        name="Ada Lovelace",
        age=36,
        height=1.65,
        price=Decimal("12.50"),
        born=dt.date(1815, 12, 10),
        seen=dt.datetime(1815, 12, 10, 9, 30),
        opens=dt.time(9, 30),
        lasts=dt.timedelta(seconds=2.5),
        tags=["alpha", "beta"],
        labels={"x", "y"},
        kinds=["bug", "feature"],
        active=True,
        verified=None,
        status="open",
        color=Color.BLUE,
        priority=Level.HIGH,
        ident=42,
        kind="person",
    )
    assert result.responses == (value,)
    assert result.attempts == 1
    assert result.response_metadata == [{"id": "Everything:e0", "json_doc_id": None}]
    assert result.skipped == ()

    # Field metadata: engine, source, spans in the ORIGINAL text.
    name = record.fields["name"]
    assert (name.engine, name.source, name.value) == (
        "clairic",
        "clairic",
        "Ada Lovelace",
    )
    assert name.span is not None
    assert TEXT[name.span[0] : name.span[1]] == "Ada Lovelace"
    assert name.confidence == 0.99
    assert name.attribution_p == 0.95
    assert name.reask is False
    tags = record.fields["tags"]
    assert tags.spans is not None
    assert [TEXT[a:b] for a, b in tags.spans] == ["alpha", "beta"]
    assert tags.attribution_p == (0.95, 0.95)
    labels = record.fields["labels"]
    assert labels.value == {"x", "y"}
    assert record.fields["active"].engine == "clef"
    assert record.fields["active"].source == "clef"
    assert record.fields["active"].value is True
    assert record.fields["verified"].value is None
    assert record.fields["verified"].source == "default"  # unstated: default applies
    assert record.fields["verified"].warnings == ("unstated",)
    assert record.fields["priority"].value == Level.HIGH
    assert record.fields["ident"].value == 42
    assert record.fields["ident"].raw == {"text": "42", "type": "int"}
    assert record.fields["kind"].source == "const"
    assert record.fields["kind"].value == "person"
    assert record.fields["born"].span is None
    assert record.entity is not None
    assert (record.entity.key, record.entity.label) == ("e0", "the Everything")
    assert record.entity.qualify_p == 0.95
    assert record.rejected == ()

    # The calls: presence, then Clairic and Clef in parallel, then the element
    # choices, then one attribution request for every Clairic value.
    seen = kinds(script.transport)
    assert seen[0] == "presence"
    assert sorted(seen[1:3]) == ["clef_fields", "extract"]
    assert seen[3:] == ["clef_fields", "attribution"]
    assert [c.kind for c in result.calls].count("attribution") == 1
    assert result.usage.requests == 5
    assert result.cost.complete is False  # Clef has no published price
    assert result.cost.unpriced_models == ("@cf/cloudflare/clef",)
    assert kind_counts(script.transport)["reask"] == 0


# --------------------------------------------------------------------------- #
# Small schemas for the rest of the tests
# --------------------------------------------------------------------------- #


class Person(BaseModel):
    """A person mentioned in the text."""

    name: str
    age: Optional[int] = None


class Ticket(BaseModel):
    """A support ticket."""

    title: str


PERSON_TEXT = "Ada Lovelace is 36 years old."


def person_script(**kwargs: Any) -> Script:
    script = Script(**kwargs)
    script.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace"),
        "age": ca("number", 36),
    }
    return script


# --------------------------------------------------------------------------- #
# Presence
# --------------------------------------------------------------------------- #


async def test_absent_entity_makes_no_record_and_no_extraction_calls() -> None:
    script = person_script(presence=0.05)
    result = await make_extractor([Person], script).ainvoke("The weather is nice.")
    assert result.records == ()
    assert result.responses == ()
    assert result.attempts == 0
    assert kinds(script.transport) == ["presence"]
    assert [c.kind for c in result.calls] == ["presence"]
    assert result.usage.requests == 1


async def test_presence_just_below_the_threshold_is_absent() -> None:
    script = person_script(presence=0.19)
    result = await make_extractor([Person], script).ainvoke(PERSON_TEXT)
    assert result.records == ()
    assert kinds(script.transport) == ["presence"]


@pytest.mark.parametrize("p", [0.2, 0.35, 0.5, 0.99])
async def test_present_or_uncertain_entity_is_extracted_and_qualify_p_kept(
    p: float,
) -> None:
    script = person_script(presence=p)
    result = await make_extractor([Person], script).ainvoke(PERSON_TEXT)
    (record,) = result.records
    assert record.entity is not None and record.entity.qualify_p == p
    assert "extract" in kinds(script.transport)


async def test_presence_question_names_the_schema() -> None:
    script = person_script()
    await make_extractor([Person], script).ainvoke(PERSON_TEXT)
    (call,) = script.calls("presence")
    question = call[1]["questions"]["presence"]
    assert question["instructions"] == (
        "Does `input` contain a Person: A person mentioned in the text?"
    )
    assert question["type"] == "noul"
    assert set(question["criteria"]) == {"true", "false"}
    assert call[1]["state"] == {"input": PERSON_TEXT}
    assert call[2].schema == "Person"


async def test_presence_thresholds_are_configurable() -> None:
    from trustcall_system_one import PresenceConfig

    script = person_script(presence=0.4)
    extractor = make_extractor(
        [Person],
        script,
        presence=PresenceConfig(threshold_absent=0.45, threshold_present=0.5),
    )
    assert (await extractor.ainvoke(PERSON_TEXT)).records == ()


async def test_forcing_a_schema_skips_presence_and_runs_only_that_schema() -> None:
    script = person_script(presence=0.0)
    extractor = make_extractor([Person, Ticket], script)
    by_class = await extractor.ainvoke(PERSON_TEXT, schema=Person)
    assert by_class.records[0].schema_name == "Person"
    assert by_class.records[0].entity is not None
    assert by_class.records[0].entity.qualify_p is None
    assert "presence" not in kinds(script.transport)
    assert {meta.schema for _, _, meta in script.transport.calls} == {"Person"}
    script.transport.calls.clear()
    by_name = await extractor.ainvoke(PERSON_TEXT, schema="Person")
    assert by_name.records[0].schema_name == "Person"
    assert "presence" not in kinds(script.transport)


async def test_forcing_an_unknown_schema_is_an_error() -> None:
    extractor = make_extractor([Person], person_script())
    with pytest.raises(ValueError, match="Ticket"):
        await extractor.ainvoke(PERSON_TEXT, schema=Ticket)
    with pytest.raises(ValueError, match="Nope"):
        await extractor.ainvoke(PERSON_TEXT, schema="Nope")


async def test_several_schemas_run_in_parallel_each_behind_its_own_gate() -> None:
    script = person_script()

    def presence(model: str, payload: dict[str, Any], meta: Any) -> Any:
        p = 0.9 if meta.schema == "Person" else 0.01
        return clef_response({"presence": noul(p)})

    script.transport = FakeTransport(
        route(
            presence=presence,
            extract=script._extract,
            clef_fields=script._clef_fields,
            attribution=script._attribution,
        )
    )
    result = await make_extractor([Ticket, Person], script).ainvoke(PERSON_TEXT)
    assert [r.schema_name for r in result.records] == ["Person"]
    assert [m.schema for _, _, m in script.calls("presence")] == ["Ticket", "Person"]
    assert result.attempts == 1


async def test_two_present_schemas_give_records_in_schema_order() -> None:
    script = person_script()
    script.clairic["title"] = ca("text", "Ada Lovelace", find="Ada Lovelace")
    result = await make_extractor([Ticket, Person], script).ainvoke(PERSON_TEXT)
    assert [r.schema_name for r in result.records] == ["Ticket", "Person"]
    assert [type(m).__name__ for m in result.responses] == ["Ticket", "Person"]
    assert result.response_metadata == [
        {"id": "Ticket:e0", "json_doc_id": None},
        {"id": "Person:e0", "json_doc_id": None},
    ]


# --------------------------------------------------------------------------- #
# Passes and instructions
# --------------------------------------------------------------------------- #


async def test_split_by_field_makes_one_clairic_request_per_field() -> None:
    from trustcall_system_one import PassesConfig

    script = person_script()
    extractor = make_extractor([Person], script, passes=PassesConfig(split="field"))
    result = await extractor.ainvoke(PERSON_TEXT)
    extracts = script.calls("extract")
    assert sorted(list(c[1]["fields"]) for c in extracts) == [["age"], ["name"]]
    assert sorted(c[2].field or "" for c in extracts) == ["age", "name"]
    assert result.records[0].value == Person(name="Ada Lovelace", age=36)


@pytest.mark.parametrize("split", ["entity", "submodel"])
async def test_other_splits_ask_a_flat_schema_in_one_request(split: str) -> None:
    from trustcall_system_one import PassesConfig

    script = person_script()
    extractor = make_extractor([Person], script, passes=PassesConfig(split=split))  # type: ignore[arg-type]
    await extractor.ainvoke(PERSON_TEXT)
    (extract,) = script.calls("extract")
    assert sorted(extract[1]["fields"]) == ["age", "name"]
    assert extract[2].field is None


async def test_instructions_go_through_the_template_without_touching_the_plan() -> None:
    script = person_script()
    extractor = make_extractor(
        [Person],
        script,
        verify=VerifyConfig(instruction_template="About {label}: {desc}"),
    )
    plan = extractor.plan_for(Person)
    before = plan.spec(("name",)).clairic
    await extractor.ainvoke(PERSON_TEXT)
    (extract,) = script.calls("extract")
    assert extract[1]["fields"]["name"]["instructions"] == "About the Person: name"
    assert plan.spec(("name",)).clairic == before  # compiled instructions unchanged
    assert plan.spec(("name",)).clairic is not None
    assert plan.spec(("name",)).clairic.instructions == "name"  # type: ignore[union-attr]


async def test_default_template_names_the_entity() -> None:
    script = person_script()
    await make_extractor([Person], script).ainvoke(PERSON_TEXT)
    (extract,) = script.calls("extract")
    assert extract[1]["fields"]["name"]["instructions"] == "the Person's name"
    state = extract[1]["state"]
    assert state.startswith('<focus entity="the Person">')
    assert f"<input>\n{PERSON_TEXT}\n</input>" in state


async def test_one_wrapper_label_replaces_the_default_label() -> None:
    script = person_script()
    extractor = make_extractor([One(Person, label="the customer")], script)
    result = await extractor.ainvoke(PERSON_TEXT)
    (extract,) = script.calls("extract")
    assert extract[1]["fields"]["name"]["instructions"] == "the customer's name"
    assert result.records[0].entity is not None
    assert result.records[0].entity.label == "the customer"


async def test_images_and_reference_date_ride_on_every_request() -> None:
    script = person_script()
    extractor = make_extractor([Person], script, reference_date="2026-01-02")
    await extractor.ainvoke(PERSON_TEXT, images=[PNG])
    assert script.transport.calls
    for _, payload, _ in script.transport.calls:
        assert payload["images"] == [PNG]
    for _, payload, _ in script.calls("extract"):
        assert payload["reference_date"] == "2026-01-02"
    script.transport.calls.clear()
    await extractor.ainvoke(PERSON_TEXT, reference_date=dt.date(2025, 5, 6))
    assert script.calls("extract")[0][1]["reference_date"] == "2025-05-06"
    assert "images" not in script.calls("extract")[0][1]


async def test_invalid_reference_date_is_rejected() -> None:
    extractor = make_extractor([Person], person_script())
    with pytest.raises(ValueError, match="reference_date"):
        await extractor.ainvoke(PERSON_TEXT, reference_date="yesterday")
    with pytest.raises(ValueError, match="reference_date"):
        make_extractor([Person], person_script(), reference_date="soon")


async def test_extraction_input_dict_and_keyword_overrides() -> None:
    script = person_script()
    extractor = make_extractor([Person], script)
    await extractor.ainvoke(
        {"text": PERSON_TEXT, "reference_date": "2024-02-03", "images": [PNG]}
    )
    payload = script.calls("extract")[0][1]
    assert payload["reference_date"] == "2024-02-03"
    assert payload["images"] == [PNG]
    script.transport.calls.clear()
    await extractor.ainvoke(
        {"text": PERSON_TEXT, "reference_date": "2024-02-03"},
        reference_date="2025-01-01",
    )
    assert script.calls("extract")[0][1]["reference_date"] == "2025-01-01"


async def test_empty_input_extracts_nothing() -> None:
    script = person_script()
    result = await make_extractor([Person], script).ainvoke("   ")
    assert result.records == ()
    assert script.transport.calls == []
    assert result.warnings


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


def filler(count: int, tag: str) -> str:
    """Filler text with no repeated stretch, so a chunk has one place in it."""
    return " ".join(f"{tag}{i}" for i in range(count)) + " "


LONG = filler(150, "q") + "Ada Lovelace is 36 years old. " + filler(150, "r")
BIG = (
    "Ada Lovelace lives here. "
    + filler(2500, "filler")
    + "She moved to Paris. "
    + filler(2500, "more")
)


class Located(BaseModel):
    name: str
    city: Optional[str] = None


def located_script() -> Script:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace"),
        "city": ca("text", "Paris", find="Paris"),
    }
    return script


def input_ranges(
    script: Script, text: str, kind: str = "extract"
) -> list[tuple[int, int]]:
    """The ranges of ``text`` that the calls of one kind carried."""
    ranges = []
    for _, payload, _ in script.calls(kind):
        state = payload["state"]
        body = state[state.index("<input>\n") + 8 : state.index("\n</input>")]
        body = body.removeprefix("…\n").removesuffix("\n…")
        start = text.index(body)
        ranges.append((start, start + len(body)))
    return ranges


def covers(ranges: list[tuple[int, int]], length: int) -> bool:
    reach = 0
    for start, end in sorted(ranges):
        if start > reach:
            return False
        reach = max(reach, end)
    return reach >= length


async def test_a_truncated_state_is_chunked_over_the_whole_input() -> None:
    script = located_script()
    script.limit = lambda state, n: 5000 if n == 0 else None
    extractor = make_extractor([Located], script, verify=VerifyConfig(window_chars=100))
    result = await extractor.ainvoke(BIG)
    record = result.records[0]
    assert record.valid
    assert record.value == Located(name="Ada Lovelace", city="Paris")
    span = record.fields["city"].span
    assert span is not None and BIG[span[0] : span[1]] == "Paris"
    name_span = record.fields["name"].span
    assert name_span is not None and BIG[name_span[0] : name_span[1]] == "Ada Lovelace"
    # the first request was cut at 5000 characters; the chunks that follow cover
    # everything, overlapping by at least window_chars
    ranges = input_ranges(script, BIG)
    full, chunks = ranges[0], sorted(ranges[1:])
    assert full == (0, len(BIG))
    assert len(chunks) >= 3
    assert covers(chunks, len(BIG))
    assert chunks[0][0] == 0 and chunks[-1][1] == len(BIG)
    for (_, end), (start, _) in itertools.pairwise(chunks):
        assert end - start >= 100
    assert max(end - start for start, end in chunks) < 5000
    assert any("clairic_chunked" in w for w in result.warnings)
    assert not any("clairic_truncated" in w for w in result.warnings)
    assert {c.kind for c in result.calls if c.kind == "extract"} == {"extract"}


async def test_chunking_merges_scalars_preferring_non_null_then_confidence() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace", confidence=0.9),
        "city": ca("text", "Paris", find="Paris", confidence=0.95),
    }
    # a second, better-attested name and a conflicting city later in the text
    text = (
        "Ada Lovelace "
        + filler(2000, "x")
        + "Ada Lovelace "
        + filler(2000, "y")
        + "Paris"
    )
    script.limit = lambda state, n: 4000 if n == 0 else None
    extractor = make_extractor(
        [Located],
        script,
        verify=VerifyConfig(window_chars=50, reask_low_confidence=False),
    )
    record = (await extractor.ainvoke(text)).records[0]
    assert record.value == Located(name="Ada Lovelace", city="Paris")
    city = record.fields["city"].span
    assert city is not None and text[city[0] : city[1]] == "Paris"


async def test_chunk_answers_for_a_scalar_are_adjudicated() -> None:
    script = Script()
    script.adjudicate = {"city": pick("a", ["a", "b", "neither"])}  # a: most confident
    texts = {"first": 0.6, "second": 0.9}

    def extract(model: str, payload: dict[str, Any], meta: Any) -> Any:
        from trustcall_system_one.testing import clairic_answer, clairic_response

        state = payload["state"]
        cut = state[:4000] if meta.kind == "extract" and len(state) > 6000 else state
        truncated = cut is not state
        answers = {}
        for name in payload["fields"]:
            if name == "name":
                at = cut.find("Ada")
                answers[name] = (
                    clairic_answer("text", "Ada", span=[at, at + 3])
                    if at >= 0
                    else clairic_answer("text", None, p_null=0.99)
                )
            else:
                if "second" in cut:
                    answers[name] = clairic_answer(
                        "text",
                        "second",
                        confidence=texts["second"],
                        span=[cut.index("second"), cut.index("second") + 6],
                    )
                elif "first" in cut:
                    answers[name] = clairic_answer(
                        "text",
                        "first",
                        confidence=texts["first"],
                        span=[cut.index("first"), cut.index("first") + 5],
                    )
                else:
                    answers[name] = clairic_answer("text", None, p_null=0.99)
        return clairic_response(answers, truncated=truncated, state_text=cut)

    script.transport = FakeTransport(
        route(
            presence=script._presence,
            extract=extract,
            attribution=script._attribution,
            adjudicate=script._adjudicate,
        )
    )
    text = "Ada. first " + filler(3000, "w") + " second " + filler(3000, "z")
    extractor = make_extractor(
        [Located],
        script,
        verify=VerifyConfig(window_chars=50, reask_low_confidence=False),
    )
    record = (await extractor.ainvoke(text)).records[0]
    assert isinstance(record.value, Located) and record.value.city == "second"
    span = record.fields["city"].span
    assert span is not None and text[span[0] : span[1]] == "second"


CITY_TEXT = (
    "Ada Lovelace. Grace lives in London. "
    + filler(2000, "w")
    + "Ada moved to Paris. "
    + filler(2000, "z")
)


def conflict_script(**kwargs: Any) -> Script:
    """Chunks that disagree: London (another person's, sure) vs Paris."""
    from trustcall_system_one.testing import clairic_answer, clairic_response

    script = Script(**kwargs)

    def extract(model: str, payload: dict[str, Any], meta: Any) -> Any:
        state = payload["state"]
        cut = state[:4000] if len(state) > 6000 else state
        answers = {}
        for name in payload["fields"]:
            if name == "name":
                at = cut.find("Ada Lovelace")
                answers[name] = (
                    clairic_answer("text", "Ada Lovelace", span=[at, at + 12])
                    if at >= 0
                    else clairic_answer("text", None, p_null=0.99)
                )
                continue
            for city, conf in (("London", 0.97), ("Paris", 0.90)):
                at = cut.find(city)
                if at >= 0:
                    answers[name] = clairic_answer(
                        "text", city, confidence=conf, span=[at, at + len(city)]
                    )
                    break
            else:
                answers[name] = clairic_answer("text", None, p_null=0.99)
        return clairic_response(answers, truncated=cut is not state, state_text=cut)

    script.transport = FakeTransport(
        route(
            presence=script._presence,
            extract=extract,
            attribution=script._attribution,
            adjudicate=script._adjudicate,
        )
    )
    return script


def conflict_extractor(script: Script, **verify: Any) -> Any:
    return make_extractor(
        [Located],
        script,
        verify=VerifyConfig(window_chars=50, reask_low_confidence=False, **verify),
    )


async def test_disagreeing_chunks_are_adjudicated_over_every_candidate() -> None:
    script = conflict_script()
    # candidates by confidence: a = London (0.97), b = Paris (0.90)
    script.adjudicate = {"city": pick("b", ["a", "b", "neither"])}
    result = await conflict_extractor(script).ainvoke(CITY_TEXT)
    record = result.records[0]
    assert isinstance(record.value, Located) and record.value.city == "Paris"
    span = record.fields["city"].span
    assert span is not None and CITY_TEXT[span[0] : span[1]] == "Paris"
    (call,) = script.calls("adjudicate")
    question = call[1]["questions"]["adjudicate"]
    assert question["criteria"] == {
        "a": "London",
        "b": "Paris",
        "neither": "The input states none of these",
    }
    assert call[2].field == "city"
    assert any("chunk_conflict" in w and "adjudicated" in w for w in result.warnings)
    # only the ladder's own attribution check ran, not the fallback
    assert [q for c in script.calls("attribution") for q in c[1]["questions"]] == [
        "attr.name",
        "attr.city",
    ]


async def test_an_indecisive_adjudication_falls_back_to_attribution() -> None:
    script = conflict_script()
    script.adjudicate = {"city": pick("neither", ["a", "b", "neither"])}
    # London is the more confident chunk answer but is another person's
    script.attribution = {"attr.city.0": 0.1, "attr.city.1": 0.95}
    result = await conflict_extractor(script).ainvoke(CITY_TEXT)
    record = result.records[0]
    assert isinstance(record.value, Located) and record.value.city == "Paris"
    asked = [q for c in script.calls("attribution") for q in c[1]["questions"]]
    assert asked[:2] == ["attr.city.0", "attr.city.1"]  # in confidence order
    assert any("first that is stated" in w for w in result.warnings)


async def test_low_probability_adjudication_is_not_decisive_either() -> None:
    script = conflict_script()
    script.adjudicate = {"city": pick("b", ["a", "b", "neither"], p=0.5)}
    script.attribution = {"attr.city.0": 0.05, "attr.city.1": 0.9}
    record = (await conflict_extractor(script).ainvoke(CITY_TEXT)).records[0]
    assert isinstance(record.value, Located) and record.value.city == "Paris"


async def test_when_no_check_settles_it_the_ladder_judges_the_top_candidate() -> None:
    script = conflict_script()
    script.adjudicate = {"city": pick("neither", ["a", "b", "neither"])}
    script.attribution = {
        "attr.city.0": 0.1,
        "attr.city.1": 0.2,
        "attr.city": 0.1,  # the ladder's own check of the standing value
    }
    result = await conflict_extractor(script).ainvoke(CITY_TEXT)
    record = result.records[0]
    assert isinstance(record.value, Located) and record.value.city is None
    assert record.fields["city"].rejected_reason == "attribution"
    assert record.fields["city"].raw == "London"  # the most confident candidate
    assert any("no check settled it" in w for w in result.warnings)


async def test_without_attribution_the_most_confident_candidate_stands() -> None:
    script = conflict_script()
    script.adjudicate = {"city": pick("neither", ["a", "b", "neither"])}
    result = await conflict_extractor(script, attribution=False).ainvoke(CITY_TEXT)
    record = result.records[0]
    assert isinstance(record.value, Located) and record.value.city == "London"
    assert script.calls("attribution") == []


class Notes(BaseModel):
    notes: list[str]


async def test_chunking_unions_list_elements_in_input_order_without_duplicates() -> (
    None
):
    text = (
        "alpha "
        + filler(1000, "a")
        + "beta "
        + filler(1000, "b")
        + "gamma "
        + filler(1000, "c")
        + "delta"
    )
    script = Script()
    script.clairic = {
        "notes": ca(
            "list", [], find_all=["delta", "alpha", "gamma", "beta"], confidence=0.95
        )
    }
    script.limit = lambda state, n: 4000 if n == 0 else None
    # the chunk answers come back in the model's own order, with the ones in the
    # overlaps reported by two chunks
    extractor = make_extractor(
        [Notes], script, verify=VerifyConfig(window_chars=3500, attribution=False)
    )
    result = await extractor.ainvoke(text)
    record = result.records[0]
    assert record.value is not None
    assert record.value == Notes(notes=["alpha", "beta", "gamma", "delta"])
    answer = record.fields["notes"]
    assert answer.spans is not None
    assert [text[a:b] for a, b in answer.spans] == ["alpha", "beta", "gamma", "delta"]
    chunks = input_ranges(script, text)[1:]
    assert any(
        text[a:b].count("beta") and text[c:d].count("beta")
        for (a, b), (c, d) in itertools.pairwise(chunks)
    ), "a list element should have sat in two chunks"


async def test_a_chunk_that_is_still_too_long_is_split_smaller_once() -> None:
    script = located_script()

    def limit(state: str, n: int) -> int | None:
        if n == 0:
            return 5000
        return 1000 if len(state) > 2000 else None

    script.limit = limit
    extractor = make_extractor([Located], script, verify=VerifyConfig(window_chars=20))
    result = await extractor.ainvoke(BIG)
    record = result.records[0]
    assert record.value == Located(name="Ada Lovelace", city="Paris")
    ranges = input_ranges(script, BIG)[1:]
    sizes = sorted({end - start for start, end in ranges})
    assert sizes[-1] > 2000 - 200  # the first round's chunks, which were too long
    assert sizes[0] < 1000  # the second round's
    assert covers([r for r in ranges if r[1] - r[0] < 1000], len(BIG))
    assert not any("clairic_truncated" in w for w in result.warnings)
    assert any("clairic_chunked" in w for w in result.warnings)


SMALL = filler(150, "s") + "Ada Lovelace moved to Paris."


async def test_chunks_that_stay_truncated_warn_that_input_was_not_read() -> None:
    script = located_script()
    script.limit = lambda state, n: 60  # shorter than the focus block: never fits
    extractor = make_extractor([Located], script, verify=VerifyConfig(window_chars=20))
    result = await extractor.ainvoke(SMALL)
    assert any("clairic_truncated" in w and "not read" in w for w in result.warnings)
    assert any("clairic_chunked" in w for w in result.warnings)
    # the full request, a first round of chunks and one smaller round: no third
    sizes = [end - start for start, end in input_ranges(script, SMALL)]
    assert sizes[0] == len(SMALL)
    assert 48 in sizes and 41 in sizes  # the first round, then smaller, down to
    assert max(sizes[-3:]) == 41  # the floor of 2 * window_chars + 1


async def test_strict_raises_when_chunks_stay_truncated() -> None:
    script = located_script()
    script.limit = lambda state, n: 60
    extractor = make_extractor(
        [Located], script, strict=True, verify=VerifyConfig(window_chars=20)
    )
    with pytest.raises(StateTruncatedError, match="not read"):
        await extractor.ainvoke(SMALL)


async def test_strict_does_not_raise_when_the_chunks_succeed() -> None:
    script = located_script()
    script.limit = lambda state, n: 5000 if n == 0 else None
    extractor = make_extractor([Located], script, strict=True)
    record = (await extractor.ainvoke(BIG)).records[0]
    assert record.value == Located(name="Ada Lovelace", city="Paris")


async def test_without_a_state_text_the_chunk_size_is_a_conservative_default() -> None:
    from trustcall_system_one._chunk import DEFAULT_CHUNK_CHARS

    script = located_script()
    script.truncated = 1  # flagged truncated, but state_text says nothing was cut
    extractor = make_extractor([Located], script)
    text = "Ada Lovelace. " + filler(3500, "x") + " Paris"
    record = (await extractor.ainvoke(text)).records[0]
    assert record.value == Located(name="Ada Lovelace", city="Paris")
    chunks = input_ranges(script, text)[1:]
    assert max(end - start for start, end in chunks) <= DEFAULT_CHUNK_CHARS
    assert covers(chunks, len(text))


async def test_a_truncated_short_input_is_still_split_in_two() -> None:
    script = person_script(truncated=1)
    extractor = make_extractor([Person], script, verify=VerifyConfig(window_chars=20))
    result = await extractor.ainvoke(LONG)
    ranges = input_ranges(script, LONG)
    assert len(ranges) == 3  # the full request and two chunks
    assert covers(ranges[1:], len(LONG))
    record = result.records[0]
    span = record.fields["name"].span
    assert span is not None and LONG[span[0] : span[1]] == "Ada Lovelace"


async def test_an_input_too_short_to_split_warns_instead_of_retrying() -> None:
    script = person_script(truncated=3)
    extractor = make_extractor([Person], script)  # window 600: nothing to split
    result = await extractor.ainvoke(PERSON_TEXT)
    assert len(script.calls("extract")) == 1
    assert any("clairic_truncated" in w for w in result.warnings)
    strict = make_extractor([Person], person_script(truncated=3), strict=True)
    with pytest.raises(StateTruncatedError):
        await strict.ainvoke(PERSON_TEXT)


async def test_a_truncated_reask_is_chunked_too() -> None:
    class Aged(BaseModel):
        name: str
        age: Optional[int] = None

    text = "Ada Lovelace is 36. " + filler(2000, "pad")
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace"),
        "age": ca("number", 36, confidence=0.7),
    }
    script.reask = {"age": ca("number", 36, confidence=0.99)}
    script.limit = lambda state, n: 3000 if len(state) > 3500 else None
    extractor = make_extractor(
        [Aged], script, verify=VerifyConfig(window_chars=50, attribution=False)
    )
    result = await extractor.ainvoke(text)
    record = result.records[0]
    assert record.value == Aged(name="Ada Lovelace", age=36)
    assert record.fields["age"].reask is True
    assert record.fields["age"].confidence == 0.99
    assert len(script.calls("reask")) > 1  # the full re-ask was cut, then chunks
    assert any("clairic_chunked" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class Adult(BaseModel):
    name: str
    age: int = Field(ge=18)


def adult_script(age: int) -> Script:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada", find="Ada"),
        "age": ca("number", age),
    }
    return script


async def test_validation_failure_is_an_invalid_record() -> None:
    script = adult_script(12)
    result = await make_extractor([Adult], script).ainvoke("Ada is 12.")
    (record,) = result.records
    assert record.valid is False
    assert record.value is None
    assert record.kind == "insert"
    assert len(record.errors) == 1 and record.errors[0].startswith("age:")
    assert record.after == {"name": "Ada", "age": 12}
    assert record.fields["age"].value == 12
    assert result.responses == ()
    assert result.response_metadata == []
    assert result.skipped == ()


async def test_strict_validation_failure_raises_with_the_record_and_calls() -> None:
    script = adult_script(12)
    extractor = make_extractor([Adult], script, strict=True)
    with pytest.raises(ExtractionValidationError) as info:
        await extractor.ainvoke("Ada is 12.")
    error = info.value
    assert error.errors and error.errors[0].startswith("age:")
    assert isinstance(error.record, ResultRecord)
    assert error.record.valid is False
    assert next(c.kind for c in error.calls) == "presence"
    assert len(error.calls) == len(script.transport.calls)


async def test_missing_required_value_is_an_invalid_record() -> None:
    script = Script()
    script.clairic = {"name": ca("text", None, p_null=0.99), "age": ca("number", 30)}
    result = await make_extractor([Person], script).ainvoke("Someone is 30.")
    (record,) = result.records
    assert record.valid is False
    assert record.errors[0].startswith("name:")


# --------------------------------------------------------------------------- #
# Usage, cost, call log
# --------------------------------------------------------------------------- #


async def test_usage_cost_and_call_records() -> None:
    script = person_script()
    result = await make_extractor([Person], script).ainvoke(PERSON_TEXT)
    assert result.usage.requests == len(script.transport.calls) == 3
    assert result.usage.input_tokens == 300
    assert result.usage.by_kind["extract"].input_tokens == 100
    assert result.usage.by_model["@cf/cloudflare/clairic"].requests == 1
    assert result.usage.by_model["@cf/cloudflare/clef"].requests == 2
    # Clairic is priced ($0.10 per million input tokens), Clef is not
    assert result.cost.usd == pytest.approx(100 * 0.10 / 1_000_000)
    assert result.cost.complete is False
    assert result.cost.unpriced_models == ("@cf/cloudflare/clef",)
    assert len(result.calls) == 3
    for call in result.calls:
        assert call.error is None
        assert call.status_code == 200
        assert call.duration_ms >= 0
        assert call.schema == "Person"
        assert call.entity == "e0"
        assert call.usage["input_tokens"] == 100
    assert sorted(c.kind for c in result.calls) == [
        "attribution",
        "extract",
        "presence",
    ]


async def test_cost_is_complete_once_clef_is_priced() -> None:
    script = person_script()
    extractor = make_extractor(
        [Person], script, prices=Prices({"clairic": 0.10, "clef": 0.30})
    )
    result = await extractor.ainvoke(PERSON_TEXT)
    assert result.cost.complete is True
    assert result.cost.usd == pytest.approx((100 * 0.10 + 200 * 0.30) / 1_000_000)


async def test_call_records_keep_the_retries_the_transport_reports() -> None:
    script = person_script()

    class Retrying:
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> TransportResponse:
            response = await script.transport.run(model, payload, meta=meta)
            return dataclasses.replace(response, retries=2, duration_ms=12.5)

    extractor = create_extractor([Person], transport=Retrying())
    result = await extractor.ainvoke(PERSON_TEXT)
    assert [c.retries for c in result.calls] == [2, 2, 2]
    assert [c.duration_ms for c in result.calls] == [12.5, 12.5, 12.5]


async def test_transport_errors_propagate() -> None:
    class Broken:
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> TransportResponse:
            raise ServerError("down", status_code=503)

    extractor = create_extractor([Person], transport=Broken())
    with pytest.raises(ServerError):
        await extractor.ainvoke(PERSON_TEXT)


async def test_default_transport_is_created_lazily_and_needs_credentials_only_then(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    extractor = create_extractor([Person])  # no credentials, no network: fine
    with pytest.raises(AuthError):
        await extractor.ainvoke(PERSON_TEXT)
    await extractor.aclose()


# --------------------------------------------------------------------------- #
# What is not built yet
# --------------------------------------------------------------------------- #


async def test_existing_is_not_supported_before_m5() -> None:
    extractor = make_extractor([Person], person_script())
    with pytest.raises(NotImplementedError, match="M5"):
        await extractor.ainvoke(PERSON_TEXT, existing={"Person": {"name": "Ada"}})
    with pytest.raises(NotImplementedError, match="M5"):
        await extractor.ainvoke({"text": PERSON_TEXT, "existing": []})
    with pytest.raises(NotImplementedError, match="M5"):
        extractor.invoke(PERSON_TEXT, existing=[])


async def test_many_is_not_supported_before_m7_and_fails_at_invoke_not_creation() -> (
    None
):
    script = person_script()
    extractor = make_extractor([Many(Person, identity="name")], script)
    with pytest.raises(NotImplementedError, match="M7"):
        await extractor.ainvoke(PERSON_TEXT)
    assert script.transport.calls == []


class Pet(BaseModel):
    name: str


class WithDeferred(BaseModel):
    name: str
    pets: list[Pet] = []
    scores: dict[str, int] = {}


async def test_collections_and_mappings_are_left_to_their_defaults_with_a_warning() -> (
    None
):
    script = Script()
    script.clairic = {"name": ca("text", "Ada", find="Ada")}
    result = await make_extractor([WithDeferred], script).ainvoke("Ada has a dog.")
    (record,) = result.records
    assert record.valid
    assert record.value == WithDeferred(name="Ada")
    assert set(record.fields) == {"name"}
    joined = "\n".join(result.warnings)
    for field_name in ("pets", "scores"):
        assert f"WithDeferred.{field_name}:" in joined
    assert "M9" in joined
    assert sorted(script.calls("extract")[0][1]["fields"]) == ["name"]


class NeedsDeferred(BaseModel):
    name: str
    pets: list[Pet]


async def test_a_required_collection_makes_the_record_invalid() -> None:
    script = Script()
    script.clairic = {"name": ca("text", "Ada", find="Ada")}
    result = await make_extractor([NeedsDeferred], script).ainvoke("Ada has a dog.")
    (record,) = result.records
    assert record.valid is False
    assert record.errors[0].startswith("pets:")
    assert any("NeedsDeferred.pets" in w and "M9" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# create_extractor
# --------------------------------------------------------------------------- #


def test_at_least_one_enable_flag_is_required() -> None:
    with pytest.raises(ValueError, match="At least one of enable_inserts"):
        create_extractor(
            [Person], enable_inserts=False, enable_updates=False, enable_deletes=False
        )
    create_extractor(
        [Person], enable_inserts=False, enable_updates=False, enable_deletes=True
    )
    create_extractor([Person], enable_inserts=True, enable_updates=False)


async def test_without_existing_records_it_inserts_whatever_enable_inserts_says() -> (
    None
):
    script = person_script()
    extractor = make_extractor(
        [Person], script, enable_inserts=False, enable_updates=True
    )
    assert extractor.enable_inserts is False
    result = await extractor.ainvoke(PERSON_TEXT)
    assert result.records[0].kind == "insert"


def test_flags_and_options_are_stored() -> None:
    extractor = create_extractor(
        [Person],
        enable_inserts=True,
        enable_deletes=True,
        enable_item_removals=True,
        existing_schema_policy="ignore",
        clairic_model="@cf/cloudflare/clairic",
        strict=True,
    )
    assert extractor.enable_inserts and extractor.enable_deletes
    assert extractor.enable_item_removals
    assert extractor.existing_schema_policy == "ignore"
    assert extractor.strict is True
    with pytest.raises(ValueError, match="existing_schema_policy"):
        create_extractor([Person], existing_schema_policy="sometimes")  # type: ignore[arg-type]


def test_schemas_are_validated_when_the_extractor_is_created() -> None:
    with pytest.raises(ValueError, match="at least one schema"):
        create_extractor([])
    with pytest.raises(ValueError, match="unique"):
        create_extractor([Person, Person])

    class Broken(BaseModel):
        pick: Literal[tuple(f"v{i}" for i in range(300))]  # type: ignore[valid-type]

    with pytest.raises(ValueError, match="255"):
        create_extractor([Broken])


def test_plan_for_uses_the_extractors_gates() -> None:
    from trustcall_system_one import GateConfig

    extractor = create_extractor(
        [Person], gates=GateConfig(required_null_threshold=0.9)
    )
    plan = extractor.plan_for(Person)
    assert plan.spec(("name",)).null_threshold == 0.9
    assert extractor.plan_for(One(Person)) is plan
    assert extractor.plan_for(Many(Person)) is plan


def test_extractor_is_frozen_and_with_options_makes_a_copy() -> None:
    transport = person_script().transport
    extractor = create_extractor([Person], transport=transport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        extractor.strict = True  # type: ignore[misc]
    strict = extractor.with_options(strict=True, enable_inserts=True)
    assert strict.strict and strict.enable_inserts
    assert not extractor.strict
    assert strict.core is extractor.core  # shares the transport and the bridge
    assert strict.transport is transport
    other = extractor.with_options(transport=person_script().transport)
    assert other.core is not extractor.core
    with pytest.raises(TypeError, match="bogus"):
        extractor.with_options(bogus=1)
    with pytest.raises(ValueError, match="At least one"):
        extractor.with_options(enable_updates=False)


def test_public_names_are_exported() -> None:
    import trustcall_system_one as package

    for name in (
        "create_extractor",
        "Extractor",
        "ExtractionInput",
        "Many",
        "One",
        "VerifyConfig",
        "PresenceConfig",
        "PassesConfig",
        "ExtractionResult",
        "ResultRecord",
        "FieldAnswer",
        "EntityInfo",
        "Mention",
        "PatchOp",
        "RejectedOp",
        "SkippedEntity",
    ):
        assert name in package.__all__
        assert hasattr(package, name)


# --------------------------------------------------------------------------- #
# The sync bridge
# --------------------------------------------------------------------------- #


def bridge_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "trustcall-system-one-loop"]


def comparable(result: ExtractionResult) -> Any:
    record = result.records[0]
    return (
        record.value,
        record.fields,
        result.responses,
        sorted(c.kind for c in result.calls),
        result.usage.requests,
    )


def test_invoke_works_from_plain_code() -> None:
    script = person_script()
    extractor = make_extractor([Person], script)
    try:
        result = extractor.invoke(PERSON_TEXT)
        assert result.records[0].value == Person(name="Ada Lovelace", age=36)
        assert len(bridge_threads()) >= 1
        thread = extractor.core._thread
        assert thread is not None and thread.daemon
        assert extractor.invoke(PERSON_TEXT).records[0].valid
        assert extractor.core._thread is thread  # one persistent thread
    finally:
        extractor.close()


async def test_invoke_works_from_inside_a_running_loop() -> None:
    script = person_script()
    extractor = make_extractor([Person], script)
    try:
        result = extractor.invoke(PERSON_TEXT)  # blocks this loop; no nesting
        assert result.records[0].valid
        assert asyncio.get_running_loop() is not extractor.core._loop
    finally:
        extractor.close()


def test_invoke_works_from_inside_asyncio_run() -> None:
    extractor = make_extractor([Person], person_script())

    async def main() -> ExtractionResult:
        return extractor.invoke(PERSON_TEXT)

    try:
        assert asyncio.run(main()).records[0].valid
    finally:
        extractor.close()


async def test_invoke_and_ainvoke_agree_on_the_same_handler() -> None:
    extractor = make_extractor([Person], person_script())
    try:
        sync_result = extractor.invoke(PERSON_TEXT)
        async_result = await extractor.ainvoke(PERSON_TEXT)
        assert comparable(sync_result) == comparable(async_result)
        schema_result = extractor.invoke(PERSON_TEXT, schema=Person)
        assert schema_result.records[0].value == sync_result.records[0].value
    finally:
        extractor.close()


def test_invoke_propagates_errors_and_stays_usable() -> None:
    script = adult_script(12)
    extractor = make_extractor([Adult], script, strict=True)
    try:
        with pytest.raises(ExtractionValidationError):
            extractor.invoke("Ada is 12.")
        script.clairic["age"] = ca("number", 40)
        assert extractor.invoke("Ada is 40.").records[0].valid
    finally:
        extractor.close()


def test_invoke_from_the_bridge_thread_itself_is_refused() -> None:
    extractor = make_extractor([Person], person_script())
    try:

        async def inner() -> None:
            extractor.invoke(PERSON_TEXT)

        with pytest.raises(RuntimeError, match="own event loop"):
            extractor.core.run_sync(inner())
    finally:
        extractor.close()


class RecordingTransport:
    """Wraps a script's transport and notes how it was closed."""

    def __init__(self, inner: FakeTransport) -> None:
        self.inner = inner
        self.aclosed_on: asyncio.AbstractEventLoop | None = None
        self.closed = False

    async def run(
        self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
    ) -> TransportResponse:
        return await self.inner.run(model, payload, meta=meta)

    async def aclose(self) -> None:
        self.aclosed_on = asyncio.get_running_loop()

    def close(self) -> None:
        self.closed = True


def test_close_stops_the_thread_and_closes_an_owned_transport_on_its_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = person_script()
    created: list[RecordingTransport] = []

    def factory(*, concurrency: Any = None) -> RecordingTransport:
        created.append(RecordingTransport(script.transport))
        return created[-1]

    monkeypatch.setattr("trustcall_system_one._extractor.CloudflareTransport", factory)
    extractor = create_extractor([Person])
    assert created == []  # lazy
    extractor.invoke(PERSON_TEXT)
    (owned,) = created
    loop = extractor.core._loop
    thread = extractor.core._thread
    assert loop is not None and thread is not None and thread.is_alive()
    extractor.close()
    assert owned.aclosed_on is loop  # closed on the loop that used it
    assert not thread.is_alive()
    assert loop.is_closed()
    assert extractor.core._thread is None
    extractor.close()  # idempotent
    # usable again afterwards, with a fresh transport and thread
    assert extractor.invoke(PERSON_TEXT).records[0].valid
    assert len(created) == 2
    assert extractor.core._thread is not thread
    extractor.close()


def test_a_given_transport_is_not_closed() -> None:
    script = person_script()
    given = RecordingTransport(script.transport)
    extractor = create_extractor([Person], transport=given)
    extractor.invoke(PERSON_TEXT)
    thread = extractor.core._thread
    extractor.close()
    assert thread is not None and not thread.is_alive()
    assert given.aclosed_on is None and given.closed is False


def test_context_manager_closes() -> None:
    script = person_script()
    with make_extractor([Person], script) as extractor:
        extractor.invoke(PERSON_TEXT)
        thread = extractor.core._thread
        assert thread is not None and thread.is_alive()
    assert thread is not None and not thread.is_alive()


async def test_async_context_manager_and_aclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = person_script()
    created: list[RecordingTransport] = []

    def factory(*, concurrency: Any = None) -> RecordingTransport:
        created.append(RecordingTransport(script.transport))
        return created[-1]

    monkeypatch.setattr("trustcall_system_one._extractor.CloudflareTransport", factory)
    async with create_extractor([Person]) as extractor:
        await extractor.ainvoke(PERSON_TEXT)
        assert extractor.core._thread is None  # ainvoke needs no bridge
    (owned,) = created
    assert owned.aclosed_on is asyncio.get_running_loop()

    # aclose also stops a bridge that was started by invoke
    extractor = create_extractor([Person])
    extractor.invoke(PERSON_TEXT)
    thread = extractor.core._thread
    assert thread is not None
    await extractor.aclose()
    assert not thread.is_alive()
    assert created[-1].aclosed_on is not None
    await extractor.aclose()  # idempotent


def test_an_atexit_hook_closes_leftovers(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[Any] = []
    unregistered: list[Any] = []
    monkeypatch.setattr(
        "trustcall_system_one._extractor.atexit.register", registered.append
    )
    monkeypatch.setattr(
        "trustcall_system_one._extractor.atexit.unregister", unregistered.append
    )
    extractor = make_extractor([Person], person_script())
    extractor.invoke(PERSON_TEXT)
    (hook,) = registered
    thread = extractor.core._thread
    assert thread is not None and thread.is_alive()
    hook()  # what the interpreter would call at exit
    assert not thread.is_alive()
    assert unregistered == [hook]
    hook()  # harmless twice


def test_the_hook_does_not_keep_the_extractor_alive() -> None:
    import gc
    import weakref

    extractor = make_extractor([Person], person_script())
    extractor.invoke(PERSON_TEXT)
    core = weakref.ref(extractor.core)
    thread = extractor.core._thread
    extractor.close()
    del extractor
    gc.collect()
    assert core() is None
    assert thread is not None and not thread.is_alive()


async def test_batch_runs_the_inputs_concurrently() -> None:
    script = person_script()
    in_flight = 0
    peak = 0

    async def presence(model: str, payload: dict[str, Any], meta: Any) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        for _ in range(500):
            if in_flight >= 3:
                break
            await asyncio.sleep(0.001)
        in_flight -= 1
        return clef_response({"presence": noul(0.9)})

    script.transport = FakeTransport(
        route(
            presence=presence,
            extract=script._extract,
            clef_fields=script._clef_fields,
            attribution=script._attribution,
        )
    )
    extractor = make_extractor([Person], script)
    results = await extractor.abatch([PERSON_TEXT, PERSON_TEXT, PERSON_TEXT])
    assert peak == 3
    assert len(results) == 3 and all(r.records[0].valid for r in results)


def test_sync_batch_goes_through_the_bridge() -> None:
    script = person_script()
    extractor = make_extractor([Person], script)
    try:
        results = extractor.batch([PERSON_TEXT, {"text": PERSON_TEXT}])
        assert [r.records[0].value for r in results] == [
            Person(name="Ada Lovelace", age=36)
        ] * 2
        assert len(script.calls("presence")) == 2
    finally:
        extractor.close()


async def test_batch_failure_cancels_the_rest() -> None:
    script = adult_script(12)
    extractor = make_extractor([Adult], script, strict=True)
    with pytest.raises(ExtractionValidationError):
        await extractor.abatch(["Ada is 12.", "Ada is 12 too."])


# --------------------------------------------------------------------------- #
# Spans and the returned state_text
# --------------------------------------------------------------------------- #


async def test_spans_are_re_anchored_on_a_changed_state_text() -> None:
    script = person_script()
    script.state_text = lambda state: "PREFIX LINE\n" + state.replace("Answer", "Reply")
    result = await make_extractor([Person], script).ainvoke(PERSON_TEXT)
    span = result.records[0].fields["name"].span
    assert span is not None
    assert PERSON_TEXT[span[0] : span[1]] == "Ada Lovelace"
    assert result.warnings == ()


async def test_spans_are_unknown_with_a_warning_when_the_input_cannot_be_found() -> (
    None
):
    script = Script()
    script.clairic = {"name": ca("text", "Ada", span=[1, 4]), "age": ca("number", 36)}
    script.state_text = lambda state: "something unrelated"
    result = await make_extractor([Person], script).ainvoke("Ada is 36.")
    record = result.records[0]
    assert record.valid
    assert record.fields["name"].span is None
    assert any("state_text_mismatch" in w for w in result.warnings)
    assert "span_unknown" in record.fields["name"].warnings


async def test_a_span_outside_the_input_block_is_dropped_with_a_note() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada", span=[2, 6]),  # inside <focus>
        "age": ca("number", 36),
    }
    result = await make_extractor([Person], script).ainvoke("Ada is 36.")
    answer = result.records[0].fields["name"]
    assert answer.span is None
    assert answer.value == "Ada"
    assert "span_outside_input" in answer.warnings


# --------------------------------------------------------------------------- #
# Many Clef questions
# --------------------------------------------------------------------------- #


async def test_more_than_64_clef_questions_are_split_and_merged() -> None:
    from pydantic import create_model

    fields: dict[str, Any] = {f"f{i}": (bool, False) for i in range(70)}
    Wide = create_model("Wide", **fields)
    script = Script()
    script.clef = {f"f{i}": {"type": "noul", "noul": 0.9} for i in range(70)}
    result = await make_extractor([Wide], script).ainvoke("Everything is true.")
    clef_calls = script.calls("clef_fields")
    assert sorted(len(c[1]["questions"]) for c in clef_calls) == [6, 64]
    (record,) = result.records
    assert record.valid
    assert all(getattr(record.value, f"f{i}") is True for i in range(70))
    assert len(record.fields) == 70
    assert [c.kind for c in result.calls].count("clef_fields") == 2
    assert "extract" not in kinds(script.transport)  # nothing for Clairic to do


# --------------------------------------------------------------------------- #
# Shutdown of the sync bridge
# --------------------------------------------------------------------------- #


class Hanging:
    """A transport whose calls never finish until cancelled."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def run(
        self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
    ) -> TransportResponse:
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


def test_close_during_a_sync_invoke_returns_promptly_and_frees_the_caller() -> None:
    import time

    hanging = Hanging()
    extractor = create_extractor([Person], transport=hanging)
    outcome: dict[str, Any] = {}

    def worker() -> None:
        started = time.monotonic()
        try:
            extractor.invoke(PERSON_TEXT)
            outcome["result"] = "returned"
        except BaseException as exc:
            outcome["result"] = exc
        outcome["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert hanging.started.wait(5)
    bridge = extractor.core._thread
    started = time.monotonic()
    extractor.close()
    assert time.monotonic() - started < 3
    thread.join(5)
    assert not thread.is_alive()  # the invoking thread did not hang
    assert isinstance(outcome["result"], TrustcallSystemOneError)
    assert "closed" in str(outcome["result"])
    assert outcome["elapsed"] < 5
    assert hanging.cancelled.wait(5)  # the call on the loop was cancelled too
    assert bridge is not None and not bridge.is_alive()


def test_close_during_a_sync_batch_frees_every_caller() -> None:
    hanging = Hanging()
    extractor = create_extractor([Person], transport=hanging)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            extractor.batch([PERSON_TEXT, PERSON_TEXT])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert hanging.started.wait(5)
    extractor.close()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(errors) == 2
    assert all(isinstance(e, TrustcallSystemOneError) for e in errors)


def test_the_extractor_restarts_lazily_after_a_close_that_cancelled_a_call() -> None:
    script = person_script()
    hanging = Hanging()

    class Switch:
        use_hanging = True

        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> TransportResponse:
            if self.use_hanging:
                return await hanging.run(model, payload, meta=meta)
            return await script.transport.run(model, payload, meta=meta)

    switch = Switch()
    extractor = create_extractor([Person], transport=switch)
    thread = threading.Thread(
        target=lambda: _swallow(extractor.invoke, PERSON_TEXT), daemon=True
    )
    thread.start()
    assert hanging.started.wait(5)
    first_bridge = extractor.core._thread
    extractor.close()
    thread.join(5)
    switch.use_hanging = False
    assert extractor.invoke(PERSON_TEXT).records[0].valid
    assert extractor.core._thread is not first_bridge
    extractor.close()


def _swallow(function: Any, *args: Any) -> None:
    with contextlib.suppress(BaseException):
        function(*args)


def test_close_from_the_bridge_thread_closes_the_owned_transport_on_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = person_script()
    created: list[RecordingTransport] = []

    def factory(*, concurrency: Any = None) -> RecordingTransport:
        created.append(RecordingTransport(script.transport))
        return created[-1]

    monkeypatch.setattr("trustcall_system_one._extractor.CloudflareTransport", factory)
    extractor = create_extractor([Person])
    extractor.invoke(PERSON_TEXT)
    (owned,) = created
    loop = extractor.core._loop
    thread = extractor.core._thread
    assert thread is not None

    async def close_myself() -> None:
        extractor.close()  # called on the bridge thread

    with pytest.raises(TrustcallSystemOneError, match="closed"):
        extractor.core.run_sync(close_myself())
    thread.join(5)
    assert not thread.is_alive()
    assert owned.aclosed_on is loop  # closed on the loop, not leaked
    assert extractor.invoke(PERSON_TEXT).records[0].valid  # and it restarts
    assert len(created) == 2
    extractor.close()


def test_the_loop_thread_lets_cancelled_tasks_finish_before_it_closes() -> None:
    finished = threading.Event()

    class Slow:
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> TransportResponse:
            try:
                await asyncio.sleep(60)
            finally:
                await asyncio.sleep(0.05)  # cleanup that needs the loop to run
                finished.set()
            raise AssertionError("unreachable")

    extractor = create_extractor([Person], transport=Slow())
    thread = threading.Thread(
        target=lambda: _swallow(extractor.invoke, PERSON_TEXT), daemon=True
    )
    thread.start()
    for _ in range(100):
        if extractor.core._loop is not None and extractor.core._pending:
            break
        threading.Event().wait(0.02)
    extractor.close()
    thread.join(5)
    assert finished.wait(5)


# --------------------------------------------------------------------------- #
# Failed invokes keep their accounting
# --------------------------------------------------------------------------- #


class FailsOn:
    """Delegates to a script, but raises on the first call of one kind."""

    def __init__(self, script: Script, kind: str, error: Exception) -> None:
        self.script, self.kind, self.error = script, kind, error

    async def run(
        self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
    ) -> TransportResponse:
        if meta.kind == self.kind:
            raise self.error
        return await self.script.transport.run(model, payload, meta=meta)


async def test_a_library_error_carries_the_calls_usage_and_cost() -> None:
    error = ServerError("down", status_code=503)
    error.retries = 4
    extractor = create_extractor(
        [Person], transport=FailsOn(person_script(), "extract", error)
    )
    with pytest.raises(ServerError) as info:
        await extractor.ainvoke(PERSON_TEXT)
    raised = info.value
    assert raised is error
    assert [c.kind for c in raised.calls].count("presence") == 1
    failed = [c for c in raised.calls if c.error is not None]
    assert [(c.kind, c.retries, c.status_code) for c in failed] == [("extract", 4, 503)]
    assert raised.usage is not None and raised.usage.requests == 1  # the presence
    assert raised.cost is not None and raised.cost.complete is False
    assert raised.cost.unpriced_models == ("@cf/cloudflare/clef",)


async def test_errors_outside_a_call_have_empty_defaults() -> None:
    plain = TrustcallSystemOneError("plain")
    assert plain.calls == () and plain.usage is None and plain.cost is None


async def test_a_strict_validation_error_has_the_full_call_log() -> None:
    extractor = make_extractor([Adult], adult_script(12), strict=True)
    with pytest.raises(ExtractionValidationError) as info:
        await extractor.ainvoke("Ada is 12.")
    assert len(info.value.calls) == 3
    assert info.value.usage is not None and info.value.usage.requests == 3
    assert info.value.cost is not None


async def test_a_foreign_error_is_reraised_unchanged_with_a_note() -> None:
    import sys

    error = TypeError("not a library error")
    extractor = create_extractor(
        [Person], transport=FailsOn(person_script(), "extract", error)
    )
    with pytest.raises(TypeError) as info:
        await extractor.ainvoke(PERSON_TEXT)
    assert info.value is error
    assert not hasattr(info.value, "calls")
    if sys.version_info >= (3, 11):
        (note,) = info.value.__notes__
        assert "2 call(s)" in note and "presence" in note
    else:
        assert not hasattr(info.value, "__notes__")


# --------------------------------------------------------------------------- #
# Batch with return_exceptions
# --------------------------------------------------------------------------- #


async def test_abatch_return_exceptions_keeps_the_others() -> None:
    script = adult_script(12)
    ages = iter([12, 40])
    original = script._extract

    def extract(model: str, payload: dict[str, Any], meta: Any) -> Any:
        script.clairic["age"] = ca("number", next(ages))
        return original(model, payload, meta)

    script.transport = FakeTransport(
        route(
            presence=script._presence,
            extract=extract,
            attribution=script._attribution,
        )
    )
    extractor = make_extractor([Adult], script, strict=True)
    outcomes = await extractor.abatch(
        ["Ada is 12.", "Ada is 40."], return_exceptions=True
    )
    assert isinstance(outcomes[0], ExtractionValidationError)
    assert isinstance(outcomes[1], ExtractionResult)
    assert outcomes[1].records[0].valid
    assert outcomes[0].calls  # the failed input kept its call log


def test_batch_return_exceptions_through_the_bridge() -> None:
    extractor = make_extractor([Adult], adult_script(12), strict=True)
    try:
        outcomes = extractor.batch(
            ["Ada is 12.", "Ada is 12 too."], return_exceptions=True
        )
        assert all(isinstance(o, ExtractionValidationError) for o in outcomes)
        with pytest.raises(ExtractionValidationError):
            extractor.batch(["Ada is 12."])
    finally:
        extractor.close()
