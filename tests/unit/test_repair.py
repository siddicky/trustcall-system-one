"""The validation repair loop: one narrow re-ask per failing field (``_repair``)."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import datetime as dt
import json
from typing import Any, Literal, Optional, TypeVar, Union

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from tests.unit.extract_helpers import Script, ca, kinds, make_extractor, pick
from trustcall_system_one import (
    ExtractionResult,
    ExtractionValidationError,
    One,
    RepairConfig,
    ResultRecord,
    compile_schema,
    s1_field,
)
from trustcall_system_one._repair import (
    Hit,
    Problem,
    classify,
    excerpt,
    locate,
    problems_of,
    signature,
)
from trustcall_system_one._wire import CLAIRIC
from trustcall_system_one.testing import noul

ON = RepairConfig()


def only(result: ExtractionResult) -> ResultRecord:
    (record,) = result.records
    return record


M = TypeVar("M", bound=BaseModel)


def value_of(record: ResultRecord, model: type[M]) -> M:
    """The record's validated value, as the model it must be."""
    assert record.valid and isinstance(record.value, model)
    return record.value


def repair_calls(world: Script) -> list[Any]:
    return world.calls("repair")


def request_for(world: Script, field: str) -> dict[str, Any]:
    """The payload of the repair call about one field."""
    (call,) = [c for c in repair_calls(world) if c[2].field == field]
    payload: dict[str, Any] = call[1]
    return payload


def instructions(world: Script, field: str) -> str:
    return str(request_for(world, field)["fields"][field]["instructions"])


async def run(
    schemas: list[Any],
    world: Script,
    text: str,
    *,
    repair: RepairConfig | None = ON,
    existing: Any = None,
    **kwargs: Any,
) -> ExtractionResult:
    extractor = make_extractor(schemas, world, repair=repair, **kwargs)
    return await extractor.ainvoke(text, existing=existing)


# --------------------------------------------------------------------------- #
# A schema with several kinds of constraint
# --------------------------------------------------------------------------- #


class Applicant(BaseModel):
    name: str
    age: int = Field(ge=18, le=130, description="age in years")
    postcode: str = Field(pattern=r"^\d{5}$", description="five-digit postcode")


APPLICANT = "Katarina Vogel, Friedrichstrasse 12, Berlin 10115, is 34 years old."


def applicant_world(age: Any = None, postcode: Any = None) -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Katarina Vogel", find="Katarina Vogel"),
        "age": age or ca("number", 34),
        "postcode": postcode or ca("text", "10115", find="10115"),
    }
    return world


# --------------------------------------------------------------------------- #
# From an error location to the field
# --------------------------------------------------------------------------- #


class City(BaseModel):
    name: str
    zip_code: str = Field(alias="zip", pattern=r"^\d{5}$")


class Place(BaseModel):
    label: str
    city: City


class Region(BaseModel):
    start: int
    end: int

    @model_validator(mode="after")
    def _ordered(self) -> "Region":
        if self.end < self.start:
            raise ValueError("end before start")
        return self


class Trip(BaseModel):
    title: str
    tags: list[str] = Field(default_factory=list, max_length=2)
    place: Place
    region: Optional[Region] = None


class Cat(BaseModel):
    kind: Literal["cat"]
    lives: int = Field(ge=1)


class Dog(BaseModel):
    kind: Literal["dog"]
    tricks: int = Field(ge=0)


class Owner(BaseModel):
    name: str
    pet: Union[Cat, Dog]


def problems_for(model: type[BaseModel], data: dict[str, Any]) -> list[Problem]:
    with pytest.raises(ValidationError) as info:
        model.model_validate(data)
    return problems_of(info.value)


def paths(hits: list[Hit]) -> list[tuple[str, ...]]:
    return [hit.entry.path for hit in hits]


def test_a_nested_error_implicates_the_leaf_at_the_end_of_its_path() -> None:
    plan = compile_schema(Trip)
    data = {
        "title": "t",
        "place": {"label": "l", "city": {"name": "n", "zip": "x"}},
    }
    hits = locate(plan, problems_for(Trip, data), {})
    assert paths(hits) == [("place", "city", "zip")]  # the key is the alias
    (hit,) = hits
    assert hit.direct and hit.entry.spec.path == ("place", "city", "zip_code")
    assert hit.entry.spec.wire_name  # the plan's own spec, with its wire name
    assert classify(hit, hit.entry.spec)[0] == "format"


def test_an_alias_is_found_by_its_validation_key() -> None:
    plan = compile_schema(City)
    (hit,) = locate(plan, problems_for(City, {"name": "n", "zip": "no"}), {})
    assert hit.entry.spec.path == ("zip_code",)
    assert hit.problems[0].loc == ("zip",)


def test_a_missing_required_leaf_implicates_that_leaf() -> None:
    plan = compile_schema(City)
    problems = problems_for(City, {"zip": "12345"})
    assert [p.type for p in problems] == ["missing"]
    (hit,) = locate(plan, problems, {})
    assert hit.entry.path == ("name",)
    assert classify(hit, hit.entry.spec)[0] == "missing"


def test_an_error_below_a_list_field_implicates_the_list() -> None:
    plan = compile_schema(Trip)
    data = {
        "title": "t",
        "tags": ["a", "b", "c"],
        "place": {"label": "l", "city": {"name": "n", "zip": "12345"}},
    }
    problems = problems_for(Trip, data)
    assert [p.type for p in problems] == ["too_long"]
    (hit,) = locate(plan, problems, {})
    assert hit.entry.path == ("tags",)
    assert classify(hit, hit.entry.spec)[0] == "length"


def test_the_union_member_segment_is_dropped_and_unchosen_members_are_ignored() -> None:
    plan = compile_schema(Owner)
    problems = problems_for(Owner, {"name": "n", "pet": {"kind": "dog", "tricks": -1}})
    # pydantic's smart union reports every member, each named in the location
    assert ("pet", "Dog", "tricks") in [p.loc for p in problems]
    assert ("pet", "Cat", "lives") in [p.loc for p in problems]
    (hit,) = locate(plan, problems, {"/pet": "Dog"})
    assert hit.entry.path == ("pet", "tricks")  # the member segment is dropped
    assert hit.entry.spec.member_labels == ("Dog",)
    assert [p.loc for p in hit.problems] == [("pet", "Dog", "tricks")]
    # the errors about the member that was not chosen are not the document's
    cat = locate(plan, problems, {"/pet": "Cat"})
    assert paths(cat) == [("pet", "lives")]
    assert cat[0].entry.spec.member_labels == ("Cat",)


def test_a_model_level_error_implicates_every_field_of_that_model() -> None:
    plan = compile_schema(Trip)
    data = {
        "title": "t",
        "place": {"label": "l", "city": {"name": "n", "zip": "12345"}},
        "region": {"start": 5, "end": 1},
    }
    problems = problems_for(Trip, data)
    assert [(p.loc, p.type) for p in problems] == [(("region",), "value_error")]
    hits = locate(plan, problems, {})
    assert paths(hits) == [("region", "start"), ("region", "end")]
    # a field named only through its model is not blamed on its own: class "other"
    assert not any(hit.direct for hit in hits)
    assert [classify(h, h.entry.spec)[0] for h in hits] == ["other", "other"]


def test_an_error_about_the_whole_record_implicates_every_field() -> None:
    plan = compile_schema(Region)
    problems = problems_for(Region, {"start": 5, "end": 1})
    assert problems[0].loc == ()
    assert paths(locate(plan, problems, {})) == [("start",), ("end",)]


def test_errors_of_things_that_are_not_asked_implicate_nothing() -> None:
    plan = compile_schema(City)
    unknown = [Problem(("bogus",), "extra_forbidden", "Extra inputs are not permitted")]
    assert locate(plan, unknown, {}) == []


def test_the_error_signature_is_the_set_of_pointer_and_type() -> None:
    plan = compile_schema(Owner)
    problems = problems_for(Owner, {"name": "n", "pet": {"kind": "dog", "tricks": -1}})
    assert signature(plan, problems, {"/pet": "Dog"}) == frozenset(
        {("/pet/tricks", "greater_than_equal")}
    )
    assert signature(plan, problems, {"/pet": "Cat"}) == frozenset(
        {("/pet/kind", "literal_error"), ("/pet/lives", "missing")}
    )


def test_the_schema_excerpt_is_ref_free() -> None:
    text = excerpt(Trip, ("place",))
    schema = json.loads(text)
    assert "$ref" not in text and "$defs" not in text
    assert set(schema["properties"]) == {"label", "city"}
    assert set(schema["properties"]["city"]["properties"]) == {"name", "zip"}
    # below a union field the excerpt follows the member that has the key
    assert json.loads(excerpt(Owner, ("pet", "tricks")))["minimum"] == 0
    # a list index stays with the list's items
    assert json.loads(excerpt(Trip, ("tags", 1)))["type"] == "string"
    # past what the schema knows, the deepest known part is shown
    assert json.loads(excerpt(Trip, ("title", "nope")))["type"] == "string"


# --------------------------------------------------------------------------- #
# The request: one narrow, input-only re-ask per failing field
# --------------------------------------------------------------------------- #


async def test_a_bound_is_stated_in_prose_and_the_same_type_is_asked() -> None:
    world = applicant_world(age=ca("number", 12))
    world.repair = {"age": ca("number", 34)}
    result = await run([Applicant], world, APPLICANT)
    record = only(result)
    assert value_of(record, Applicant).age == 34
    assert kinds(world.transport) == [
        "presence",
        "extract",
        "attribution",
        "repair",  # ONE re-ask, for the one field that failed
        "attribution",  # the repaired value is verified like any answer
    ]
    payload = request_for(world, "age")
    assert list(payload["fields"]) == ["age"]  # a single-field request
    field = payload["fields"]["age"]
    assert (field["type"], field["integer"]) == ("number", True)
    assert field["instructions"].startswith("the Applicant's age in years")
    assert "Must satisfy: at least 18." in field["instructions"]
    assert (
        "The previous answer 12 was rejected because: Input should be greater "
        "than or equal to 18." in field["instructions"]
    )
    # nothing but the failing field was asked again
    assert [c[2].field for c in repair_calls(world)] == ["age"]
    assert result.attempts == 2


async def test_the_state_has_the_focus_the_input_and_three_blocks_and_no_record() -> (
    None
):
    world = applicant_world(age=ca("number", 12))
    world.repair = {"age": ca("number", 34)}
    await run([Applicant], world, APPLICANT)
    state: str = request_for(world, "age")["state"]
    assert state.startswith('<focus entity="the Applicant">')
    assert f"<input>\n{APPLICANT}\n</input>" in state
    order = [
        state.index(tag)
        for tag in (
            "<input>",
            "</input>",
            "<previous_attempt>",
            "</previous_attempt>",
            "<validation_errors>",
            "</validation_errors>",
            "<expected_schema>",
            "</expected_schema>",
        )
    ]
    assert order == sorted(order)
    previous = state.split("<previous_attempt>\n")[1].split("\n</previous_attempt>")[0]
    assert json.loads(previous) == 12  # just that field's rejected value
    errors = state.split("<validation_errors>\n")[1].split("\n</validation_errors>")[0]
    (error,) = json.loads(errors)
    assert error == {
        "field": "age",
        "message": "Input should be greater than or equal to 18",
        "type": "greater_than_equal",
        "input": 12,
    }
    schema = state.split("<expected_schema>\n")[1].split("\n</expected_schema>")[0]
    assert json.loads(schema)["minimum"] == 18
    # input only: an update's record is never in a repair request (see below)
    assert "<existing" not in state


async def test_a_parsing_error_asks_for_a_number() -> None:
    class Person(BaseModel):
        name: str
        age: int = Field(
            json_schema_extra=s1_field(clairic_type="text"), description="age"
        )

    world = Script()
    world.clairic = {
        "name": ca("text", "Ada", find="Ada"),
        "age": ca(
            "text", "34 years", find="34 years"
        ),  # text: pydantic cannot parse it
    }
    world.repair = {"age": ca("number", 34)}
    record = only(await run([Person], world, "Ada is 34 years old."))
    assert value_of(record, Person).age == 34
    field = request_for(world, "age")["fields"]["age"]
    assert (field["type"], field["integer"]) == ("number", True)
    assert "Must satisfy: a whole number." in field["instructions"]
    assert (
        "The previous answer '34 years' was rejected because" in field["instructions"]
    )
    assert "unable to parse string as an integer" in field["instructions"]


async def test_a_date_error_asks_for_a_date() -> None:
    class Event(BaseModel):
        title: str
        day: dt.date = Field(
            json_schema_extra=s1_field(clairic_type="text"), description="the day"
        )

    world = Script()
    world.clairic = {
        "title": ca("text", "Launch", find="Launch"),
        "day": ca("text", "3rd of March", find="3rd of March"),
    }
    world.repair = {"day": ca("date", "2024-03-03")}
    record = only(await run([Event], world, "Launch on the 3rd of March."))
    assert value_of(record, Event).day == dt.date(2024, 3, 3)
    field = request_for(world, "day")["fields"]["day"]
    assert field["type"] == "date"
    assert "Must satisfy: an ISO 8601 date (YYYY-MM-DD)." in field["instructions"]


class Account(BaseModel):
    owner: str
    status: Literal["active", "inactive"]

    @field_validator("status", mode="before")
    @classmethod
    def _shout(cls, value: Any) -> Any:
        return value.upper() if value == "inactive" else value  # no longer allowed


async def test_a_bad_literal_is_re_asked_as_a_clef_choice() -> None:
    world = Script()
    world.clairic = {"owner": ca("text", "Ada", find="Ada")}
    world.clef = {"status": pick("inactive", ["active", "inactive"])}
    world.repair = {"status": pick("active", ["active", "inactive"])}
    record = only(await run([Account], world, "Ada's account is closed."))
    assert value_of(record, Account).status == "active"
    (call,) = repair_calls(world)
    model, payload, meta = call
    assert model != CLAIRIC  # no Clairic for a choice
    assert (meta.kind, meta.field) == ("repair", "status")
    question = payload["questions"]["status"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"active", "inactive"}  # nothing else to pick
    assert "Must satisfy: one of: 'active', 'inactive'." in question["instructions"]
    assert (
        "The previous answer 'inactive' was rejected because"
        in question["instructions"]
    )
    state = payload["state"]
    assert state["input"] == "Ada's account is closed."
    assert state["previous_attempt"] == "inactive"
    assert state["validation_errors"][0]["type"] == "literal_error"
    assert state["expected_schema"]["enum"] == ["active", "inactive"]
    assert "record" not in state
    field = record.fields["status"]
    assert field.repaired and field.engine == "clef"


class Alert(BaseModel):
    text: str
    urgent: bool

    @field_validator("urgent", mode="before")
    @classmethod
    def _maybe(cls, value: Any) -> Any:
        return "maybe" if value is True else value


async def test_a_bool_error_is_re_asked_as_a_clef_noul() -> None:
    world = Script()
    world.clairic = {"text": ca("text", "server down", find="server down")}
    world.clef = {"urgent": noul(0.95)}
    world.repair = {"urgent": noul(0.03)}
    record = only(await run([Alert], world, "The server down alert is not urgent."))
    assert value_of(record, Alert).urgent is False
    (call,) = repair_calls(world)
    question = call[1]["questions"]["urgent"]
    assert question["type"] == "noul"
    assert "Must satisfy: true or false." in question["instructions"]
    assert call[1]["state"]["validation_errors"][0]["type"] == "bool_parsing"
    assert record.fields["urgent"].repaired


class Profile(BaseModel):
    nickname: str = Field(max_length=5, description="nickname")
    tags: list[str] = Field(min_length=2, description="tags")


PROFILE = "Bartholomew, known as Bart, is into chess and golf."


async def test_lengths_are_stated_in_prose_and_all_failures_are_one_attempt() -> None:
    world = Script()
    world.clairic = {
        "nickname": ca("text", "Bartholomew", find="Bartholomew"),
        "tags": ca("list", ["chess"], find_all=["chess"]),
    }
    world.repair = {
        "nickname": ca("text", "Bart", find="Bart"),
        "tags": ca("list", ["chess", "golf"], find_all=["chess", "golf"]),
    }
    result = await run([Profile], world, PROFILE)
    record = only(result)
    profile = value_of(record, Profile)
    assert (profile.nickname, profile.tags) == ("Bart", ["chess", "golf"])
    # two failing fields, in parallel: ONE attempt (the fan-out rule)
    assert sorted(c[2].field or "" for c in repair_calls(world)) == ["nickname", "tags"]
    assert result.attempts == 2
    assert "Must satisfy: at most 5 characters." in instructions(world, "nickname")
    assert "Must satisfy: at least 2 items." in instructions(world, "tags")
    assert request_for(world, "tags")["fields"]["tags"]["type"] == "list"
    assert {f.repaired for f in (record.fields["nickname"], record.fields["tags"])} == {
        True
    }


class Scores(BaseModel):
    first: int = Field(ge=0)
    second: int = Field(ge=0)
    third: int = Field(ge=0)


async def test_a_fan_out_of_three_failing_fields_is_one_attempt() -> None:
    world = Script()
    world.clairic = {
        "first": ca("number", -1),
        "second": ca("number", -2),
        "third": ca("number", -3),
    }
    world.repair = {
        "first": ca("number", 1),
        "second": ca("number", 2),
        "third": ca("number", 3),
    }
    result = await run([Scores], world, "The scores were 1, 2 and 3.")
    record = only(result)
    assert record.valid and record.value == Scores(first=1, second=2, third=3)
    assert len(repair_calls(world)) == 3
    assert result.attempts == 2  # not 4
    names = kinds(world.transport)
    first = names.index("repair")
    assert names[first : first + 3] == ["repair"] * 3  # one round, back to back
    # their answers were verified together: one attribution request for all three
    assert names[first + 3 :] == ["attribution"]


async def test_a_required_null_is_re_asked_with_a_strict_null_threshold() -> None:
    world = applicant_world(age=ca("number", None, p_null=0.99))
    world.repair = {"age": ca("number", 34, p_null=0.02)}
    result = await run([Applicant], world, APPLICANT)
    record = only(result)
    assert value_of(record, Applicant).age == 34
    field = request_for(world, "age")["fields"]["age"]
    assert field["null_threshold"] == 0.95
    assert "The input states this; quote it." in field["instructions"]
    assert "Must satisfy: a value that the input states." in field["instructions"]
    errors = json.loads(
        request_for(world, "age")["state"]
        .split("<validation_errors>\n")[1]
        .split("\n</validation_errors>")[0]
    )
    assert errors[0]["type"] == "int_type"  # a null gets a type error, not "missing"


async def test_a_repair_that_is_null_again_is_not_used_and_not_repeated() -> None:
    world = applicant_world(age=ca("number", None, p_null=0.99))
    world.repair = {"age": ca("number", None, p_null=0.99)}
    result = await run([Applicant], world, APPLICANT)
    record = only(result)
    assert not record.valid
    assert len(repair_calls(world)) == 1  # the same error again: no second round
    assert result.attempts == 2
    assert any("failed" in note for note in record.fields["age"].warnings)


class Booking(BaseModel):
    start: dt.date = Field(description="the first day")
    end: dt.date = Field(description="the last day")

    @model_validator(mode="after")
    def _ordered(self) -> "Booking":
        if self.end < self.start:
            raise ValueError("the end must not be before the start")
        return self


async def test_a_model_level_error_re_asks_every_field_of_the_model_once() -> None:
    world = Script()
    world.clairic = {
        "start": ca("date", "2024-05-10"),
        "end": ca("date", "2024-05-01"),
    }
    world.repair = {"start": ca("date", "2024-05-01"), "end": ca("date", "2024-05-10")}
    result = await run([Booking], world, "Booked from 1 May 2024 to 10 May 2024.")
    record = only(result)
    assert value_of(record, Booking).start == dt.date(2024, 5, 1)
    assert sorted(c[2].field or "" for c in repair_calls(world)) == ["end", "start"]
    assert result.attempts == 2
    # "anything else": the same type, with the error in hand
    assert request_for(world, "start")["fields"]["start"]["type"] == "date"
    assert "the end must not be before the start" in instructions(world, "start")
    state = request_for(world, "end")["state"]
    assert '"field": "(record)"' in state and '"type": "value_error"' in state


# --------------------------------------------------------------------------- #
# Patterns and other formats: Clairic only quotes the input
# --------------------------------------------------------------------------- #


async def test_a_pattern_failure_is_re_asked_once_if_the_input_could_satisfy_it() -> (
    None
):
    world = applicant_world(postcode=ca("text", "Berlin 10115", find="Berlin 10115"))
    world.repair = {"postcode": ca("text", "10115", find="10115")}
    result = await run([Applicant], world, APPLICANT)
    record = only(result)
    assert value_of(record, Applicant).postcode == "10115"
    (call,) = repair_calls(world)
    text = call[1]["fields"]["postcode"]["instructions"]
    assert "Must satisfy: text that matches the pattern ^\\d{5}$" in text
    assert "quoted verbatim from the input" in text
    assert "The previous answer 'Berlin 10115' was rejected because" in text
    answer = record.fields["postcode"]
    assert answer.repaired and answer.span is not None
    start, end = answer.span
    assert APPLICANT[start:end] == "10115"  # the new span is in the input


async def test_a_pattern_answer_without_a_span_in_the_input_is_not_accepted() -> None:
    world = applicant_world(postcode=ca("text", "Berlin 10115", find="Berlin 10115"))
    world.repair = {"postcode": ca("text", "10115")}  # matches, but quotes nothing
    result = await run([Applicant], world, APPLICANT)
    record = only(result)
    assert not record.valid
    assert len(repair_calls(world)) == 1  # failed fast: no second try
    assert result.attempts == 2
    notes = record.fields["postcode"].warnings
    assert any("does not point at text in the input" in n for n in notes)
    assert not record.fields["postcode"].repaired


async def test_a_pattern_answer_that_still_does_not_match_fails_fast() -> None:
    world = applicant_world(postcode=ca("text", "Berlin 10115", find="Berlin 10115"))
    world.repair = {"postcode": ca("text", "Berlin", find="Berlin")}
    result = await run([Applicant], world, APPLICANT)
    assert not only(result).valid
    assert len(repair_calls(world)) == 1
    assert any(
        "still does not meet the format" in n
        for n in only(result).fields["postcode"].warnings
    )


async def test_a_pattern_answer_read_from_the_rejected_attempt_is_not_accepted() -> (
    None
):
    world = applicant_world(postcode=ca("text", "Berlin 10115", find="Berlin 10115"))
    # the quoted form exists only in the <previous_attempt> block, not in the input
    world.repair = {"postcode": ca("text", '"Berlin 10115"', find='"Berlin 10115"')}
    result = await run([Applicant], world, APPLICANT)
    record = only(result)
    assert not record.valid and len(repair_calls(world)) == 1
    assert any("not from it" in n for n in record.fields["postcode"].warnings)


async def test_without_a_candidate_in_the_input_a_pattern_field_is_never_asked() -> (
    None
):
    text = "Katarina Vogel lives in Berlin and is 34 years old."
    world = applicant_world(postcode=ca("text", "Berlin", find="Berlin"))
    result = await run([Applicant], world, text)
    record = only(result)
    assert not record.valid
    assert repair_calls(world) == []  # no request at all
    assert result.attempts == 1  # and no round was spent
    assert any(
        "the input holds no text that could satisfy the format" in n
        for n in record.fields["postcode"].warnings
    )


async def test_pattern_re_asks_can_be_switched_off() -> None:
    world = applicant_world(postcode=ca("text", "Berlin 10115", find="Berlin 10115"))
    result = await run(
        [Applicant], world, APPLICANT, repair=RepairConfig(allow_pattern_reask=False)
    )
    assert not only(result).valid
    assert repair_calls(world) == []
    assert any(
        "allow_pattern_reask" in n for n in only(result).fields["postcode"].warnings
    )


async def test_other_fields_are_still_repaired_while_a_pattern_field_gives_up() -> None:
    text = "Katarina Vogel lives in Berlin and is 34 years old."
    world = applicant_world(
        age=ca("number", 12), postcode=ca("text", "Berlin", find="Berlin")
    )
    world.repair = {"age": ca("number", 34)}
    result = await run([Applicant], world, text)
    assert not only(result).valid  # the postcode cannot be fixed
    assert [c[2].field for c in repair_calls(world)] == ["age"]  # but age was asked
    assert result.attempts == 2


async def test_another_format_error_needs_a_piece_of_the_input_that_validates() -> None:
    from pydantic import HttpUrl

    class Page(BaseModel):
        title: str
        link: HttpUrl = Field(description="the page's address")

    world = Script()
    world.clairic = {
        "title": ca("text", "Docs", find="Docs"),
        "link": ca("text", "docs.example.com", find="docs.example.com"),
    }
    world.repair = {
        "link": ca("text", "https://docs.example.com", find="https://docs.example.com")
    }
    # a url that the input spells in full could be quoted: one re-ask
    text = "Docs live at docs.example.com, see https://docs.example.com for more."
    record = only(await run([Page], world, text))
    assert record.valid and len(repair_calls(world)) == 1
    # a url that the input never spells with a scheme cannot be: no re-ask
    world = Script()
    world.clairic = {
        "title": ca("text", "Docs", find="Docs"),
        "link": ca("text", "docs.example.com", find="docs.example.com"),
    }
    result = await run([Page], world, "Docs live at docs.example.com.")
    assert not only(result).valid and repair_calls(world) == []


async def test_a_repair_request_that_is_too_long_is_chunked_with_its_blocks() -> None:
    class Where(BaseModel):
        name: str
        zip: str = Field(pattern=r"^\d{5}$", description="the zip code")

    filler = " ".join(f"w{i}" for i in range(2500))
    text = f"Ada Lovelace lives in WA 98402. {filler}"
    world = Script()
    world.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace"),
        "zip": ca("text", "WA 98402", find="WA 98402"),
    }
    world.repair = {"zip": ca("text", "98402", find="98402")}
    # the first request that follows the extraction (the repair) is cut short
    world.limit = lambda state, n: 4000 if n == 1 else None
    result = await run([Where], world, text)
    record = only(result)
    assert record.valid and value_of(record, Where).zip == "98402"
    calls = repair_calls(world)
    assert len(calls) >= 3  # the whole request, then the chunks that cover the input
    for _, payload, _ in calls:
        state = payload["state"]
        assert "<validation_errors>" in state and "<previous_attempt>" in state
    span = record.fields["zip"].span
    assert span is not None and text[span[0] : span[1]] == "98402"
    assert result.attempts == 2  # the chunks of one request are not rounds


# --------------------------------------------------------------------------- #
# Attempts, repeats and the cap
# --------------------------------------------------------------------------- #


class Adult(BaseModel):
    name: str
    age: int = Field(ge=18)


ADULT = "Ada is 12."


def adult_world(*repairs: Any) -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Ada", find="Ada"),
        "age": ca("number", 12),
    }
    world.repair = {"age": list(repairs)}
    return world


async def test_the_loop_stops_when_the_error_signature_repeats() -> None:
    # every re-ask answers something else that is still too small: the same
    # (path, error type) comes back, so asking again would not help
    world = adult_world(ca("number", 10), ca("number", 11), ca("number", 13))
    result = await run([Adult], world, ADULT)
    assert not only(result).valid
    assert len(repair_calls(world)) == 1
    assert result.attempts == 2


async def test_without_stop_on_repeat_max_attempts_is_the_only_guard() -> None:
    world = adult_world(ca("number", 10), ca("number", 11), ca("number", 13))
    config = RepairConfig(stop_on_repeat=False, max_attempts=2)
    result = await run([Adult], world, ADULT, repair=config)
    assert not only(result).valid
    assert len(repair_calls(world)) == 2  # exactly the cap
    assert result.attempts == 3  # the extraction plus two repair rounds
    # a later answer is the one that was tried last
    state = repair_calls(world)[1][1]["state"]
    assert "<previous_attempt>\n10\n</previous_attempt>" in state


async def test_a_re_ask_that_gives_back_the_rejected_value_changes_nothing() -> None:
    world = adult_world(ca("number", 12))
    result = await run([Adult], world, ADULT)
    record = only(result)
    assert not record.valid and len(repair_calls(world)) == 1
    assert any(
        "is the value that was rejected" in n for n in record.fields["age"].warnings
    )


async def test_a_repair_that_works_in_a_later_round_counts_every_round() -> None:
    class Pair(BaseModel):
        low: int = Field(ge=0)
        high: int = Field(ge=0, le=100)

    world = Script()
    world.clairic = {"low": ca("number", -5), "high": ca("number", 500)}
    # round 1 fixes `low` and answers a different wrong value for `high`;
    # the errors are no longer the same set, so round 2 asks `high` again
    world.repair = {
        "low": ca("number", 5),
        "high": [ca("number", 1000), ca("number", 50)],
    }
    result = await run([Pair], world, "low 5, high 50")
    record = only(result)
    assert record.valid and record.value == Pair(low=5, high=50)
    assert [c[2].field for c in repair_calls(world)] == ["low", "high", "high"]
    assert result.attempts == 3
    # `low` was valid after round 1 and was not asked again
    assert sum(c[2].field == "low" for c in repair_calls(world)) == 1


# --------------------------------------------------------------------------- #
# Repaired values are verified like any answer
# --------------------------------------------------------------------------- #


async def test_a_repaired_value_that_fails_attribution_is_not_used() -> None:
    world = applicant_world(age=ca("number", 12))
    world.repair = {"age": ca("number", 34)}
    # the first check passes the (too small) 12; the repaired 34 is refused
    world.attribution = {"attr.age": [0.95, 0.05]}
    result = await run([Applicant], world, APPLICANT)
    record = only(result)
    assert not record.valid
    assert record.after is not None and record.after["age"] == 12  # not 34
    assert kinds(world.transport).count("attribution") == 2
    age = record.fields["age"]
    assert not age.repaired
    assert any("did not pass verification (attribution)" in n for n in age.warnings)


async def test_a_repaired_value_goes_through_the_whole_ladder() -> None:
    # a low-confidence repair is re-asked ("reask") before it is attributed
    world = applicant_world(age=ca("number", 12))
    world.repair = {"age": ca("number", 34, confidence=0.6)}
    world.reask = {"age": ca("number", 34, confidence=0.9)}
    record = only(await run([Applicant], world, APPLICANT))
    assert record.valid and record.fields["age"].reask
    names = kinds(world.transport)
    assert names.index("repair") < names.index("reask")


# --------------------------------------------------------------------------- #
# Terminal outcomes of an insert
# --------------------------------------------------------------------------- #


class Pet(BaseModel):
    name: str
    nickname: Optional[str] = Field(default=None, max_length=5)


async def test_an_optional_field_that_still_fails_goes_back_to_none() -> None:
    world = Script()
    world.clairic = {
        "name": ca("text", "Rex", find="Rex"),
        "nickname": ca("text", "Bartholomew", find="Bartholomew"),
    }
    world.repair = {"nickname": ca("text", None, p_null=0.99)}
    result = await run([Pet], world, "Rex, also called Bartholomew.")
    record = only(result)
    assert record.valid and record.value == Pet(name="Rex", nickname=None)
    assert record.after == {"name": "Rex", "nickname": None}
    assert result.skipped == () and len(result.responses) == 1
    dropped = [r for r in record.rejected if r.path == "/nickname"]
    assert [(r.op, r.reason) for r in dropped] == [(None, "validation")]
    assert "String should have at most 5 characters" in dropped[0].detail
    answer = record.fields["nickname"]
    assert answer.value is None and answer.rejected_reason == "validation"
    assert not answer.repaired


async def test_a_required_field_that_still_fails_makes_the_record_invalid() -> None:
    world = adult_world(ca("number", 12))
    result = await run([Adult], world, ADULT)
    record = only(result)
    assert record.valid is False and record.value is None
    assert record.after == {"name": "Ada", "age": 12}
    assert record.errors == ("age: Input should be greater than or equal to 18",)
    assert result.responses == () and result.response_metadata == []
    (skipped,) = result.skipped
    assert (skipped.schema_name, skipped.reason) == ("Adult", "validation_failed")
    assert skipped.detail == "age: Input should be greater than or equal to 18"
    assert skipped.entity is record.entity


async def test_strict_raises_when_the_repair_did_not_fix_it() -> None:
    world = adult_world(ca("number", 12))
    extractor = make_extractor([Adult], world, repair=ON, strict=True)
    with pytest.raises(ExtractionValidationError) as info:
        await extractor.ainvoke(ADULT)
    error = info.value
    assert error.errors == ["age: Input should be greater than or equal to 18"]
    assert isinstance(error.record, ResultRecord) and error.record.valid is False
    assert "repair" in [c.kind for c in error.calls]  # the log shows the re-ask
    assert len(error.calls) == len(world.transport.calls)


async def test_strict_does_not_raise_when_the_repair_fixes_it() -> None:
    world = adult_world(ca("number", 30))
    result = await run([Adult], world, ADULT, strict=True)
    assert only(result).valid and result.skipped == ()


async def test_a_valid_record_is_never_repaired() -> None:
    world = applicant_world()
    result = await run([Applicant], world, APPLICANT)
    assert only(result).valid and repair_calls(world) == []
    assert result.attempts == 1
    assert not any(f.repaired for f in only(result).fields.values())


async def test_with_repair_off_the_record_is_as_it_was_before() -> None:
    world = adult_world(ca("number", 30))
    result = await run([Adult], world, ADULT, repair=RepairConfig(enabled=False))
    record = only(result)
    assert not record.valid and repair_calls(world) == []
    assert result.attempts == 1
    assert record.errors == ("age: Input should be greater than or equal to 18",)
    assert result.skipped == ()  # just an invalid record, as before the loop


async def test_with_repair_off_an_optional_failure_still_invalidates_the_record() -> (
    None
):
    world = Script()
    world.clairic = {
        "name": ca("text", "Rex", find="Rex"),
        "nickname": ca("text", "Bartholomew", find="Bartholomew"),
    }
    off = RepairConfig(enabled=False)
    result = await run([Pet], world, "Rex, also called Bartholomew.", repair=off)
    record = only(result)
    assert not record.valid  # nothing re-asks and nothing is dropped to its default
    assert record.errors[0].startswith("nickname:")
    assert repair_calls(world) == []
    assert not [r for r in record.rejected if r.reason == "validation"]


async def test_attempts_is_the_maximum_over_the_schemas() -> None:
    world = Script()
    world.clairic = {
        "name": ca("text", "Ada", find="Ada"),
        "age": ca("number", 12),
        "nickname": ca("text", None, p_null=0.99),
    }
    world.repair = {"age": ca("number", 30)}
    result = await run([Adult, Pet], world, ADULT)
    by_schema = {r.schema_name: r for r in result.records}
    assert by_schema["Adult"].valid and by_schema["Pet"].valid
    assert result.attempts == 2  # one record needed a repair round, the other did not


# --------------------------------------------------------------------------- #
# Inside sub-models and unions
# --------------------------------------------------------------------------- #


class Addr(BaseModel):
    city: str
    zip: Optional[str] = Field(default=None, pattern=r"^\d{5}$")


class Writer(BaseModel):
    name: str
    address: Optional[Addr] = None


class Novel(BaseModel):
    title: str
    author: Writer


NOVEL = "Dune was written by Frank Herbert, who lives in Tacoma, WA 98402."


def novel_world() -> Script:
    world = Script()
    world.clairic = {
        "title": ca("text", "Dune", find="Dune"),
        "author.name": ca("text", "Frank Herbert", find="Frank Herbert"),
        "author.address.city": ca("text", "Tacoma", find="Tacoma"),
        "author.address.zip": ca("text", "WA 98402", find="WA 98402"),
    }
    return world


async def test_an_optional_sub_model_is_repaired_before_it_is_dropped() -> None:
    world = novel_world()
    world.repair = {"author.address.zip": ca("text", "98402", find="98402")}
    record = only(await run([Novel], world, NOVEL))
    novel = value_of(record, Novel)
    assert novel.author.address == Addr(city="Tacoma", zip="98402")
    assert [c[2].field for c in repair_calls(world)] == ["author.address.zip"]
    assert record.fields["author.address.zip"].repaired
    assert not [r for r in record.rejected if r.reason == "invalid_optional_group"]


async def test_an_optional_sub_model_that_stays_wrong_is_dropped_as_before() -> None:
    world = novel_world()
    world.repair = {"author.address.zip": ca("text", None, p_null=0.99)}
    record = only(await run([Novel], world, NOVEL))
    novel = value_of(record, Novel)
    assert novel.author.address is None  # the group is absent, not repaired
    assert [r.reason for r in record.rejected if r.path == "/author/address"] == [
        "invalid_optional_group"
    ]
    assert len(repair_calls(world)) == 1


# --------------------------------------------------------------------------- #
# Updates
# --------------------------------------------------------------------------- #


class Member(BaseModel):
    name: str
    age: int = Field(ge=0, le=130)


MEMBER = {"name": "Dana", "age": 40}
MEMBER_TEXT = "Dana turned 410 today. No wait, she turned 41."


def member_world() -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana", find="Dana"),
        "age": ca("number", 410),
    }
    world.repair = {"age": ca("number", 41)}
    return world


async def update(
    world: Script, record: Any = None, text: str = MEMBER_TEXT, **kwargs: Any
) -> ExtractionResult:
    extractor = make_extractor([One(Member)], world, repair=ON, **kwargs)
    return await extractor.ainvoke(
        text, existing=[("m-1", "Member", MEMBER if record is None else record)]
    )


async def test_an_update_whose_patch_breaks_validation_is_repaired() -> None:
    world = member_world()
    result = await update(world)
    record = only(result)
    assert record.valid and record.kind == "update"
    assert [(p.op, p.path, p.value) for p in record.patches] == [
        ("replace", "/age", 41)
    ]
    assert record.after == {"name": "Dana", "age": 41}
    assert result.attempts == 2
    assert [c[2].field for c in repair_calls(world)] == ["age"]
    age = record.fields["age"]
    assert age.repaired and age.gate_p is not None
    assert not [r for r in record.rejected if r.reason == "validation"]


async def test_a_repair_request_of_an_update_never_shows_the_record() -> None:
    world = member_world()
    await update(world)
    state: str = request_for(world, "age")["state"]
    assert "<existing" not in state
    assert json.dumps(MEMBER) not in state and '"age": 40' not in state
    assert state.startswith('<focus entity="the Member">')
    assert "Dana turned 410 today" in state  # the input is there
    assert "<previous_attempt>\n410\n</previous_attempt>" in state
    assert "greater_than" not in state and "less_than_equal" in state
    # and the cross-check equivalent is not needed: no `crosscheck` after a repair
    names = kinds(world.transport)
    assert "crosscheck" not in names[names.index("repair") :]


async def test_a_repaired_update_value_is_judged_by_the_gate_again() -> None:
    world = member_world()
    world.gate = {"/age": [0.95, 0.95]}
    await update(world)
    gated = [c[1]["state"]["proposed_change"] for c in world.calls("gate")]
    # the first operation (410) was judged before it broke validation, and the
    # operation the repaired value makes (41) is judged on its own
    assert [g["to"] for g in gated] == [410, 41]
    assert [g["path"] for g in gated] == ["/age", "/age"]


async def test_a_repaired_update_value_the_gate_refuses_is_not_applied() -> None:
    world = member_world()
    world.gate = {"/age": [0.95, 0.05]}
    record = only(await update(world))
    assert record.valid
    assert record.patches == () and record.after == record.before
    refused = [r for r in record.rejected if r.reason == "gate_noul"]
    assert [(r.op, r.path) for r in refused] == [("replace", "/age")]


async def test_an_update_repair_that_fails_attribution_is_dropped_as_before() -> None:
    world = member_world()
    world.attribution = {"attr.age": [0.95, 0.05]}
    record = only(await update(world))
    # the repair was not used, so the update ends as it did before the loop:
    # the operation that breaks validation is dropped
    assert record.valid and record.patches == ()
    assert [
        (r.op, r.path, r.reason) for r in record.rejected if r.reason == "validation"
    ] == [("replace", "/age", "validation")]
    assert record.after == record.before
    assert len(repair_calls(world)) == 1


async def test_an_error_the_record_already_had_is_not_the_updates_to_repair() -> None:
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana", find="Dana"),
        "age": ca("number", None, p_null=0.98),
    }
    broken = {"name": "Dana", "age": 200}  # invalid before the update
    record = only(await update(world, broken, "Dana says hello."))
    assert not record.valid
    assert repair_calls(world) == []  # no operation touched it
    assert record.errors and record.errors[0].startswith("age:")


async def test_an_error_below_a_union_member_in_an_update_is_repaired() -> None:
    world = Script()
    world.clairic = {
        "name": ca("text", "Sam", find="Sam"),
        "pet.Dog.tricks": ca("number", -2),
        "pet.Cat.lives": ca("number", None, p_null=0.99),
    }
    world.clef = {"pet": pick("dog", ["cat", "dog"])}
    world.repair = {"pet.Dog.tricks": ca("number", 2)}
    extractor = make_extractor([One(Owner)], world, repair=ON)
    record = only(
        await extractor.ainvoke(
            "Sam's dog knows 2 tricks. Wait, minus 2?",
            existing={"Owner": {"name": "Sam", "pet": {"kind": "dog", "tricks": 3}}},
        )
    )
    assert record.valid, record.errors
    assert record.after == {"name": "Sam", "pet": {"kind": "dog", "tricks": 2}}
    assert [c[2].field for c in repair_calls(world)] == ["pet.Dog.tricks"]
    assert [(p.op, p.path) for p in record.patches] == [("replace", "/pet/tricks")]
    assert record.fields["pet.tricks"].repaired


# --------------------------------------------------------------------------- #
# The record never reaches a repair request
# --------------------------------------------------------------------------- #


class Plan(BaseModel):
    name: str
    tier: Literal["gold", "silver"]
    seats: int

    @model_validator(mode="after")
    def _allowed(self) -> "Plan":
        if self.tier == "silver" and self.seats > 5:
            raise ValueError("this combination is not allowed")  # names no field
        return self


class NamedPlan(Plan):
    @model_validator(mode="after")
    def _named(self) -> "NamedPlan":
        return self


PLAN = {"name": "Acme", "tier": "gold", "seats": 3, "secret_note": "CEO is Zed"}
PLAN_TEXT = "Acme now has 8 seats on silver."


def plan_world(**repair: Any) -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Acme", find="Acme"),
        "seats": ca("number", 8),
    }
    world.clef = {"tier": pick("silver", ["gold", "silver"])}
    world.repair = {
        "seats": ca("number", 5),
        "tier": pick("silver", ["gold", "silver"]),
    }
    world.repair.update(repair)
    return world


async def update_plan(world: Script, schema: Any = Plan) -> ExtractionResult:
    extractor = make_extractor([One(schema)], world, repair=ON)
    return await extractor.ainvoke(PLAN_TEXT, existing={schema.__name__: PLAN})


async def test_a_model_level_error_on_an_update_leaks_no_record_value() -> None:
    world = plan_world()
    await update_plan(world)
    calls = repair_calls(world)
    assert {c[2].field for c in calls} == {"tier", "seats"}  # Clef and Clairic
    assert any(c[0] != CLAIRIC for c in calls) and any(c[0] == CLAIRIC for c in calls)
    for _, payload, _ in calls:
        text = json.dumps(payload)
        assert "CEO is Zed" not in text  # not even a key the schema does not know
        assert "secret_note" not in text
        # nor the record's own values, as text in a state or as keys of a Clef one
        assert 'seats\\": 3' not in text and '"seats": 3' not in text
    # the error is there, without the document that pydantic attaches to it
    state = request_for(world, "seats")["state"]
    (error,) = json.loads(
        state.split("<validation_errors>\n")[1].split("\n</validation_errors>")[0]
    )
    assert error["type"] == "value_error" and "input" not in error


async def test_the_previous_attempt_of_a_list_is_the_answer_not_the_patched_list() -> (
    None
):
    class Notes(BaseModel):
        name: str
        notes: list[str] = Field(default_factory=list, max_length=2)

    world = Script()
    world.clairic = {
        "name": ca("text", "Dana", find="Dana"),
        "notes": ca("list", ["asked about X"], find_all=["asked about X"]),
    }
    world.repair = {"notes": ca("list", None, p_null=0.99)}
    extractor = make_extractor([One(Notes)], world, repair=ON)
    record = only(
        await extractor.ainvoke(
            "Dana asked about X.",
            existing={"Notes": {"name": "Dana", "notes": ["secret a", "secret b"]}},
        )
    )
    assert not record.valid or record.patches == ()  # three notes do not fit
    (call,) = repair_calls(world)
    text = json.dumps(call[1])
    assert "secret a" not in text and "secret b" not in text
    state = call[1]["state"]
    assert '<previous_attempt>\n["asked about X"]\n</previous_attempt>' in state


# --------------------------------------------------------------------------- #
# A model-level error: minimal repairs
# --------------------------------------------------------------------------- #


class Stay(BaseModel):
    guest: str
    room: Optional[str] = None
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _order(self) -> "Stay":
        if self.end < self.start:
            raise ValueError("the end must not be before the start")
        return self


class VagueStay(Stay):
    @model_validator(mode="after")
    def _vague(self) -> "VagueStay":
        return self


class Vague(BaseModel):
    guest: str
    room: Optional[str] = None
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _order(self) -> "Vague":
        if self.end < self.start:
            raise ValueError("the dates are inconsistent")  # names no field
        return self


STAY = "Ann Lee booked Suite 12 from 1 May 2024 to 10 May 2024."


def stay_world(**repair: Any) -> Script:
    world = Script()
    world.clairic = {
        "guest": ca("text", "Ann Lee", find="Ann Lee"),
        "room": ca("text", "12", find="12"),
        "start": ca("date", "2024-05-10"),
        "end": ca("date", "2024-05-01"),
    }
    world.repair = {
        "guest": ca("text", "Ann", find="Ann"),
        "room": ca("text", "Suite 12", find="Suite 12"),
        "start": ca("date", "2024-05-01"),
        "end": ca("date", "2024-05-10"),
    }
    world.repair.update(repair)
    return world


async def test_a_message_that_names_fields_re_asks_only_those() -> None:
    world = stay_world()
    record = only(await run([Stay], world, STAY))
    stay = value_of(record, Stay)
    assert sorted(c[2].field or "" for c in repair_calls(world)) == ["end", "start"]
    assert (stay.guest, stay.room) == ("Ann Lee", "12")  # unrelated: not touched
    assert (stay.start, stay.end) == (dt.date(2024, 5, 1), dt.date(2024, 5, 10))
    assert record.fields["start"].repaired and not record.fields["guest"].repaired


async def test_an_unnamed_model_level_error_keeps_only_the_answers_it_needed() -> None:
    # nothing in the message says which field is wrong: all four are re-asked and
    # installed, then taken back one by one while the record stays valid
    world = stay_world(start=ca("date", "2024-05-10"))  # `start` has nothing new
    record = only(await run([Vague], world, STAY))
    stay = value_of(record, Vague)
    assert sorted(c[2].field or "" for c in repair_calls(world)) == [
        "end",
        "guest",
        "room",
        "start",
    ]
    assert (stay.guest, stay.room) == ("Ann Lee", "12")  # the changes were reverted
    assert stay.end == dt.date(2024, 5, 10)  # the one that fixed it stays
    assert record.fields["end"].repaired
    assert not record.fields["guest"].repaired and not record.fields["room"].repaired
    assert any("reverted" in n for n in record.fields["guest"].warnings)


async def test_unneeded_answers_are_taken_back_when_the_error_stays() -> None:
    # the dates cannot be fixed (their re-asks repeat the first answers): the
    # changes to the unrelated fields removed no error, so none of them stays
    world = stay_world(start=ca("date", "2024-05-10"), end=ca("date", "2024-05-01"))
    record = only(await run([Vague], world, STAY))
    assert not record.valid
    assert record.after is not None
    assert (record.after["guest"], record.after["room"]) == ("Ann Lee", "12")
    assert not record.fields["guest"].repaired
    assert any("reverted" in n for n in record.fields["room"].warnings)


# --------------------------------------------------------------------------- #
# An update repair never goes back to the record
# --------------------------------------------------------------------------- #


async def test_a_repair_back_to_the_records_value_keeps_the_operation_rejected() -> (
    None
):
    world = plan_world(tier=pick("gold", ["gold", "silver"]))  # the record's tier
    record = only(await update_plan(world))
    refused = [r for r in record.rejected if r.reason == "validation" and r.op]
    assert [(r.op, r.path) for r in refused] == [("replace", "/tier")]
    assert "the record's own value" in refused[0].detail
    tier = record.fields["tier"]
    assert not tier.repaired
    assert any("failed" in n for n in tier.warnings)
    # the seats were repaired to 5, but silver still needs the tier it was given:
    # the update is valid only with the tier operation, so it stays as asked
    assert record.valid and record.after is not None
    assert record.after["tier"] == "silver" and record.after["seats"] == 5


async def test_a_repair_that_still_changes_the_value_is_used() -> None:
    world = plan_world(tier=pick("silver", ["gold", "silver"]))
    record = only(await update_plan(world))
    assert record.valid and not [r for r in record.rejected if r.op]


# --------------------------------------------------------------------------- #
# The error signature ignores list positions
# --------------------------------------------------------------------------- #


def test_list_positions_do_not_change_the_error_signature() -> None:
    plan = compile_schema(Profile)
    first = [Problem(("tags", 2), "string_too_short", "m")]
    second = [Problem(("tags", 3), "string_too_short", "m")]
    assert signature(plan, first, {}) == signature(plan, second, {})
    assert signature(plan, first, {}) == frozenset({("/tags/*", "string_too_short")})


async def test_an_update_with_the_loop_off_drops_the_operation_as_before() -> None:
    world = member_world()
    extractor = make_extractor([One(Member)], world, repair=RepairConfig(enabled=False))
    result = await extractor.ainvoke(MEMBER_TEXT, existing=[("m-1", "Member", MEMBER)])
    record = only(result)
    assert record.valid and record.patches == () and repair_calls(world) == []
    assert result.attempts == 1
    assert [r.reason for r in record.rejected if r.reason == "validation"] == [
        "validation"
    ]
