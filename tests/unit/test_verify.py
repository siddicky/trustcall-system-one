"""Tests of the verification ladder, driven through the extractor."""

# ruff: noqa: UP045 - tests deliberately spell Optional and Union

import datetime as dt
import math
from typing import Any, Literal, Optional

import pytest
from pydantic import BaseModel, Field, create_model

from tests.unit.extract_helpers import Script, ca, kinds, make_extractor, pick
from trustcall_system_one import (
    ResultRecord,
    VerifyConfig,
    compile_schema,
    s1_field,
)
from trustcall_system_one._verify import Resolved, accept, agree, display


class Person(BaseModel):
    """A person mentioned in the text."""

    name: str
    city: Optional[str] = None
    age: Optional[int] = None


TEXT = "Ada Lovelace lives in London and is 36 years old."


def person_script() -> Script:
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace"),
        "city": ca("text", "London", find="London"),
        "age": ca("number", 36),
    }
    return script


def person(record: ResultRecord) -> Person:
    assert isinstance(record.value, Person)
    return record.value


PADDED = "x" * 40 + " " + TEXT + " " + "y" * 40
RECHECK = VerifyConfig(window_chars=10, borderline_null=False, isolate_decisions=False)
"""A narrow window: the re-check's state differs from the first check's (a state that
does not differ is answered from the run, which is what the memo is for)."""


async def run_person(
    script: Script, *, text: str = TEXT, **kwargs: Any
) -> ResultRecord:
    extractor = make_extractor([Person], script, **kwargs)
    result = await extractor.ainvoke(text)
    (record,) = result.records
    return record


# --------------------------------------------------------------------------- #
# Answer-level thresholds
# --------------------------------------------------------------------------- #


async def test_a_clean_run_makes_no_verification_calls_besides_attribution() -> None:
    script = person_script()
    record = await run_person(script)
    assert record.valid
    assert record.value == Person(name="Ada Lovelace", city="London", age=36)
    assert kinds(script.transport)[0] == "presence"
    assert sorted(kinds(script.transport)[1:]) == ["attribution", "extract"]


async def test_low_confidence_below_the_floor_is_rejected() -> None:
    script = person_script()
    script.clairic["city"] = ca("text", "London", find="London", confidence=0.4)
    record = await run_person(script)
    assert record.value == Person(name="Ada Lovelace", age=36)
    answer = record.fields["city"]
    assert answer.rejected_reason == "low_confidence"
    assert answer.value is None
    assert answer.source == "default"
    (rejected,) = record.rejected
    assert (rejected.op, rejected.path, rejected.reason) == (
        None,
        "/city",
        "low_confidence",
    )
    assert rejected.p == 0.4
    assert "reask" not in kinds(script.transport)  # rejected answers are not re-asked


async def test_field_level_min_confidence_overrides_the_default() -> None:
    class Strict(BaseModel):
        name: Optional[str] = Field(
            None, json_schema_extra=s1_field(min_confidence=0.95)
        )

    script = Script()
    script.clairic = {"name": ca("text", "Ada", find="Ada", confidence=0.9)}
    script.reask = {"name": ca("text", "Ada", find="Ada", confidence=0.9)}
    script.adjudicate = {}
    extractor = make_extractor([Strict], script)
    result = await extractor.ainvoke("Ada is here.")
    assert result.records[0].fields["name"].rejected_reason == "low_confidence"


async def test_p_null_above_the_threshold_rejects_a_value() -> None:
    script = person_script()
    # city is optional: null_threshold 0.3; p_null 0.9 is outside the null band
    script.clairic["city"] = ca("text", "London", find="London", p_null=0.9)
    record = await run_person(script)
    assert record.fields["city"].rejected_reason == "p_null"
    assert record.fields["city"].p_null == 0.9
    assert person(record).city is None


async def test_unparseable_answer_is_rejected_not_raised() -> None:
    script = person_script()
    script.clairic["age"] = ca("number", "many")
    record = await run_person(script)
    assert record.fields["age"].rejected_reason == "unparseable"
    assert record.rejected[0].detail
    assert record.valid


async def test_guess_is_never_used_as_a_value() -> None:
    script = person_script()
    script.clairic["city"] = ca("text", None, guess="Paris", p_null=0.95)
    record = await run_person(script)
    assert person(record).city is None
    assert record.fields["city"].guess == "Paris"
    assert record.fields["city"].value is None
    assert record.fields["city"].rejected_reason is None  # a null is not a rejection


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #


async def test_attribution_asks_one_noul_per_value_in_one_request() -> None:
    script = person_script()
    await run_person(script)
    (call,) = script.calls("attribution")
    questions = call[1]["questions"]
    assert sorted(questions) == ["attr.age", "attr.city", "attr.name"]
    assert questions["attr.city"]["instructions"] == (
        'Does `input` state that city of `entity.label` is "London", or does it '
        "follow by simple arithmetic or date math from figures the input states "
        "about `entity.label`?"
    )
    assert call[1]["state"] == {
        "input": TEXT,
        "entity": {"label": "the Person", "mentions": []},
    }
    assert questions["attr.city"]["type"] == "noul"
    assert set(questions["attr.city"]["criteria"]) == {"true", "false"}


async def test_attribution_below_drop_removes_the_value() -> None:
    script = person_script()
    script.attribution = {"attr.city": 0.1}
    record = await run_person(script)
    assert person(record).city is None
    answer = record.fields["city"]
    assert answer.rejected_reason == "attribution"
    assert answer.attribution_p == 0.1
    assert answer.value is None
    assert [r.reason for r in record.rejected] == ["attribution"]
    assert "reask" not in kinds(script.transport)


async def test_attribution_at_or_above_keep_keeps_the_value() -> None:
    script = person_script()
    script.attribution = {"attr.city": 0.75}
    record = await run_person(script)
    assert person(record).city == "London"
    assert record.fields["city"].attribution_p == 0.75


async def test_a_bleed_at_0_62_is_no_longer_kept() -> None:
    """A live run kept another person's city at 0.62, just over the old 0.6."""
    script = person_script()
    script.attribution = {"attr.city": [0.62, 0.3]}
    script.reask = {"city": ca("text", "London", find="London")}
    record = await run_person(script, text=PADDED, verify=RECHECK)
    assert record.fields["city"].rejected_reason == "attribution"
    assert record.fields["city"].reask is True  # the middle band re-checks first
    assert kinds(script.transport).count("attribution") == 2


async def test_middle_band_reasks_in_isolation_then_checks_again() -> None:
    text = "x" * 40 + " Ada lives in London. " + "y" * 40
    script = person_script()
    script.clairic["name"] = ca("text", "Ada", find="Ada")
    script.clairic["city"] = ca("text", "London", find="London")
    script.clairic["age"] = ca("number", None, p_null=0.99)
    script.attribution = {"attr.city": [0.45, 0.8]}
    script.reask = {"city": ca("text", "London", find="London")}
    extractor = make_extractor([Person], script, verify=RECHECK)
    result = await extractor.ainvoke(text)
    record = result.records[0]
    assert person(record).city == "London"
    assert record.fields["city"].reask is True
    assert record.fields["city"].attribution_p == 0.8
    seen = kinds(script.transport)
    assert seen.count("reask") == 1
    assert seen.count("attribution") == 2
    # the re-ask and the second check both see only the neighbourhood
    reask_state = script.calls("reask")[0][1]["state"]
    assert "…" in reask_state
    assert "x" * 40 not in reask_state
    assert script.calls("reask")[0][1]["fields"]["city"]["instructions"].endswith(
        "Answer only from the text near these mentions."
    )
    second = script.calls("attribution")[1][1]
    assert "…" in second["state"]["input"]
    assert "y" * 40 not in second["state"]["input"]
    # the span still maps to the original text
    span = record.fields["city"].span
    assert span is not None and text[span[0] : span[1]] == "London"


async def test_middle_band_that_stays_low_drops_the_value() -> None:
    script = person_script()
    script.attribution = {"attr.city": [0.45, 0.4]}
    script.reask = {"city": ca("text", "London", find="London")}
    record = await run_person(script, text=PADDED, verify=RECHECK)
    assert record.fields["city"].rejected_reason == "attribution"
    assert record.fields["city"].attribution_p == 0.4
    assert record.fields["city"].reask is True


async def test_middle_band_whose_reask_finds_nothing_drops_the_value() -> None:
    script = person_script()
    script.attribution = {"attr.city": 0.45}
    script.reask = {"city": ca("text", None, p_null=0.99)}
    record = await run_person(script, text=PADDED, verify=RECHECK)
    assert record.fields["city"].rejected_reason == "attribution"
    assert kinds(script.transport).count("attribution") == 1


async def test_list_elements_are_attributed_one_by_one() -> None:
    class Notes(BaseModel):
        notes: list[str]

    text = "Notes: red, green, blue."
    script = Script()
    script.clairic = {
        "notes": ca("list", ["red", "green", "blue"], find_all=["red", "green", "blue"])
    }
    script.attribution = {
        "attr.notes.0": 0.9,
        "attr.notes.1": 0.05,
        "attr.notes.2": 0.8,
    }
    extractor = make_extractor([Notes], script)
    result = await extractor.ainvoke(text)
    record = result.records[0]
    assert record.value == Notes(notes=["red", "blue"])
    answer = record.fields["notes"]
    assert answer.attribution_p == (0.9, 0.05, 0.8)
    assert answer.spans is not None
    assert [text[a:b] for a, b in answer.spans] == ["red", "blue"]
    assert [(r.path, r.reason) for r in record.rejected] == [
        ("/notes/1", "attribution")
    ]
    assert answer.warnings == ("dropped 1 element(s): attribution",)
    (call,) = script.calls("attribution")
    question = call[1]["questions"]["attr.notes.1"]
    assert question["instructions"].startswith(
        'According to `input`, is "green" one of the notes of `entity.label`?'
    )
    assert question["criteria"] == {"true": "Yes.", "false": "No."}


async def test_list_with_every_element_dropped_is_rejected() -> None:
    class Notes(BaseModel):
        notes: Optional[list[str]] = None

    script = Script()
    script.clairic = {"notes": ca("list", ["a"], find_all=["a"])}
    script.attribution = {"attr.notes.0": 0.0}
    extractor = make_extractor([Notes], script)
    result = await extractor.ainvoke("a")
    record = result.records[0]
    assert record.fields["notes"].rejected_reason == "attribution"
    assert record.value == Notes()


async def test_attribution_can_be_switched_off() -> None:
    script = person_script()
    await run_person(script, verify=VerifyConfig(attribution=False))
    assert "attribution" not in kinds(script.transport)


async def test_clef_engine_values_skip_attribution() -> None:
    class Flag(BaseModel):
        status: Literal["open", "closed"]

    script = Script()
    script.clef = {"status": pick("open", ["open", "closed"])}
    extractor = make_extractor([Flag], script)
    result = await extractor.ainvoke("It is open.")
    assert result.records[0].value == Flag(status="open")
    assert sorted(kinds(script.transport)) == ["clef_fields", "presence"]


# --------------------------------------------------------------------------- #
# Low-confidence re-ask and adjudication
# --------------------------------------------------------------------------- #


def low_confidence_script(first: str = "London") -> Script:
    script = person_script()
    script.clairic["city"] = ca("text", first, find=first, confidence=0.7)
    return script


async def test_low_confidence_reask_that_agrees_keeps_the_better_answer() -> None:
    script = low_confidence_script()
    script.reask = {"city": ca("text", "london", find="London", confidence=0.97)}
    record = await run_person(script)
    answer = record.fields["city"]
    assert person(record).city == "london"  # the higher confidence
    assert answer.confidence == 0.97
    assert answer.reask is True
    assert "adjudicate" not in kinds(script.transport)
    # one re-ask for the one low-confidence field, with the narrowed state
    (reask,) = script.calls("reask")
    assert list(reask[1]["fields"]) == ["city"]


async def test_low_confidence_reask_keeps_the_first_when_it_is_more_confident() -> None:
    script = low_confidence_script()
    script.reask = {"city": ca("text", "London", find="London", confidence=0.6)}
    record = await run_person(script)
    assert record.fields["city"].confidence == 0.7
    assert record.fields["city"].reask is True


async def test_agreement_is_fuzzy_for_text() -> None:
    script = person_script()
    script.clairic["city"] = ca("text", "Londonn", find="London", confidence=0.7)
    script.reask = {"city": ca("text", "London", find="London", confidence=0.99)}
    record = await run_person(script)
    assert "adjudicate" not in kinds(script.transport)
    assert record.fields["city"].value == "London"


@pytest.mark.parametrize(
    ("verdict", "city", "reason"),
    [
        ("a", "London", None),
        ("b", "Paris", None),
        ("neither", None, "adjudication_failed"),
    ],
)
async def test_disagreeing_reask_goes_to_adjudication(
    verdict: str, city: str | None, reason: str | None
) -> None:
    text = "Ada Lovelace lives in London but works in Paris and is 36 years old."
    script = person_script()
    script.clairic["city"] = ca("text", "London", find="London", confidence=0.7)
    script.reask = {"city": ca("text", "Paris", find="Paris", confidence=0.9)}
    script.adjudicate = {"city": pick(verdict, ["a", "b", "neither"], 0.9)}
    extractor = make_extractor([Person], script)
    result = await extractor.ainvoke(text)
    record = result.records[0]
    answer = record.fields["city"]
    assert answer.value == city
    assert answer.rejected_reason == reason
    assert answer.reask is True
    (call,) = script.calls("adjudicate")
    question = call[1]["questions"]["adjudicate"]
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "a": "London",
        "b": "Paris",
        "neither": "The input states neither",
    }
    assert set(call[1]["state"]) == {"input", "entity"}


async def test_adjudication_below_the_minimum_nulls_the_field() -> None:
    text = "Ada Lovelace lives in London but works in Paris."
    script = person_script()
    script.clairic["city"] = ca("text", "London", find="London", confidence=0.7)
    script.reask = {"city": ca("text", "Paris", find="Paris", confidence=0.9)}
    script.adjudicate = {"city": pick("a", ["a", "b", "neither"], 0.5)}
    extractor = make_extractor([Person], script)
    record = (await extractor.ainvoke(text)).records[0]
    assert record.fields["city"].rejected_reason == "adjudication_failed"
    assert record.fields["city"].rejected_reason in [r.reason for r in record.rejected]
    assert person(record).city is None


async def test_reask_that_finds_nothing_is_adjudicated_against_no_value() -> None:
    script = low_confidence_script()
    script.reask = {"city": ca("text", None, p_null=0.99)}
    script.adjudicate = {"city": pick("b", ["a", "b", "neither"])}
    record = await run_person(script)
    (call,) = script.calls("adjudicate")
    assert call[1]["questions"]["adjudicate"]["criteria"]["b"] == (
        "The input states no value for this."
    )
    assert record.fields["city"].value is None
    assert record.fields["city"].warnings == ("adjudicated_to_null",)


async def test_reask_can_be_switched_off() -> None:
    script = low_confidence_script()
    record = await run_person(script, verify=VerifyConfig(reask_low_confidence=False))
    assert "reask" not in kinds(script.transport)
    assert record.fields["city"].value == "London"


async def test_reasked_values_go_through_attribution_again() -> None:
    text = "Ada Lovelace lives in London but works in Paris and is 36 years old."
    script = person_script()
    script.clairic["city"] = ca("text", "London", find="London", confidence=0.7)
    script.reask = {"city": ca("text", "Paris", find="Paris", confidence=0.9)}
    script.adjudicate = {"city": pick("b", ["a", "b", "neither"], 0.9)}
    script.attribution = {"attr.city": 0.05}
    extractor = make_extractor([Person], script)
    record = (await extractor.ainvoke(text)).records[0]
    (call,) = script.calls("attribution")
    assert (
        call[1]["questions"]["attr.city"]["instructions"]
        == 'Does `input` state that city of `entity.label` is "Paris", or does it '
        "follow by simple arithmetic or date math from figures the input states "
        "about `entity.label`?"
    )
    assert record.fields["city"].rejected_reason == "attribution"


# --------------------------------------------------------------------------- #
# Borderline null
# --------------------------------------------------------------------------- #


async def test_borderline_null_that_the_input_states_is_reasked() -> None:
    script = person_script()
    script.clairic["city"] = ca("text", None, p_null=0.5)
    script.null_check = {"nullcheck.city": 0.9}
    script.reask = {"city": ca("text", "London", find="London", p_null=0.2)}
    record = await run_person(script)
    assert person(record).city == "London"
    assert record.fields["city"].reask is True
    (check,) = script.calls("null_check")
    assert list(check[1]["questions"]) == ["nullcheck.city"]
    assert check[1]["questions"]["nullcheck.city"]["instructions"] == (
        "Does `input` state the Person's city?"
    )
    (reask,) = script.calls("reask")
    assert reask[1]["fields"]["city"]["null_threshold"] == 0.95
    assert (
        "city"
        in script.calls("attribution")[0][1]["questions"]["attr.city"]["instructions"]
    )


async def test_borderline_null_that_is_confirmed_stays_null() -> None:
    script = person_script()
    script.clairic["city"] = ca("text", None, p_null=0.5)
    script.null_check = {"nullcheck.city": 0.1}
    record = await run_person(script)
    assert person(record).city is None
    assert "reask" not in kinds(script.transport)


async def test_borderline_null_still_null_after_the_reask() -> None:
    script = person_script()
    script.clairic["city"] = ca("text", None, p_null=0.5)
    script.null_check = {"nullcheck.city": 0.9}
    script.reask = {"city": ca("text", None, p_null=0.9)}
    record = await run_person(script)
    assert person(record).city is None
    assert record.fields["city"].reask is True


async def test_borderline_non_null_value_the_input_does_not_state_is_nulled() -> None:
    class Guarded(BaseModel):
        name: str
        age: Optional[int] = Field(None, json_schema_extra=s1_field(null_threshold=0.9))

    script = Script()
    script.clairic = {
        "name": ca("text", "Ada", find="Ada"),
        "age": ca("number", 36, p_null=0.5),
    }
    script.null_check = {"nullcheck.age": 0.1}
    extractor = make_extractor([Guarded], script)
    record = (await extractor.ainvoke("Ada is here.")).records[0]
    assert record.value == Guarded(name="Ada")
    assert record.fields["age"].rejected_reason == "null_check"
    assert [(r.reason, r.p) for r in record.rejected] == [("null_check", 0.1)]
    assert "attr.age" not in script.calls("attribution")[0][1]["questions"]


async def test_borderline_value_between_the_confirm_thresholds_is_left_alone() -> None:
    class Guarded(BaseModel):
        age: Optional[int] = Field(None, json_schema_extra=s1_field(null_threshold=0.9))

    script = Script()
    script.clairic = {"age": ca("number", 36, p_null=0.5)}
    script.null_check = {"nullcheck.age": 0.5}
    extractor = make_extractor([Guarded], script)
    record = (await extractor.ainvoke("Ada is 36.")).records[0]
    assert record.value == Guarded(age=36)


async def test_borderline_band_is_inclusive_and_configurable() -> None:
    script = person_script()
    script.clairic["city"] = ca("text", None, p_null=0.25)
    script.null_check = {"nullcheck.city": 0.1}
    await run_person(script)
    assert len(script.calls("null_check")) == 1
    script = person_script()
    script.clairic["city"] = ca("text", None, p_null=0.25)
    await run_person(script, verify=VerifyConfig(null_band=(0.3, 0.7)))
    assert script.calls("null_check") == []
    script = person_script()
    script.clairic["city"] = ca("text", None, p_null=0.5)
    await run_person(script, verify=VerifyConfig(borderline_null=False))
    assert script.calls("null_check") == []


# --------------------------------------------------------------------------- #
# Clef answers
# --------------------------------------------------------------------------- #


class Flags(BaseModel):
    active: bool = False
    verified: Optional[bool] = None
    status: Literal["open", "closed"] = "open"


def flags(record: ResultRecord) -> "Flags":
    assert isinstance(record.value, Flags)
    return record.value


async def run_flags(clef: dict[str, Any]) -> ResultRecord:
    script = Script()
    script.clef = clef
    extractor = make_extractor([Flags], script)
    return (await extractor.ainvoke("The ticket was closed.")).records[0]


async def test_noul_inside_the_abstain_band_is_rejected() -> None:
    record = await run_flags(
        {
            "active": {"type": "noul", "noul": 0.55},
            "verified": pick("unstated", ["yes", "no", "unstated"]),
            "status": pick("closed", ["open", "closed", "unstated"]),
        }
    )
    assert record.fields["active"].rejected_reason == "abstain"
    assert record.value == Flags(active=False, status="closed")
    assert ("/active", "abstain") in [(r.path, r.reason) for r in record.rejected]


async def test_noul_outside_the_band_is_a_bool() -> None:
    record = await run_flags(
        {
            "active": {"type": "noul", "noul": 0.05},
            "verified": pick("yes", ["yes", "no", "unstated"]),
            "status": pick("closed", ["open", "closed", "unstated"]),
        }
    )
    assert record.fields["active"].value is False
    assert record.value == Flags(active=False, verified=True, status="closed")


async def test_unstated_on_defaulted_and_nullable_fields_gives_the_default() -> None:
    record = await run_flags(
        {
            "active": {"type": "noul", "noul": 0.9},
            "verified": pick("unstated", ["yes", "no", "unstated"]),
            "status": pick("unstated", ["open", "closed", "unstated"]),
        }
    )
    assert record.value == Flags(active=True, verified=None, status="open")
    for name in ("verified", "status"):
        answer = record.fields[name]
        assert answer.value is None
        assert answer.source == "default"
        assert answer.rejected_reason is None
        assert answer.warnings == ("unstated",)
    assert record.rejected == ()


async def test_clef_choice_below_the_probability_floor_is_rejected() -> None:
    record = await run_flags(
        {
            "active": {"type": "noul", "noul": 0.9},
            "verified": pick("unstated", ["yes", "no", "unstated"]),
            "status": {
                "type": "choice",
                "choice": "closed",
                "probabilities": {"open": 0.4, "closed": 0.5, "unstated": 0.1},
                "confidence": 0.9,
            },
        }
    )
    assert record.fields["status"].rejected_reason == "low_choice_prob"
    assert flags(record).status == "open"


async def test_clef_choice_with_low_confidence_is_rejected() -> None:
    record = await run_flags(
        {
            "active": {"type": "noul", "noul": 0.9},
            "verified": pick("unstated", ["yes", "no", "unstated"]),
            "status": {
                "type": "choice",
                "choice": "closed",
                "probabilities": {"open": 0.1, "closed": 0.8, "unstated": 0.1},
                "confidence": 0.3,
            },
        }
    )
    assert record.fields["status"].rejected_reason == "low_confidence"


async def test_list_elements_matching_no_option_are_dropped() -> None:
    class Tags(BaseModel):
        kinds: list[Literal["bug", "feature"]]

    script = Script()
    script.clairic = {
        "kinds": ca("list", ["bug", "wish"], find_all=["bug", "wish"], confidence=0.99)
    }
    script.clef = {
        "kinds.el0": pick("bug", ["bug", "feature", "none_of_these"]),
        "kinds.el1": pick("none_of_these", ["bug", "feature", "none_of_these"]),
    }
    extractor = make_extractor([Tags], script)
    text = "A bug and a wish."
    record = (await extractor.ainvoke(text)).records[0]
    assert record.value == Tags(kinds=["bug"])
    answer = record.fields["kinds"]
    assert answer.spans is not None
    assert [text[a:b] for a, b in answer.spans] == ["bug"]
    assert any("wish" in note for note in answer.warnings)


async def test_set_of_enum_elements_dedupes() -> None:
    class Tags(BaseModel):
        kinds: set[Literal["bug", "feature"]]

    script = Script()
    script.clairic = {"kinds": ca("list", ["bug", "bugs"], find_all=["bug", "bugs"])}
    script.clef = {
        "kinds.el0": pick("bug", ["bug", "feature", "none_of_these"]),
        "kinds.el1": pick("bug", ["bug", "feature", "none_of_these"]),
    }
    extractor = make_extractor([Tags], script)
    record = (await extractor.ainvoke("bug bugs")).records[0]
    assert record.value == Tags(kinds={"bug"})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_agree_text_numbers_lists() -> None:
    assert agree("London", "  london ")
    assert agree("Londonn", "London")
    assert not agree("London", "Paris")
    assert agree(5, 5.0)
    assert agree(dt.date(2020, 1, 1), dt.date(2020, 1, 1))
    assert not agree(dt.date(2020, 1, 1), dt.date(2020, 1, 2))
    assert agree(["a", "b"], ["B", "a"])
    assert not agree(["a", "b"], ["a"])
    assert not agree(["a", "a"], ["a", "b"])
    assert agree(None, None)
    assert not agree(None, "x")
    assert not agree(True, 1.5)


def test_display_formats_values_for_questions() -> None:
    assert display(dt.date(2020, 1, 2)) == "2020-01-02"
    assert display(dt.datetime(2020, 1, 2, 3, 4)) == "2020-01-02T03:04:00"
    assert display(["a", 1]) == "a; 1"
    from decimal import Decimal

    assert display(Decimal("1E+2")) == "100"


def test_accept_applies_the_answer_level_thresholds() -> None:
    plan = compile_schema(Person)
    city = plan.spec(("city",))
    assert accept(city, Resolved(value="x", p_null=0.0, confidence=0.9)) is None
    assert accept(city, Resolved(value=None, p_null=0.99)) is None
    reason = accept(city, Resolved(value="x", p_null=0.5, confidence=0.9))
    assert reason is not None and reason[0] == "p_null"
    assert accept(city, Resolved(value="x", p_null=0.5), null_threshold=0.95) is None
    low = accept(city, Resolved(value="x", p_null=0.0, confidence=0.1))
    assert low is not None and low[0] == "low_confidence"
    bad = accept(city, Resolved(ok=False, reason="unparseable", detail="nope"))
    assert bad == ("unparseable", "nope", None)


async def test_clef_questions_name_the_entity_at_the_description() -> None:
    class Review(BaseModel):
        best: bool = False  # "best" also occurs earlier in the compiled question
        rating: Literal["best", "worst"] = "best"

    script = Script()
    script.clef = {
        "best": {"type": "noul", "noul": 0.9},
        "rating": pick("best", ["best", "worst", "unstated"]),
    }
    extractor = make_extractor([Review], script)
    plan = extractor.plan_for(Review)
    before = plan.spec(("best",)).clef
    await extractor.ainvoke("The best.")
    (call,) = script.calls("clef_fields")
    assert call[1]["questions"]["best"]["instructions"] == (
        "Does the input say this is true: the Review's best?"
    )
    assert call[1]["questions"]["rating"]["instructions"] == (
        "Pick the option that best fits: the Review's rating"
    )
    assert plan.spec(("best",)).clef == before


# --------------------------------------------------------------------------- #
# A missing attribution answer is not a pass
# --------------------------------------------------------------------------- #


class Lossy:
    """Wraps a script's transport and leaves some Clef answers out.

    ``drop`` maps a question id to how many of its answers to leave out, counting
    calls that carried that question.
    """

    def __init__(self, script: Script, drop: dict[str, int]) -> None:
        self.script = script
        self.drop = dict(drop)

    async def run(
        self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
    ) -> Any:
        import dataclasses

        response = await self.script.transport.run(model, payload, meta=meta)
        if meta.kind != "attribution":
            return response
        body = dict(response.body)
        answers = dict(body["answers"])
        for qid in list(answers):
            if self.drop.get(qid, 0) > 0:
                self.drop[qid] -= 1
                del answers[qid]
        body["answers"] = answers
        return dataclasses.replace(response, body=body)


async def run_lossy(
    script: Script, drop: dict[str, int], **kwargs: Any
) -> tuple[ResultRecord, Script]:
    from trustcall_system_one import create_extractor

    extractor = create_extractor(
        [Person],
        transport=Lossy(script, drop),
        verify=RECHECK,
        **kwargs,
    )
    return (await extractor.ainvoke(PADDED)).records[0], script


async def test_a_missing_scalar_attribution_is_rechecked_not_passed() -> None:
    script = person_script()
    script.reask = {"city": ca("text", "London", find="London")}
    record, _ = await run_lossy(script, {"attr.city": 1})
    assert person(record).city == "London"  # the recheck answered and passed
    assert record.fields["city"].reask is True
    assert record.fields["city"].attribution_p == 0.95
    assert len(script.calls("attribution")) == 2


async def test_a_scalar_attribution_that_stays_missing_is_dropped() -> None:
    script = person_script()
    script.reask = {"city": ca("text", "London", find="London")}
    record, _ = await run_lossy(script, {"attr.city": 2})
    assert person(record).city is None
    assert record.fields["city"].rejected_reason == "attribution_missing"
    assert [(r.path, r.reason) for r in record.rejected] == [
        ("/city", "attribution_missing")
    ]


async def test_a_missing_attribution_with_a_reask_that_finds_nothing_is_dropped() -> (
    None
):
    script = person_script()
    script.reask = {"city": ca("text", None, p_null=0.99)}
    record, _ = await run_lossy(script, {"attr.city": 1})
    assert record.fields["city"].rejected_reason == "attribution"


async def test_a_missing_element_attribution_drops_the_element_as_missing() -> None:
    class Notes(BaseModel):
        notes: list[str]

    from trustcall_system_one import create_extractor

    script = Script()
    script.clairic = {"notes": ca("list", ["red", "green"], find_all=["red", "green"])}
    script.reask = {"notes": ca("list", ["red", "green"], find_all=["red", "green"])}
    extractor = create_extractor(
        [Notes],
        transport=Lossy(script, {"attr.notes.1": 2}),
        verify=VerifyConfig(borderline_null=False),
    )
    record = (await extractor.ainvoke("red green")).records[0]
    assert record.value == Notes(notes=["red"])
    assert [(r.path, r.reason) for r in record.rejected] == [
        ("/notes/1", "attribution_missing")
    ]
    assert record.fields["notes"].attribution_p is not None


async def test_a_missing_element_attribution_is_none_never_nan() -> None:
    class Notes(BaseModel):
        notes: list[str]

    from trustcall_system_one import create_extractor

    script = Script()
    script.clairic = {"notes": ca("list", ["red", "green"], find_all=["red", "green"])}
    script.reask = {"notes": ca("list", ["red", "green"], find_all=["red", "green"])}
    extractor = create_extractor(
        [Notes],
        transport=Lossy(script, {"attr.notes.1": 2}),
        verify=VerifyConfig(borderline_null=False),
    )
    record = (await extractor.ainvoke("red green")).records[0]
    assert record.fields["notes"].attribution_p == (0.95, None)
    # no NaN anywhere on the public result
    assert not any(
        isinstance(p, float) and math.isnan(p)
        for answer in record.fields.values()
        for p in (
            answer.attribution_p
            if isinstance(answer.attribution_p, tuple)
            else (answer.attribution_p,)
        )
    )


async def test_every_element_missing_rejects_the_field_as_missing() -> None:
    class Notes(BaseModel):
        notes: Optional[list[str]] = None

    from trustcall_system_one import create_extractor

    script = Script()
    script.clairic = {"notes": ca("list", ["red"], find_all=["red"])}
    script.reask = {"notes": ca("list", ["red"], find_all=["red"])}
    extractor = create_extractor(
        [Notes],
        transport=Lossy(script, {"attr.notes.0": 2}),
        verify=VerifyConfig(borderline_null=False),
    )
    record = (await extractor.ainvoke("red")).records[0]
    assert record.fields["notes"].rejected_reason == "attribution_missing"


# --------------------------------------------------------------------------- #
# Dropped elements: one index space
# --------------------------------------------------------------------------- #


async def test_every_dropped_list_element_is_rejected_at_its_original_position() -> (
    None
):
    class Numbers(BaseModel):
        nums: list[int]

    script = Script()
    script.clairic = {
        "nums": ca("list", ["1", "x", "3", "4"], find_all=["1", "x", "3", "4"])
    }
    script.attribution = {"attr.nums.1": 0.0}  # the element "3" (kept index 1)
    extractor = make_extractor([Numbers], script)
    record = (await extractor.ainvoke("1 x 3 4")).records[0]
    assert record.value == Numbers(nums=[1, 4])
    by_path = {r.path: r for r in record.rejected}
    assert set(by_path) == {"/nums/1", "/nums/2"}
    # "x" failed to parse (position 1 of the model's list); "3" is position 2 of
    # it, although it was the second element left after parsing
    assert by_path["/nums/1"].reason == "unparseable"
    assert by_path["/nums/2"].reason == "attribution"
    assert by_path["/nums/2"].detail == "3"


async def test_enum_list_drops_share_the_index_space() -> None:
    class Kinds(BaseModel):
        kinds: list[Literal["bug", "feature", "docs"]]

    script = Script()
    script.clairic = {
        "kinds": ca(
            "list",
            ["bug", "wish", "feature", "docs"],
            find_all=["bug", "wish", "feature", "docs"],
        )
    }
    options = ["bug", "feature", "docs", "none_of_these"]
    script.clef = {
        "kinds.el0": pick("bug", options),
        "kinds.el1": pick("none_of_these", options),
        "kinds.el2": pick("feature", options),
        "kinds.el3": pick("docs", options),
    }
    script.attribution = {"attr.kinds.2": 0.0}  # "docs": third element left
    extractor = make_extractor([Kinds], script)
    record = (await extractor.ainvoke("bug wish feature docs")).records[0]
    assert record.value == Kinds(kinds=["bug", "feature"])
    by_path = {r.path: r.reason for r in record.rejected}
    assert by_path == {"/kinds/1": "none_of_these", "/kinds/3": "attribution"}


# --------------------------------------------------------------------------- #
# Templating rebuilds questions from their parts
# --------------------------------------------------------------------------- #


async def test_a_clipped_clef_question_is_rebuilt_not_patched() -> None:
    from trustcall_system_one import Limits

    long_description = "whether the review " + "really " * 12 + "praises the product"

    class Review(BaseModel):
        praises: bool = Field(False, description=long_description)

    script = Script()
    script.clef = {"praises": {"type": "noul", "noul": 0.9}}
    extractor = make_extractor(
        [Review], script, limits=Limits(max_instruction_chars=60)
    )
    compiled = extractor.plan_for(Review).spec(("praises",)).clef
    assert compiled is not None
    assert long_description not in str(compiled.instructions)  # clipped by the compiler
    await extractor.ainvoke("It praises.")
    (call,) = script.calls("clef_fields")
    text = call[1]["questions"]["praises"]["instructions"]
    assert len(text) <= 60
    assert text.startswith("Does the input say this is true: the Review's whether")
    assert text.endswith("...")


async def test_the_template_wraps_the_description_not_the_context_or_hints() -> None:
    from trustcall_system_one import Limits

    # a name over 64 characters is shortened, which adds parent context
    definitions: dict[str, Any] = {
        "f" * 70: (Optional[int], Field(None, ge=0, description="the count"))
    }
    Wide = create_model("Wide", **definitions)
    plan = compile_schema(Wide)
    spec = plan.specs[0]
    assert spec.context == "Wide" and spec.hints == ("Must be at least 0.",)
    script = Script()
    script.clairic = {spec.wire_name: ca("number", 3)}
    extractor = make_extractor([Wide], script, limits=Limits())
    await extractor.ainvoke("The count is 3.")
    (call,) = script.calls("extract")
    assert call[1]["fields"][spec.wire_name]["instructions"] == (
        "Wide: the Wide's count. Must be at least 0."
    )


async def test_reasked_codes_that_differ_by_a_digit_do_not_agree() -> None:
    class Book(BaseModel):
        isbn: Optional[str] = None

    text = "ISBN 978-3-16-148410-0 (once misprinted as 978-3-16-148410-1)."
    script = Script()
    script.clairic = {
        "isbn": ca(
            "text", "978-3-16-148410-0", find="978-3-16-148410-0", confidence=0.6
        )
    }
    script.reask = {"isbn": ca("text", "978-3-16-148410-1", find="978-3-16-148410-1")}
    script.adjudicate = {"isbn": pick("a", ["a", "b", "neither"])}
    extractor = make_extractor([Book], script)
    record = (await extractor.ainvoke(text)).records[0]
    assert isinstance(record.value, Book) and record.value.isbn == "978-3-16-148410-0"
    assert len(script.calls("adjudicate")) == 1  # they disagree: Clef decides


async def test_reasked_near_identical_text_still_agrees() -> None:
    class Who(BaseModel):
        name: Optional[str] = None

    text = "Ada Lovelace wrote it; Ada Lovelaces notes."
    script = Script()
    script.clairic = {
        "name": ca("text", "Ada Lovelace", find="Ada Lovelace", confidence=0.6)
    }
    script.reask = {"name": ca("text", "Ada Lovelaces", find="Ada Lovelaces")}
    record = (await make_extractor([Who], script).ainvoke(text)).records[0]
    assert isinstance(record.value, Who) and record.value.name is not None
    assert script.calls("adjudicate") == []


def test_a_re_ask_of_a_list_field_does_not_put_a_null_threshold_on_the_wire() -> None:
    """``model_copy`` skips validation, so the copy has to leave it out itself."""
    from trustcall_system_one._run import Run
    from trustcall_system_one._verify import Ctx
    from trustcall_system_one.testing import FakeTransport

    class Tagged(BaseModel):
        """Something with tags."""

        name: str
        tags: list[str] = []

    plan = compile_schema(Tagged)
    ctx = Ctx(
        run=Run(transport=FakeTransport(lambda *a: None)),
        plan=plan,
        text="x",
        label="the Tagged",
    )
    tags, name = plan.spec(("tags",)), plan.spec(("name",))
    assert ctx.clairic_field(tags, null_threshold=0.95).null_threshold is None
    assert ctx.clairic_field(tags).null_threshold is None
    assert ctx.clairic_field(name, null_threshold=0.95).null_threshold == 0.95


def test_values_are_shown_to_the_attribution_question_the_way_they_read() -> None:
    assert display(1080.0) == "1080"  # a float total is not "1080.0"
    assert display(12.5) == "12.5"
    assert display(float("inf")) == "inf"
    assert display([1.0, 2.5]) == "1; 2.5"


async def test_attribution_accepts_what_the_input_directly_implies() -> None:
    """A live run rejected total=1080, stated as parts (4 x $250 + $80), at p=0.04."""

    class Invoice(BaseModel):
        invoice_number: str
        total: float = Field(description="the amount due including shipping")

    text = "Invoice INV-2291: 4 pallets at $250.00 each, shipping $80.00."
    script = Script()
    script.clairic = {
        "invoice_number": ca("text", "INV-2291", find="INV-2291"),
        "total": ca("number", 1080),  # computed: stated only as its parts
    }
    result = await make_extractor([Invoice], script).ainvoke(text)
    (call,) = script.calls("attribution")
    question = call[1]["questions"]["attr.total"]
    assert question["instructions"] == (
        "Does `input` state or ask for the amount due including shipping of "
        '`entity.label` to be "1080", or does it follow by simple arithmetic or '
        "date math from figures the input states about `entity.label`?"
    )
    assert "simple arithmetic or date math" in question["criteria"]["true"]
    assert "concerns another entity" in question["criteria"]["false"]
    assert result.records[0].valid
