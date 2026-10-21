"""Tests for converting model answers into Python values."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import datetime as dt
import enum
from decimal import Decimal
from typing import Annotated, Any, Literal, Optional, Union

import pytest
from pydantic import Field, create_model

from trustcall_system_one import FieldSpec, GateConfig, compile_schema, s1_field
from trustcall_system_one._coerce import Coerced, coerce
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ClairicAnswer,
    NoulAnswer,
    ScoreAnswer,
)


def spec_for(annotation: Any, default: Any = ..., **field_kwargs: Any) -> FieldSpec:
    model = create_model("M", f=(annotation, Field(default, **field_kwargs)))
    return compile_schema(model).specs[0]


def answer(type_: str, value: Any, **extra: Any) -> ClairicAnswer:
    return ClairicAnswer(type=type_, value=value, **extra)


class Level(enum.Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


# --------------------------------------------------------------------------- #
# Clairic values
# --------------------------------------------------------------------------- #


def test_null_value_is_ok_and_never_falls_back_to_the_guess() -> None:
    result = coerce(
        spec_for(str),
        answer("text", None, guess="maybe", p_null=0.9, confidence=0.2),
    )
    assert result == Coerced(
        value=None, ok=True, reason=None, p_null=0.9, confidence=0.2, guess="maybe"
    )


def test_text() -> None:
    result = coerce(spec_for(str), answer("text", "Ada", span=[3, 6], confidence=0.9))
    assert (result.value, result.ok, result.span, result.confidence) == (
        "Ada",
        True,
        (3, 6),
        0.9,
    )


def test_text_accepts_scalars_but_not_containers() -> None:
    assert coerce(spec_for(str), answer("text", 12)).value == "12"
    bad = coerce(spec_for(str), answer("text", {"a": 1}))
    assert bad.ok is False and bad.value is None and bad.reason


@pytest.mark.parametrize(
    ("annotation", "raw", "expected"),
    [
        (int, 3, 3),
        (int, 3.0, 3),
        (int, "7", 7),
        (float, 3, 3.0),
        (float, 2.5, 2.5),
        (float, "2.5", 2.5),
        (Decimal, 1.1, Decimal("1.1")),
        (Decimal, 7, Decimal("7")),
        (Decimal, "12.50", Decimal("12.50")),
    ],
)
def test_numbers(annotation: Any, raw: Any, expected: Any) -> None:
    result = coerce(spec_for(annotation), answer("number", raw))
    assert result.ok
    assert result.value == expected
    assert type(result.value) is type(expected)


@pytest.mark.parametrize("raw", [3.5, "3.5", True, "abc", float("inf"), [1]])
def test_integer_field_rejects_non_integral_numbers(raw: Any) -> None:
    result = coerce(spec_for(int), answer("number", raw))
    assert result.ok is False
    assert result.value is None
    assert result.reason


def test_non_integral_reason_names_the_problem() -> None:
    result = coerce(spec_for(int), answer("number", 3.5))
    assert result.reason is not None
    assert "whole number" in result.reason


def test_bounded_integer_estimate_is_rounded() -> None:
    model = create_model(
        "M", f=(Annotated[int, Field(ge=0, le=10)], Field(..., description="x"))
    )
    from trustcall_system_one import Limits

    plan = compile_schema(model, limits=Limits(estimate_for_bounded_numbers=True))
    assert coerce(plan.specs[0], answer("estimate", 6.6, mean=6.6)).value == 7


def test_dates_times_and_datetimes() -> None:
    assert coerce(spec_for(dt.date), answer("date", "2024-02-29")).value == dt.date(
        2024, 2, 29
    )
    assert coerce(spec_for(dt.time), answer("time", "13:45:10")).value == dt.time(
        13, 45, 10
    )
    assert coerce(spec_for(dt.time), answer("time", "13:45Z")).value == dt.time(
        13, 45, tzinfo=dt.timezone.utc
    )
    spec = spec_for(dt.datetime)
    assert coerce(spec, answer("datetime", "2024-02-29T10:30:00")).value == (
        dt.datetime(2024, 2, 29, 10, 30)
    )
    assert coerce(spec, answer("datetime", "2024-02-29T10:30:00Z")).value == (
        dt.datetime(2024, 2, 29, 10, 30, tzinfo=dt.timezone.utc)
    )
    assert coerce(spec, answer("datetime", "2024-02-29T10:30:00+02:00")).value == (
        dt.datetime(2024, 2, 29, 10, 30, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    )


def test_a_date_field_accepts_a_datetime_string() -> None:
    result = coerce(spec_for(dt.date), answer("date", "2024-02-29T00:00:00Z"))
    assert result.value == dt.date(2024, 2, 29)


@pytest.mark.parametrize(
    ("annotation", "type_", "raw"),
    [
        (dt.date, "date", "29/02/2024"),
        (dt.date, "date", "2023-02-29"),
        (dt.date, "date", 20240229),
        (dt.time, "time", "25:00"),
        (dt.time, "time", "noon"),
        (dt.datetime, "datetime", "yesterday"),
        (dt.datetime, "datetime", 12345),
    ],
)
def test_unparseable_temporal_values_are_not_ok(
    annotation: Any, type_: str, raw: Any
) -> None:
    result = coerce(spec_for(annotation), answer(type_, raw))
    assert result.ok is False
    assert result.value is None
    assert result.reason


@pytest.mark.parametrize(
    ("unit", "raw", "expected"),
    [
        ("seconds", 90, dt.timedelta(seconds=90)),
        ("minutes", 1.5, dt.timedelta(seconds=90)),
        ("hours", 2, dt.timedelta(hours=2)),
        ("days", 3, dt.timedelta(days=3)),
        ("weeks", 1, dt.timedelta(weeks=1)),
        ("months", 2, dt.timedelta(days=2 * 30.436875)),
        ("years", 1, dt.timedelta(days=365.2425)),
    ],
)
def test_durations_follow_the_spec_unit(unit: str, raw: Any, expected: Any) -> None:
    spec = spec_for(dt.timedelta, json_schema_extra=s1_field(unit=unit))
    result = coerce(spec, answer("duration", raw))
    assert result.ok
    assert result.value == expected


def test_default_duration_unit_is_seconds_and_text_forms_parse() -> None:
    spec = spec_for(dt.timedelta)
    assert coerce(spec, answer("duration", 45)).value == dt.timedelta(seconds=45)
    assert coerce(spec, answer("duration", "PT1H30M")).value == dt.timedelta(minutes=90)
    assert coerce(spec, answer("duration", "oops")).ok is False
    assert coerce(spec, answer("duration", float("nan"))).ok is False


def test_lossy_duration_units_are_flagged_on_the_spec() -> None:
    assert (
        "lossy_duration"
        in spec_for(dt.timedelta, json_schema_extra=s1_field(unit="months")).warnings
    )


def test_huge_duration_is_not_ok_rather_than_raising() -> None:
    spec = spec_for(dt.timedelta, json_schema_extra=s1_field(unit="years"))
    assert coerce(spec, answer("duration", 10**9)).ok is False


# --------------------------------------------------------------------------- #
# Lists
# --------------------------------------------------------------------------- #


def test_text_and_number_lists() -> None:
    assert coerce(spec_for(list[str]), answer("list", ["a", "b"])).value == ["a", "b"]
    assert coerce(spec_for(list[int]), answer("list", [1, 2.0, "3"])).value == [1, 2, 3]
    assert coerce(spec_for(list[float]), answer("list", [1, 2.5])).value == [1.0, 2.5]
    assert coerce(spec_for(list[Decimal]), answer("list", [1.5, "2"])).value == [
        Decimal("1.5"),
        Decimal("2"),
    ]
    assert coerce(spec_for(list[dt.date]), answer("list", ["2024-01-02"])).value == [
        dt.date(2024, 1, 2)
    ]


def test_list_drops_unparseable_elements_and_says_so() -> None:
    result = coerce(spec_for(list[int]), answer("list", [1, 2.5, None, "x", 4]))
    assert result.ok
    assert result.value == [1, 4]
    assert result.reason is not None
    assert result.reason.startswith("dropped 3 element(s)")


def test_list_that_is_not_a_list_is_not_ok() -> None:
    result = coerce(spec_for(list[str]), answer("list", "a, b"))
    assert result.ok is False


def test_empty_list_is_a_valid_value() -> None:
    result = coerce(spec_for(list[str]), answer("list", []))
    assert result.ok and result.value == [] and result.reason is None


def test_sets_dedupe_keeping_first_occurrence() -> None:
    result = coerce(spec_for(set[str]), answer("list", ["b", "a", "b", "c", "a"]))
    assert result.value == ["b", "a", "c"]
    result = coerce(spec_for(frozenset[int]), answer("list", [3, 1, 3.0, 2, 1]))
    assert result.value == [3, 1, 2]
    assert coerce(spec_for(list[str]), answer("list", ["b", "b"])).value == ["b", "b"]


def test_lossy_datetime_time_and_timedelta_lists_parse_per_element() -> None:
    result = coerce(
        spec_for(list[dt.datetime]),
        answer("list", ["2024-01-02T03:04:05Z", "later", "2024-02-03T00:00:00"]),
    )
    assert result.value == [
        dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 2, 3),
    ]
    assert result.ok
    assert result.reason is not None and "dropped 1" in result.reason
    result = coerce(
        spec_for(list[dt.time]), answer("list", ["09:30", "9am", "17:00:15"])
    )
    assert result.value == [dt.time(9, 30), dt.time(17, 0, 15)]
    assert "dropped 1" in (result.reason or "")


def test_timedelta_list_parses_text_durations() -> None:
    spec = spec_for(list[dt.timedelta], json_schema_extra=s1_field(unit="minutes"))
    result = coerce(
        spec,
        answer(
            "list",
            [
                "2 hours",
                "PT30M",
                "1:30:00",
                "1h 15m",
                "45",
                10,
                "1 hour and 5 mins",
                "soon",
            ],
        ),
    )
    assert result.value == [
        dt.timedelta(hours=2),
        dt.timedelta(minutes=30),
        dt.timedelta(hours=1, minutes=30),
        dt.timedelta(hours=1, minutes=15),
        dt.timedelta(minutes=45),
        dt.timedelta(minutes=10),
        dt.timedelta(hours=1, minutes=5),
    ]
    assert "dropped 1" in (result.reason or "")


def test_list_of_literals_keeps_raw_text_for_the_per_element_choice() -> None:
    spec = spec_for(list[Literal["a", "b"]])
    assert coerce(spec, answer("list", ["a", "zzz"])).value == ["a", "zzz"]


def test_a_set_of_datetimes_dedupes_after_parsing() -> None:
    result = coerce(
        spec_for(set[dt.datetime]),
        answer("list", ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00+00:00"]),
    )
    assert result.value == [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)]


# --------------------------------------------------------------------------- #
# Spans and metadata
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([0, 4], (0, 4)),
        ([5, 5], (5, 5)),
        ([3, 2], None),
        ([-1, 2], None),
        ([1, 2, 3], None),
        ([1], None),
        ([], None),
        ([True, 2], None),
        (None, None),
    ],
)
def test_span_is_accepted_only_when_well_formed(raw: Any, expected: Any) -> None:
    result = coerce(spec_for(str), answer("text", "x", span=raw))
    assert result.span == expected
    assert result.ok


def test_floats_in_a_span_are_dropped_by_the_wire_model() -> None:
    result = coerce(spec_for(str), answer("text", "x", span=[1.5, 2]))
    assert result.span is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([[0, 2], [5, 9]], ((0, 2), (5, 9))),
        ([[0, 2], [9, 5]], None),
        ([[0, 2], [5]], None),
        ([[0, 2, 3]], None),
        ([[-1, 0]], None),
        ([], None),
        (None, None),
    ],
)
def test_spans_need_every_pair_to_be_well_formed(raw: Any, expected: Any) -> None:
    result = coerce(spec_for(list[str]), answer("list", ["a", "b"], spans=raw))
    assert result.spans == expected


def test_metadata_is_carried_on_failures_and_nulls() -> None:
    result = coerce(
        spec_for(int),
        answer("number", 2.5, p_null=0.1, confidence=0.8, guess=2, span=[0, 3]),
    )
    assert result.ok is False
    assert (result.p_null, result.confidence, result.guess, result.span) == (
        0.1,
        0.8,
        2,
        (0, 3),
    )


def test_union_text_value_is_returned_as_text() -> None:
    result = coerce(spec_for(Union[int, str]), answer("text", "forty-two"))
    assert result.value == "forty-two"


def test_clairic_answer_for_a_clef_or_skipped_spec_is_not_ok() -> None:
    for annotation in (bool, Literal["a", "b"], Any):
        result = coerce(spec_for(annotation), answer("text", "x"))
        assert result.ok is False
        assert result.reason


# --------------------------------------------------------------------------- #
# Clef
# --------------------------------------------------------------------------- #


def noul(p: float) -> NoulAnswer:
    return NoulAnswer(type="noul", noul=p)


@pytest.mark.parametrize(
    ("p", "value", "ok"),
    [
        (1.0, True, True),
        (0.9, True, True),
        (0.65, True, True),
        (0.64, None, False),
        (0.5, None, False),
        (0.36, None, False),
        (0.35, False, True),
        (0.1, False, True),
        (0.0, False, True),
    ],
)
def test_noul_uses_the_margin_around_one_half(p: float, value: Any, ok: bool) -> None:
    result = coerce(spec_for(bool), noul(p))
    assert (result.value, result.ok) == (value, ok)
    assert result.score == p
    if not ok:
        assert result.reason == "abstain"


def test_noul_margin_comes_from_the_gate_config() -> None:
    spec = spec_for(bool)
    assert coerce(spec, noul(0.6)).ok is False
    assert coerce(spec, noul(0.6), GateConfig(noul_margin=0.05)).value is True
    assert coerce(spec, noul(0.4), gates=GateConfig(noul_margin=0.05)).value is False
    wide = GateConfig(noul_margin=0.4)
    assert coerce(spec, noul(0.85), wide).ok is False
    assert coerce(spec, noul(0.9), wide).value is True


def test_noul_with_zero_margin_still_abstains_at_one_half() -> None:
    gates = GateConfig(noul_margin=0)
    spec = spec_for(bool)
    assert coerce(spec, noul(0.5), gates).ok is False
    assert coerce(spec, noul(0.51), gates).value is True
    assert coerce(spec, noul(0.49), gates).value is False


def choice(name: str, **probabilities: float) -> ChoiceAnswer:
    return ChoiceAnswer(
        type="choice", choice=name, probabilities=probabilities, confidence=0.8
    )


def test_choice_maps_the_option_id_back_to_the_value() -> None:
    spec = spec_for(Literal["in progress", "done"])
    result = coerce(spec, choice("in_progress", in_progress=0.7, done=0.3))
    assert (result.value, result.ok, result.confidence, result.score) == (
        "in progress",
        True,
        0.8,
        0.7,
    )


def test_choice_over_an_enum_returns_the_enum_value() -> None:
    result = coerce(spec_for(Level), choice("mid", low=0.1, mid=0.8, high=0.1))
    assert result.value == "mid"


def test_unknown_choice_is_not_ok() -> None:
    result = coerce(spec_for(Literal["a", "b"]), choice("c", a=0.1, b=0.1, c=0.8))
    assert result.ok is False
    assert result.value is None
    assert result.reason is not None and "'c'" in result.reason


def test_optional_bool_choice_can_say_unstated() -> None:
    spec = spec_for(Optional[bool])
    yes = coerce(spec, choice("yes", yes=0.9, no=0.05, unstated=0.05))
    no = coerce(spec, choice("no", yes=0.05, no=0.9, unstated=0.05))
    unstated = coerce(spec, choice("unstated", yes=0.1, no=0.1, unstated=0.8))
    assert (yes.value, yes.ok) == (True, True)
    assert (no.value, no.ok) == (False, True)
    assert (unstated.value, unstated.ok) == (None, True)


def test_union_choice_returns_the_member_type_name() -> None:
    model = create_model("M", f=(Union[int, str], Field(..., description="x")))
    spec = compile_schema(model).specs[0]
    result = coerce(spec, choice("int", int=0.9, str=0.1))
    assert result.value == "int"


def score(value: float, **probabilities: float) -> ScoreAnswer:
    return ScoreAnswer(
        type="score",
        score=value,
        legend={},
        probabilities=probabilities,
        confidence=0.6,
    )


def test_score_takes_the_most_probable_level_and_keeps_the_weighted_score() -> None:
    spec = spec_for(Level, json_schema_extra=s1_field(clef="score"))
    result = coerce(spec, score(1.3, **{"0": 0.1, "1": 0.5, "2": 0.4}))
    assert result.value == "mid"
    assert result.score == 1.3
    assert result.confidence == 0.6
    assert result.ok


def test_score_ties_go_to_the_lower_level() -> None:
    spec = spec_for(Level, json_schema_extra=s1_field(clef="score"))
    assert coerce(spec, score(1.0, **{"0": 0.4, "1": 0.4, "2": 0.2})).value == "low"


def test_score_without_probabilities_rounds_the_weighted_score() -> None:
    spec = spec_for(Level, json_schema_extra=s1_field(clef="score"))
    assert coerce(spec, score(1.6)).value == "high"


def test_score_with_an_unknown_level_is_not_ok() -> None:
    spec = spec_for(Level, json_schema_extra=s1_field(clef="score"))
    result = coerce(spec, score(5.0, **{"7": 1.0}))
    assert result.ok is False
    assert result.reason is not None and "'7'" in result.reason


@pytest.mark.parametrize(
    ("annotation", "wrong"),
    [
        (bool, choice("a", a=1.0)),
        (bool, score(1.0, **{"0": 1.0})),
        (Literal["a", "b"], noul(0.9)),
        (Literal["a", "b"], score(1.0, **{"0": 1.0})),
        (str, noul(0.9)),
        (str, choice("a", a=1.0)),
        (Any, noul(0.9)),
    ],
)
def test_wrong_answer_type_is_not_ok_and_never_raises(
    annotation: Any, wrong: Any
) -> None:
    result = coerce(spec_for(annotation), wrong)
    assert result.ok is False
    assert result.value is None
    assert result.reason


def test_score_answer_for_a_choice_spec_is_not_ok() -> None:
    spec = spec_for(Level)
    assert coerce(spec, score(1.0, **{"0": 1.0})).ok is False
    score_spec = spec_for(Level, json_schema_extra=s1_field(clef="score"))
    assert coerce(score_spec, choice("low", low=1.0)).ok is False


def test_coerced_is_frozen() -> None:
    result = coerce(spec_for(str), answer("text", "x"))
    with pytest.raises(AttributeError):
        result.value = "y"  # type: ignore[misc]
