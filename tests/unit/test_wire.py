"""Tests for the wire models: request constraints and response parsing."""

import base64
from typing import Any

import pytest
from pydantic import ValidationError

from trustcall_system_one import CLAIRIC, CLEF, CLEF_FLASH, ModelResponseError
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ClairicField,
    ClairicRequest,
    ClefRequest,
    NoulAnswer,
    ScoreAnswer,
    body_model,
    parse_clairic,
    parse_clef,
)

TEXT = {"type": "text"}
IMG = "data:image/png;base64,AAAA"


def clairic(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "clairic", "state": "hello", "fields": {"a": TEXT}}
    body.update(overrides)
    return body


def noul_q(**overrides: Any) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul", "instructions": "Is it?"}
    q.update(overrides)
    return q


def clef(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "clef",
        "state": "hello",
        "questions": {"q": noul_q()},
    }
    body.update(overrides)
    return body


def choice_q(n: int) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {f"opt{i}": None for i in range(n)},
    }


def score_q(n: int) -> dict[str, Any]:
    return {
        "type": "score",
        "instructions": "Rate",
        "criteria": [f"level {i}" for i in range(n)],
    }


# --------------------------------------------------------------------------- #
# Clairic request
# --------------------------------------------------------------------------- #


def test_clairic_minimal_request_is_valid() -> None:
    ClairicRequest.model_validate(clairic())


@pytest.mark.parametrize("model", ["clairic", CLAIRIC])
def test_clairic_model_selector_accepts_both_forms(model: str) -> None:
    ClairicRequest.model_validate(clairic(model=model))


@pytest.mark.parametrize("model", ["clef", "clairic ", "@cf/cloudflare/clef", ""])
def test_clairic_model_selector_rejects_others(model: str) -> None:
    with pytest.raises(ValidationError):
        ClairicRequest.model_validate(clairic(model=model))


@pytest.mark.parametrize("name", ["a" * 64, "_x", "a.b-c_d", "A1", "a-", "a."])
def test_clairic_field_name_accepted(name: str) -> None:
    ClairicRequest.model_validate(clairic(fields={name: TEXT}))


@pytest.mark.parametrize("name", ["a" * 65, "1a", "", "-a", ".a", "a b", "a\n", "é"])
def test_clairic_field_name_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        ClairicRequest.model_validate(clairic(fields={name: TEXT}))


def test_clairic_requires_at_least_one_field() -> None:
    with pytest.raises(ValidationError):
        ClairicRequest.model_validate(clairic(fields={}))


def test_clairic_list_requires_items() -> None:
    with pytest.raises(ValidationError, match="items"):
        ClairicField.model_validate({"type": "list"})
    assert (
        ClairicField.model_validate({"type": "list", "items": "text"}).items == "text"
    )


@pytest.mark.parametrize("kind", ["estimate", "forecast"])
def test_estimate_and_forecast_require_min_and_max(kind: str) -> None:
    for spec in ({"type": kind}, {"type": kind, "min": 0}, {"type": kind, "max": 1}):
        with pytest.raises(ValidationError, match="min and max"):
            ClairicField.model_validate(spec)
    ClairicField.model_validate({"type": kind, "min": 0, "max": 10})


def test_estimate_bounds_accept_numbers_and_iso_dates() -> None:
    ClairicField.model_validate({"type": "estimate", "min": 0.5, "max": 3})
    ClairicField.model_validate(
        {"type": "estimate", "min": "2026-01-01", "max": "2026-12-31"}
    )
    with pytest.raises(ValidationError, match="ISO date"):
        ClairicField.model_validate({"type": "estimate", "min": "soon", "max": 3})


def test_integer_bounds_keep_their_type() -> None:
    spec = ClairicField.model_validate({"type": "estimate", "min": 0, "max": 10})
    assert isinstance(spec.min, int)
    assert spec.to_payload() == {"type": "estimate", "min": 0, "max": 10}


@pytest.mark.parametrize("decimals", [0, 6])
def test_decimals_boundary_accepted(decimals: int) -> None:
    ClairicField.model_validate({"type": "number", "decimals": decimals})


@pytest.mark.parametrize("decimals", [-1, 7])
def test_decimals_past_boundary_rejected(decimals: int) -> None:
    with pytest.raises(ValidationError):
        ClairicField.model_validate({"type": "number", "decimals": decimals})


@pytest.mark.parametrize("threshold", [0.0001, 0.5, 1])
def test_null_threshold_accepted(threshold: float) -> None:
    ClairicField.model_validate({"type": "text", "null_threshold": threshold})


@pytest.mark.parametrize("kind", ["text", "number", "date", "time", "datetime"])
def test_null_threshold_is_accepted_on_the_other_types(kind: str) -> None:
    ClairicField.model_validate({"type": kind, "null_threshold": 0.5})


def test_a_list_field_takes_no_null_threshold() -> None:
    """Observed live (error 3030, HTTP 422); the published schema does not say so."""
    with pytest.raises(
        ValidationError, match="list fields don't take 'null_threshold'"
    ):
        ClairicField.model_validate(
            {"type": "list", "items": "text", "null_threshold": 0.5}
        )
    ClairicField.model_validate({"type": "list", "items": "text"})
    # so a whole request is refused before it is sent
    with pytest.raises(ValidationError, match="null_threshold"):
        ClairicRequest.model_validate(
            clairic(
                fields={"f": {"type": "list", "items": "text", "null_threshold": 0.3}}
            )
        )


@pytest.mark.parametrize("threshold", [0, -0.1, 1.01])
def test_null_threshold_rejected(threshold: float) -> None:
    with pytest.raises(ValidationError):
        ClairicField.model_validate({"type": "text", "null_threshold": threshold})


def test_forecast_horizon_and_interval() -> None:
    base = {"type": "forecast", "min": 0, "max": 1}
    ClairicField.model_validate({**base, "horizon": 1, "interval": "quarter"})
    ClairicField.model_validate({**base, "horizon": 64})
    for horizon in (0, 65):
        with pytest.raises(ValidationError):
            ClairicField.model_validate({**base, "horizon": horizon})
    with pytest.raises(ValidationError):
        ClairicField.model_validate({**base, "interval": "decade"})


@pytest.mark.parametrize("extra", [{"horizon": 3}, {"interval": "day"}])
def test_horizon_and_interval_only_for_forecast(extra: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="only allowed for forecast"):
        ClairicField.model_validate({"type": "estimate", "min": 0, "max": 1, **extra})


def test_duration_unit_validated() -> None:
    for unit in ("seconds", "days", "years"):
        ClairicField.model_validate({"type": "duration", "unit": unit})
    ClairicField.model_validate({"type": "duration"})
    with pytest.raises(ValidationError, match="duration unit"):
        ClairicField.model_validate({"type": "duration", "unit": "fortnights"})
    # Other types use ``unit`` as a free label.
    ClairicField.model_validate({"type": "number", "unit": "USD"})


def test_step_accepts_positive_number_or_month_year() -> None:
    base = {"type": "estimate", "min": 0, "max": 10}
    for step in (1, 0.5, "month", "year"):
        ClairicField.model_validate({**base, "step": step})
    for step in (0, -1, "week"):
        with pytest.raises(ValidationError):
            ClairicField.model_validate({**base, "step": step})


def test_unknown_field_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ClairicField.model_validate({"type": "text", "bogus": 1})
    with pytest.raises(ValidationError):
        ClairicField.model_validate({"type": "blob"})


def test_clairic_images_limit() -> None:
    ClairicRequest.model_validate(clairic(images=[IMG] * 4))
    with pytest.raises(ValidationError):
        ClairicRequest.model_validate(clairic(images=[IMG] * 5))


def test_image_forms() -> None:
    ClairicRequest.model_validate(clairic(images=["DATA:image/png;base64,AAAA"]))
    ClairicRequest.model_validate(
        clairic(images=[{"content_type": "image/png", "base64": "AAAA"}])
    )
    with pytest.raises(ValidationError):
        ClairicRequest.model_validate(clairic(images=["https://example.com/a.png"]))
    with pytest.raises(ValidationError):
        ClairicRequest.model_validate(clairic(images=[{"content_type": "image/png"}]))


def test_clairic_empty_state_needs_images() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        ClairicRequest.model_validate(clairic(state=""))
    with pytest.raises(ValidationError, match="non-empty"):
        ClairicRequest.model_validate({"model": "clairic", "fields": {"a": TEXT}})
    ClairicRequest.model_validate(clairic(state="", images=[IMG]))


def test_clairic_options_and_reference_date() -> None:
    req = ClairicRequest.model_validate(
        clairic(
            reference_date="2026-10-21",
            options={"include_quantiles": True, "return_state_text": False},
        )
    )
    assert req.to_payload()["options"] == {
        "include_quantiles": True,
        "return_state_text": False,
    }
    with pytest.raises(ValidationError):
        ClairicRequest.model_validate(clairic(options={"nope": True}))


def test_clairic_to_payload_drops_none() -> None:
    req = ClairicRequest.model_validate(
        clairic(fields={"a": {"type": "text", "instructions": None}})
    )
    assert req.to_payload() == {
        "model": "clairic",
        "state": "hello",
        "fields": {"a": {"type": "text"}},
    }


# --------------------------------------------------------------------------- #
# Clef request
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["clef", "clef-flash", " clef-flash ", "\tclef\n"])
def test_clef_model_selector_accepted(model: str) -> None:
    ClefRequest.model_validate(clef(model=model))


@pytest.mark.parametrize("model", ["jev", "clef-fast", "", CLEF])
def test_clef_model_selector_rejected(model: str) -> None:
    with pytest.raises(ValidationError):
        ClefRequest.model_validate(clef(model=model))


@pytest.mark.parametrize("n", [1, 64])
def test_clef_question_count_accepted(n: int) -> None:
    ClefRequest.model_validate(clef(questions={f"q{i}": noul_q() for i in range(n)}))


@pytest.mark.parametrize("n", [0, 65])
def test_clef_question_count_rejected(n: int) -> None:
    with pytest.raises(ValidationError):
        ClefRequest.model_validate(
            clef(questions={f"q{i}": noul_q() for i in range(n)})
        )


def test_clef_question_id_length_and_charset() -> None:
    ClefRequest.model_validate(clef(questions={"q" * 100: noul_q()}))
    ClefRequest.model_validate(clef(questions={"a.b-c_9": noul_q()}))
    for bad in ("q" * 101, "", "has space", "é"):
        with pytest.raises(ValidationError):
            ClefRequest.model_validate(clef(questions={bad: noul_q()}))


@pytest.mark.parametrize("n", [2, 255])
def test_choice_option_count_accepted(n: int) -> None:
    ClefRequest.model_validate(clef(questions={"q": choice_q(n)}))


@pytest.mark.parametrize("n", [0, 1, 256])
def test_choice_option_count_rejected(n: int) -> None:
    with pytest.raises(ValidationError):
        ClefRequest.model_validate(clef(questions={"q": choice_q(n)}))


def test_choice_option_ids_must_be_non_empty() -> None:
    question = {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"": "x", "b": None},
    }
    with pytest.raises(ValidationError, match="non-empty"):
        ClefRequest.model_validate(clef(questions={"q": question}))


def test_choice_criteria_values_may_be_structured_or_null() -> None:
    question = {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"a": "text", "b": {"k": 1}, "c": ["x"], "d": None},
    }
    req = ClefRequest.model_validate(clef(questions={"q": question}))
    # A null option description is meaningful and must survive serialisation.
    assert req.to_payload()["questions"]["q"]["criteria"]["d"] is None


@pytest.mark.parametrize("n", [2, 10])
def test_score_level_count_accepted(n: int) -> None:
    ClefRequest.model_validate(clef(questions={"q": score_q(n)}))


@pytest.mark.parametrize("n", [0, 1, 11])
def test_score_level_count_rejected(n: int) -> None:
    with pytest.raises(ValidationError):
        ClefRequest.model_validate(clef(questions={"q": score_q(n)}))


@pytest.mark.parametrize("instructions", ["", {}, []])
def test_instructions_must_be_non_empty(instructions: Any) -> None:
    with pytest.raises(ValidationError):
        ClefRequest.model_validate(
            clef(questions={"q": noul_q(instructions=instructions)})
        )


def test_instructions_may_be_structured() -> None:
    ClefRequest.model_validate(
        clef(questions={"q": noul_q(instructions={"question": "Is it?", "ref": [1]})})
    )
    ClefRequest.model_validate(clef(questions={"q": noul_q(instructions=["Is it?"])}))


def test_unknown_question_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ClefRequest.model_validate(
            clef(questions={"q": {"type": "rank", "instructions": "x"}})
        )


def test_clef_state_may_be_structured() -> None:
    ClefRequest.model_validate(clef(state={"user": {"name": "Ada"}, "tags": ["a"]}))
    ClefRequest.model_validate(clef(state=[1, 2, 3]))
    ClefRequest.model_validate(clef(state="text"))
    with pytest.raises(ValidationError):
        ClefRequest.model_validate({"model": "clef", "questions": {"q": noul_q()}})


def test_clef_images_limit() -> None:
    ClefRequest.model_validate(clef(images=[IMG] * 4))
    with pytest.raises(ValidationError):
        ClefRequest.model_validate(clef(images=[IMG] * 5))


def test_noul_criteria_serialise_as_true_and_false() -> None:
    req = ClefRequest.model_validate(
        clef(
            questions={
                "q": noul_q(criteria={"true": "it is", "false": {"it": "is not"}}),
                "r": noul_q(criteria={"true": "yes"}),
                "s": noul_q(),
            }
        )
    )
    questions = req.to_payload()["questions"]
    assert questions["q"]["criteria"] == {"true": "it is", "false": {"it": "is not"}}
    assert questions["r"]["criteria"] == {"true": "yes"}
    assert "criteria" not in questions["s"]


def test_clef_to_payload_round_trips_structured_state() -> None:
    state = {"a": None, "b": [1, {"c": None}]}
    payload = ClefRequest.model_validate(clef(state=state)).to_payload()
    assert payload["state"] == state


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #

DOCUMENTED_CLAIRIC: dict[str, Any] = {
    "model": "clairic",
    "answers": {
        "invoice_number": {
            "type": "text",
            "p_null": 3.8e-10,
            "span": [8, 16],
            "guess": "INV-2291",
            "confidence": 1,
            "value": "INV-2291",
        },
        "due_date": {
            "type": "date",
            "p_null": 1.9e-08,
            "guess": "2026-10-02",
            "confidence": 0.9999,
            "value": "2026-10-02",
        },
        "total": {
            "type": "number",
            "p_null": 2.3e-08,
            "guess": 1080,
            "confidence": 0.9999,
            "value": 1080,
        },
        "contacts": {
            "type": "list",
            "value": ["Dana Ruiz", "Priya Shah"],
            "spans": [[120, 129], [131, 141]],
            "confidence": 0.9999,
        },
        "po_number": {
            "type": "text",
            "p_null": 1,
            "span": [8, 16],
            "guess": "INV-2291",
            "confidence": 0.9995,
            "value": None,
        },
        "days_to_pay": {
            "type": "estimate",
            "value": 30,
            "mean": 29.535,
            "median": 30,
            "p10": 21,
            "p90": 38,
            "p_below": 5.9e-06,
            "p_above": 1.2e-05,
        },
    },
    "truncated": False,
    "usage": {
        "state_tokens": 61,
        "image_tokens": 0,
        "schema_tokens": 472,
        "input_tokens": 655,
        "output_tokens": 0,
        "passes": 3,
    },
}


def test_parse_documented_clairic_response() -> None:
    parsed = parse_clairic(DOCUMENTED_CLAIRIC)
    assert parsed.model == "clairic"
    assert parsed.truncated is False
    assert parsed.state_text is None
    assert set(parsed.answers) == set(DOCUMENTED_CLAIRIC["answers"])

    invoice = parsed.answers["invoice_number"]
    assert invoice.type == "text"
    assert invoice.value == "INV-2291"
    assert invoice.span == [8, 16]
    assert invoice.p_null == pytest.approx(3.8e-10)
    assert invoice.confidence == 1.0

    total = parsed.answers["total"]
    assert total.value == 1080
    assert total.guess == 1080

    contacts = parsed.answers["contacts"]
    assert contacts.value == ["Dana Ruiz", "Priya Shah"]
    assert contacts.spans == [[120, 129], [131, 141]]
    assert contacts.span is None
    assert contacts.p_null is None

    po = parsed.answers["po_number"]
    assert po.value is None
    assert po.guess == "INV-2291"
    assert po.p_null == 1.0

    days = parsed.answers["days_to_pay"]
    assert (days.value, days.mean, days.median) == (30, 29.535, 30)
    assert (days.p10, days.p90) == (21, 38)
    assert days.p_below == pytest.approx(5.9e-06)
    assert days.p_above == pytest.approx(1.2e-05)

    usage = parsed.usage
    assert (usage.state_tokens, usage.image_tokens, usage.schema_tokens) == (61, 0, 472)
    assert (usage.input_tokens, usage.output_tokens, usage.passes) == (655, 0, 3)


def test_clairic_usage_optional_fields_default_to_none() -> None:
    body = {
        "model": "clairic",
        "answers": {},
        "truncated": True,
        "state_text": "abc",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    parsed = parse_clairic(body)
    assert parsed.truncated is True
    assert parsed.state_text == "abc"
    assert parsed.usage.passes is None


def test_clairic_missing_truncated_raises_with_raw() -> None:
    body = {k: v for k, v in DOCUMENTED_CLAIRIC.items() if k != "truncated"}
    with pytest.raises(ModelResponseError) as info:
        parse_clairic(body)
    assert info.value.raw is body
    assert "truncated" in str(info.value)


@pytest.mark.parametrize("key", ["model", "answers", "usage"])
def test_clairic_missing_required_keys_raise(key: str) -> None:
    body = {k: v for k, v in DOCUMENTED_CLAIRIC.items() if k != key}
    with pytest.raises(ModelResponseError):
        parse_clairic(body)


def test_clairic_answer_requires_type_and_value() -> None:
    for missing in ("type", "value"):
        answer = {"type": "text", "value": "x"}
        del answer[missing]
        body = {**DOCUMENTED_CLAIRIC, "answers": {"a": answer}}
        with pytest.raises(ModelResponseError):
            parse_clairic(body)


def test_clairic_usage_requires_token_counts() -> None:
    body = {**DOCUMENTED_CLAIRIC, "usage": {"input_tokens": 1}}
    with pytest.raises(ModelResponseError):
        parse_clairic(body)


def test_clairic_tolerates_unknown_keys() -> None:
    body = {
        **DOCUMENTED_CLAIRIC,
        "extra_top": 1,
        "answers": {"a": {"type": "text", "value": "x", "new_stat": [1]}},
    }
    parsed = parse_clairic(body)
    assert parsed.model_extra == {"extra_top": 1}
    assert parsed.answers["a"].model_extra == {"new_stat": [1]}


def test_clairic_non_object_body_raises() -> None:
    with pytest.raises(ModelResponseError):
        parse_clairic([])  # type: ignore[arg-type]


DOCUMENTED_CLEF: dict[str, Any] = {
    "model": "clef",
    "answers": {
        "is_refund": {"type": "noul", "noul": 0.97},
        "topic": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.8, "tech": 0.15, "other": 0.05},
            "confidence": 0.8,
        },
        "urgency": {
            "type": "score",
            "score": 1.7,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
            "confidence": 0.5,
        },
    },
    "usage": {"input_tokens": 412, "output_tokens": 0},
}


def test_parse_clef_response() -> None:
    parsed = parse_clef(DOCUMENTED_CLEF)
    assert parsed.model == "clef"
    assert parsed.usage.input_tokens == 412
    noul_answer = parsed.answers["is_refund"]
    assert isinstance(noul_answer, NoulAnswer)
    assert noul_answer.noul == 0.97
    choice_answer = parsed.answers["topic"]
    assert isinstance(choice_answer, ChoiceAnswer)
    assert choice_answer.choice == "billing"
    assert choice_answer.probabilities["tech"] == 0.15
    score_answer = parsed.answers["urgency"]
    assert isinstance(score_answer, ScoreAnswer)
    assert score_answer.score == 1.7
    assert score_answer.legend["2"] == "high"
    assert score_answer.confidence == 0.5


@pytest.mark.parametrize(
    "span", ["8-16", [1.5, 2], ["a", "b"], [[1, 2]], [True, 2], {"s": 1}, 7]
)
def test_malformed_span_does_not_fail_the_response(span: Any) -> None:
    answer = {"type": "text", "value": "x", "span": span, "spans": span}
    parsed = parse_clairic({**DOCUMENTED_CLAIRIC, "answers": {"a": answer}})
    assert parsed.answers["a"].span is None
    assert parsed.answers["a"].value == "x"


def test_spans_of_any_length_are_accepted() -> None:
    answer = {"type": "list", "value": [], "span": [1, 2, 3], "spans": [[1], [2, 3, 4]]}
    parsed = parse_clairic({**DOCUMENTED_CLAIRIC, "answers": {"a": answer}})
    assert parsed.answers["a"].span == [1, 2, 3]
    assert parsed.answers["a"].spans == [[1], [2, 3, 4]]


def test_spans_with_a_bad_item_become_none() -> None:
    answer = {"type": "list", "value": [], "spans": [[1, 2], ["x"]]}
    parsed = parse_clairic({**DOCUMENTED_CLAIRIC, "answers": {"a": answer}})
    assert parsed.answers["a"].spans is None


def test_probabilities_within_rounding_noise_are_clipped() -> None:
    body = {
        "model": "clef",
        "answers": {
            "n": {"type": "noul", "noul": 1.0000005},
            "m": {"type": "noul", "noul": -5e-7},
            "c": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 1.0000005, "b": -1e-7},
                "confidence": 1.0000009,
            },
            "s": {
                "type": "score",
                "score": 0.0,
                "legend": {},
                "probabilities": {"0": 1.0},
                "confidence": 1,
            },
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    answers = parse_clef(body).answers
    assert isinstance(answers["n"], NoulAnswer)
    assert answers["n"].noul == 1.0
    assert isinstance(answers["m"], NoulAnswer)
    assert answers["m"].noul == 0.0
    assert isinstance(answers["c"], ChoiceAnswer)
    assert answers["c"].probabilities == {"a": 1.0, "b": 0.0}
    assert answers["c"].confidence == 1.0


@pytest.mark.parametrize(
    "bad",
    [1.001, -0.001, 2, float("nan"), "0.5x", True, "1.5", "0.5", "1", None, [0.5]],
)
def test_probabilities_outside_tolerance_are_rejected(bad: Any) -> None:
    body = {
        "model": "clef",
        "answers": {"q": {"type": "noul", "noul": bad}},
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    with pytest.raises(ModelResponseError):
        parse_clef(body)


def test_clef_missing_noul_raises_with_raw() -> None:
    body = {
        "model": "clef",
        "answers": {"q": {"type": "noul"}},
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    with pytest.raises(ModelResponseError) as info:
        parse_clef(body)
    assert info.value.raw is body


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}},
        {"type": "score", "score": 1, "probabilities": {}, "confidence": 1},
        {"type": "mystery", "noul": 0.5},
        {"type": "noul", "noul": 1.5},
        {"type": "noul", "noul": -0.1},
        {"type": "choice", "choice": "a", "probabilities": {"a": 1.2}, "confidence": 1},
    ],
)
def test_clef_invalid_answers_raise(answer: dict[str, Any]) -> None:
    body = {**DOCUMENTED_CLEF, "answers": {"q": answer}}
    with pytest.raises(ModelResponseError):
        parse_clef(body)


@pytest.mark.parametrize("key", ["model", "answers", "usage"])
def test_clef_missing_required_keys_raise(key: str) -> None:
    body = {k: v for k, v in DOCUMENTED_CLEF.items() if k != key}
    with pytest.raises(ModelResponseError):
        parse_clef(body)


def test_clef_tolerates_unknown_keys() -> None:
    body = {
        **DOCUMENTED_CLEF,
        "extra_top": True,
        "answers": {"q": {"type": "noul", "noul": 0.5, "note": "hi"}},
    }
    parsed = parse_clef(body)
    answer = parsed.answers["q"]
    assert isinstance(answer, NoulAnswer)
    assert answer.model_extra == {"note": "hi"}
    assert parsed.model_extra == {"extra_top": True}


# --------------------------------------------------------------------------- #
# body_model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (CLAIRIC, "clairic"),
        (CLEF, "clef"),
        (CLEF_FLASH, "clef-flash"),
        ("clairic", "clairic"),
        ("clef", "clef"),
        ("clef-flash", "clef-flash"),
    ],
)
def test_body_model_maps_ids_and_bare_names(model_id: str, expected: str) -> None:
    assert body_model(model_id) == expected


@pytest.mark.parametrize("bad", ["jev", "", "@cf/cloudflare/jev", " clef", "CLEF"])
def test_body_model_rejects_unknown(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown System One model"):
        body_model(bad)


def test_model_id_constants() -> None:
    assert CLAIRIC == "@cf/cloudflare/clairic"
    assert CLEF == "@cf/cloudflare/clef"
    assert CLEF_FLASH == "@cf/cloudflare/clef-flash"


# --------------------------------------------------------------------------- #
# Numeric strictness, dates and instructions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(bound: float) -> None:
    for key in ("min", "max", "step"):
        spec = {"type": "estimate", "min": 0, "max": 1, key: bound}
        with pytest.raises(ValidationError):
            ClairicField.model_validate(spec)


@pytest.mark.parametrize("key", ["min", "max", "step", "decimals", "horizon"])
def test_booleans_are_rejected_where_numbers_are_expected(key: str) -> None:
    spec = {"type": "forecast", "min": 0, "max": 1, "horizon": 2, key: True}
    with pytest.raises(ValidationError, match="not a boolean"):
        ClairicField.model_validate(spec)


@pytest.mark.parametrize(
    "date_text",
    [
        "2026-1-5",
        "20260105",
        "2026-01-05T00:00",
        "\u0662\u0660\u0662\u0666-\u0660\u0661-\u0660\u0665",
        "2026-13-01",
        "2026-01-05\n",
    ],
)
def test_bounds_must_be_strict_iso_dates(date_text: str) -> None:
    with pytest.raises(ValidationError, match="ISO date"):
        ClairicField.model_validate({"type": "estimate", "min": date_text, "max": 3})


def test_whitespace_instructions_are_rejected() -> None:
    for blank in ("   ", "\n\t"):
        with pytest.raises(ValidationError):
            ClefRequest.model_validate(
                clef(questions={"q": noul_q(instructions=blank)})
            )


# --------------------------------------------------------------------------- #
# Image and body size limits
# --------------------------------------------------------------------------- #

MIB = 1024 * 1024


def b64(size: int) -> str:
    return base64.b64encode(b"\0" * size).decode()


def b64_image(size: int, content_type: str = "image/png") -> dict[str, str]:
    return {"content_type": content_type, "base64": b64(size)}


@pytest.mark.parametrize(
    "content_type", ["image/png", "image/jpeg", "image/jpg", "image/webp", "IMAGE/PNG"]
)
def test_image_types_accepted(content_type: str) -> None:
    ClairicRequest.model_validate(clairic(images=[b64_image(3, content_type)]))
    ClairicRequest.model_validate(
        clairic(images=[f"data:{content_type};base64,{b64(3)}"])
    )


@pytest.mark.parametrize(
    "content_type", ["image/gif", "image/svg+xml", "text/plain", ""]
)
def test_other_image_types_rejected(content_type: str) -> None:
    with pytest.raises(ValidationError, match="image type"):
        ClairicRequest.model_validate(clairic(images=[b64_image(3, content_type)]))
    with pytest.raises(ValidationError, match="image type"):
        ClefRequest.model_validate(
            clef(images=[f"data:{content_type};base64,{b64(3)}"])
        )


@pytest.mark.parametrize("payload", ["AAA", "!!!!", "AA=A", "AAAA="])
def test_invalid_base64_rejected(payload: str) -> None:
    with pytest.raises(ValidationError, match="base64"):
        ClairicRequest.model_validate(
            clairic(images=[{"content_type": "image/png", "base64": payload}])
        )
    with pytest.raises(ValidationError, match="base64"):
        ClairicRequest.model_validate(
            clairic(images=[f"data:image/png;base64,{payload}"])
        )


def test_data_url_must_be_base64() -> None:
    with pytest.raises(ValidationError, match="data:image/png;base64"):
        ClairicRequest.model_validate(clairic(images=["data:image/png,AAAA"]))


def test_single_image_limit_is_four_mib_decoded() -> None:
    ClairicRequest.model_validate(clairic(images=[b64_image(4 * MIB)]))
    with pytest.raises(ValidationError, match="4 MiB"):
        ClairicRequest.model_validate(clairic(images=[b64_image(4 * MIB + 1)]))
    with pytest.raises(ValidationError, match="4 MiB"):
        ClefRequest.model_validate(
            clef(images=[f"data:image/webp;base64,{b64(4 * MIB + 1)}"])
        )


def test_total_image_limit_is_eight_mib_decoded() -> None:
    ClairicRequest.model_validate(clairic(images=[b64_image(4 * MIB)] * 2))
    with pytest.raises(ValidationError, match="8 MiB"):
        ClairicRequest.model_validate(
            clairic(images=[b64_image(4 * MIB)] * 2 + [b64_image(1)])
        )
    with pytest.raises(ValidationError, match="8 MiB"):
        ClefRequest.model_validate(clef(images=[b64_image(3 * MIB)] * 3))


def test_clairic_body_limit_is_sixteen_mib() -> None:
    ClairicRequest.model_validate(clairic(state="a" * (16 * MIB - 1000)))
    with pytest.raises(ValidationError, match="request body"):
        ClairicRequest.model_validate(clairic(state="a" * (16 * MIB)))


def test_clef_body_limit_is_thirteen_mib() -> None:
    ClefRequest.model_validate(clef(state="a" * (13 * MIB - 1000)))
    with pytest.raises(ValidationError, match="request body"):
        ClefRequest.model_validate(clef(state="a" * (13 * MIB)))
    # Fine for Clairic, too big for Clef: the two limits differ.
    ClairicRequest.model_validate(clairic(state="a" * (14 * MIB)))


def test_mime_wrapped_base64_is_accepted() -> None:
    raw = b64(300)
    wrapped = "\r\n".join(raw[i : i + 76] for i in range(0, len(raw), 76)) + "\r\n"
    assert "\n" in wrapped
    ClairicRequest.model_validate(
        clairic(images=[{"content_type": "image/png", "base64": wrapped}])
    )
    ClairicRequest.model_validate(clairic(images=[f"data:image/png;base64,{wrapped}"]))
    # Whitespace stripping must not make otherwise invalid data valid.
    with pytest.raises(ValidationError, match="base64"):
        ClairicRequest.model_validate(
            clairic(images=[{"content_type": "image/png", "base64": "AA\nA"}])
        )
