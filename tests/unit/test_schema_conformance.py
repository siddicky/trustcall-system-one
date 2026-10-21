"""Conformance of the wire models and builders with the published schemas."""

from typing import Any

import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis_jsonschema import from_schema

from tests.unit.schema_helpers import load_schema, validator
from trustcall_system_one._wire import (
    ClairicRequest,
    ClefRequest,
    parse_clairic,
    parse_clef,
)

PNG = "data:image/png;base64,AAAA"

CLAIRIC_REQUESTS: list[dict[str, Any]] = [
    {"model": "clairic", "state": "x", "fields": {"a": {"type": "text"}}},
    {
        "model": "@cf/cloudflare/clairic",
        "state": "",
        "images": [PNG, {"content_type": "image/webp", "base64": "AAAA"}],
        "reference_date": "2026-10-21",
        "options": {"include_quantiles": True, "return_state_text": False},
        "fields": {
            "t": {"type": "text", "instructions": "name", "null_threshold": 0.3},
            "n": {"type": "number", "integer": True, "decimals": 2, "unit": "USD"},
            "d": {"type": "date"},
            "tm": {"type": "time"},
            "dt": {"type": "datetime"},
            "du": {"type": "duration", "unit": "weeks", "integer": True},
            "l": {"type": "list", "items": "number", "decimals": 1},
            "e": {
                "type": "estimate",
                "min": 0,
                "max": 10.5,
                "step": 0.5,
                "scale": "log",
            },
            "ed": {
                "type": "estimate",
                "min": "2026-01-01",
                "max": "2026-12-31",
                "step": "month",
            },
            "f": {
                "type": "forecast",
                "min": 0,
                "max": 100,
                "horizon": 12,
                "interval": "quarter",
            },
        },
    },
]

CLEF_REQUESTS: list[dict[str, Any]] = [
    {
        "model": "clef",
        "state": "text",
        "questions": {"q": {"type": "noul", "instructions": "Is it?"}},
    },
    {
        "model": " clef-flash ",
        "state": {"records": [{"id": 1, "note": None}], "tags": ["a"]},
        "images": [PNG],
        "questions": {
            "n": {
                "type": "noul",
                "instructions": {"question": "Is it?", "ref": [1, 2]},
                "criteria": {"true": "it is", "false": ["it is not"]},
            },
            "c": {
                "type": "choice",
                "instructions": ["Pick", {"data": 1}],
                "criteria": {"a": None, "b": "B", "c": {"k": "v"}, "d": ["x"]},
            },
            "s": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["lo", {"k": "mid"}, ["hi"]],
            },
        },
    },
]


@pytest.mark.parametrize("body", CLAIRIC_REQUESTS)
def test_clairic_requests_validate_against_the_published_input_schema(
    body: dict[str, Any],
) -> None:
    payload = ClairicRequest.model_validate(body).to_payload()
    validator("clairic-input").validate(payload)
    # No key was dropped except ``None`` ones: the payload keeps every input key.
    assert set(payload) == set(body)


@pytest.mark.parametrize("body", CLEF_REQUESTS)
def test_clef_requests_validate_against_the_published_input_schema(
    body: dict[str, Any],
) -> None:
    payload = ClefRequest.model_validate(body).to_payload()
    validator("clef-input").validate(payload)
    assert set(payload) == set(body)
    for qid, question in body["questions"].items():
        assert payload["questions"][qid]["type"] == question["type"]


def test_the_autouse_fixture_checks_to_payload_against_the_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # conftest wraps to_payload so every request built anywhere in the unit
    # suite is checked; prove the wrapper really validates by breaking it.
    request = ClairicRequest.model_validate(CLAIRIC_REQUESTS[0])
    assert request.to_payload()["model"] == "clairic"
    monkeypatch.setattr(ClairicRequest, "model_dump", lambda self, **kw: {"model": "x"})
    with pytest.raises(jsonschema.ValidationError):
        request.to_payload()


HYPOTHESIS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large,
    ],
)


def _small(schema: dict[str, Any]) -> dict[str, Any]:
    """Bound collection sizes so generation stays fast."""
    bounded = dict(schema)
    answers = bounded["properties"]["answers"]
    bounded["properties"] = {
        **bounded["properties"],
        "answers": {**answers, "maxProperties": 4},
    }
    return bounded


@HYPOTHESIS
@given(from_schema(_small(load_schema("clairic-output"))))
def test_generated_clairic_responses_parse(body: dict[str, Any]) -> None:
    validator("clairic-output").validate(body)
    parsed = parse_clairic(body)
    assert set(parsed.answers) == set(body["answers"])
    assert parsed.truncated is body["truncated"]
    assert parsed.usage.input_tokens == body["usage"]["input_tokens"]


@HYPOTHESIS
@given(from_schema(_small(load_schema("clef-output"))))
def test_generated_clef_responses_parse(body: dict[str, Any]) -> None:
    validator("clef-output").validate(body)
    parsed = parse_clef(body)
    assert set(parsed.answers) == set(body["answers"])
    for qid, answer in body["answers"].items():
        assert parsed.answers[qid].type == answer["type"]
