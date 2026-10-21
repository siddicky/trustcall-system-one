"""Tests for FakeTransport and the response builders."""

import datetime
import decimal
from typing import Any

import pytest
from pydantic import ValidationError

from tests.unit.schema_helpers import validator
from trustcall_system_one import CLAIRIC, CLEF, CLEF_FLASH, CallMeta, ModelResponseError
from trustcall_system_one._wire import parse_clairic, parse_clef
from trustcall_system_one.testing import (
    FakeTransport,
    choice,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    score,
)
from trustcall_system_one.transport import Transport, TransportResponse

META = CallMeta(kind="extract", schema="Person")
CLAIRIC_PAYLOAD = {
    "model": "clairic",
    "state": "Ada is 36",
    "fields": {"name": {"type": "text"}},
}
CLEF_PAYLOAD = {
    "model": "clef",
    "state": {"a": 1},
    "questions": {"q": {"type": "noul", "instructions": "Is it?"}},
}


def clairic_handler(
    model: str, payload: dict[str, Any], meta: CallMeta
) -> dict[str, Any]:
    return clairic_response(
        {name: clairic_answer("text", "Ada", span=(0, 3)) for name in payload["fields"]}
    )


async def test_records_calls_and_returns_the_handler_body() -> None:
    fake = FakeTransport(clairic_handler)
    assert isinstance(fake, Transport)
    result = await fake.run(CLAIRIC, CLAIRIC_PAYLOAD, meta=META)
    assert isinstance(result, TransportResponse)
    assert result.body["answers"]["name"]["value"] == "Ada"
    assert (result.retries, result.status_code) == (0, 200)
    assert result.duration_ms >= 0
    assert fake.calls == [(CLAIRIC, CLAIRIC_PAYLOAD, META)]


async def test_async_handler_is_awaited() -> None:
    async def handler(
        model: str, payload: dict[str, Any], meta: CallMeta
    ) -> dict[str, Any]:
        return clef_response({"q": noul(0.9)}, model="clef-flash")

    fake = FakeTransport(handler)
    result = await fake.run(
        CLEF_FLASH, {**CLEF_PAYLOAD, "model": "clef-flash"}, meta=META
    )
    assert result.body["answers"]["q"]["noul"] == 0.9


async def test_handler_sees_a_copy_and_calls_keep_the_original() -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        payload["fields"].clear()
        return clairic_response({"name": clairic_answer("text", "Ada")})

    fake = FakeTransport(handler)
    payload = {**CLAIRIC_PAYLOAD, "fields": {"name": {"type": "text"}}}
    await fake.run(CLAIRIC, payload, meta=META)
    assert fake.calls[0][1]["fields"] == {"name": {"type": "text"}}
    assert payload["fields"] == {"name": {"type": "text"}}


async def test_invalid_clairic_payload_is_rejected_before_the_handler() -> None:
    called = False

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        nonlocal called
        called = True
        return clairic_response({})

    fake = FakeTransport(handler)
    bad = {**CLAIRIC_PAYLOAD, "fields": {"1bad name": {"type": "text"}}}
    with pytest.raises(ValidationError):
        await fake.run(CLAIRIC, bad, meta=META)
    assert not called
    assert len(fake.calls) == 1  # still recorded for diagnosis


async def test_invalid_clef_payload_is_rejected() -> None:
    fake = FakeTransport(lambda m, p, meta: clef_response({}))
    bad = {**CLEF_PAYLOAD, "questions": {}}
    with pytest.raises(ValidationError):
        await fake.run(CLEF, bad, meta=META)


async def test_unknown_model_is_rejected() -> None:
    fake = FakeTransport(lambda m, p, meta: {})
    with pytest.raises(ValueError, match="unknown System One model"):
        await fake.run("@cf/other/model", CLAIRIC_PAYLOAD, meta=META)


async def test_response_missing_required_keys_is_rejected() -> None:
    clairic_fake = FakeTransport(
        lambda m, p, meta: {"model": "clairic", "answers": {}, "usage": {}}
    )
    with pytest.raises(ModelResponseError):
        await clairic_fake.run(CLAIRIC, CLAIRIC_PAYLOAD, meta=META)

    clef_fake = FakeTransport(
        lambda m, p, meta: {
            "model": "clef",
            "answers": {"q": {"type": "noul"}},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
    )
    with pytest.raises(ModelResponseError):
        await clef_fake.run(CLEF, CLEF_PAYLOAD, meta=META)


async def test_none_from_handler_raises_assertion_with_context() -> None:
    fake = FakeTransport(lambda m, p, meta: None)
    with pytest.raises(AssertionError) as info:
        await fake.run(CLAIRIC, CLAIRIC_PAYLOAD, meta=META)
    message = str(info.value)
    assert CLAIRIC in message
    assert "kind='extract'" in message
    assert "Ada is 36" in message


async def test_key_error_from_handler_raises_assertion() -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        return {"answers": {}}["nope"]

    fake = FakeTransport(handler)
    with pytest.raises(AssertionError, match="unhandled call") as info:
        await fake.run(CLAIRIC, CLAIRIC_PAYLOAD, meta=META)
    assert isinstance(info.value.__cause__, KeyError)


async def test_other_handler_errors_propagate() -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await FakeTransport(handler).run(CLAIRIC, CLAIRIC_PAYLOAD, meta=META)


def test_clairic_answer_builder() -> None:
    answer = clairic_answer("text", "Ada")
    assert answer == {
        "type": "text",
        "value": "Ada",
        "p_null": 1e-6,
        "confidence": 0.99,
    }

    full = clairic_answer(
        "list",
        ["a", "b"],
        p_null=0.2,
        confidence=0.5,
        span=(1, 2),
        spans=[(1, 2), (3, 4)],
        guess="g",
        mean=1.5,
    )
    assert full["span"] == [1, 2]
    assert full["spans"] == [[1, 2], [3, 4]]
    assert full["guess"] == "g"
    assert full["mean"] == 1.5


def test_clairic_response_builder_parses() -> None:
    body = clairic_response(
        {
            "name": clairic_answer("text", "Ada", span=(0, 3)),
            "age": clairic_answer("number", None, p_null=1.0, guess=36),
            "tags": clairic_answer("list", ["x"], spans=[(0, 1)]),
            "eta": clairic_answer("estimate", 3, mean=3.1, median=3, p10=1, p90=5),
        },
        truncated=True,
        state_text="Ada is 36",
        input_tokens=42,
    )
    parsed = parse_clairic(body)
    assert parsed.truncated is True
    assert parsed.state_text == "Ada is 36"
    assert parsed.usage.input_tokens == 42
    assert parsed.answers["name"].span == [0, 3]
    assert parsed.answers["age"].value is None
    assert parsed.answers["tags"].spans == [[0, 1]]
    assert parsed.answers["eta"].p90 == 5

    default = clairic_response({})
    assert default["truncated"] is False
    assert "state_text" not in default
    assert default["usage"]["input_tokens"] == 100
    parse_clairic(default)


def test_clef_builders_parse() -> None:
    body = clef_response(
        {
            "a": noul(0.25),
            "b": choice("x", {"x": 0.7, "y": 0.3}),
            "c": score(1.2, {"0": 0.1, "1": 0.6, "2": 0.3}, legend={"0": "lo"}),
            "d": choice("y", {"x": 0.4, "y": 0.6}, confidence=0.1),
        },
        input_tokens=7,
        model="clef-flash",
    )
    parsed = parse_clef(body)
    assert parsed.model == "clef-flash"
    assert parsed.usage.input_tokens == 7
    assert body["answers"]["b"]["confidence"] == 0.7  # defaults to max probability
    assert body["answers"]["c"]["confidence"] == 0.6
    assert body["answers"]["c"]["legend"] == {"0": "lo"}
    assert body["answers"]["d"]["confidence"] == 0.1
    assert score(0.0, {"0": 1.0})["legend"] == {}
    assert clef_response({})["model"] == "clef"
    parse_clef(clef_response({}))


# --------------------------------------------------------------------------- #
# Strictness: the fake must be at least as strict as the real API
# --------------------------------------------------------------------------- #


def clef_q(question: dict[str, Any], model: str = "clef") -> dict[str, Any]:
    return {"model": model, "state": "text", "questions": {"q": question}}


CHOICE_Q = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": "B"}}
SCORE_Q = {"type": "score", "instructions": "Rate", "criteria": ["lo", "mid", "hi"]}


def answering(body: dict[str, Any]) -> FakeTransport:
    return FakeTransport(lambda m, p, meta: body)


@pytest.mark.parametrize("bare", ["clairic", "clef", "clef-flash"])
async def test_bare_model_names_are_rejected_as_urls(bare: str) -> None:
    fake = FakeTransport(lambda m, p, meta: clairic_response({}))
    payload = CLAIRIC_PAYLOAD if bare == "clairic" else CLEF_PAYLOAD
    with pytest.raises(ValueError, match="full id"):
        await fake.run(bare, payload, meta=META)
    assert fake.calls == []


@pytest.mark.parametrize(
    ("model", "selector"),
    [(CLEF_FLASH, "clef"), (CLEF, "clef-flash"), (CLEF, " clef-flash ")],
)
async def test_url_and_body_model_must_agree_for_clef(
    model: str, selector: str
) -> None:
    fake = FakeTransport(lambda m, p, meta: clef_response({"q": noul(0.5)}))
    with pytest.raises(AssertionError, match="does not match body model"):
        await fake.run(
            model, clef_q({"type": "noul", "instructions": "?"}, selector), meta=META
        )


async def test_url_and_body_model_must_agree_across_families() -> None:
    fake = FakeTransport(lambda m, p, meta: clairic_response({}))
    with pytest.raises(ValidationError):  # a clef body is not a valid clairic body
        await fake.run(CLAIRIC, CLEF_PAYLOAD, meta=META)
    with pytest.raises(ValidationError):
        await fake.run(CLEF, CLAIRIC_PAYLOAD, meta=META)


async def test_clairic_body_may_use_the_full_id_selector() -> None:
    fake = FakeTransport(clairic_handler)
    await fake.run(CLAIRIC, {**CLAIRIC_PAYLOAD, "model": CLAIRIC}, meta=META)


@pytest.mark.parametrize(
    "field",
    [
        {"type": "number", "integer": "yes"},
        {"type": "number", "integer": 1},
        {"type": "number", "decimals": "2"},
        {"type": "number", "decimals": 2.0},
        {"type": "number", "decimals": True},
        {"type": "text", "null_threshold": "0.5"},
        {"type": "estimate", "min": "1", "max": 2},
        {"type": "estimate", "min": True, "max": 2},
    ],
)
async def test_lax_types_are_rejected_in_strict_mode(field: dict[str, Any]) -> None:
    fake = FakeTransport(clairic_handler)
    payload = {**CLAIRIC_PAYLOAD, "fields": {"f": field}}
    with pytest.raises(ValidationError):
        await fake.run(CLAIRIC, payload, meta=META)


async def test_lax_clef_types_are_rejected_in_strict_mode() -> None:
    fake = FakeTransport(lambda m, p, meta: clef_response({"q": noul(0.5)}))
    payload = clef_q({"type": "noul", "instructions": "?"})
    payload["questions"]["q"]["criteria"] = {"true": {"k": 1}}
    await fake.run(CLEF, payload, meta=META)  # structured criteria values are fine
    with pytest.raises(ValidationError):
        await fake.run(
            CLEF, {**payload, "images": "data:image/png;base64,AAAA"}, meta=META
        )


@pytest.mark.parametrize(
    "value",
    [datetime.date(2026, 1, 1), decimal.Decimal("1.5"), {1, 2}, b"bytes", object()],
)
async def test_payloads_that_cannot_be_json_encoded_are_rejected(value: Any) -> None:
    fake = FakeTransport(lambda m, p, meta: clef_response({"q": noul(0.5)}))
    payload = clef_q({"type": "noul", "instructions": "?"})
    payload["state"] = {"when": value}
    with pytest.raises(TypeError):
        await fake.run(CLEF, payload, meta=META)
    assert len(fake.calls) == 1  # recorded for diagnosis


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
async def test_non_finite_numbers_in_a_payload_are_rejected(bad: float) -> None:
    fake = FakeTransport(clairic_handler)
    payload = {
        **CLAIRIC_PAYLOAD,
        "fields": {"f": {"type": "estimate", "min": bad, "max": 2}},
    }
    with pytest.raises(ValueError):
        await fake.run(CLAIRIC, payload, meta=META)


async def test_payload_is_json_round_tripped_before_the_handler() -> None:
    seen: list[dict[str, Any]] = []

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        seen.append(payload)
        return clef_response({"q": noul(0.5)})

    fake = FakeTransport(handler)
    payload = clef_q({"type": "noul", "instructions": "?"})
    payload["state"] = {1: ("a", "b")}  # int key and tuple, as json would send them
    await fake.run(CLEF, payload, meta=META)
    assert seen[0]["state"] == {"1": ["a", "b"]}
    assert fake.calls[0][1]["state"] == {"1": ["a", "b"]}


@pytest.mark.parametrize(
    "body",
    [
        clairic_response({}),  # no answers for the requested field
        clairic_response(
            {"name": clairic_answer("text", "A"), "extra": clairic_answer("text", "B")}
        ),
        clairic_response({"other": clairic_answer("text", "A")}),
    ],
)
async def test_clairic_answer_keys_must_equal_the_requested_fields(
    body: dict[str, Any],
) -> None:
    with pytest.raises(AssertionError, match="differ from the requested fields"):
        await answering(body).run(CLAIRIC, CLAIRIC_PAYLOAD, meta=META)


async def test_clairic_response_model_must_match() -> None:
    body = {**clairic_response({"name": clairic_answer("text", "A")}), "model": "clef"}
    with pytest.raises(AssertionError, match="response model"):
        await answering(body).run(CLAIRIC, CLAIRIC_PAYLOAD, meta=META)


async def test_clef_answer_keys_must_equal_the_asked_questions() -> None:
    payload = clef_q({"type": "noul", "instructions": "?"})
    for answers in ({}, {"q": noul(0.5), "r": noul(0.5)}, {"r": noul(0.5)}):
        with pytest.raises(AssertionError, match="differ from the asked questions"):
            await answering(clef_response(answers)).run(CLEF, payload, meta=META)


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ({"type": "noul", "instructions": "?"}, choice("a", {"a": 1.0})),
        (CHOICE_Q, noul(0.5)),
        (SCORE_Q, choice("a", {"a": 1.0})),
        (CHOICE_Q, score(1.0, {"0": 1.0})),
    ],
)
async def test_clef_answer_type_must_match_the_question_type(
    question: dict[str, Any], answer: dict[str, Any]
) -> None:
    with pytest.raises(AssertionError, match="answered as"):
        await answering(clef_response({"q": answer})).run(
            CLEF, clef_q(question), meta=META
        )


async def test_choice_must_be_one_of_the_offered_options() -> None:
    fake = answering(clef_response({"q": choice("zzz", {"a": 0.5, "b": 0.5})}))
    with pytest.raises(AssertionError, match="is not one of"):
        await fake.run(CLEF, clef_q(CHOICE_Q), meta=META)


async def test_choice_probabilities_must_be_a_subset_of_the_options() -> None:
    fake = answering(clef_response({"q": choice("a", {"a": 0.5, "zzz": 0.5})}))
    with pytest.raises(AssertionError, match="options that were not offered"):
        await fake.run(CLEF, clef_q(CHOICE_Q), meta=META)
    ok_subset = answering(clef_response({"q": choice("a", {"a": 1.0})}))
    await ok_subset.run(CLEF, clef_q(CHOICE_Q), meta=META)


@pytest.mark.parametrize(
    "probabilities",
    [
        {"0": 0.5, "1": 0.5},
        {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.4},
        {"1": 0.3, "2": 0.3, "3": 0.4},
        {"a": 1.0},
    ],
)
async def test_score_probability_keys_must_be_the_level_indexes(
    probabilities: dict[str, float],
) -> None:
    fake = answering(clef_response({"q": score(1.0, probabilities)}))
    with pytest.raises(AssertionError, match="probabilities must be keyed"):
        await fake.run(CLEF, clef_q(SCORE_Q), meta=META)
    good = answering(clef_response({"q": score(1.0, {"0": 0.2, "1": 0.3, "2": 0.5})}))
    await good.run(CLEF, clef_q(SCORE_Q), meta=META)


async def test_clef_response_model_must_echo_the_body_model() -> None:
    payload = clef_q({"type": "noul", "instructions": "?"}, "clef-flash")
    with pytest.raises(AssertionError, match="response model"):
        await answering(clef_response({"q": noul(0.5)})).run(
            CLEF_FLASH, payload, meta=META
        )
    await answering(clef_response({"q": noul(0.5)}, model="clef-flash")).run(
        CLEF_FLASH, payload, meta=META
    )
    with pytest.raises(AssertionError, match="response model"):
        await answering(clef_response({"q": noul(0.5)}, model="clef-flash")).run(
            CLEF, clef_q({"type": "noul", "instructions": "?"}), meta=META
        )


def test_clef_response_builder_validates_the_model_name() -> None:
    assert clef_response({}, model="clef-flash")["model"] == "clef-flash"
    with pytest.raises(ValueError, match="clef"):
        clef_response({}, model="jev")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Builders against the published output schemas
# --------------------------------------------------------------------------- #


def test_builder_outputs_validate_against_the_published_output_schemas() -> None:
    clairic_validator = validator("clairic-output")
    clef_validator = validator("clef-output")
    for body in (
        clairic_response({}),
        clairic_response(
            {
                "a": clairic_answer("text", "x", span=(0, 1), guess="x"),
                "b": clairic_answer("list", ["x"], spans=[(0, 1)]),
                "c": clairic_answer("estimate", 3, mean=3.0, p10=1, p90=5),
                "d": clairic_answer("number", None, p_null=1.0),
            },
            truncated=True,
            state_text="x",
            input_tokens=7,
        ),
    ):
        clairic_validator.validate(body)
    for body in (
        clef_response({}),
        clef_response(
            {
                "a": noul(0.0),
                "b": choice("x", {"x": 0.6, "y": 0.4}),
                "c": choice("x", {"x": 0.6, "y": 0.4}, confidence=0.1),
                "d": score(1.5, {"0": 0.25, "1": 0.5, "2": 0.25}, legend={"0": "lo"}),
                "e": score(0.0, {"0": 1.0}),
            },
            model="clef-flash",
        ),
    ):
        clef_validator.validate(body)
