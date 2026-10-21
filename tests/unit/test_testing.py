"""Tests for FakeTransport and the response builders."""

from typing import Any

import pytest
from pydantic import ValidationError

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
from trustcall_system_one.transport import Transport

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
    assert result["answers"]["name"]["value"] == "Ada"
    assert fake.calls == [(CLAIRIC, CLAIRIC_PAYLOAD, META)]
    assert fake.last_retries == 0


async def test_async_handler_is_awaited() -> None:
    async def handler(
        model: str, payload: dict[str, Any], meta: CallMeta
    ) -> dict[str, Any]:
        return clef_response({"q": noul(0.9)}, model="clef-flash")

    fake = FakeTransport(handler)
    result = await fake.run(
        CLEF_FLASH, {**CLEF_PAYLOAD, "model": "clef-flash"}, meta=META
    )
    assert result["answers"]["q"]["noul"] == 0.9


async def test_handler_sees_a_copy_and_calls_keep_the_original() -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        payload["fields"].clear()
        return clairic_response({})

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
    assert parsed.answers["name"].span == (0, 3)
    assert parsed.answers["age"].value is None
    assert parsed.answers["tags"].spans == [(0, 1)]
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
