"""The Clef request budget (every request is estimated and kept within it) and the
containment of a request the provider refuses (one 422 must not abort the invoke).

The live trigger: table-heavy spec sheets made a coref incremental assignment of
16 questions over 250 entities (69,639 tokens against a maximum of 65,529); the
HTTP 422 aborted the whole ``invoke``.
"""

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    AuthError,
    BadRequestError,
    CorefConfig,
    Limits,
    Many,
    UncertainOutcomeError,
    create_extractor,
)
from trustcall_system_one._errors import (
    RequestTooLargeError,
    is_request_rejection,
    is_size_rejection,
)
from trustcall_system_one._render import (
    estimate_request_tokens,
    estimate_text_tokens,
    estimate_tokens,
    rebase_spans,
)
from trustcall_system_one._run import Run
from trustcall_system_one._wire import ChoiceQuestion, NoulCriteria, NoulQuestion
from trustcall_system_one.testing import FakeTransport, FakeWorld, noul
from trustcall_system_one.testing import clef_response as clef_reply
from trustcall_system_one.transport import CallMeta

BUDGET = 56_000


class Part(BaseModel):
    """A part number listed in a table."""

    name: str = Field(description="the part number")


def table_world(n: int = 120) -> tuple[FakeWorld, str, list[str]]:
    """A table-heavy sheet naming ``n`` distinct parts, one per row."""
    names = [f"PN-{i:04d}" for i in range(n)]
    rows = [
        f"| {i + 1:>3} | {name} | CHECK {i * 37 % 1000:>6} | {i * 91 % 9999:>7} FH |"
        for i, name in enumerate(names)
    ]
    text = "\n".join(
        ["| No | Part | Check | Hours |", "| --- | --- | --- | --- |", *rows]
    )
    world = FakeWorld(text)
    for name in names:
        world.entity(name, schema=Part, mentions=[name], fields={"name": name})
    return world, text, names


def sized(transport: FakeTransport) -> list[tuple[str, float]]:
    """The estimated size of every Clef request a transport received."""
    return [
        (meta.kind, estimate_request_tokens(body["state"], body["questions"]))
        for model, body, meta in transport.calls
        if "questions" in body
    ]


async def test_a_table_heavy_sheet_never_sends_a_request_over_budget() -> None:
    world, text, names = table_world()
    unguarded = world.transport()
    loose = create_extractor(
        [Many(Part)],
        transport=unguarded,
        coref=CorefConfig(strategy="incremental"),
        limits=Limits(clef_state_token_budget=10_000_000),
    )
    await loose.ainvoke(text)
    # the point of the test: without a budget the same sheet DOES build such requests
    assert max(size for _, size in sized(unguarded)) > BUDGET

    world, text, names = table_world()  # a transport keeps every call it saw
    transport = world.transport()
    extractor = create_extractor(
        [Many(Part)], transport=transport, coref=CorefConfig(strategy="incremental")
    )
    result = await extractor.ainvoke(text)
    sizes = sized(transport)
    assert sizes and max(size for _, size in sizes) <= BUDGET
    assert any(kind == "coref" for kind, _ in sizes)
    assert (
        sorted(r.value.name for r in result.records if isinstance(r.value, Part))
        == names
    )
    assert any("coref_options_capped" in w for w in result.warnings)


async def test_a_422_on_one_call_leaves_the_other_entities_intact() -> None:
    world, text, names = table_world(12)
    victim = names[3]
    handler = world.transport()._handler  # the world's own answers

    def refuse_one(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        if meta.kind == "extract" and victim in payload["state"]:
            raise BadRequestError(
                "Cloudflare API returned HTTP 422: AiError: Request requires "
                "69639 tokens; the maximum is 65529",
                status_code=422,
            )
        return handler(model, payload, meta)

    transport = FakeTransport(refuse_one)
    extractor = create_extractor([Many(Part)], transport=transport)
    result = await extractor.ainvoke(text)
    got = {r.value.name for r in result.records if isinstance(r.value, Part)}
    assert got >= set(names) - {victim}
    assert any(w.startswith("request_rejected") for w in result.warnings)
    # the refused request was not sent a second time
    asked = [
        json.dumps(body, sort_keys=True)
        for _, body, meta in transport.calls
        if meta.kind == "extract" and victim in body["state"]
    ]
    assert asked and len(asked) == len(set(asked))


async def test_a_422_on_a_coref_request_keeps_the_mentions_as_separate_entities() -> (
    None
):
    world, text, names = table_world(20)
    handler = world.transport()._handler

    def refuse_coref(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        if meta.kind == "coref":
            raise BadRequestError(
                "Cloudflare API returned HTTP 422: Request requires 69639 tokens; "
                "the maximum is 65529",
                status_code=422,
            )
        return handler(model, payload, meta)

    extractor = create_extractor(
        [Many(Part)],
        transport=FakeTransport(refuse_coref),
        coref=CorefConfig(strategy="incremental"),
    )
    result = await extractor.ainvoke(text)
    assert len([r for r in result.records if r.value is not None]) == len(names)
    assert any(w.startswith("request_rejected") for w in result.warnings)


async def test_a_size_rejection_is_sent_again_as_two_smaller_requests() -> None:
    seen: list[int] = []

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        count = len(payload["questions"])
        seen.append(count)
        if count > 4:
            raise BadRequestError(
                "HTTP 422: Request requires 70000 tokens; the maximum is 65529",
                status_code=422,
            )
        return clef_reply(
            {q: noul(0.9) for q in payload["questions"]}, model=payload["model"]
        )

    run = Run(transport=FakeTransport(handler))
    questions = {
        f"q{i}": NoulQuestion(
            type="noul",
            instructions=f"Is {i} even?",
            criteria=NoulCriteria(true="yes", false="no"),
        )
        for i in range(8)
    }
    answers = await run.call_clef(questions, {"input": "x"}, meta=CallMeta(kind="gate"))
    assert seen == [8, 4, 4]  # the refused request is never repeated
    assert set(answers) == set(questions)
    assert any("clef_request_resplit" in w for w in run.warnings)


@pytest.mark.parametrize("error", [AuthError("bad key", status_code=401)])
async def test_auth_errors_still_abort(error: Exception) -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        raise error

    run = Run(transport=FakeTransport(handler))
    question = NoulQuestion(
        type="noul", instructions="?", criteria=NoulCriteria(true="y", false="n")
    )
    with pytest.raises(AuthError):
        await run.call_clef({"q": question}, {"input": "x"}, meta=CallMeta(kind="gate"))


async def test_an_uncertain_outcome_is_never_swallowed_or_repeated() -> None:
    calls: list[int] = []

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        calls.append(1)
        raise UncertainOutcomeError("timed out after send")

    run = Run(transport=FakeTransport(handler))
    question = NoulQuestion(
        type="noul", instructions="?", criteria=NoulCriteria(true="y", false="n")
    )
    with pytest.raises(UncertainOutcomeError):
        await run.call_clef({"q": question}, {"input": "x"}, meta=CallMeta(kind="gate"))
    assert len(calls) == 1


async def test_strict_mode_raises_a_refused_request() -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        raise BadRequestError("HTTP 422: too many tokens", status_code=422)

    run = Run(transport=FakeTransport(handler), strict=True)
    question = NoulQuestion(
        type="noul", instructions="?", criteria=NoulCriteria(true="y", false="n")
    )
    with pytest.raises(BadRequestError):
        await run.call_clef({"q": question}, {"input": "x"}, meta=CallMeta(kind="gate"))


async def test_a_not_found_is_not_a_request_rejection() -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        raise BadRequestError("HTTP 404: no such model", status_code=404)

    run = Run(transport=FakeTransport(handler))
    question = NoulQuestion(
        type="noul", instructions="?", criteria=NoulCriteria(true="y", false="n")
    )
    with pytest.raises(BadRequestError):
        await run.call_clef({"q": question}, {"input": "x"}, meta=CallMeta(kind="gate"))


def test_which_errors_count_as_a_refused_request() -> None:
    assert is_request_rejection(BadRequestError("x", status_code=422))
    assert is_request_rejection(BadRequestError("x", status_code=413))
    assert is_request_rejection(RequestTooLargeError("x"))
    assert not is_request_rejection(BadRequestError("x", status_code=404))
    assert not is_request_rejection(BadRequestError("x"))
    assert not is_request_rejection(AuthError("x", status_code=401))
    assert not is_request_rejection(UncertainOutcomeError("x"))
    assert is_size_rejection(
        BadRequestError(
            "Request requires 69639 tokens; the maximum is 65529", status_code=422
        )
    )
    assert not is_size_rejection(BadRequestError("bad field", status_code=422))


# --------------------------------------------------------------------------- #
# The estimate and the central guard
# --------------------------------------------------------------------------- #


def test_a_choice_option_costs_per_question_in_the_estimate() -> None:
    options = {f"c{i}": f"Entity {i}" for i in range(100)}
    options["__new__"] = "none"
    question = ChoiceQuestion(type="choice", instructions="Which?", criteria=options)
    one = estimate_request_tokens({"input": "x"}, {"a": question})
    sixteen = estimate_request_tokens(
        {"input": "x"}, {f"a{i}": question for i in range(16)}
    )
    assert sixteen > 15 * (one - estimate_request_tokens({"input": "x"}))
    assert estimate_tokens({"input": "x"}) < 10  # the old estimate saw none of it


def test_dense_table_text_costs_more_per_character_than_padded_text() -> None:
    dense = "| 17 | LPT STAGE 6 DISK | 3A2996 | 33`942 | 15`323 | 20`000 |\n" * 200
    padded = ("| Name" + " " * 40 + "|" + "-" * 40 + "|\n") * 200
    assert len(dense) / estimate_text_tokens(dense) < 2.6
    assert len(padded) / estimate_text_tokens(padded) > 3.5


def test_rebase_spans_maps_spans_into_a_narrowed_input_and_drops_the_rest() -> None:
    state = {
        "input": "unchanged",
        "mentions": {"m0": {"text": "a", "span": [105, 106]}},
        "clusters": {"c0": {"mentions": [{"text": "far", "span": [900, 903]}]}},
    }
    moved = rebase_spans(state, 1000, [(100, 120)])
    assert moved["input"] == "unchanged"
    assert moved["mentions"]["m0"]["span"] == [7, 8]  # after the leading "…\n"
    assert "span" not in moved["clusters"]["c0"]["mentions"][0]
    assert moved["clusters"]["c0"]["mentions"][0]["text"] == "far"


async def test_the_guard_splits_questions_and_narrows_the_input() -> None:
    sent: list[dict[str, Any]] = []

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        sent.append(payload)
        return clef_reply(
            {q: noul(0.5) for q in payload["questions"]}, model=payload["model"]
        )

    run = Run(
        transport=FakeTransport(handler), limits=Limits(clef_state_token_budget=3_000)
    )
    text = "filler words go here. " * 400 + "NEEDLE is here. " + "more filler. " * 400
    start = text.index("NEEDLE")
    questions = {
        f"q{i}": NoulQuestion(
            type="noul",
            instructions="Is the text about `entity.label`? " + "pad " * 40,
            criteria=NoulCriteria(true="yes", false="no"),
        )
        for i in range(30)
    }
    state = {
        "input": text,
        "entity": {
            "label": "NEEDLE",
            "mentions": [{"text": "NEEDLE", "span": [start, start + 6]}],
        },
    }
    answers = await run.call_clef(questions, state, meta=CallMeta(kind="gate"))
    assert set(answers) == set(questions)
    assert len(sent) > 1
    for payload in sent:
        assert estimate_request_tokens(payload["state"], payload["questions"]) <= 3_000
        assert "NEEDLE" in payload["state"]["input"]
    assert any(w.startswith("clef_request_split") for w in run.warnings)


async def test_a_request_that_cannot_fit_is_dropped_before_it_is_sent() -> None:
    sent: list[Any] = []

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        sent.append(payload)
        return clef_reply(
            {q: noul(0.5) for q in payload["questions"]}, model=payload["model"]
        )

    run = Run(
        transport=FakeTransport(handler), limits=Limits(clef_state_token_budget=1_000)
    )
    huge = NoulQuestion(
        type="noul",
        instructions="x " * 2_000,
        criteria=NoulCriteria(true="yes", false="no"),
    )
    answers = await run.call_clef(
        {"q": huge}, {"input": "short"}, meta=CallMeta(kind="gate")
    )
    assert answers == {} and sent == []
    assert any(w.startswith("request_rejected") for w in run.warnings)


async def test_the_transport_may_advertise_a_smaller_budget() -> None:
    class Small(FakeTransport):
        state_token_budget = 1_500

    run = Run(transport=Small(lambda m, p, c: None))
    assert run.clef_budget() == 1_500
    assert Run(transport=FakeTransport(lambda m, p, c: None)).clef_budget() == BUDGET
