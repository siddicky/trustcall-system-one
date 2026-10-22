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
    ServerError,
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
    state_spans,
)
from trustcall_system_one._run import Run
from trustcall_system_one._wire import (
    ChoiceQuestion,
    ClairicField,
    NoulCriteria,
    NoulQuestion,
)
from trustcall_system_one.testing import FakeTransport, FakeWorld, noul
from trustcall_system_one.testing import clef_response as clef_reply
from trustcall_system_one.transport import CallMeta, FallbackTransport

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
        if meta.kind == "extract" and meta.entity == "e3":
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
        if meta.kind == "extract" and meta.entity == "e3"
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
    size = "Request requires 69639 tokens; the maximum is 65529"
    assert is_request_rejection(BadRequestError(size, status_code=422))
    assert is_request_rejection(BadRequestError(size, status_code=400))
    assert is_request_rejection(BadRequestError("payload too large", status_code=413))
    assert is_request_rejection(
        BadRequestError("error: max_tokens_exceeded", status_code=422)
    )
    assert is_request_rejection(RequestTooLargeError("x"))
    # any other client error is a bug that every request shares: it raises
    assert not is_request_rejection(BadRequestError("x", status_code=422))
    assert not is_request_rejection(
        BadRequestError("list fields don't take 'null_threshold'", status_code=422)
    )
    assert not is_request_rejection(BadRequestError(size, status_code=404))
    assert not is_request_rejection(BadRequestError(size))
    assert not is_request_rejection(AuthError("x", status_code=401))
    assert not is_request_rejection(UncertainOutcomeError("x"))
    assert is_size_rejection(BadRequestError(size, status_code=422))
    assert not is_size_rejection(BadRequestError("bad field", status_code=422))


@pytest.mark.parametrize(
    "mark",
    [
        {"maybe_sent": True},
        {"partial_usage": {"input_tokens": 5}},
    ],
)
def test_an_error_that_may_have_been_charged_is_never_a_rejection(
    mark: dict[str, Any],
) -> None:
    err = BadRequestError("HTTP 413: payload too large", status_code=413)
    for name, value in mark.items():
        setattr(err, name, value)
    assert not is_request_rejection(err) and not is_size_rejection(err)


def test_a_primary_that_may_have_run_makes_the_fallbacks_refusal_uncontainable() -> (
    None
):
    fallback_error = BadRequestError("HTTP 413: payload too large", status_code=413)
    fallback_error.primary_error = UncertainOutcomeError("timeout after send")
    assert not is_request_rejection(fallback_error)
    chained = BadRequestError("HTTP 413: payload too large", status_code=413)
    chained.__cause__ = ServerError("HTTP 503", status_code=503)
    assert not is_request_rejection(chained)


async def test_an_uncertain_primary_behind_a_fallback_is_raised() -> None:
    """probe_more `fallback`: the primary's request may have run; nothing is resent."""
    primary_calls: list[int] = []
    fallback_calls: list[int] = []

    class Primary(FakeTransport):
        def classify_failure(self, exc: Exception) -> str | None:
            return "uncertain" if isinstance(exc, UncertainOutcomeError) else None

    def primary(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        primary_calls.append(len(payload["questions"]))
        if len(payload["questions"]) > 4:
            raise UncertainOutcomeError(
                "read timeout after send; may have been charged"
            )
        return clef_reply(
            {q: noul(0.9) for q in payload["questions"]}, model=payload["model"]
        )

    def fallback(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        fallback_calls.append(len(payload["questions"]))
        raise BadRequestError(
            "Decisions API returned HTTP 413: payload too large", status_code=413
        )

    run = Run(
        transport=FallbackTransport(
            Primary(primary), FakeTransport(fallback), on=("unavailable", "uncertain")
        )
    )
    questions = {
        f"q{i}": NoulQuestion(
            type="noul",
            instructions=f"Is {i} even?",
            criteria=NoulCriteria(true="y", false="n"),
        )
        for i in range(8)
    }
    with pytest.raises(BadRequestError):
        await run.call_clef(questions, {"input": "x"}, meta=CallMeta(kind="gate"))
    assert primary_calls == [8] and fallback_calls == [8]  # no half was sent again
    assert not any("request_rejected" in w for w in run.warnings)


async def test_a_field_error_422_is_raised_not_contained() -> None:
    """A 422 about the request's FIELDS (code 3030) is a bug every request shares."""

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        raise BadRequestError(
            "Cloudflare API returned HTTP 422: list fields don't take "
            "'null_threshold' (3030)",
            status_code=422,
        )

    run = Run(transport=FakeTransport(handler))
    question = NoulQuestion(
        type="noul", instructions="?", criteria=NoulCriteria(true="y", false="n")
    )
    with pytest.raises(BadRequestError, match="3030"):
        await run.call_clef({"q": question}, {"input": "x"}, meta=CallMeta(kind="gate"))
    with pytest.raises(BadRequestError, match="3030"):
        await run.call_clairic(
            {"f": ClairicField(type="text", instructions="x")},
            "text",
            meta=CallMeta(kind="extract"),
        )


async def test_the_same_refusal_over_and_over_trips_the_circuit_breaker() -> None:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        raise BadRequestError(
            "HTTP 413: payload too large", status_code=413
        )  # every request is "too large"

    run = Run(transport=FakeTransport(handler))
    question = NoulQuestion(
        type="noul", instructions="?", criteria=NoulCriteria(true="y", false="n")
    )
    contained = 0
    with pytest.raises(BadRequestError):
        for n in range(30):
            state = {"input": f"text {n}"}  # a different request each time
            await run.call_clef({"q": question}, state, meta=CallMeta(kind="gate"))
            contained += 1
    assert contained == 8  # the ninth is raised


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


async def test_a_refused_coref_does_not_split_a_repeated_name_in_two() -> None:
    """probe_more `coref`: PN-0003 appears twice; a refused coref inserted it twice."""
    names = [f"PN-{i:04d}" for i in range(20)]
    rows = [f"| {i + 1} | {n} |" for i, n in enumerate(names)]
    rows.append("| 21 | PN-0003 (repeat of row 4) |")
    text = "\n".join(["| No | Part |", "|---|---|", *rows])
    world = FakeWorld(text)
    for n in names:
        world.entity(
            n,
            schema=Part,
            mentions=[(n, 0), (n, 1)] if n == "PN-0003" else [n],
            fields={"name": n},
        )
    handler = world.transport()._handler

    def refuse(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        if meta.kind == "coref":
            raise BadRequestError(
                "HTTP 422: Request requires 69639 tokens; the maximum is 65529",
                status_code=422,
            )
        return handler(model, payload, meta)

    for strategy in ("incremental", "pairwise"):
        result = await create_extractor(
            [Many(Part)],
            transport=FakeTransport(refuse),
            coref=CorefConfig(strategy=strategy),
        ).ainvoke(text)
        got = [r.value.name for r in result.records if isinstance(r.value, Part)]
        assert sorted(got) == names, strategy  # PN-0003 once, every other part once
        assert any(w.startswith("request_rejected") for w in result.warnings)


def test_a_records_own_span_key_is_never_read_or_rewritten() -> None:
    record = {"span": [2010, 2015], "note": "the span of years"}
    state = {
        "input": "x" * 100,
        "record": record,
        "records": {"r0": {"span": [1, 2]}},
        "proposed_change": {"to": {"span": [3, 4]}},
        "mentions": {"m0": {"text": "a", "span": [10, 12]}},
    }
    assert state_spans(state) == [(10, 12)]
    moved = rebase_spans(state, 100, [(5, 20)])
    assert moved["record"] == record and moved["records"] == state["records"]
    assert moved["proposed_change"] == state["proposed_change"]
    assert moved["mentions"]["m0"]["span"] == [7, 9]
