"""Sticky "maybe sent" of Cloudflare, the primary's spend, and saturated answers.

The rows of the table of probe ``exp_31bdaee_checks.py`` (a Cloudflare attempt that
may have run, followed by one that certainly did not) are tests here.
"""

import json
from typing import Any

import httpx
import pytest

from tests.unit.test_decisions import (
    ALL,
    EXAMPLE,
    META,
    REVIEW,
    decisions,
    experiential,
    review_questions,
)
from trustcall_system_one import (
    CLEF,
    CloudflareTransport,
    DecisionsTransport,
    FallbackTransport,
    ModelResponseError,
    RateLimitError,
    TransportError,
    UncertainOutcomeError,
)
from trustcall_system_one._config import GateConfig, Prices
from trustcall_system_one._decisions import (
    from_decisions_response,
    to_decisions_request,
)
from trustcall_system_one._diff import Change
from trustcall_system_one._gates import threshold
from trustcall_system_one._run import Run
from trustcall_system_one._wire import ClefRequest

PAYLOAD: dict[str, Any] = {
    "model": "clef",
    "state": "s",
    "questions": {"q": {"type": "noul", "instructions": "?"}},
}
RT = httpx.ReadTimeout("slow")
CE = httpx.ConnectError("refused")
R429 = httpx.Response(429, json={"errors": [{"message": "rl"}]})
R429_LONG = httpx.Response(
    429, json={"errors": [{"message": "rl"}]}, headers={"Retry-After": "3600"}
)
R503 = httpx.Response(503, json={"errors": [{"message": "down"}]})
R408 = httpx.Response(408, json={})


async def no_sleep(_: float) -> None:
    return None


def cloudflare(
    script: list[Any], **kwargs: Any
) -> tuple[CloudflareTransport, list[int]]:
    sent: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        item = script[min(len(sent), len(script) - 1)]
        sent.append(1)
        if isinstance(item, Exception):
            raise item
        response: httpx.Response = item
        return response

    transport = CloudflareTransport(
        account_id="a" * 32,
        api_token="t",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=no_sleep,
        **kwargs,
    )
    return transport, sent


# the probe's table: (attempts in order, retries allowed, what the last error is,
# its category, whether a request may have been run, attempts made)
ROWS = [
    ("connect errors only", [CE], 2, TransportError, "unavailable", False, 3),
    ("ReadTimeout x all", [RT], 2, TransportError, "uncertain", True, 3),
    ("503 x all", [R503], 2, TransportError, "uncertain", True, 3),
    ("408 x all", [R408], 2, TransportError, "uncertain", True, 3),
    ("429 x all", [R429], 2, RateLimitError, "unavailable", False, 3),
    (
        "ReadTimeout then ConnectError",
        [RT, CE, CE],
        2,
        TransportError,
        "uncertain",
        True,
        3,
    ),
    ("503 then ConnectError", [R503, CE, CE], 2, TransportError, "uncertain", True, 3),
    ("ReadTimeout then 429", [RT, R429, R429], 2, RateLimitError, "uncertain", True, 3),
    (
        "ReadTimeout then a 429 with a long Retry-After",
        [RT, R429_LONG],
        5,
        RateLimitError,
        "uncertain",
        True,
        2,
    ),
    (
        "RemoteProtocolError then ConnectError",
        [httpx.RemoteProtocolError("drop"), CE, CE],
        2,
        TransportError,
        "uncertain",
        True,
        3,
    ),
    (
        "ConnectError then 429",
        [CE, R429, R429],
        2,
        RateLimitError,
        "unavailable",
        False,
        3,
    ),
    (
        "ConnectTimeout then ConnectError",
        [httpx.ConnectTimeout("t"), CE, CE],
        2,
        TransportError,
        "unavailable",
        False,
        3,
    ),
]


@pytest.mark.parametrize(
    ("script", "retries", "kind", "category", "maybe_sent", "attempts"),
    [row[1:] for row in ROWS],
    ids=[row[0] for row in ROWS],
)
async def test_a_cloudflare_failure_remembers_every_attempt_that_may_have_run(
    script: list[Any],
    retries: int,
    kind: type[Exception],
    category: str,
    maybe_sent: bool,
    attempts: int,
) -> None:
    transport, sent = cloudflare(script, max_retries=retries)
    with pytest.raises(kind) as info:
        await transport.run(CLEF, PAYLOAD, meta=META)
    assert len(sent) == attempts
    assert info.value.maybe_sent is maybe_sent  # type: ignore[attr-defined]
    assert transport.classify_failure(info.value) == category


@pytest.mark.parametrize(
    ("script", "retries", "falls_back"),
    [
        ([RT, CE, CE], 2, False),  # the probe's wrong fallback
        ([R503, CE, CE], 2, False),
        ([RT, R429, R429], 2, False),
        ([CE], 2, True),
        ([R429], 1, True),
    ],
)
async def test_the_default_fallback_waits_for_certainty(
    script: list[Any], retries: int, falls_back: bool
) -> None:
    primary, _ = cloudflare(script, max_retries=retries)
    backup, backup_server = decisions()
    transport = FallbackTransport(primary, backup)
    if falls_back:
        assert (
            await transport.run(CLEF, PAYLOAD, meta=META)
        ).fallback_from == "cloudflare"
        assert len(backup_server.requests) == 1
    else:
        with pytest.raises(TransportError):
            await transport.run(CLEF, PAYLOAD, meta=META)
        assert backup_server.requests == []  # the request may have been run
    # with the opt-in a request that may have run falls back too
    again, _ = cloudflare(script, max_retries=retries)
    backup2, _ = decisions()
    answered = await FallbackTransport(
        again, backup2, on={"unavailable", "uncertain"}
    ).run(CLEF, PAYLOAD, meta=META)
    assert answered.provider == "decisions"


async def test_a_deadline_error_after_an_attempt_that_may_have_run_is_uncertain() -> (
    None
):
    # a timeout, then a 429 whose wait does not fit the deadline
    long_wait = httpx.Response(
        429, json={"errors": [{"message": "rl"}]}, headers={"Retry-After": "60"}
    )
    transport, _ = cloudflare([RT, long_wait], max_retries=5, call_deadline=30.0)
    with pytest.raises(TransportError, match="deadline") as info:
        await transport.run(CLEF, PAYLOAD, meta=META)
    assert info.value.maybe_sent is True
    assert transport.classify_failure(info.value) == "uncertain"
    # without an earlier attempt that may have run it is not
    transport, _ = cloudflare([long_wait], max_retries=5, call_deadline=30.0)
    with pytest.raises(TransportError, match="deadline") as info2:
        await transport.run(CLEF, PAYLOAD, meta=META)
    assert info2.value.maybe_sent is False
    assert transport.classify_failure(info2.value) == "unavailable"


async def test_a_slot_timeout_carries_the_flag_too() -> None:
    transport, _ = cloudflare([CE], max_retries=0, queue_timeout=0.01)
    # hold every slot so the next request waits for one and times out
    held = [transport.limiter.slot(CLEF) for _ in range(transport.limiter.limit(CLEF))]
    entered = []
    try:
        for slot in held:
            entered.append(slot)
            await slot.__aenter__()
        with pytest.raises(TransportError, match="request slot") as info:
            await transport.run(CLEF, PAYLOAD, meta=META)
        assert info.value.maybe_sent is False
        assert transport.classify_failure(info.value) == "unavailable"
    finally:
        for slot in entered:
            await slot.__aexit__(None, None, None)


# --------------------------------------------------------------------------- #
# The primary's charge when the fallback answers
# --------------------------------------------------------------------------- #


async def test_the_primarys_charge_and_id_are_logged_when_the_fallback_answers() -> (
    None
):
    # probe: the primary is charged $0.003 for a bad answer, the fallback $0.001
    bad = {
        "model": "clef",
        "usage": {"input_tokens": 5, "output_tokens": 0, "cost": 0.003},
    }
    primary, _ = experiential(
        httpx.Response(200, json=bad, headers={"x-correlation-id": "corr-prim"})
    )
    body = {
        "model": "gpt-6-luna-decisions",
        "answers": EXAMPLE["answers"],
        "usage": {"input_tokens": 9, "output_tokens": 0, "cost": 0.001},
    }
    backup, _ = decisions(httpx.Response(200, json=body))
    run = Run(
        transport=FallbackTransport(primary, backup, on={"unavailable", "bad_response"})
    )
    await run.call_clef(review_questions(), REVIEW["state"], meta=META)
    first, second = run.calls
    assert (first.provider, first.request_id, first.cost_usd) == (
        "experiential",
        "corr-prim",
        0.003,
    )
    assert first.error is not None and first.error.startswith("ModelResponseError")
    assert (second.provider, second.fallback_from, second.cost_usd) == (
        "decisions",
        "experiential",
        0.001,
    )
    assert run.usage.reported_cost_usd == pytest.approx(0.004)
    assert Prices().cost(run.usage).reported_usd == pytest.approx(0.004)
    assert sum(c.cost_usd or 0 for c in run.calls) == pytest.approx(0.004)


async def test_the_primarys_status_and_retries_are_logged_too() -> None:
    down = httpx.Response(
        503, json={"error": "down"}, headers={"x-request-id": "req-503"}
    )
    primary, _ = experiential(down)
    backup, _ = decisions()
    run = Run(transport=FallbackTransport(primary, backup, on=ALL))
    await run.call_clef(review_questions(), REVIEW["state"], meta=META)
    first = run.calls[0]
    assert (first.status_code, first.request_id, first.retries) == (503, "req-503", 0)
    assert first.provider == "experiential"


# --------------------------------------------------------------------------- #
# Saturated answers are bare picks
# --------------------------------------------------------------------------- #


def saturated_yes() -> dict[str, Any]:
    return {
        "model": "gpt-6-luna-decisions",
        "answers": [
            {
                "type": "choice",
                "name": "gate",
                "choice": "yes",
                "probabilities": [
                    {"value": "yes", "probability": 1.0},
                    {"value": "no", "probability": 0.0},
                ],
                "confidence": 1.0,
            }
        ],
        "usage": {"input_tokens": 9, "output_tokens": 0},
    }


async def test_a_saturated_yes_cannot_authorise_a_removal_but_passes_replace() -> None:
    backup, _ = decisions(httpx.Response(200, json=saturated_yes()))
    run = Run(transport=backup)
    questions = ClefRequest.model_validate(
        {
            "model": "clef",
            "state": "s",
            "questions": {"gate": {"type": "noul", "instructions": "?"}},
        }
    ).questions
    answers = await run.call_clef(questions, "s", meta=META)
    p = answers["gate"].noul  # type: ignore[union-attr]
    assert p == 0.8
    assert answers["gate"].model_extra["hard"] is True  # type: ignore[index]
    gates = GateConfig()
    removal = Change(op="remove", path="/name", field_path="name")
    clear = Change(op="replace", path="/name", value=None, field_path="name")
    replace = Change(op="replace", path="/name", value="x", field_path="name")
    add = Change(op="add", path="/name", value="x", field_path="name")
    assert p < threshold(gates, removal)  # refused
    assert p < threshold(gates, clear)  # a set-null is a removal too
    assert p >= threshold(gates, replace)  # passes
    assert p >= threshold(gates, add)  # passes
    assert gates.remove_threshold == 0.85


def test_every_saturated_shape_is_hard_and_capped() -> None:
    _, plans = to_decisions_request(REVIEW)
    clef = from_decisions_response(EXAMPLE, plans, clef_model="clef")
    answer = clef["answers"]["review"]
    assert answer["hard"] is True and answer["confidence"] == 0.8
    assert max(answer["probabilities"].values()) == 0.8
    # a graded answer is left alone, and so is its confidence (when lower)
    graded = json.loads(json.dumps(EXAMPLE))
    graded["answers"][0]["probabilities"] = [
        {"value": "accept", "probability": 0.1},
        {"value": "review", "probability": 0.9},
    ]
    answer = from_decisions_response(graded, plans, clef_model="clef")["answers"][
        "review"
    ]
    assert "hard" not in answer
    assert answer["probabilities"] == {"accept": pytest.approx(0.1), "review": 0.9}
    assert answer["confidence"] == 0.9  # the reported 1.0 cannot exceed the top


def test_confidence_follows_the_probabilities() -> None:
    _, plans = to_decisions_request(REVIEW)
    body = json.loads(json.dumps(EXAMPLE))
    body["answers"][0]["probabilities"] = [
        {"value": "accept", "probability": 0.5},
        {"value": "review", "probability": 0.5},
    ]
    answer = from_decisions_response(body, plans, clef_model="clef")["answers"][
        "review"
    ]
    assert answer["confidence"] == 0.5  # {0.5, 0.5} does not report 1.0
    body["answers"][0]["probabilities"] = [
        {"value": "accept", "probability": 0.2},
        {"value": "review", "probability": 1.0},
    ]  # sums to 1.2: normalised, the confidence follows the normalised top
    answer = from_decisions_response(body, plans, clef_model="clef")["answers"][
        "review"
    ]
    assert answer["confidence"] == pytest.approx(1.0 / 1.2)
    assert answer["probabilities"]["review"] == pytest.approx(1.0 / 1.2)


def test_the_default_cap_sits_below_the_removal_threshold() -> None:
    from trustcall_system_one._decisions import DEFAULT_HARD_CAP

    gates = GateConfig()
    assert DEFAULT_HARD_CAP == 0.8
    assert gates.replace_threshold <= DEFAULT_HARD_CAP < gates.remove_threshold
    assert DEFAULT_HARD_CAP >= 0.75  # the attribution keep threshold


# --------------------------------------------------------------------------- #
# A Decisions call that fails with no response is recorded under its model
# --------------------------------------------------------------------------- #


async def test_a_decisions_failure_without_a_response_is_filed_under_its_model() -> (
    None
):
    from trustcall_system_one._decisions import DECISIONS_MODEL

    backup, _ = decisions(httpx.ReadTimeout("slow"))
    run = Run(transport=backup)
    with pytest.raises(UncertainOutcomeError):
        await run.call_clef(review_questions(), REVIEW["state"], meta=META)
    (record,) = run.calls
    assert record.model == DECISIONS_MODEL and record.provider == "decisions"
    # a refusal before sending is too (no key)
    keyless = DecisionsTransport(api_key=None)
    run = Run(transport=keyless)
    with pytest.raises(Exception, match="API key"):
        await run.call_clef(review_questions(), REVIEW["state"], meta=META)
    assert run.calls[0].model == DECISIONS_MODEL
    # and a charged but bad response keeps its cost under that model
    bad = {
        "model": DECISIONS_MODEL,
        "answers": [],
        "usage": {"input_tokens": 3, "output_tokens": 0, "cost": 0.002},
    }
    broken, _ = decisions(httpx.Response(200, json=bad))
    run = Run(transport=broken)
    with pytest.raises(ModelResponseError):
        await run.call_clef(review_questions(), REVIEW["state"], meta=META)
    assert run.usage.by_model[DECISIONS_MODEL].reported_cost_usd == 0.002
