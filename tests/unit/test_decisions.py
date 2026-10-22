"""The Decisions API as a fallback for Clef: translation, transport and fallback.

Every request goes to ``httpx.MockTransport``. The shape of a Decisions response comes
from one live probe (``docs/schemas/experiential-decisions-response.example.json``);
the other question types and the answers without probabilities are not observed, so
they are tested from what the translation documents.
"""

import asyncio
import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from evals.cli import _parser, _settings
from evals.runner import Settings, _have_credentials, count_unknown_outcomes
from tests.unit.schema_helpers import validator
from tests.unit.test_experiential import Tag
from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CLEF_FLASH,
    AuthError,
    BadRequestError,
    CallMeta,
    CloudflareTransport,
    DecisionsTransport,
    ExperientialTransport,
    FallbackTransport,
    ModelResponseError,
    RateLimitError,
    RoutingTransport,
    ServerError,
    TransportError,
    TransportResponse,
    UncertainOutcomeError,
    clef_fallback_from_env,
    create_extractor,
    system_one_transport,
)
from trustcall_system_one._decisions import (
    DECISIONS_MODEL,
    DecisionsRequest,
    from_decisions_response,
    render_state,
    to_decisions_request,
)
from trustcall_system_one._run import FALLBACK_WARNING, Run
from trustcall_system_one._wire import ClefRequest, parse_clef
from trustcall_system_one.transport import DEFAULT_FALLBACK_ON

KEY = "xpl_" + "dc12ba34" * 5
TOKEN = re.compile(r'<<<([0-9a-f]{16}) "')
ALL = {"unavailable", "uncertain", "bad_response"}
META = CallMeta(kind="gate", schema="Person", entity="Ada Lovelace")
DECISIONS_URL = "https://api.experientiallabs.ai/v1/decisions"
EXAMPLE: dict[str, Any] = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "docs/schemas/experiential-decisions-response.example.json"
    ).read_text()
)

# The user's probe request, as a Clef request.
REVIEW: dict[str, Any] = {
    "model": "clef",
    "state": (
        "Review a proposed change before a human decides whether to merge it. "
        "Change: reject an empty project name before writing a record. Tests: the "
        "new empty-name test and existing creation tests pass; no integration tests "
        "were run."
    ),
    "questions": {
        "review": {
            "type": "choice",
            "instructions": "Does the supplied evidence support accepting this change?",
            "criteria": {
                "accept": "The described change meets the task.",
                "review": "Evidence is missing.",
            },
        }
    },
}
MIXED: dict[str, Any] = {
    "model": "clef-flash",
    "state": {
        "input": "Ada Lovelace wrote the first program.\nShe lived in London.",
        "entity": {"label": "the Person", "name": "Ada"},
    },
    "questions": {
        "z_last": {
            "type": "noul",
            "instructions": "Is Ada the author?",
            "criteria": {"true": "Ada is the author.", "false": "Someone else is."},
        },
        "a_first": {
            "type": "choice",
            "instructions": {"question": "Where?", "options_note": "pick one"},
            "criteria": {"london": "London", "paris": None, "rome": {"city": "Rome"}},
        },
        "m_score": {
            "type": "score",
            "instructions": "How famous?",
            "criteria": ["unknown", "known", ["very", "famous"]],
        },
    },
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EXPERIENTIAL_API_KEY",
        "TRUSTCALL_S1_CLEF_PROVIDER",
        "TRUSTCALL_S1_CLEF_FALLBACK",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# A fake Decisions server
# --------------------------------------------------------------------------- #


def answer_for(question: Mapping[str, Any], pick: int = 0) -> dict[str, Any]:
    """The answer a model could give: ``pick`` gets 0.7, the rest share 0.3."""
    values = [c["value"] for c in question["choices"]]
    rest = 0.3 / (len(values) - 1)
    return {
        "type": "choice",
        "name": question["name"],
        "choice": values[pick],
        "probabilities": [
            {"value": v, "probability": 0.7 if i == pick else rest}
            for i, v in enumerate(values)
        ],
        "confidence": 0.7,
    }


def serve(request: httpx.Request) -> dict[str, Any]:
    body = json.loads(request.content)
    return {
        "model": body["model"],
        "answers": [answer_for(q) for q in body["questions"]],
        "usage": {
            "input_tokens": 50,
            "output_tokens": 0,
            "cost": 0.0,
            "is_byok": False,
        },
        "provider": "openai",
    }


class Server:
    """A MockTransport handler that records requests and answers from ``serve``."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.delays: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            return httpx.Response(200, json=serve(request))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)


def decisions(
    *responses: httpx.Response | Exception, **kwargs: Any
) -> tuple[DecisionsTransport, Server]:
    server = Server(*responses)
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    kwargs.setdefault("api_key", KEY)
    return DecisionsTransport(client=client, sleep=server.sleep, **kwargs), server


def experiential(
    *responses: httpx.Response | Exception, **kwargs: Any
) -> tuple[ExperientialTransport, Server]:
    server = Server(*responses)
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    kwargs.setdefault("api_key", KEY)
    return ExperientialTransport(client=client, sleep=server.sleep, **kwargs), server


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #


def test_the_users_example_is_reproduced() -> None:
    body, plans = to_decisions_request(REVIEW)
    assert body == {
        "model": "gpt-6-luna-decisions",
        "input": REVIEW["state"],
        "questions": [
            {
                "type": "choice",
                "name": "review",
                "instructions": REVIEW["questions"]["review"]["instructions"],
                "choices": [
                    {
                        "value": "accept",
                        "description": "The described change meets the task.",
                    },
                    {"value": "review", "description": "Evidence is missing."},
                ],
            }
        ],
    }
    assert [(p.qid, p.kind, p.values) for p in plans] == [
        ("review", "choice", ("accept", "review"))
    ]


def test_the_model_is_configurable() -> None:
    body, _ = to_decisions_request(REVIEW, model="another-model")
    assert body["model"] == "another-model"
    assert DECISIONS_MODEL == "gpt-6-luna-decisions"


def test_a_structured_state_becomes_fenced_sections() -> None:
    body, _ = to_decisions_request(MIXED)
    text = body["input"]
    token = TOKEN.search(text).group(1)  # type: ignore[union-attr]
    # the user's text is verbatim (real newlines, no JSON escaping) in its section
    assert (
        f'<<<{token} "input">>>\nAda Lovelace wrote the first program.\n'
        f"She lived in London.\n<<<END {token}>>>"
    ) in text
    assert (
        f'<<<{token} "entity">>>\n{{\n  "label": "the Person",\n  "name": "Ada"\n}}\n'
        f"<<<END {token}>>>"
    ) in text
    assert text.index('"input"') < text.index('"entity"')
    assert text.startswith("The sections below are fenced")
    # the same request is rendered the same way
    assert to_decisions_request(MIXED)[0]["input"] == text


def test_render_state_forms() -> None:
    assert render_state("plain") == "plain"
    assert render_state({}) == "{}"
    assert render_state([1, "é"]) == '[\n  1,\n  "\u00e9"\n]'
    assert render_state(3) == "3"
    text = render_state({"a": "x", "b": [1]})
    token = TOKEN.search(text).group(1)  # type: ignore[union-attr]
    assert f'<<<{token} "a">>>\nx\n<<<END {token}>>>' in text
    assert f'<<<{token} "b">>>\n[\n  1\n]\n<<<END {token}>>>' in text


def test_questions_keep_their_order_and_ids() -> None:
    body, plans = to_decisions_request(MIXED)
    assert [q["name"] for q in body["questions"]] == ["z_last", "a_first", "m_score"]
    assert [p.qid for p in plans] == ["z_last", "a_first", "m_score"]
    assert all(q["type"] == "choice" for q in body["questions"])


def test_noul_becomes_yes_and_no() -> None:
    body, plans = to_decisions_request(MIXED)
    noul = body["questions"][0]
    assert noul["choices"] == [
        {"value": "yes", "description": "Ada is the author."},
        {"value": "no", "description": "Someone else is."},
    ]
    assert plans[0].kind == "noul"
    bare = {
        **REVIEW,
        "questions": {"q": {"type": "noul", "instructions": "Is it?"}},
    }
    (question,) = to_decisions_request(bare)[0]["questions"]
    assert [c["value"] for c in question["choices"]] == ["yes", "no"]
    assert all(c["description"] for c in question["choices"])


def test_choice_criteria_become_choices() -> None:
    body, plans = to_decisions_request(MIXED)
    choice = body["questions"][1]
    # a structured instruction is fenced like a state
    token = TOKEN.search(choice["instructions"]).group(1)  # type: ignore[union-attr]
    assert (
        f'<<<{token} "question">>>\nWhere?\n<<<END {token}>>>' in choice["instructions"]
    )
    assert f'<<<{token} "options_note">>>\npick one\n<<<END' in choice["instructions"]
    assert choice["choices"] == [
        {"value": "london", "description": "London"},
        {"value": "paris"},  # no description to give
        {"value": "rome", "description": '{\n  "city": "Rome"\n}'},
    ]
    assert plans[1].values == ("london", "paris", "rome")


def test_score_levels_become_numbered_choices() -> None:
    body, plans = to_decisions_request(MIXED)
    score = body["questions"][2]
    assert [c["value"] for c in score["choices"]] == ["0", "1", "2"]
    assert score["choices"][2]["description"] == '[\n  "very",\n  "famous"\n]'
    assert score["instructions"].startswith("How famous?")
    assert "0 (lowest) to 2 (highest)" in score["instructions"]
    assert plans[2].kind == "score"
    assert plans[2].legend == ("unknown", "known", ["very", "famous"])


def test_the_translated_request_is_valid_both_ways() -> None:
    validator("clef-input").validate(MIXED)
    validator("clef-input").validate(REVIEW)
    for payload in (MIXED, REVIEW):
        body, _ = to_decisions_request(payload)
        assert DecisionsRequest.model_validate(body).to_payload() == body
        json.dumps(body, allow_nan=False)


def test_unsupported_and_invalid_requests_are_refused_clearly() -> None:
    with pytest.raises(BadRequestError, match="cannot take images"):
        to_decisions_request(
            {**REVIEW, "images": ["data:image/png;base64,AAAA"]},
        )
    with pytest.raises(BadRequestError, match="not a valid Clef request"):
        to_decisions_request({**REVIEW, "model": "clairic"})
    with pytest.raises(BadRequestError, match="not a valid Clef request"):
        to_decisions_request({**REVIEW, "questions": {}})
    with pytest.raises(BadRequestError):
        to_decisions_request({"model": "clef"})


# --------------------------------------------------------------------------- #
# Response translation
# --------------------------------------------------------------------------- #


def test_the_observed_response_becomes_a_clef_choice_answer() -> None:
    _, plans = to_decisions_request(REVIEW)
    clef = from_decisions_response(EXAMPLE, plans, clef_model="clef")
    parsed = parse_clef(clef)
    validator("clef-output").validate(clef)
    assert parsed.model == "clef"
    answer = clef["answers"]["review"]
    # the live answer is saturated (0.0 / 1.0, confidence 1.0): a bare pick, so it is
    # tagged hard and capped below the remove / set-null gate (0.85)
    assert answer == {
        "type": "choice",
        "choice": "review",
        "probabilities": {"accept": pytest.approx(0.2), "review": 0.8},
        "confidence": 0.8,
        "provider": "decisions",
        "hard": True,
    }
    assert clef["usage"] == {"input_tokens": 201, "output_tokens": 0, "cost": 0.0}
    assert (clef["provider"], clef["decisions_model"]) == ("decisions", DECISIONS_MODEL)


def response_for(
    plans_payload: Mapping[str, Any], answers: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "model": DECISIONS_MODEL,
        "answers": answers,
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


def probs(**given: float) -> list[dict[str, Any]]:
    return [{"value": k, "probability": v} for k, v in given.items()]


def test_noul_is_the_probability_of_yes() -> None:
    payload = {**REVIEW, "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    _, plans = to_decisions_request(payload)

    def noul(**kwargs: Any) -> Any:
        body = response_for(payload, [{"type": "choice", "name": "q", **kwargs}])
        return from_decisions_response(body, plans, clef_model="clef")["answers"]["q"]

    assert noul(choice="yes", probabilities=probs(yes=0.8, no=0.2)) == {
        "type": "noul",
        "noul": 0.8,
        "provider": "decisions",
    }
    # probabilities that do not sum to one (beyond 0.02) are normalised
    assert math.isclose(
        noul(choice="no", probabilities=probs(yes=0.1, no=0.3))["noul"], 0.25
    )
    # a missing option has probability 0 and the rest is normalised: 1.0 is a bare
    # yes, which is capped
    assert noul(choice="yes", probabilities=probs(yes=0.6))["noul"] == 0.8
    assert noul(choice="yes", probabilities=probs(yes=0.9, no=0.2))[
        "noul"
    ] == pytest.approx(0.9 / 1.1)
    # within 0.02 they are left alone
    assert noul(choice="yes", probabilities=probs(yes=0.51, no=0.48))["noul"] == 0.51


def test_an_answer_without_probabilities_is_hard_capped_and_tagged() -> None:
    payload = MIXED
    _, plans = to_decisions_request(payload)
    body = response_for(
        payload,
        [
            {"type": "choice", "name": "z_last", "choice": "yes"},
            {"type": "choice", "name": "a_first", "choice": "paris"},
            {"type": "choice", "name": "m_score", "choice": "2"},
        ],
    )
    clef = from_decisions_response(body, plans, clef_model="clef-flash")
    parse_clef(clef)
    validator("clef-output").validate(clef)
    assert clef["model"] == "clef-flash"
    answers = clef["answers"]
    # a bare pick is never 1.0: the cap (0.8) shows in the numbers and in the tag
    assert answers["z_last"] == {
        "type": "noul",
        "noul": 0.8,
        "provider": "decisions",
        "hard": True,
    }
    assert answers["a_first"]["probabilities"] == {
        "london": pytest.approx(0.1),
        "paris": 0.8,
        "rome": pytest.approx(0.1),
    }
    assert answers["a_first"]["confidence"] == 0.8
    assert answers["m_score"]["score"] == pytest.approx(0.1 + 1.6)
    assert all(a["hard"] is True for a in answers.values())
    no = response_for(
        payload,
        [
            {"type": "choice", "name": n, "choice": c}
            for n, c in [("z_last", "no"), ("a_first", "london"), ("m_score", "0")]
        ],
    )
    assert from_decisions_response(no, plans, clef_model="clef")["answers"]["z_last"][
        "noul"
    ] == pytest.approx(0.2)
    # the cap is a knob; 1.0 gives a bare pick full weight
    full = from_decisions_response(body, plans, clef_model="clef", hard_cap=1.0)
    assert full["answers"]["z_last"]["noul"] == 1.0


def test_a_score_is_the_probability_weighted_level_with_its_legend() -> None:
    payload = {
        **REVIEW,
        "questions": {
            "s": {
                "type": "score",
                "instructions": "Rate.",
                "criteria": ["bad", "ok", "good"],
            }
        },
    }
    _, plans = to_decisions_request(payload)
    body = response_for(
        payload,
        [
            {
                "type": "choice",
                "name": "s",
                "choice": "1",
                "probabilities": probs(**{"0": 0.2, "1": 0.5, "2": 0.3}),
                "confidence": 0.5,
            }
        ],
    )
    clef = from_decisions_response(body, plans, clef_model="clef")
    answer = clef["answers"]["s"]
    assert math.isclose(answer["score"], 0.5 + 0.6)
    assert answer["legend"] == {"0": "bad", "1": "ok", "2": "good"}
    assert answer["probabilities"] == {"0": 0.2, "1": 0.5, "2": 0.3}
    assert answer["confidence"] == 0.5
    parse_clef(clef)
    validator("clef-output").validate(clef)


def test_answers_come_back_in_the_order_of_the_questions() -> None:
    _, plans = to_decisions_request(MIXED)
    body, _ = to_decisions_request(MIXED)
    shuffled = response_for(MIXED, [answer_for(q) for q in reversed(body["questions"])])
    clef = from_decisions_response(shuffled, plans, clef_model="clef-flash")
    assert list(clef["answers"]) == ["z_last", "a_first", "m_score"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b["answers"].pop(), "missing"),
        (lambda b: b["answers"].append({**b["answers"][0], "name": "extra"}), "extra"),
        (lambda b: b["answers"].append(dict(b["answers"][0])), "answered twice"),
        (lambda b: b["answers"][0].update(choice="maybe"), "not an option"),
        (
            lambda b: b["answers"][0].update(probabilities=probs(maybe=1.0)),
            "unknown options",
        ),
        (lambda b: b.pop("usage"), "does not match the schema"),
        (lambda b: b.update(answers="none"), "does not match the schema"),
    ],
)
def test_a_response_that_does_not_fit_the_questions_is_refused(
    mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    _, plans = to_decisions_request(REVIEW)
    body = json.loads(json.dumps(EXAMPLE))
    mutate(body)
    with pytest.raises(ModelResponseError, match=message):
        from_decisions_response(body, plans, clef_model="clef")


def test_a_response_that_is_not_an_object_is_refused() -> None:
    _, plans = to_decisions_request(REVIEW)
    with pytest.raises(ModelResponseError):
        from_decisions_response(["no"], plans, clef_model="clef")


def test_a_round_trip_through_a_fake_server_for_every_question_type() -> None:
    for payload in (MIXED, REVIEW):
        body, plans = to_decisions_request(payload)
        server_reply = {
            "model": body["model"],
            "answers": [answer_for(q, pick=1) for q in body["questions"]],
            "usage": {"input_tokens": 9, "output_tokens": 0, "cost": 0.5},
        }
        clef = from_decisions_response(server_reply, plans, clef_model=payload["model"])
        validator("clef-output").validate(clef)
        parsed = parse_clef(clef)
        wanted = ClefRequest.model_validate(payload).questions
        assert list(parsed.answers) == list(wanted)
        for qid, question in wanted.items():
            assert parsed.answers[qid].type == question.type
        assert parsed.usage.cost == 0.5


# --------------------------------------------------------------------------- #
# DecisionsTransport
# --------------------------------------------------------------------------- #


async def test_one_post_to_the_decisions_endpoint_returns_a_clef_response() -> None:
    headers = {"x-correlation-id": "corr-1"}
    transport, server = decisions(httpx.Response(200, json=EXAMPLE, headers=headers))
    result = await transport.run(CLEF, REVIEW, meta=META)
    (request,) = server.requests
    assert str(request.url) == DECISIONS_URL
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    sent = json.loads(request.content)
    assert sent == to_decisions_request(REVIEW)[0]
    assert "Ada Lovelace" not in request.content.decode()  # CallMeta stays local
    assert parse_clef(result.body).answers["review"].type == "choice"
    assert (result.provider, result.request_id, result.status_code) == (
        "decisions",
        "corr-1",
        200,
    )
    assert result.retries == 0


@pytest.mark.parametrize("model", [CLEF, CLEF_FLASH, "clef", "clef-flash"])
async def test_it_serves_clef_models_only(model: str) -> None:
    transport, server = decisions()
    payload = {**REVIEW, "model": "clef-flash" if "flash" in model else "clef"}
    result = await transport.run(model, payload, meta=META)
    assert result.body["model"] == payload["model"]
    assert transport.serves(model)
    assert len(server.requests) == 1


@pytest.mark.parametrize("model", [CLAIRIC, "clairic", "gpt-4"])
async def test_other_models_are_refused_before_sending(model: str) -> None:
    transport, server = decisions()
    assert not transport.serves(model)
    with pytest.raises(BadRequestError, match="Clef only"):
        await transport.run(model, REVIEW, meta=META)
    assert server.requests == []


async def test_images_are_refused_before_sending() -> None:
    transport, server = decisions()
    with pytest.raises(BadRequestError, match="images"):
        await transport.run(
            CLEF, {**REVIEW, "images": ["data:image/png;base64,AAAA"]}, meta=META
        )
    assert server.requests == []


async def test_model_and_url_are_configurable() -> None:
    transport, server = decisions(
        decisions_model="other-decisions", base_url="https://example.test/d/"
    )
    await transport.run(CLEF, REVIEW, meta=META)
    assert str(server.requests[0].url) == "https://example.test/d"
    assert json.loads(server.requests[0].content)["model"] == "other-decisions"
    with pytest.raises(ValueError, match="decisions_model"):
        DecisionsTransport(decisions_model=" ")


async def test_a_missing_key_is_an_auth_error() -> None:
    transport, server = decisions(api_key=None)
    with pytest.raises(AuthError):
        await transport.run(CLEF, REVIEW, meta=META)
    assert server.requests == []


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("slow"), httpx.ReadError("reset"), OSError("tls")],
    ids=lambda e: type(e).__name__,
)
async def test_a_failure_after_sending_is_one_attempt_and_uncertain(
    error: Exception,
) -> None:
    transport, server = decisions(error, httpx.Response(200, json=EXAMPLE))
    with pytest.raises(UncertainOutcomeError):
        await transport.run(CLEF, REVIEW, meta=META)
    assert len(server.requests) == 1
    assert server.delays == []


@pytest.mark.parametrize("status", [408, 500, 503])
async def test_5xx_is_one_attempt_and_uncertain_with_the_decisions_name(
    status: int,
) -> None:
    response = httpx.Response(
        status,
        json={"error": {"message": "overloaded"}},
        headers={"x-request-id": "r-5"},
    )
    transport, server = decisions(response, httpx.Response(200, json=EXAMPLE))
    with pytest.raises(
        UncertainOutcomeError, match="Decisions API returned HTTP"
    ) as info:
        await transport.run(CLEF, REVIEW, meta=META)
    assert "overloaded" in str(info.value)
    assert len(server.requests) == 1


async def test_connect_errors_and_429_with_retry_after_are_retried() -> None:
    limited = httpx.Response(429, json={"error": "slow"}, headers={"Retry-After": "2"})
    transport, server = decisions(
        httpx.ConnectError("refused"), limited, httpx.Response(200, json=EXAMPLE)
    )
    result = await transport.run(CLEF, REVIEW, meta=META)
    assert result.retries == 2
    assert len(server.requests) == 3
    assert 2.0 in server.delays


async def test_a_429_without_retry_after_is_not_retried() -> None:
    transport, server = decisions(httpx.Response(429, json={"error": "slow"}))
    with pytest.raises(RateLimitError):
        await transport.run(CLEF, REVIEW, meta=META)
    assert len(server.requests) == 1


async def test_a_200_with_an_error_or_a_bad_answer_is_refused() -> None:
    transport, _ = decisions(
        httpx.Response(200, json={"error": {"message": "no credit"}})
    )
    with pytest.raises(BadRequestError, match="no credit"):
        await transport.run(CLEF, REVIEW, meta=META)
    wrong = {**EXAMPLE, "answers": [{**EXAMPLE["answers"][0], "name": "other"}]}
    transport, _ = decisions(httpx.Response(200, json=wrong))
    with pytest.raises(ModelResponseError, match="do not match the questions"):
        await transport.run(CLEF, REVIEW, meta=META)
    transport, _ = decisions(httpx.Response(200, text="<html>"))
    with pytest.raises(ModelResponseError, match="non-JSON"):
        await transport.run(CLEF, REVIEW, meta=META)


def test_the_key_is_known_to_the_recorder() -> None:
    assert DecisionsTransport(api_key=KEY).sensitive_values() == (KEY,)


# --------------------------------------------------------------------------- #
# FallbackTransport
# --------------------------------------------------------------------------- #


class Raises:
    """A transport that raises ``error`` and counts its calls."""

    def __init__(self, error: BaseException, provider: str = "primary") -> None:
        self.error = error
        self.provider = provider
        self.calls: list[tuple[str, Mapping[str, Any], CallMeta, float | None]] = []

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.calls.append((model, payload, meta, timeout))
        raise self.error


class Answers:
    """A transport that answers and counts its calls."""

    provider = "backup"

    def __init__(self, request_id: str | None = "bk-1") -> None:
        self.calls: list[tuple[str, Mapping[str, Any], CallMeta, float | None]] = []
        self.request_id = request_id
        self.closed: list[str] = []

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.calls.append((model, payload, meta, timeout))
        return TransportResponse(
            body={"from": "backup"}, status_code=200, request_id=self.request_id
        )

    async def aclose(self) -> None:
        self.closed.append("aclose")

    def close(self) -> None:
        self.closed.append("close")


# nothing ran: the default triggers
TRIGGERS: list[Exception] = [
    RateLimitError("429", status_code=429),
    TransportError("could not connect; the request was not sent"),
]
# the request may have run, or certainly did: opt-ins
OPT_INS: list[Exception] = [
    ServerError("502", status_code=502),
    UncertainOutcomeError("timed out; outcome unknown"),
    ModelResponseError("not a clef response"),
]
NON_TRIGGERS: list[Exception] = [
    AuthError("401", status_code=401),
    BadRequestError("422", status_code=422),
    ValueError("a bug"),
]


@pytest.mark.parametrize("error", TRIGGERS, ids=lambda e: type(e).__name__)
async def test_each_default_trigger_sends_the_request_to_the_fallback_once(
    error: Exception,
) -> None:
    primary, backup = Raises(error), Answers()
    transport = FallbackTransport(primary, backup)
    result = await transport.run(CLEF, REVIEW, meta=META, timeout=7.0)
    assert len(primary.calls) == 1
    assert backup.calls == [(CLEF, REVIEW, META, 7.0)]  # same request, meta included
    assert result.body == {"from": "backup"}
    assert (result.provider, result.fallback_from) == ("backup", "primary")
    assert result.fallback_error is not None
    assert result.fallback_error.startswith(type(error).__name__)
    assert result.request_id == "bk-1"


@pytest.mark.parametrize("error", OPT_INS, ids=lambda e: type(e).__name__)
async def test_uncertain_and_bad_responses_do_not_fall_back_by_default(
    error: Exception,
) -> None:
    # "a timeout can leave a charged, unknown outcome; do not auto-retry": the
    # fallback asks the same question again, so it needs an explicit opt-in
    primary, backup = Raises(error), Answers()
    with pytest.raises(type(error)) as info:
        await FallbackTransport(primary, backup).run(CLEF, REVIEW, meta=META)
    assert info.value is error
    assert backup.calls == []
    opted_in = FallbackTransport(Raises(error), backup, on=ALL)
    assert (await opted_in.run(CLEF, REVIEW, meta=META)).provider == "backup"
    assert len(backup.calls) == 1


def test_the_default_trigger_is_unavailable_only() -> None:
    assert frozenset({"unavailable"}) == DEFAULT_FALLBACK_ON


@pytest.mark.parametrize("error", NON_TRIGGERS, ids=lambda e: type(e).__name__)
async def test_auth_and_bad_request_errors_never_fall_back(error: Exception) -> None:
    primary, backup = Raises(error), Answers()
    with pytest.raises(type(error)) as info:
        await FallbackTransport(primary, backup).run(CLEF, REVIEW, meta=META)
    assert info.value is error
    assert backup.calls == []


async def test_a_success_never_touches_the_fallback() -> None:
    backup = Answers()
    primary = Answers()
    primary.provider = "primary"
    result = await FallbackTransport(primary, backup).run(CLEF, REVIEW, meta=META)
    assert len(primary.calls) == 1 and backup.calls == []
    assert result.fallback_from is None and result.fallback_error is None


async def test_a_cancellation_is_not_a_trigger() -> None:
    primary, backup = Raises(asyncio.CancelledError()), Answers()
    with pytest.raises(asyncio.CancelledError):
        await FallbackTransport(primary, backup).run(CLEF, REVIEW, meta=META)
    assert backup.calls == []


async def test_unavailable_only_leaves_out_uncertain_outcomes_and_bad_responses() -> (
    None
):
    for error, falls in [
        (RateLimitError("429"), True),
        (TransportError("not sent"), True),
        (ServerError("502"), False),  # a 5xx may have run
        (UncertainOutcomeError("unknown"), False),
        (ModelResponseError("bad"), False),
    ]:
        primary, backup = Raises(error), Answers()
        transport = FallbackTransport(primary, backup, on={"unavailable"})
        if falls:
            assert (await transport.run(CLEF, REVIEW, meta=META)).provider == "backup"
        else:
            with pytest.raises(type(error)):
                await transport.run(CLEF, REVIEW, meta=META)
            assert backup.calls == []


async def test_on_takes_categories_and_exception_classes() -> None:
    only_uncertain = FallbackTransport(
        Raises(UncertainOutcomeError("u")), Answers(), on="uncertain"
    )
    assert (await only_uncertain.run(CLEF, REVIEW, meta=META)).provider == "backup"
    by_class = FallbackTransport(Raises(ServerError("5")), Answers(), on=[ServerError])
    assert (await by_class.run(CLEF, REVIEW, meta=META)).provider == "backup"
    with pytest.raises(RateLimitError):
        await FallbackTransport(
            Raises(RateLimitError("429")), Answers(), on=[ServerError]
        ).run(CLEF, REVIEW, meta=META)
    # "unavailable" does not catch the specific TransportError subclasses
    with pytest.raises(UncertainOutcomeError):
        await FallbackTransport(
            Raises(UncertainOutcomeError("u")), Answers(), on={"unavailable"}
        ).run(CLEF, REVIEW, meta=META)


def test_invalid_triggers_are_rejected() -> None:
    for bad in (AuthError, BadRequestError):
        with pytest.raises(ValueError, match="never falls back"):
            FallbackTransport(Answers(), Answers(), on=[bad])
    with pytest.raises(ValueError, match="never falls back"):
        FallbackTransport(Answers(), Answers(), on=[Exception])  # would include them
    with pytest.raises(ValueError, match="unknown fallback trigger"):
        FallbackTransport(Answers(), Answers(), on={"sometimes"})
    with pytest.raises(ValueError, match="selects no error"):
        FallbackTransport(Answers(), Answers(), on=set())
    with pytest.raises(ValueError, match="not a fallback trigger"):
        FallbackTransport(Answers(), Answers(), on=[3])  # type: ignore[list-item]


async def test_when_the_fallback_fails_too_its_error_carries_the_primarys() -> None:
    primary_error = UncertainOutcomeError("primary timed out")
    root = ValueError("the fallback's own cause")
    fallback_error = ServerError("fallback down")
    fallback_error.__cause__ = root
    primary, backup = Raises(primary_error), Raises(fallback_error, "backup")
    with pytest.raises(ServerError) as info:
        await FallbackTransport(primary, backup, on=ALL).run(CLEF, REVIEW, meta=META)
    assert info.value is fallback_error
    assert info.value.primary_error is primary_error
    assert info.value.__cause__ is root  # its own cause is not clobbered
    assert (primary_error.provider, fallback_error.provider) == ("primary", "backup")
    assert len(primary.calls) == 1 and len(backup.calls) == 1  # no third attempt


async def test_a_fallback_that_cannot_serve_the_model_leaves_the_primarys_error() -> (
    None
):
    primary_error = RateLimitError("429")
    decisions_transport, server = decisions()
    transport = FallbackTransport(Raises(primary_error), decisions_transport)
    with pytest.raises(RateLimitError) as info:
        await transport.run(CLAIRIC, REVIEW, meta=META)
    assert info.value is primary_error
    assert server.requests == []


async def test_each_provider_sees_exactly_one_request() -> None:
    first, first_server = experiential(httpx.ReadTimeout("took too long"))
    second, second_server = decisions(httpx.Response(200, json=EXAMPLE))
    # a timeout is an uncertain outcome: the fallback is an explicit opt-in
    result = await FallbackTransport(first, second, on=ALL).run(CLEF, REVIEW, meta=META)
    assert len(first_server.requests) == 1
    assert len(second_server.requests) == 1
    assert first_server.delays == second_server.delays == []
    assert (result.provider, result.fallback_from) == ("decisions", "experiential")
    assert result.fallback_error is not None
    assert "UncertainOutcomeError" in result.fallback_error
    assert parse_clef(result.body).answers["review"].type == "choice"
    assert result.served_model == DECISIONS_MODEL


async def test_a_5xx_primary_and_a_5xx_fallback_make_two_requests_not_more() -> None:
    first, first_server = experiential(httpx.Response(503, json={"error": "down"}))
    second, second_server = decisions(httpx.Response(502, json={"error": "down"}))
    with pytest.raises(UncertainOutcomeError) as info:
        await FallbackTransport(first, second, on=ALL).run(CLEF, REVIEW, meta=META)
    assert (len(first_server.requests), len(second_server.requests)) == (1, 1)
    assert isinstance(info.value.primary_error, UncertainOutcomeError)
    # and by default a 5xx of the primary is not a reason to ask again
    first, first_server = experiential(httpx.Response(503, json={"error": "down"}))
    second, second_server = decisions()
    with pytest.raises(UncertainOutcomeError):
        await FallbackTransport(first, second).run(CLEF, REVIEW, meta=META)
    assert second_server.requests == []


async def test_cloudflare_as_the_primary_is_classified_per_failure() -> None:
    """Only a failure where nothing could have run falls back by default."""

    def cloudflare(*responses: httpx.Response | Exception) -> tuple[Any, Server]:
        server = Server(*responses)
        client = httpx.AsyncClient(transport=httpx.MockTransport(server))
        return (
            CloudflareTransport(
                account_id="acct",
                api_token="tok",
                client=client,
                max_retries=1,
                sleep=server.sleep,
            ),
            server,
        )

    sent_and_lost = [
        httpx.ReadTimeout("slow"),  # sent, no answer
        httpx.ReadError("reset"),  # dropped after sending
        httpx.RemoteProtocolError("closed"),
    ]
    for error in sent_and_lost:
        primary, server = cloudflare(error)
        backup, backup_server = decisions()
        with pytest.raises(TransportError):
            await FallbackTransport(primary, backup).run(CLEF, REVIEW, meta=META)
        assert backup_server.requests == [], type(error).__name__
        assert len(server.requests) == 2  # Cloudflare's own retry
    for response in (
        httpx.Response(500, json={"errors": [{"message": "boom"}]}),
        httpx.Response(408, json={"errors": [{"message": "timeout"}]}),
    ):
        primary, _ = cloudflare(response)
        backup, backup_server = decisions()
        with pytest.raises(TransportError):
            await FallbackTransport(primary, backup).run(CLEF, REVIEW, meta=META)
        assert backup_server.requests == [], response.status_code
    # nothing ran: the connection failed, or the request was refused
    for error2 in (
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("handshake"),
        httpx.Response(
            429, json={"errors": [{"message": "limit"}]}, headers={"Retry-After": "1"}
        ),
    ):
        primary, _ = cloudflare(error2)
        backup, backup_server = decisions()
        result = await FallbackTransport(primary, backup).run(CLEF, REVIEW, meta=META)
        assert result.fallback_from == "cloudflare"
        assert len(backup_server.requests) == 1
    # with the opt-in, a lost request falls back too
    primary, _ = cloudflare(httpx.ReadTimeout("slow"))
    backup, backup_server = decisions()
    result = await FallbackTransport(primary, backup, on={"uncertain"}).run(
        CLEF, REVIEW, meta=META
    )
    assert result.provider == "decisions" and len(backup_server.requests) == 1


async def test_the_pair_exposes_credentials_and_closes_both() -> None:
    backup = Answers()
    pair = FallbackTransport(ExperientialTransport(api_key=KEY), backup)
    assert pair.sensitive_values() == (KEY,)
    await pair.aclose()
    pair.close()
    assert backup.closed == ["aclose", "close"]
    async with FallbackTransport(Answers(), backup):
        pass
    assert backup.closed[-1] == "aclose"


async def test_it_combines_with_routing() -> None:
    cloudflare, backup = Answers(), Answers()
    router = RoutingTransport(
        {"clef": FallbackTransport(Raises(RateLimitError("429")), backup)},
        default=cloudflare,
    )
    clef = await router.run(CLEF, REVIEW, meta=META)
    assert clef.fallback_from == "primary"
    await router.run(CLAIRIC, {"model": "clairic"}, meta=META)
    assert len(cloudflare.calls) == 1 and len(backup.calls) == 1


# --------------------------------------------------------------------------- #
# Records and warnings
# --------------------------------------------------------------------------- #


def review_questions() -> Any:
    return ClefRequest.model_validate(REVIEW).questions


async def test_the_call_log_names_the_provider_and_the_fallback() -> None:
    first, _ = experiential(httpx.Response(504, json={"error": "gateway"}))
    second, _ = decisions(httpx.Response(200, json=EXAMPLE))
    run = Run(transport=FallbackTransport(first, second, on=ALL))
    answers = await run.call_clef(review_questions(), REVIEW["state"], meta=META)
    assert answers["review"].type == "choice"
    primary_attempt, answered = run.calls
    assert primary_attempt.provider == "experiential"
    assert primary_attempt.error is not None
    assert primary_attempt.error.startswith("UncertainOutcomeError")
    assert answered.error is None
    assert (answered.provider, answered.fallback_from) == ("decisions", "experiential")
    assert answered.model == DECISIONS_MODEL  # accounted under the model that answered
    assert answered.cost_usd == 0.0
    assert (
        run.usage.requests == 1
    )  # the fallback's answer; the failed call has no usage
    assert count_unknown_outcomes(run.calls) == 1  # the primary may have been charged
    (warning,) = run.warnings
    assert warning.startswith(f"{FALLBACK_WARNING}: 1 Clef request(s)")
    assert "decisions" in warning and "experiential" in warning
    assert "used for: gate:Person" in warning


async def test_the_warning_counts_and_stays_one_line() -> None:
    first, _ = experiential(httpx.Response(503, json={"error": "down"}))
    second, _ = decisions()
    run = Run(transport=FallbackTransport(first, second, on=ALL))
    for i in range(3):
        questions = ClefRequest.model_validate(
            {**REVIEW, "questions": {f"q{i}": REVIEW["questions"]["review"]}}
        ).questions
        await run.call_clef(questions, REVIEW["state"], meta=META)
    assert run.fallback_used == 3
    (warning,) = run.warnings
    assert warning.startswith(f"{FALLBACK_WARNING}: 3 Clef request(s)")


async def test_no_fallback_no_warning_and_providers_are_recorded() -> None:
    primary, _ = experiential(httpx.Response(200, json=EXAMPLE_CLEF))
    run = Run(transport=FallbackTransport(primary, Answers()))
    await run.call_clef(review_questions(), REVIEW["state"], meta=META)
    (record,) = run.calls
    assert (record.provider, record.fallback_from) == ("experiential", None)
    assert run.warnings == []


EXAMPLE_CLEF: dict[str, Any] = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "docs/schemas/experiential-clef-response.example.json"
    ).read_text()
)


class _Person(BaseModel):
    name: str


# --------------------------------------------------------------------------- #
# Selection from the environment
# --------------------------------------------------------------------------- #


def test_the_fallback_is_off_unless_asked_for() -> None:
    assert clef_fallback_from_env({}) == "none"
    assert clef_fallback_from_env({"TRUSTCALL_S1_CLEF_FALLBACK": ""}) == "none"
    assert clef_fallback_from_env({"TRUSTCALL_S1_CLEF_FALLBACK": " None "}) == "none"
    assert clef_fallback_from_env({"EXPERIENTIAL_API_KEY": KEY}) == "none"
    assert (
        clef_fallback_from_env({"TRUSTCALL_S1_CLEF_FALLBACK": " Decisions"})
        == "decisions"
    )
    with pytest.raises(ValueError, match="TRUSTCALL_S1_CLEF_FALLBACK"):
        clef_fallback_from_env({"TRUSTCALL_S1_CLEF_FALLBACK": "openai"})


def test_the_factory_builds_the_fallback() -> None:
    router = system_one_transport("experiential", clef_fallback="decisions")
    assert isinstance(router, RoutingTransport)
    clef = router.transport_for(CLEF)
    assert isinstance(clef, FallbackTransport)
    assert isinstance(clef._primary, ExperientialTransport)
    assert isinstance(clef._fallback, DecisionsTransport)
    assert isinstance(router.transport_for(CLEF_FLASH), FallbackTransport)
    assert isinstance(router.transport_for(CLAIRIC), CloudflareTransport)


def test_clef_on_cloudflare_can_have_the_fallback_too() -> None:
    router = system_one_transport("cloudflare", clef_fallback="decisions")
    assert isinstance(router, RoutingTransport)
    clef = router.transport_for(CLEF)
    assert isinstance(clef, FallbackTransport)
    assert clef._primary is router.transport_for(CLAIRIC)  # one Cloudflare transport
    assert isinstance(clef._fallback, DecisionsTransport)


def test_without_a_fallback_nothing_changes() -> None:
    assert isinstance(
        system_one_transport("cloudflare", clef_fallback="none"), CloudflareTransport
    )
    router = system_one_transport("experiential", clef_fallback="none")
    assert isinstance(router, RoutingTransport)
    assert isinstance(router.transport_for(CLEF), ExperientialTransport)
    with pytest.raises(ValueError, match="unknown Clef fallback"):
        system_one_transport("cloudflare", clef_fallback="openai")  # type: ignore[arg-type]


def test_the_factory_uses_the_given_transports_and_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_FALLBACK", "decisions")
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    cloudflare = CloudflareTransport(account_id="a", api_token="t")
    backup = DecisionsTransport(api_key=KEY)
    router = system_one_transport(cloudflare=cloudflare, decisions=backup)
    assert isinstance(router, RoutingTransport)
    clef = router.transport_for(CLEF)
    assert isinstance(clef, FallbackTransport)
    assert clef._primary is cloudflare and clef._fallback is backup


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, CloudflareTransport),
        ({"EXPERIENTIAL_API_KEY": KEY}, CloudflareTransport),  # a key alone: no
        ({"TRUSTCALL_S1_CLEF_FALLBACK": "decisions"}, RoutingTransport),
        ({"TRUSTCALL_S1_CLEF_FALLBACK": "none"}, CloudflareTransport),
        (
            {
                "TRUSTCALL_S1_CLEF_PROVIDER": "experiential",
                "TRUSTCALL_S1_CLEF_FALLBACK": "decisions",
            },
            RoutingTransport,
        ),
    ],
)
def test_the_extractors_default_transport_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: type
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    extractor = create_extractor([_Person])
    try:
        transport = extractor.core.transport()
        assert type(transport) is expected
        if env.get("TRUSTCALL_S1_CLEF_FALLBACK") == "decisions":
            assert isinstance(transport, RoutingTransport)
            assert isinstance(transport.transport_for(CLEF), FallbackTransport)
    finally:
        extractor.close()


def test_a_bad_fallback_variable_fails_when_the_extractor_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_FALLBACK", "openai")
    with pytest.raises(ValueError, match="TRUSTCALL_S1_CLEF_FALLBACK"):
        create_extractor([_Person])
    create_extractor([_Person], transport=Tag("x")).close()  # an explicit transport


def test_evals_take_a_fallback_and_it_needs_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _parser().parse_args(["run", "--live", "--clef-fallback", "decisions"])
    assert _settings(args, {}).clef_fallback == "decisions"
    assert _settings(_parser().parse_args(["run"]), {}).clef_fallback is None
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--clef-fallback", "openai"])
    _settings(_parser().parse_args(["run", "--clef-fallback", "decisions"]), {})
    assert "no effect without --live" in capsys.readouterr().err
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    assert _have_credentials("cloudflare", "none")
    assert not _have_credentials("cloudflare", "decisions")
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    assert _have_credentials("cloudflare", "decisions")
    assert Settings().clef_fallback is None


def test_the_live_tests_need_the_key_for_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.live.conftest import credentials_present

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    assert credentials_present()
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_FALLBACK", "decisions")
    assert not credentials_present()
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", KEY)
    assert credentials_present()


@pytest.mark.parametrize(
    ("yes", "hard"),
    [
        (1.0, True),
        (0.9999, True),  # not 1.0, still a hard pick: capped below remove (0.85)
        (0.9991, True),
        (0.001, True),  # a near-certain no is just as bare
        (0.995, False),  # 0.005 from 1: graded, 0.001 is the line
        (0.9, False),
    ],
)
def test_near_saturated_probabilities_are_hard_picks_too(
    yes: float, hard: bool
) -> None:
    payload = {**REVIEW, "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    _, plans = to_decisions_request(payload)
    body = response_for(
        payload,
        [
            {
                "type": "choice",
                "name": "q",
                "choice": "yes" if yes >= 0.5 else "no",
                "probabilities": probs(yes=yes, no=1 - yes),
            }
        ],
    )
    answer = from_decisions_response(body, plans, clef_model="clef")["answers"]["q"]
    assert answer.get("hard", False) is hard
    if hard:
        assert answer["noul"] in (pytest.approx(0.8), pytest.approx(0.2))
        assert answer["noul"] < 0.85 or answer["noul"] == pytest.approx(0.2)
    else:
        assert answer["noul"] == pytest.approx(yes)


def test_a_near_saturated_yes_cannot_authorise_a_removal() -> None:
    from trustcall_system_one import GateConfig

    payload = {**REVIEW, "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    _, plans = to_decisions_request(payload)
    body = response_for(
        payload,
        [
            {
                "type": "choice",
                "name": "q",
                "choice": "yes",
                "probabilities": probs(yes=0.9999, no=0.0001),
            }
        ],
    )
    noul_p = from_decisions_response(body, plans, clef_model="clef")["answers"]["q"][
        "noul"
    ]
    gates = GateConfig()
    assert noul_p < gates.remove_threshold  # no removal or set-null
    assert noul_p >= gates.add_threshold and noul_p >= gates.replace_threshold
