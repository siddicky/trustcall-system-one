"""One Clef request carries at most as many questions as its transport takes.

Experiential Labs answers a 400 ("questions must map 1 through 32 ids to typed
questions") above 32; Cloudflare takes 64. Pairwise coref of 9 mentions asks 36
questions and broke the README scenario on Experiential. Everything that groups Clef
questions asks ``Run.clef_batch`` (``transport.max_questions``) instead of 64.
"""

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from tests.unit.schema_helpers import validator
from trustcall_system_one import (
    CLEF,
    CallMeta,
    CloudflareTransport,
    DecisionsTransport,
    ExperientialTransport,
    FallbackTransport,
    RoutingTransport,
)
from trustcall_system_one._coref import cluster_mentions
from trustcall_system_one._match import Candidate, flag_deletes
from trustcall_system_one._run import Run
from trustcall_system_one._types import Mention
from trustcall_system_one.testing import FakeTransport, clef_response, noul
from trustcall_system_one.transport import DEFAULT_MAX_QUESTIONS, max_questions

KEY = "xpl_" + "ab12cd34" * 5
ENDPOINT_LIMIT = 32


class Server:
    """An Experiential endpoint that enforces its documented limit."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.bodies: list[dict[str, Any]] = []
        self.schema = validator("clef-input")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        questions = body["questions"]
        self.sizes.append(len(questions))
        self.schema.validate(body)  # the published Clef input, as the library sends it
        if not 1 <= len(questions) <= ENDPOINT_LIMIT:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "questions must map 1 through 32 ids to typed "
                        "questions",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_request",
                    }
                },
            )
        answers = {qid: noul(0.03) for qid in questions}
        return httpx.Response(200, json=clef_response(answers))


def experiential() -> tuple[ExperientialTransport, Server]:
    server = Server()
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return ExperientialTransport(api_key=KEY, client=client), server


# --------------------------------------------------------------------------- #
# What each transport says
# --------------------------------------------------------------------------- #


def test_each_transport_reports_its_limit() -> None:
    assert (
        max_questions(CloudflareTransport(account_id="a" * 32, api_token="t"), CLEF)
        == 64
    )
    assert max_questions(ExperientialTransport(api_key=KEY), CLEF) == 32
    assert max_questions(DecisionsTransport(api_key=KEY), CLEF) == 32  # unknown: safe
    assert max_questions(FakeTransport(lambda m, p, meta: None), CLEF) == (
        DEFAULT_MAX_QUESTIONS
    )


def test_a_routing_transport_reports_the_limit_of_the_transport_serving_the_model() -> (
    None
):
    router = RoutingTransport(
        {"clef": ExperientialTransport(api_key=KEY)},
        default=CloudflareTransport(account_id="a" * 32, api_token="t"),
    )
    assert max_questions(router, CLEF) == 32
    assert max_questions(router, "@cf/cloudflare/clairic") == 64


def test_a_fallback_reports_the_smaller_limit_of_primary_and_fallback() -> None:
    cloudflare = CloudflareTransport(account_id="a" * 32, api_token="t")
    decisions = DecisionsTransport(api_key=KEY)
    experiential_ = ExperientialTransport(api_key=KEY)
    assert max_questions(FallbackTransport(cloudflare, decisions), CLEF) == 32
    assert max_questions(FallbackTransport(experiential_, decisions), CLEF) == 32
    # the fallback does not serve Clairic: only the primary's limit counts there
    assert max_questions(FallbackTransport(cloudflare, decisions), "clairic") == 64
    # and a router around a fallback is that fallback's limit
    router = RoutingTransport(
        {"clef": FallbackTransport(cloudflare, decisions)}, default=cloudflare
    )
    assert max_questions(router, CLEF) == 32


@pytest.mark.parametrize("said", [0, -3, True, "many", None])
def test_a_nonsense_limit_is_the_default(said: Any) -> None:
    class Odd:
        max_questions = said

    assert max_questions(Odd(), CLEF) == DEFAULT_MAX_QUESTIONS


def test_a_method_or_an_attribute_both_work() -> None:
    class Method:
        def max_questions(self, model: str) -> int:
            return 7 if model == CLEF else 9

    class Attribute:
        max_questions = 5

    assert max_questions(Method(), CLEF) == 7
    assert max_questions(Attribute(), CLEF) == 5
    assert Run(transport=Attribute()).clef_batch() == 5  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# What groups questions obeys it
# --------------------------------------------------------------------------- #


def nine_mentions() -> tuple[str, list[Mention]]:
    names = [
        "Anna Smith",
        "Ben Jones",
        "Cara Lee",
        "Dev Patel",
        "Eli Brown",
        "Fay Green",
        "Gus White",
        "Hana Black",
        "Ivo Gray",
    ]
    source = ". ".join(f"{name} came" for name in names) + "."
    found: list[Mention] = []
    cursor = 0
    for name in names:
        at = source.index(name, cursor)
        found.append(Mention(name, (at, at + len(name))))
        cursor = at + len(name)
    return source, found


async def test_36_pairs_through_experiential_are_32_and_4() -> None:
    transport, server = experiential()
    run = Run(transport=transport)
    source, mentions = nine_mentions()
    clustering = await cluster_mentions(
        run, source=source, mentions=mentions, noun="Person", schema="Person"
    )
    assert len(clustering.pairs) == 36
    assert sorted(server.sizes) == [4, 32]
    assert all(
        sorted(body) == ["model", "questions", "state"] for body in server.bodies
    )
    assert clustering.clusters == tuple((i,) for i in range(9))


async def test_the_same_36_pairs_on_a_64_transport_are_one_request() -> None:
    calls: list[int] = []

    def handler(model: str, payload: Mapping[str, Any], meta: CallMeta) -> Any:
        calls.append(len(payload["questions"]))
        return clef_response({qid: noul(0.03) for qid in payload["questions"]})

    run = Run(transport=FakeTransport(handler))  # the default limit: 64
    source, mentions = nine_mentions()
    await cluster_mentions(
        run, source=source, mentions=mentions, noun="Person", schema="Person"
    )
    assert calls == [36]


async def test_call_clef_splits_70_questions_into_32_32_and_6() -> None:
    transport, server = experiential()
    run = Run(transport=transport)
    from trustcall_system_one._wire import ClefRequest

    many = {
        f"q{i}": {"type": "noul", "instructions": f"Is {i} odd?"} for i in range(70)
    }
    questions: dict[str, Any] = {}
    for start in (0, 35):  # the wire model takes 64 at most: build it in two parts
        part = dict(list(many.items())[start : start + 35])
        questions.update(
            ClefRequest.model_validate(
                {"model": "clef", "state": "s", "questions": part}
            ).questions
        )
    answers = await run.call_clef(questions, "s", meta=CallMeta(kind="coref"))
    assert len(answers) == 70
    assert sorted(server.sizes) == [6, 32, 32]


async def test_deletes_are_asked_in_requests_the_endpoint_takes() -> None:
    from tests.unit.test_match import Person
    from trustcall_system_one import Many  # noqa: F401  (public API in use)
    from trustcall_system_one._compile import compile_schema

    transport, server = experiential()
    run = Run(transport=transport)
    candidates = [
        Candidate(str(i), {"name": f"Person {i}", "relationship": None, "notes": []})
        for i in range(40)
    ]
    found = await flag_deletes(
        run, compile_schema(Person), "Please remove nobody.", candidates, ("name",)
    )
    assert len(found) == 40
    assert max(server.sizes) <= ENDPOINT_LIMIT
    assert sorted(server.sizes, reverse=True)[0] == 32


async def test_a_fallback_pair_chunks_by_the_smaller_limit() -> None:
    # Cloudflare (64) behind a Decisions fallback (32): the fallback must take
    # every request, so requests are cut at 32 whoever answers
    class Counting(FakeTransport):
        max_questions = 64

    sizes: list[int] = []

    def handler(model: str, payload: Mapping[str, Any], meta: CallMeta) -> Any:
        sizes.append(len(payload["questions"]))
        return clef_response({qid: noul(0.5) for qid in payload["questions"]})

    primary = Counting(handler)
    backup = DecisionsTransport(api_key=KEY)
    run = Run(transport=FallbackTransport(primary, backup))
    assert run.clef_batch() == 32


# --------------------------------------------------------------------------- #
# The recording and cassette wrappers forward the limit
# --------------------------------------------------------------------------- #


async def test_a_recording_transport_around_experiential_chunks_by_32(
    tmp_path: Any,
) -> None:
    from trustcall_system_one.testing import (
        CassetteTransport,
        RecordingTransport,
    )

    inner, server = experiential()
    path = tmp_path / "cassette.jsonl"
    recorder = RecordingTransport(inner, path)
    assert max_questions(recorder, CLEF) == 32
    run = Run(transport=recorder)
    source, mentions = nine_mentions()
    await cluster_mentions(
        run, source=source, mentions=mentions, noun="Person", schema="Person"
    )
    assert sorted(server.sizes) == [4, 32]  # not one request of 36: a 400 live
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row["max_questions"] for row in rows} == {32}  # recorded with the lines

    # the cassette replays with the limit it was recorded with, no inner needed
    cassette = CassetteTransport(path)
    assert max_questions(cassette, CLEF) == 32
    replay = Run(transport=cassette)
    await cluster_mentions(
        replay, source=source, mentions=mentions, noun="Person", schema="Person"
    )
    assert cassette.misses == [] and cassette.hits == 2
    assert max_questions(cassette.fork(), CLEF) == 32


def test_a_cassette_without_a_recorded_limit_asks_its_inner_transport_else_64(
    tmp_path: Any,
) -> None:
    from trustcall_system_one.testing import CassetteTransport

    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps({"key": "k", "model": CLEF, "request": {}, "response": {}}) + "\n"
    )
    assert max_questions(CassetteTransport(path), CLEF) == DEFAULT_MAX_QUESTIONS
    inner = ExperientialTransport(api_key=KEY)
    live = CassetteTransport(path, strict=False, inner=inner)
    assert max_questions(live, CLEF) == 32
    # a recorded 64 does not widen an inner that takes 32
    wide = tmp_path / "wide.jsonl"
    wide.write_text(
        json.dumps(
            {
                "key": "k",
                "model": CLEF,
                "request": {},
                "response": {},
                "max_questions": 64,
            }
        )
        + "\n"
    )
    assert max_questions(CassetteTransport(wide), CLEF) == 64
    assert max_questions(CassetteTransport(wide, strict=False, inner=inner), CLEF) == 32
