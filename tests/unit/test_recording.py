"""RecordingTransport, CassetteTransport and the trace converter."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    BadRequestError,
    CallMeta,
    CloudflareTransport,
    TransportError,
    TransportResponse,
)
from trustcall_system_one.testing import (
    CassetteMissError,
    CassetteTransport,
    FakeTransport,
    RecordingTransport,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    request_key,
    route,
    scrub,
    trace_to_cassette,
)

META = CallMeta(kind="extract", schema="Person", entity="e0", field="name", scope="s")
SECRETS = {
    "CLOUDFLARE_ACCOUNT_ID": "acct-0123456789abcdef",
    "CLOUDFLARE_API_TOKEN": "tok-SECRETSECRETSECRET",
    "CLOUDFLARE_AUTH_TOKEN": "auth-OTHERSECRET99999",
}


def clairic_payload(name: str = "name", state: str = "Ada is 36.") -> dict[str, Any]:
    return {
        "model": "clairic",
        "state": f"<input>\n{state}\n</input>",
        "fields": {name: {"type": "text", "instructions": "the name"}},
        "options": {"return_state_text": True},
    }


def clef_payload(question: str = "q") -> dict[str, Any]:
    return {
        "model": "clef",
        "state": {"input": "Ada is 36."},
        "questions": {
            question: {
                "type": "noul",
                "instructions": "Is it?",
                "criteria": {"true": "yes", "false": "no"},
            }
        },
    }


def world(counter: list[int] | None = None) -> FakeTransport:
    """A fake model whose Clef answer changes with each call (to tell calls apart)."""
    seen = counter if counter is not None else []

    def extract(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clairic_response(
            {name: clairic_answer("text", "Ada") for name in payload["fields"]},
            state_text=payload["state"],
        )

    def decide(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        seen.append(1)
        return clef_response(
            {qid: noul(round(0.1 * len(seen), 2)) for qid in payload["questions"]},
            model=payload["model"],
        )

    return FakeTransport(route(extract=extract, gate=decide, presence=decide))


GATE = CallMeta(kind="gate", schema="Person", scope="s")


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def test_the_key_ignores_dict_order_and_tells_requests_apart() -> None:
    a = {"x": 1, "y": {"b": 2, "a": [1, 2]}}
    b = {"y": {"a": [1, 2], "b": 2}, "x": 1}
    assert request_key(CLAIRIC, a) == request_key(CLAIRIC, b)
    assert request_key(CLAIRIC, a) != request_key(CLEF, a)
    assert request_key(CLAIRIC, a) != request_key(CLAIRIC, {**a, "x": 2})
    assert request_key(CLAIRIC, {"state": "é"}) == request_key(CLAIRIC, {"state": "é"})


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


async def test_every_call_is_written_as_one_line_with_the_documented_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deep" / "cassette.jsonl"
    recording = RecordingTransport(world(), path)
    payload = clairic_payload()
    result = await recording.run(CLAIRIC, payload, meta=META)
    await recording.run(CLEF, clef_payload(), meta=GATE)
    first, second = lines(path)
    assert set(first) >= {
        "key",
        "model",
        "kind",
        "scope",
        "request",
        "response",
        "status",
        "duration_ms",
    }
    assert first["key"] == request_key(CLAIRIC, payload)
    assert (first["model"], first["kind"], first["scope"]) == (CLAIRIC, "extract", "s")
    assert first["request"] == payload
    assert first["response"] == result.body
    assert first["status"] == 200 and first["duration_ms"] >= 0
    assert (second["model"], second["kind"]) == (CLEF, "gate")
    assert recording.written == 2


async def test_a_failed_call_is_written_and_raised(tmp_path: Path) -> None:
    class Failing:
        async def run(
            self, model: str, payload: Any, *, meta: CallMeta, timeout: Any = None
        ) -> TransportResponse:
            error = BadRequestError("nope")
            error.status_code = 422
            raise error

    path = tmp_path / "c.jsonl"
    with pytest.raises(BadRequestError):
        await RecordingTransport(Failing(), path, record_errors=True).run(
            CLAIRIC, clairic_payload(), meta=META
        )
    (line,) = lines(path)
    assert line["response"] is None and line["status"] == 422
    assert line["error"] == {"type": "BadRequestError", "message": "nope"}
    with pytest.raises(BadRequestError) as caught:
        await CassetteTransport(path).run(CLAIRIC, clairic_payload(), meta=META)
    assert caught.value.status_code == 422


async def test_append_false_starts_a_fresh_file(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    await RecordingTransport(world(), path).run(CLAIRIC, clairic_payload(), meta=META)
    await RecordingTransport(world(), path).run(CLAIRIC, clairic_payload(), meta=META)
    assert len(lines(path)) == 2
    fresh = RecordingTransport(world(), path, append=False)
    assert lines(path) == []
    await fresh.run(CLAIRIC, clairic_payload(), meta=META)
    assert len(lines(path)) == 1


async def test_no_credentials_reach_the_file_even_when_they_are_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    seen_headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        body = json.loads(request.content)
        answer = clairic_response(
            {name: clairic_answer("text", "Ada") for name in body["fields"]},
            state_text=body["state"],
        )
        # a hostile server that echoes the account id and the token in the body
        answer["echo"] = {
            "header": request.headers["Authorization"],
            "url": str(request.url),
        }
        return httpx.Response(
            200, json={"result": answer, "success": True}, headers={"x-acct": "x"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cloudflare = CloudflareTransport(client=client)  # credentials from the environment
    path = tmp_path / "c.jsonl"
    # the payload itself mentions a secret value: it is removed from the line too
    payload = clairic_payload(state=f"my token is {SECRETS['CLOUDFLARE_API_TOKEN']}")
    result = await RecordingTransport(cloudflare, path).run(CLAIRIC, payload, meta=META)
    assert result.status_code == 200
    # the real call did carry the credentials ...
    assert SECRETS["CLOUDFLARE_API_TOKEN"] in seen_headers[0]["Authorization"]
    # ... and none of them is in the file, in any form
    text = path.read_text()
    for value in SECRETS.values():
        assert value not in text
    assert "authorization" not in text.lower()
    assert "Bearer tok" not in text and "Bearer auth" not in text
    assert "<redacted>" in text
    await client.aclose()


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


async def record(tmp_path: Path, *calls: tuple[str, dict[str, Any], CallMeta]) -> Path:
    path = tmp_path / "c.jsonl"
    recording = RecordingTransport(world(), path)
    for model, payload, meta in calls:
        await recording.run(model, payload, meta=meta)
    return path


async def test_a_recorded_call_replays_with_its_response_status_and_timing(
    tmp_path: Path,
) -> None:
    path = await record(tmp_path, (CLAIRIC, clairic_payload(), META))
    expected = lines(path)[0]
    cassette = CassetteTransport(path)
    result = await cassette.run(CLAIRIC, clairic_payload(), meta=META)
    assert result.body == expected["response"]
    assert (result.status_code, result.retries) == (200, 0)
    assert result.duration_ms == expected["duration_ms"]
    assert (cassette.hits, len(cassette.misses), len(cassette)) == (1, 0, 1)
    result.body["answers"].clear()  # a caller cannot change the recording
    again = CassetteTransport(path)
    assert (await again.run(CLAIRIC, clairic_payload(), meta=META)).body["answers"]


async def test_a_missing_key_names_the_kind_and_the_field(tmp_path: Path) -> None:
    path = await record(tmp_path, (CLAIRIC, clairic_payload(), META))
    cassette = CassetteTransport(path)
    other = CallMeta(kind="reask", schema="Person", field="relationship")
    with pytest.raises(CassetteMissError) as caught:
        await cassette.run(CLAIRIC, clairic_payload(name="relationship"), meta=other)
    error = caught.value
    assert (error.kind, error.field, error.schema) == (
        "reask",
        "relationship",
        "Person",
    )
    assert "'reask'" in str(error) and "'relationship'" in str(error)
    assert error.key == request_key(CLAIRIC, clairic_payload(name="relationship"))
    assert isinstance(error, AssertionError)
    assert [m[2].kind for m in cassette.misses] == ["reask"]


async def test_duplicate_keys_replay_in_recorded_order_then_run_out(
    tmp_path: Path,
) -> None:
    path = await record(
        tmp_path,
        (CLEF, clef_payload(), GATE),
        (CLEF, clef_payload(), GATE),
        (CLEF, clef_payload(), GATE),
    )
    recorded = [row["response"] for row in lines(path)]
    assert len({json.dumps(r) for r in recorded}) == 3  # the three differ
    cassette = CassetteTransport(path)
    replayed = [
        (await cassette.run(CLEF, clef_payload(), meta=GATE)).body for _ in range(3)
    ]
    assert replayed == recorded
    with pytest.raises(CassetteMissError, match="already replayed"):
        await cassette.run(CLEF, clef_payload(), meta=GATE)
    cassette.reset()
    first = await cassette.run(CLEF, clef_payload(), meta=GATE)
    assert first.body == recorded[0]
    assert len(cassette.unused()) == 2


async def test_not_strict_falls_through_to_the_inner_transport(tmp_path: Path) -> None:
    path = await record(tmp_path, (CLAIRIC, clairic_payload(), META))
    inner = world()
    cassette = CassetteTransport(path, strict=False, inner=inner)
    await cassette.run(CLAIRIC, clairic_payload(), meta=META)
    assert inner.calls == []  # a hit never reaches the inner transport
    result = await cassette.run(CLAIRIC, clairic_payload(name="other"), meta=META)
    assert "other" in result.body["answers"]
    assert [p["fields"] for _, p, _ in inner.calls] == [
        clairic_payload(name="other")["fields"]
    ]
    assert (cassette.hits, cassette.forwarded, len(cassette.misses)) == (1, 1, 1)


async def test_strict_never_uses_the_inner_transport(tmp_path: Path) -> None:
    path = await record(tmp_path, (CLAIRIC, clairic_payload(), META))
    inner = world()
    cassette = CassetteTransport(path, strict=True, inner=inner)
    with pytest.raises(CassetteMissError):
        await cassette.run(CLAIRIC, clairic_payload(name="other"), meta=META)
    assert inner.calls == []
    # not strict, but nothing to ask: still a miss
    with pytest.raises(CassetteMissError):
        await CassetteTransport(path, strict=False).run(
            CLAIRIC, clairic_payload(name="other"), meta=META
        )


async def test_a_directory_or_a_list_of_files_is_one_cassette(tmp_path: Path) -> None:
    folder = tmp_path / "many"
    folder.mkdir()
    for i, name in enumerate(["b", "a"]):
        await RecordingTransport(world(), folder / f"{name}.jsonl").run(
            CLAIRIC, clairic_payload(name=f"f{i}"), meta=META
        )
    (folder / "notes.txt").write_text("ignored")
    assert len(CassetteTransport(folder)) == 2
    assert len(CassetteTransport([folder / "a.jsonl", folder / "b.jsonl"])) == 2


def test_a_file_that_is_not_a_cassette_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"nothing": 1}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        CassetteTransport(bad)
    with pytest.raises(FileNotFoundError):
        CassetteTransport(tmp_path / "missing.jsonl")


async def test_a_cassette_serves_concurrent_callers(tmp_path: Path) -> None:
    import asyncio

    path = await record(
        tmp_path,
        *((CLAIRIC, clairic_payload(name=f"f{i}"), META) for i in range(20)),
    )
    cassette = CassetteTransport(path)
    results = await asyncio.gather(
        *(
            cassette.run(CLAIRIC, clairic_payload(name=f"f{i}"), meta=META)
            for i in range(20)
        )
    )
    assert [list(r.body["answers"]) for r in results] == [[f"f{i}"] for i in range(20)]
    assert cassette.unused() == []


# --------------------------------------------------------------------------- #
# The trace converter
# --------------------------------------------------------------------------- #


def trace_line(
    model: str, request: dict[str, Any], result: dict[str, Any], http: int = 200
) -> dict[str, Any]:
    return {
        "url": f"https://api.cloudflare.com/client/v4/accounts/<ACCOUNT>/ai/run/{model}",
        "http": http,
        "request": request,
        "response": {"result": result, "success": True, "errors": [], "messages": []},
    }


async def test_a_wire_trace_becomes_a_cassette_that_replays_by_key(
    tmp_path: Path,
) -> None:
    fake = world()
    first = clairic_payload()
    answer = (await fake.run(CLAIRIC, first, meta=META)).body
    second = clef_payload()
    decision = (await fake.run(CLEF, second, meta=GATE)).body
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                trace_line(CLAIRIC, first, answer),
                trace_line(CLEF, second, decision),
            )
        )
        + "\n"
    )
    cassette_path = tmp_path / "out.jsonl"
    assert trace_to_cassette(trace, cassette_path) == 2
    rows = lines(cassette_path)
    assert [r["model"] for r in rows] == [CLAIRIC, CLEF]
    assert rows[0]["response"] == answer  # the REST envelope is gone
    cassette = CassetteTransport(cassette_path)
    assert (await cassette.run(CLAIRIC, first, meta=META)).body == answer
    assert (await cassette.run(CLEF, second, meta=GATE)).body == decision


def test_a_line_that_is_not_a_wire_trace_is_refused(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"request": {}}\n')
    with pytest.raises(ValueError, match="not a wire trace line"):
        trace_to_cassette(trace, tmp_path / "o.jsonl")


async def test_a_fork_replays_the_same_recordings_from_the_start(
    tmp_path: Path,
) -> None:
    path = await record(tmp_path, (CLAIRIC, clairic_payload(), META))
    first = CassetteTransport(path)
    await first.run(CLAIRIC, clairic_payload(), meta=META)
    second = first.fork()
    assert (second.hits, len(second)) == (0, 1)
    await second.run(CLAIRIC, clairic_payload(), meta=META)
    # the original is spent; the fork has its own cursor and its own settings
    with pytest.raises(CassetteMissError):
        await first.run(CLAIRIC, clairic_payload(), meta=META)
    inner = world()
    open_fork = first.fork(strict=False, inner=inner)
    await open_fork.run(CLAIRIC, clairic_payload(name="x"), meta=META)
    assert open_fork.forwarded == 1 and len(inner.calls) == 1


# --------------------------------------------------------------------------- #
# Recorded failures, and what the files may contain
# --------------------------------------------------------------------------- #


class Flaky:
    """Fails the first call with a 429, answers the others."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self, model: str, payload: Any, *, meta: CallMeta, timeout: Any = None
    ) -> TransportResponse:
        from trustcall_system_one import RateLimitError

        self.calls += 1
        if self.calls == 1:
            raise RateLimitError("429 too many requests")
        return TransportResponse(
            body=clef_response({"q": noul(0.9)}, model="clef"), status_code=200
        )


async def test_failures_are_not_recorded_unless_asked(tmp_path: Path) -> None:
    from trustcall_system_one import RateLimitError

    path = tmp_path / "c.jsonl"
    with pytest.raises(RateLimitError):
        await RecordingTransport(Flaky(), path).run(CLEF, clef_payload(), meta=GATE)
    assert not path.exists() or path.read_text() == ""


async def test_a_recorded_failure_is_skipped_when_there_is_a_live_fallback(
    tmp_path: Path,
) -> None:
    from trustcall_system_one import RateLimitError

    path = tmp_path / "c.jsonl"
    live = Flaky()
    with pytest.raises(RateLimitError):
        await RecordingTransport(live, path, record_errors=True).run(
            CLEF, clef_payload(), meta=GATE
        )
    assert len(lines(path)) == 1 and lines(path)[0]["error"]
    # without a fallback the recorded failure is what the cassette says
    with pytest.raises(RateLimitError):
        await CassetteTransport(path).run(CLEF, clef_payload(), meta=GATE)
    # with one, it is a transient accident: the request goes out, and is recorded
    cassette = CassetteTransport(
        path, strict=False, inner=RecordingTransport(live, path)
    )
    result = await cassette.run(CLEF, clef_payload(), meta=GATE)
    assert result.status_code == 200
    assert (cassette.hits, cassette.forwarded, live.calls) == (0, 1, 2)
    # the next run replays the success, not the failure
    again = CassetteTransport(path, strict=False, inner=RecordingTransport(live, path))
    assert (await again.run(CLEF, clef_payload(), meta=GATE)).status_code == 200
    assert (again.hits, again.forwarded) == (1, 0)


ACCOUNT = "0123456789abcdef0123456789abcdef"
TOKEN = "secret-token-123456"


def refusing_cloudflare(gateway: str | None) -> CloudflareTransport:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return CloudflareTransport(
        account_id=ACCOUNT,
        api_token=TOKEN,
        gateway=gateway,
        client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
        max_retries=0,
    )


@pytest.mark.parametrize("gateway", [None, "my-gw"])
async def test_an_explicit_account_id_never_reaches_a_cassette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: str | None
) -> None:
    for name in SECRETS:
        monkeypatch.delenv(name, raising=False)  # nothing in the environment
    path = tmp_path / "c.jsonl"
    with pytest.raises(TransportError):
        await RecordingTransport(
            refusing_cloudflare(gateway), path, record_errors=True
        ).run(CLEF, clef_payload(), meta=GATE)
    text = path.read_text()
    assert "Request to https://" in text  # the error message is there ...
    assert ACCOUNT not in text and TOKEN not in text  # ... without the ids
    assert "<redacted>" in text


def test_scrub_removes_urls_ids_and_tokens() -> None:
    text = (
        "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run/@cf/x "
        "https://gateway.ai.cloudflare.com/v1/abc123/gw/workers-ai/@cf/x "
        "Authorization: Bearer abcdefghijklmnop and extra-secret"
    )
    out = scrub(text, extra=["extra-secret"])
    assert "abc123" not in out and "abcdefghijklmnop" not in out
    assert "extra-secret" not in out
    assert "/ai/run/@cf/x" in out and "/gw/workers-ai/" in out  # the rest is kept
