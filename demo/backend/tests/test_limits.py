"""Tests of the request body limits (app/limits.py)."""

import json
from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from app import main as app_main
from app.limits import (
    DOCUMENT_LIMIT,
    EXTRACT_LIMIT,
    SCHEMA_LIMIT,
    BodyLimitMiddleware,
)

SOURCE = "from pydantic import BaseModel\nclass A(BaseModel):\n    x: int\n"


@pytest.fixture
def untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if a request that should be rejected early reaches an endpoint."""

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the request reached the endpoint")

    monkeypatch.setattr(app_main, "read_document", boom)
    monkeypatch.setattr(app_main, "parse_models", boom)
    monkeypatch.setattr(app_main, "run_extraction", boom)


def test_a_declared_upload_over_the_limit_is_413_before_parsing(
    client: TestClient, untouched: None
) -> None:
    data = b"a" * (DOCUMENT_LIMIT + 1)
    response = client.post(
        "/api/document", files={"file": ("big.txt", data, "text/plain")}
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"


def test_a_lying_content_length_is_413_without_reading_the_body(
    client: TestClient, untouched: None
) -> None:
    response = client.post(
        "/api/document",
        content=b"tiny",
        headers={
            "content-length": str(DOCUMENT_LIMIT + 1),
            "content-type": "multipart/form-data; boundary=x",
        },
    )
    assert response.status_code == 413


def test_an_upload_at_the_file_limit_still_reaches_the_endpoint(
    client: TestClient,
) -> None:
    """10 MB of file plus the multipart framing fits; the endpoint judges the file."""
    data = b"a" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/document", files={"file": ("big.txt", data, "text/plain")}
    )
    assert response.status_code == 413
    assert "MB" in response.json()["detail"]["message"]  # from read_document


def test_json_bodies_have_a_limit(client: TestClient, untouched: None) -> None:
    big = json.dumps({"source": "#" * SCHEMA_LIMIT})
    response = client.post(
        "/api/schema/parse", content=big, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"

    huge = json.dumps({"text": "x" * EXTRACT_LIMIT, "schemas": []})
    response = client.post(
        "/api/extract", content=huge, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413


def test_a_body_under_the_limit_is_untouched(client: TestClient) -> None:
    response = client.post("/api/schema/parse", json={"source": SOURCE})
    assert response.status_code == 200


def test_chunked_bodies_without_content_length_are_counted(
    client: TestClient, untouched: None
) -> None:
    chunk = b" " * 65_536

    def body() -> Iterator[bytes]:
        yield b'{"source": "'
        for _ in range(40):  # 2.6 MB, no Content-Length header
            yield chunk
        yield b'"}'

    response = client.post(
        "/api/schema/parse",
        content=body(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"


def test_chunked_upload_to_the_document_endpoint(
    client: TestClient, untouched: None
) -> None:
    def body() -> Iterator[bytes]:
        yield b"--x\r\nContent-Disposition: form-data; "
        yield b"name=file; filename=a.txt\r\n\r\n"
        for _ in range(DOCUMENT_LIMIT // 65_536 + 2):
            yield b"a" * 65_536

    response = client.post(
        "/api/document",
        content=body(),
        headers={"content-type": "multipart/form-data; boundary=x"},
    )
    assert response.status_code == 413


def test_the_413_carries_cors_headers(client: TestClient, untouched: None) -> None:
    response = client.post(
        "/api/schema/parse",
        content=b"{}",
        headers={
            "origin": "http://localhost:5173",
            "content-length": str(SCHEMA_LIMIT + 1),
            "content-type": "application/json",
        },
    )
    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_an_invalid_content_length_is_400(client: TestClient) -> None:
    response = client.post(
        "/api/schema/parse",
        content=b"{}",
        headers={"content-length": "lots", "content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "bad_request"


def test_other_paths_are_not_limited(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200


# -- The middleware on its own: the stream stops being read at the limit -------- #


async def _run(
    app: BodyLimitMiddleware, chunks: list[bytes], headers: list[tuple[bytes, bytes]]
) -> tuple[list[Message], int]:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/up",
        "headers": headers,
    }
    pending = list(chunks)
    pulled = 0

    async def receive() -> Message:
        nonlocal pulled
        pulled += 1
        body = pending.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent, pulled


async def _drain(scope: Scope, receive: Receive, send: Send) -> None:
    more = True
    while more:
        message = await receive()
        more = message.get("more_body", False)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def test_the_stream_stops_being_read_past_the_limit() -> None:
    middleware = BodyLimitMiddleware(_drain, {"/up": 100})
    chunks = [b"x" * 40] * 10
    with pytest.raises(HTTPException) as info:
        await _run(middleware, chunks, [])
    assert info.value.status_code == 413
    assert isinstance(info.value.detail, dict)
    assert info.value.detail["code"] == "too_large"


async def test_a_stream_at_the_limit_passes() -> None:
    middleware = BodyLimitMiddleware(_drain, {"/up": 100})
    sent, pulled = await _run(middleware, [b"x" * 50, b"x" * 50], [])
    assert sent[0]["status"] == 200 and pulled == 2


async def test_content_length_is_judged_before_any_read() -> None:
    middleware = BodyLimitMiddleware(_drain, {"/up": 100})
    sent, pulled = await _run(middleware, [b"x"], [(b"content-length", b"101")])
    assert sent[0]["status"] == 413 and pulled == 0
