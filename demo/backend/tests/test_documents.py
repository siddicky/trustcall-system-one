"""Tests of POST /api/document."""

import inspect

import pytest
from fastapi.testclient import TestClient

from app import documents
from app import main as app_main
from app._sandbox import Limits
from tests.pdf import make_pdf


def upload(client: TestClient, name: str, data: bytes, mime: str = "text/plain"):
    return client.post("/api/document", files={"file": (name, data, mime)})


def test_markdown_upload(client: TestClient) -> None:
    body = "# Notes\n\nDana moved to **Berlin** \u2013 caf\u00e9.\n"
    response = upload(client, "notes.md", body.encode())
    assert response.status_code == 200
    assert response.json() == {
        "filename": "notes.md",
        "kind": "markdown",
        "text": body,
        "pages": None,
        "chars": len(body),
    }


def test_markdown_extension_and_text_kind(client: TestClient) -> None:
    assert upload(client, "A.MARKDOWN", b"x").json()["kind"] == "markdown"
    plain = upload(client, "a.txt", b"hello").json()
    assert (plain["kind"], plain["text"], plain["pages"]) == ("text", "hello", None)


def test_invalid_utf8_is_replaced(client: TestClient) -> None:
    response = upload(client, "a.txt", b"caf\xe9")
    assert response.status_code == 200
    assert response.json()["text"] == "caf\ufffd"


def test_pdf_round_trips_its_text(client: TestClient) -> None:
    pdf = make_pdf(["Dana moved to Berlin.", "She is Head of Ops (Northwind)."])
    response = upload(client, "note.pdf", pdf, "application/pdf")
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "pdf" and body["pages"] == 2
    assert body["text"] == ("Dana moved to Berlin.\n\nShe is Head of Ops (Northwind).")
    assert body["chars"] == len(body["text"])


def test_pdf_without_text_is_422(client: TestClient) -> None:
    response = upload(client, "scan.pdf", make_pdf([""]), "application/pdf")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "pdf_no_text"


def test_garbage_pdf_is_422(client: TestClient) -> None:
    response = upload(client, "bad.pdf", b"this is not a pdf", "application/pdf")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "pdf_parse_error" and detail["message"]


def test_unsupported_type_is_415(client: TestClient) -> None:
    response = upload(client, "setup.exe", b"MZ", "application/octet-stream")
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_type"


def test_too_many_characters_is_413(client: TestClient) -> None:
    response = upload(client, "big.txt", b"a" * 200_001)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"


def test_too_many_bytes_is_413(client: TestClient) -> None:
    response = upload(client, "big.txt", b"a" * (10 * 1024 * 1024 + 1))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"


def test_missing_file_is_422(client: TestClient) -> None:
    assert client.post("/api/document").status_code == 422


def test_more_than_300_pages_is_413(client: TestClient) -> None:
    pdf = make_pdf(["page"] * (documents.MAX_PAGES + 1))
    response = upload(client, "long.pdf", pdf, "application/pdf")
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "too_large" and "300 pages" in detail["message"]


def test_300_pages_are_read(client: TestClient) -> None:
    pdf = make_pdf(["page"] * documents.MAX_PAGES)
    response = upload(client, "long.pdf", pdf, "application/pdf")
    assert response.status_code == 200
    assert response.json()["pages"] == documents.MAX_PAGES


def test_a_pdf_with_too_much_text_is_413(client: TestClient) -> None:
    pdf = make_pdf(["a" * 50_000] * 5)  # 250,000 characters
    response = upload(client, "wall.pdf", pdf, "application/pdf")
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "too_large" and "200,000" in detail["message"]


def test_a_pdf_that_takes_too_long_is_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents, "PDF_LIMITS", Limits(timeout=0.01))
    pdf = make_pdf(["Dana moved to Berlin."])
    response = upload(client, "slow.pdf", pdf, "application/pdf")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "pdf_parse_error" and "too long" in detail["message"]


def test_a_worker_that_runs_out_of_memory_is_a_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker dies on its own memory limit; the server answers and lives on."""
    monkeypatch.setattr(
        documents, "PDF_LIMITS", Limits(timeout=20.0, memory_bytes=16 * 1024 * 1024)
    )
    pdf = make_pdf(["Dana moved to Berlin."])
    response = upload(client, "small.pdf", pdf, "application/pdf")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "pdf_parse_error"


def test_the_pdf_worker_has_no_credentials(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app._sandbox import child_env

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "secret")
    assert "CLOUDFLARE_API_TOKEN" not in child_env()
    pdf = make_pdf(["Dana moved to Berlin."])
    assert upload(client, "a.pdf", pdf, "application/pdf").status_code == 200


def test_the_document_endpoint_does_not_block_the_event_loop() -> None:
    assert not inspect.iscoroutinefunction(app_main.document)
