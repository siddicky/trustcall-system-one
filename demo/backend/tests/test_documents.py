"""Tests of POST /api/document."""

from fastapi.testclient import TestClient

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
