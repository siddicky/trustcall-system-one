"""Tests of GET /api/health and CORS."""

import pytest
from fastapi.testclient import TestClient
from trustcall_system_one import __version__

from app.extract import detect_provider


def test_no_provider(client: TestClient) -> None:
    assert client.get("/api/health").json() == {
        "ok": True,
        "provider": None,
        "version": __version__,
    }


def test_cloudflare(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    assert client.get("/api/health").json()["provider"] == "cloudflare"


def test_cloudflare_needs_both_variables(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    assert client.get("/api/health").json()["provider"] is None


def test_typesafe_needs_the_provider_variable_and_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "key")
    assert detect_provider() is None  # a key alone never switches provider
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    assert detect_provider() == "typesafe"
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert detect_provider() is None
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "nonsense")
    assert detect_provider() is None


def test_cors_allows_the_dev_origin_only(client: TestClient) -> None:
    allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers
