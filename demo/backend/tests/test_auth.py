"""Tests of the bearer-token gate on the API routes."""

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.auth import AUDIENCE, verify_token

SECRET = "test-token-secret"


def mint(
    email: str = "ada@example.com",
    *,
    secret: str = SECRET,
    exp: float | None = None,
    aud: str = AUDIENCE,
) -> str:
    """Mint a token the way the frontend's /api/auth/token does."""
    body = {"email": email, "aud": aud, "exp": exp or int(time.time()) + 600}
    payload = base64.urlsafe_b64encode(json.dumps(body).encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_TOKEN_SECRET", SECRET)


def parse(client: TestClient, token: str | None = None) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    res = client.post("/api/schema/parse", json={"source": "x"}, headers=headers)
    return res.status_code


def test_open_without_a_secret(client: TestClient) -> None:
    assert parse(client) != 401


@pytest.mark.usefixtures("gated")
def test_health_stays_public(client: TestClient) -> None:
    assert client.get("/api/health").status_code == 200


@pytest.mark.usefixtures("gated")
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/document"),
        ("post", "/api/schema/parse"),
        ("post", "/api/extract"),
    ],
)
def test_routes_need_a_token(client: TestClient, method: str, path: str) -> None:
    res = client.request(method, path)
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "unauthorized"
    assert res.headers["www-authenticate"] == "Bearer"


@pytest.mark.usefixtures("gated")
def test_valid_token_passes(client: TestClient) -> None:
    assert parse(client, mint()) != 401


@pytest.mark.usefixtures("gated")
@pytest.mark.parametrize(
    "token",
    [
        mint(secret="other"),
        mint(exp=time.time() - 1),
        mint(aud="someone-else"),
        "garbage",
        ".abc",
    ],
    ids=["wrong-secret", "expired", "wrong-audience", "garbage", "no-payload"],
)
def test_bad_tokens_are_rejected(client: TestClient, token: str) -> None:
    assert parse(client, token) == 401


@pytest.mark.usefixtures("gated")
def test_needs_the_bearer_scheme(client: TestClient) -> None:
    res = client.post(
        "/api/schema/parse",
        json={"source": "x"},
        headers={"Authorization": f"Basic {mint()}"},
    )
    assert res.status_code == 401


def test_verify_token_returns_the_email() -> None:
    assert verify_token(mint("zoë@example.com"), SECRET) == "zoë@example.com"


def test_swapped_payload_is_rejected() -> None:
    signature = mint().rpartition(".")[2]
    forged = mint("eve@example.com", secret="other").rpartition(".")[0]
    assert verify_token(f"{forged}.{signature}", SECRET) is None
