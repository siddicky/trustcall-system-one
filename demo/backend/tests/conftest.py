"""Shared fixtures: a test client with no provider credentials in the environment."""

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from trustcall_system_one import Transport

from app.main import app, get_transport

ENV_VARS = (
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_AUTH_TOKEN",
    "TRUSTCALL_S1_PROVIDER",
    "TYPESAFE_API_KEY",
    "TRUSTCALL_S1_CLEF_PROVIDER",
    "TRUSTCALL_S1_ESCALATE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without provider credentials."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client; dependency overrides are removed afterwards."""
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def use_transport(client: TestClient) -> Callable[[Transport], None]:
    """Return a function that makes the app use the given transport."""

    def use(transport: Transport) -> None:
        app.dependency_overrides[get_transport] = lambda: transport

    return use
