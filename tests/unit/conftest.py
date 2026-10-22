"""Unit-test fixtures: stay offline and check requests against published schemas."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pytest_socket import disable_socket, enable_socket

from tests.unit.schema_helpers import validator
from trustcall_system_one._wire import ClairicRequest, ClefRequest


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    """Block network sockets even when pytest runs without ``--disable-socket``."""
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()


@pytest.fixture(autouse=True)
def _no_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's Clef provider settings out of the tests."""
    monkeypatch.delenv("TRUSTCALL_S1_CLEF_PROVIDER", raising=False)
    monkeypatch.delenv("EXPERIENTIAL_API_KEY", raising=False)
    monkeypatch.delenv("TRUSTCALL_S1_CLEF_FALLBACK", raising=False)


@pytest.fixture(autouse=True)
def _requests_match_published_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate every ``to_payload()`` result against ``docs/schemas/*-input.json``."""
    pairs: list[tuple[type[Any], str]] = [
        (ClairicRequest, "clairic-input"),
        (ClefRequest, "clef-input"),
    ]
    for cls, schema_name in pairs:
        check = validator(schema_name).validate
        original: Callable[[Any], dict[str, Any]] = cls.to_payload

        def to_payload(
            self: Any,
            _original: Callable[[Any], dict[str, Any]] = original,
            _check: Callable[[Any], None] = check,
        ) -> dict[str, Any]:
            payload = _original(self)
            _check(payload)
            return payload

        monkeypatch.setattr(cls, "to_payload", to_payload)
