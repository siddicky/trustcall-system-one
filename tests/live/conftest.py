"""Fixtures of the live smoke tests: a Cloudflare transport that can record."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from trustcall_system_one import CloudflareTransport, Transport
from trustcall_system_one.testing import RecordingTransport

LIVE_ENV = "TRUSTCALL_S1_LIVE"
RECORD_ENV = "TRUSTCALL_S1_RECORD"


def credentials_present() -> bool:
    """Whether the Cloudflare account id and an API token are in the environment."""
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get(
        "CLOUDFLARE_AUTH_TOKEN"
    )
    return bool(os.environ.get("CLOUDFLARE_ACCOUNT_ID") and token)


def live_enabled() -> bool:
    """``TRUSTCALL_S1_LIVE=1`` and the credentials are set."""
    return os.environ.get(LIVE_ENV) == "1" and credentials_present()


SKIP_REASON = (
    f"live tests need {LIVE_ENV}=1 and CLOUDFLARE_ACCOUNT_ID plus "
    "CLOUDFLARE_API_TOKEN in the environment"
)


@pytest.fixture
def live_transport() -> Iterator[Callable[[str], Transport]]:
    """Make the transport of one scenario: live, recording when asked to.

    ``TRUSTCALL_S1_RECORD=<dir>`` writes each scenario's requests and responses to
    ``<dir>/smoke_<name>.jsonl`` (never headers or credentials).
    """
    opened: list[CloudflareTransport] = []

    def make(name: str) -> Transport:
        live = CloudflareTransport()
        opened.append(live)
        folder = os.environ.get(RECORD_ENV)
        if not folder:
            return live
        return RecordingTransport(
            live, Path(folder) / f"smoke_{name}.jsonl", append=False
        )

    yield make
    for transport in opened:
        transport.close()
