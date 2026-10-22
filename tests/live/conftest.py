"""Fixtures of the live smoke tests: the live transport, which can record."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from trustcall_system_one import (
    CloudflareTransport,
    RoutingTransport,
    Transport,
    TypeSafeTransport,
    clef_fallback_from_env,
    clef_provider_from_env,
    provider_from_env,
    system_one_transport,
)
from trustcall_system_one.testing import RecordingTransport

LIVE_ENV = "TRUSTCALL_S1_LIVE"
RECORD_ENV = "TRUSTCALL_S1_RECORD"


def credentials_present() -> bool:
    """Whether the credentials of the providers in use are in the environment.

    With ``TRUSTCALL_S1_PROVIDER=typesafe`` only ``TYPESAFE_API_KEY`` is needed
    (Jev answers both models). Otherwise Clairic always needs the Cloudflare
    account id and an API token. With Clef on Experiential Labs or the Decisions API
    (``TRUSTCALL_S1_CLEF_PROVIDER``) its key is needed too, as it is for the
    Decisions fallback
    (``TRUSTCALL_S1_CLEF_FALLBACK=decisions``).
    """
    if provider_from_env() == "typesafe":
        return bool((os.environ.get("TYPESAFE_API_KEY") or "").strip())
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get(
        "CLOUDFLARE_AUTH_TOKEN"
    )
    if not (os.environ.get("CLOUDFLARE_ACCOUNT_ID") and token):
        return False
    if (
        clef_provider_from_env() in ("experiential", "decisions")
        or clef_fallback_from_env() == "decisions"
    ):
        return bool((os.environ.get("EXPERIENTIAL_API_KEY") or "").strip())
    return True


def live_enabled() -> bool:
    """``TRUSTCALL_S1_LIVE=1`` and the credentials are set."""
    return os.environ.get(LIVE_ENV) == "1" and credentials_present()


SKIP_REASON = (
    f"live tests need {LIVE_ENV}=1 and CLOUDFLARE_ACCOUNT_ID plus "
    "CLOUDFLARE_API_TOKEN in the environment (and EXPERIENTIAL_API_KEY when "
    "Clef runs on Experiential Labs), or TRUSTCALL_S1_PROVIDER=typesafe with "
    "TYPESAFE_API_KEY"
)


@pytest.fixture
def live_transport() -> Iterator[Callable[[str], Transport]]:
    """Make the transport of one scenario: live, recording when asked to.

    Clef goes to Experiential Labs only when
    ``TRUSTCALL_S1_CLEF_PROVIDER`` is ``experiential`` or ``decisions``, else to
    Cloudflare; both models go to TypeSafe's Jev only when
    ``TRUSTCALL_S1_PROVIDER=typesafe``.

    ``TRUSTCALL_S1_RECORD=<dir>`` writes each scenario's requests and responses to
    ``<dir>/smoke_<name>.jsonl`` (never headers or credentials).
    """
    opened: list[CloudflareTransport | RoutingTransport | TypeSafeTransport] = []

    def make(name: str) -> Transport:
        live = system_one_transport()
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
