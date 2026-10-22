"""Selecting the provider (TypeSafe-only, Decisions as the primary), scrubbing, evals.

Offline. A key in the environment never switches a provider: only
``TRUSTCALL_S1_PROVIDER`` / ``TRUSTCALL_S1_CLEF_PROVIDER`` or an argument does.
The TypeSafe key below is made up and deliberately not hexadecimal; the key of the
real shape is built at run time so that no key-like literal is in the tree.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from evals.cases import CASES
from evals.cli import _parser, _settings
from evals.cli import main as evals_main
from evals.report import suite_markdown, write_suite
from evals.runner import Settings, SuiteResult, combination, run_suite
from trustcall_system_one import (
    CLEF,
    CallMeta,
    CloudflareTransport,
    DecisionsTransport,
    ExperientialTransport,
    RoutingTransport,
    TypeSafeTransport,
    clef_provider_from_env,
    create_extractor,
    provider_from_env,
    system_one_transport,
)
from trustcall_system_one.testing import RecordingTransport, scrub
from trustcall_system_one.transport import max_questions

KEY = "apikey_" + "ghjk-mnpq" * 3
# the shape of a real key: "apikey_" + 36 hex + "_" + 64 hex, assembled at run time
REAL_SHAPE = "_".join(
    ["apikey", "0123456789abcdef" * 2 + "0123", "fedcba9876543210" * 4]
)  # joined at run time: no key-shaped constant reaches the bytecode either
META = CallMeta(kind="attribution")
PAYLOAD: dict[str, Any] = {
    "model": "clef",
    "state": "I was charged twice.",
    "questions": {"q": {"type": "noul", "instructions": "Is it about billing?"}},
}
BODY = {
    "model": "jev-1.14.0",
    "answers": {"q": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 5, "output_tokens": 1},
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
        "EXPERIENTIAL_API_KEY",
        "TYPESAFE_API_KEY",
        "TRUSTCALL_S1_PROVIDER",
        "TRUSTCALL_S1_CLEF_PROVIDER",
        "TRUSTCALL_S1_CLEF_FALLBACK",
        "TRUSTCALL_S1_CLEF_FALLBACK_ON",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# TypeSafe-only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "cloudflare"),
        ("", "cloudflare"),
        ("Cloudflare", "cloudflare"),
        (" TypeSafe ", "typesafe"),
    ],
)
def test_provider_from_env(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str
) -> None:
    if value is not None:
        monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", value)
    assert provider_from_env() == expected
    assert provider_from_env({"TRUSTCALL_S1_PROVIDER": "typesafe"}) == "typesafe"


def test_an_unknown_provider_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "openai")
    with pytest.raises(ValueError, match="TRUSTCALL_S1_PROVIDER"):
        provider_from_env()
    with pytest.raises(ValueError, match="unknown provider"):
        system_one_transport(provider="openai")  # type: ignore[arg-type]


def test_a_key_alone_never_switches_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    assert isinstance(system_one_transport(), CloudflareTransport)
    ex = create_extractor([_Schema])
    assert isinstance(ex.core.transport(), CloudflareTransport)
    ex.core.close()


def test_the_argument_selects_typesafe_for_both_models() -> None:
    transport = system_one_transport(provider="typesafe")
    assert isinstance(transport, TypeSafeTransport)
    assert transport.serves("@cf/cloudflare/clairic")
    assert transport.serves(CLEF)
    own = TypeSafeTransport(api_key=KEY)
    assert system_one_transport(provider="typesafe", typesafe=own) is own


def test_the_environment_selects_typesafe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    assert isinstance(system_one_transport(), TypeSafeTransport)


class _Schema(BaseModel):
    name: str | None = None


def test_the_extractor_default_is_typesafe_without_cloudflare_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    ex = create_extractor([_Schema])
    transport = ex.core.transport()
    assert isinstance(transport, TypeSafeTransport)
    assert transport is ex.core.transport()  # created once
    ex.core.close()


def test_a_bad_provider_fails_when_the_extractor_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "nope")
    with pytest.raises(ValueError, match="TRUSTCALL_S1_PROVIDER"):
        create_extractor([_Schema])


def test_typesafe_only_means_no_other_company_gets_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="answers Clef with Jev itself"):
        system_one_transport("experiential", provider="typesafe")
    with pytest.raises(ValueError, match="answers Clef with Jev itself"):
        system_one_transport("decisions", provider="typesafe")
    with pytest.raises(ValueError, match="TypeSafe only"):
        system_one_transport(provider="typesafe", clef_fallback="decisions")
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "experiential")
    with pytest.raises(ValueError, match="answers Clef with Jev itself"):
        system_one_transport()
    monkeypatch.delenv("TRUSTCALL_S1_CLEF_PROVIDER")
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_FALLBACK", "decisions")
    with pytest.raises(ValueError, match="TypeSafe only"):
        system_one_transport()


def test_a_run_on_typesafe_needs_no_cloudflare_credential_and_sends_nothing_there() -> (
    None
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = system_one_transport(
        provider="typesafe", typesafe=TypeSafeTransport(api_key=KEY, client=client)
    )
    import asyncio

    sent = asyncio.run(transport.run(CLEF, PAYLOAD, meta=META))
    assert sent.body["answers"]["q"]["noul"] == 0.9
    assert {r.url.host for r in seen} == {"api.typesafe.ai"}


# --------------------------------------------------------------------------- #
# The Decisions API as the primary for Clef
# --------------------------------------------------------------------------- #


def test_clef_provider_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "Decisions")
    assert clef_provider_from_env() == "decisions"
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "gpt")
    with pytest.raises(ValueError, match="'decisions'"):
        clef_provider_from_env()


def test_decisions_answers_all_clef_requests_and_clairic_stays_on_cloudflare() -> None:
    transport = system_one_transport("decisions")
    assert isinstance(transport, RoutingTransport)
    clef = transport.transport_for(CLEF)
    assert isinstance(clef, DecisionsTransport)
    assert transport.transport_for("@cf/cloudflare/clef-flash") is clef
    assert isinstance(
        transport.transport_for("@cf/cloudflare/clairic"), CloudflareTransport
    )
    # the Decisions limits apply: 32 questions, a hard pick is capped at 0.8
    assert max_questions(transport, CLEF) == 32
    assert max_questions(transport, "@cf/cloudflare/clairic") == 64
    assert clef._hard_cap == 0.8


def test_decisions_is_an_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", "xpl_" + "ab12cd34" * 5)
    assert isinstance(system_one_transport(), CloudflareTransport)
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "decisions")
    transport = system_one_transport()
    assert isinstance(transport, RoutingTransport)
    assert isinstance(transport.transport_for(CLEF), DecisionsTransport)


def test_decisions_given_as_an_object_is_used() -> None:
    own = DecisionsTransport(api_key="xpl_" + "ab12cd34" * 5, decisions_hard_cap=0.7)
    transport = system_one_transport("decisions", decisions=own)
    assert isinstance(transport, RoutingTransport)
    assert transport.transport_for(CLEF) is own


def test_decisions_as_primary_and_as_fallback_is_a_contradiction() -> None:
    with pytest.raises(ValueError, match="already is the Decisions API"):
        system_one_transport("decisions", clef_fallback="decisions")


def test_the_other_clef_providers_are_unchanged() -> None:
    assert isinstance(system_one_transport("cloudflare"), CloudflareTransport)
    routed = system_one_transport("experiential")
    assert isinstance(routed, RoutingTransport)
    assert isinstance(routed.transport_for(CLEF), ExperientialTransport)
    assert not isinstance(routed.transport_for(CLEF), DecisionsTransport)
    with pytest.raises(ValueError, match="unknown Clef provider"):
        system_one_transport("openai")  # type: ignore[arg-type]


def test_the_extractor_default_follows_the_clef_provider_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "decisions")
    ex = create_extractor([_Schema])
    transport = ex.core.transport()
    assert isinstance(transport, RoutingTransport)
    assert isinstance(transport.transport_for(CLEF), DecisionsTransport)
    ex.core.close()


# --------------------------------------------------------------------------- #
# Scrubbing
# --------------------------------------------------------------------------- #


def test_scrub_removes_the_exact_value_the_header_and_the_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    text = (
        f"failed with {KEY}; headers {{'Authorization': 'Bearer {KEY}'}}; "
        f"other {REAL_SHAPE} and Bearer {REAL_SHAPE}"
    )
    cleaned = scrub(text)
    assert KEY not in cleaned
    assert REAL_SHAPE not in cleaned
    assert "0123456789abcdef" not in cleaned
    assert "Bearer <redacted>" in cleaned


def test_scrub_removes_an_explicit_key_and_the_apikey_shape() -> None:
    cleaned = scrub(
        f"key {KEY} and {REAL_SHAPE}",
        extra=TypeSafeTransport(api_key=KEY).sensitive_values(),
    )
    assert KEY not in cleaned
    assert REAL_SHAPE not in cleaned
    assert scrub("apikey_short is a word") == "apikey_short is a word"
    assert scrub("the apikey field") == "the apikey field"


async def test_a_recorded_call_that_holds_the_key_is_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=BODY))
    )
    transport = TypeSafeTransport(client=client)
    leaky = {**PAYLOAD, "state": f"my key is {KEY}"}
    path = tmp_path / "cassette.jsonl"
    recording = RecordingTransport(transport, path)
    with pytest.warns(RuntimeWarning, match="credential"):
        await recording.run(CLEF, leaky, meta=META)
    assert recording.written == 0
    assert not path.exists()
    # an explicit key (not in the environment) is protected through sensitive_values
    monkeypatch.delenv("TYPESAFE_API_KEY")
    explicit = TypeSafeTransport(api_key=KEY, client=client)
    recording = RecordingTransport(explicit, path)
    with pytest.warns(RuntimeWarning, match="credential"):
        await recording.run(CLEF, leaky, meta=META)
    assert recording.written == 0


async def test_a_clean_call_is_recorded_without_any_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=BODY))
    )
    path = tmp_path / "cassette.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await RecordingTransport(TypeSafeTransport(client=client), path).run(
            CLEF, PAYLOAD, meta=META
        )
    line = json.loads(path.read_text())
    assert line["response"]["answers"]["q"]["noul"] == 0.9
    assert KEY not in path.read_text()


async def test_an_error_message_never_holds_the_key(tmp_path: Path) -> None:
    echo = httpx.Response(401, json={"detail": f"bad key {KEY}"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: echo))
    from trustcall_system_one import AuthError

    path = tmp_path / "cassette.jsonl"
    transport = TypeSafeTransport(api_key=KEY, client=client)
    with pytest.raises(AuthError):
        await RecordingTransport(transport, path, record_errors=True).run(
            CLEF, PAYLOAD, meta=META
        )
    assert KEY not in path.read_text()


# --------------------------------------------------------------------------- #
# Evals
# --------------------------------------------------------------------------- #


def test_the_evals_cli_takes_a_provider(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _parser().parse_args(["run", "--live", "--provider", "typesafe"])
    assert _settings(args, {}).provider == "typesafe"
    assert _settings(_parser().parse_args(["run"]), {}).provider is None
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--provider", "openai"])
    decisions = _parser().parse_args(["run", "--live", "--clef-provider", "decisions"])
    assert _settings(decisions, {}).clef_provider == "decisions"
    # a live run on TypeSafe without the key stops before any request
    status = evals_main(
        ["run", "--live", "--provider", "typesafe", "--quiet", "--cases", CASES[0].id]
    )
    assert status == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
    # a Cloudflare credential does not help; the Decisions API needs its own key
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    status = evals_main(
        [
            "run",
            "--live",
            "--clef-provider",
            "decisions",
            "--quiet",
            "--cases",
            CASES[0].id,
        ]
    )
    assert status == 2
    assert "EXPERIENTIAL_API_KEY" in capsys.readouterr().err


def test_a_provider_flag_without_live_says_it_does_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    status = evals_main(
        [
            "run",
            "--provider",
            "typesafe",
            "--quiet",
            "--cases",
            CASES[0].id,
            "--out",
            str(tmp_path),
        ]
    )
    assert status == 0
    assert "--provider" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("provider", "clef", "fallback", "extraction", "decisions"),
    [
        ("cloudflare", "cloudflare", "none", "cloudflare clairic", "cloudflare clef"),
        (
            "cloudflare",
            "experiential",
            "none",
            "cloudflare clairic",
            "experiential clef",
        ),
        (
            "cloudflare",
            "decisions",
            "none",
            "cloudflare clairic",
            "experiential decisions (gpt-6-luna-decisions, hard cap 0.8)",
        ),
        (
            "cloudflare",
            "cloudflare",
            "decisions",
            "cloudflare clairic",
            "cloudflare clef + fallback decisions",
        ),
        ("typesafe", "typesafe", "none", "typesafe jev", "typesafe jev"),
    ],
)
def test_combination_labels(
    provider: str, clef: str, fallback: str, extraction: str, decisions: str
) -> None:
    assert combination(provider, clef, fallback) == (extraction, decisions)


async def test_the_suite_records_which_combination_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay with nothing recorded sends nothing, yet knows who would answer."""
    cassettes = tmp_path / "cassettes"
    cassettes.mkdir()
    (cassettes / "empty.jsonl").write_text("")
    cases = CASES[:1]
    seen: dict[str, SuiteResult] = {}
    for name, settings in {
        "default": Settings(replay=cassettes),
        "decisions": Settings(replay=cassettes, clef_provider="decisions"),
        "typesafe": Settings(replay=cassettes, provider="typesafe"),
    }.items():
        seen[name] = await run_suite(cases, settings)
    assert (seen["default"].extraction, seen["default"].decisions) == (
        "cloudflare clairic",
        "cloudflare clef",
    )
    assert seen["decisions"].decisions.startswith(
        "experiential decisions (gpt-6-luna-decisions"
    )
    assert seen["decisions"].clef_provider == "decisions"
    assert (seen["typesafe"].extraction, seen["typesafe"].decisions) == (
        "typesafe jev",
        "typesafe jev",
    )
    assert seen["typesafe"].provider == "typesafe"
    js, md = write_suite(seen["typesafe"], tmp_path / "out")
    data = json.loads(js.read_text())
    assert data["provider"] == "typesafe"
    assert data["extraction"] == "typesafe jev"
    assert data["decisions"] == "typesafe jev"
    assert "Extraction: typesafe jev; decisions: typesafe jev" in md.read_text()
    assert "Extraction: typesafe jev" in suite_markdown(seen["typesafe"])


async def test_an_offline_suite_says_it_was_scripted(tmp_path: Path) -> None:
    suite = await run_suite(CASES[:2], Settings())
    assert (suite.extraction, suite.decisions) == ("scripted world", "scripted world")
    js, _ = write_suite(suite, tmp_path)
    assert json.loads(js.read_text())["extraction"] == "scripted world"


async def test_run_suite_live_typesafe_needs_only_the_typesafe_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        await run_suite(CASES[:1], Settings(live=True, provider="typesafe"))
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        await run_suite(CASES[:1], Settings(live=True))  # the environment decides
    monkeypatch.delenv("TRUSTCALL_S1_PROVIDER")
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)  # a key alone selects nothing
    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        await run_suite(CASES[:1], Settings(live=True))


def test_the_live_tests_and_the_examples_honour_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from tests.live.conftest import credentials_present

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import _common

    assert credentials_present() is False
    assert _common.live_available({}) is False
    assert _common.live_available({"TYPESAFE_API_KEY": KEY}) is False  # a key alone: no
    typesafe = {"TRUSTCALL_S1_PROVIDER": "typesafe", "TYPESAFE_API_KEY": KEY}
    assert _common.live_available(typesafe) is True
    assert _common.live_available({"TRUSTCALL_S1_PROVIDER": "typesafe"}) is False
    assert _common.resolve_offline(False, typesafe) is False
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    assert credentials_present() is False
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    assert credentials_present() is True  # no Cloudflare credential needed
    monkeypatch.delenv("TRUSTCALL_S1_PROVIDER")
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "decisions")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    assert (
        credentials_present() is False
    )  # the Decisions API needs the Experiential key
    monkeypatch.setenv("EXPERIENTIAL_API_KEY", "k")
    assert credentials_present() is True
