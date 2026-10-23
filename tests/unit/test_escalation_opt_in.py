"""The escalation opt-in is explicit and never sticks on a transport passed in.

Every text is synthetic; nothing touches the network (a transport that would send is
never run).
"""

from typing import Any

import pytest
from pydantic import BaseModel

from evals.runner import Settings, run_suite
from tests.unit.test_escalation_evals import CASE
from trustcall_system_one import (
    CallMeta,
    CloudflareTransport,
    EscalatingTransport,
    EscalationConfig,
    TypeSafeTransport,
    create_extractor,
    system_one_transport,
)
from trustcall_system_one.testing import (
    CassetteTransport,
    FakeTransport,
    RecordingTransport,
)
from trustcall_system_one.transport import (
    FallbackTransport,
    FieldTypeRoutingTransport,
    KindRoutingTransport,
    RoutingTransport,
    escalation_of,
    max_questions,
    prefers_keep_uncertain,
    state_token_budget,
)


class _Doc(BaseModel):
    name: str


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TRUSTCALL_S1_PROVIDER",
        "TRUSTCALL_S1_CLEF_PROVIDER",
        "TRUSTCALL_S1_CLEF_FALLBACK",
        "TRUSTCALL_S1_ESCALATE",
        "TRUSTCALL_S1_ESCALATE_MODE",
        "TRUSTCALL_S1_ESCALATE_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


def cloudflare() -> CloudflareTransport:
    return CloudflareTransport(account_id="0" * 32, api_token="t" * 24)


# --------------------------------------------------------------------------- #
# A transport that was passed in is not changed
# --------------------------------------------------------------------------- #


def test_the_opt_in_never_sticks_on_the_transport_that_was_passed_in() -> None:
    given = cloudflare()
    tier = system_one_transport(cloudflare=given, escalate="openai")
    assert isinstance(tier, EscalatingTransport)
    assert tier._inner is given  # the wrapper holds it; the object itself is unchanged
    assert escalation_of(tier) is not None and escalation_of(tier).enabled  # type: ignore[union-attr]
    assert "escalation" not in vars(given) and escalation_of(given) is None
    # a later transport built on the same object, and an extractor on it, send nothing
    later = system_one_transport(cloudflare=given)
    assert later is given and escalation_of(later) is None
    extractor = create_extractor([_Doc], transport=given)
    assert not extractor._escalation_for_run().enabled
    assert (
        escalation_of(system_one_transport(cloudflare=given, escalate="none")) is None
    )
    # the extractor on the wrapper does carry it
    assert create_extractor([_Doc], transport=tier).core.escalation(None) is not None


def test_the_wrapper_delegates_everything_else() -> None:
    given = cloudflare()
    tier = EscalatingTransport(given, EscalationConfig(enabled=True))
    assert max_questions(tier, "@cf/cloudflare/clef") == max_questions(
        given, "@cf/cloudflare/clef"
    )
    assert state_token_budget(tier, "@cf/cloudflare/clef") == state_token_budget(
        given, "@cf/cloudflare/clef"
    )
    assert prefers_keep_uncertain(tier, "@cf/cloudflare/clef") is False
    assert tier.serves("@cf/cloudflare/clef")
    assert tier.sensitive_values() == given.sensitive_values()
    assert tier.classify_failure(ValueError("x")) == given.classify_failure(
        ValueError("x")
    )
    assert tier.provider == given.provider


async def test_the_wrapper_closes_what_it_wraps() -> None:
    closed: list[str] = []

    class Inner(FakeTransport):
        async def aclose(self) -> None:
            closed.append("aclose")

        def close(self) -> None:
            closed.append("close")

    inner = Inner(lambda *a: {})
    wrapper = EscalatingTransport(inner, EscalationConfig(enabled=True))
    await wrapper.aclose()
    wrapper.close()
    assert closed == ["aclose", "close"]


async def test_the_wrapper_runs_the_call_on_the_wrapped_transport() -> None:
    seen: list[tuple[str, float | None]] = []

    class Inner:
        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: float | None = None
        ) -> str:
            seen.append((model, timeout))
            return "answer"

    wrapper = EscalatingTransport(Inner(), EscalationConfig(enabled=True))  # type: ignore[arg-type]
    reply = await wrapper.run("m", {}, meta=CallMeta(kind="extract"), timeout=3.0)
    assert str(reply) == "answer" and seen == [("m", 3.0)]


def test_the_wrappers_forward_the_choice(tmp_path: Any) -> None:
    config = EscalationConfig(enabled=True)
    carrier = EscalatingTransport(cloudflare(), config)
    assert escalation_of(RecordingTransport(carrier, tmp_path / "r.jsonl")) is config
    assert escalation_of(CassetteTransport(tmp_path, inner=carrier)) is config
    assert escalation_of(CassetteTransport(tmp_path)) is None
    assert escalation_of(FallbackTransport(carrier, cloudflare())) is config
    assert escalation_of(FallbackTransport(cloudflare(), carrier)) is None
    assert escalation_of(RoutingTransport({}, default=carrier)) is config
    assert escalation_of(KindRoutingTransport({}, default=carrier)) is config
    assert escalation_of(FieldTypeRoutingTransport({}, default=carrier)) is config
    plain = cloudflare()
    assert escalation_of(RoutingTransport({}, default=plain)) is None
    assert escalation_of(object()) is None


# --------------------------------------------------------------------------- #
# TypeSafe only
# --------------------------------------------------------------------------- #


def test_the_environment_cannot_combine_typesafe_with_the_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "openai")
    with pytest.raises(ValueError, match="TypeSafe only"):
        create_extractor([_Doc])
    # the explicit API refuses it the same way
    with pytest.raises(ValueError, match="TypeSafe only"):
        system_one_transport(provider="typesafe", escalate="openai")
    # either alone is fine
    monkeypatch.delenv("TRUSTCALL_S1_ESCALATE")
    create_extractor([_Doc]).close()
    monkeypatch.delenv("TRUSTCALL_S1_PROVIDER")
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "openai")
    create_extractor([_Doc]).close()


def test_an_explicit_tier_cannot_combine_with_a_typesafe_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    with pytest.raises(ValueError, match="TypeSafe only"):
        create_extractor([_Doc], escalation=EscalationConfig(enabled=True))
    # a transport that was passed in is the caller's explicit choice
    create_extractor(
        [_Doc],
        transport=TypeSafeTransport(api_key="k" * 20),
        escalation=EscalationConfig(enabled=True),
    )


async def test_the_eval_runner_refuses_typesafe_with_the_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    with pytest.raises(ValueError, match="TypeSafe only"):
        await run_suite([CASE], Settings(escalate="openai"))
    monkeypatch.delenv("TRUSTCALL_S1_PROVIDER")
    monkeypatch.setenv("TRUSTCALL_S1_ESCALATE", "openai")
    with pytest.raises(ValueError, match="TypeSafe only"):
        await run_suite([CASE], Settings(provider="typesafe"))
