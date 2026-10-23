"""Regressions of the review of the integrated fixes (probes turned into tests)."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import Script, ca, make_extractor
from trustcall_system_one import CallMeta
from trustcall_system_one._sentences import ITEMS_VETO, _hedged_veto
from trustcall_system_one._wire import ChoiceAnswer
from trustcall_system_one.testing import choice, clef_response

__all__: list[str] = []

# --------------------------------------------------------------------------- #
# S3: the items veto is for hedging readers only (probe_items)
# --------------------------------------------------------------------------- #


class Contact(BaseModel):
    """A contact."""

    name: str = Field(description="the person's full name")
    phones: list[str] = Field(default_factory=list, description="phone numbers")


PHONES = "Dana Ruiz is our new lead. Call her on 555-0101 or 555-0199 any time."


class _Kind(Script):
    p_items = 0.7

    def _list_kind(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {
                q: choice(
                    "items", {"items": self.p_items, "statements": 1 - self.p_items}
                )
                for q in payload["questions"]
            }
        )


@pytest.mark.parametrize("p_items", [0.7, 0.6, 0.55])
async def test_phone_numbers_stay_items_when_clef_grades_them_as_items(
    p_items: float,
) -> None:
    world = _Kind(
        clairic={
            "name": ca("text", "Dana Ruiz", find="Dana Ruiz", confidence=0.99),
            "phones": ca(
                "list", None, find_all=["555-0101", "555-0199"], confidence=0.3
            ),
        },
        reask={
            "phones": ca(
                "list", None, find_all=["555-0101", "555-0199"], confidence=0.3
            )
        },
        sentences={"555-0101": 0.9},
    )
    world.p_items = p_items
    (record,) = (await make_extractor([Contact], world).ainvoke(PHONES)).records
    # the graded answer of a provider that is not hedging keeps the plain rule: the
    # list is NOT read by sentences, so the phones are never the sentence
    phones = (record.after or {}).get("phones") or []
    assert "Call her on 555-0101 or 555-0199 any time." not in phones
    assert not any("sentences" in w for w in record.fields["phones"].warnings)


def _answer(items: float, *, hard: bool, confidence: float) -> ChoiceAnswer:
    body: dict[str, Any] = {
        "type": "choice",
        "choice": "items",
        "probabilities": {"items": items, "statements": 1 - items},
        "confidence": confidence,
    }
    if hard:
        body["hard"] = True
    return ChoiceAnswer.model_validate(body)


def test_the_veto_is_clamped_to_the_hard_cap() -> None:
    from types import SimpleNamespace

    ctx: Any = SimpleNamespace(run=SimpleNamespace(calls=[]))
    spec: Any = SimpleNamespace(wire_name="notes")
    assert (
        _hedged_veto(ctx, spec, _answer(0.8, hard=True, confidence=0.8)) == ITEMS_VETO
    )
    # a lower cap: a hard "items" holds the cap, which is the veto, so it vetoes
    low = _hedged_veto(ctx, spec, _answer(0.6, hard=True, confidence=0.6))
    assert low == 0.6
    assert low is not None and low <= 0.6
    # a graded answer with no Decisions provider on record does not hedge
    assert _hedged_veto(ctx, spec, _answer(0.7, hard=False, confidence=0.4)) is None
    decisions = SimpleNamespace(kind="list_kind", field="notes", provider="decisions")
    ctx.run.calls.append(decisions)
    assert (
        _hedged_veto(ctx, spec, _answer(0.7, hard=False, confidence=0.4)) == ITEMS_VETO
    )
    ctx.run.calls[:] = [
        SimpleNamespace(kind="list_kind", field="notes", provider="cloudflare")
    ]
    assert _hedged_veto(ctx, spec, _answer(0.7, hard=False, confidence=0.4)) is None
