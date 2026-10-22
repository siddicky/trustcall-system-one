"""Offline replays of recorded live traffic (``tests/recorded``).

Two kinds of cassette live there:

* ``smoke_<name>.jsonl``, written by ``make record`` (``tests/live/test_smoke.py``).
  Each is replayed here, strictly, through the same scenario and the same
  invariants as the live smoke test. A scenario that has not been recorded yet is
  skipped; the day it is, every ``make test`` re-checks it. If a change to the
  library alters the requests, the replay fails with a ``CassetteMissError`` naming
  the call: record again.
* ``legacy_05207aa_*.jsonl``, converted from the wire traces of the second live run
  (``trace_to_cassette``; commit 05207aa). They keep the real answers of the real
  models for examples 03 and 06. Since then M9 reworded the attribution question
  (arithmetic and date math are named in the true-case), so those requests no
  longer hash to the recorded keys, and the key of every other request still does.
  The tests below pin exactly that: all but the attribution requests are replayed
  strictly, the attribution requests are answered with what the model said to the
  same questions about the same entity (the one deliberate loosening, in the test
  and not in ``CassetteTransport``), and the outcomes of the live run are
  asserted. Example 05 is not kept: Clairic's list answers changed the mentions it
  discovers (the span/value realignment of M9), so nearly every request after
  discovery has a new key.
"""

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from evals.scenarios import CONTACT_EXISTING, CONTACT_NOTE, PASSAGE, Colleague
from evals.smoke import SCENARIOS, Scenario
from trustcall_system_one import (
    CallMeta,
    Limits,
    Many,
    One,
    TransportResponse,
    create_extractor,
)
from trustcall_system_one.testing import CassetteMissError, CassetteTransport

RECORDED = Path(__file__).resolve().parents[1] / "recorded"


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    email: str
    phone: str | None = None
    city: str | None = None
    company: str | None = None
    title: str | None = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


# --------------------------------------------------------------------------- #
# Cassettes written by `make record`
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_a_recorded_smoke_scenario_replays_offline(scenario: Scenario) -> None:
    path = RECORDED / f"smoke_{scenario.name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} is not recorded yet: `make record` writes it")
    cassette = CassetteTransport(path)
    extractor = create_extractor(
        scenario.schemas(), transport=cassette, **scenario.options
    )
    result = extractor.invoke(scenario.text, existing=scenario.existing)
    extractor.close()
    assert cassette.misses == [], "the requests changed since the cassette was recorded"
    scenario.check(result)


# --------------------------------------------------------------------------- #
# The traces of the second live run
# --------------------------------------------------------------------------- #


def quoted(question: dict[str, Any]) -> str | None:
    """The value an attribution question asks about (its first quoted text)."""
    found = re.search(r'"((?:[^"\\]|\\.)*)"', question["instructions"])
    return found.group(1) if found else None


class RewordedAttribution:
    """Answers an attribution request with the recorded answer to the same questions.

    The key is the question ids (``attr.city``), the entity, the whole state
    (byte-identical) and the value each question quotes; the wording of the
    questions is what changed, so it is the one thing not compared. Any other kind
    of call is a miss that fails the test.
    """

    def __init__(self, path: Path) -> None:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.rows = [
            r
            for r in rows
            if all(
                q.startswith("attr.") for q in r["request"].get("questions", {"": 0})
            )
        ]
        self.served: list[tuple[str, ...]] = []

    async def run(
        self,
        model: str,
        payload: Any,
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        assert meta.kind == "attribution", f"a {meta.kind!r} request changed"
        ids = set(payload["questions"])
        label = payload["state"]["entity"]["label"]
        for row in self.rows:
            request = row["request"]
            # the same questions about the same value, over a byte-identical state:
            # only the wording of the question may differ
            if (
                set(request["questions"]) == ids
                and request["state"] == payload["state"]
                and request["state"]["entity"]["label"] == label
                and all(
                    quoted(request["questions"][q]) == quoted(payload["questions"][q])
                    for q in ids
                )
            ):
                self.served.append(tuple(sorted(ids)))
                return TransportResponse(
                    body=copy.deepcopy(row["response"]), status_code=200
                )
        raise AssertionError(f"no recorded attribution for {sorted(ids)} about {label}")


def replay(
    name: str, schemas: list[Any], text: str, **kwargs: Any
) -> tuple[Any, CassetteTransport, RewordedAttribution]:
    path = RECORDED / f"legacy_05207aa_{name}.jsonl"
    inner = RewordedAttribution(path)
    cassette = CassetteTransport(path, strict=False, inner=inner)
    # the library of that commit had no sentence reading: Clairic's list only
    extractor = create_extractor(
        schemas, transport=cassette, limits=Limits(list_strategy="list"), **kwargs
    )
    return extractor, cassette, inner


def test_the_recorded_update_of_a_contact_gives_the_live_patch() -> None:
    extractor, cassette, inner = replay(
        "03_update_contact",
        [One(Contact)],
        CONTACT_NOTE,
    )
    result = extractor.invoke(
        CONTACT_NOTE, existing=[("crm-4711", "Contact", CONTACT_EXISTING)]
    )
    # 8 of the 9 recorded requests hash to the same key; the ninth is the
    # attribution request whose wording changed
    assert (cassette.hits, len(cassette.misses), len(cassette)) == (8, 1, 9)
    assert [m[2].kind for m in cassette.misses] == ["attribution"]
    assert len(inner.served) == 1
    (record,) = result.records
    assert record.kind == "update" and record.json_doc_id == "crm-4711" and record.valid
    assert [(p.op, p.path, p.value) for p in record.patches] == [
        ("replace", "/phone", "+49 30 5555 0198"),
        ("replace", "/city", "Berlin"),
        ("replace", "/company", "Northwind"),
        ("replace", "/title", "Head of Ops"),
    ]
    # what the note does not state stays: the live run appended no note either
    # (Clairic's list confidence was 3.2e-11), and "Dana" did not rename the contact
    assert record.after is not None
    assert record.after["name"] == CONTACT_EXISTING["name"]
    assert record.after["email"] == CONTACT_EXISTING["email"]
    assert record.after["notes"] == CONTACT_EXISTING["notes"]
    reasons = {r.path: r.reason for r in record.rejected}
    assert reasons["/email"] == "echo_span"
    assert reasons["/notes"] == "low_confidence"
    assert reasons["/name"] == "identity_subset"
    cassette.reset()


def test_the_recorded_attribution_guard_gives_the_live_records() -> None:
    extractor, cassette, inner = replay(
        "06_attribution_guard",
        [Many(Colleague, identity="name")],
        PASSAGE,
    )
    result = extractor.invoke(PASSAGE)
    assert (cassette.hits, len(cassette.misses), len(cassette)) == (7, 2, 9)
    assert {m[2].kind for m in cassette.misses} == {"attribution"}
    assert len(inner.served) == 2
    people = {
        r.entity.label: r for r in result.records if r.entity is not None and r.valid
    }
    assert set(people) == {"Priya Raman", "Tom Becker"}
    priya, tom = people["Priya Raman"].value, people["Tom Becker"].value
    assert priya is not None and tom is not None
    assert (priya.name, priya.city, priya.role) == ("Priya Raman", "Lisbon", None)
    assert (tom.name, tom.role) == ("Tom Becker", "finance lead")
    # the live run kept Tom's Lisbon: attributed 0.7736 against a keep threshold of
    # 0.75. The cassette makes that margin visible, and a regression test for it.
    assert tom.city == "Lisbon"
    attribution = people["Tom Becker"].fields["city"].attribution_p
    assert isinstance(attribution, float) and attribution == pytest.approx(
        0.7736, abs=1e-3
    )


def test_a_cassette_miss_names_the_request_that_changed() -> None:
    path = RECORDED / "legacy_05207aa_06_attribution_guard.jsonl"
    extractor = create_extractor(
        [Many(Colleague, identity="name")], transport=CassetteTransport(path)
    )
    with pytest.raises(CassetteMissError) as caught:
        extractor.invoke(PASSAGE)
    assert caught.value.kind == "attribution"
    assert caught.value.schema == "Colleague"
    assert "changed since the cassette was recorded" in str(caught.value)
