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

import collections
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
    DiscoveryConfig,
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


def mention_texts(state: dict[str, Any]) -> list[str]:
    return [m["text"] for m in state["entity"].get("mentions", [])]


def same_entity(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Whether a reworded request is about the entity the recording was about.

    The mentions are the same; the label is the same or has grown (a single record's
    label carries what it is called now: "the Contact" became "the Contact Dana
    Whitfield").
    """
    before, after = old["entity"], new["entity"]
    return mention_texts(old) == mention_texts(new) and str(after["label"]).startswith(
        str(before["label"])
    )


def _asks(row: dict[str, Any], prefix: str) -> bool:
    questions = row["request"].get("questions", {})
    return bool(questions) and all(q.startswith(prefix) for q in questions)


class Reworded:
    """Answers the requests whose wording or packing changed with what the model said.

    Attribution: one question at a time (a decision gets a request of its own now,
    the recording batched them), matched on the question id, the input, the
    entity's mentions and the value the question quotes; the wording of the
    question and the label (a single record's carries its identity now) are what
    changed, so they are the things not compared. Qualification: the same entities
    in the same order (the question no longer carries the schema's description).
    Presence: the same input (the question names the kind of thing, "person", and
    not the schema's title). Any other kind of call is a miss that fails the test.
    """

    def __init__(self, path: Path) -> None:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # converted traces carry no call kind: the question ids tell what a row is
        self.attribution = [r for r in rows if _asks(r, "attr.")]
        self.qualify = [r for r in rows if _asks(r, "qualify.")]
        self.gate = [r for r in rows if _asks(r, "gate")]
        self.presence = [r for r in rows if _asks(r, "presence")]
        self.served: list[tuple[str, ...]] = []

    def _answer(self, row: dict[str, Any], qids: list[str]) -> TransportResponse:
        body = copy.deepcopy(row["response"])
        body["answers"] = {q: body["answers"][q] for q in qids}
        return TransportResponse(body=body, status_code=200)

    async def run(
        self,
        model: str,
        payload: Any,
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        if meta.kind == "qualify":
            labels = [e["label"] for e in payload["state"]["entities"].values()]
            for row in self.qualify:
                shown = [
                    e["label"] for e in row["request"]["state"]["entities"].values()
                ]
                if shown == labels and set(row["request"]["questions"]) == set(
                    payload["questions"]
                ):
                    self.served.append(("qualify", *sorted(payload["questions"])))
                    return self._answer(row, list(payload["questions"]))
            raise AssertionError(f"no recorded qualification of {labels}")
        if meta.kind == "presence":  # it now names the kind of thing, not the title
            for row in self.presence:
                if row["request"]["state"]["input"] == payload["state"]["input"]:
                    self.served.append(("presence",))
                    return self._answer(row, ["presence"])
            raise AssertionError("no recorded presence of this input")
        if meta.kind == "gate":  # the label of a single record carries its identity
            state = payload["state"]
            for row in self.gate:
                request = row["request"]["state"]
                if (
                    request["input"] == state["input"]
                    and request["proposed_change"] == state["proposed_change"]
                    and request["record"] == state["record"]
                    and same_entity(request, state)
                ):
                    self.served.append(("gate", state["proposed_change"]["path"]))
                    return self._answer(row, ["gate"])
            raise AssertionError(f"no recorded gate for {state['proposed_change']}")
        assert meta.kind == "attribution", f"a {meta.kind!r} request changed"
        state = payload["state"]
        found: dict[str, TransportResponse] = {}
        for qid, question in payload["questions"].items():
            for row in self.attribution:
                request = row["request"]
                other = request["questions"].get(qid)
                if (
                    other is not None
                    and request["state"]["input"] == state["input"]
                    and same_entity(request["state"], state)
                    and quoted(other) == quoted(question)
                ):
                    found[qid] = self._answer(row, [qid])
                    break
            else:
                raise AssertionError(
                    f"no recorded attribution for {qid} about "
                    f"{state['entity']['label']}"
                )
        body = copy.deepcopy(next(iter(found.values())).body)
        body["answers"] = {q: r.body["answers"][q] for q, r in found.items()}
        self.served.append(tuple(sorted(found)))
        return TransportResponse(body=body, status_code=200)


def replay(
    name: str, schemas: list[Any], text: str, **kwargs: Any
) -> tuple[Any, CassetteTransport, Reworded]:
    path = RECORDED / f"legacy_05207aa_{name}.jsonl"
    inner = Reworded(path)
    cassette = CassetteTransport(path, strict=False, inner=inner)
    # the library of that commit had no sentence reading: Clairic's list only; and
    # it asked a record's presence of nobody
    extractor = create_extractor(
        schemas,
        transport=cassette,
        limits=Limits(list_strategy="list"),
        discovery=DiscoveryConfig(record_presence=0),
        **kwargs,
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
    # the requests that did not change hash to their keys still, and exactly these
    # do (presence, the reading with the record, the input-only reading and one
    # re-ask): 4 of the 9 recorded. The attribution questions (one per request now,
    # five of them) and
    # the gates (a single record's label carries its identity, four) are answered
    # with what the model said to the same questions
    assert cassette.hits == 4 and len(cassette) == 9
    assert collections.Counter(m[2].kind for m in cassette.misses) == {
        "attribution": 5,
        "gate": 4,
    }
    assert len(inner.served) == 9
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
    assert collections.Counter(m[2].kind for m in cassette.misses) == {
        "attribution": 5,
        "qualify": 1,
        "presence": 1,
    }
    assert cassette.hits == 5 and len(cassette) == 9 and len(inner.served) == 7
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
    # presence is the first request M10c reworded (it names the kind of thing)
    assert caught.value.kind == "presence"
    assert caught.value.schema == "Colleague"
    assert "changed since the cassette was recorded" in str(caught.value)
