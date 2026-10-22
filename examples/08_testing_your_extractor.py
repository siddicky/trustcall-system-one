"""How to unit-test code that calls your extractor, with no network and no keys.

Runs today. ``uv run python examples/08_testing_your_extractor.py``, or run the
very same functions under pytest:
``uv run pytest examples/08_testing_your_extractor.py``.

This file is a miniature test module for an imaginary app function,
``parse_meeting_request(text, transport)``. It shows the three things that make
offline tests of an extractor trustworthy:

* ``FakeTransport`` takes a handler ``(model, payload, meta) -> response body`` and
  validates BOTH sides against the published Clairic and Clef schemas, stricter
  than the real API. A request the library should not send, or a fake answer that
  does not fit the request (a missing field, a choice nobody offered), fails your
  test instead of passing it for the wrong reason. Test 4 shows it.
* ``route(presence=..., extract=..., ...)`` dispatches on ``CallMeta.kind``, so each
  kind of call (presence gate, Clairic extraction, Clef decisions, attribution
  check) gets its own small handler. ``transport.calls`` records every request, so a
  test can also assert what was NOT asked (test 2: no extraction after a failed
  presence gate).
* Spans are computed with ``str.find`` over the state the library actually rendered
  (``find_span`` below), so the offsets in your fake answers are the real ones.

``testing.FakeWorld`` (test 7) answers every call kind of a multi-entity run from one
declaration of text, entities and faults, so a test says what the text contains and
asserts what the library made of it; for a single entity the hand-written handler
table above is all you need.

Look at the output: one line per test, then the request log of the last test.
There is no live mode here; the point is that tests never need one.
"""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import heading, paint, print_banner
from pydantic import BaseModel, Field

from trustcall_system_one import (
    CallMeta,
    Extractor,
    Many,
    Transport,
    create_extractor,
)
from trustcall_system_one.testing import (
    FakeTransport,
    FakeWorld,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    route,
)

# --------------------------------------------------------------------------- #
# The code under test: your application
# --------------------------------------------------------------------------- #


class MeetingRequest(BaseModel):
    """A request to schedule a meeting."""

    requester: str = Field(description="person asking for the meeting")
    topic: str = Field(description="what the meeting is about")
    proposed_date: date | None = Field(None, description="date the requester proposes")
    urgent: bool = Field(False, description="the requester says it is urgent")


def make_extractor(transport: Transport | None = None) -> Extractor:
    """Build the extractor ONCE per process (it owns a connection pool and a loop)."""
    return create_extractor([MeetingRequest], transport=transport)


def parse_meeting_request(text: str, extractor: Extractor) -> MeetingRequest | None:
    """Your function: the first meeting request in ``text``, or ``None``."""
    result = extractor.invoke(text)
    return next((r for r in result.responses if isinstance(r, MeetingRequest)), None)


# --------------------------------------------------------------------------- #
# Test helpers: a tiny fake world for one email
# --------------------------------------------------------------------------- #

EMAIL = (
    "Hi team, Maria Alvarez here. Could we meet on 2024-06-18 to go over the "
    "Q3 roadmap review? It is urgent, the board sees it on Friday."
)

Body = dict[str, Any]


def find_span(state: str, needle: str) -> list[int]:
    """Span of ``needle`` in the state Clairic is sent (after the <input> tag)."""
    start = state.index(needle, state.index("<input>"))
    return [start, start + len(needle)]


def fake_models(
    *,
    presence: float = 0.97,
    attribution: dict[str, float] | None = None,
    urgent: float = 0.93,
) -> FakeTransport:
    """A transport whose "models" read ``EMAIL`` like a sensible model would."""
    attribution = attribution or {}

    def gate(model: str, payload: Body, meta: CallMeta) -> Body:
        return clef_response({"presence": noul(presence)})

    def extract(model: str, payload: Body, meta: CallMeta) -> Body:
        state = payload["state"]
        known = {
            "requester": clairic_answer(
                "text", "Maria Alvarez", span=find_span(state, "Maria Alvarez")
            ),
            "topic": clairic_answer(
                "text", "Q3 roadmap review", span=find_span(state, "Q3 roadmap review")
            ),
            "proposed_date": clairic_answer("date", "2024-06-18"),
        }
        # FakeTransport insists on exactly the requested fields.
        answers = {name: known[name] for name in payload["fields"]}
        return clairic_response(answers, state_text=state)

    def decide(model: str, payload: Body, meta: CallMeta) -> Body:
        return clef_response({q: noul(urgent) for q in payload["questions"]})

    def check_attribution(model: str, payload: Body, meta: CallMeta) -> Body:
        answers = {q: noul(attribution.get(q, 0.96)) for q in payload["questions"]}
        return clef_response(answers)

    return FakeTransport(
        route(
            presence=gate,
            extract=extract,
            clef_fields=decide,
            attribution=check_attribution,
        )
    )


@dataclass
class World:
    """A fake transport and the one extractor that talks to it."""

    transport: FakeTransport
    extractor: Extractor


@contextmanager
def world(transport: FakeTransport | None = None, **faults: Any) -> Iterator[World]:
    """One fake world and ONE extractor for a whole test, closed afterwards.

    With pytest this is a fixture: ``@pytest.fixture`` and ``yield`` instead of
    ``@contextmanager``. Build the extractor once and reuse it for every call in
    the test, as your application does; ``faults`` are the knobs of ``fake_models``.
    """
    fake = transport or fake_models(**faults)
    extractor = make_extractor(fake)
    try:
        yield World(fake, extractor)
    finally:
        extractor.close()


def kinds(transport: FakeTransport) -> list[str]:
    """The kinds of call a test run made, in order."""
    return [meta.kind for _, _, meta in transport.calls]


# --------------------------------------------------------------------------- #
# The tests (pytest collects these; main() below also runs them)
# --------------------------------------------------------------------------- #


def test_extracts_the_request() -> None:
    with world() as w:
        request = parse_meeting_request(EMAIL, w.extractor)
    assert request == MeetingRequest(
        requester="Maria Alvarez",
        topic="Q3 roadmap review",
        proposed_date=date(2024, 6, 18),
        urgent=True,
    )


def test_text_without_a_request_makes_no_extraction_calls() -> None:
    with world(presence=0.02) as w:
        assert parse_meeting_request("Lunch menu: soup and bread.", w.extractor) is None
    assert kinds(w.transport) == ["presence"]  # nothing else was asked


def test_a_value_the_text_does_not_support_is_dropped() -> None:
    # The attribution check says the date is not stated about this request.
    with world(attribution={"attr.proposed_date": 0.05}) as w:
        request = parse_meeting_request(EMAIL, w.extractor)
    assert request is not None
    assert request.proposed_date is None  # dropped, not trusted
    assert request.requester == "Maria Alvarez"  # the rest is unaffected


def test_the_fake_rejects_answers_that_do_not_fit_the_request() -> None:
    def forgetful(model: str, payload: Body, meta: CallMeta) -> Body | None:
        # answers a question that was not asked, and not the one that was
        return clef_response({"is_it_a_meeting": noul(0.97)})

    with world(FakeTransport(forgetful)) as w:
        try:
            parse_meeting_request(EMAIL, w.extractor)
        except AssertionError as error:  # with pytest: `with pytest.raises(...)`
            assert "differ from the asked questions" in str(error)
        else:
            raise AssertionError("the fake should have refused the wrong answer")


def test_the_fake_rejects_an_incomplete_extraction_answer() -> None:
    def forgetful(model: str, payload: Body, meta: CallMeta) -> Body:
        if meta.kind == "presence":
            return clef_response({"presence": noul(0.97)})
        if meta.kind == "extract":  # answers only one of the requested fields
            return clairic_response({"requester": clairic_answer("text", None)})
        # every other kind answers correctly, so exactly one call is wrong
        return clef_response({q: noul(0.9) for q in payload["questions"]})

    with world(FakeTransport(forgetful)) as w:
        try:
            parse_meeting_request(EMAIL, w.extractor)
        except AssertionError as error:  # raised inside a concurrent step, as itself
            assert "differ from the requested" in str(error)
        else:
            raise AssertionError("the fake should have refused the incomplete answer")


def test_the_request_carries_the_text_and_the_field_descriptions() -> None:
    with world() as w:
        parse_meeting_request(EMAIL, w.extractor)
        parse_meeting_request(EMAIL, w.extractor)  # the same extractor, reused
    extracts = [c for c in w.transport.calls if c[2].kind == "extract"]
    assert len(extracts) == 2
    _, payload, _ = extracts[0]
    assert EMAIL in payload["state"]
    described = " ".join(f["instructions"] for f in payload["fields"].values())
    assert "person asking for the meeting" in described


class Attendee(BaseModel):
    """Someone who will attend a meeting."""

    name: str = Field(description="full name")
    role: str | None = Field(None, description="the person's role")


ATTENDEES = "Maria Alvarez will present. Tom Becker, the finance lead, will take notes."


def attendee_world() -> FakeWorld:
    """Declare WHAT the text contains; the world answers every kind of call."""
    world = FakeWorld(ATTENDEES)
    world.entity(
        "Maria Alvarez",
        schema=Attendee,
        mentions=["Maria Alvarez"],
        fields={"name": "Maria Alvarez"},
    )
    world.entity(
        "Tom Becker",
        schema=Attendee,
        mentions=["Tom Becker"],
        fields={"name": "Tom Becker", "role": "finance lead"},
    )
    return world


def test_a_fake_world_answers_every_call_of_a_multi_entity_run() -> None:
    extractor = create_extractor(
        [Many(Attendee, identity="name")], transport=attendee_world().transport()
    )
    with extractor:
        result = extractor.invoke(ATTENDEES)
    maria, tom = result.responses
    assert (maria.name, maria.role) == ("Maria Alvarez", None)  # type: ignore[attr-defined]
    assert (tom.name, tom.role) == ("Tom Becker", "finance lead")  # type: ignore[attr-defined]
    assert [r.entity.label for r in result.records if r.entity] == [
        "Maria Alvarez",
        "Tom Becker",
    ]


def test_a_fault_injected_into_the_world_is_caught_by_attribution() -> None:
    world = attendee_world()
    world.bleed(
        "Maria Alvarez", "role", "finance lead"
    )  # Tom's role, offered for Maria
    extractor = create_extractor(
        [Many(Attendee, identity="name")], transport=world.transport()
    )
    with extractor:
        result = extractor.invoke(ATTENDEES)
    maria, tom = result.responses
    assert maria.role is None  # type: ignore[attr-defined]
    assert tom.role == "finance lead"  # type: ignore[attr-defined]
    assert [(r.path, r.reason) for r in result.records[0].rejected] == [
        ("/role", "attribution")
    ]


TESTS: list[Callable[[], None]] = [
    test_extracts_the_request,
    test_text_without_a_request_makes_no_extraction_calls,
    test_a_value_the_text_does_not_support_is_dropped,
    test_the_fake_rejects_answers_that_do_not_fit_the_request,
    test_the_fake_rejects_an_incomplete_extraction_answer,
    test_the_request_carries_the_text_and_the_field_descriptions,
    test_a_fake_world_answers_every_call_of_a_multi_entity_run,
    test_a_fault_injected_into_the_world_is_caught_by_attribution,
]


def main(offline: bool = True) -> dict[str, bool]:
    """Run the tests above and print one line each; return ``{name: passed}``.

    Args:
        offline: Ignored: these tests never use the network, which is the point.
    """
    del offline
    print_banner(True, live_hint=False)
    heading("Running the example's own tests against a FakeTransport")
    outcome: dict[str, bool] = {}
    for test in TESTS:
        try:
            test()
        except AssertionError as error:
            outcome[test.__name__] = False
            print(f"  {paint('FAIL', 'red')}  {test.__name__}: {error}")
        else:
            outcome[test.__name__] = True
            print(f"  {paint('ok', 'green')}    {test.__name__}")
    print()
    with world() as w:
        parse_meeting_request(EMAIL, w.extractor)
    heading("What the library asked the fake (transport.calls) for one email")
    for _, _, meta in w.transport.calls:
        print(f"  {meta.kind:<12} entity={meta.entity} field={meta.field}")
    print()
    return outcome


if __name__ == "__main__":
    main()
