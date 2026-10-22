"""Carry-overs fixed with M9: partial re-anchoring, the identity guard, list spans.

* R1: a ``state_text`` that only partly re-anchors (CRLF turned into LF) must not
  make correct answers fail the span rule.
* R2: the identity guard does not veto a rename the coreference cluster links.
* L1: Clairic's list ``value`` and ``spans`` are not index-aligned.
* L2: a required list the input says nothing about is ``[]``, not ``null``.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Any, Optional

from pydantic import BaseModel, Field

from tests.unit.extract_helpers import Script, ca, make_extractor
from trustcall_system_one import Many, RepairConfig, create_extractor
from trustcall_system_one._collection import parse_mentions
from trustcall_system_one._render import render_clairic
from trustcall_system_one._run import Run
from trustcall_system_one._verify import align_list_spans
from trustcall_system_one._wire import ClairicAnswer, ClairicResponse
from trustcall_system_one.testing import (
    FakeTransport,
    FakeWorld,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    route,
)


class Contact(BaseModel):
    """A contact."""

    name: str = Field(description="full name")
    city: str = Field(description="the city they live in")


CRLF_TEXT = "Dana Ruiz called today about the invoice.\r\nShe lives in Lyon now."


def crlf_transport(answers: dict[str, tuple[str, str]]) -> FakeTransport:
    """Clairic read the state with its line endings normalised (CRLF -> LF).

    ``answers`` maps a field to the value and the piece of the state that is
    returned as its span (the value itself when empty).
    """

    def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
        if "fields" in payload:
            seen = payload["state"].replace("\r\n", "\n")
            found = {}
            for name in payload["fields"]:
                value, anchor = answers[name]
                anchor = anchor or value
                at = seen.index(anchor)
                found[name] = clairic_answer(
                    "text", value, span=[at, at + len(anchor)], confidence=0.99
                )
            return clairic_response(found, state_text=seen)
        questions = payload["questions"]
        return clef_response({q: noul(0.95) for q in questions}, model=payload["model"])

    return FakeTransport(handler)


# --------------------------------------------------------------------------- #
# R1: the span rule and a state_text that only partly re-anchors
# --------------------------------------------------------------------------- #


async def test_a_correct_answer_after_a_partial_re_anchor_is_kept_and_relocated() -> (
    None
):
    transport = crlf_transport({"name": ("Dana Ruiz", ""), "city": ("Lyon", "")})
    extractor = create_extractor([Contact], transport=transport)
    result = await extractor.ainvoke(CRLF_TEXT)
    (record,) = result.records
    assert record.valid and record.after == {"name": "Dana Ruiz", "city": "Lyon"}
    city = record.fields["city"]
    assert city.span is not None
    assert CRLF_TEXT[city.span[0] : city.span[1]] == "Lyon"
    assert "span_relocated" in city.warnings
    assert "span_outside_input" not in city.warnings
    assert record.fields["name"].span == (0, 9)  # still mapped, not relocated
    assert result.warnings == ()


async def test_a_value_that_is_not_in_the_input_is_still_rejected() -> None:
    # the state has the text after the partial re-anchor point, but not "Paris"
    transport = crlf_transport({"name": ("Dana Ruiz", ""), "city": ("Paris", "Lyon")})
    extractor = create_extractor(
        [Contact], transport=transport, repair=RepairConfig(enabled=False)
    )
    result = await extractor.ainvoke(CRLF_TEXT)
    (record,) = result.records
    assert record.fields["city"].value is None
    assert record.fields["city"].rejected_reason == "span_outside_input"
    assert not record.valid


async def test_a_header_span_is_still_rejected_after_a_partial_re_anchor() -> None:
    def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
        if "fields" in payload:
            seen = payload["state"].replace("\r\n", "\n")
            head = seen.index("Answer only")  # the focus header, before the input
            found = {
                "name": clairic_answer(
                    "text", "Dana Ruiz", span=[head, head + 9], confidence=0.99
                ),
                "city": clairic_answer(
                    "text",
                    "Lyon",
                    span=[seen.index("Lyon"), seen.index("Lyon") + 4],
                    confidence=0.99,
                ),
            }
            return clairic_response(found, state_text=seen)
        return clef_response(
            {q: noul(0.95) for q in payload["questions"]}, model=payload["model"]
        )

    extractor = create_extractor(
        [Contact],
        transport=FakeTransport(handler),
        repair=RepairConfig(enabled=False),
    )
    result = await extractor.ainvoke(CRLF_TEXT)
    (record,) = result.records
    name = record.fields["name"]
    assert name.value is None and name.rejected_reason == "span_outside_input"
    assert record.fields["city"].value == "Lyon"  # the genuine one is kept


async def test_a_state_text_that_does_not_match_at_all_keeps_the_answer_spanless() -> (
    None
):
    def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
        if "fields" in payload:
            found = {
                "name": clairic_answer("text", "Dana Ruiz", span=[3, 12]),
                "city": clairic_answer("text", "Lyon", span=[40, 44]),
            }
            return clairic_response(found, state_text="something unrelated")
        return clef_response(
            {q: noul(0.95) for q in payload["questions"]}, model=payload["model"]
        )

    extractor = create_extractor([Contact], transport=FakeTransport(handler))
    result = await extractor.ainvoke(CRLF_TEXT)
    (record,) = result.records
    assert record.valid and record.after == {"name": "Dana Ruiz", "city": "Lyon"}
    assert record.fields["city"].span is None
    assert any("state_text_mismatch" in w for w in result.warnings)


async def test_a_list_element_after_a_partial_re_anchor_is_kept() -> None:
    class Trip(BaseModel):
        """A trip."""

        stops: list[str] = Field(description="places visited")

    text = "Start in Paris.\r\nThen Lyon and finally Nice."

    def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
        seen = payload["state"].replace("\r\n", "\n")
        stops = ["Paris", "Lyon", "Nice"]
        at = [seen.index(s) for s in stops]
        found = {
            "stops": clairic_answer(
                "list",
                stops,
                spans=[[a, a + len(s)] for a, s in zip(at, stops, strict=True)],
                confidence=0.99,
            )
        }
        return clairic_response(found, state_text=seen)

    transport = FakeTransport(
        route(
            default=lambda m, p, meta: clef_response(
                {q: noul(0.95) for q in p["questions"]}, model=p["model"]
            ),
            extract=handler,
        )
    )
    result = await create_extractor([Trip], transport=transport).ainvoke(text)
    (record,) = result.records
    assert record.valid and record.after == {"stops": ["Paris", "Lyon", "Nice"]}
    spans = record.fields["stops"].spans
    assert spans is not None
    assert [text[a:b] for a, b in spans] == ["Paris", "Lyon", "Nice"]


# --------------------------------------------------------------------------- #
# R2: the identity guard and a rename the cluster links
# --------------------------------------------------------------------------- #


class Person(BaseModel):
    """A person."""

    name: str = Field(description="full name")
    city: Optional[str] = Field(None, description="the city they live in")


async def test_a_married_name_is_not_an_identity_conflict() -> None:
    text = (
        "Sarah Johnson got married last spring and is now Sarah Miller. "
        "She moved to Leeds."
    )
    world = FakeWorld(text)
    world.entity(
        "Sarah Miller",
        schema=Person,
        mentions=["Sarah Johnson", "Sarah Miller"],
        fields={"name": "Sarah Miller", "city": "Leeds"},
        matches={"name": "Sarah Johnson"},
    )
    existing = [("s", "Person", {"name": "Sarah Johnson", "city": "York"})]
    extractor = create_extractor([Many(Person)], transport=world.transport())
    result = await extractor.ainvoke(text, existing=existing)
    (record,) = result.records
    assert [(p.op, p.path, p.value) for p in record.patches] == [
        ("replace", "/name", "Sarah Miller"),
        ("replace", "/city", "Leeds"),
    ]
    assert record.after == {"name": "Sarah Miller", "city": "Leeds"}
    assert not [r for r in record.rejected if r.reason == "identity_conflict"]
    assert result.skipped == ()


async def test_another_name_with_no_compatible_mention_is_still_refused() -> None:
    text = "Emma Watson moved to Lyon."
    world = FakeWorld(text)
    world.entity(
        "Emma Watson",
        schema=Person,
        mentions=["Emma Watson"],
        fields={"name": "Emma Watson", "city": "Lyon"},
        matches={"name": "Emma Thompson"},
    )
    existing = [("a", "Person", {"name": "Emma Thompson", "city": "Paris"})]
    extractor = create_extractor([Many(Person)], transport=world.transport())
    result = await extractor.ainvoke(text, existing=existing)
    (record,) = result.records
    assert record.patches == ()
    assert {r.reason for r in record.rejected} == {"identity_conflict"}
    assert [s.reason for s in result.skipped] == ["ambiguous_match"]


# --------------------------------------------------------------------------- #
# L1: Clairic's list value and spans are not index-aligned
# --------------------------------------------------------------------------- #

LIVE_TEXT = "Emma met Sarah today. Later Sarah left. Olivia Davis arrived."


def live_answer(text: str = LIVE_TEXT) -> tuple[Any, ClairicAnswer]:
    """The live shape: four spans (Sarah twice) for three values."""
    state = render_clairic(text, label="Person", focus=("<focus>", ("x",)))
    base = state.input_span[0]
    pieces = [
        (0, "Emma"),
        (text.index("Sarah"), "Sarah"),
        (text.index("Sarah", text.index("Later")), "Sarah"),
        (text.index("Olivia Davis"), "Olivia Davis"),
    ]
    spans = [[base + at, base + at + len(t)] for at, t in pieces]
    answer = ClairicAnswer(
        type="list", value=["Emma", "Sarah", "Olivia Davis"], spans=spans
    )
    return state, answer


def test_misaligned_values_and_spans_give_the_right_mentions() -> None:
    state, answer = live_answer()
    run = Run(transport=FakeTransport(lambda *a: None))
    mentions = parse_mentions(LIVE_TEXT, answer, state, run)
    assert [(m.text, m.span) for m in mentions] == [
        ("Emma", (0, 4)),
        ("Sarah", (9, 14)),
        ("Sarah", (28, 33)),  # the second Sarah is found, though no value names it
        ("Olivia Davis", (40, 52)),
    ]
    assert run.warnings == []  # no false mention_not_in_input


def test_the_spans_of_a_list_answer_are_re_paired_by_text() -> None:
    state, answer = live_answer()
    response = ClairicResponse(
        model="clairic",
        answers={"mentions": answer},
        truncated=False,
        usage={"input_tokens": 1, "output_tokens": 0},
    )
    fixed = align_list_spans(response, state).answers["mentions"]
    assert fixed.spans is not None and len(fixed.spans) == 3
    assert [state.text[a:b] for a, b in fixed.spans] == [
        "Emma",
        "Sarah",
        "Olivia Davis",
    ]
    assert fixed._raw_spans == answer.spans  # what the model said is kept aside
    # an answer that already lines up is not touched
    again = align_list_spans(
        response.model_copy(update={"answers": {"mentions": fixed}}), state
    )
    assert again.answers["mentions"] is fixed


def test_a_value_with_no_covering_span_takes_every_whole_word_occurrence() -> None:
    text = "Ann met Bob. Bob left. Bobby stayed."
    state = render_clairic(text, label="Person", focus=("<focus>", ("x",)))
    base = state.input_span[0]
    answer = ClairicAnswer(
        type="list", value=["Ann", "Bob"], spans=[[base, base + 3]]
    )  # no span for Bob
    mentions = parse_mentions(text, answer, state)
    assert [(m.text, m.span) for m in mentions] == [
        ("Ann", (0, 3)),
        ("Bob", (8, 11)),
        ("Bob", (13, 16)),  # not the "Bob" inside "Bobby"
    ]


async def test_discovery_with_misaligned_spans_builds_no_duplicate_entity() -> None:
    world = FakeWorld(LIVE_TEXT)
    for name, mentions in (
        ("Emma", ["Emma"]),
        ("Sarah", ["Sarah"]),
        ("Olivia Davis", ["Olivia Davis"]),
    ):
        world.entity(
            name, schema=Person, mentions=mentions, fields={"name": name, "city": None}
        )

    def discover(model: str, payload: dict[str, Any], meta: Any) -> Any:
        state = payload["state"]
        base = state.index("<input>\n") + len("<input>\n")
        spans = []
        for needle, nth in (
            ("Emma", 0),
            ("Sarah", 0),
            ("Sarah", 1),
            ("Olivia Davis", 0),
        ):
            at = base - 1
            for _ in range(nth + 1):
                at = state.index(needle, at + 1)
            spans.append([at, at + len(needle)])
        return clairic_response(
            {
                "mentions": clairic_answer(
                    "list",
                    ["Emma", "Sarah", "Olivia Davis"],
                    spans=spans,
                    confidence=0.9,
                )
            }
        )

    transport = FakeTransport(route(default=world.handler, discover=discover))
    extractor = create_extractor([Many(Person)], transport=transport)
    result = await extractor.ainvoke(LIVE_TEXT)
    assert [r.entity.label for r in result.records if r.entity] == [
        "Emma",
        "Sarah",
        "Olivia Davis",
    ]
    assert not any("mention_not_in_input" in w for w in result.warnings)
    sarah = result.records[1].entity
    assert sarah is not None and [m.span for m in sarah.mentions] == [(9, 14), (28, 33)]


# --------------------------------------------------------------------------- #
# L2: a required list the input says nothing about
# --------------------------------------------------------------------------- #


class Tagged(BaseModel):
    """A thing with tags."""

    name: str = Field(description="name")
    tags: list[str] = Field(description="tags")


async def test_a_required_list_that_is_null_is_assembled_empty() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Widget", find="Widget"),
        "tags": ca("list", None),
    }
    result = await make_extractor([Tagged], script).ainvoke("Widget is here.")
    (record,) = result.records
    assert record.valid and record.after == {"name": "Widget", "tags": []}


async def test_a_required_list_with_every_element_rejected_is_empty() -> None:
    script = Script()
    script.clairic = {
        "name": ca("text", "Widget", find="Widget"),
        "tags": ca("list", ["red"], find_all=["red"]),  # not in the input
    }
    result = await make_extractor([Tagged], script).ainvoke("Widget is here.")
    (record,) = result.records
    assert record.valid and record.after is not None
    assert record.after["tags"] == []


async def test_an_optional_list_keeps_its_default_and_an_update_makes_no_op() -> None:
    class WithDefault(BaseModel):
        """A thing."""

        name: str = Field(description="name")
        tags: list[str] = Field(["x"], description="tags")

    script = Script()
    script.clairic = {
        "name": ca("text", "Widget", find="Widget"),
        "tags": ca("list", None),
    }
    result = await make_extractor([WithDefault], script).ainvoke("Widget is here.")
    assert result.records[0].after == {"name": "Widget", "tags": ["x"]}

    update = Script()
    update.clairic = {
        "name": ca("text", "Widget", find="Widget"),
        "tags": ca("list", None),
    }
    existing = {"Tagged": {"name": "Widget", "tags": ["a"]}}
    done = await make_extractor([Tagged], update).ainvoke(
        "Widget is here.", existing=existing
    )
    (record,) = done.records
    assert record.patches == () and record.after == {"name": "Widget", "tags": ["a"]}
