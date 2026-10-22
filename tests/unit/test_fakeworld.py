"""``FakeWorld``: one declaration of text and entities answers every call kind."""

# ruff: noqa: UP007, UP045 - the schemas deliberately spell Optional and Union

import enum
from datetime import date
from typing import Any, Literal, Optional, Union, cast

import pytest
from pydantic import BaseModel

from trustcall_system_one import CallKind, CallMeta, Many, One, create_extractor
from trustcall_system_one._render import render_clairic
from trustcall_system_one._wire import CLAIRIC, CLEF
from trustcall_system_one.testing import (
    FakeTransport,
    FakeWorld,
    WorldEntity,
    clef_response,
    noul,
    route,
)

Body = dict[str, Any]


class Person(BaseModel):
    """Someone the speaker knows."""

    name: str
    relationship: Optional[str] = None
    notes: list[str] = []


TEXT = (
    "My friend Emma Thompson climbs rock walls. Later Emma met Michael Chen, "
    "who plays guitar. Sunny barked."
)


def people(text: str = TEXT) -> FakeWorld:
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={"name": "Emma Thompson", "notes": ["climbs rock walls"]},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={"name": "Michael Chen", "notes": ["plays guitar"]},
    )
    world.non_entity("Sunny", schema=Person, mentions=["Sunny"])
    return world


async def ask(
    world: FakeWorld,
    kind: str,
    *,
    clairic: bool = False,
    state: Any = None,
    schema: str = "Person",
    **body: Any,
) -> Body:
    """Send one hand-made request to the world's transport; return its answers."""
    transport = world.transport()
    if clairic:
        payload = {"model": "clairic", "state": state, **body}
        model = CLAIRIC
    else:
        payload = {"model": "clef", "state": state, **body}
        model = CLEF
    response = await transport.run(
        model, payload, meta=CallMeta(kind=cast(CallKind, kind), schema=schema)
    )
    answers: Body = response.body["answers"]
    return answers


def question(text: str, kind: str = "noul", **extra: Any) -> Body:
    return {"type": kind, "instructions": text, **extra}


def focus(label: str, text: str, spans: str = "") -> str:
    line = f"The input mentions {label} as: {spans}.\n" if spans else ""
    return (
        f'<focus schema="Person" entity="{label}">\nAnswer each field for {label} '
        f"only.\n{line}</focus>\n<input>\n{text}\n</input>"
    )


# --------------------------------------------------------------------------- #
# Declaring
# --------------------------------------------------------------------------- #


def test_a_mention_that_is_not_in_the_text_is_refused() -> None:
    world = FakeWorld("Hello Ann.")
    with pytest.raises(ValueError, match="not in the text"):
        world.entity("Bob", schema=Person, mentions=["Bob"])
    with pytest.raises(ValueError, match="not in the text"):
        world.non_entity("Cat", schema=Person, mentions=["Cat"])


def test_entity_returns_the_declared_entity() -> None:
    world = FakeWorld("Hello Ann.")
    ann = world.entity(
        "Ann", schema=Person, mentions=["Ann"], fields={"name": "Ann"}, matches="r1"
    )
    assert isinstance(ann, WorldEntity)
    assert (ann.label, ann.schema, ann.real, ann.matches) == (
        "Ann",
        "Person",
        True,
        "r1",
    )
    assert ann.flat == {"name": "Ann"}


def test_the_schema_may_be_given_by_name() -> None:
    world = FakeWorld("Hello Ann.")
    assert world.entity("Ann", schema="Person", mentions=["Ann"]).schema == "Person"


# --------------------------------------------------------------------------- #
# Discovery: spans are found in the rendered state and map back to the text
# --------------------------------------------------------------------------- #

DISCOVER = {"mentions": {"type": "list", "items": "text"}}


async def test_discovery_lists_every_occurrence_with_spans_in_the_state() -> None:
    world = people()
    rendered = render_clairic(TEXT, label="Person")
    answers = await ask(
        world,
        "discover",
        clairic=True,
        state=rendered.text,
        fields=DISCOVER,
    )
    mentions = answers["mentions"]
    # "Emma" inside "Emma Thompson" is not a separate mention; Sunny is listed
    assert mentions["value"] == [
        "Emma Thompson",
        "Emma",
        "Michael Chen",
        "Sunny",
    ]
    for text, (start, end) in zip(mentions["value"], mentions["spans"], strict=True):
        assert rendered.text[start:end] == text
        original = rendered.to_original((start, end))
        assert original is not None and TEXT[original[0] : original[1]] == text


async def test_discovery_of_a_narrowed_state_lists_only_what_it_contains() -> None:
    text = "Emma Thompson is here. " + "x " * 400 + "Michael Chen is there."
    world = FakeWorld(text)
    world.entity("Emma Thompson", schema=Person, mentions=["Emma Thompson"])
    world.entity("Michael Chen", schema=Person, mentions=["Michael Chen"])
    start = text.index("Michael")
    rendered = render_clairic(text, label="Person", windows=[(start - 10, len(text))])
    answers = await ask(
        world, "discover", clairic=True, state=rendered.text, fields=DISCOVER
    )
    (span,) = answers["mentions"]["spans"]
    assert answers["mentions"]["value"] == ["Michael Chen"]  # nothing else
    assert rendered.text[span[0] : span[1]] == "Michael Chen"


async def test_a_plain_string_mention_covers_every_occurrence_and_a_pair_one() -> None:
    text = "Emma sings. Emma dances."
    world = FakeWorld(text)
    world.entity("A", schema=Person, mentions=[("Emma", 0)])
    world.entity("B", schema=Person, mentions=[("Emma", 1)])
    answers = await ask(
        world,
        "discover",
        clairic=True,
        state=render_clairic(text, label="Person").text,
        fields=DISCOVER,
    )
    assert answers["mentions"]["value"] == ["Emma", "Emma"]


async def test_the_broadened_discovery_also_lists_descriptive_mentions() -> None:
    text = "Emma Thompson came with my cousin."
    world = FakeWorld(text)
    world.entity("Emma Thompson", schema=Person, mentions=["Emma Thompson"])
    world.entity("cousin", schema=Person, mentions=[], descriptive=["my cousin"])
    state = render_clairic(text, label="Person").text
    plain = await ask(world, "discover", clairic=True, state=state, fields=DISCOVER)
    wide = await ask(
        world,
        "discover",
        clairic=True,
        state=state,
        fields={
            "mentions": {
                "type": "list",
                "items": "text",
                "instructions": "Also include descriptive references such as 'my "
                "cousin' when no name is given.",
            }
        },
    )
    assert plain["mentions"]["value"] == ["Emma Thompson"]
    assert wide["mentions"]["value"] == ["Emma Thompson", "my cousin"]


async def test_a_state_that_is_not_the_worlds_text_is_an_error() -> None:
    world = people()
    state = render_clairic("some other text", label="Person").text
    with pytest.raises(AssertionError, match="not the input"):
        await ask(world, "discover", clairic=True, state=state, fields=DISCOVER)


async def test_the_count_is_the_number_of_entities_unless_overridden() -> None:
    world = people()
    state = render_clairic(TEXT, label="Person").text
    fields = {"count": {"type": "number", "integer": True}}
    answers = await ask(world, "count", clairic=True, state=state, fields=fields)
    assert answers["count"]["value"] == 2  # Sunny is not an entity
    world.count(Person, 5)
    answers = await ask(world, "count", clairic=True, state=state, fields=fields)
    assert answers["count"]["value"] == 5


# --------------------------------------------------------------------------- #
# Coreference, qualification, presence
# --------------------------------------------------------------------------- #


def mention_entry(text: str, which: int = 0) -> Body:
    at = -1
    for _ in range(which + 1):
        at = TEXT.index(text, at + 1)
    return {"text": text, "span": [at, at + len(text)], "context": TEXT}


async def test_coref_pairs_follow_the_owners_of_the_mentions() -> None:
    world = people()
    state = {
        "input": TEXT,
        "mentions": {
            "m0": mention_entry("Emma Thompson"),
            "m1": mention_entry("Emma"),
            "m2": mention_entry("Michael Chen"),
        },
    }
    qs = {
        "pair.m0.m1": question("same?"),
        "pair.m0.m2": question("same?"),
        "pair.m1.m2": question("same?"),
    }
    answers = await ask(world, "coref", state=state, questions=qs)
    assert answers["pair.m0.m1"]["noul"] == 0.96
    assert answers["pair.m0.m2"]["noul"] == 0.03
    assert answers["pair.m1.m2"]["noul"] == 0.03


async def test_a_coref_fault_changes_one_pair_in_either_order() -> None:
    world = people().coref("Emma", "Emma Thompson", 0.3)
    state = {
        "input": TEXT,
        "mentions": {"m0": mention_entry("Emma Thompson"), "m1": mention_entry("Emma")},
    }
    answers = await ask(
        world, "coref", state=state, questions={"pair.m0.m1": question("same?")}
    )
    assert answers["pair.m0.m1"]["noul"] == 0.3


async def test_an_assignment_picks_the_cluster_of_the_same_entity() -> None:
    world = people()
    first = mention_entry("Emma Thompson")
    options = {
        "c0": f'Emma Thompson: mentioned as "Emma Thompson" ({first["span"][0]}-'
        f"{first['span'][1]})",
        "c1": 'Michael Chen: mentioned as "Michael Chen" (58-70)',
        "__new__": "none of them",
    }
    state = {"input": TEXT, "mention": mention_entry("Emma")}
    answers = await ask(
        world,
        "coref",
        state=state,
        questions={"assign.m2": question("which?", "choice", criteria=options)},
    )
    assert answers["assign.m2"]["choice"] == "c0"
    # a mention of nobody the options hold starts a new cluster
    state = {"input": TEXT, "mention": mention_entry("Sunny")}
    answers = await ask(
        world,
        "coref",
        state=state,
        questions={"assign.m3": question("which?", "choice", criteria=options)},
    )
    assert answers["assign.m3"]["choice"] == "__new__"


async def test_a_merge_check_compares_the_owners_of_the_two_sides() -> None:
    world = people()
    state = {
        "input": TEXT,
        "a": {"mentions": [mention_entry("Emma Thompson")]},
        "b": {"mentions": [mention_entry("Emma")]},
    }
    answers = await ask(
        world, "coref", state=state, questions={"merge": question("same?")}
    )
    assert answers["merge"]["noul"] == 0.96


async def test_qualification_says_yes_to_an_entity_and_no_to_a_non_entity() -> None:
    world = people()
    for label, span, p in (
        ("Emma Thompson", mention_entry("Emma Thompson")["span"], 0.97),
        ("Sunny", mention_entry("Sunny")["span"], 0.03),
    ):
        state = {
            "input": TEXT,
            "entity": {"label": label, "mentions": [{"text": label, "span": span}]},
        }
        answers = await ask(
            world, "qualify", state=state, questions={"qualify": question("is it?")}
        )
        assert answers["qualify"]["noul"] == p


async def test_presence_follows_whether_the_schema_has_an_entity() -> None:
    world = people()
    answers = await ask(
        world, "presence", state={"input": TEXT}, questions={"presence": question("?")}
    )
    assert answers["presence"]["noul"] == 0.97
    empty = FakeWorld("Nothing here.")
    answers = await ask(
        empty, "presence", state={"input": "x"}, questions={"presence": question("?")}
    )
    assert answers["presence"]["noul"] == 0.03


# --------------------------------------------------------------------------- #
# Extraction answers
# --------------------------------------------------------------------------- #


async def test_extraction_answers_carry_spans_nearest_the_entitys_mentions() -> None:
    text = "Tea for Emma. Michael wants tea."
    world = FakeWorld(text)
    world.entity(
        "Emma",
        schema=Person,
        mentions=["Emma"],
        fields={"name": "Emma", "relationship": "tea"},
    )
    world.entity(
        "Michael",
        schema=Person,
        mentions=["Michael"],
        fields={"name": "Michael", "relationship": "tea"},
    )
    fields = {"relationship": {"type": "text"}}
    for label, wanted in (("Emma", text.index("tea")), ("Michael", text.rindex("tea"))):
        state = focus(label, text)
        answers = await ask(world, "extract", clairic=True, state=state, fields=fields)
        start, end = answers["relationship"]["span"]
        assert state[start:end] == "tea"
        assert start - state.index("<input>\n") - len("<input>\n") == wanted


async def test_a_missing_field_is_a_confident_null() -> None:
    world = people()
    answers = await ask(
        world,
        "extract",
        clairic=True,
        state=focus("Emma Thompson", TEXT),
        fields={"relationship": {"type": "text"}},
    )
    assert answers["relationship"]["value"] is None
    assert answers["relationship"]["p_null"] >= 0.9


async def test_a_value_that_is_not_in_the_text_is_an_error() -> None:
    world = FakeWorld("Hello Ann.")
    world.entity("Ann", schema=Person, mentions=["Ann"], fields={"name": "Annabel"})
    with pytest.raises(AssertionError, match="Clairic only quotes"):
        await ask(
            world,
            "extract",
            clairic=True,
            state=focus("Ann", "Hello Ann."),
            fields={"name": {"type": "text"}},
        )


async def test_numbers_dates_and_lists_are_answered_by_type() -> None:
    text = "Ann is 36, joined on 2020-05-01 and likes jam and tea."
    world = FakeWorld(text)
    world.entity(
        "Ann",
        schema=Person,
        mentions=["Ann"],
        fields={
            "age": 36.0,
            "joined": date(2020, 5, 1),
            "likes": ["jam", "tea"],
            "scores": [1, 2],
        },
    )
    answers = await ask(
        world,
        "extract",
        clairic=True,
        state=focus("Ann", text),
        fields={
            "age": {"type": "number"},
            "joined": {"type": "date"},
            "likes": {"type": "list", "items": "text"},
            "scores": {"type": "list", "items": "number"},
        },
    )
    assert answers["age"]["value"] == 36
    assert answers["joined"]["value"] == "2020-05-01"
    assert answers["likes"]["value"] == ["jam", "tea"]
    assert len(answers["likes"]["spans"]) == 2
    assert answers["scores"]["value"] == [1, 2]


async def test_nested_paths_and_union_member_wire_names_are_found() -> None:
    text = "Ann lives in Rome and has a dog."
    world = FakeWorld(text)
    world.entity(
        "Ann",
        schema=Person,
        mentions=["Ann"],
        fields={"address": {"city": "Rome"}, "pet": {"kind": "dog", "toy": "dog"}},
    )
    answers = await ask(
        world,
        "extract",
        clairic=True,
        state=focus("Ann", text),
        fields={
            "address.city": {"type": "text"},
            "pet.Dog.toy": {"type": "text"},  # a union member inside the wire name
        },
    )
    assert answers["address.city"]["value"] == "Rome"
    assert answers["pet.Dog.toy"]["value"] == "dog"


async def test_bleed_adds_to_lists_and_replaces_scalars_in_primary_asks_only() -> None:
    world = people()
    world.bleed("Michael Chen", "notes", "climbs rock walls")
    world.bleed("Michael Chen", "relationship", "My friend")
    fields = {
        "notes": {"type": "list", "items": "text"},
        "relationship": {"type": "text"},
    }
    state = focus("Michael Chen", TEXT)
    primary = await ask(world, "extract", clairic=True, state=state, fields=fields)
    assert primary["notes"]["value"] == ["plays guitar", "climbs rock walls"]
    assert primary["relationship"]["value"] == "My friend"
    cross = await ask(world, "crosscheck", clairic=True, state=state, fields=fields)
    assert cross["notes"]["value"] == ["plays guitar", "climbs rock walls"]
    again = await ask(world, "reask", clairic=True, state=state, fields=fields)
    assert again["notes"]["value"] == ["plays guitar"]  # a re-ask is clean
    assert again["relationship"]["value"] is None


async def test_low_confidence_lowers_the_first_answer_only() -> None:
    world = people()
    world.low_confidence("Emma Thompson", "name", 0.4)
    fields = {"name": {"type": "text"}}
    state = focus("Emma Thompson", TEXT)
    first = await ask(world, "extract", clairic=True, state=state, fields=fields)
    second = await ask(world, "reask", clairic=True, state=state, fields=fields)
    assert first["name"]["confidence"] == 0.4
    assert second["name"]["confidence"] >= 0.9


async def test_the_focused_entity_is_found_by_its_mention_spans() -> None:
    text = "Emma sings. Emma dances."
    world = FakeWorld(text)
    world.entity(
        "Emma", schema=Person, mentions=[("Emma", 0)], fields={"name": "sings"}
    )
    world.entity(
        "Emma", schema=Person, mentions=[("Emma", 1)], fields={"name": "dances"}
    )
    fields = {"name": {"type": "text"}}
    for spans, wanted in (
        ("at characters 0-4", "sings"),
        ("at characters 12-16", "dances"),
    ):
        state = focus("Emma", text, f'"Emma" {spans}')
        answers = await ask(world, "extract", clairic=True, state=state, fields=fields)
        assert answers["name"]["value"] == wanted


# --------------------------------------------------------------------------- #
# Clef-typed fields and verification
# --------------------------------------------------------------------------- #


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


async def test_clef_questions_are_answered_from_the_value() -> None:
    world = FakeWorld("Ann is vip.")
    world.entity(
        "Ann",
        schema=Person,
        mentions=["Ann"],
        fields={"vip": True, "tier": "gold", "level": 2, "pet": {"kind": "dog"}},
    )
    state = {
        "input": "Ann is vip.",
        "entity": {"label": "Ann", "mentions": [{"text": "Ann", "span": [0, 3]}]},
    }
    answers = await ask(
        world,
        "clef_fields",
        state=state,
        questions={
            "vip": question("vip?"),
            "tier": question(
                "tier?", "choice", criteria={"gold": "Gold", "silver": "Silver"}
            ),
            "unknown": question(
                "x?", "choice", criteria={"a": "A", "b": "B", "unstated": "no"}
            ),
            "pet": question(
                "pet?", "choice", criteria={"cat": "Cat: ...", "dog": "Dog: ..."}
            ),
            "level": question("level?", "score", criteria=["low", "mid", "high"]),
        },
    )
    assert answers["vip"]["noul"] == 0.95
    assert answers["tier"]["choice"] == "gold"
    assert answers["unknown"]["choice"] == "unstated"
    assert answers["pet"]["choice"] == "dog"
    assert answers["level"]["probabilities"]["2"] == 0.8


async def test_a_false_bool_and_a_missing_field_are_noes() -> None:
    world = FakeWorld("Ann is here.")
    world.entity("Ann", schema=Person, mentions=["Ann"], fields={"vip": False})
    state = {"input": "Ann is here.", "entity": {"label": "Ann", "mentions": []}}
    answers = await ask(
        world,
        "clef_fields",
        state=state,
        questions={"vip": question("?"), "other": question("?")},
    )
    assert answers["vip"]["noul"] == 0.05 and answers["other"]["noul"] == 0.05


async def test_choices_match_enum_names_and_model_instances() -> None:
    class Dog(BaseModel):
        kind: str = "dog"

    world = FakeWorld("Ann is here.")
    world.entity(
        "Ann",
        schema=Person,
        mentions=["Ann"],
        fields={"colour": "red", "pet": Dog(), "age": 36},
    )
    state = {"input": "Ann is here.", "entity": {"label": "Ann", "mentions": []}}
    answers = await ask(
        world,
        "clef_fields",
        state=state,
        questions={
            "colour": question(
                "?",
                "choice",
                criteria={"RED": "Colour.RED (red)", "BLUE": "Colour.BLUE (blue)"},
            ),
            "pet": question("?", "choice", criteria={"Cat": "a cat", "Dog": "a dog"}),
            "age.__type": question(
                "?", "choice", criteria={"str": "a str", "int": "an int"}
            ),
        },
    )
    assert answers["colour"]["choice"] == "RED"
    assert answers["pet"]["choice"] == "Dog"
    assert answers["age.__type"]["choice"] == "int"


async def test_element_choices_pick_the_option_an_element_is() -> None:
    world = FakeWorld("Ann is here.")
    world.entity("Ann", schema=Person, mentions=["Ann"])
    state = {"input": "Ann is here.", "entity": {"label": "Ann", "mentions": []}}
    options = {"a": "alpha", "b": "beta", "none_of_these": "None of the others."}
    answers = await ask(
        world,
        "clef_fields",
        state=state,
        questions={
            "tags.el0": question(
                'Which option is "beta" in the input, for x?',
                "choice",
                criteria=options,
            ),
            "tags.el1": question(
                'Which option is "zeta" in the input, for x?',
                "choice",
                criteria=options,
            ),
        },
    )
    assert answers["tags.el0"]["choice"] == "b"
    assert answers["tags.el1"]["choice"] == "none_of_these"


def attribution_state(label: str) -> Body:
    return {
        "input": TEXT,
        "entity": {"label": label, "mentions": []},
    }


async def test_attribution_is_yes_when_the_value_belongs_to_the_entity() -> None:
    world = people()
    questions = {
        "attr.notes.0": question(
            'Does `input` state this about `entity.label`: "climbs rock walls"?'
        ),
        "attr.name": question(
            'Does `input` state that name of `entity.label` is "Emma Thompson"?'
        ),
    }
    emma = await ask(
        world,
        "attribution",
        state=attribution_state("Emma Thompson"),
        questions=questions,
    )
    assert emma["attr.notes.0"]["noul"] == 0.95 and emma["attr.name"]["noul"] == 0.95
    michael = await ask(
        world,
        "attribution",
        state=attribution_state("Michael Chen"),
        questions=questions,
    )
    assert michael["attr.notes.0"]["noul"] == 0.04
    assert michael["attr.name"]["noul"] == 0.04


async def test_attribution_of_a_value_nobody_has_is_no() -> None:
    world = people()
    answers = await ask(
        world,
        "attribution",
        state=attribution_state("Emma Thompson"),
        questions={
            "attr.notes.0": question(
                'Does `input` state this about `entity.label`: "teaches yoga"?'
            )
        },
    )
    assert answers["attr.notes.0"]["noul"] == 0.04


async def test_null_checks_and_stated_checks_follow_the_field() -> None:
    world = people()
    state = attribution_state("Emma Thompson")
    qs = {"nullcheck.notes": question("?"), "nullcheck.relationship": question("?")}
    answers = await ask(world, "null_check", state=state, questions=qs)
    assert answers["nullcheck.notes"]["noul"] == 0.95
    assert answers["nullcheck.relationship"]["noul"] == 0.05
    stated = await ask(
        world, "stated", state=state, questions={"stated.name": question("?")}
    )
    assert stated["stated.name"]["noul"] == 0.95


async def test_adjudication_picks_the_option_that_is_the_value() -> None:
    world = people()
    options = {"a": "plays drums", "b": "plays guitar", "neither": "neither"}
    response = await world.transport().run(
        CLEF,
        {
            "model": "clef",
            "state": attribution_state("Michael Chen"),
            "questions": {"adjudicate": question("?", "choice", criteria=options)},
        },
        meta=CallMeta(kind="adjudicate", schema="Person", field="notes"),
    )
    assert response.body["answers"]["adjudicate"]["choice"] == "b"
    nobody = {"a": "plays drums", "b": "sings", "neither": "neither"}
    response = await world.transport().run(
        CLEF,
        {
            "model": "clef",
            "state": attribution_state("Michael Chen"),
            "questions": {"adjudicate": question("?", "choice", criteria=nobody)},
        },
        meta=CallMeta(kind="adjudicate", schema="Person", field="notes"),
    )
    assert response.body["answers"]["adjudicate"]["choice"] == "neither"


async def test_list_dedup_is_yes_only_for_the_same_text() -> None:
    world = people()
    qs = {
        "dedup.0": question(
            'Is "Plays Guitar" the same fact as "plays guitar", as far as '
            "`entity.label` goes?"
        ),
        "dedup.1": question(
            'Is "sings" the same fact as "plays guitar", as far as `entity.label` goes?'
        ),
    }
    answers = await ask(
        world, "list_dedup", state=attribution_state("Emma Thompson"), questions=qs
    )
    assert answers["dedup.0"]["noul"] == 0.95 and answers["dedup.1"]["noul"] == 0.05


async def test_the_gate_says_yes_to_what_is_new_compared_to_the_record() -> None:
    world = people()
    record = {"name": "Emma Thompson", "notes": ["climbs rock walls"], "age": 36}

    def state(path: str, to: Any) -> Body:
        return {
            "input": TEXT,
            "entity": {"label": "Emma Thompson", "mentions": []},
            "record": record,
            "proposed_change": {"op": "add", "path": path, "field": path, "to": to},
        }

    async def p(path: str, to: Any) -> float:
        answers = await ask(
            world, "gate", state=state(path, to), questions={"gate": question("?")}
        )
        return float(answers["gate"]["noul"])

    assert await p("/notes/-", "plays chess") == 0.9  # a new element
    assert await p("/notes/-", "Climbs rock walls") == 0.05  # held, folded
    assert await p("/name", "Emma Thompson") == 0.05  # unchanged
    assert await p("/name", "Emma") == 0.9  # a different value
    assert await p("/age", 37) == 0.9
    assert await p("/age", 36) == 0.05
    assert await p("/nickname", "Em") == 0.9  # the record lacks the key


async def test_match_and_delete_are_stubbed_for_m8() -> None:
    world = people()
    state = attribution_state("Emma Thompson")
    options = {"r0": "a record", "__new__": "none"}
    matched = await ask(
        world,
        "match",
        state=state,
        questions={"match": question("?", "choice", criteria=options)},
    )
    assert matched["match"]["choice"] == "__new__"
    deleted = await ask(
        world, "delete", state=state, questions={"delete": question("?")}
    )
    assert deleted["delete"]["noul"] == 0.03


async def test_a_call_kind_the_world_cannot_answer_is_an_error() -> None:
    world = people()
    with pytest.raises(AssertionError, match="no answer for 'raw'"):
        await ask(world, "raw", state={"input": "x"}, questions={"q": question("?")})


async def test_the_handler_can_be_wrapped_to_override_one_kind() -> None:
    world = people()

    def presence(model: str, payload: Body, meta: CallMeta) -> Any:
        return clef_response(
            {qid: noul(0.35) for qid in payload["questions"]}, model=payload["model"]
        )

    transport = FakeTransport(route(default=world.handler, presence=presence))
    response = await transport.run(
        CLEF,
        {
            "model": "clef",
            "state": {"input": TEXT},
            "questions": {"presence": question("?")},
        },
        meta=CallMeta(kind="presence", schema="Person"),
    )
    assert response.body["answers"]["presence"]["noul"] == 0.35
    other = await transport.run(
        CLEF,
        {"model": "clef", "state": {"input": TEXT}, "questions": {"q": question("?")}},
        meta=CallMeta(kind="qualify", schema="Person"),
    )
    assert "q" in other.body["answers"]


def test_transport_is_created_once() -> None:
    world = people()
    assert world.transport() is world.transport()


# --------------------------------------------------------------------------- #
# A One schema end to end: nested groups, a union, Clef-typed fields
# --------------------------------------------------------------------------- #


class Address(BaseModel):
    city: str
    zip: Optional[str] = None


class Cat(BaseModel):
    kind: Literal["cat"]
    lives: int


class Dog(BaseModel):
    kind: Literal["dog"]
    tricks: list[str] = []


class Profile(BaseModel):
    """A customer profile."""

    name: str
    age: int
    joined: date
    vip: bool
    tier: Literal["gold", "silver"]
    address: Address
    shipping: Optional[Address] = None
    pet: Union[Cat, Dog]
    nicknames: list[str] = []


PROFILE_TEXT = (
    "Ada Byron, 36, joined on 2020-05-01 and is a VIP on the gold tier. She lives "
    "in London. Her dog knows roll over and sit. Nicknames: Countess, Enchantress."
)


async def test_a_one_schema_is_answered_end_to_end() -> None:
    world = FakeWorld(PROFILE_TEXT)
    world.entity(
        "the Profile",
        schema=Profile,
        mentions=["Ada Byron"],
        fields={
            "name": "Ada Byron",
            "age": 36,
            "joined": date(2020, 5, 1),
            "vip": True,
            "tier": "gold",
            "address": {"city": "London"},
            "pet": {"kind": "dog", "tricks": ["roll over", "sit"]},
            "nicknames": ["Countess", "Enchantress"],
        },
    )
    extractor = create_extractor([One(Profile)], transport=world.transport())
    result = await extractor.ainvoke(PROFILE_TEXT)
    (record,) = result.records
    assert record.valid, record.errors
    profile = record.value
    assert isinstance(profile, Profile)
    assert profile.name == "Ada Byron" and profile.age == 36
    assert profile.joined == date(2020, 5, 1)
    assert profile.vip is True and profile.tier == "gold"
    assert profile.address.city == "London"
    assert profile.shipping is None  # the group is not in the world: absent
    assert isinstance(profile.pet, Dog) and profile.pet.tricks == ["roll over", "sit"]
    assert profile.nicknames == ["Countess", "Enchantress"]
    assert result.skipped == () and record.rejected == ()


async def test_many_schemas_run_on_the_same_world_as_one_schemas() -> None:
    world = people()
    extractor = create_extractor(
        [Many(Person, identity="name")], transport=world.transport()
    )
    result = await extractor.ainvoke(TEXT)
    assert [r.entity.label for r in result.records if r.entity] == [
        "Emma Thompson",
        "Michael Chen",
    ]
