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
    line = (
        f"The entity is mentioned in the input at characters {spans}.\n"
        if spans
        else ""
    )
    return (
        f'<focus schema="Person" entity="{label}">\nAnswer each field for the '
        f"entity named above only.\n{line}</focus>\n<input>\n{text}\n</input>"
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
    options = {"c0": "Emma Thompson", "c1": "Michael Chen", "__new__": "none of them"}
    # one request assigns a batch of mentions; the clusters are described once
    state = {
        "input": TEXT,
        "mentions": {"m2": mention_entry("Emma"), "m3": mention_entry("Sunny")},
        "clusters": {
            "c0": {
                "label": "Emma Thompson",
                "mentions": [
                    {
                        k: v
                        for k, v in mention_entry("Emma Thompson").items()
                        if k != "context"
                    }
                ],
            },
            "c1": {
                "label": "Michael Chen",
                "mentions": [
                    {
                        k: v
                        for k, v in mention_entry("Michael Chen").items()
                        if k != "context"
                    }
                ],
            },
        },
    }
    answers = await ask(
        world,
        "coref",
        state=state,
        questions={
            "assign.m2": question("which?", "choice", criteria=options),
            "assign.m3": question("which?", "choice", criteria=options),
        },
    )
    assert answers["assign.m2"]["choice"] == "c0"
    # a mention of nobody the options hold starts a new cluster
    assert answers["assign.m3"]["choice"] == "__new__"


async def test_a_merge_check_looks_at_every_mention_of_both_sides() -> None:
    world = people()
    same = {
        "input": TEXT,
        "a": {"mentions": [mention_entry("Emma Thompson")]},
        "b": {"mentions": [mention_entry("Emma")]},
    }
    answers = await ask(
        world, "coref", state=same, questions={"merge": question("same?")}
    )
    assert answers["merge"]["noul"] == 0.96
    # the first mention of each side is one person; a later one is not
    mixed = {
        "input": TEXT,
        "a": {"mentions": [mention_entry("Emma Thompson")]},
        "b": {"mentions": [mention_entry("Emma"), mention_entry("Michael Chen")]},
    }
    answers = await ask(
        world, "coref", state=mixed, questions={"merge": question("same?")}
    )
    assert answers["merge"]["noul"] == 0.03


async def test_qualification_answers_every_entity_of_one_request() -> None:
    world = people()
    emma, sunny = mention_entry("Emma Thompson")["span"], mention_entry("Sunny")["span"]
    state = {
        "input": TEXT,
        "entities": {
            "e0": {
                "label": "Emma Thompson",
                "mentions": [{"text": "Emma Thompson", "span": emma}],
            },
            "e1": {"label": "Sunny", "mentions": [{"text": "Sunny", "span": sunny}]},
        },
    }
    answers = await ask(
        world,
        "qualify",
        state=state,
        questions={"qualify.e0": question("is it?"), "qualify.e1": question("is it?")},
    )
    assert answers["qualify.e0"]["noul"] == 0.97  # an entity
    assert answers["qualify.e1"]["noul"] == 0.03  # a non-entity


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
        ("0-4", "sings"),
        ("12-16", "dances"),
    ):
        state = focus("Emma", text, spans)
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


def match_world() -> FakeWorld:
    world = FakeWorld(TEXT)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={"name": "Emma Thompson"},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={"name": "Michael Chen"},
        matches="chen",  # a string that occurs in the record's JSON text
    )
    world.entity(
        "Newcomer",
        schema=Person,
        mentions=["Sunny"],
        fields={"name": "Newcomer"},
    )
    return world


def match_state(label: str, start: int, end: int) -> Body:
    return {
        "input": TEXT,
        "entity": {"label": label, "mentions": [{"text": label, "span": [start, end]}]},
    }


RECORDS = {
    "rec_0": {"name": "Michael Chen", "notes": []},
    "rec_1": {"name": "Emma Thompson", "notes": ["x"]},
    "rec_2": {"name": "Zed", "notes": []},
    "__new__": "A Person not among the listed records",
}


async def test_a_match_picks_the_record_the_entity_declares() -> None:
    world = match_world()
    questions = {"match": question("?", "choice", criteria=RECORDS)}
    start = TEXT.index("Emma Thompson")
    emma = await ask(
        world,
        "match",
        state=match_state("Emma Thompson", start, start + 13),
        questions=questions,
    )
    assert emma["match"]["choice"] == "rec_1"
    assert emma["match"]["probabilities"]["rec_1"] == 0.92
    assert emma["match"]["probabilities"]["rec_0"] == pytest.approx(0.08 / 3)
    start = TEXT.index("Michael Chen")
    michael = await ask(
        world,
        "match",
        state=match_state("Michael Chen", start, start + 12),
        questions=questions,
    )
    assert michael["match"]["choice"] == "rec_0"


async def test_an_entity_without_matches_is_new() -> None:
    world = match_world()
    start = TEXT.index("Sunny")
    new = await ask(
        world,
        "match",
        state=match_state("Newcomer", start, start + 5),
        questions={"match": question("?", "choice", criteria=RECORDS)},
    )
    assert new["match"]["choice"] == "__new__"
    assert new["match"]["probabilities"]["__new__"] == 0.92


async def test_a_declared_match_that_fits_no_offered_record_is_an_error() -> None:
    world = match_world()
    start = TEXT.index("Emma Thompson")
    offered = {k: v for k, v in RECORDS.items() if k != "rec_1"}
    with pytest.raises(AssertionError, match="fits 0 of the 2 records offered"):
        await ask(
            world,
            "match",
            state=match_state("Emma Thompson", start, start + 13),
            questions={"match": question("?", "choice", criteria=offered)},
        )


async def test_a_declared_match_that_fits_two_records_is_an_error() -> None:
    world = match_world()
    start = TEXT.index("Michael Chen")
    two = {**RECORDS, "rec_3": {"name": "Michael Chen", "notes": ["again"]}}
    with pytest.raises(AssertionError, match="fits 2 of the 4 records offered"):
        await ask(
            world,
            "match",
            state=match_state("Michael Chen", start, start + 12),
            questions={"match": question("?", "choice", criteria=two)},
        )


async def test_a_match_state_that_names_another_entitys_span_is_an_error() -> None:
    world = match_world()
    start = TEXT.index("Michael Chen")
    with pytest.raises(AssertionError, match="belong to"):
        await ask(
            world,
            "match",
            state=match_state("Emma Thompson", start, start + 12),
            questions={"match": question("?", "choice", criteria=RECORDS)},
        )


async def test_a_model_or_dict_matches_by_the_keys_it_names() -> None:
    world = FakeWorld(TEXT)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
        matches=Person(name="Emma Thompson", notes=["x"]),
    )
    start = TEXT.index("Emma Thompson")
    full = {
        "rec_0": {"name": "Emma Thompson", "relationship": None, "notes": []},
        "rec_1": {"name": "Emma Thompson", "relationship": None, "notes": ["x"]},
        "__new__": "none",
    }
    answers = await ask(
        world,
        "match",
        state=match_state("Emma Thompson", start, start + 13),
        questions={"match": question("?", "choice", criteria=full)},
    )
    assert answers["match"]["choice"] == "rec_1"


async def delete_asks(world: FakeWorld, ids: str, count: int) -> Body:
    transport = world.transport()
    response = await transport.run(
        CLEF,
        {
            "model": "clef",
            "state": {
                "input": TEXT,
                "records": {f"r{k}": {"name": "x"} for k in range(count)},
            },
            "questions": {f"delete.r{k}": question("?") for k in range(count)},
        },
        meta=CallMeta(kind="delete", schema="Person", field=ids),
    )
    answers: Body = response.body["answers"]
    return answers


async def test_a_delete_is_unlikely_unless_the_world_declares_it() -> None:
    world = people()
    world.delete("r2", 0.93).delete("r3")
    answers = await delete_asks(world, "r1,r2,r3", 3)
    assert [answers[f"delete.r{k}"]["noul"] for k in range(3)] == [0.05, 0.93, 0.97]


async def test_a_delete_request_must_name_one_record_id_per_question() -> None:
    with pytest.raises(AssertionError, match="record id"):
        await delete_asks(people(), "r1", 2)


async def test_declared_match_odds_pick_the_likeliest_and_share_the_rest() -> None:
    world = FakeWorld(TEXT)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
        match_odds={"Emma Thompson": 0.55, "__new__": 0.3},
        match_confidence=0.4,
    )
    start = TEXT.index("Emma Thompson")
    answers = await ask(
        world,
        "match",
        state=match_state("Emma Thompson", start, start + 13),
        questions={"match": question("?", "choice", criteria=RECORDS)},
    )
    match = answers["match"]
    assert match["choice"] == "rec_1"
    assert match["probabilities"]["rec_1"] == 0.55
    assert match["probabilities"]["__new__"] == 0.3
    assert match["probabilities"]["rec_0"] == pytest.approx(0.075)
    assert match["probabilities"]["rec_2"] == pytest.approx(0.075)
    assert match["confidence"] == 0.4


async def test_declared_match_odds_must_name_exactly_one_record() -> None:
    world = FakeWorld(TEXT)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
        match_odds={"nobody": 0.5},
    )
    start = TEXT.index("Emma Thompson")
    with pytest.raises(AssertionError, match=r"match_odds key 'nobody' .* fits 0"):
        await ask(
            world,
            "match",
            state=match_state("Emma Thompson", start, start + 13),
            questions={"match": question("?", "choice", criteria=RECORDS)},
        )


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
        meta=CallMeta(kind="null_check", schema="Person"),
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


# --------------------------------------------------------------------------- #
# The world is strict about the requests it is sent
# --------------------------------------------------------------------------- #


async def test_the_world_resolves_the_label_first_and_refuses_a_mismatch() -> None:
    world = people()
    fields = {"name": {"type": "text"}}
    # the label and the spans agree: fine
    emma = TEXT.index("Emma Thompson")
    state = focus("Emma Thompson", TEXT, f"{emma}-{emma + 13}")
    answers = await ask(world, "extract", clairic=True, state=state, fields=fields)
    assert answers["name"]["value"] == "Emma Thompson"
    # the label names Emma, the spans are Michael's: that is a bug in the request
    michael = TEXT.index("Michael Chen")
    state = focus("Emma Thompson", TEXT, f"{michael}-{michael + 12}")
    with pytest.raises(AssertionError, match="belong to 'Michael Chen'"):
        await ask(world, "extract", clairic=True, state=state, fields=fields)


async def test_two_entities_with_one_label_need_spans_to_be_told_apart() -> None:
    text = "Emma sings. Emma dances."
    world = FakeWorld(text)
    world.entity(
        "Emma", schema=Person, mentions=[("Emma", 0)], fields={"name": "sings"}
    )
    world.entity(
        "Emma", schema=Person, mentions=[("Emma", 1)], fields={"name": "dances"}
    )
    with pytest.raises(AssertionError, match="2 entities are called 'Emma'"):
        await ask(
            world,
            "extract",
            clairic=True,
            state=focus("Emma", text),
            fields={"name": {"type": "text"}},
        )


async def test_a_clef_state_without_the_entitys_mentions_is_refused() -> None:
    world = people()
    state = {
        "input": "Nothing about anybody here.",
        "entity": {
            "label": "Emma Thompson",
            "mentions": [{"text": "Emma Thompson", "span": [10, 23]}],
        },
    }
    with pytest.raises(AssertionError, match="narrowed away from the entity"):
        await ask(
            world,
            "null_check",
            state=state,
            questions={"nullcheck.name": question("?")},
        )
    # switched off, the world answers anyway
    lax = FakeWorld(TEXT, check_state=False)
    lax.entity(
        "Emma Thompson", schema=Person, mentions=["Emma Thompson"], fields={"name": "x"}
    )
    answers = await ask(
        lax, "null_check", state=state, questions={"nullcheck.name": question("?")}
    )
    assert answers["nullcheck.name"]["noul"] == 0.95


async def test_a_narrowed_state_finds_its_entity_by_the_context() -> None:
    text = "Emma sings. " + "x " * 50 + "Emma dances."
    world = FakeWorld(text)
    world.entity(
        "Emma", schema=Person, mentions=[("Emma", 0)], fields={"name": "sings"}
    )
    world.entity(
        "Emma", schema=Person, mentions=[("Emma", 1)], fields={"name": "dances"}
    )
    late = text.rindex("Emma")
    state = {
        "input": "…\nEmma dances.",
        "entity": {
            "label": "Emma",
            "mentions": [{"text": "Emma", "context": text[late - 20 :]}],  # no span
        },
    }
    answers = await ask(
        world, "null_check", state=state, questions={"nullcheck.name": question("?")}
    )
    assert answers["nullcheck.name"]["noul"] == 0.95  # resolved, not ambiguous


async def test_discovery_noise_makes_the_answers_messy_and_spans_still_map() -> None:
    text = "We met the Smiths. Sarah's kids waved. She left. Bob came."
    world = FakeWorld(text)
    world.entity("Smiths", schema=Person, mentions=["Smiths"])
    world.entity("Sarah", schema=Person, mentions=["Sarah"])
    world.entity("Bob", schema=Person, mentions=["Bob"])
    state = render_clairic(text, label="Person").text

    async def listed() -> Body:
        answers = await ask(
            world, "discover", clairic=True, state=state, fields=DISCOVER
        )
        mentions: Body = answers["mentions"]
        return mentions

    assert (await listed())["value"] == ["Smiths", "Sarah", "Bob"]
    world.noise(determiners=True)
    assert (await listed())["value"] == ["the Smiths", "Sarah", "Bob"]
    world.noise(possessives=True)
    assert (await listed())["value"] == ["Smiths", "Sarah's", "Bob"]
    world.noise(pronouns=True)
    assert (await listed())["value"] == ["Smiths", "Sarah", "She", "Bob"]
    world.noise(shuffle=True)
    assert (await listed())["value"] == ["Bob", "Sarah", "Smiths"]
    world.noise(missing_span=True)
    answer = await listed()
    assert answer["spans"][1] == [] and len(answer["spans"]) == 3
    assert answer["spans"][0] != []
    world.noise()  # all off again
    assert (await listed())["value"] == ["Smiths", "Sarah", "Bob"]
