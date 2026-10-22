"""The collection engine: every entity of a schema in one input, inserted."""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

import asyncio
from collections import Counter
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    CallMeta,
    CorefConfig,
    DiscoveryConfig,
    ExtractionResult,
    Extractor,
    Many,
    One,
    create_extractor,
)
from trustcall_system_one._collection import (
    PRONOUNS,
    Scope,
    clean_mention,
    collect,
    identity_fields,
    parse_mentions,
)
from trustcall_system_one._render import render_clairic
from trustcall_system_one._run import Run
from trustcall_system_one._wire import ClairicAnswer
from trustcall_system_one.testing import (
    FakeTransport,
    FakeWorld,
    clef_response,
    noul,
    route,
)

Body = dict[str, Any]


class Person(BaseModel):
    """Someone the speaker knows or is being introduced to."""

    name: str = Field(description="full name")
    relationship: Optional[str] = Field(None, description="how they relate to me")
    notes: list[str] = Field([], description="things worth remembering")


class Event(BaseModel):
    """A thing that happens at a time."""

    title: str
    when: Optional[str] = None


TEXT = (
    "I ran into Emma Thompson, my college friend, at the park yesterday. "
    "She was walking her new puppy, a golden retriever named Sunny. "
    "Emma got promoted to Senior Marketing Manager last month and has taken up rock "
    "climbing. Then Michael Chen called, my coworker. He is now a Data Scientist and "
    "is thinking of going vegan. My neighbor Sarah is teaching special education. "
    "Sarah's kids are growing up fast. My friend introduced me to her cousin Olivia "
    "Davis, a graphic designer who sketches in her free time."
)


def readme_world() -> FakeWorld:
    """The README scenario's people, without any record of them."""
    world = FakeWorld(TEXT)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={
            "name": "Emma Thompson",
            "relationship": "my college friend",
            "notes": [
                "promoted to Senior Marketing Manager last month",
                "taken up rock climbing",
            ],
        },
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={
            "name": "Michael Chen",
            "relationship": "my coworker",
            "notes": ["a Data Scientist", "thinking of going vegan"],
        },
    )
    world.entity(
        "Sarah",
        schema=Person,
        mentions=["Sarah"],
        fields={
            "name": "Sarah",
            "relationship": "My neighbor",
            "notes": ["teaching special education", "kids are growing up fast"],
        },
    )
    world.entity(
        "Olivia Davis",
        schema=Person,
        mentions=["Olivia Davis"],
        fields={
            "name": "Olivia Davis",
            "relationship": "her cousin",
            "notes": ["a graphic designer", "sketches in her free time"],
        },
    )
    world.non_entity("Sunny", schema=Person, mentions=["Sunny"])
    return world


def extractor_for(world: FakeWorld, *schemas: Any, **options: Any) -> Extractor:
    chosen = schemas or (Many(Person, identity="name"),)
    return create_extractor(list(chosen), transport=world.transport(), **options)


async def invoke(
    world: FakeWorld, text: str | None = None, *schemas: Any, **options: Any
) -> ExtractionResult:
    extractor = extractor_for(world, *schemas, **options)
    return await extractor.ainvoke(text if text is not None else world.text)


def kinds(result: ExtractionResult) -> list[str]:
    return [call.kind for call in result.calls]


def of_kind(world: FakeWorld, kind: str) -> list[tuple[str, Body, CallMeta]]:
    return [c for c in world.transport().calls if c[2].kind == kind]


def labels(result: ExtractionResult) -> list[str]:
    return [r.entity.label for r in result.records if r.entity]


# --------------------------------------------------------------------------- #
# Mentions: pronouns, determiners, spans
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "word",
    [
        "she",
        "She",
        "her",
        "hers",
        "he",
        "him",
        "they",
        "them",
        "I",
        "me",
        "we",
        "someone",
        "who",
        "it",
        "themselves",
        "yourself",
    ],
)
def test_a_pronoun_is_never_a_mention(word: str) -> None:
    assert word.lower() in PRONOUNS
    assert clean_mention(word, (4, 4 + len(word))) is None


def test_a_mention_of_only_pronouns_and_punctuation_is_dropped() -> None:
    assert clean_mention("  she, ") is None
    assert clean_mention("the") is None
    assert clean_mention("") is None
    assert clean_mention("... ") is None


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("the Smiths", "Smiths"),
        ("The Hendersons", "Hendersons"),
        ("a neighbor", "neighbor"),
        ("an old friend", "old friend"),
        ("this Emma", "Emma"),
        ("Emma's", "Emma"),
        ("Emma" + chr(0x2019) + "s", "Emma"),
        ("the cousin's", "cousin"),
        ("Dr. Who", "Dr. Who"),
        ("my cousin", "my cousin"),  # a descriptive reference keeps "my"
        ('"Sarah"', "Sarah"),
    ],
)
def test_determiners_and_possessives_are_stripped(raw: str, cleaned: str) -> None:
    mention = clean_mention(raw)
    assert mention is not None and mention.text == cleaned and mention.span is None


def test_stripping_moves_the_span_with_the_text() -> None:
    source = "We saw the Smiths' dog and Emma's cat."
    for raw, text in (("the Smiths", "Smiths"), ("Emma's", "Emma")):
        start = source.index(raw)
        mention = clean_mention(raw, (start, start + len(raw)))
        assert mention is not None and mention.span is not None
        assert source[mention.span[0] : mention.span[1]] == text


def rendered_with(
    text: str, mention_texts: list[str], **windows: Any
) -> tuple[Any, Any]:
    """A rendered discovery state and an answer whose spans are offsets in it."""
    state = render_clairic(text, label="Person", focus=("<focus>", ("x",)), **windows)
    spans, values = [], []
    for mention in mention_texts:
        at = state.text.index(mention, state.input_span[0])
        spans.append([at, at + len(mention)])
        values.append(mention)
    answer = ClairicAnswer(type="list", value=values, spans=spans)
    return state, answer


def test_spans_are_mapped_back_to_offsets_in_the_original_input() -> None:
    text = "Emma Thompson met the Smiths and Emma's friend. She left."
    state, answer = rendered_with(
        text, ["Emma Thompson", "the Smiths", "Emma's", "She"]
    )
    mentions = parse_mentions(text, answer, state)
    assert [m.text for m in mentions] == ["Emma Thompson", "Smiths", "Emma"]
    for mention in mentions:
        assert mention.span is not None
        assert text[mention.span[0] : mention.span[1]] == mention.text
    assert state.input_span[0] > 0  # the state has a header: offsets differ


def test_spans_are_mapped_through_narrowed_windows() -> None:
    text = "Emma Thompson is here. " + "x " * 300 + "Michael Chen is there."
    start = text.index("Michael")
    state, answer = rendered_with(
        text, ["Michael Chen"], windows=[(start - 5, len(text))]
    )
    (mention,) = parse_mentions(text, answer, state)
    assert mention.span == (start, start + len("Michael Chen"))


def test_an_item_without_a_span_is_found_by_searching_the_input() -> None:
    text = "Ann met Bob. Bob left."
    answer = ClairicAnswer(type="list", value=["Ann", "Bob", "Bob"])
    mentions = parse_mentions(text, answer, None)
    assert [(m.text, m.span) for m in mentions] == [
        ("Ann", (0, 3)),
        ("Bob", (8, 11)),
        ("Bob", (13, 16)),  # each search starts after the previous mention
    ]


def test_an_item_that_is_not_in_the_input_is_dropped_with_a_warning() -> None:
    run = Run(transport=FakeTransport(lambda *a: None))
    answer = ClairicAnswer(type="list", value=["Ann", "Zed"])
    mentions = parse_mentions("Ann is here.", answer, None, run)
    assert [m.text for m in mentions] == ["Ann"]
    assert any("mention_not_in_input" in w for w in run.warnings)


def test_a_wrong_span_is_corrected_from_the_text_next_to_it() -> None:
    text = "Hello Ann Lee, hello Bob."
    answer = ClairicAnswer(type="list", value=["Ann Lee"], spans=[[0, 5]])
    (mention,) = parse_mentions(text, answer, render_clairic(text, label="x"))
    assert mention.span is None or text[mention.span[0] : mention.span[1]] == "Ann Lee"


def test_overlapping_mentions_keep_the_longer_and_order_is_by_position() -> None:
    text = "Emma Thompson, Michael."
    answer = ClairicAnswer(
        type="list",
        value=["Michael", "Emma", "Emma Thompson"],
        spans=[[0, 0], [0, 0], [0, 0]],
    )
    mentions = parse_mentions(text, answer, None)
    assert [m.text for m in mentions] == ["Emma Thompson", "Michael"]


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def test_the_identity_is_the_wrappers_then_the_schemas_own() -> None:
    from trustcall_system_one import compile_schema

    plan = compile_schema(Person)
    assert identity_fields(plan, Many(Person)) == ("name",)  # a field named "name"
    assert identity_fields(plan, Many(Person, identity="relationship")) == (
        "relationship",
    )
    assert identity_fields(plan, Many(Person, identity=["name", "relationship"])) == (
        "name",
        "relationship",
    )
    with pytest.raises(ValueError, match="no field of"):
        identity_fields(plan, Many(Person, identity="nickname"))


async def test_a_compound_identity_is_named_in_the_discovery_request() -> None:
    world = readme_world()
    await invoke(world, None, Many(Person, identity=["name", "relationship"]))
    ((_, payload, _),) = of_kind(world, "discover")
    text = payload["fields"]["mentions"]["instructions"]
    assert "by their name (full name) and relationship (how they relate to me)" in text


async def test_without_an_identity_field_discovery_follows_the_description() -> None:
    class Reading(BaseModel):
        """A temperature reading from a sensor."""

        value: float
        taken: Optional[float] = None

    world = FakeWorld("The sensor read 21.5 degrees, then the probe read 19.")
    world.entity("r1", schema=Reading, mentions=["21.5"], fields={"value": 21.5})
    world.entity("r2", schema=Reading, mentions=["19"], fields={"value": 19})
    result = await invoke(world, None, Many(Reading))
    assert len(result.records) == 2
    assert any("no_identity_field" in w for w in result.warnings)
    ((_, payload, _),) = of_kind(world, "discover")
    text = payload["fields"]["mentions"]["instructions"]
    assert "Every mention of a Reading (A temperature reading from a sensor)" in text
    assert " by their " not in text


async def test_many_label_is_what_prompts_call_an_entity() -> None:
    world = readme_world()
    await invoke(world, None, Many(Person, identity="name", label="friend"))
    ((_, payload, _),) = of_kind(world, "discover")
    assert (
        "Every mention of a friend by their name"
        in (payload["fields"]["mentions"]["instructions"])
    )
    ((_, presence, _),) = of_kind(world, "presence")
    assert presence["questions"]["presence"]["instructions"].startswith(
        "Does `input` mention any friend"
    )


# --------------------------------------------------------------------------- #
# The README scenario, without existing records
# --------------------------------------------------------------------------- #


async def test_the_readme_people_without_existing_records() -> None:
    world = readme_world()
    result = await invoke(world)
    assert labels(result) == ["Emma Thompson", "Michael Chen", "Sarah", "Olivia Davis"]
    assert all(r.kind == "insert" and r.json_doc_id is None for r in result.records)
    assert all(r.valid for r in result.records)
    emma, michael, sarah, olivia = result.responses
    assert isinstance(emma, Person)
    assert (emma.name, emma.relationship) == ("Emma Thompson", "my college friend")
    assert emma.notes == [
        "promoted to Senior Marketing Manager last month",
        "taken up rock climbing",
    ]
    assert isinstance(michael, Person) and michael.notes[0] == "a Data Scientist"
    assert isinstance(sarah, Person) and sarah.name == "Sarah"
    assert isinstance(olivia, Person) and olivia.relationship == "her cousin"
    # Sunny is a dog
    ((skip),) = result.skipped
    assert (skip.reason, skip.schema_name, skip.entity.label) == (
        "not_qualified",
        "Person",
        "Sunny",
    )
    assert skip.entity.qualify_p is not None and skip.entity.qualify_p < 0.5
    assert result.warnings == ()
    assert result.untouched == () and result.deleted == ()


async def test_the_call_sequence_of_the_readme_people() -> None:
    world = readme_world()
    result = await invoke(world)
    counts = Counter(kinds(result))
    assert counts == {
        "presence": 1,
        "discover": 1,
        "count": 1,
        "coref": 1,  # 7 mentions: 21 pairs in one request
        "qualify": 5,  # one per entity, Sunny included
        "extract": 4,  # one per qualifying entity
        "attribution": 4,
    }
    first = kinds(result)[:3]
    assert sorted(first) == ["count", "discover", "presence"]  # the parallel step
    assert kinds(result)[3] == "coref"
    assert kinds(result)[4:9] == ["qualify"] * 5


async def test_the_three_discovery_requests_are_in_flight_together() -> None:
    world = readme_world()
    in_flight = 0
    peak = 0
    inner = world.handler

    async def slow(model: str, payload: Body, meta: CallMeta) -> Any:
        nonlocal in_flight, peak
        if meta.kind in ("presence", "discover", "count"):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
        return inner(model, payload, meta)

    extractor = create_extractor(
        [Many(Person, identity="name")], transport=FakeTransport(slow)
    )
    await extractor.ainvoke(TEXT)
    assert peak == 3


async def test_entity_records_carry_their_mentions_with_spans() -> None:
    world = readme_world()
    result = await invoke(world)
    for record in result.records:
        entity = record.entity
        assert entity is not None and entity.mentions
        assert entity.first_mention == entity.mentions[0]
        for mention in entity.mentions:
            assert mention.span is not None
            assert TEXT[mention.span[0] : mention.span[1]] == mention.text
        assert entity.qualify_p is not None and entity.qualify_p > 0.5
        assert entity.parent is None and dict(entity.match_probs) == {}
    emma = result.records[0].entity
    assert emma is not None
    assert [m.text for m in emma.mentions] == ["Emma Thompson", "Emma"]
    assert emma.label == "Emma Thompson"  # the mention with the most tokens


async def test_records_are_ordered_by_first_mention_and_keys_by_cluster() -> None:
    world = FakeWorld("Zed met Amy. Then Bob came.")
    # declared in reverse order of the text
    world.entity("Bob", schema=Person, mentions=["Bob"], fields={"name": "Bob"})
    world.entity("Amy", schema=Person, mentions=["Amy"], fields={"name": "Amy"})
    world.entity("Zed", schema=Person, mentions=["Zed"], fields={"name": "Zed"})
    result = await invoke(world)
    assert labels(result) == ["Zed", "Amy", "Bob"]
    assert [r.entity.key for r in result.records if r.entity] == ["e0", "e1", "e2"]
    assert [m.name for m in result.responses] == ["Zed", "Amy", "Bob"]  # type: ignore[attr-defined]
    assert result.response_metadata == [
        {"id": "Person:e0", "json_doc_id": None},
        {"id": "Person:e1", "json_doc_id": None},
        {"id": "Person:e2", "json_doc_id": None},
    ]


async def test_a_skipped_entity_keeps_its_place_in_the_numbering() -> None:
    world = readme_world()
    result = await invoke(world)
    keys = {r.entity.label: r.entity.key for r in result.records if r.entity}
    (skip,) = result.skipped
    assert skip.entity.key == "e1"  # Sunny is mentioned between Emma and Emma
    assert keys == {
        "Emma Thompson": "e0",
        "Michael Chen": "e2",
        "Sarah": "e3",
        "Olivia Davis": "e4",
    }


# --------------------------------------------------------------------------- #
# Presence
# --------------------------------------------------------------------------- #


async def test_a_text_without_entities_makes_no_extraction_calls() -> None:
    world = FakeWorld("The weather is nice and the bus was late.")
    result = await invoke(world)
    assert result.records == () and result.responses == () and result.skipped == ()
    assert set(kinds(result)) <= {"presence", "discover", "count"}
    assert "extract" not in kinds(result) and "coref" not in kinds(result)
    assert "qualify" not in kinds(result)
    assert result.attempts == 0


def with_presence(world: FakeWorld, p: float) -> FakeTransport:
    def presence(model: str, payload: Body, meta: CallMeta) -> Any:
        return clef_response(
            {qid: noul(p) for qid in payload["questions"]}, model=payload["model"]
        )

    return FakeTransport(route(default=world.handler, presence=presence))


async def test_presence_below_the_floor_stops_everything_after_discovery() -> None:
    world = readme_world()
    transport = with_presence(world, 0.1)
    extractor = create_extractor([Many(Person, identity="name")], transport=transport)
    result = await extractor.ainvoke(TEXT)
    assert result.records == ()
    assert sorted(kinds(result)) == ["count", "discover", "presence"]


async def test_borderline_presence_goes_on_when_discovery_found_mentions() -> None:
    world = readme_world()
    extractor = create_extractor(
        [Many(Person, identity="name")], transport=with_presence(world, 0.3)
    )
    result = await extractor.ainvoke(TEXT)
    assert len(result.records) == 4


async def test_borderline_presence_without_mentions_stops() -> None:
    world = FakeWorld("The weather is nice.")
    extractor = create_extractor(
        [Many(Person, identity="name")], transport=with_presence(world, 0.3)
    )
    result = await extractor.ainvoke(world.text)
    assert result.records == ()
    assert "coref" not in kinds(result) and "extract" not in kinds(result)


async def test_the_presence_thresholds_are_configurable() -> None:
    world = readme_world()
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=with_presence(world, 0.3),
        discovery=DiscoveryConfig(presence_absent=0.4, presence_present=0.5),
    )
    assert (await extractor.ainvoke(TEXT)).records == ()


async def test_a_forced_schema_skips_the_presence_gate_only() -> None:
    world = readme_world()
    extractor = create_extractor(
        [Many(Person, identity="name")], transport=with_presence(world, 0.01)
    )
    result = await extractor.ainvoke(TEXT, schema=Person)
    assert "presence" not in kinds(result)
    assert len(result.records) == 4  # even though the gate would have said no
    assert {"discover", "count"} <= set(kinds(result))


# --------------------------------------------------------------------------- #
# Qualification
# --------------------------------------------------------------------------- #


async def test_qualification_skips_what_is_not_a_person() -> None:
    world = readme_world()
    result = await invoke(world)
    assert "Sunny" not in labels(result)
    qualify = of_kind(world, "qualify")
    assert len(qualify) == 5
    sunny = next(
        payload
        for _, payload, _ in qualify
        if payload["state"]["entity"]["label"] == "Sunny"
    )
    assert sunny["questions"]["qualify"]["instructions"].startswith(
        "Is `entity.label` a Person"
    )
    at = TEXT.index("Sunny")
    assert sunny["state"]["entity"]["mentions"] == [
        {"text": "Sunny", "span": [at, at + 5]}
    ]
    assert sunny["state"]["input"] == TEXT


async def test_a_non_entity_costs_no_extraction_request() -> None:
    world = readme_world()
    result = await invoke(world)
    extracting = {c.entity for c in result.calls if c.kind == "extract"}
    assert "e1" not in extracting  # Sunny
    assert extracting == {"e0", "e2", "e3", "e4"}


# --------------------------------------------------------------------------- #
# Per-entity extraction: the focus header and the templated instructions
# --------------------------------------------------------------------------- #


async def test_every_entity_is_asked_with_its_own_focus_header() -> None:
    world = readme_world()
    await invoke(world)
    extracts = of_kind(world, "extract")
    assert len(extracts) == 4
    by_entity = {meta.entity: payload for _, payload, meta in extracts}
    emma = by_entity["e0"]["state"]
    assert emma.startswith('<focus schema="Person" entity="Emma Thompson">\n')
    assert (
        "Answer each field for Emma Thompson only. Ignore facts about any other person."
        in emma
    )
    first = TEXT.index("Emma Thompson")
    second = TEXT.index("Emma got")
    assert (
        f'The input mentions Emma Thompson as: "Emma Thompson" at characters '
        f'{first}-{first + 13}, "Emma" at characters {second}-{second + 4}.'
    ) in emma
    assert emma.index("</focus>") < emma.index("<input>")
    assert TEXT in emma
    sarah = by_entity["e3"]["state"]
    assert sarah.startswith('<focus schema="Person" entity="Sarah">\n')
    assert "Answer each field for Sarah only." in sarah


async def test_instructions_are_templated_with_the_entity_label() -> None:
    world = readme_world()
    await invoke(world)
    extracts = {meta.entity: payload for _, payload, meta in of_kind(world, "extract")}
    emma = extracts["e0"]["fields"]
    assert emma["name"]["instructions"].startswith("Emma Thompson's full name")
    assert emma["relationship"]["instructions"].startswith(
        "Emma Thompson's how they relate to me"
    )
    michael = extracts["e2"]["fields"]
    assert michael["notes"]["instructions"].startswith(
        "Michael Chen's things worth remembering"
    )
    for payload in extracts.values():
        assert "the Person" not in payload["fields"]["name"]["instructions"]


async def test_the_attribution_state_names_the_entity_and_its_mentions() -> None:
    world = readme_world()
    await invoke(world)
    attribution = [c for c in of_kind(world, "attribution") if c[2].entity == "e0"]
    assert len(attribution) == 1
    _, payload, _ = attribution[0]
    entity = payload["state"]["entity"]
    assert entity["label"] == "Emma Thompson"
    assert [m["text"] for m in entity["mentions"]] == ["Emma Thompson", "Emma"]
    assert all(m["span"][1] > m["span"][0] for m in entity["mentions"])
    assert payload["state"]["input"] == TEXT
    instructions = [q["instructions"] for q in payload["questions"].values()]
    assert any("`entity.label`" in text for text in instructions)


async def test_the_presence_gate_is_not_asked_again_per_entity() -> None:
    world = readme_world()
    result = await invoke(world)
    assert kinds(result).count("presence") == 1
    ((_, payload, _),) = of_kind(world, "presence")
    assert list(payload["questions"]) == ["presence"]
    assert payload["questions"]["presence"]["instructions"].startswith(
        "Does `input` mention any Person (Someone the speaker knows"
    )


async def test_the_stored_identity_value_is_extracted_like_any_field() -> None:
    world = readme_world()
    result = await invoke(world)
    sarah = result.records[2]
    assert sarah.fields["name"].span is not None
    start, end = sarah.fields["name"].span
    assert TEXT[start:end] == "Sarah"  # the label "Sarah" is only framing


async def test_each_entitys_sub_groups_and_clef_fields_carry_its_entity() -> None:
    class Address(BaseModel):
        city: str

    class Customer(BaseModel):
        """A customer."""

        name: str
        vip: bool
        home: Optional[Address] = None

    text = "Ann lives in Rome and is a VIP. Bob is not."
    world = FakeWorld(text)
    world.entity(
        "Ann",
        schema=Customer,
        mentions=["Ann"],
        fields={"name": "Ann", "vip": True, "home": {"city": "Rome"}},
    )
    world.entity("Bob", schema=Customer, mentions=["Bob"], fields={"name": "Bob"})
    result = await invoke(world, None, Many(Customer, identity="name"))
    ann, bob = result.responses
    assert isinstance(ann, Customer) and isinstance(bob, Customer)
    assert ann.vip is True and ann.home is not None and ann.home.city == "Rome"
    assert bob.vip is False and bob.home is None
    for _, payload, meta in world.transport().calls:
        if meta.kind in ("clef_fields", "presence") and meta.entity:
            assert payload["state"]["entity"]["label"] in ("Ann", "Bob")


# --------------------------------------------------------------------------- #
# Attribution: a fact stated about one person lands on another
# --------------------------------------------------------------------------- #


async def test_a_bled_value_is_rejected_on_the_wrong_entity_and_kept_on_the_right() -> (
    None
):
    world = readme_world()
    # the friend's remark about rock climbing is Emma's; offered for Michael too
    world.bleed("Michael Chen", "notes", "taken up rock climbing")
    result = await invoke(world)
    emma, michael = result.records[0], result.records[1]
    assert "taken up rock climbing" in emma.value.notes  # type: ignore[union-attr]
    assert "taken up rock climbing" not in michael.value.notes  # type: ignore[union-attr]
    assert michael.value.notes == ["a Data Scientist", "thinking of going vegan"]  # type: ignore[union-attr]
    rejected = [(r.path, r.reason) for r in michael.rejected]
    assert rejected == [("/notes/2", "attribution")]
    assert emma.rejected == ()
    # the answer says why: the attribution probability was low for that element
    attributed = michael.fields["notes"].attribution_p
    assert isinstance(attributed, tuple) and attributed[2] is not None
    assert attributed[2] < 0.3 and attributed[0] and attributed[0] > 0.9


async def test_a_bled_scalar_is_refused_and_the_field_stays_empty() -> None:
    world = readme_world()
    world.bleed("Michael Chen", "relationship", "My neighbor")  # Sarah's
    result = await invoke(world)
    michael = result.records[1]
    assert michael.value.relationship is None  # type: ignore[union-attr]
    assert [(r.path, r.reason) for r in michael.rejected] == [
        ("/relationship", "attribution")
    ]
    assert result.records[2].value.relationship == "My neighbor"  # type: ignore[union-attr]


async def test_a_unsure_first_answer_is_re_asked_for_that_entity() -> None:
    world = readme_world()
    world.low_confidence("Sarah", "relationship", 0.5)
    result = await invoke(world)
    sarah = result.records[2]
    assert sarah.fields["relationship"].reask is True
    reasks = of_kind(world, "reask")
    assert len(reasks) == 1
    assert reasks[0][2].entity == "e3"
    assert '<focus schema="Person" entity="Sarah">' in reasks[0][1]["state"]


# --------------------------------------------------------------------------- #
# Two people who are both called Emma
# --------------------------------------------------------------------------- #

TWO_EMMAS = (
    "Emma sings in my choir, my friend. My sister Emma paints and I visit Emma often."
)


def two_emmas() -> FakeWorld:
    world = FakeWorld(TWO_EMMAS)
    world.entity(
        "Emma",
        schema=Person,
        mentions=[("Emma", 0)],
        fields={"name": "Emma", "relationship": "my friend", "notes": ["sings"]},
    )
    world.entity(
        "Emma",
        schema=Person,
        mentions=[("Emma", 1), ("Emma", 2)],
        fields={"name": "Emma", "relationship": "My sister", "notes": ["paints"]},
    )
    return world


async def test_two_distinct_emmas_become_two_records() -> None:
    world = two_emmas()
    result = await invoke(world)
    assert len(result.records) == 2
    first, second = result.records
    assert first.entity is not None and second.entity is not None
    assert first.entity.label == second.entity.label == "Emma"
    assert [m.span for m in first.entity.mentions] == [(0, 4)]
    assert len(second.entity.mentions) == 2
    assert first.value.relationship == "my friend"  # type: ignore[union-attr]
    assert second.value.relationship == "My sister"  # type: ignore[union-attr]
    assert first.value.notes == ["sings"] and second.value.notes == ["paints"]  # type: ignore[union-attr]
    assert not result.warnings


async def test_two_emmas_are_told_apart_by_spans_in_every_request() -> None:
    world = two_emmas()
    await invoke(world)
    extracts = {meta.entity: payload for _, payload, meta in of_kind(world, "extract")}
    assert "at characters 0-4" in extracts["e0"]["state"]
    assert "at characters 0-4" not in extracts["e1"]["state"]
    attribution = {
        meta.entity: payload for _, payload, meta in of_kind(world, "attribution")
    }
    assert attribution["e0"]["state"]["entity"]["mentions"] == [
        {"text": "Emma", "span": [0, 4]}
    ]
    assert len(attribution["e1"]["state"]["entity"]["mentions"]) == 2


# --------------------------------------------------------------------------- #
# Coreference through the engine
# --------------------------------------------------------------------------- #


async def test_the_incremental_strategy_finds_the_same_entities() -> None:
    world = readme_world()
    result = await invoke(world, coref=CorefConfig(strategy="incremental"))
    assert labels(result) == ["Emma Thompson", "Michael Chen", "Sarah", "Olivia Davis"]
    coref = of_kind(world, "coref")
    assert len(coref) == 6  # 7 mentions: all but the first are assigned
    assert all(next(iter(c[1]["questions"])).startswith("assign.m") for c in coref)


async def test_auto_switches_to_incremental_above_the_threshold() -> None:
    world = readme_world()
    result = await invoke(world, coref=CorefConfig(max_pairwise_mentions=5))
    assert len(result.records) == 4
    assert all(
        next(iter(c[1]["questions"])).startswith("assign.")
        for c in of_kind(world, "coref")
    )


# --------------------------------------------------------------------------- #
# Count reconciliation
# --------------------------------------------------------------------------- #


async def test_a_count_above_the_entities_found_triggers_a_broadened_discovery() -> (
    None
):
    world = FakeWorld("Emma Thompson came to dinner. My cousin came too.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
    )
    world.entity(
        "My cousin",
        schema=Person,
        mentions=[],
        descriptive=["My cousin"],
        fields={"name": "My cousin"},
    )
    result = await invoke(world)
    discoveries = of_kind(world, "discover")
    assert len(discoveries) == 2  # the plain one, then the broadened one
    plain = discoveries[0][1]["fields"]["mentions"]["instructions"]
    wide = discoveries[1][1]["fields"]["mentions"]["instructions"]
    assert "descriptive references" not in plain
    assert wide.endswith(
        "Also include descriptive references such as 'my cousin' when no name is given."
    )
    assert labels(result) == ["Emma Thompson", "My cousin"]
    assert result.warnings == ()
    # the descriptive mention is a mention like any other: with its span
    cousin = result.records[1].entity
    assert cousin is not None and cousin.mentions[0].span is not None
    assert world.text[slice(*cousin.mentions[0].span)] == "My cousin"


async def test_a_count_that_stays_above_warns_and_invents_nobody() -> None:
    world = FakeWorld("Emma Thompson came to dinner. Michael Chen did not.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={"name": "Michael Chen"},
    )
    world.count(Person, 3)  # the model counts a person who is not there
    result = await invoke(world)
    assert labels(result) == ["Emma Thompson", "Michael Chen"]
    assert len(of_kind(world, "discover")) == 2  # one broadened attempt, no more
    (warning,) = [w for w in result.warnings if "count_exceeds_discovered" in w]
    assert "the count is 3 but only 2 Person(s) were found" in warning
    assert "no entity was invented" in warning


async def test_a_count_below_the_entities_found_checks_for_a_merge() -> None:
    world = FakeWorld("Emma Thompson came. Later Emma left.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={"name": "Emma Thompson"},
    )
    # the pairwise model is unsure Emma is Emma Thompson: two clusters at first
    world.coref("Emma", "Emma Thompson", 0.3)
    result = await invoke(world)
    assert labels(result) == ["Emma Thompson"]  # merged by the cluster-level check
    assert result.warnings == ()
    merges = [c for c in of_kind(world, "coref") if "merge" in c[1]["questions"]]
    assert len(merges) == 1
    state = merges[0][1]["state"]
    assert set(state) == {"input", "a", "b"}
    assert state["a"]["label"] == "Emma Thompson" and state["b"]["label"] == "Emma"
    assert state["a"]["mentions"][0]["span"] == [0, 13]
    assert merges[0][1]["questions"]["merge"]["instructions"] == (
        "Do `a` and `b` refer to the same Person?"
    )
    assert len(of_kind(world, "discover")) == 1  # no broadened discovery
    emma = result.records[0].entity
    assert emma is not None and [m.text for m in emma.mentions] == [
        "Emma Thompson",
        "Emma",
    ]


async def test_a_merge_the_check_refuses_leaves_a_warning() -> None:
    world = FakeWorld("Ann Lee came. Ann Kim left.")
    world.entity(
        "Ann Lee", schema=Person, mentions=["Ann Lee"], fields={"name": "Ann Lee"}
    )
    world.entity(
        "Ann Kim", schema=Person, mentions=["Ann Kim"], fields={"name": "Ann Kim"}
    )
    world.coref("Ann Lee", "Ann Kim", 0.3)  # a candidate for a merge ...
    world.count(Person, 1)  # ... because the model counted one person
    result = await invoke(world)
    assert labels(result) == ["Ann Lee", "Ann Kim"]  # the check says: two people
    (warning,) = [w for w in result.warnings if "count_below_discovered" in w]
    assert "the count is 1 but 2 Person(s) were found" in warning
    assert (
        len([c for c in of_kind(world, "coref") if "merge" in c[1]["questions"]]) == 1
    )


async def test_clusters_that_look_nothing_alike_are_not_even_checked() -> None:
    world = FakeWorld("Ann Lee came. Bob Roe left.")
    world.entity(
        "Ann Lee", schema=Person, mentions=["Ann Lee"], fields={"name": "Ann Lee"}
    )
    world.entity(
        "Bob Roe", schema=Person, mentions=["Bob Roe"], fields={"name": "Bob Roe"}
    )
    world.count(Person, 1)
    result = await invoke(world)
    assert len(result.records) == 2
    assert not [c for c in of_kind(world, "coref") if "merge" in c[1]["questions"]]
    assert any("count_below_discovered" in w for w in result.warnings)


def crowd(n: int) -> FakeWorld:
    names = [f"Person{chr(65 + i)}{chr(97 + i)}" for i in range(n)]
    text = " ".join(f"{name} came." for name in names)
    world = FakeWorld(text)
    for name in names:
        world.entity(name, schema=Person, mentions=[name], fields={"name": name})
    return world


async def test_above_the_trust_limit_a_mismatch_only_warns() -> None:
    world = crowd(11)
    world.count(Person, 13)  # more than count_trust_max (10) on both sides
    result = await invoke(world)
    assert len(result.records) == 11
    assert len(of_kind(world, "discover")) == 1  # no broadened discovery
    (warning,) = [w for w in result.warnings if "count_exceeds_discovered" in w]
    assert "not trusted above 10" in warning


async def test_the_trust_limit_is_configurable() -> None:
    world = crowd(11)
    world.count(Person, 13)
    await invoke(world, discovery=DiscoveryConfig(count_trust_max=20))
    assert len(of_kind(world, "discover")) == 2


async def test_broadening_can_be_switched_off() -> None:
    world = FakeWorld("Emma Thompson came to dinner. My cousin came too.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
    )
    world.entity(
        "My cousin",
        schema=Person,
        mentions=[],
        descriptive=["My cousin"],
        fields={"name": "My cousin"},
    )
    result = await invoke(
        world, discovery=DiscoveryConfig(broaden_on_count_mismatch=False)
    )
    assert labels(result) == ["Emma Thompson"]
    assert len(of_kind(world, "discover")) == 1
    assert any("count_exceeds_discovered" in w for w in result.warnings)


async def test_a_matching_count_asks_nothing_more() -> None:
    world = readme_world()
    result = await invoke(world)
    assert len(of_kind(world, "discover")) == 1
    assert not [c for c in of_kind(world, "coref") if "merge" in c[1]["questions"]]
    assert result.warnings == ()


# --------------------------------------------------------------------------- #
# Mixed schemas, existing records, validation
# --------------------------------------------------------------------------- #


async def test_a_many_schema_and_a_one_schema_run_in_parallel() -> None:
    text = "Emma Thompson organises the Spring Fair on 2024-04-20. Michael Chen helps."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={"name": "Michael Chen"},
    )
    world.entity(
        "the Event",
        schema=Event,
        mentions=["Spring Fair"],
        fields={"title": "Spring Fair", "when": "2024-04-20"},
    )
    result = await invoke(world, text, Many(Person, identity="name"), Event)
    assert [r.schema_name for r in result.records] == ["Person", "Person", "Event"]
    emma, michael, event = result.responses
    assert isinstance(emma, Person) and emma.name == "Emma Thompson"
    assert isinstance(michael, Person) and michael.name == "Michael Chen"
    assert isinstance(event, Event) and event.title == "Spring Fair"
    assert result.records[2].entity is not None and result.records[2].entity.key == "e0"
    # both ran: Person through discovery, Event through its own presence gate
    counts = Counter(kinds(result))
    assert counts["discover"] == 1 and counts["count"] == 1
    assert counts["presence"] == 2  # Person (collection) and Event
    presence_schemas = {meta.schema for _, _, meta in of_kind(world, "presence")}
    assert presence_schemas == {"Person", "Event"}


async def test_a_forced_many_schema_leaves_the_others_out() -> None:
    world = readme_world()
    world.entity("the Event", schema=Event, mentions=["park"], fields={"title": "park"})
    extractor = extractor_for(world, Many(Person, identity="name"), Event)
    result = await extractor.ainvoke(TEXT, schema="Person")
    assert {r.schema_name for r in result.records} == {"Person"}
    assert "presence" not in kinds(result)


async def test_many_with_existing_records_is_not_implemented_before_m8() -> None:
    world = readme_world()
    extractor = extractor_for(world, Many(Person, identity="name"))
    records = [("0", "Person", {"name": "Emma Thompson"})]
    with pytest.raises(NotImplementedError, match="M8"):
        await extractor.ainvoke(TEXT, existing=records)
    with pytest.raises(NotImplementedError, match="M8"):
        await extractor.ainvoke(TEXT, existing={"Person": {"name": "Emma Thompson"}})
    assert world.transport().calls == []  # nothing was spent


async def test_list_form_existing_records_imply_many_which_needs_m8() -> None:
    world = readme_world()
    extractor = extractor_for(world, Person)
    with pytest.raises(NotImplementedError, match="M8"):
        await extractor.ainvoke(
            TEXT, existing=[("0", "Person", {"name": "Emma Thompson"})]
        )
    assert world.transport().calls == []


async def test_existing_records_of_another_schema_do_not_stop_a_many() -> None:
    text = "Emma Thompson organises the Spring Fair on 2024-04-20. Michael Chen helps."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={"name": "Michael Chen"},
    )
    world.entity(
        "the Event",
        schema=Event,
        mentions=["Spring Fair"],
        fields={"title": "Spring Fair", "when": "2024-04-20"},
    )
    extractor = extractor_for(world, Many(Person, identity="name"), One(Event))
    result = await extractor.ainvoke(
        text, existing={"Event": {"title": "Spring Fair", "when": None}}
    )
    assert [(r.schema_name, r.kind) for r in result.records] == [
        ("Person", "insert"),
        ("Person", "insert"),
        ("Event", "update"),
    ]
    assert result.untouched == ()


async def test_the_scope_refuses_existing_records_until_m8() -> None:
    from trustcall_system_one import compile_schema

    scope = Scope(compile_schema(Person), TEXT, Many(Person), existing=({"name": "x"},))
    run = Run(transport=readme_world().transport())
    with pytest.raises(NotImplementedError, match="M8"):
        await collect(scope, run)


async def test_collect_returns_the_entities_and_skips_without_extracting() -> None:
    from trustcall_system_one import compile_schema

    world = readme_world()
    scope = Scope(compile_schema(Person), TEXT, Many(Person, identity="name"))
    run = Run(transport=world.transport())
    outcome = await collect(scope, run)
    assert [e.label for e in outcome.entities] == [
        "Emma Thompson",
        "Michael Chen",
        "Sarah",
        "Olivia Davis",
    ]
    assert [s.reason for s in outcome.skipped] == ["not_qualified"]
    assert outcome.presence is not None and outcome.presence > 0.9
    assert outcome.count == 4
    assert scope.parent is None and scope.pointer == ""
    assert "extract" not in {c[2].kind for c in world.transport().calls}


async def test_an_entity_that_fails_validation_is_invalid_and_skipped() -> None:
    class Strict(BaseModel):
        """A strict person."""

        name: str
        email: str = Field(description="email address")

    text = "Ann <ann@x.io> wrote. Bob did not give an address."
    world = FakeWorld(text)
    world.entity(
        "Ann",
        schema=Strict,
        mentions=["Ann"],
        fields={"name": "Ann", "email": "ann@x.io"},
    )
    world.entity("Bob", schema=Strict, mentions=["Bob"], fields={"name": "Bob"})
    result = await invoke(world, text, Many(Strict, identity="name"))
    ann, bob = result.records
    assert ann.valid and not bob.valid and bob.value is None
    assert [m.name for m in result.responses] == ["Ann"]  # type: ignore[attr-defined]
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label) == ("validation_failed", "Bob")
    assert result.attempts >= 2  # the repair loop ran for Bob


async def test_extraction_errors_propagate_from_a_concurrent_entity() -> None:
    world = readme_world()

    def broken(model: str, payload: Body, meta: CallMeta) -> Any:
        if meta.kind == "extract" and "Michael Chen" in payload["state"][:120]:
            raise RuntimeError("boom")
        return world.handler(model, payload, meta)

    extractor = create_extractor(
        [Many(Person, identity="name")], transport=FakeTransport(broken)
    )
    with pytest.raises(RuntimeError, match="boom"):
        await extractor.ainvoke(TEXT)


async def test_a_one_schema_is_unchanged_by_the_collection_engine() -> None:
    text = "Emma Thompson, my college friend, climbs."
    world = FakeWorld(text)
    world.entity(
        "the Person",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "relationship": "my college friend"},
    )
    result = await invoke(world, text, One(Person))
    (record,) = result.records
    assert record.valid and record.entity is not None
    assert record.entity.label == "the Person" and record.entity.mentions == ()
    extract = of_kind(world, "extract")[0][1]["state"]
    assert extract.startswith('<focus entity="the Person">')  # the old header
    assert "discover" not in kinds(result)


async def test_the_configs_are_accepted_kept_and_replaceable() -> None:
    discovery = DiscoveryConfig(count_trust_max=3)
    coref = CorefConfig(strategy="pairwise")
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=readme_world().transport(),
        discovery=discovery,
        coref=coref,
    )
    assert extractor.discovery is discovery and extractor.coref is coref
    changed = extractor.with_options(coref=CorefConfig(strategy="incremental"))
    assert changed.coref.strategy == "incremental" and changed.discovery is discovery
    defaults = create_extractor([Person], transport=readme_world().transport())
    assert defaults.discovery == DiscoveryConfig() and defaults.coref == CorefConfig()


async def test_strict_mode_raises_for_an_entity_that_stays_invalid() -> None:
    from trustcall_system_one import ExtractionValidationError

    class Strict(BaseModel):
        """A strict person."""

        name: str
        email: str

    world = FakeWorld("Ann has no address.")
    world.entity("Ann", schema=Strict, mentions=["Ann"], fields={"name": "Ann"})
    extractor = extractor_for(world, Many(Strict, identity="name"), strict=True)
    with pytest.raises(ExtractionValidationError, match="Strict failed validation"):
        await extractor.ainvoke(world.text)


async def test_an_any_record_is_left_untouched_by_a_many_schema() -> None:
    world = readme_world()
    extractor = extractor_for(world, Many(Person, identity="name"))
    result = await extractor.ainvoke(
        TEXT, existing=[("keep", "__any__", {"name": "Emma Thompson"})]
    )
    assert len(result.records) == 4  # the Many inserted, as it has no records
    assert result.untouched == ("keep",)
    assert any(
        "a Many schema cannot claim a record before M8" in w for w in result.warnings
    )


async def test_the_collection_works_in_a_batch_and_through_the_sync_bridge() -> None:
    world = readme_world()
    extractor = extractor_for(world, Many(Person, identity="name"))
    first, second = await extractor.abatch([TEXT, TEXT])
    assert len(first.records) == len(second.records) == 4
    with extractor_for(readme_world()) as sync:
        assert len(sync.invoke(TEXT).records) == 4


async def test_discovery_of_a_long_input_is_chunked_and_spans_stay_original() -> None:
    filler = " ".join(f"word{i}" for i in range(200))
    text = f"Ann Lee came. {filler} Bob Roe stayed. {filler} Cy Ng left. {filler}"
    world = FakeWorld(text)
    for name in ("Ann Lee", "Bob Roe", "Cy Ng"):
        world.entity(name, schema=Person, mentions=[name], fields={"name": name})
    limit = 1500
    inner = world.handler

    def cut(model: str, payload: Body, meta: CallMeta) -> Any:
        state = payload["state"]
        if meta.kind != "discover" or not isinstance(state, str):
            return inner(model, payload, meta)
        start = state.index("<input>\n") + len("<input>\n")
        end = state.rindex("\n</input>")
        if end - start <= limit:
            return inner(model, payload, meta)
        # Clairic read only the beginning of the input and says so
        read = state[: start + limit]
        body = inner(model, {**payload, "state": read + "\n</input>"}, meta)
        assert isinstance(body, dict)
        return {**body, "truncated": True, "state_text": read}

    extractor = create_extractor(
        [Many(Person, identity="name")], transport=FakeTransport(cut)
    )
    result = await extractor.ainvoke(text)
    assert labels(result) == ["Ann Lee", "Bob Roe", "Cy Ng"]
    assert any("clairic_chunked" in w for w in result.warnings)
    for record in result.records:
        assert record.entity is not None
        (mention,) = record.entity.mentions
        assert mention.span is not None
        assert text[slice(*mention.span)] == record.entity.label
    assert len([c for c in result.calls if c.kind == "discover"]) > 1
