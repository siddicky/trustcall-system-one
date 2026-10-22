"""A ``Many`` schema with existing records: match, update, insert, delete, untouched."""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from collections import Counter
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from trustcall_system_one import (
    AmbiguousMatchError,
    CallMeta,
    CorefConfig,
    ExtractionResult,
    Extractor,
    Many,
    MatchConfig,
    One,
    create_extractor,
)
from trustcall_system_one._wire import ChoiceQuestion
from trustcall_system_one.testing import (
    FakeTransport,
    FakeWorld,
    choice,
    clef_response,
    route,
)

Body = dict[str, Any]
Record = tuple[str, str, dict[str, Any]]

CONVERSATION = """\
Me: I ran into Emma Thompson at the park yesterday. She was walking her new puppy, a golden retriever named Sunny. She mentioned she got promoted to Senior Marketing Manager last month.
Friend: That's great news for Emma! How's she enjoying the new role?
Me: She seems to be thriving. Oh, and did you know she's taken up rock climbing? She invited me to join her at the climbing gym sometime.
Friend: Wow, rock climbing? That's quite a change from hiking. Speaking of friends, have you heard from Michael Chen recently?
Me: Actually, yes. We had a video call last week. He's switched jobs and is now working as a Data Scientist at a startup. He's also mentioned he's thinking of going vegan.
Friend: That's a big change for Michael! Both career and diet-wise. How about your neighbor, Sarah? Is she still teaching?
Me: Sarah's doing well. Her kids are growing up fast - her oldest just started middle school. She's still teaching, but now she's focusing on special education. She's really passionate about it.
Friend: That's wonderful. Oh, before I forget, I wanted to introduce you to my cousin who just moved to town. Her name is Olivia Davis, she's a 27-year-old graphic designer. She's looking to meet new people and expand her social circle. I thought you two might get along well.
Me: That sounds great! I'd love to meet her. Maybe we could all get together for coffee next week?
Friend: Perfect! I'll set it up. Olivia loves art and is always sketching in her free time. She also volunteers at the local animal shelter on weekends."""  # noqa: E501


class Person(BaseModel):
    """Someone the user knows or interacts with."""

    name: str
    relationship: Optional[str] = Field(None, description="how they relate to me")
    notes: list[str] = Field([], description="memories and observations")


class Event(BaseModel):
    """A thing that happens at a time."""

    title: str
    when: Optional[str] = None


EMMA = {
    "name": "Emma Thompson",
    "relationship": "College friend",
    "notes": ["Loves hiking", "Works in marketing", "Has a dog named Max"],
}
MICHAEL = {
    "name": "Michael Chen",
    "relationship": "Coworker",
    "notes": ["Great at problem-solving", "Vegetarian", "Plays guitar"],
}
SARAH = {
    "name": "Sarah Johnson",
    "relationship": "Neighbor",
    "notes": ["Has two kids", "Loves gardening", "Makes amazing cookies"],
}
EXISTING = [("0", "Person", EMMA), ("1", "Person", MICHAEL), ("2", "Person", SARAH)]

EMMA_NOTES = [
    "walking her new puppy, a golden retriever named Sunny",
    "got promoted to Senior Marketing Manager last month",
    "taken up rock climbing",
]
MICHAEL_NOTES = [
    "switched jobs and is now working as a Data Scientist at a startup",
    "thinking of going vegan",
]
SARAH_NOTES = [
    "her oldest just started middle school",
    "focusing on special education",
    "really passionate about it",
]
OLIVIA_NOTES = [
    "a 27-year-old graphic designer",
    "looking to meet new people and expand her social circle",
    "loves art and is always sketching in her free time",
    "volunteers at the local animal shelter on weekends",
]


def people_world(*, bleed: bool = True) -> FakeWorld:
    """The README conversation: three people we have records of, a fourth, a dog."""
    world = FakeWorld(CONVERSATION)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={"name": "Emma Thompson", "notes": EMMA_NOTES},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen", "Michael"],
        fields={"name": "Michael Chen", "notes": MICHAEL_NOTES},
        matches={"name": "Michael Chen"},
    )
    world.entity(
        "Sarah",
        schema=Person,
        mentions=["Sarah"],
        fields={"name": "Sarah", "notes": SARAH_NOTES},
        matches="Sarah Johnson",  # a string that occurs in the record
    )
    world.entity(
        "Olivia Davis",
        schema=Person,
        mentions=["Olivia Davis", "Olivia"],
        fields={
            "name": "Olivia Davis",
            "relationship": "my cousin",
            "notes": OLIVIA_NOTES,
        },
    )
    world.non_entity("Sunny", schema=Person, mentions=["Sunny"])
    if bleed:
        # Emma's remark sits right before Michael's name: offered for Michael,
        # refused by the attribution check
        world.bleed("Michael Chen", "notes", "taken up rock climbing")
    return world


def extractor_for(world: FakeWorld, *schemas: Any, **options: Any) -> Extractor:
    chosen = schemas or (Many(Person, identity="name"),)
    return create_extractor(list(chosen), transport=world.transport(), **options)


async def invoke(
    world: FakeWorld,
    existing: Any,
    *schemas: Any,
    text: str | None = None,
    **options: Any,
) -> ExtractionResult:
    extractor = extractor_for(world, *schemas, **options)
    return await extractor.ainvoke(
        text if text is not None else world.text, existing=existing
    )


def of_kind(world: FakeWorld, kind: str) -> list[tuple[str, Body, CallMeta]]:
    return [c for c in world.transport().calls if c[2].kind == kind]


def kinds(result: ExtractionResult) -> Counter[str]:
    return Counter(call.kind for call in result.calls)


def ops(record: Any) -> list[tuple[str, str, Any]]:
    return [(p.op, p.path, p.value) for p in record.patches]


# --------------------------------------------------------------------------- #
# The README scenario, end to end
# --------------------------------------------------------------------------- #


async def test_the_readme_scenario_updates_three_people_and_inserts_olivia() -> None:
    world = people_world()
    result = await invoke(world, EXISTING, enable_inserts=True)
    assert [(r.json_doc_id, r.kind, r.entity.label) for r in result.records] == [  # type: ignore[union-attr]
        ("0", "update", "Emma Thompson"),
        ("1", "update", "Michael Chen"),
        ("2", "update", "Sarah"),
        (None, "insert", "Olivia Davis"),
    ]
    emma, michael, sarah, olivia = result.records
    # the patches: notes are appended, one element at a time, and nothing else
    assert ops(emma) == [("add", "/notes/-", note) for note in EMMA_NOTES]
    assert ops(michael) == [("add", "/notes/-", note) for note in MICHAEL_NOTES]
    assert ops(sarah) == [("add", "/notes/-", note) for note in SARAH_NOTES]
    assert olivia.patches == ()
    # the records after the update: existing notes untouched, in place
    assert emma.after == {**EMMA, "notes": [*EMMA["notes"], *EMMA_NOTES]}
    assert michael.after == {**MICHAEL, "notes": [*MICHAEL["notes"], *MICHAEL_NOTES]}
    assert sarah.after == {**SARAH, "notes": [*SARAH["notes"], *SARAH_NOTES]}
    assert emma.before == EMMA and sarah.before == SARAH
    # "Sarah" is not a rename of "Sarah Johnson"
    assert [(r.reason, r.path) for r in sarah.rejected if r.path == "/name"] == [
        ("identity_subset", "/name")
    ]
    assert sarah.after is not None and sarah.after["name"] == "Sarah Johnson"
    # Emma's rock climbing, offered for Michael, was refused by attribution
    assert [r.reason for r in michael.rejected if r.path.startswith("/notes")] == [
        "attribution"
    ]
    # Olivia: a verbatim span, not a paraphrase
    assert olivia.value is not None
    assert olivia.value.relationship == "my cousin"  # type: ignore[attr-defined]
    assert olivia.value.notes == OLIVIA_NOTES  # type: ignore[attr-defined]
    # Sunny is a dog
    assert [(s.reason, s.entity.label) for s in result.skipped] == [
        ("not_qualified", "Sunny")
    ]
    assert result.untouched == () and result.deleted == ()
    assert [m["json_doc_id"] for m in result.response_metadata] == [
        "0",
        "1",
        "2",
        None,
    ]
    assert len(result.responses) == 4
    names = [p.name for p in result.responses]  # type: ignore[attr-defined]
    assert names == ["Emma Thompson", "Michael Chen", "Sarah Johnson", "Olivia Davis"]
    assert result.warnings == ()


async def test_the_readme_scenario_reports_the_match_probabilities() -> None:
    world = people_world()
    result = await invoke(world, EXISTING, enable_inserts=True)
    for record, expected in zip(
        result.records, ["0", "1", "2", "__new__"], strict=True
    ):
        assert record.entity is not None
        probs = record.entity.match_probs
        assert set(probs) == {"0", "1", "2", "__new__"}
        assert probs[expected] == 0.92
        assert max(probs, key=lambda k: probs[k]) == expected
    # entities keep their numbers over ALL entities, Sunny included
    assert [r.entity.key for r in result.records if r.entity] == [
        "e0",
        "e2",
        "e3",
        "e4",
    ]
    assert result.skipped[0].entity.key == "e1"  # Sunny is the second mention


async def test_the_call_sequence_with_existing_records() -> None:
    world = people_world()
    result = await invoke(world, EXISTING, enable_inserts=True)
    order = [call.kind for call in result.calls]
    assert sorted(order[:3]) == ["count", "discover", "presence"]
    counts = kinds(result)
    assert counts["coref"] >= 1 and counts["qualify"] == 1
    assert counts["match"] == 4  # one choice per entity that qualifies
    assert counts["delete"] == 0  # enable_deletes is off
    # matching comes after qualification and before any extraction
    assert order.index("match") > order.index("qualify")
    assert order.index("match") < order.index("extract")
    # only the update path runs for matched records: an input-only cross-check and
    # gates; the insert of Olivia has none of them
    assert counts["crosscheck"] >= 3 and counts["gate"] == 8


async def test_the_options_of_a_match_request_are_the_records_as_json() -> None:
    world = people_world()
    await invoke(world, EXISTING, enable_inserts=True)
    requests = of_kind(world, "match")
    assert len(requests) == 4
    _, payload, meta = requests[0]
    assert meta.schema == "Person" and meta.entity == "e0"
    question = payload["questions"]["match"]
    assert question["instructions"] == "Which listed record is `entity.label`?"
    assert question["criteria"]["rec_0"] == EMMA
    assert question["criteria"]["rec_2"] == SARAH
    assert question["criteria"]["__new__"] == "A Person not among the listed records"
    ChoiceQuestion.model_validate(question)
    state = payload["state"]
    assert set(state) == {"input", "entity"}
    assert state["entity"]["label"] == "Emma Thompson"
    assert [m["text"] for m in state["entity"]["mentions"]] == [
        "Emma Thompson",
        "Emma",
    ]


async def test_records_are_ordered_by_first_mention_not_by_the_order_of_existing() -> (
    None
):
    world = people_world()
    reordered = [EXISTING[2], EXISTING[0], EXISTING[1]]
    result = await invoke(world, reordered, enable_inserts=True)
    assert [r.json_doc_id for r in result.records] == ["0", "1", "2", None]
    assert result.untouched == ()


async def test_without_enable_inserts_olivia_is_skipped() -> None:
    world = people_world()
    result = await invoke(world, EXISTING)  # enable_inserts defaults to False
    assert [(r.json_doc_id, r.kind) for r in result.records] == [
        ("0", "update"),
        ("1", "update"),
        ("2", "update"),
    ]
    assert [(s.reason, s.entity.label) for s in result.skipped] == [
        ("not_qualified", "Sunny"),
        ("inserts_disabled", "Olivia Davis"),
    ]
    skip = next(s for s in result.skipped if s.reason == "inserts_disabled")
    assert skip.entity.first_mention is not None
    assert skip.entity.first_mention.text == "Olivia Davis"
    assert "enable_inserts is off" in skip.detail
    assert len(result.responses) == 3
    # no extraction was spent on her
    extracted = {m.entity for _, _, m in of_kind(world, "extract")}
    assert extracted == {"e0", "e2", "e3"}  # Emma, Michael, Sarah: not Olivia (e4)


async def test_skipped_entities_are_in_first_mention_order() -> None:
    world = people_world()
    result = await invoke(world, EXISTING)
    keys = [int(s.entity.key[1:]) for s in result.skipped]
    assert keys == sorted(keys)
    assert [s.reason for s in result.skipped] == ["not_qualified", "inserts_disabled"]


async def test_without_enable_updates_matched_records_are_skipped_not_untouched() -> (
    None
):
    world = people_world()
    result = await invoke(world, EXISTING, enable_inserts=True, enable_updates=False)
    assert [(r.json_doc_id, r.kind) for r in result.records] == [(None, "insert")]
    reasons = [(s.reason, s.entity.label) for s in result.skipped]
    assert ("updates_disabled", "Emma Thompson") in reasons
    assert ("updates_disabled", "Michael Chen") in reasons
    assert ("updates_disabled", "Sarah") in reasons
    assert result.untouched == ()  # the three records were dealt with (skipped)
    assert not any(m.kind == "gate" for _, _, m in world.transport().calls)


async def test_a_record_the_input_does_not_mention_is_untouched() -> None:
    world = people_world()
    zed: Record = (
        "3",
        "Person",
        {"name": "Zed Qux", "relationship": None, "notes": []},
    )
    result = await invoke(world, [*EXISTING, zed], enable_inserts=True)
    assert [r.json_doc_id for r in result.records] == ["0", "1", "2", None]
    assert result.untouched == ("3",)
    assert all("e" not in (r.json_doc_id or "") for r in result.records)
    # no extraction for it, and it was an option for every entity
    first = of_kind(world, "match")[0][1]["questions"]["match"]["criteria"]
    assert first["rec_3"]["name"] == "Zed Qux"


async def test_no_entities_leave_every_record_untouched_without_extraction() -> None:
    text = "The weather was fine and nothing else happened."
    world = FakeWorld(text)
    world.entity("Ann", schema=Event, mentions=["weather"], fields={"title": "x"})
    result = await invoke(world, EXISTING, text=text, enable_inserts=True)
    assert result.records == () and result.skipped == ()
    assert result.untouched == ("0", "1", "2")
    assert result.responses == () and result.deleted == ()
    used = kinds(result)
    assert used["extract"] == used["match"] == used["gate"] == used["delete"] == 0
    assert set(used) <= {"presence", "discover", "count"}


async def test_deletes_are_not_asked_when_nobody_is_mentioned() -> None:
    text = "The weather was fine and nothing else happened."
    world = FakeWorld(text)
    world.entity("x", schema=Event, mentions=["weather"], fields={"title": "x"})
    result = await invoke(
        world, EXISTING, text=text, enable_inserts=True, enable_deletes=True
    )
    assert result.untouched == ("0", "1", "2")
    assert kinds(result)["delete"] == 0


async def test_dict_form_existing_and_an_implied_many_run_the_same_matching() -> None:
    world = FakeWorld("Emma Thompson was here.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    # dict form with an explicit Many: one record, still matched
    result = await invoke(world, {"Person": EMMA})
    (record,) = result.records
    assert (record.kind, record.json_doc_id) == ("update", "Person")
    assert ops(record) == [("add", "/notes/-", "was here")]
    # list form without a wrapper implies Many
    world = FakeWorld("Emma Thompson was here.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    extractor = create_extractor([Person], transport=world.transport())
    result = await extractor.ainvoke(
        "Emma Thompson was here.", existing=[("e-1", "Person", EMMA)]
    )
    (record,) = result.records
    assert (record.kind, record.json_doc_id) == ("update", "e-1")
    assert kinds(result)["match"] == 1


async def test_a_bare_model_list_gets_its_minted_ids_back() -> None:
    world = FakeWorld("Emma Thompson was here.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    model = Person(**EMMA)
    result = await invoke(world, [model, Person(name="Zed", notes=[])])
    (record,) = result.records
    assert record.json_doc_id is not None and len(record.json_doc_id) == 36
    assert len(result.untouched) == 1


async def test_a_record_of_another_schema_is_left_alone_next_to_a_many() -> None:
    text = "Emma Thompson organises the Spring Fair on 2024-04-20."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["organises the Spring Fair"]},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "the Event",
        schema=Event,
        mentions=["Spring Fair"],
        fields={"title": "Spring Fair", "when": "2024-04-20"},
    )
    result = await invoke(
        world,
        [
            ("0", "Person", EMMA),
            ("ev", "Event", {"title": "Spring Fair", "when": None}),
        ],
        Many(Person, identity="name"),
        One(Event),
    )
    assert [(r.schema_name, r.kind, r.json_doc_id) for r in result.records] == [
        ("Person", "update", "0"),
        ("Event", "update", "ev"),
    ]
    assert result.untouched == ()


# --------------------------------------------------------------------------- #
# "__any__" records and Many
# --------------------------------------------------------------------------- #


async def test_a_many_schema_can_claim_an_any_record() -> None:
    world = FakeWorld("Emma Thompson was here. Bob Roe too.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Bob Roe",
        schema=Person,
        mentions=["Bob Roe"],
        fields={"name": "Bob Roe", "notes": ["too"]},
    )
    result = await invoke(
        world,
        [("anything", "__any__", EMMA)],
        Many(Person, identity="name"),
        enable_inserts=True,
    )
    assert [(r.kind, r.json_doc_id) for r in result.records] == [
        ("update", "anything"),
        ("insert", None),
    ]
    assert result.untouched == ()
    assert result.warnings == ()  # one candidate schema: nothing to decide


async def test_the_any_claim_is_presence_based_between_a_many_and_a_one() -> None:
    world = FakeWorld("Emma Thompson was here.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    result = await invoke(
        world,
        [("anything", "__any__", EMMA)],
        Many(Person, identity="name"),
        One(Event),  # the input says nothing about an Event
    )
    (record,) = result.records
    assert (record.schema_name, record.kind, record.json_doc_id) == (
        "Person",
        "update",
        "anything",
    )
    presences = {m.schema for _, _, m in of_kind(world, "presence")}
    assert presences == {"Person", "Event"}


async def test_an_any_record_is_claimed_by_one_schema_only() -> None:
    world = FakeWorld("Emma Thompson was here.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "x", schema=Event, mentions=["here"], fields={"title": "here", "when": None}
    )
    result = await invoke(
        world,
        [("anything", "__any__", EMMA), ("other", "__any__", EMMA)],
        Many(Person, identity="name"),
        Many(Event, identity="title"),
    )
    # the only present schema whose model validates it takes the first record; the
    # second goes to the other one, so no record is shared and nothing is updated twice
    assert [(r.schema_name, r.json_doc_id) for r in result.records] == [
        ("Person", "anything")
    ]
    assert result.untouched == ("other",)
    claims = [w for w in result.warnings if "was given to" in w]
    assert len(claims) == 2
    assert "'anything' was given to Person" in claims[0]
    assert "'other' was given to Event" in claims[1]


async def test_a_many_with_its_own_records_does_not_claim_an_any_record() -> None:
    world = FakeWorld("Emma Thompson was here.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    result = await invoke(
        world,
        [("0", "Person", EMMA), ("loose", "__any__", {"name": "Loose"})],
        Many(Person, identity="name"),
    )
    assert [r.json_doc_id for r in result.records] == ["0"]
    assert result.untouched == ("loose",)
    (warning,) = [w for w in result.warnings if "'loose'" in w]
    assert "every schema has records of its own" in warning


# --------------------------------------------------------------------------- #
# Ambiguity and collisions with a scripted matcher
# --------------------------------------------------------------------------- #


def matcher(
    world: FakeWorld,
    table: dict[str, dict[str, float]],
    *,
    confidence: float | None = None,
    **handlers: Any,
) -> FakeTransport:
    """A transport answering ``match`` as scripted, by entity label and record name."""

    def answer(model: str, payload: Body, meta: CallMeta) -> Any:
        label = payload["state"]["entity"]["label"]
        options = dict(payload["questions"]["match"]["criteria"])
        wanted = table[label]
        probs = {
            option: wanted.get(
                "__new__" if option == "__new__" else options[option]["name"], 0.0
            )
            for option in options
        }
        best = max(probs, key=lambda o: probs[o])
        return clef_response(
            {"match": choice(best, probs, confidence)}, model=payload["model"]
        )

    return FakeTransport(route(default=world.handler, match=answer, **handlers))


def two_people() -> FakeWorld:
    text = "Emma Thompson met Bob Roe. Later Bob thanked Emma Thompson."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["met Bob Roe"]},
    )
    world.entity(
        "Bob Roe",
        schema=Person,
        mentions=["Bob Roe", "Bob"],
        fields={"name": "Bob Roe", "notes": ["thanked Emma Thompson"]},
    )
    return world


RECORDS: list[Record] = [
    ("0", "Person", {"name": "Emma Thompson", "relationship": None, "notes": []}),
    ("1", "Person", {"name": "Bob Roe", "relationship": None, "notes": []}),
]


async def test_an_entity_below_the_accept_threshold_is_ambiguous_with_candidates() -> (
    None
):
    world = two_people()
    transport = matcher(
        world,
        {
            "Emma Thompson": {"Emma Thompson": 0.5, "Bob Roe": 0.3, "__new__": 0.2},
            "Bob Roe": {"Bob Roe": 0.95, "__new__": 0.05},
        },
    )
    extractor = create_extractor([Many(Person, identity="name")], transport=transport)
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    assert [(r.json_doc_id, r.entity.label) for r in result.records] == [  # type: ignore[union-attr]
        ("1", "Bob Roe")
    ]
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label) == ("ambiguous_match", "Emma Thompson")
    assert dict(skip.candidates) == {"0": 0.5, "1": 0.3, "__new__": 0.2}
    assert [c[0] for c in skip.candidates] == ["0", "1", "__new__"]  # most likely first
    # the record she might have been is untouched
    assert result.untouched == ("0",)
    assert not any(c.kind == "extract" and c.entity == "e0" for c in result.calls)


async def test_a_confidence_below_the_minimum_is_ambiguous_too() -> None:
    world = two_people()
    transport = matcher(
        world,
        {
            "Emma Thompson": {"Emma Thompson": 0.95, "__new__": 0.05},
            "Bob Roe": {"Bob Roe": 0.95, "__new__": 0.05},
        },
        confidence=0.3,
    )
    extractor = create_extractor([Many(Person, identity="name")], transport=transport)
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    assert result.records == ()
    assert [s.reason for s in result.skipped] == ["ambiguous_match"] * 2
    assert result.untouched == ("0", "1")
    # lowering the minimum lets them through
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=matcher(
            world,
            {
                "Emma Thompson": {"Emma Thompson": 0.95, "__new__": 0.05},
                "Bob Roe": {"Bob Roe": 0.95, "__new__": 0.05},
            },
            confidence=0.3,
        ),
        match=MatchConfig(min_confidence=0.3),
    )
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    assert [r.json_doc_id for r in result.records] == ["0", "1"]


async def test_on_ambiguous_insert_inserts_an_entity_that_is_likely_new() -> None:
    world = two_people()
    # not sure of any answer (low confidence): Emma is most likely new, Bob a record
    table = {
        "Emma Thompson": {"Emma Thompson": 0.05, "__new__": 0.95},
        "Bob Roe": {"Bob Roe": 0.95, "__new__": 0.05},
    }
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=matcher(world, table, confidence=0.3),
        enable_inserts=True,
        match=MatchConfig(on_ambiguous="insert"),
    )
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    assert [(r.json_doc_id, r.kind) for r in result.records] == [(None, "insert")]
    # Bob is not likely new: "insert" does not make a duplicate of his record
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label) == ("ambiguous_match", "Bob Roe")
    # inserts still obey enable_inserts
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=matcher(world, table, confidence=0.3),
        match=MatchConfig(on_ambiguous="insert"),
    )
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    assert [s.reason for s in result.skipped] == [
        "inserts_disabled",
        "ambiguous_match",
    ]


async def test_declared_match_odds_exercise_the_ambiguous_paths() -> None:
    text = "Emma Thompson met Bob Roe."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["met Bob Roe"]},
        match_odds={"Emma Thompson": 0.55, "Bob Roe": 0.05, "__new__": 0.4},
    )
    world.entity(
        "Bob Roe",
        schema=Person,
        mentions=["Bob Roe"],
        fields={"name": "Bob Roe", "notes": ["met Emma"]},
        match_odds={"Bob Roe": 0.9, "__new__": 0.1},
        match_confidence=0.3,  # sure of the option, not of the answer
    )
    result = await invoke(world, RECORDS, Many(Person, identity="name"), text=text)
    assert result.records == ()
    reasons = {s.entity.label: s for s in result.skipped}
    assert reasons["Emma Thompson"].candidates == (
        ("0", 0.55),
        ("__new__", 0.4),
        ("1", 0.05),
    )
    assert "confidence is 0.30" in reasons["Bob Roe"].detail
    assert result.untouched == ("0", "1")


async def test_on_ambiguous_raise_stops_before_any_extraction() -> None:
    world = two_people()
    transport = matcher(
        world,
        {
            "Emma Thompson": {"Emma Thompson": 0.5, "__new__": 0.5},
            "Bob Roe": {"Bob Roe": 0.95, "__new__": 0.05},
        },
    )
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=transport,
        match=MatchConfig(on_ambiguous="raise"),
    )
    with pytest.raises(AmbiguousMatchError, match="Emma Thompson") as caught:
        await extractor.ainvoke(world.text, existing=RECORDS)
    assert dict(caught.value.candidates) == {"0": 0.5, "__new__": 0.5}
    seen = {call[2].kind for call in transport.calls}
    assert "extract" not in seen and "match" in seen
    assert any(c.kind == "match" for c in caught.value.calls)  # the spend is reported


async def test_two_entities_claiming_one_record_warn_and_the_loser_falls_back() -> None:
    world = two_people()
    table = {
        "Emma Thompson": {"Emma Thompson": 0.9, "Bob Roe": 0.05, "__new__": 0.05},
        "Bob Roe": {"Emma Thompson": 0.7, "Bob Roe": 0.25, "__new__": 0.05},
    }
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=matcher(world, table),
        match=MatchConfig(accept=0.2),
    )
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    # Emma keeps her record; Bob falls back to his own
    assert [(r.json_doc_id, r.entity.label) for r in result.records] == [  # type: ignore[union-attr]
        ("0", "Emma Thompson"),
        ("1", "Bob Roe"),
    ]
    (warning,) = [w for w in result.warnings if w.startswith("match_collision")]
    assert "Emma Thompson (e0)" in warning and "Bob Roe (e1)" in warning
    assert "'0'" in warning and "p=0.90, 0.70" in warning
    assert result.untouched == ()


async def test_a_collision_loser_with_no_acceptable_option_is_ambiguous() -> None:
    world = two_people()
    table = {
        "Emma Thompson": {"Emma Thompson": 0.9, "__new__": 0.1},
        "Bob Roe": {"Emma Thompson": 0.7, "__new__": 0.3},
    }
    extractor = create_extractor(
        [Many(Person, identity="name")], transport=matcher(world, table)
    )
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    assert [r.json_doc_id for r in result.records] == ["0"]
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label) == ("ambiguous_match", "Bob Roe")
    assert dict(skip.candidates) == {"0": 0.7, "__new__": 0.3}
    assert any(w.startswith("match_collision") for w in result.warnings)
    assert result.untouched == ("1",)


def one_entity_in_two_forms() -> FakeWorld:
    text = "Emma Thompson arrived. Later Emma left."
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={"name": "Emma Thompson", "notes": ["arrived", "left"]},
        matches={"name": "Emma Thompson"},
    )
    # coreference doubts that the two forms are one, and the count agrees with
    # the two clusters, so the discovery stage keeps two entities
    world.coref("Emma Thompson", "Emma", 0.2)
    world.count(Person, 2)
    return world


async def test_collision_merge_joins_two_entities_that_are_one_and_matches_again() -> (
    None
):
    world = one_entity_in_two_forms()
    extractor = extractor_for(
        world, Many(Person, identity="name"), match=MatchConfig(collision="merge")
    )
    result = await extractor.ainvoke(world.text, existing=[("0", "Person", EMMA)])
    (record,) = result.records
    assert (record.kind, record.json_doc_id) == ("update", "0")
    assert record.entity is not None
    assert [m.text for m in record.entity.mentions] == ["Emma Thompson", "Emma"]
    assert record.entity.label == "Emma Thompson"
    assert ops(record) == [("add", "/notes/-", "arrived"), ("add", "/notes/-", "left")]
    assert any(w.startswith("match_merged") for w in result.warnings)
    assert not any(w.startswith("match_collision") for w in result.warnings)
    counts = kinds(result)
    assert counts["match"] == 3  # both entities, then the merged one
    merges = [c for c in of_kind(world, "coref") if "merge" in c[1]["questions"]]
    assert len(merges) >= 1


async def test_collision_warn_keeps_two_entities_and_the_second_is_ambiguous() -> None:
    world = one_entity_in_two_forms()
    result = await invoke(world, [("0", "Person", EMMA)], Many(Person, identity="name"))
    assert [(r.json_doc_id, r.entity.label) for r in result.records] == [  # type: ignore[union-attr]
        ("0", "Emma Thompson")
    ]
    (skip,) = result.skipped
    assert skip.reason == "ambiguous_match" and skip.entity.label == "Emma"
    assert any(w.startswith("match_collision") for w in result.warnings)


# --------------------------------------------------------------------------- #
# Prefilter on 300 records
# --------------------------------------------------------------------------- #


async def test_300_existing_records_are_prefiltered_and_the_true_match_is_kept() -> (
    None
):
    world = FakeWorld("Emma Thompson was here. Bob joined her.")
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["was here"]},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Bob",
        schema=Person,
        mentions=["Bob"],
        fields={"name": "Bob", "notes": ["joined her"]},
        matches={"name": "Robert Smith"},
    )
    records = [
        (f"r{i}", "Person", {"name": f"Zed Qux{i:03d}", "notes": []})
        for i in range(300)
    ]
    records[299] = ("r299", "Person", {"name": "Emma Thompson", "notes": []})
    records[298] = ("r298", "Person", {"name": "Robert Smith", "notes": []})
    result = await invoke(world, records, Many(Person, identity="name"))
    assert [(r.json_doc_id, r.kind) for r in result.records] == [
        ("r299", "update"),
        ("r298", "update"),
    ]
    for _, payload, _ in of_kind(world, "match"):
        assert len(payload["questions"]["match"]["criteria"]) == 255
    (warning,) = [w for w in result.warnings if w.startswith("match_prefiltered")]
    assert "300 existing Person records" in warning
    assert len(result.untouched) == 298
    assert set(result.untouched).isdisjoint({"r298", "r299"})


# --------------------------------------------------------------------------- #
# Deletes
# --------------------------------------------------------------------------- #


async def test_a_record_the_input_says_to_remove_is_deleted_last() -> None:
    world = people_world()
    world.delete("3", 0.95)
    zed: Record = (
        "3",
        "Person",
        {"name": "Zed Qux", "relationship": None, "notes": ["x"]},
    )
    result = await invoke(
        world, [*EXISTING, zed], enable_inserts=True, enable_deletes=True
    )
    assert [(r.json_doc_id, r.kind) for r in result.records] == [
        ("0", "update"),
        ("1", "update"),
        ("2", "update"),
        (None, "insert"),
        ("3", "delete"),
    ]
    delete = result.records[-1]
    assert delete.before == zed[2] and delete.after is None
    assert delete.value is None and delete.valid and delete.patches == ()
    assert delete.entity is None and delete.delete_p == 0.95
    assert [r.delete_p for r in result.records[:-1]] == [None] * 4
    assert result.deleted == ("3",)
    assert result.untouched == ()  # a deleted record was dealt with
    assert len(result.responses) == 4  # a delete has no model
    assert len(result.response_metadata) == 4
    # one request for the four records, one question each over a shared state
    ((_, payload, meta),) = of_kind(world, "delete")
    assert meta.entity is None and meta.field == "0,1,2,3"
    assert set(payload["state"]) == {"input", "records"}
    assert list(payload["state"]["records"]) == ["r0", "r1", "r2", "r3"]
    assert list(payload["questions"]) == [f"delete.r{k}" for k in range(4)]


async def test_a_long_list_of_records_costs_one_delete_request_per_64() -> None:
    world = people_world(bleed=False)
    records: list[Record] = [
        (f"z{i}", "Person", {"name": f"Zed Qux{i:03d}", "notes": []})
        for i in range(130)
    ]
    result = await invoke(
        world, [*EXISTING, *records], enable_inserts=True, enable_deletes=True
    )
    asked = of_kind(world, "delete")
    assert sorted(len(p["questions"]) for _, p, _ in asked) == [5, 64, 64]
    assert kinds(result)["delete"] == 3
    assert result.deleted == ()


async def test_deletes_follow_the_threshold() -> None:
    world = people_world(bleed=False)
    world.delete("3", 0.8)  # below 0.85
    zed: Record = (
        "3",
        "Person",
        {"name": "Zed Qux", "relationship": None, "notes": []},
    )
    result = await invoke(
        world, [*EXISTING, zed], enable_inserts=True, enable_deletes=True
    )
    assert result.deleted == () and result.untouched == ("3",)
    result = await invoke(
        world,
        [*EXISTING, zed],
        enable_inserts=True,
        enable_deletes=True,
        match=MatchConfig(delete_threshold=0.75),
    )
    assert result.deleted == ("3",)


async def test_no_deletes_unless_enabled() -> None:
    world = people_world(bleed=False)
    world.delete("3", 0.99)
    zed: Record = (
        "3",
        "Person",
        {"name": "Zed Qux", "relationship": None, "notes": []},
    )
    result = await invoke(world, [*EXISTING, zed], enable_inserts=True)
    assert result.deleted == () and result.untouched == ("3",)
    assert of_kind(world, "delete") == []


async def test_a_record_both_updated_and_deleted_is_a_conflict() -> None:
    world = people_world()
    world.delete("1", 0.97)  # Michael is matched AND flagged, and the text changes him
    result = await invoke(world, EXISTING, enable_inserts=True, enable_deletes=True)
    assert [(r.json_doc_id, r.kind) for r in result.records] == [
        ("0", "update"),
        ("2", "update"),
        (None, "insert"),
    ]
    conflict = next(s for s in result.skipped if s.reason == "conflict_update_delete")
    assert conflict.entity.label == "Michael Chen"
    assert conflict.candidates == (("1", 0.92),)
    assert "neither updated nor deleted" in conflict.detail
    assert "operation(s) passed the gate" in conflict.detail
    assert result.deleted == ()  # an operation went through the gate: no delete
    assert result.untouched == ()  # the conflicting record was dealt with
    # the update ran to find out, and its operations are not in the result
    michael = [
        m
        for _, _, m in world.transport().calls
        if m.kind == "gate" and m.entity == "e2"
    ]
    assert michael


def silent_michael_world() -> FakeWorld:
    """Michael is mentioned and matched, but the text says nothing new about him."""
    fresh = FakeWorld(CONVERSATION)
    fresh.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={"name": "Emma Thompson", "notes": EMMA_NOTES},
        matches={"name": "Emma Thompson"},
    )
    fresh.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen", "Michael"],
        fields={"name": "Michael Chen"},  # no notes: nothing to change
        matches={"name": "Michael Chen"},
    )
    return fresh


async def test_a_mentioned_record_with_nothing_to_change_is_deleted() -> None:
    """ "Please remove Michael": mentioned, matched, flagged, nothing else changes."""
    world = silent_michael_world()
    world.delete("1", 0.97)
    result = await invoke(world, EXISTING, enable_deletes=True)
    assert [(r.json_doc_id, r.kind) for r in result.records] == [
        ("0", "update"),
        ("1", "delete"),
    ]
    assert result.deleted == ("1",)
    assert result.skipped == () or all(
        s.reason != "conflict_update_delete" for s in result.skipped
    )
    # the update was run to find out that nothing survived the gate
    assert any(
        m.kind == "extract" and m.entity == "e1" for _, _, m in world.transport().calls
    )


async def test_a_matched_flagged_record_is_deleted_when_updates_are_disabled() -> None:
    world = people_world()
    world.delete("1", 0.97)
    result = await invoke(
        world,
        EXISTING,
        enable_inserts=True,
        enable_updates=False,
        enable_deletes=True,
    )
    # no update could change him, so the removal stands; no request was spent on him
    assert [(r.json_doc_id, r.kind) for r in result.records] == [
        (None, "insert"),
        ("1", "delete"),
    ]
    assert result.deleted == ("1",)
    reasons = {s.entity.label: s.reason for s in result.skipped}
    assert "Michael Chen" not in reasons
    assert reasons["Emma Thompson"] == "updates_disabled"
    assert result.untouched == ()


async def test_a_flagged_record_an_ambiguous_entity_may_be_is_not_deleted() -> None:
    world = two_people()
    table = {
        "Emma Thompson": {"Emma Thompson": 0.5, "Bob Roe": 0.3, "__new__": 0.2},
        "Bob Roe": {"Bob Roe": 0.95, "__new__": 0.05},
    }
    delete_p = {"0": 0.95}

    def answer(model: str, payload: Body, meta: CallMeta) -> Any:
        ids = (meta.field or "").split(",")
        return clef_response(
            {
                qid: {"type": "noul", "noul": delete_p.get(rid, 0.05)}
                for qid, rid in zip(payload["questions"], ids, strict=True)
            },
            model=payload["model"],
        )

    transport = matcher(world, table, delete=answer)
    extractor = create_extractor(
        [Many(Person, identity="name")], transport=transport, enable_deletes=True
    )
    result = await extractor.ainvoke(world.text, existing=RECORDS)
    # record "0" is flagged, and Emma (ambiguous) most likely is that record
    assert result.deleted == ()
    assert [r.kind for r in result.records] == ["update"]  # Bob
    assert result.untouched == ("0",)
    (warning,) = [w for w in result.warnings if w.startswith("match_delete_held")]
    assert "'0'" in warning and "Emma Thompson" in warning


async def test_deletes_run_with_the_matching_not_after_it() -> None:
    world = people_world()
    result = await invoke(world, EXISTING, enable_inserts=True, enable_deletes=True)
    order = [c.kind for c in result.calls]
    assert order.count("delete") == 1  # one request for the three records
    assert order.index("delete") < order.index("extract")
    assert result.deleted == ()


async def test_deletes_make_no_difference_to_the_update_of_unflagged_records() -> None:
    world = people_world()
    result = await invoke(world, EXISTING, enable_inserts=True, enable_deletes=True)
    plain = await invoke(
        people_world(), EXISTING, enable_inserts=True, enable_deletes=False
    )
    assert [r.patches for r in result.records] == [r.patches for r in plain.records]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_extractor_carries_the_match_settings() -> None:
    config = MatchConfig(accept=0.7)
    extractor = create_extractor([Many(Person)], match=config)
    assert extractor.match is config
    assert create_extractor([Many(Person)]).match == MatchConfig()
    assert extractor.with_options(match=MatchConfig()).match == MatchConfig()


async def test_coreference_settings_still_apply_next_to_matching() -> None:
    world = people_world()
    result = await invoke(
        world,
        EXISTING,
        enable_inserts=True,
        coref=CorefConfig(strategy="pairwise"),
    )
    assert [r.json_doc_id for r in result.records] == ["0", "1", "2", None]


# --------------------------------------------------------------------------- #
# A confident wrong match must not rewrite another entity's record
# --------------------------------------------------------------------------- #


class Resident(BaseModel):
    """A person."""

    name: str = Field(description="full name")
    city: Optional[str] = Field(None, description="the city they live in")
    notes: list[str] = Field([], description="things worth remembering")


WATSON = "Emma Watson moved to Lyon and took up fencing."
THOMPSON: Record = (
    "a",
    "Resident",
    {"name": "Emma Thompson", "city": "Paris", "notes": ["likes tea"]},
)


def watson_world() -> FakeWorld:
    world = FakeWorld(WATSON)
    world.entity(
        "Emma Watson",
        schema=Resident,
        mentions=["Emma Watson"],
        fields={"name": "Emma Watson", "city": "Lyon", "notes": ["took up fencing"]},
        matches={"name": "Emma Thompson"},  # Clef picks the wrong Emma, p=0.92
    )
    return world


async def test_an_entity_with_another_name_than_its_record_is_refused() -> None:
    world = watson_world()
    extractor = create_extractor([Many(Resident)], transport=world.transport())
    result = await extractor.ainvoke(WATSON, existing=[THOMPSON])
    (record,) = result.records
    # nothing was applied: the record is as it was, and every proposed operation
    # is reported
    assert (record.kind, record.json_doc_id) == ("update", "a")
    assert record.patches == () and record.after == record.before == THOMPSON[2]
    assert [(r.op, r.path, r.reason) for r in record.rejected] == [
        ("replace", "/name", "identity_conflict"),
        ("replace", "/city", "identity_conflict"),
        ("add", "/notes/-", "identity_conflict"),
    ]
    assert "Emma Watson" in record.rejected[0].detail
    assert "Emma Thompson" in record.rejected[0].detail
    (skip,) = result.skipped
    assert (skip.reason, skip.entity.label) == ("ambiguous_match", "Emma Watson")
    assert skip.detail.startswith("identity conflict:")
    assert dict(skip.candidates) == {"a": 0.92, "__new__": pytest.approx(0.08)}
    # the change gate was never asked
    assert kinds(result)["gate"] == 0
    assert result.responses[0].name == "Emma Thompson"  # type: ignore[attr-defined]


async def test_a_short_form_or_diminutive_of_the_records_name_still_updates() -> None:
    text = "Liz moved to Lyon and took up fencing."
    world = FakeWorld(text)
    world.entity(
        "Liz",
        schema=Resident,
        mentions=["Liz"],
        fields={"name": "Liz", "city": "Lyon", "notes": ["took up fencing"]},
        matches={"name": "Elizabeth Smith"},
    )
    existing: Record = (
        "b",
        "Resident",
        {"name": "Elizabeth Smith", "city": "Paris", "notes": []},
    )
    extractor = create_extractor([Many(Resident)], transport=world.transport())
    result = await extractor.ainvoke(text, existing=[existing])
    (record,) = result.records
    assert result.skipped == ()
    assert [(p.op, p.path) for p in record.patches] == [
        ("replace", "/city"),
        ("add", "/notes/-"),
    ]
    assert record.after is not None and record.after["name"] == "Elizabeth Smith"
    assert [r.reason for r in record.rejected if r.path == "/name"] == [
        "identity_subset"
    ]


async def test_a_schema_without_an_identity_field_skips_the_guard() -> None:
    class Reading(BaseModel):
        """A temperature reading."""

        value: float
        note: Optional[str] = None

    text = "The probe read 21.5 degrees."
    world = FakeWorld(text)
    world.entity(
        "r",
        schema=Reading,
        mentions=["probe"],
        fields={"value": 21.5},
        matches={"value": 19.0},
    )
    extractor = create_extractor([Many(Reading)], transport=world.transport())
    result = await extractor.ainvoke(
        text, existing=[("r1", "Reading", {"value": 19.0, "note": None})]
    )
    (record,) = result.records
    assert [(p.op, p.path) for p in record.patches] == [("replace", "/value")]
    assert result.skipped == ()
    assert any("no_identity_field" in w for w in result.warnings)


async def test_the_guard_applies_per_entity_and_leaves_the_others_alone() -> None:
    text = "Emma Watson moved to Lyon. Later Bob Roe thanked her."
    world = FakeWorld(text)
    world.entity(
        "Emma Watson",
        schema=Resident,
        mentions=["Emma Watson"],
        fields={"name": "Emma Watson", "city": "Lyon"},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Bob Roe",
        schema=Resident,
        mentions=["Bob Roe"],
        fields={"name": "Bob Roe", "notes": ["thanked her"]},
        matches={"name": "Bob Roe"},
    )
    existing = [
        THOMPSON,
        ("b", "Resident", {"name": "Bob Roe", "city": None, "notes": []}),
    ]
    extractor = create_extractor([Many(Resident)], transport=world.transport())
    result = await extractor.ainvoke(text, existing=existing)
    thompson, bob = result.records
    assert thompson.patches == () and [p.path for p in bob.patches] == ["/notes/-"]
    assert [s.entity.label for s in result.skipped] == ["Emma Watson"]
