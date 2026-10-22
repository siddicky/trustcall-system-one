"""The sentence-based reading of open-ended ``list[str]`` fields.

Splitting and scoping are pure; the strategies ("list", "sentences", "auto") run on
scripted transports and on ``FakeWorld``.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import Script, ca, make_extractor
from trustcall_system_one import (
    Limits,
    Many,
    One,
    VerifyConfig,
    create_extractor,
    s1_field,
)
from trustcall_system_one._compile import compile_schema
from trustcall_system_one._sentences import (
    Sentence,
    candidate_sentences,
    eligible,
    split_sentences,
)
from trustcall_system_one.testing import FakeWorld

# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def model_of(record: Any) -> Any:
    """The validated model of a record, typed loosely (the schemas are local)."""
    return record.value


def texts(text: str) -> list[str]:
    return [s.text for s in split_sentences(text)]


def test_plain_sentences_are_split_with_their_offsets() -> None:
    text = "Dana moved to Berlin. She starts on Monday! Is that right? Yes."
    found = split_sentences(text)
    assert [s.text for s in found] == [
        "Dana moved to Berlin.",
        "She starts on Monday!",
        "Is that right?",
        "Yes.",
    ]
    assert all(text[s.start : s.end] == s.text for s in found)
    assert found[1].span == (22, 43)


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Wu will call tomorrow. Bring the files.",
        "Mrs. Alvarez and Mr. Chen met Prof. Okafor. They talked.",
        "Prices vary, e.g. in Paris. Ask first.",
        "That is, i.e. the second one. Fine.",
        "She lives in the U.S. now. He does not.",
        "Meet J. R. Smith at noon. Bring tea.",
        "Order No. 5 shipped on Oct. 5 from the depot. It arrived.",
    ],
)
def test_abbreviations_and_initials_do_not_end_a_sentence(text: str) -> None:
    found = texts(text)
    assert len(found) == 2, found
    assert found[0] + " " + found[1] == text


def test_a_word_that_is_also_an_abbreviation_still_ends_a_sentence() -> None:
    assert texts("I said no. Then I left. She sat. He ran.") == [
        "I said no.",
        "Then I left.",
        "She sat.",
        "He ran.",
    ]


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("The fee is 3.5 percent. It rose to $1.5M. Nice.", 3),
        ("See example.com/path. Then mail me at a.b@x.org. Done.", 3),
        ("Version 2.0.1 shipped. Rollback was 0.5 days.", 2),
        ("It cost 12.50. He paid 7.", 2),
    ],
)
def test_decimals_and_addresses_are_not_boundaries(text: str, count: int) -> None:
    assert len(texts(text)) == count


def test_lower_case_after_a_period_continues_the_sentence() -> None:
    assert texts("It was approx. five miles away. Then rain.") == [
        "It was approx. five miles away.",
        "Then rain.",
    ]
    assert texts("Wait... what happened? OK.") == ["Wait... what happened?", "OK."]


def test_quotes_and_brackets_after_the_period_stay_with_the_sentence() -> None:
    assert texts('She said "I am leaving." Then she left. (It rained.) Fine.') == [
        'She said "I am leaving."',
        "Then she left.",
        "(It rained.)",
        "Fine.",
    ]


def test_lines_bullets_labels_and_clauses() -> None:
    text = (
        "Me: Emma got promoted\nFriend: Great news; she earned it\n"
        "- likes tea\n2. owns a dog"
    )
    found = split_sentences(text)
    assert [s.text for s in found] == [
        "Emma got promoted",
        "Great news",
        "she earned it",
        "likes tea",
        "owns a dog",
    ]
    assert all(text[s.start : s.end] == s.text for s in found)


def test_a_colon_in_the_middle_is_not_a_speaker_label() -> None:
    note = "Quick note after catching up with Dana: she moved to Berlin."
    assert texts(note) == [note]
    assert texts("Meeting at 5: bring snacks.") == ["Meeting at 5: bring snacks."]


def test_pieces_without_letters_and_empty_input_give_nothing() -> None:
    assert split_sentences("") == []
    assert split_sentences("  \n \n") == []
    assert texts("... \n--- \nReal one.") == ["Real one."]


# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #

TALK = (
    "Emma Thompson got promoted. She loves hiking. The weather was fine. "
    "Michael Chen plays guitar. He is vegan. Lunch was good."
)


def test_without_mentions_every_sentence_is_a_candidate() -> None:
    sentences = split_sentences(TALK)
    assert candidate_sentences(sentences) == sentences


def test_a_mentioned_entity_gets_its_sentences_and_the_ones_after() -> None:
    sentences = split_sentences(TALK)
    emma = candidate_sentences(sentences, spans=[(0, 13)], follow=1)
    assert [s.text for s in emma] == [
        "Emma Thompson got promoted.",
        "She loves hiking.",
    ]
    michael = candidate_sentences(sentences, spans=[(68, 80)], follow=2)
    assert [s.text for s in michael] == [
        "Michael Chen plays guitar.",
        "He is vegan.",
        "Lunch was good.",
    ]


def test_mention_texts_are_matched_as_whole_words_when_there_is_no_span() -> None:
    sentences = split_sentences("Emmanuel is here. Emma left. Nothing else.")
    found = candidate_sentences(sentences, texts=["Emma"], follow=0)
    assert [s.text for s in found] == ["Emma left."]


def test_follow_zero_keeps_only_the_mentioning_sentences() -> None:
    sentences = split_sentences(TALK)
    found = candidate_sentences(sentences, spans=[(0, 13), (68, 80)], follow=0)
    assert [s.text for s in found] == [
        "Emma Thompson got promoted.",
        "Michael Chen plays guitar.",
    ]


def test_a_sentence_is_a_value_object() -> None:
    assert Sentence("a", 0, 1).span == (0, 1)


# --------------------------------------------------------------------------- #
# The strategies on a One
# --------------------------------------------------------------------------- #


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    city: Optional[str] = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


TEXT = (
    "Dana Ruiz called from Lyon. She wants to meet in May. Her mobile died. "
    "Dr. Wu will follow up."
)


def contact_script() -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana Ruiz", find="Dana Ruiz"),
        "city": ca("text", "Lyon", find="Lyon"),
    }
    world.sentences = {"meet in May": 0.9, "follow up": 0.8}
    return world


def asked_fields(world: Script, kind: str) -> list[set[str]]:
    return [set(payload["fields"]) for _, payload, _ in world.calls(kind)]


async def test_sentences_become_the_elements_verbatim_with_their_spans() -> None:
    world = contact_script()
    extractor = make_extractor(
        [One(Contact)], world, limits=Limits(list_strategy="sentences")
    )
    result = await extractor.ainvoke(TEXT)
    (record,) = result.records
    assert record.valid and record.value is not None
    assert model_of(record).notes == [
        "She wants to meet in May.",
        "Dr. Wu will follow up.",
    ]
    notes = record.fields["notes"]
    assert notes.spans is not None
    assert [TEXT[a:b] for a, b in notes.spans] == model_of(record).notes
    assert "list_strategy:sentences" in notes.warnings
    # Clairic was never asked about the field; Clef was asked once, per sentence
    assert all("notes" not in fields for fields in asked_fields(world, "extract"))
    ((_, payload, meta),) = world.calls("sentences")
    assert meta.field == "notes"
    assert list(payload["questions"]) == [f"sentence.{i}" for i in range(4)]
    asked = [q["instructions"] for q in payload["questions"].values()]
    assert 'Dr. Wu will follow up."?' in asked[3]
    assert "things worth remembering" in asked[0]
    assert "`entity.label`" in asked[0]
    # and the usual ladder judged each element
    assert world.calls("attribution")


async def test_a_threshold_decides_which_sentences_are_kept() -> None:
    world = contact_script()
    extractor = make_extractor(
        [One(Contact)],
        world,
        limits=Limits(list_strategy="sentences"),
        verify=VerifyConfig(sentence_keep=0.85),
    )
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert record.value is not None
    assert model_of(record).notes == ["She wants to meet in May."]


async def test_no_sentence_kept_leaves_a_required_list_empty() -> None:
    world = contact_script()
    world.sentences = {}
    extractor = make_extractor(
        [One(Contact)], world, limits=Limits(list_strategy="sentences")
    )
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert record.value is not None and model_of(record).notes == []


async def test_more_than_64_candidates_are_asked_in_batches_of_64() -> None:
    text = "Dana Ruiz called from Lyon. " + " ".join(
        f"Item number {i} is fine." for i in range(150)
    )
    world = contact_script()
    world.sentences = {"number 7 ": 0.9, "number 149 ": 0.9}
    extractor = make_extractor(
        [One(Contact)], world, limits=Limits(list_strategy="sentences")
    )
    (record,) = (await extractor.ainvoke(text)).records
    sizes = sorted(len(p["questions"]) for _, p, _ in world.calls("sentences"))
    assert sizes == [23, 64, 64]
    assert record.value is not None
    assert model_of(record).notes == [
        "Item number 7 is fine.",
        "Item number 149 is fine.",
    ]


async def test_more_candidates_than_the_cap_are_left_out_with_a_warning() -> None:
    text = "Dana Ruiz called from Lyon. " + " ".join(
        f"Item number {i} is fine." for i in range(30)
    )
    world = contact_script()
    extractor = make_extractor(
        [One(Contact)],
        world,
        limits=Limits(list_strategy="sentences", max_sentences=10),
    )
    result = await extractor.ainvoke(text)
    assert sum(len(p["questions"]) for _, p, _ in world.calls("sentences")) == 10
    assert any("max_sentences" in w for w in result.warnings)


class Pinned(BaseModel):
    """A record whose notes are always read by sentences."""

    name: str
    notes: list[str] = Field(
        default_factory=list,
        description="things worth remembering",
        json_schema_extra=s1_field(list_strategy="sentences"),
    )
    tags: list[str] = Field(default_factory=list, description="short labels")


async def test_a_field_can_pin_its_strategy_against_the_global_default() -> None:
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana Ruiz", find="Dana Ruiz"),
        "tags": ca("list", ["Lyon"], find_all=["Lyon"], confidence=0.97),
    }
    world.sentences = {"meet in May": 0.9}
    extractor = make_extractor(
        [One(Pinned)], world, limits=Limits(list_strategy="list")
    )
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert record.value is not None
    assert model_of(record).notes == ["She wants to meet in May."]
    assert model_of(record).tags == ["Lyon"]  # still Clairic's list
    assert asked_fields(world, "extract") == [{"name", "tags"}]


def test_list_strategy_is_validated() -> None:
    assert s1_field(list_strategy="auto")["trustcall_system_one"] == {
        "list_strategy": "auto"
    }
    with pytest.raises(ValueError, match="list_strategy"):
        s1_field(list_strategy="words")
    with pytest.raises(ValueError, match="list_strategy"):
        Limits(list_strategy="words")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="list_auto_below"):
        Limits(list_auto_below=0)
    with pytest.raises(ValueError, match="max_sentences"):
        Limits(max_sentences=0)
    with pytest.raises(ValueError, match="sentence_keep"):
        VerifyConfig(sentence_keep=1.5)
    assert Limits().list_strategy == "auto"
    assert VerifyConfig().sentence_keep == 0.6


def test_only_a_list_of_free_text_takes_a_list_strategy() -> None:
    class Bad(BaseModel):
        """Not a list of text."""

        scores: list[int] = Field(
            default_factory=list,
            description="scores",
            json_schema_extra=s1_field(list_strategy="sentences"),
        )

    with pytest.raises(ValueError, match=r"scores.*list\[str\]"):
        compile_schema(Bad)
    plan = compile_schema(Contact)
    notes = next(spec for spec in plan.specs if spec.pretty == "notes")
    city = next(spec for spec in plan.specs if spec.pretty == "city")
    assert eligible(notes) and not eligible(city)


# --------------------------------------------------------------------------- #
# auto
# --------------------------------------------------------------------------- #


def junk_script(confidence: float) -> Script:
    world = contact_script()
    world.clairic["notes"] = ca(
        "list",
        ["Dana", "Lyon"],
        find_all=["Dana", "Lyon"],
        confidence=confidence,
        p_null=0.05,
    )
    return world


async def test_auto_falls_back_to_sentences_when_the_list_answer_is_not_trusted() -> (
    None
):
    world = junk_script(1e-9)
    extractor = make_extractor([One(Contact)], world)  # "auto" is the default
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert record.value is not None
    assert model_of(record).notes == [
        "She wants to meet in May.",
        "Dr. Wu will follow up.",
    ]
    # Clairic was asked first, then Clef read the sentences
    assert any("notes" in f for f in asked_fields(world, "extract"))
    assert len(world.calls("sentences")) == 1
    warnings = record.fields["notes"].warnings
    assert any(w.startswith("list_strategy:auto->sentences") for w in warnings)


async def test_auto_keeps_a_confident_list_answer() -> None:
    world = junk_script(0.95)
    world.sentences = {}
    extractor = make_extractor([One(Contact)], world)
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert world.calls("sentences") == []
    assert record.value is not None
    assert model_of(record).notes == ["Dana", "Lyon"]


async def test_the_auto_threshold_is_a_limit() -> None:
    world = junk_script(0.6)
    extractor = make_extractor(
        [One(Contact)], world, limits=Limits(list_auto_below=0.7)
    )
    await extractor.ainvoke(TEXT)
    assert len(world.calls("sentences")) == 1


async def test_auto_leaves_a_null_list_answer_alone() -> None:
    world = contact_script()
    world.clairic["notes"] = ca("list", None, p_null=0.97)
    extractor = make_extractor([One(Contact)], world)
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert world.calls("sentences") == []
    assert record.value is not None and model_of(record).notes == []


async def test_list_strategy_list_never_reads_sentences() -> None:
    world = junk_script(1e-9)
    extractor = make_extractor(
        [One(Contact)], world, limits=Limits(list_strategy="list")
    )
    await extractor.ainvoke(TEXT)
    assert world.calls("sentences") == []


# --------------------------------------------------------------------------- #
# Scoped to an entity
# --------------------------------------------------------------------------- #

TALK2 = (
    "Emma Thompson got promoted to Senior Marketing Manager. She loves hiking. "
    "The weather was fine. The bus was late. Today was long. "
    "Michael Chen plays the guitar. He is thinking of going vegan. The sea was calm."
)


class Person(BaseModel):
    """Someone the speaker knows."""

    name: str = Field(description="full name")
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


def people_world() -> FakeWorld:
    world = FakeWorld(TALK2)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={
            "name": "Emma Thompson",
            "notes": [
                "Emma Thompson got promoted to Senior Marketing Manager.",
                "She loves hiking.",
            ],
        },
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={"name": "Michael Chen", "notes": ["He is thinking of going vegan."]},
    )
    return world


async def test_an_entity_is_asked_about_its_own_sentences_only() -> None:
    world = people_world()
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=world.transport(),
        limits=Limits(list_strategy="sentences", sentence_follow=1),
    )
    result = await extractor.ainvoke(TALK2)
    calls = [c for c in world.transport().calls if c[2].kind == "sentences"]
    seen = {
        c[1]["state"]["entity"]["label"]: [
            q["instructions"].split(': "', 1)[1].rstrip('"?')
            for q in c[1]["questions"].values()
        ]
        for c in calls
    }
    assert seen == {
        "Emma Thompson": [
            "Emma Thompson got promoted to Senior Marketing Manager.",
            "She loves hiking.",
        ],
        "Michael Chen": [
            "Michael Chen plays the guitar.",
            "He is thinking of going vegan.",
        ],
    }
    by_name = {model_of(r).name: model_of(r).notes for r in result.records if r.value}
    assert by_name == {
        "Emma Thompson": [
            "Emma Thompson got promoted to Senior Marketing Manager.",
            "She loves hiking.",
        ],
        "Michael Chen": ["He is thinking of going vegan."],
    }


# --------------------------------------------------------------------------- #
# Updates
# --------------------------------------------------------------------------- #

RECORD: dict[str, Any] = {
    "name": "Dana Ruiz",
    "city": "Lyon",
    "notes": ["Prefers email over phone", "Met at the 2023 summit"],
}
UPDATE_TEXT = (
    "Dana Ruiz called today. She wants to meet in May. She prefers email to phone. "
    "Her mobile died."
)


def update_script() -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana Ruiz", find="Dana Ruiz"),
        "city": ca("text", None, p_null=0.97),
        "notes": ca(
            "list",
            ["called", "wants"],
            find_all=["called", "wants"],
            confidence=1e-9,
            p_null=0.05,
        ),
    }
    world.sentences = {"meet in May": 0.9, "prefers email": 0.9}
    world.dedup = {"She prefers email to phone.": 0.93}
    return world


async def test_an_update_appends_a_verbatim_sentence_and_leaves_the_notes_alone() -> (
    None
):
    world = update_script()
    world.sentences = {"meet in May": 0.9}
    extractor = make_extractor([One(Contact)], world)
    result = await extractor.ainvoke(
        UPDATE_TEXT, existing=[("rec-1", "Contact", RECORD)]
    )
    (record,) = result.records
    assert record.valid and record.after is not None
    assert [(p.op, p.path, p.value) for p in record.patches] == [
        ("add", "/notes/-", "She wants to meet in May.")
    ]
    assert record.after["notes"] == [
        "Prefers email over phone",
        "Met at the 2023 summit",
        "She wants to meet in May.",
    ]
    assert record.after["name"] == "Dana Ruiz" and record.after["city"] == "Lyon"
    gated = [c[1]["state"]["proposed_change"]["path"] for c in world.calls("gate")]
    assert gated == ["/notes/-"]


async def test_sentences_need_no_input_only_re_read_of_the_field() -> None:
    world = update_script()
    world.sentences = {"meet in May": 0.9}
    extractor = make_extractor(
        [One(Contact)], world, limits=Limits(list_strategy="sentences")
    )
    (record,) = (
        await extractor.ainvoke(UPDATE_TEXT, existing=[("rec-1", "Contact", RECORD)])
    ).records
    assert [(p.op, p.path) for p in record.patches] == [("add", "/notes/-")]
    # they come from the input alone: Clairic is never asked, with or without the
    # record, and the field is not cross-checked
    for kind in ("extract", "crosscheck"):
        assert all("notes" not in f for f in asked_fields(world, kind))
    assert record.fields["notes"].crosscheck_agree is True


async def test_a_near_duplicate_of_an_existing_note_is_rejected() -> None:
    world = update_script()
    extractor = make_extractor([One(Contact)], world)
    result = await extractor.ainvoke(
        UPDATE_TEXT, existing=[("rec-1", "Contact", RECORD)]
    )
    (record,) = result.records
    assert record.after is not None
    assert [(p.op, p.path) for p in record.patches] == [("add", "/notes/-")]
    assert record.after["notes"][-1] == "She wants to meet in May."
    assert "She prefers email to phone." not in record.after["notes"]
    rejected = {(r.path, r.reason) for r in record.rejected}
    assert ("/notes/-", "duplicate") in rejected
    assert len(world.calls("list_dedup")) == 1


async def test_an_identical_sentence_is_no_change() -> None:
    world = update_script()
    world.sentences = {"Prefers email over phone": 0.9}
    extractor = make_extractor([One(Contact)], world)
    result = await extractor.ainvoke(
        "Dana Ruiz called. Prefers email over phone.",
        existing=[("rec-1", "Contact", RECORD)],
    )
    (record,) = result.records
    assert record.patches == ()
    assert record.after == RECORD
