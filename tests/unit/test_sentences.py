"""The sentence-based reading of open-ended ``list[str]`` fields.

Splitting and scoping are pure; the strategies ("list", "sentences", "auto") run on
scripted transports and on ``FakeWorld``.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

import json
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
from trustcall_system_one.testing import FakeTransport, FakeWorld, route

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
        "Me: Nice\nFriend: Yes\n- likes tea\n2. owns a dog"
    )
    found = split_sentences(text)
    assert [s.text for s in found] == [
        "Emma got promoted",
        "Great news",
        "she earned it",
        "Nice",
        "Yes",
        "likes tea",
        "owns a dog",
    ]
    assert all(text[s.start : s.end] == s.text for s in found)
    assert [s.speaker for s in found] == [
        "Me",
        "Friend",
        "Friend",
        "Me",
        "Friend",
        None,
        None,
    ]
    assert text[slice(*found[0].label_span or (0, 0))] == "Me"


def test_a_colon_in_the_middle_is_not_a_speaker_label() -> None:
    note = "Quick note after catching up with Dana: she moved to Berlin."
    assert texts(note) == [note]
    assert texts("Meeting at 5: bring snacks.") == ["Meeting at 5: bring snacks."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Her email: a@b.co. Next sentence.", ["Her email: a@b.co.", "Next sentence."]),
        ("Plan A: go home. Plan B: stay.", ["Plan A: go home.", "Plan B: stay."]),
        ("Note: Sarah prefers tea.", ["Note: Sarah prefers tea."]),
        # each label starts one line only: not a transcript
        (
            "Sarah: I just got promoted.\nMe: Congrats!",
            ["Sarah: I just got promoted.", "Me: Congrats!"],
        ),
    ],
)
def test_a_word_and_a_colon_is_text_unless_the_label_repeats(
    text: str, expected: list[str]
) -> None:
    found = split_sentences(text)
    assert [s.text for s in found] == expected
    assert all(s.speaker is None for s in found)


def test_in_a_transcript_a_speaker_s_lines_are_her_candidates() -> None:
    text = (
        "Me: How was your week?\nSarah: I just got promoted to team lead.\n"
        "Me: Congrats!\nSarah: Thanks. I also adopted a cat."
    )
    sentences = split_sentences(text)
    assert [s.text for s in sentences if s.speaker == "Sarah"] == [
        "I just got promoted to team lead.",
        "Thanks.",
        "I also adopted a cat.",
    ]
    label = (text.index("Sarah"), text.index("Sarah") + 5)
    with_span = candidate_sentences(sentences, spans=[label], follow=0)
    assert [s.text for s in with_span] == ["I just got promoted to team lead."]
    by_text = candidate_sentences(sentences, texts=["Sarah"], follow=0)
    assert [s.text for s in by_text] == [
        "I just got promoted to team lead.",
        "Thanks.",
        "I also adopted a cat.",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("So did I. We left early.", ["So did I.", "We left early."]),
        ("She got a B. Then she retook it.", ["She got a B.", "Then she retook it."]),
        ("Take vitamin C. It helps.", ["Take vitamin C.", "It helps."]),
        ("J. R. R. Tolkien wrote it. Fine.", ["J. R. R. Tolkien wrote it.", "Fine."]),
        ("Meet J. Smith at noon. Bring tea.", ["Meet J. Smith at noon.", "Bring tea."]),
        ("He arrived at 5 p.m. Then dinner.", ["He arrived at 5 p.m.", "Then dinner."]),
        ("Call me at 5 p.m. tomorrow.", ["Call me at 5 p.m. tomorrow."]),
        (
            "Visit https://example.com/a.b?c=d. Then call.",
            ["Visit https://example.com/a.b?c=d.", "Then call."],
        ),
        (
            "Sie wohnt in Berlin。Er wohnt in Wien。",
            ["Sie wohnt in Berlin。", "Er wohnt in Wien。"],
        ),
        (
            "她住在北京。他住在上海\uff01你呢\uff1f",
            ["她住在北京。", "他住在上海\uff01", "你呢\uff1f"],
        ),
        # documented limits
        ("We met on Main St. Then lunch.", ["We met on Main St. Then lunch."]),
        (
            "I work at Acme Inc. The office is in Lyon.",
            ["I work at Acme Inc. The office is in Lyon."],
        ),
        ("she moved. her kid started school.", ["she moved. her kid started school."]),
    ],
)
def test_initials_times_urls_and_cjk(text: str, expected: list[str]) -> None:
    assert texts(text) == expected


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


TWO_EMMAS = (
    "Emma Thompson got promoted. Emma loves hiking. The bus was late. Today was long. "
    "It rained. The sea was calm. We ate. Emma Watson moved to Paris. "
    "Emma adopted a cat."
)


def test_two_people_called_emma_are_told_apart_by_their_spans() -> None:
    sentences = split_sentences(TWO_EMMAS)

    def at(text: str, n: int = 0) -> tuple[int, int]:
        i = -1
        for _ in range(n + 1):
            i = TWO_EMMAS.index(text, i + 1)
        return (i, i + len(text))

    thompson = [at("Emma Thompson"), at("Emma", 1)]
    watson = [at("Emma Watson"), at("Emma", 3)]
    # with spans the bare text "Emma" is not looked for: Watson's sentences stay out
    found = candidate_sentences(
        sentences, spans=thompson, texts=[], follow=1, others=[*thompson, *watson]
    )
    assert [s.text for s in found] == [
        "Emma Thompson got promoted.",
        "Emma loves hiking.",
        "The bus was late.",
    ]
    other = candidate_sentences(
        sentences, spans=watson, texts=[], follow=2, others=[*thompson, *watson]
    )
    assert [s.text for s in other] == [
        "Emma Watson moved to Paris.",
        "Emma adopted a cat.",
    ]


def test_the_reach_of_a_mention_stops_at_a_sentence_that_names_another_entity() -> None:
    text = (
        "Emma Thompson got promoted. She smiled. Michael Chen plays guitar. "
        "He is vegan."
    )
    sentences = split_sentences(text)
    emma = [(0, 13)]
    michael = [(text.index("Michael"), text.index("Michael") + 12)]
    found = candidate_sentences(
        sentences, spans=emma, follow=4, others=[*emma, *michael]
    )
    assert [s.text for s in found] == ["Emma Thompson got promoted.", "She smiled."]
    # without knowing the others the window runs on
    assert len(candidate_sentences(sentences, spans=emma, follow=4)) == 4


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
    world.reask = {"notes": world.clairic["notes"]}  # asked again: just as unsure
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
    world.reask = {"notes": world.clairic["notes"]}
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


# --------------------------------------------------------------------------- #
# The "auto" rule: re-ask first, sentences only for statements
# --------------------------------------------------------------------------- #

LANGUAGES_TEXT = (
    "Dana Ruiz speaks Spanish and French. She also knows some Portuguese. "
    "Call her Monday."
)


class Speaker(BaseModel):
    """A contact with languages."""

    name: str
    languages: list[str] = Field(
        default_factory=list, description="languages the person speaks"
    )


def languages_script(first: float, again: float | None) -> Script:
    world = Script()
    values = ["Spanish", "French", "Portuguese"]
    world.clairic = {
        "name": ca("text", "Dana Ruiz", find="Dana Ruiz"),
        "languages": ca("list", values, find_all=values, confidence=first, p_null=0.05),
    }
    if again is not None:
        world.reask = {
            "languages": ca(
                "list", values, find_all=values, confidence=again, p_null=0.05
            )
        }
    world.sentences = {"Spanish": 0.9, "Portuguese": 0.8}
    world.list_kinds = {"languages": "items"}
    return world


async def test_a_defined_list_clairic_is_unsure_of_stays_a_list_after_re_ask() -> None:
    world = languages_script(0.4, 0.9)
    (record,) = (
        await make_extractor([One(Speaker)], world).ainvoke(LANGUAGES_TEXT)
    ).records
    assert model_of(record).languages == ["Spanish", "French", "Portuguese"]
    assert len(world.calls("reask")) == 1
    assert world.calls("sentences") == [] and world.calls("list_kind") == []
    assert "list_strategy:auto re-ask" in record.fields["languages"].warnings


async def test_a_defined_items_list_that_stays_unsure_is_rejected_not_sentences() -> (
    None
):
    world = languages_script(0.4, 0.4)
    (record,) = (
        await make_extractor([One(Speaker)], world).ainvoke(LANGUAGES_TEXT)
    ).records
    assert model_of(record).languages == []
    assert record.fields["languages"].rejected_reason == "low_confidence"
    assert world.calls("sentences") == []
    (call,) = world.calls("list_kind")
    assert call[1]["state"]["field"]["name"] == "languages"
    assert set(call[1]["questions"]["kind"]["criteria"]) == {"items", "statements"}
    assert any("not statements" in w for w in record.fields["languages"].warnings)


async def test_notes_that_stay_unsure_after_the_re_ask_are_read_by_sentences() -> None:
    world = junk_script(1e-6)
    (record,) = (await make_extractor([One(Contact)], world).ainvoke(TEXT)).records
    assert model_of(record).notes == [
        "She wants to meet in May.",
        "Dr. Wu will follow up.",
    ]
    assert len(world.calls("reask")) == 1
    assert len(world.calls("list_kind")) == 1 and len(world.calls("sentences")) == 1


async def test_the_re_ask_can_rescue_a_list_the_first_pass_was_unsure_of() -> None:
    world = junk_script(1e-6)
    world.reask = {
        "notes": ca("list", ["Dana", "Lyon"], find_all=["Dana", "Lyon"], confidence=0.9)
    }
    (record,) = (await make_extractor([One(Contact)], world).ainvoke(TEXT)).records
    assert model_of(record).notes == ["Dana", "Lyon"]
    assert world.calls("sentences") == [] and world.calls("list_kind") == []


async def test_the_kind_of_a_field_is_asked_once_for_all_its_entities() -> None:
    text = (
        "Olivia Davis is a graphic designer. Olivia loves art and is always sketching. "
        "Priya Raman is a flatmate. Priya enjoys running every morning before work."
    )
    world = FakeWorld(text)
    for name, notes in (
        ("Olivia Davis", ["Olivia loves art and is always sketching."]),
        ("Priya Raman", ["Priya enjoys running every morning before work."]),
    ):
        entity = world.entity(
            name,
            schema=Person,
            mentions=[name],
            fields={"name": name, "notes": notes},
        )
        world.low_confidence(entity, "notes", 1e-9, persistent=True)
    extractor = create_extractor(
        [Many(Person, identity="name")], transport=world.transport()
    )
    result = await extractor.ainvoke(text)
    kinds = [m.kind for _, _, m in world.transport().calls]
    assert kinds.count("list_kind") == 1  # one question, two entities
    assert kinds.count("sentences") == 2
    by_name = {model_of(r).name: model_of(r).notes for r in result.records}
    assert by_name["Olivia Davis"] == ["Olivia loves art and is always sketching."]
    assert by_name["Priya Raman"] == ["Priya enjoys running every morning before work."]


async def test_an_explicit_sentences_strategy_skips_the_classification() -> None:
    world = Script()
    world.clairic = {"name": ca("text", "Dana Ruiz", find="Dana Ruiz")}
    world.clairic["tags"] = ca("list", ["Lyon"], find_all=["Lyon"], confidence=0.97)
    world.sentences = {"meet in May": 0.9}
    extractor = make_extractor([One(Pinned)], world)
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert model_of(record).notes == ["She wants to meet in May."]
    assert world.calls("list_kind") == [] and world.calls("reask") == []


# --------------------------------------------------------------------------- #
# Sentences meet the rest of the pipeline
# --------------------------------------------------------------------------- #


class Capped(BaseModel):
    """At most two notes."""

    name: str
    notes: list[str] = Field(
        default_factory=list, max_length=2, description="things worth remembering"
    )


class Floored(BaseModel):
    """A high minimum confidence on the notes."""

    name: str
    notes: list[str] = Field(
        default_factory=list,
        description="things worth remembering",
        json_schema_extra=s1_field(min_confidence=0.8),
    )


class AtLeastThree(BaseModel):
    """Needs three notes."""

    name: str
    notes: list[str] = Field(
        default_factory=list, min_length=3, description="things worth remembering"
    )


def sentences_script(**probabilities: float) -> Script:
    world = Script()
    world.clairic = {"name": ca("text", "Dana Ruiz", find="Dana Ruiz")}
    world.sentences = {
        "meet in May": 0.9,
        "follow up": 0.8,
        "mobile": 0.95,
        **{k.replace("_", " "): v for k, v in probabilities.items()},
    }
    return world


async def test_a_list_over_max_length_keeps_the_surest_sentences() -> None:
    world = sentences_script()
    extractor = make_extractor(
        [One(Capped)], world, limits=Limits(list_strategy="sentences")
    )
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert record.valid
    # three kept (0.9, 0.95, 0.8): the two surest stay, in input order
    assert model_of(record).notes == ["She wants to meet in May.", "Her mobile died."]
    assert world.calls("repair") == []  # Clairic's list type is never asked
    assert all(
        "notes" not in set(c[1].get("fields", ())) for c in world.transport.calls
    )
    notes = record.fields["notes"]
    assert any("kept the 2 surest of 3" in w for w in notes.warnings)


async def test_a_too_short_sentence_list_is_terminal_not_re_asked() -> None:
    world = sentences_script(mobile=0.1)
    extractor = make_extractor(
        [One(AtLeastThree)], world, limits=Limits(list_strategy="sentences")
    )
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert world.calls("repair") == []
    assert any("not re-asked of Clairic" in w for w in record.fields["notes"].warnings)


async def test_an_update_over_max_length_keeps_room_for_the_record_s_notes() -> None:
    world = sentences_script()
    world.cross = {}
    extractor = make_extractor(
        [One(Capped)], world, limits=Limits(list_strategy="sentences")
    )
    existing = [("rec-1", "Capped", {"name": "Dana Ruiz", "notes": ["Likes jazz"]})]
    (record,) = (await extractor.ainvoke(TEXT, existing=existing)).records
    assert record.valid and record.after is not None
    # the record holds one note, the limit is two: one sentence fits, the surest
    assert record.after["notes"] == ["Likes jazz", "Her mobile died."]
    assert world.calls("repair") == []


async def test_an_element_below_min_confidence_is_dropped_not_the_whole_list() -> None:
    world = sentences_script(follow_up=0.65)
    extractor = make_extractor(
        [One(Floored)], world, limits=Limits(list_strategy="sentences")
    )
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert model_of(record).notes == ["She wants to meet in May.", "Her mobile died."]
    assert record.fields["notes"].rejected_reason is None


async def test_a_list_read_by_sentences_starts_with_the_other_requests() -> None:
    import asyncio

    world = sentences_script()
    seen = asyncio.Event()
    inner = world.transport

    async def sentences(model: str, payload: Any, meta: Any) -> Any:
        seen.set()
        return inner._handler(model, payload, meta)

    async def extract(model: str, payload: Any, meta: Any) -> Any:
        # this request cannot finish until the sentences request has been sent: a
        # reader that waited for it would deadlock
        await asyncio.wait_for(seen.wait(), 2)
        return inner._handler(model, payload, meta)

    world.transport = FakeTransport(
        route(default=inner._handler, extract=extract, sentences=sentences)
    )
    extractor = make_extractor(
        [One(Capped)], world, limits=Limits(list_strategy="sentences")
    )
    (record,) = (await extractor.ainvoke(TEXT)).records
    assert record.valid


async def test_the_question_tokens_count_against_the_clef_state_budget() -> None:
    text = "Dana Ruiz called. " + " ".join(
        f"Item number {i} is entirely fine, unremarkable and of no further interest."
        for i in range(199)
    )
    budget = 12_000
    world = sentences_script()
    extractor = make_extractor(
        [One(Capped)],
        world,
        limits=Limits(list_strategy="sentences", clef_state_token_budget=budget),
    )
    await extractor.ainvoke(text)
    calls = world.calls("sentences")
    # 64 questions would not fit next to the input: the requests are smaller ones
    sizes = [len(c[1]["questions"]) for c in calls]
    assert max(sizes) < 64 and sum(sizes) == 200
    for _, payload, _ in calls:
        assert len(json.dumps(payload)) / 3.5 < budget, len(json.dumps(payload)) / 3.5
    # with room to spare they are the full 64
    roomy = sentences_script()
    huge = make_extractor(
        [One(Capped)],
        roomy,
        limits=Limits(list_strategy="sentences", clef_state_token_budget=60_000),
    )
    await huge.ainvoke(text)
    assert sorted(len(c[1]["questions"]) for c in roomy.calls("sentences")) == [
        8,
        64,
        64,
        64,
    ]


# --------------------------------------------------------------------------- #
# A yes to someone else's sentence
# --------------------------------------------------------------------------- #


async def test_a_sentence_about_the_wrong_person_is_removed_by_attribution() -> None:
    text = (
        "Emma Thompson got promoted. Michael Chen plays guitar. She loves hiking. "
        "He is thinking of going vegan."
    )
    world = FakeWorld(text)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "notes": ["Emma Thompson got promoted."]},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen"],
        fields={
            "name": "Michael Chen",
            "notes": ["Michael Chen plays guitar.", "He is thinking of going vegan."],
        },
    )
    # asked about Michael, the question says yes to Emma's pronoun sentence
    world.bleed("Michael Chen", "notes", "She loves hiking.")
    extractor = create_extractor(
        [Many(Person, identity="name")],
        transport=world.transport(),
        limits=Limits(list_strategy="sentences"),
    )
    result = await extractor.ainvoke(text)
    people = {model_of(r).name: r for r in result.records}
    michael = people["Michael Chen"]
    raw = {row["sentence"]: row["p"] for row in michael.fields["notes"].raw}
    assert raw["She loves hiking."] >= 0.9  # the sentence question said yes ...
    assert "She loves hiking." not in model_of(michael).notes  # ... attribution said no
    assert any(
        r.reason == "attribution" and "She loves hiking" in r.detail
        for r in michael.rejected
    )
    assert model_of(michael).notes == [
        "Michael Chen plays guitar.",
        "He is thinking of going vegan.",
    ]
    assert model_of(people["Emma Thompson"]).notes == ["Emma Thompson got promoted."]
