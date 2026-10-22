"""M10c: what the first live runs showed, fixed (see docs/DESIGN.md).

* notes that only repeat what the update writes elsewhere are dropped (novelty),
* the notes gate judges against the record after the update's other operations,
* attribution names the field of a list element and asks for requested numbers,
* a single record's Clef label carries what it is called,
* decisions are asked one per request.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import Script, ca, kinds, make_extractor
from trustcall_system_one import Limits, One, VerifyConfig, s1_field
from trustcall_system_one._gates import record_after
from trustcall_system_one.testing import FakeTransport, FakeWorld


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    phone: Optional[str] = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


RECORD: dict[str, Any] = {"name": "Dana Ruiz", "phone": "+1 415 555 0134", "notes": []}
TEXT = (
    "Dana Ruiz called today. Her mobile is now +49 30 5555 0198. "
    "She wants to meet in May."
)


def script() -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana Ruiz", find="Dana Ruiz"),
        "phone": ca("text", "+49 30 5555 0198", find="+49 30 5555 0198"),
        "notes": ca("list", ["called"], find_all=["called"], confidence=1e-9),
    }
    world.reask = {"notes": world.clairic["notes"]}
    world.sentences = {"mobile": 0.9, "meet in May": 0.9}
    return world


async def update(world: Script, **options: Any) -> Any:
    extractor = make_extractor([One(Contact)], world, **options)
    result = await extractor.ainvoke(TEXT, existing=[("c-1", "Contact", RECORD)])
    (record,) = result.records
    return record


# --------------------------------------------------------------------------- #
# Fix 2: the notes novelty filter and the record after the other operations
# --------------------------------------------------------------------------- #


async def test_a_note_that_repeats_an_applied_change_is_dropped_as_not_novel() -> None:
    world = script()
    world.novelty = {"mobile": 0.1}
    record = await update(world)
    assert [(p.op, p.path) for p in record.patches] == [
        ("replace", "/phone"),
        ("add", "/notes/-"),
    ]
    assert record.after is not None
    assert record.after["notes"] == ["She wants to meet in May."]
    (dropped,) = [r for r in record.rejected if r.reason == "not_novel"]
    assert dropped.path == "/notes/-" and dropped.p == 0.1
    assert "adds nothing" in dropped.detail


async def test_novelty_and_the_notes_gate_see_the_record_after_the_other_ops() -> None:
    world = script()
    await update(world)
    assert world.calls("novelty")
    for _, payload, _ in world.calls("novelty"):
        assert payload["state"]["record"]["phone"] == "+49 30 5555 0198"
    gates = {
        c[1]["state"]["proposed_change"]["path"]: c[1]["state"]["record"]
        for c in world.calls("gate")
    }
    assert gates["/phone"]["phone"] == "+1 415 555 0134"  # the change itself: as it was
    assert gates["/notes/-"]["phone"] == "+49 30 5555 0198"  # a note: after the rest


async def test_each_sentence_gets_its_own_novelty_request() -> None:
    world = script()
    await update(world, verify=VerifyConfig(isolate_decisions=True))
    assert len(world.calls("novelty")) == 2  # "mobile" and "meet in May"
    assert all(len(c[1]["questions"]) == 1 for c in world.calls("novelty"))
    batched = script()
    await update(batched, verify=VerifyConfig(isolate_decisions=False))
    assert len(batched.calls("novelty")) == 1


async def test_note_novelty_zero_switches_the_check_off() -> None:
    world = script()
    world.novelty = {"mobile": 0.0}
    record = await update(world, verify=VerifyConfig(note_novelty=0))
    assert "novelty" not in kinds(world.transport)
    assert record.after is not None and len(record.after["notes"]) == 2


async def test_the_floor_is_the_knob() -> None:
    world = script()
    world.novelty = {"meet in May": 0.45}
    record = await update(world, verify=VerifyConfig(note_novelty=0.4))
    assert (
        record.after is not None
        and "She wants to meet in May." in (record.after["notes"])
    )
    world = script()
    world.novelty = {"meet in May": 0.45}
    record = await update(world)  # 0.5
    assert (
        record.after is not None
        and "She wants to meet in May." not in (record.after["notes"])
    )


async def test_list_strategy_elements_are_not_asked_about_novelty() -> None:
    """A list read by Clairic holds short named values, not sentences copied from
    the input, so there is nothing to repeat another field; only sentence-read
    elements are checked."""

    class Tags(BaseModel):
        name: str
        tags: list[str] = Field(default_factory=list, description="tags")

    world = Script()
    world.clairic = {
        "name": ca("text", "Dana", find="Dana"),
        "tags": ca("list", ["red", "blue"], find_all=["red", "blue"]),
    }
    extractor = make_extractor([One(Tags)], world)
    await extractor.ainvoke(
        "Dana is red and blue.",
        existing=[("t-1", "Tags", {"name": "Dana", "tags": []})],
    )
    assert "novelty" not in kinds(world.transport)


def test_record_after_applies_the_operations_that_apply() -> None:
    from trustcall_system_one._diff import Change

    before = {"name": "Bob", "phone": "1", "notes": []}
    changes = [
        Change("replace", "/phone", "2", "1", "phone"),
        Change("remove", "/missing", None, None, "missing"),  # cannot apply
    ]
    after = record_after(before, changes)
    assert after["phone"] == "2" and before["phone"] == "1"


async def test_fakeworld_answers_the_novelty_kind() -> None:
    class Note(BaseModel):
        name: str
        notes: list[str] = Field(default_factory=list, description="notes")

    text = "Dana called. Her mobile died. She moves to Rome."
    world = FakeWorld(text)
    world.entity(
        "the Note",
        schema=Note,
        mentions=["Dana"],
        fields={"name": "Dana", "notes": ["Her mobile died.", "She moves to Rome."]},
    )
    world.redundant("mobile")
    extractor = make_extractor_from_world(world)
    result = await extractor.ainvoke(
        text, existing=[("n-1", "Note", {"name": "Dana", "notes": []})]
    )
    (record,) = result.records
    assert record.after is not None
    assert record.after["notes"] == ["She moves to Rome."]
    assert any(r.reason == "not_novel" for r in record.rejected)


def make_extractor_from_world(world: FakeWorld) -> Any:
    from trustcall_system_one import create_extractor

    class Note(BaseModel):
        name: str
        notes: list[str] = Field(default_factory=list, description="notes")

    return create_extractor(
        [One(Note)],
        transport=world.transport(),
        limits=Limits(list_strategy="sentences"),
    )


# --------------------------------------------------------------------------- #
# Fix 4: attribution wording
# --------------------------------------------------------------------------- #


async def test_a_list_element_is_asked_as_one_of_the_fields_values() -> None:
    class Invoice(BaseModel):
        invoice_number: str
        contacts: list[str] = Field(default_factory=list, description="the contacts")

    text = "Invoice INV-2291. Contact: Dana Ruiz, Priya Shah."
    world = Script()
    world.clairic = {
        "invoice_number": ca("text", "INV-2291", find="INV-2291"),
        "contacts": ca(
            "list", ["Dana Ruiz", "Priya Shah"], find_all=["Dana Ruiz", "Priya Shah"]
        ),
    }
    await make_extractor([Invoice], world).ainvoke(text)
    (call,) = world.calls("attribution")
    question = call[1]["questions"]["attr.contacts.1"]
    assert question["instructions"] == (
        'According to `input`, is "Priya Shah" one of the contacts of `entity.label`? '
        "Something the input asks to include counts."
    )
    assert question["criteria"] == {"true": "Yes.", "false": "No."}


async def test_only_a_quantity_may_be_asked_for() -> None:
    class Line(BaseModel):
        name: str
        quantity: int = Field(description="number of units ordered")
        total: float = Field(description="the amount due")

    world = Script()
    world.clairic = {
        "name": ca("text", "blue mugs", find="blue mugs"),
        "quantity": ca("number", 2),
        "total": ca("number", 20),
    }
    await make_extractor([Line], world).ainvoke("Please add two more blue mugs, 20.")
    (call,) = world.calls("attribution")
    questions = call[1]["questions"]
    assert questions["attr.quantity"]["instructions"] == (
        "Does `input` ask for 2 of `entity.label` (as the number of units ordered)?"
    )
    # a total is stated, not asked for; a name is stated
    assert questions["attr.total"]["instructions"].startswith(
        "Does `input` state that the amount due of `entity.label` is"
    )
    assert questions["attr.name"]["instructions"].startswith(
        "Does `input` state that name of `entity.label` is"
    )


# --------------------------------------------------------------------------- #
# Fix 5: the label of a single record
# --------------------------------------------------------------------------- #


async def test_a_single_records_clef_label_carries_what_it_is_called() -> None:
    world = script()
    await update(world)
    states = [c[1]["state"] for c in world.calls("attribution")]
    assert states and all(
        s["entity"]["label"] == "the Contact Dana Ruiz" for s in states
    )
    # Clairic is told nothing of the kind: its instructions keep the plain label
    extract = world.calls("extract")[0][1]
    assert "Dana Ruiz's" not in str(extract["fields"]["phone"]["instructions"])


async def test_a_record_without_an_identity_is_named_by_its_short_id() -> None:
    class Reading(BaseModel):
        value: Optional[float] = None

    world = Script()
    world.clairic = {"value": ca("number", 12.5)}
    extractor = make_extractor([One(Reading)], world)
    await extractor.ainvoke("Reading: 12.5", existing=[("INV-2291", "Reading", {})])
    (call,) = world.calls("attribution")
    assert call[1]["state"]["entity"]["label"] == "the Reading INV-2291"
    opaque = Script()
    opaque.clairic = {"value": ca("number", 12.5)}
    await make_extractor([One(Reading)], opaque).ainvoke(
        "Reading: 12.5",
        existing=[("0b9f2c1e-77aa-4d3e-9c2b-1a2b3c4d5e6f", "Reading", {})],
    )
    (call,) = opaque.calls("attribution")
    assert call[1]["state"]["entity"]["label"] == "the Reading"


# --------------------------------------------------------------------------- #
# Fix 7: the identity of a new entity falls back to its mention
# --------------------------------------------------------------------------- #


class Member(BaseModel):
    """A person."""

    name: str
    city: Optional[str] = None


async def test_a_new_entity_whose_name_extracts_badly_takes_its_mention() -> None:
    from trustcall_system_one import Many, create_extractor

    text = "Emma Thompson lives in Rome. Bob Roe lives in Oslo."
    world = FakeWorld(text)
    emma = world.entity(
        "Emma Thompson",
        schema=Member,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "city": "Rome"},
    )
    world.entity(
        "Bob Roe",
        schema=Member,
        mentions=["Bob Roe"],
        fields={"name": "Bob Roe", "city": "Oslo"},
    )
    world.low_confidence(emma, "name", 0.2, persistent=True)
    extractor = create_extractor(
        [Many(Member, identity="name")], transport=world.transport()
    )
    result = await extractor.ainvoke(text)
    emma_record = next(
        r for r in result.records if r.entity and r.entity.label == "Emma Thompson"
    )
    assert emma_record.valid and emma_record.after is not None
    assert emma_record.after["name"] == "Emma Thompson"
    answer = emma_record.fields["name"]
    assert answer.span == (0, 13) and answer.rejected_reason is None
    assert answer.warnings == ("identity_from_mention: low_confidence",)
    assert [r.reason for r in emma_record.rejected] == ["low_confidence"]
    bob = next(r for r in result.records if r.entity and r.entity.label == "Bob Roe")
    assert bob.fields["name"].warnings == ()  # a good name is left alone


async def test_an_update_never_takes_its_identity_from_the_mention() -> None:
    from trustcall_system_one import Many, create_extractor

    text = "Emma Thompson lives in Rome."
    world = FakeWorld(text)
    emma = world.entity(
        "Emma Thompson",
        schema=Member,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson", "city": "Rome"},
        matches="Emma Thompson",
    )
    world.low_confidence(emma, "name", 0.2, persistent=True)
    extractor = create_extractor(
        [Many(Member, identity="name")], transport=world.transport()
    )
    result = await extractor.ainvoke(
        text, existing=[("0", "Member", {"name": "Emma Thompson", "city": "Austin"})]
    )
    (record,) = result.records
    assert "identity_from_mention" not in " ".join(record.fields["name"].warnings)


# --------------------------------------------------------------------------- #
# Fix 9: the count of names is not a missed entity
# --------------------------------------------------------------------------- #


async def test_two_names_of_one_person_are_no_count_mismatch() -> None:
    from trustcall_system_one import Many, create_extractor

    text = "Sarah Johnson got married and is now Sarah Miller."
    world = FakeWorld(text)
    world.entity(
        "Sarah",
        schema=Member,
        mentions=["Sarah Johnson", "Sarah Miller"],
        fields={"name": "Sarah Miller"},
    )
    world.count(Member, 2)  # the model counted the names
    transport = world.transport()
    extractor = create_extractor([Many(Member, identity="name")], transport=transport)
    result = await extractor.ainvoke(text)
    assert not [w for w in result.warnings if "count_" in w]
    assert [m.kind for _, _, m in transport.calls].count(
        "discover"
    ) == 1  # no broadening


async def test_a_count_the_mentions_do_not_cover_still_warns_and_broadens() -> None:
    from trustcall_system_one import Many, create_extractor

    text = "Bob Roe lives in Oslo."
    world = FakeWorld(text)
    world.entity(
        "Bob Roe", schema=Member, mentions=["Bob Roe"], fields={"name": "Bob Roe"}
    )
    world.count(Member, 3)
    transport = world.transport()
    extractor = create_extractor([Many(Member, identity="name")], transport=transport)
    result = await extractor.ainvoke(text)
    assert any(w.startswith("count_exceeds_discovered") for w in result.warnings)
    assert [m.kind for _, _, m in transport.calls].count("discover") == 2


# --------------------------------------------------------------------------- #
# C: the sentence requests fit the Clef budget, state and questions together
# --------------------------------------------------------------------------- #


class Note(BaseModel):
    name: str
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


async def _sentence_requests(text: str, budget: int) -> list[tuple[float, int]]:
    from trustcall_system_one import Limits
    from trustcall_system_one._sentences import _tokens

    world = Script()
    world.clairic = {"name": ca("text", "Dana Ruiz", find="Dana Ruiz")}
    world.sentences_default = 0.9
    extractor = make_extractor(
        [One(Note)],
        world,
        limits=Limits(list_strategy="sentences", clef_state_token_budget=budget),
    )
    await extractor.ainvoke(text)
    return [
        (_tokens(p["state"]) + _tokens(p["questions"]), len(p["questions"]))
        for _, p, _ in world.calls("sentences")
    ]


async def test_long_sentences_are_split_until_state_and_questions_fit() -> None:
    text = "Dana Ruiz called. " + " ".join(
        f"Item {i} " + " ".join(["word"] * 300) + "." for i in range(40)
    )
    budget = 6_000
    sent = await _sentence_requests(text, budget)
    assert len(sent) > 1  # more than one request was needed
    assert all(total <= budget for total, _ in sent), sent


async def test_the_estimate_for_wide_text_never_falls_below_a_token_per_character() -> (
    None
):
    from trustcall_system_one._sentences import _tokens

    wide = "你好" * 300
    assert _tokens({"x": wide}) >= 600
    assert _tokens({"x": "word " * 300}) < 600
    text = "Dana Ruiz called. " + " ".join(
        "这是一个很长的句子" * 20 + f"{i}。" for i in range(30)
    )
    budget = 4_000
    sent = await _sentence_requests(text, budget)
    assert all(total <= budget for total, _ in sent), sent


# --------------------------------------------------------------------------- #
# Nits: a time followed by a day, and the list_kind cache key
# --------------------------------------------------------------------------- #


def test_a_time_followed_by_a_day_or_month_does_not_split() -> None:
    from trustcall_system_one._sentences import split_sentences

    def texts(text: str) -> list[str]:
        return [s.text for s in split_sentences(text)]

    assert texts("We meet at 5 p.m. Monday to plan.") == [
        "We meet at 5 p.m. Monday to plan."
    ]
    assert texts("Call me at 9 a.m. March is busy.") == [
        "Call me at 9 a.m. March is busy."
    ]
    # a month that is a modal, or a day or month followed by a modal or a pronoun,
    # starts a new sentence
    assert texts("Dinner is at 5 p.m. May I suggest the fish?") == [
        "Dinner is at 5 p.m.",
        "May I suggest the fish?",
    ]
    assert texts("Breakfast is at 9 a.m. April will be warm.") == [
        "Breakfast is at 9 a.m.",
        "April will be warm.",
    ]
    # a weekday goes on whatever follows: only the month words (May, April) are
    # also verbs and names that start a sentence
    assert texts("We start at 9 a.m. Monday we rest.") == [
        "We start at 9 a.m. Monday we rest."
    ]
    assert texts("Call me at 5 p.m. Friday this week.") == [
        "Call me at 5 p.m. Friday this week."
    ]
    assert texts("Call at 5 p.m. May 3 is the date.") == [
        "Call at 5 p.m. May 3 is the date."
    ]
    # a sentence that starts with anything else still ends at the time
    assert texts("We meet at 5 p.m. Then we leave.") == [
        "We meet at 5 p.m.",
        "Then we leave.",
    ]
    assert texts("She left at 9 a.m. He stayed.") == [
        "She left at 9 a.m.",
        "He stayed.",
    ]


async def test_two_models_with_one_class_name_do_not_share_a_list_kind() -> None:
    from trustcall_system_one import CallMeta, compile_schema
    from trustcall_system_one._run import Run
    from trustcall_system_one._sentences import list_kind
    from trustcall_system_one._verify import Ctx
    from trustcall_system_one.testing import FakeTransport, choice, clef_response

    def make(module: str) -> type[BaseModel]:
        return type(
            "Item",
            (BaseModel,),
            {
                "__module__": module,
                "__qualname__": "Item",
                "__doc__": "An item.",
                "__annotations__": {"name": str, "notes": list[str]},
                "notes": Field(default_factory=list, description="things to remember"),
            },
        )

    asked: list[str] = []

    def answer(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        asked.append(meta.schema)
        return clef_response(
            {"kind": choice("statements", {"statements": 0.9, "items": 0.1})},
            model=payload["model"],
        )

    run = Run(transport=FakeTransport(lambda m, p, meta: answer(m, p, meta)))
    kinds_seen = []
    for module in ("pkg_a.models", "pkg_b.models"):
        plan = compile_schema(make(module))
        ctx = Ctx(run=run, plan=plan, text="x", label="the Item")
        kinds_seen.append(await list_kind(ctx, plan.spec(("notes",))))
    assert kinds_seen == ["statements", "statements"]
    # the same class name in two modules is two cache entries (the two requests are
    # byte-identical here, so the run's Clef memo sends one of them)
    keys = [schema for schema, _ in run.list_kinds]
    assert [(k.__module__, k.__name__) for k in keys] == [
        ("pkg_a.models", "Item"),
        ("pkg_b.models", "Item"),
    ]
    assert all(isinstance(k, type) for k in keys)  # keyed by the class itself
    assert len(asked) == 1
    # and the same model asks once, however many entities have the field
    again = Ctx(run=run, plan=plan, text="y", label="the Item")
    await list_kind(again, plan.spec(("notes",)))
    assert len(run.list_kinds) == 2


# --------------------------------------------------------------------------- #
# M10c review: the count is covered by DISTINCT mentions only
# --------------------------------------------------------------------------- #


async def _count_case(
    text: str, entities: list[tuple[str, list[Any], list[str], dict[str, Any]]], n: int
) -> tuple[Any, int]:
    from trustcall_system_one import Many, create_extractor

    world = FakeWorld(text)
    for label, mentions, descriptive, fields in entities:
        world.entity(
            label,
            schema=Member,
            mentions=mentions,
            descriptive=descriptive,
            fields=fields,
        )
    world.count(Member, n)
    transport = world.transport()
    extractor = create_extractor(
        [Many(Member, identity="name")], transport=transport, enable_inserts=True
    )
    result = await extractor.ainvoke(text)
    return result, [m.kind for _, _, m in transport.calls].count("discover")


async def test_a_name_said_twice_does_not_hide_a_missed_descriptive_mention() -> None:
    result, discovers = await _count_case(
        "Sarah called. Sarah and my cousin went hiking in Oslo.",
        [
            ("Sarah", ["Sarah", ("Sarah", 1)], [], {"name": "Sarah"}),
            ("my cousin", [], ["my cousin"], {"name": "my cousin"}),
        ],
        2,
    )
    assert discovers == 2  # the broadened discovery ran
    assert len(result.records) + len(result.skipped) == 2


async def test_a_name_said_once_with_a_cousin_still_broadens() -> None:
    _, discovers = await _count_case(
        "Sarah and my cousin went hiking in Oslo.",
        [
            ("Sarah", ["Sarah"], [], {"name": "Sarah"}),
            ("my cousin", [], ["my cousin"], {"name": "my cousin"}),
        ],
        2,
    )
    assert discovers == 2


async def test_two_distinct_names_of_one_person_are_still_covered() -> None:
    result, discovers = await _count_case(
        "Sarah Johnson got married and is now Sarah Miller.",
        [("Sarah", ["Sarah Johnson", "Sarah Miller"], [], {"name": "Sarah Miller"})],
        2,
    )
    assert discovers == 1 and not [w for w in result.warnings if "count_" in w]


# --------------------------------------------------------------------------- #
# M10c review B2: the identity fallback only names a proper noun
# --------------------------------------------------------------------------- #


class Subscriber(BaseModel):
    """A newsletter subscriber."""

    email: str
    city: Optional[str] = None


class Ticket(BaseModel):
    """A support ticket."""

    id: str
    summary: Optional[str] = None


class Account(BaseModel):
    """A customer account."""

    login: str = Field(
        json_schema_extra=s1_field(identity=True), description="The account's name"
    )
    city: Optional[str] = None


class Machine(BaseModel):
    """A machine."""

    serial: str = Field(
        json_schema_extra=s1_field(identity=True), description="Serial number"
    )
    city: Optional[str] = None


async def _fallback(
    schema: type[BaseModel],
    text: str,
    label: str,
    mention: str,
    fields: dict[str, Any],
) -> Any:
    from trustcall_system_one import Many, create_extractor

    world = FakeWorld(text)
    world.entity(label, schema=schema, mentions=[mention], fields=fields)
    extractor = create_extractor(
        [Many(schema)], transport=world.transport(), enable_inserts=True
    )
    return await extractor.ainvoke(text)


@pytest.mark.parametrize(
    ("schema", "text", "label", "fields"),
    [
        (Subscriber, "Dana Ruiz from Lyon signed up.", "Dana Ruiz", {"city": "Lyon"}),
        (
            Ticket,
            "The printer ticket is about a paper jam.",
            "The printer ticket",
            {"summary": "paper jam"},
        ),
        (
            Machine,
            "Dana Ruiz serviced it in Lyon.",
            "Dana Ruiz",
            {"city": "Lyon"},
        ),
    ],
)
async def test_the_fallback_never_invents_an_email_an_id_or_a_serial(
    schema: type[BaseModel], text: str, label: str, fields: dict[str, Any]
) -> None:
    result = await _fallback(schema, text, label, label, fields)
    assert not any(r.valid for r in result.records)  # nothing invented
    assert not [w for w in result.warnings if "identity_from_mention" in w]
    assert [s.reason for s in result.skipped] == ["validation_failed"]


@pytest.mark.parametrize("mention", ["my cousin", "the client", "the new hire"])
async def test_a_descriptive_reference_is_not_a_name(mention: str) -> None:
    text = f"I met {mention} in Oslo."
    result = await _fallback(Member, text, mention, mention, {"city": "Oslo"})
    assert not any(r.valid for r in result.records)
    assert not [w for w in result.warnings if "identity_from_mention" in w]


async def test_a_name_like_marked_identity_takes_a_proper_noun_and_warns() -> None:
    result = await _fallback(
        Account,
        "Dana Ruiz from Lyon opened an account.",
        "Dana Ruiz",
        "Dana Ruiz",
        {"city": "Lyon"},
    )
    (record,) = result.records
    assert record.after == {"login": "Dana Ruiz", "city": "Lyon"}
    assert any("identity_from_mention" in w for w in result.warnings)


async def test_the_fallback_is_a_run_warning_for_a_plain_name_field() -> None:
    result = await _fallback(
        Member,
        "Dana Ruiz from Lyon signed up.",
        "Dana Ruiz",
        "Dana Ruiz",
        {"city": "Lyon"},
    )
    (record,) = result.records
    assert record.after == {"name": "Dana Ruiz", "city": "Lyon"}
    assert record.fields["name"].warnings == ("identity_from_mention: no value",)
    assert [w for w in result.warnings if "identity_from_mention: no value" in w]


# --------------------------------------------------------------------------- #
# M10c review S3: novelty is judged against the accepted changes only
# --------------------------------------------------------------------------- #


async def test_a_note_is_judged_against_the_record_without_a_refused_change() -> None:
    world = script()
    world.gate = {"/phone": 0.1}  # the phone change is refused by its gate
    world.novelty = {"mobile": 0.9}  # and the note about it is news: it was not written
    record = await update(world)
    assert [r.reason for r in record.rejected if r.path == "/phone"] == ["gate_noul"]
    assert record.after is not None
    assert record.after["phone"] == "+1 415 555 0134"
    assert "Her mobile is now +49 30 5555 0198." in record.after["notes"]
    for _, payload, _ in world.calls("novelty"):
        assert payload["state"]["record"]["phone"] == "+1 415 555 0134"
    gates = {
        c[1]["state"]["proposed_change"]["path"]: c[1]["state"]["record"]
        for c in world.calls("gate")
    }
    assert gates["/notes/-"]["phone"] == "+1 415 555 0134"


async def test_the_other_fields_gates_run_before_the_novelty_check() -> None:
    world = script()
    await update(world)
    order = [m.kind for _, _, m in world.transport.calls]
    last_other = max(
        i
        for i, (_, payload, m) in enumerate(world.transport.calls)
        if m.kind == "gate" and payload["state"]["proposed_change"]["path"] == "/phone"
    )
    assert last_other < order.index("novelty")


class Employee(BaseModel):
    """An employee."""

    name: str
    city: Optional[str] = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


async def test_a_note_next_to_an_applied_city_keeps_what_the_city_does_not_say() -> (
    None
):
    text = "Dana Ruiz works here. She moved to Berlin to lead the Ops team."
    world = FakeWorld(text)
    world.entity(
        "Dana Ruiz",
        schema=Employee,
        mentions=["Dana Ruiz"],
        fields={
            "name": "Dana Ruiz",
            "city": "Berlin",
            "notes": ["She moved to Berlin to lead the Ops team."],
        },
    )
    from trustcall_system_one import create_extractor

    extractor = create_extractor(
        [One(Employee)],
        transport=world.transport(),
        limits=Limits(list_strategy="sentences"),
    )
    result = await extractor.ainvoke(
        text,
        existing=[
            ("e-1", "Employee", {"name": "Dana Ruiz", "city": "Oslo", "notes": []})
        ],
    )
    (record,) = result.records
    assert record.after is not None
    assert record.after["city"] == "Berlin"
    assert record.after["notes"] == ["She moved to Berlin to lead the Ops team."]
    assert not [r for r in record.rejected if r.reason == "not_novel"]


async def test_fakeworld_novelty_depends_on_what_the_record_holds() -> None:
    from trustcall_system_one import create_extractor

    text = "Dana Ruiz works here. She moved to Berlin."
    world = FakeWorld(text)
    world.entity(
        "Dana Ruiz",
        schema=Employee,
        mentions=["Dana Ruiz"],
        fields={
            "name": "Dana Ruiz",
            "city": "Berlin",
            "notes": ["She moved to Berlin."],
        },
    )
    extractor = create_extractor(
        [One(Employee)],
        transport=world.transport(),
        limits=Limits(list_strategy="sentences"),
    )
    result = await extractor.ainvoke(
        text,
        existing=[
            ("e-1", "Employee", {"name": "Dana Ruiz", "city": "Oslo", "notes": []})
        ],
    )
    (record,) = result.records
    # the city is written, and nothing else is in the sentence: not news
    assert record.after is not None and record.after["notes"] == []
    assert [r.reason for r in record.rejected if r.path == "/notes/-"] == ["not_novel"]


# --------------------------------------------------------------------------- #
# M10c review S4: only an ordered or requested quantity may be asked for
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "description", "asked"),
    [
        ("quantity", "number of units ordered", True),
        ("quantity", None, True),
        ("qty", "how many", True),
        ("units", "number of units to order", True),
        ("copies", "number of copies wanted", True),
        ("items", "number of items requested", True),
        ("amount", "the number of units the customer wants to buy", True),
        ("unit_price", "price per unit in EUR", False),
        ("unit_cost", "Unit cost", False),
        ("invoice_number", "The number of the invoice", False),
        ("pages", "page count", False),
        ("sold", "copies sold to date", False),
        ("shipped", "total units shipped so far", False),
        ("quantity", "total units shipped so far", False),
        ("total", "the amount due", False),
        ("size", "the number of seats", False),
        ("tickets", "how many tickets", True),
        ("seats", "number of seats to book", True),
        ("guests", "number of guests", True),
        ("capacity", "number of seats in the hall", False),
        # M10e: bare count words are requested quantities on a line of an order
        ("units", None, True),
        ("count", None, True),
        ("pieces", None, True),
        ("boxes", None, True),
        ("num_items", None, True),
        ("nr_of_units", None, True),
        ("qtyOrdered", None, True),
        ("quantityOrdered", None, True),
        ("n", None, False),
        ("amount", None, False),
        ("page_count", None, False),
        # and a count that is only stated is not, however it is worded
        ("age", "how many years old the person is", False),
        ("employees", "how many employees the company has", False),
        ("children", "how many children she has", False),
        ("stars", "how many stars the review gives", False),
        ("guests", "how many guests to reserve for", True),
    ],
)
def test_only_an_ordered_quantity_is_asked_for(
    name: str, description: str | None, asked: bool
) -> None:
    from pydantic import create_model

    from trustcall_system_one._compile import compile_schema
    from trustcall_system_one._verify import requestable

    field = (int, Field(description=description) if description else Field())
    plan = compile_schema(create_model("Line", **{name: field}))  # type: ignore[call-overload]
    (spec,) = plan.specs
    assert requestable(spec, plan) is asked, (name, description)


@pytest.mark.parametrize(
    ("description", "phrase"),
    [
        ("the contacts", "one of the contacts"),
        ("List of tags", "one of the tags"),
        ("the list of the person's nicknames", "one of the nicknames"),
        ("the person's nickname", "a nickname"),
        ("things worth remembering", "one of the things worth remembering"),
        ("email address", "an email address"),
        ("contacts (order → items)", "one of the contacts (order → items)"),
        ("", "one of the things the field asks for"),
    ],
)
def test_the_element_phrase_never_doubles_a_word(description: str, phrase: str) -> None:
    from trustcall_system_one._verify import _element_phrase

    assert _element_phrase(description) == phrase


def test_the_ask_for_wording_does_not_double_the_article() -> None:
    from trustcall_system_one._verify import _attribution_question

    question = _attribution_question(
        "the number of units ordered", 2, element=False, ask_for=True
    )
    assert question.instructions == (
        "Does `input` ask for 2 of `entity.label` (as the number of units ordered)?"
    )
    element = _attribution_question("List of the tags", "red", element=True)
    assert "one of the list of" not in element.instructions
    assert 'is "red" one of the tags of `entity.label`?' in element.instructions


async def test_two_classes_with_one_module_and_name_do_not_share_a_list_kind() -> None:
    from trustcall_system_one import CallMeta, compile_schema
    from trustcall_system_one._run import Run
    from trustcall_system_one._sentences import list_kind
    from trustcall_system_one._verify import Ctx
    from trustcall_system_one.testing import FakeTransport, choice, clef_response

    def make() -> type[BaseModel]:
        class Item(BaseModel):
            notes: list[str] = Field(default_factory=list, description="notes")

        return Item

    def answer(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {"kind": choice("statements", {"items": 0.1, "statements": 0.9}, 0.9)},
            model=payload["model"],
        )

    first, second = make(), make()
    assert (first.__module__, first.__qualname__) == (
        second.__module__,
        second.__qualname__,
    )
    run = Run(transport=FakeTransport(answer))
    for model in (first, second):
        plan = compile_schema(model)
        ctx = Ctx(run=run, plan=plan, text="x", label="the Item")
        await list_kind(ctx, plan.spec(("notes",)))
    assert {k for k, _ in run.list_kinds} == {first, second}


@pytest.mark.parametrize(
    ("text", "clipped"),
    [
        ("Dana Ruiz", "Dana Ruiz"),
        ("x" * 40, "x" * 40),
        (
            "Northwind Logistics and Freight Forwarding GmbH",
            "Northwind Logistics and Freight…",
        ),
        (
            "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj",
            "aaaa bbbb cccc dddd eeee ffff gggg hhhh…",
        ),
        (
            "Alexander the Great of Macedon and Persia, King",
            "Alexander the Great of Macedon and…",
        ),
        ("x" * 60, "x" * 39 + "…"),  # one word longer than the limit: cut inside it
    ],
)
def test_a_record_label_is_cut_at_a_word_boundary(text: str, clipped: str) -> None:
    from trustcall_system_one._entity import _clip_label

    out = _clip_label(text)
    assert len(out) <= 40
    assert out == clipped


# --------------------------------------------------------------------------- #
# M10c round 2 (live 5): qualify and presence name the KIND of thing
# --------------------------------------------------------------------------- #

PEOPLE = {"Emma", "Maria", "Jon", "Olivia Davis", "Priya Raman", "Bob Roe"}
ANIMALS = {"Sunny", "Rex"}


def _terse_clef(model: str, payload: dict[str, Any], meta: Any) -> Any:
    """A Clef that, as live, doubts a bare name is a "Member" but not a "person"."""
    from trustcall_system_one.testing import clef_response, noul

    answers = {}
    for qid, question in payload["questions"].items():
        text = question["instructions"]
        if meta.kind == "presence":
            answers[qid] = noul(0.9 if "any person" in text else 0.21)
            continue
        label = payload["state"]["entities"][qid.split(".", 1)[1]]["label"]
        if label in ANIMALS:
            answers[qid] = noul(0.02)
        elif label in PEOPLE:
            answers[qid] = noul(0.95 if "a person" in text else 0.2)
        else:
            answers[qid] = noul(0.5)
    return clef_response(answers, model=payload["model"])


async def _people(
    text: str,
    people: list[tuple[str, dict[str, Any]]],
    animals: list[str] = (),  # type: ignore[assignment]
    existing: Any = None,
    matches: dict[str, str] | None = None,
) -> Any:
    from trustcall_system_one import Many, create_extractor
    from trustcall_system_one.testing import route

    world = FakeWorld(text)
    for name, fields in people:
        world.entity(
            name,
            schema=Member,
            mentions=[name],
            fields={"name": name, **fields},
            matches=(matches or {}).get(name),
        )
    for name in animals:
        world.non_entity(name, schema=Member, mentions=[name])
    transport = FakeTransport(
        route(default=world.handler, qualify=_terse_clef, presence=_terse_clef)
    )
    extractor = create_extractor(
        [Many(Member, identity="name")], transport=transport, enable_inserts=True
    )
    return await extractor.ainvoke(text, existing=existing)


async def test_a_bare_first_name_is_a_person_for_a_terse_schema_title() -> None:
    """short_form: "Is Emma a Member?" scored 0.24 live."""
    result = await _people(
        "Emma moved to Denver.",
        [("Emma", {"city": "Denver"})],
        existing=[("0", "Member", {"name": "Emma Thompson", "city": "Austin"})],
        matches={"Emma": "Emma Thompson"},
    )
    (record,) = result.records
    assert record.kind == "update" and record.json_doc_id == "0"
    assert [(p.op, p.path, p.value) for p in record.patches] == [
        ("replace", "/city", "Denver")
    ]
    assert result.skipped == ()


async def test_two_bare_first_names_both_qualify() -> None:
    """bleed_update: Maria 0.189 and Jon 0.226 live, so nothing was processed."""
    result = await _people(
        "Maria lives in Lisbon. Jon lives in Oslo.",
        [("Maria", {"city": "Lisbon"}), ("Jon", {"city": "Oslo"})],
    )
    assert sorted(r.entity.label for r in result.records if r.entity) == [
        "Jon",
        "Maria",
    ]
    assert result.skipped == ()


async def test_people_in_a_story_qualify_and_the_dog_does_not() -> None:
    """notes_people_insert: presence 0.21 and qualify 0.11 and 0.02 live; Sunny
    (the README's dog) is still refused."""
    result = await _people(
        "Olivia Davis is a designer. Priya Raman is her flatmate. They walk Sunny.",
        [("Olivia Davis", {}), ("Priya Raman", {})],
        animals=["Sunny"],
    )
    assert sorted(r.entity.label for r in result.records if r.entity) == [
        "Olivia Davis",
        "Priya Raman",
    ]
    assert [(s.reason, s.entity.label) for s in result.skipped] == [
        ("not_qualified", "Sunny")
    ]


async def test_the_presence_question_names_the_kind_too() -> None:
    from trustcall_system_one import Many, create_extractor
    from trustcall_system_one.testing import FakeTransport, route

    text = "Emma moved to Denver."
    world = FakeWorld(text)
    world.entity("Emma", schema=Member, mentions=["Emma"], fields={"name": "Emma"})
    transport = FakeTransport(route(default=world.handler))
    await create_extractor(
        [Many(Member, identity="name")], transport=transport, enable_inserts=True
    ).ainvoke(text)
    asked = [p for _, p, m in transport.calls if m.kind == "presence"]
    assert asked[0]["questions"]["presence"]["instructions"] == (
        "Does `input` mention any person?"
    )


async def test_many_kind_names_the_kind_and_the_knob_moves_the_floor() -> None:
    from trustcall_system_one import DiscoveryConfig, Many, create_extractor
    from trustcall_system_one.testing import FakeTransport, clef_response, noul, route

    text = "Emma moved to Denver."
    world = FakeWorld(text)
    world.entity("Emma", schema=Member, mentions=["Emma"], fields={"name": "Emma"})

    def seen_kind(model: str, payload: dict[str, Any], meta: Any) -> Any:
        return clef_response(
            {qid: noul(0.45) for qid in payload["questions"]}, model=payload["model"]
        )

    for floor, expected in ((0.4, 1), (0.5, 0)):
        transport = FakeTransport(route(default=world.handler, qualify=seen_kind))
        result = await create_extractor(
            [Many(Member, identity="name", kind="human being")],
            transport=transport,
            enable_inserts=True,
            discovery=DiscoveryConfig(qualify_min=floor),
        ).ainvoke(text)
        assert len(result.records) == expected
        ((_, payload, _),) = [c for c in transport.calls if c[2].kind == "qualify"]
        assert payload["questions"]["qualify.e0"]["instructions"] == (
            "Is `entities.e0.label` a human being?"
        )


# --------------------------------------------------------------------------- #
# M10d: the pairwise confirmation, through the extractor
# --------------------------------------------------------------------------- #


def _yes(p: float) -> Any:
    from trustcall_system_one.testing import clef_response, noul

    def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
        return clef_response(
            {q: noul(p) for q in payload["questions"]}, model=payload["model"]
        )

    return handler


async def _confirm_case(
    text: str,
    mentions: list[str],
    fields: dict[str, Any],
    *,
    odds: dict[str, float] | None,
    confidence: float | None = None,
    confirm: float = 0.75,
    inserts: bool = True,
) -> Any:
    """A new person ("matches=" left out) next to the record Emma Thompson, and a
    gullible pairwise Clef that says ``confirm`` whatever the truth."""
    from trustcall_system_one import Many, create_extractor

    world = FakeWorld(text)
    world.entity(
        mentions[0],
        schema=Member,
        mentions=mentions,
        fields=fields,
        match_odds=odds,
        match_confidence=confidence,
        confirm=confirm,
    )
    transport = world.transport()
    result = await create_extractor(
        [Many(Member, identity="name")], transport=transport, enable_inserts=inserts
    ).ainvoke(
        text,
        existing=[("0", "Member", {"name": "Emma Thompson", "city": "Austin"})],
    )
    asked = [m.kind for _, _, m in transport.calls].count("match_confirm")
    return result, asked


UNSURE = {"Emma Thompson": 0.17, "__new__": 0.83}


async def test_an_unsure_short_form_is_matched_by_the_confirmation() -> None:
    result, asked = await _confirm_case(
        "Emma just moved to Denver.",
        ["Emma"],
        {"name": "Emma", "city": "Denver"},
        odds=UNSURE,
        confidence=0.4377,
        confirm=0.85,
    )
    (record,) = result.records
    assert asked == 1 and record.kind == "update" and record.json_doc_id == "0"
    assert [(p.op, p.path, p.value) for p in record.patches] == [
        ("replace", "/city", "Denver")
    ]
    assert any(w.startswith("match_confirmed") for w in result.warnings)


async def test_a_sure_new_hire_is_not_written_onto_the_record_of_a_gullible_clef() -> (
    None
):
    """M10d review: "A new hire, Emma, joined the Oslo office" with a pairwise 0.75."""
    result, asked = await _confirm_case(
        "A new hire, Emma, joined the Oslo office today.",
        ["Emma"],
        {"name": "Emma", "city": "Oslo"},
        odds=None,  # the choice says __new__ at 0.92 and is sure
    )
    (record,) = result.records
    assert asked == 0
    assert record.kind == "insert" and record.json_doc_id is None
    assert record.after is not None and record.after["city"] == "Oslo"
    assert result.untouched == ("0",)


async def test_a_full_name_that_is_not_the_record_is_a_new_person_not_a_conflict() -> (
    None
):
    """M10d review: "Emma Davis, my new neighbour" was lost to `identity_conflict`."""
    text = "Emma Davis, my new neighbour, moved in next door in Oslo. Emma has a cat."
    mentions: list[Any] = ["Emma Davis", ("Emma", 1)]
    fields = {"name": "Emma Davis", "city": "Oslo"}
    result, asked = await _confirm_case(text, mentions, fields, odds=None)
    (record,) = result.records
    assert asked == 0 and record.kind == "insert"
    assert record.after is not None and record.after["name"] == "Emma Davis"
    # even when the choice is unsure, a full name that contradicts the record is
    # never confirmed onto it: nothing is written to Emma Thompson
    result, asked = await _confirm_case(
        text, mentions, fields, odds=UNSURE, confidence=0.3
    )
    assert asked == 0 and not [p for r in result.records for p in r.patches]


async def test_a_confirmed_match_whose_name_conflicts_falls_back_to_new() -> None:
    """The pairwise check said yes ("Emma"), the extraction found "Emma Davis": the
    entity is treated as new (an insert) and never dropped."""
    text = "Emma moved to Oslo. Her surname is Davis: Emma Davis."
    result, asked = await _confirm_case(
        text,
        ["Emma"],
        {"name": "Emma Davis", "city": "Oslo"},
        odds=UNSURE,
        confidence=0.3,
        confirm=0.9,
    )
    assert asked == 1
    (record,) = result.records
    assert record.kind == "insert" and record.after is not None
    assert record.after["name"] == "Emma Davis"
    assert not [p for r in result.records for p in r.patches]  # Emma Thompson untouched
    assert any(w.startswith("match_confirmed_reverted") for w in result.warnings)
    assert result.skipped == ()


async def test_a_conflicting_confirmed_match_with_inserts_off_is_a_visible_skip() -> (
    None
):
    text = "Emma moved to Oslo. Her surname is Davis: Emma Davis."
    result, _ = await _confirm_case(
        text,
        ["Emma"],
        {"name": "Emma Davis", "city": "Oslo"},
        odds=UNSURE,
        confidence=0.3,
        confirm=0.9,
        inserts=False,
    )
    assert result.records == ()
    assert [s.reason for s in result.skipped] == ["inserts_disabled"]


# --------------------------------------------------------------------------- #
# M10d review: the count is covered by distinct names of QUALIFIED entities
# --------------------------------------------------------------------------- #


async def test_a_full_name_then_its_first_name_is_one_name_for_the_count() -> None:
    result, discovers = await _count_case(
        "Sarah Johnson called. Sarah and my cousin went hiking in Oslo.",
        [
            (
                "Sarah Johnson",
                ["Sarah Johnson", "Sarah"],
                [],
                {"name": "Sarah Johnson"},
            ),
            ("my cousin", [], ["my cousin"], {"name": "my cousin"}),
        ],
        2,
    )
    assert discovers == 2  # the broadened discovery ran
    assert len(result.records) + len(result.skipped) == 2


async def test_a_possessive_and_a_case_variant_are_the_same_name_too() -> None:
    _, discovers = await _count_case(
        "Sarah's bike broke. SARAH and my cousin went hiking in Oslo.",
        [
            ("Sarah", ["Sarah's", "SARAH"], [], {"name": "Sarah"}),
            ("my cousin", [], ["my cousin"], {"name": "my cousin"}),
        ],
        2,
    )
    assert discovers == 2


async def test_a_refused_non_person_does_not_cover_the_count() -> None:
    from trustcall_system_one import Many, create_extractor

    text = "Sarah walked Sunny with my cousin in Oslo."
    world = FakeWorld(text)
    world.entity("Sarah", schema=Member, mentions=["Sarah"], fields={"name": "Sarah"})
    world.entity(
        "my cousin",
        schema=Member,
        mentions=[],
        descriptive=["my cousin"],
        fields={"name": "my cousin"},
    )
    world.non_entity("Sunny", schema=Member, mentions=["Sunny"])
    world.count(Member, 2)
    transport = world.transport()
    result = await create_extractor(
        [Many(Member, identity="name")], transport=transport, enable_inserts=True
    ).ainvoke(text)
    assert [m.kind for _, _, m in transport.calls].count("discover") == 2
    assert {s.entity.label for s in result.skipped} >= {"Sunny"}


@pytest.mark.parametrize(
    ("mention", "text", "name"),
    [
        ("Dana Ruiz", "Dana Ruiz called.", True),
        ("my cousin", "I met my cousin.", False),
        ("Client", "Client called from Oslo about the invoice.", False),
        ("Manager", "Manager flew to Oslo.", False),
        ("Mom", "Mom called.", False),
        ("Cousin", "Cousin came by.", False),
        ("CEO", "CEO spoke.", False),
        ("He", "He left.", False),
        ("I", "I left.", False),
        ("Someone", "Someone in Oslo rang.", False),
        ("Everyone", "Everyone left.", False),
        # a single capital that only starts the sentence is no evidence ...
        ("Sunrise", "Sunrise came late.", False),
        # ... unless the word is written with a capital inside a sentence too
        ("Sunrise", "Sunrise came late. We met Sunrise at noon.", True),
        ("Anna", "We met Anna. Anna called.", True),
        ("my cousin Anna", "I met my cousin Anna.", True),
        ("The Printer Ticket", "The Printer Ticket is open.", False),
        ("The Hague", "We flew to The Hague.", True),
        ("The Beatles", "We saw The Beatles.", True),
    ],
)
def test_a_proper_noun_is_not_a_pronoun_a_role_or_a_sentence_initial_capital(
    mention: str, text: str, name: bool
) -> None:
    from trustcall_system_one._entity import _proper_noun

    at = text.index(mention.split()[0])
    span = (at, at + len(mention))
    assert _proper_noun(mention, text, span) is name


@pytest.mark.parametrize("mention", ["Client", "Manager", "Someone"])
async def test_the_fallback_never_invents_a_name_from_a_role_or_pronoun(
    mention: str,
) -> None:
    result = await _fallback(
        Member, f"{mention} called from Oslo.", mention, mention, {"city": "Oslo"}
    )
    assert not any(r.valid for r in result.records)
    assert not [w for w in result.warnings if "identity_from_mention" in w]


def test_a_bare_count_word_is_not_requested_outside_an_order_like_schema() -> None:
    from pydantic import create_model

    from trustcall_system_one._compile import compile_schema
    from trustcall_system_one._verify import requestable

    plan = compile_schema(create_model("Survey", units=(int, Field())))
    (spec,) = plan.specs
    assert requestable(spec, plan) is False
    assert requestable(spec) is False  # and without a plan at all


@pytest.mark.parametrize(
    ("name", "description", "relative"),
    [
        ("quantity", None, True),
        ("units", None, True),
        ("count", None, True),
        ("num_items", None, True),
        ("qty_ordered", None, True),
        ("amount", None, True),
        ("n", None, True),
        ("pieces", "pieces wanted", True),
        ("unit_price", "price per unit", False),
        ("total", "the amount due", False),
        ("order_id", "the id", False),
        ("sold", "copies sold to date", False),
        ("birth_year", None, False),
        ("age", None, False),
        ("age", "how old", False),
        ("founded", "year founded", False),
        ("totalPrice", None, False),
    ],
)
def test_a_relative_edit_is_checked_for_any_number_that_is_not_only_stated(
    name: str, description: str | None, relative: bool
) -> None:
    from pydantic import create_model

    from trustcall_system_one._compile import compile_schema
    from trustcall_system_one._verify import relative_candidate

    field = (int, Field(description=description) if description else Field())
    plan = compile_schema(create_model("Line", **{name: field}))  # type: ignore[call-overload]
    (spec,) = plan.specs
    assert relative_candidate(spec) is relative, (name, description)


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("PersonModel", "person"),
        ("HTTPRequest", "http request"),
        ("Item_v2", "item"),
        ("CustomerOrder", "customer order"),
        ("UserDTO", "user"),
        ("invoice_line_schema", "invoice line"),
        ("OrderRecordV3", "order"),
        ("Member", "member"),
        ("Data", "data"),  # nothing but a suffix keeps it
        ("PersonInfoData", "person"),
    ],
)
def test_a_class_name_is_split_cleaned_and_lower_cased_as_a_kind(
    title: str, kind: str
) -> None:
    from trustcall_system_one._collection import _title_kind

    assert _title_kind(title) == kind


async def test_the_title_fallback_is_used_cleaned_in_the_questions() -> None:
    from trustcall_system_one import Many, create_extractor

    class GadgetModel(BaseModel):
        """Something we keep."""

        name: str

    text = "The Zorb is here."
    world = FakeWorld(text)
    world.entity("Zorb", schema=GadgetModel, mentions=["Zorb"], fields={"name": "Zorb"})
    transport = FakeTransport(world.handler)
    await create_extractor(
        [Many(GadgetModel, identity="name")], transport=transport, enable_inserts=True
    ).ainvoke(text)
    ((_, payload, _),) = [c for c in transport.calls if c[2].kind == "qualify"]
    assert payload["questions"]["qualify.e0"]["instructions"] == (
        "Is `entities.e0.label` a gadget?"
    )


# --------------------------------------------------------------------------- #
# M10f: the new hire is inserted, named by the mention (live 8)
# --------------------------------------------------------------------------- #


def _doubts_the_name(name_p: float) -> Any:
    """A Clef that doubts the extracted name ("does the input state Emma's full
    name?") the way Experiential did: null check 0.02, attribution 0.45."""
    from trustcall_system_one.testing import clef_response, noul

    def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
        answers = {
            qid: noul(name_p if qid in ("attr.name", "nullcheck.name") else 0.95)
            for qid in payload["questions"]
        }
        return clef_response(answers, model=payload["model"])

    return handler


async def test_a_name_the_clef_doubts_is_still_the_mention_of_a_new_person() -> None:
    from trustcall_system_one import Many, create_extractor
    from trustcall_system_one.testing import route

    text = "A new hire, Emma, joined the Oslo office today."
    world = FakeWorld(text)
    world.entity(
        "Emma",
        schema=Member,
        mentions=["Emma"],
        fields={"name": "Emma", "city": "Oslo"},
    )
    transport = FakeTransport(
        route(default=world.handler, attribution=_doubts_the_name(0.45))
    )
    result = await create_extractor(
        [Many(Member, identity="name")], transport=transport, enable_inserts=True
    ).ainvoke(
        text, existing=[("0", "Member", {"name": "Emma Thompson", "city": "Austin"})]
    )
    assert [m.kind for _, _, m in transport.calls].count("reask") >= 1  # the band
    (record,) = result.records
    assert record.kind == "insert" and record.valid
    assert record.after is not None and record.after["name"] == "Emma"
    assert record.after["city"] == "Oslo"
    assert result.skipped == () and result.untouched == ("0",)


def test_the_fallback_covers_the_checks_that_doubt_a_stated_name() -> None:
    from trustcall_system_one._entity import _FALLBACK_REASONS

    assert {"null_check", "attribution", "attribution_missing", "p_null"} <= (
        _FALLBACK_REASONS
    )
    assert "validation" not in _FALLBACK_REASONS  # a real error is not papered over


# --------------------------------------------------------------------------- #
# M10f follow-up: the new_final band is a known risk, pinned here
# --------------------------------------------------------------------------- #


async def test_a_gullible_pairwise_in_the_new_final_band_matches_the_wrong_record() -> (
    None
):
    """KNOWN RISK, pinned on purpose: `__new__` at 0.85 (below `new_final` 0.9) and a
    gullible pairwise 0.9 write a NEW "Emma" onto Emma Thompson. The correct live
    matches (short_form 0.83 then 0.87, Maria 0.84 then 0.87) have the same shape, so
    no margin tells them apart. See docs/DESIGN.md (known limits)."""
    odds = {"Emma Thompson": 0.15, "__new__": 0.85}
    text = "A new hire, Emma, joined the Oslo office today."
    fields = {"name": "Emma", "city": "Oslo"}
    result, asked = await _confirm_case(
        text, ["Emma"], fields, odds=odds, confidence=0.9, confirm=0.9
    )
    (record,) = result.records
    assert asked == 1 and record.kind == "update" and record.json_doc_id == "0"


async def test_the_new_final_band_can_be_opted_out_of() -> None:
    from trustcall_system_one import Many, MatchConfig, create_extractor

    odds = {"Emma Thompson": 0.15, "__new__": 0.85}
    text = "A new hire, Emma, joined the Oslo office today."
    for match in (MatchConfig(new_final=0.7), MatchConfig(confirm_accept=0)):
        world = FakeWorld(text)
        world.entity(
            "Emma",
            schema=Member,
            mentions=["Emma"],
            fields={"name": "Emma", "city": "Oslo"},
            match_odds=odds,
            match_confidence=0.9,
            confirm=0.9,
        )
        transport = world.transport()
        result = await create_extractor(
            [Many(Member, identity="name")],
            transport=transport,
            enable_inserts=True,
            match=match,
        ).ainvoke(
            text,
            existing=[("0", "Member", {"name": "Emma Thompson", "city": "Austin"})],
        )
        (record,) = result.records
        assert record.kind == "insert", match  # a new person, Emma Thompson untouched
        assert "match_confirm" not in [m.kind for _, _, m in transport.calls]
