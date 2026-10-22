"""M10c: what the first live runs showed, fixed (see docs/DESIGN.md).

* notes that only repeat what the update writes elsewhere are dropped (novelty),
* the notes gate judges against the record after the update's other operations,
* attribution names the field of a list element and asks for requested numbers,
* a single record's Clef label carries what it is called,
* decisions are asked one per request.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Any, Optional

from pydantic import BaseModel, Field

from tests.unit.extract_helpers import Script, ca, kinds, make_extractor
from trustcall_system_one import Limits, One, VerifyConfig
from trustcall_system_one._gates import record_after
from trustcall_system_one.testing import FakeWorld


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
    assert sorted(key for key, _ in run.list_kinds) == [
        "pkg_a.models.Item",
        "pkg_b.models.Item",
    ]
    assert len(asked) == 1
    # and the same model asks once, however many entities have the field
    again = Ctx(run=run, plan=plan, text="y", label="the Item")
    await list_kind(again, plan.spec(("notes",)))
    assert len(run.list_kinds) == 2
