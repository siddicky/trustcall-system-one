"""Record-driven presence and the qualify wording (M10c).

The existing records say which entities the input is about when discovery missed a
mention or qualification refused it ("Emma just moved to Denver" against the record
"Emma Thompson"); they never make an entity twice, and they never attach a new
person to a record that only shares a first name.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Any, Optional

import pytest
from pydantic import BaseModel

from trustcall_system_one import (
    CallMeta,
    DiscoveryConfig,
    Many,
    create_extractor,
)
from trustcall_system_one._collection import _derive_kind, _locate, _present
from trustcall_system_one._compile import compile_schema
from trustcall_system_one._types import Mention
from trustcall_system_one.testing import (
    FakeTransport,
    FakeWorld,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    route,
)


class Member(BaseModel):
    """A person in an address book."""

    name: str
    city: Optional[str] = None


RECORDS = [
    ("0", "Member", {"name": "Emma Thompson", "city": "Austin"}),
    ("1", "Member", {"name": "Bob Roe", "city": "Rome"}),
]
TEXT = "Emma just moved to Denver."


def emma_world() -> FakeWorld:
    world = FakeWorld(TEXT)
    world.entity(
        "Emma",
        schema=Member,
        mentions=["Emma"],
        fields={"name": "Emma", "city": "Denver"},
        matches="Emma Thompson",
    )
    return world


def refuse_qualify(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
    """Live: "Is Emma a Member (A person in an address book)?" was 0.37."""
    return clef_response({qid: noul(0.37) for qid in payload["questions"]})


def nothing_found(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
    answer = clairic_answer("list", [], spans=[], p_null=0.99, confidence=0.5)
    return clairic_response(
        {name: answer for name in payload["fields"]}, state_text=payload["state"]
    )


def kinds_of(transport: FakeTransport) -> list[str]:
    return [meta.kind for _, _, meta in transport.calls]


async def run(transport: FakeTransport, text: str = TEXT, **options: Any) -> Any:
    extractor = create_extractor(
        [Many(Member, identity="name")], transport=transport, **options
    )
    return await extractor.ainvoke(text, existing=RECORDS)


# --------------------------------------------------------------------------- #
# Adoption
# --------------------------------------------------------------------------- #


async def test_a_refused_first_name_is_brought_back_by_its_record() -> None:
    world = emma_world()
    transport = FakeTransport(route(default=world.handler, qualify=refuse_qualify))
    result = await run(transport)
    (record,) = result.records
    assert record.json_doc_id == "0" and record.kind == "update"
    assert [(p.op, p.path, p.value) for p in record.patches] == [
        ("replace", "/city", "Denver")
    ]
    assert result.skipped == ()
    # one question about the record that no qualified entity fits, none for Bob
    asked = [c for c in transport.calls if c[2].kind == "record_presence"]
    assert [c[2].field for c in asked] == ["0"]
    assert asked[0][1]["state"]["record"]["name"] == "Emma Thompson"
    assert list(asked[0][1]["questions"]) == ["record.0"]


async def test_without_record_presence_the_refusal_stands() -> None:
    world = emma_world()
    transport = FakeTransport(route(default=world.handler, qualify=refuse_qualify))
    result = await run(transport, discovery=DiscoveryConfig(record_presence=0))
    assert result.records == ()
    assert [s.reason for s in result.skipped] == ["not_qualified"]
    assert "record_presence" not in kinds_of(transport)


async def test_a_mention_discovery_missed_is_found_in_the_text() -> None:
    world = emma_world()
    transport = FakeTransport(route(default=world.handler, discover=nothing_found))
    result = await run(transport)
    (record,) = result.records
    assert record.json_doc_id == "0"
    assert record.entity is not None
    (mention,) = record.entity.mentions
    assert (mention.text, mention.span) == ("Emma", (0, 4))
    assert [(p.op, p.path) for p in record.patches] == [("replace", "/city")]


async def test_an_entity_discovery_found_is_never_made_twice() -> None:
    world = emma_world()
    transport = FakeTransport(world.handler)  # qualifies normally
    result = await run(transport)
    (record,) = result.records
    assert record.json_doc_id == "0"
    # the record is fitted by a qualified entity: it is not even asked about
    assert "record_presence" not in kinds_of(transport)
    assert [c for c in kinds_of(transport) if c == "match"] == ["match"]


async def test_a_record_the_input_does_not_name_is_not_asked_about() -> None:
    world = emma_world()
    transport = FakeTransport(route(default=world.handler, discover=nothing_found))
    await run(transport)
    asked = [c[2].field for c in transport.calls if c[2].kind == "record_presence"]
    assert asked == ["0"]  # Bob Roe shares no word with the input


async def test_a_no_from_clef_adopts_nothing() -> None:
    world = emma_world()

    def no(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response({qid: noul(0.2) for qid in payload["questions"]})

    transport = FakeTransport(
        route(
            default=world.handler,
            qualify=refuse_qualify,
            record_presence=no,
        )
    )
    result = await run(transport)
    assert result.records == ()


async def test_too_many_records_skip_the_check() -> None:
    world = emma_world()
    transport = FakeTransport(route(default=world.handler, qualify=refuse_qualify))
    result = await run(transport, discovery=DiscoveryConfig(record_presence_max=0))
    assert result.records == () and "record_presence" not in kinds_of(transport)


async def test_a_new_person_with_a_shared_first_name_is_not_attached() -> None:
    """ "Emma Watson" is a new member: "Emma Thompson" shares only the first name."""
    text = "Emma Watson moved to Denver."
    world = FakeWorld(text)
    world.entity(
        "Emma Watson",
        schema=Member,
        mentions=["Emma Watson"],
        fields={"name": "Emma Watson", "city": "Denver"},
    )
    transport = FakeTransport(world.handler)
    result = await run(transport, text, enable_inserts=True)
    (record,) = result.records
    assert record.kind == "insert" and record.json_doc_id is None
    assert result.untouched == ("0", "1")


async def test_a_first_name_alone_is_decided_by_the_match_not_by_the_record() -> None:
    """The entity is qualified (so the record is not asked about): the match decides,
    and when Clef's top option is __new__ the record is not forced on it."""
    text = "Emma is new in town."
    world = FakeWorld(text)
    world.entity(
        "Emma", schema=Member, mentions=["Emma"], fields={"name": "Emma", "city": None}
    )  # no matches=: a new person
    transport = FakeTransport(world.handler)
    result = await run(transport, text, enable_inserts=True)
    (record,) = result.records
    assert record.kind == "insert"
    assert "record_presence" not in kinds_of(transport)


# --------------------------------------------------------------------------- #
# Helpers and wording
# --------------------------------------------------------------------------- #


def test_the_mention_is_located_by_the_whole_identity_else_its_first_word() -> None:
    text = "Please tell Emma Thompson that Emma is late."
    assert _locate(text, ["Emma Thompson"]) == Mention("Emma Thompson", (12, 25))
    assert _locate("Emma is late.", ["Emma Thompson"]) == Mention("Emma", (0, 4))
    assert _locate("Nobody here.", ["Emma Thompson"]) is None
    assert _locate("Remma is late.", ["Emma"]) is None  # whole words only


def test_a_proper_name_is_located_in_its_own_case() -> None:
    assert _locate("Tom, please pay the bill by Friday.", ["Bill Gates"]) is None
    assert _locate("Tom, ask Bill about it.", ["Bill Gates"]) == Mention(
        "Bill", (9, 13)
    )
    # a lower-case identity word ("blue mugs") is found in any case
    assert _locate("Add two Blue Mugs.", ["blue mugs"]) == Mention("Blue Mugs", (8, 17))


def test_an_input_without_case_names_a_record_in_any_case() -> None:
    # the original lower-case input of the test above, and an all-caps one
    text = "Please tell emma thompson that emma is late."
    assert _locate(text, ["Emma Thompson"]) == Mention("emma thompson", (12, 25))
    assert _locate("talked to emma today", ["Emma Thompson"]) == Mention(
        "emma", (10, 14)
    )
    assert _locate("EMMA THOMPSON CALLED", ["Emma Thompson"]) == Mention(
        "EMMA THOMPSON", (0, 13)
    )
    assert _present(["Emma"], "talked to emma today")
    assert _present(["Emma"], "EMMA CALLED")
    # a mix of cases is information again: the bill is not Bill
    assert not _present(["Bill"], "Tom, please pay the bill by Friday.")
    # and a word written in capitals in the record is found in any case
    assert _locate("Call ibm about it.", ["IBM"]) == Mention("ibm", (5, 8))


def test_short_words_and_stopwords_locate_nothing() -> None:
    assert _locate("Ed and Jo came to the office.", ["Ed Jo"]) is None
    assert _locate("The Company met.", ["The Company"]) is None
    assert _locate("Dr Ng came.", ["Dr Ng"]) is None


KINDS: list[tuple[str, str | None]] = [
    ("A customer order.", "customer order"),
    ("A person in an address book.", "person"),
    ("Someone the speaker knows or is being introduced to.", "person"),
    ("One product line on an order.", "product line"),
    ("An invoice with its number and total.", "invoice"),
    ("Invoice, paid or open.", "invoice"),
    ("A colleague named in an announcement.", "colleague"),
    ("CRM contact for sales.", "CRM contact"),
    # meta nouns say a schema is a description, not what it describes
    ("Details of a person", "person"),
    ("Information about a person.", "person"),
    ("Schema for a customer", "customer"),
    ("Data model for an invoice", "invoice"),
    ("A record of a customer", "customer"),
    ("Contact information for a person", "contact"),
    ("User profile", "user"),
    ("Customer record", "customer"),
    ("An entry in the address book", None),
    # a possessive owner, a plural, a lead-in
    ("Person's contact details", "contact"),
    ("The user's pet", "pet"),
    ("People the user knows", "person"),
    ("Customers", "customer"),
    ("This class represents a contact.", "contact"),
    ("Represents a person", "person"),
    ("Stores info on employees", "employee"),
    ("e.g. a book", "book"),
    # punctuation: blank line, parenthesis, abbreviation dots, a colon
    ("A person\n\nAttributes:\n    name: the name", "person"),
    ("A family member (parent, child, sibling)", "family member"),
    ("An item (SKU) in the cart", "item"),
    ("A U.S. state", "US state"),
    ("A pet: dog, cat or other animal", "pet"),
    # nothing usable: the title, with a warning
    ("", None),
    ("Something the user owns", None),
    ("Extracted named entity", None),
    ("Anyone mentioned", "person"),
]


@pytest.mark.parametrize(("doc", "kind"), KINDS)
def test_the_kind_of_a_schema_is_the_leading_noun_phrase_of_its_description(
    doc: str, kind: str | None
) -> None:
    model = type("Thing", (BaseModel,), {"__doc__": doc, "__annotations__": {}})
    assert _derive_kind(compile_schema(model)) == kind, doc


@pytest.mark.parametrize(
    ("noun", "expected"),
    [
        ("person", "a"),
        ("invoice", "an"),
        ("user", "a"),
        ("hour", "an"),
        ("LLM-generated summary", "an"),
        ("CEO", "a"),
        ("FAQ entry", "an"),
        ("unit", "a"),
        ("item", "an"),
    ],
)
def test_the_article_agrees_with_how_the_noun_is_said(noun: str, expected: str) -> None:
    from trustcall_system_one._coref import article

    assert article(noun) == expected


async def test_a_label_does_not_change_the_kind_and_a_fallback_warns() -> None:
    from trustcall_system_one import Many, create_extractor

    text = "Emma moved to Denver."
    world = FakeWorld(text)
    world.entity("Emma", schema=Member, mentions=["Emma"], fields={"name": "Emma"})
    transport = FakeTransport(world.handler)
    await create_extractor(
        [Many(Member, identity="name", label="address book entry")],
        transport=transport,
        enable_inserts=True,
    ).ainvoke(text)
    ((_, payload, _),) = [c for c in transport.calls if c[2].kind == "qualify"]
    assert payload["questions"]["qualify.e0"]["instructions"] == (
        "Is `entities.e0.label` a person?"  # not "an address book entry"
    )

    class Gadget(BaseModel):
        """Something we keep."""

        name: str

    world = FakeWorld("The Zorb is here.")
    world.entity("Zorb", schema=Gadget, mentions=["Zorb"], fields={"name": "Zorb"})
    result = await create_extractor(
        [Many(Gadget, identity="name")],
        transport=world.transport(),
        enable_inserts=True,
    ).ainvoke("The Zorb is here.")
    assert any(w.startswith("kind_fallback") for w in result.warnings)


async def test_the_qualify_question_has_no_context_sentence() -> None:
    world = emma_world()
    transport = FakeTransport(world.handler)
    await run(transport)
    ((_, payload, _),) = [c for c in transport.calls if c[2].kind == "qualify"]
    (question,) = payload["questions"].values()
    assert question["instructions"] == "Is `entities.e0.label` a person?"
    assert question["criteria"]["false"] == (
        "The entity is clearly something else, not a person."
    )


# --------------------------------------------------------------------------- #
# M10c review: what the text does not mean is not revived
# --------------------------------------------------------------------------- #


class Company(BaseModel):
    """A company."""

    name: str
    city: Optional[str] = None


def says(p: float) -> Any:
    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response({q: noul(p) for q in payload["questions"]})

    return handler


async def test_the_question_is_plain_and_asked_without_the_first_name_nudge() -> None:
    world = emma_world()
    transport = FakeTransport(route(default=world.handler, discover=nothing_found))
    await run(transport)
    ((_, payload, _),) = [c for c in transport.calls if c[2].kind == "record_presence"]
    assert payload["questions"]["record.0"]["instructions"] == (
        "Does `input` refer to the same Member as `record`?"
    )


async def test_a_bill_to_pay_does_not_ask_for_bill_gates() -> None:
    text = "Tom, please pay the bill by Friday."
    world = FakeWorld(text)
    world.entity(
        "Tom",
        schema=Member,
        mentions=["Tom"],
        fields={"name": "Tom", "city": None},
        matches="Tom Hardy",
    )
    seen: list[Any] = []

    def record_presence(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        seen.append(payload["state"]["record"])
        return clef_response({q: noul(0.9) for q in payload["questions"]})

    transport = FakeTransport(
        route(default=world.handler, record_presence=record_presence)
    )
    records = [
        ("0", "Member", {"name": "Bill Gates", "city": "Seattle"}),
        ("1", "Member", {"name": "Tom Hardy", "city": None}),
    ]
    extractor = create_extractor([Many(Member, identity="name")], transport=transport)
    result = await extractor.ainvoke(text, existing=records)
    assert seen == []  # "bill" is not "Bill", and Tom's record is fitted already
    assert [r.json_doc_id for r in result.records] == ["1"]


async def test_a_dog_refused_far_below_the_floor_is_not_revived() -> None:
    text = "We walked Sunny, our dog, in the park."
    world = FakeWorld(text)
    world.non_entity("Sunny", schema=Member, mentions=["Sunny"])
    transport = FakeTransport(
        route(
            default=world.handler,
            record_presence=says(0.9),
            presence=says(0.9),
        )
    )
    records = [("0", "Member", {"name": "Sunny Patel", "city": "Leeds"})]
    extractor = create_extractor(
        [Many(Member, identity="name")], transport=transport, enable_inserts=True
    )
    result = await extractor.ainvoke(text, existing=records)
    assert result.records == ()
    assert [s.reason for s in result.skipped] == ["not_qualified"]
    assert "match" not in kinds_of(transport)


async def test_a_revived_entity_keeps_the_qualify_probability_it_had() -> None:
    world = emma_world()
    transport = FakeTransport(route(default=world.handler, qualify=refuse_qualify))
    result = await run(transport)
    (record,) = result.records
    assert record.entity is not None and record.entity.qualify_p == 0.37


async def test_a_sibling_company_sharing_a_word_overlaps_no_second_mention() -> None:
    text = "Acme Labs moved to Berlin."
    world = FakeWorld(text)
    world.entity(
        "Acme Labs",
        schema=Company,
        mentions=["Acme Labs"],
        fields={"name": "Acme Labs", "city": "Berlin"},
        matches="Acme Labs",
    )
    transport = FakeTransport(
        route(default=world.handler, record_presence=says(0.6), discover=nothing_found)
    )
    records = [
        ("0", "Company", {"name": "Acme Corp", "city": "Austin"}),
        ("1", "Company", {"name": "Acme Labs", "city": "Rome"}),
    ]
    extractor = create_extractor([Many(Company, identity="name")], transport=transport)
    result = await extractor.ainvoke(text, existing=records)
    spans = [m.span for r in result.records if r.entity for m in r.entity.mentions]
    assert len(spans) == len(set(spans))  # no mention twice
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            assert a is None or b is None or a[1] <= b[0] or b[1] <= a[0]


async def test_the_record_cap_is_a_warning_not_a_silence() -> None:
    text = "Emma moved to Coast Road."
    world = FakeWorld(text)
    world.entity(
        "Emma", schema=Member, mentions=["Emma"], fields={"name": "Emma", "city": None}
    )
    transport = FakeTransport(world.handler)
    records = [
        (str(i), "Member", {"name": f"Coast Member{i}", "city": None}) for i in range(3)
    ]
    extractor = create_extractor(
        [Many(Member, identity="name")],
        transport=transport,
        discovery=DiscoveryConfig(record_presence_max=2),
    )
    result = await extractor.ainvoke(text, existing=records)
    assert "record_presence" not in kinds_of(transport)
    assert any(w.startswith("record_presence_capped") for w in result.warnings)
    # at the cap there is no warning
    transport = FakeTransport(world.handler)
    extractor = create_extractor(
        [Many(Member, identity="name")],
        transport=transport,
        discovery=DiscoveryConfig(record_presence_max=3),
    )
    result = await extractor.ainvoke(text, existing=records)
    assert not any("record_presence_capped" in w for w in result.warnings)
    assert kinds_of(transport).count("record_presence") == 3


def test_fakeworld_calls_a_record_with_blank_values_absent() -> None:
    world = emma_world()
    meta = CallMeta(kind="record_presence", schema="Member", field="0")

    def ask(record: dict[str, Any]) -> float:
        payload = {
            "model": "clef",
            "state": {"input": TEXT, "record": record},
            "questions": {"record.0": {"type": "noul", "instructions": "?"}},
        }
        response: Any = world.handler("@cf/cloudflare/clef", payload, meta)
        return float(response["answers"]["record.0"]["noul"])

    assert ask({"name": "Emma Thompson"}) > 0.9
    assert ask({"name": "", "city": "   "}) < 0.1  # nothing to compare is no fit


async def test_a_lower_case_or_all_caps_input_still_asks_about_its_record() -> None:
    for text, mention, city in (
        ("talked to emma today, she moved to denver", "emma", "denver"),
        ("EMMA THOMPSON CALLED AND MOVED TO DENVER", "EMMA THOMPSON", "DENVER"),
    ):
        world = FakeWorld(text)
        world.entity(
            mention,
            schema=Member,
            mentions=[mention],
            fields={"name": mention, "city": city},
            matches="Emma Thompson",
        )
        transport = FakeTransport(route(default=world.handler, discover=nothing_found))
        result = await run(transport, text)
        asked = [c[2].field for c in transport.calls if c[2].kind == "record_presence"]
        assert asked == ["0"], text
        assert [r.json_doc_id for r in result.records] == ["0"], text
