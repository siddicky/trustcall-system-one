"""Record-driven presence and the qualify wording (M10c).

The existing records say which entities the input is about when discovery missed a
mention or qualification refused it ("Emma just moved to Denver" against the record
"Emma Thompson"); they never make an entity twice, and they never attach a new
person to a record that only shares a first name.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Any, Optional

from pydantic import BaseModel

from trustcall_system_one import (
    CallMeta,
    DiscoveryConfig,
    Many,
    create_extractor,
)
from trustcall_system_one._collection import _locate, _type_description
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
    text = "Please tell emma thompson that Emma is late."
    assert _locate(text, ["Emma Thompson"]) == Mention("emma thompson", (12, 25))
    assert _locate("Emma is late.", ["Emma Thompson"]) == Mention("Emma", (0, 4))
    assert _locate("Nobody here.", ["Emma Thompson"]) is None
    assert _locate("Remma is late.", ["Emma"]) is None  # whole words only


def test_only_a_type_description_goes_into_the_qualify_question() -> None:
    def described(doc: str) -> str:
        model = type("M", (BaseModel,), {"__doc__": doc, "__annotations__": {}})
        return _type_description(compile_schema(model))

    assert described("A customer order.") == "A customer order"
    assert described("A person in an address book.") == ""
    assert described("Someone the speaker knows or is being introduced to.") == ""
    assert described("One product line on an order.") == ""
    assert described("An invoice with its number and total.") == ""
    assert described("Invoice, paid or open.") == ""


async def test_the_qualify_question_has_no_context_sentence() -> None:
    world = emma_world()
    transport = FakeTransport(world.handler)
    await run(transport)
    ((_, payload, _),) = [c for c in transport.calls if c[2].kind == "qualify"]
    (question,) = payload["questions"].values()
    assert question["instructions"] == "Is `entities.e0.label` a Member?"
