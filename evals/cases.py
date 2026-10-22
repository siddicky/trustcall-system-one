"""The eval dataset: small cases, each with a scripted world for offline runs.

A case states the input, the schemas, the existing records, the operations and
values the run must produce, and what must not happen. The ``world`` of a case is
a ``FakeWorld`` that answers like an ideal model (plus, in some cases, a fault
injected on purpose, such as a fact offered for the wrong person), so the harness
and the library's guards can be exercised without a network. Live, the same cases
are scored on what the real models do; exact values belong here, not in the
smoke tests.
"""

# ruff: noqa: E501

from typing import Any

from evals.model import Case, MustNot, Op, Rejected, Val
from evals.scenarios import (
    CONTACT_EXISTING,
    CONTACT_NOTE,
    CONVERSATION,
    EXISTING_PEOPLE,
    INVOICE_TEXT,
    ORDER_EMAIL,
    ORDER_EXISTING,
    PASSAGE,
    Bill,
    Colleague,
    Contact,
    Invoice,
    LineItem,
    Member,
    Order,
    Person,
    TotalOnly,
)
from trustcall_system_one import Many, One
from trustcall_system_one.testing import (
    FakeTransport,
    FakeWorld,
    clef_response,
    noul,
    route,
)

__all__ = ["CASES"]

# --------------------------------------------------------------------------- #
# Worlds
# --------------------------------------------------------------------------- #

EMMA_PROMOTED = "She mentioned she got promoted to Senior Marketing Manager last month."
EMMA_CLIMBING = "Oh, and did you know she's taken up rock climbing?"
MICHAEL_JOB = "He's switched jobs and is now working as a Data Scientist at a startup."
MICHAEL_VEGAN = "He's also mentioned he's thinking of going vegan."
SARAH_SPECIAL = "She's still teaching, but now she's focusing on special education."
OLIVIA_DESIGNER = "Her name is Olivia Davis, she's a 27-year-old graphic designer."
OLIVIA_ART = "Olivia loves art and is always sketching in her free time."
OLIVIA_SHELTER = "She also volunteers at the local animal shelter on weekends."


def people_world() -> FakeWorld:
    """The README conversation. Emma's remark about climbing is offered for Michael."""
    world = FakeWorld(CONVERSATION)
    world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson", "Emma"],
        fields={"name": "Emma Thompson", "notes": [EMMA_PROMOTED, EMMA_CLIMBING]},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Michael Chen",
        schema=Person,
        mentions=["Michael Chen", "Michael"],
        fields={"name": "Michael Chen", "notes": [MICHAEL_JOB, MICHAEL_VEGAN]},
        matches={"name": "Michael Chen"},
    )
    world.entity(
        "Sarah",
        schema=Person,
        mentions=["Sarah"],
        fields={"name": "Sarah", "notes": [SARAH_SPECIAL]},
        matches="Sarah Johnson",
    )
    world.entity(
        "Olivia Davis",
        schema=Person,
        mentions=["Olivia Davis", "Olivia"],
        fields={
            "name": "Olivia Davis",
            "relationship": "my cousin",
            "notes": [OLIVIA_DESIGNER, OLIVIA_ART, OLIVIA_SHELTER],
        },
    )
    world.non_entity("Sunny", schema=Person, mentions=["Sunny"])
    world.bleed("Michael Chen", "notes", EMMA_CLIMBING)
    return world


def member(name: str, **fields: Any) -> tuple[str, dict[str, Any]]:
    return name, {"name": name, "city": None, "phone": None, "notes": [], **fields}


# --------------------------------------------------------------------------- #
# Wrappers for answers a FakeWorld cannot express
# --------------------------------------------------------------------------- #


class Overridden:
    """A world with one kind of answer replaced."""

    def __init__(self, world: FakeWorld, **by_kind: Any) -> None:
        self.world = world
        self.by_kind = by_kind

    def transport(self) -> FakeTransport:
        return FakeTransport(route(default=self.world.handler, **self.by_kind))


def near_duplicates(p: float) -> Any:
    """A ``list_dedup`` answer: every pair the library asks about is the same fact."""

    def answer(model: str, payload: dict[str, Any], meta: Any) -> Any:
        return clef_response(
            {qid: noul(p) for qid in payload["questions"]}, model=payload["model"]
        )

    return answer


# --------------------------------------------------------------------------- #
# The cases
# --------------------------------------------------------------------------- #


# What is distinctive about each person of the README conversation: a note about one
# of them never sits on another's record (not as its own note, not merged into one).
PEOPLE_FACTS: dict[str, tuple[str, ...]] = {
    "0": ("promoted", "rock climbing", "puppy", "Sunny", "climbing gym"),
    "1": ("Data Scientist", "vegan", "video call", "switched jobs"),
    "2": ("special education", "middle school", "teaching", "kids"),
    "Olivia": ("graphic designer", "sketching", "animal shelter", "27-year-old"),
}


def bleed_rules(facts: dict[str, tuple[str, ...]]) -> list[MustNot]:
    """``must_not`` rules: each person's facts are never on anyone else's record.

    One rule on the operations (a note added to the wrong record) and one on the
    final document (a note that merges two people's sentences), per fact and per
    other person; they count as bleed.
    """
    rules: list[MustNot] = []
    for owner, words in facts.items():
        for other in facts:
            if other == owner:
                continue
            for word in words:
                rules.append(MustNot.value_at(other, "/notes", contains=word))
    return rules


def readme_people() -> Case:
    return Case(
        id="readme_people",
        text=CONVERSATION,
        schemas=lambda: [Many(Person, identity="name")],
        existing=EXISTING_PEOPLE,
        options={"enable_inserts": True},
        # the must-have notes ...
        expect_ops=[
            Op(
                "add",
                "/notes/-",
                contains="promoted to Senior Marketing Manager",
                who="0",
            ),
            Op("add", "/notes/-", contains="Data Scientist", who="1"),
            Op("add", "/notes/-", contains="special education", who="2"),
            Op("insert", who="Olivia"),
        ],
        # ... and the true extras a model may legitimately add (the remark about
        # climbing is Emma's, the diet is Michael's): they neither fail the case
        # nor count against precision. Bleed still fails, through must_not.
        allow_ops=[
            Op("add", "/notes/-", contains="rock climbing", who="0"),
            Op("add", "/notes/-", contains="going vegan", who="1"),
            Op("add", "/notes/-", contains="puppy", who="0"),
            Op("add", "/notes/-", contains="Sunny", who="0"),
            # Sarah's sentences, each true of her and each allowed once
            Op("add", "/notes/-", contains="doing well", who="2"),
            Op("add", "/notes/-", contains="middle school", who="2"),
            Op("add", "/notes/-", contains="passionate", who="2"),
        ],
        expect_values=[
            Val("Olivia", "/name", "Olivia Davis"),
            # required, as in the README: a new person without it is not inserted
            # (live: "cousin" scored 0.75 on attribution and the record was invalid),
            # so a run that inserts Olivia holds the speaker's own word for it
            Val("Olivia", "/relationship", contains="cousin"),
            Val("Olivia", "/notes", contains="graphic designer"),
        ],
        must_not=[
            MustNot.insert("Sunny"),
            MustNot.op("replace", "/name", who="2"),
            MustNot.op("remove"),
            MustNot.delete(),
            *bleed_rules(PEOPLE_FACTS),
        ],
        tags=("people", "update", "insert", "notes", "readme"),
        world=people_world,
        note="The trustcall README scenario. Live (05207aa): Emma skipped as ambiguous, "
        "no note ever added (Clairic's list type), Olivia invalid.",
    )


def colleague_bleed() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(PASSAGE)
        w.entity(
            "Priya Raman",
            schema=Colleague,
            mentions=["Priya Raman"],
            fields={"name": "Priya Raman", "city": "Lisbon"},
        )
        w.entity(
            "Tom Becker",
            schema=Colleague,
            mentions=["Tom Becker"],
            fields={"name": "Tom Becker", "role": "finance lead"},
        )
        w.bleed("Priya Raman", "role", "finance lead")
        w.bleed("Tom Becker", "city", "Lisbon")
        return w

    return Case(
        id="colleague_bleed",
        text=PASSAGE,
        schemas=lambda: [Many(Colleague, identity="name")],
        expect_ops=[Op("insert", who="Priya"), Op("insert", who="Tom")],
        expect_values=[
            Val("Priya", "/name", "Priya Raman"),
            Val("Priya", "/city", "Lisbon"),
            Val("Priya", "/role", None),
            Val("Tom", "/name", "Tom Becker"),
            Val("Tom", "/role", "finance lead"),
            Val("Tom", "/city", None),
        ],
        must_not=[
            MustNot.value_at("Tom", "/city", contains="Lisbon"),
            MustNot.value_at("Priya", "/role", contains="finance"),
        ],
        tags=("people", "insert", "bleed"),
        world=world,
        note="Examples/06. Live (05207aa): Tom kept 'Lisbon' (attribution 0.7736, "
        "keep threshold 0.75): a known bleed.",
    )


def computed_total() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(INVOICE_TEXT)
        w.entity(
            "the TotalOnly",
            schema=TotalOnly,
            mentions=["INV-2291"],
            fields={"total": 1080},
        )
        return w

    return Case(
        id="computed_total",
        text=INVOICE_TEXT,
        schemas=lambda: [One(TotalOnly)],
        expect_ops=[Op("insert")],
        expect_values=[Val("TotalOnly", "/total", 1080)],
        tags=("invoice", "insert", "computed"),
        world=world,
        note="4 x $250 + $80 shipping = 1080, which the text never states: arithmetic "
        "from stated figures must be accepted. Live: attribution 0.7695 (keep 0.75).",
    )


def invoice_contacts() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(INVOICE_TEXT)
        w.entity(
            "the Invoice",
            schema=Invoice,
            mentions=["INV-2291"],
            fields={
                "invoice_number": "INV-2291",
                "total": 1080,
                "contacts": ["Dana Ruiz", "Priya Shah"],
            },
        )
        return w

    return Case(
        id="invoice_contacts",
        text=INVOICE_TEXT,
        schemas=lambda: [One(Invoice)],
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Invoice", "/invoice_number", "INV-2291"),
            Val("Invoice", "/total", 1080),
            Val("Invoice", "/contacts", contains="Dana Ruiz"),
            Val("Invoice", "/contacts", contains="Priya Shah"),
        ],
        tags=("invoice", "insert", "list"),
        world=world,
        note="A contacts list of defined items: Clairic's list type is confident "
        "(live 0.9995).",
    )


def contacts_update_add() -> Case:
    existing = [
        (
            "INV-2291",
            "Invoice",
            {"invoice_number": "INV-2291", "total": 1080.0, "contacts": ["Dana Ruiz"]},
        )
    ]
    text = "Re INV-2291: please also copy Priya Shah on the reminders."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Invoice",
            schema=Invoice,
            mentions=["INV-2291"],
            fields={"invoice_number": "INV-2291", "contacts": ["Priya Shah"]},
        )
        return w

    return Case(
        id="contacts_update_add",
        text=text,
        schemas=lambda: [One(Invoice)],
        existing=existing,
        expect_ops=[Op("add", "/contacts/-", "Priya Shah", who="INV-2291")],
        must_not=[MustNot.op("remove"), MustNot.op("replace", "/contacts")],
        tags=("invoice", "update", "list"),
        world=world,
        note="A list gains one element and keeps the others.",
    )


def other_invoice_total() -> Case:
    existing = [("INV-2291", "Bill", {"invoice_number": "INV-2291", "total": 1080.0})]
    text = "Invoice INV-2300 from Fabrikam totals $500.00. Nothing has changed on INV-2291."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Bill",
            schema=Bill,
            mentions=["INV-2291"],
            fields={"invoice_number": "INV-2291"},
        )
        w.bleed(
            "the Bill", "total", 500
        )  # the other invoice's total, offered for this one
        return w

    return Case(
        id="other_invoice_total",
        text=text,
        schemas=lambda: [One(Bill)],
        existing=existing,
        expect_ops=[],
        must_not=[
            MustNot.op("replace", "/total"),
            MustNot.value_at("INV-2291", "/total", 500),
        ],
        tags=("invoice", "update", "bleed"),
        world=world,
        note="A total that belongs to another invoice must be rejected, not applied.",
    )


def total_correction() -> Case:
    existing = [("INV-2291", "Bill", {"invoice_number": "INV-2291", "total": 1080.0})]
    text = "Correction: the total of INV-2291 is $1,125.00, not $1,080.00."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Bill",
            schema=Bill,
            mentions=["INV-2291"],
            fields={"invoice_number": "INV-2291", "total": 1125},
        )
        return w

    return Case(
        id="total_correction",
        text=text,
        schemas=lambda: [One(Bill)],
        existing=existing,
        expect_ops=[Op("replace", "/total", 1125, who="INV-2291")],
        must_not=[MustNot.op("remove")],
        tags=("invoice", "update"),
        world=world,
        note="An explicit correction of a number replaces it.",
    )


def bill_insert() -> Case:
    text = "Invoice INV-2300 from Fabrikam totals $500.00, due on receipt."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Bill",
            schema=Bill,
            mentions=["INV-2300"],
            fields={"invoice_number": "INV-2300", "total": 500},
        )
        return w

    return Case(
        id="bill_insert",
        text=text,
        schemas=lambda: [One(Bill)],
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Bill", "/invoice_number", "INV-2300"),
            Val("Bill", "/total", 500),
        ],
        tags=("invoice", "insert"),
        world=world,
        note="A plain insert of two stated fields.",
    )


def rename() -> Case:
    text = "Sarah Johnson got married last spring and is now Sarah Miller."
    existing = [("0", "Member", member("Sarah Johnson", city="Austin")[1])]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Sarah Miller",
            schema=Member,
            mentions=["Sarah Johnson", "Sarah Miller"],
            fields={"name": "Sarah Miller"},
            matches="Sarah Johnson",
        )
        return w

    return Case(
        id="rename",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        existing=existing,
        expect_ops=[Op("replace", "/name", "Sarah Miller", who="0")],
        # live 5: the whole sentence is also appended as a note (attribution 0.97): it
        # is grounded, so it may happen; the rename must, and nothing may be deleted
        allow_ops=[Op("add", "/notes/-", contains="got married", who="0")],
        must_not=[MustNot.insert(), MustNot.delete(), MustNot.op("replace", "/city")],
        tags=("people", "update", "identity"),
        world=world,
        note="A rename of the identity field: coreference links both names to the record.",
    )


def short_form() -> Case:
    text = "Emma just moved to Denver."
    existing = [("0", "Member", member("Emma Thompson", city="Austin")[1])]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Emma",
            schema=Member,
            mentions=["Emma"],
            fields={"name": "Emma", "city": "Denver"},
            matches="Emma Thompson",
        )
        return w

    return Case(
        id="short_form",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        existing=existing,
        expect_ops=[Op("replace", "/city", "Denver", who="0")],
        # live 7: the sentence itself is also a grounded note (attribution 0.95)
        allow_ops=[Op("add", "/notes/-", contains="moved to Denver", who="0")],
        must_not=[MustNot.op("replace", "/name"), MustNot.insert(), MustNot.delete()],
        tags=("people", "update", "identity"),
        world=world,
        note="'Emma' is a short form of the record's 'Emma Thompson': matched, and not a rename.",
    )


def _other_emma(case_id: str, text: str, mentions: list[Any], note: str) -> Case:
    label = str(mentions[0])
    existing = [("0", "Member", member("Emma Thompson", city="Austin")[1])]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        # truly new; a gullible pairwise Clef would say yes to the record anyway
        w.entity(
            label,
            schema=Member,
            mentions=mentions,
            fields={"name": label, "city": "Oslo"},
            confirm=0.75,
        )
        return w

    return Case(
        id=case_id,
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        existing=existing,
        options={"enable_inserts": True},
        expect_ops=[Op("insert", who=label)],
        must_not=[MustNot.op(who="0"), MustNot.delete()],
        tags=("people", "insert", "identity", "negative"),
        world=world,
        note=note,
    )


def different_emma() -> Case:
    return _other_emma(
        "different_emma",
        "Emma Davis, my new neighbour, moved in next door in Oslo.",
        ["Emma Davis"],
        "A new Emma next to the existing Emma Thompson is a new person: her city "
        "is never written onto Emma Thompson, whatever the pairwise check says.",
    )


def new_hire_same_first_name() -> Case:
    return _other_emma(
        "new_hire_same_first_name",
        "A new hire, Emma, joined the Oslo office today.",
        ["Emma"],
        "A bare first name that the choice question calls new at 0.9 or more "
        "(match.new_final) is not Emma Thompson: a gullible pairwise answer does "
        "not override it. Below 0.9 the pairwise check can still decide (a known "
        "risk, see docs/DESIGN.md).",
    )


def order_line_items() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(ORDER_EMAIL)
        w.entity(
            "the Order",
            schema=Order,
            mentions=["SO-10442"],
            fields={"order_id": "SO-10442"},
        )
        w.entity(
            "tea towels",
            schema=LineItem,
            parent="Order",
            mentions=["tea towels"],
            fields={"name": "tea towels", "quantity": 1},
            matches={"name": "Tea towels"},
        )
        w.entity(
            "blue mugs",
            schema=LineItem,
            parent="Order",
            mentions=["blue mugs"],
            fields={"name": "blue mugs", "quantity": 2},
        )
        return w

    return Case(
        id="order_line_items",
        text=ORDER_EMAIL,
        schemas=lambda: [One(Order)],
        existing=[("SO-10442", "Order", ORDER_EXISTING)],
        options={"enable_inserts": True},
        expect_ops=[
            Op("replace", "/items/1/quantity", 1, who="SO-10442"),
            Op("add", "/items/-", who="SO-10442"),
        ],
        expect_values=[
            Val("SO-10442", "/items/1/quantity", 1),
            Val("SO-10442", "/items/3/name", "blue mugs"),
            Val("SO-10442", "/items/3/quantity", 2),
        ],
        must_not=[
            MustNot.op(path="/items/0"),
            MustNot.op(path="/items/2"),
            MustNot.op("remove"),
        ],
        tags=("order", "update", "nested"),
        world=world,
        note="Examples/07: one line edited, one added, the others untouched.",
    )


def order_remove_item() -> Case:
    text = "Hi Sam, it's Ines about order SO-10442: please drop the steel kettle."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Order",
            schema=Order,
            mentions=["SO-10442"],
            fields={"order_id": "SO-10442"},
        )
        w.entity(
            "steel kettle",
            schema=LineItem,
            parent="Order",
            mentions=["steel kettle"],
            fields={"name": "steel kettle"},
            matches={"name": "Steel kettle"},
        )
        w.delete("0", 0.96, parent="Order")
        return w

    return Case(
        id="order_remove_item",
        text=text,
        schemas=lambda: [One(Order)],
        existing=[("SO-10442", "Order", ORDER_EXISTING)],
        options={"enable_item_removals": True},
        expect_ops=[Op("remove", "/items/0", who="SO-10442")],
        expect_values=[Val("SO-10442", "/items/0/name", "Tea towels")],
        must_not=[
            MustNot.op(path="/items/1"),
            MustNot.op(path="/items/2"),
            MustNot.op("add"),
        ],
        tags=("order", "update", "nested", "delete"),
        world=world,
        note="Removing one line of an order removes that line only.",
    )


def relative_edit() -> Case:
    text = "Please add two more blue mugs to SO-10442."
    existing = [
        (
            "SO-10442",
            "Order",
            {
                "order_id": "SO-10442",
                "customer": "Ines Romero",
                "items": [{"name": "Blue mugs", "quantity": 3, "sku": "MUG-BLU"}],
            },
        )
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Order",
            schema=Order,
            mentions=["SO-10442"],
            fields={"order_id": "SO-10442"},
        )
        w.entity(
            "blue mugs",
            schema=LineItem,
            parent="Order",
            mentions=["blue mugs"],
            fields={"name": "blue mugs", "quantity": 2},
            matches={"name": "Blue mugs"},
        )
        return w

    return Case(
        id="relative_edit",
        text=text,
        schemas=lambda: [One(Order)],
        existing=existing,
        options={"enable_inserts": True},
        # nothing is applied: "two more" is a request to add 2 to what is there, which
        # the library does not compute, and writing the stated 2 over the 3 loses
        # data (a replace of the quantity is a violation, not an alternative)
        expect_ops=[],
        expect_rejected=[
            Rejected("relative_edit_unsupported", "/items/0/quantity", "SO-10442")
        ],
        must_not=[
            MustNot.op("replace", "/items/0/quantity", who="SO-10442"),
            MustNot.value_at("SO-10442", "/items/0/quantity", 2),
            MustNot.value_at("SO-10442", "/items/0/quantity", 5),
            MustNot.op("add", "/items/-"),
        ],
        tags=("order", "update", "nested", "relative"),
        world=world,
        note="'Two more' is a change relative to the record's 3: it is neither "
        "written as 2 (data loss) nor computed as 5; the operation is refused as "
        "relative_edit_unsupported (docs/DESIGN.md).",
    )


def relative_mixed() -> Case:
    text = (
        "Hi Sam, Ines about order SO-10442: two more of the steel kettle, and the "
        "tea towels at 5."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Order",
            schema=Order,
            mentions=["SO-10442"],
            fields={"order_id": "SO-10442"},
        )
        for name, record, quantity in (
            ("steel kettle", "Steel kettle", 2),  # the model reads "two"
            ("tea towels", "Tea towels", 5),
        ):
            w.entity(
                name,
                schema=LineItem,
                parent="Order",
                mentions=[name],
                fields={"name": name, "quantity": quantity},
                matches={"name": record},
            )
        return w

    return Case(
        id="relative_mixed",
        text=text,
        schemas=lambda: [One(Order)],
        existing=[("SO-10442", "Order", ORDER_EXISTING)],
        options={"enable_inserts": True},
        # the absolute value is written, the relative one is refused: one phrase of
        # the input about one line never decides another line
        expect_ops=[Op("replace", "/items/1/quantity", 5, who="SO-10442")],
        expect_rejected=[
            Rejected("relative_edit_unsupported", "/items/0/quantity", "SO-10442")
        ],
        must_not=[
            MustNot.op("replace", "/items/0/quantity", who="SO-10442"),
            MustNot.value_at("SO-10442", "/items/0/quantity", 2),
            MustNot.op("add", "/items/-"),
        ],
        tags=("order", "update", "nested", "relative"),
        world=world,
        note="'Two more of the steel kettle' is relative (refused) while 'the tea "
        "towels at 5' is a new value (applied): the question is about the change.",
    )


def delete_request() -> Case:
    text = "Please remove Bob Roe from my contacts."
    existing = [
        ("0", "Member", member("Alice Poe", city="Rome")[1]),
        ("1", "Member", member("Bob Roe", city="Oslo")[1]),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Bob Roe",
            schema=Member,
            mentions=["Bob Roe"],
            fields={"name": "Bob Roe"},
            matches={"name": "Bob Roe"},
        )
        w.delete("1", 0.97)
        return w

    return Case(
        id="delete_request",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        existing=existing,
        options={"enable_deletes": True},
        expect_ops=[Op("delete", who="1")],
        must_not=[MustNot.delete("0"), MustNot.insert()],
        tags=("people", "delete"),
        world=world,
        note="A removal request deletes that record and no other.",
    )


def change_not_delete() -> Case:
    text = "Bob Roe's new phone number is 555-0199."
    existing = [
        ("0", "Member", member("Alice Poe", city="Rome")[1]),
        ("1", "Member", member("Bob Roe", city="Oslo", phone="555-0100")[1]),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Bob Roe",
            schema=Member,
            mentions=["Bob Roe"],
            fields={"name": "Bob Roe", "phone": "555-0199"},
            matches={"name": "Bob Roe"},
        )
        return w

    return Case(
        id="change_not_delete",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        existing=existing,
        options={"enable_deletes": True},
        expect_ops=[Op("replace", "/phone", "555-0199", who="1")],
        must_not=[MustNot.delete(), MustNot.op("replace", "/name"), MustNot.insert()],
        tags=("people", "update", "delete"),
        world=world,
        note="A change request on a record must never delete it (the 'flagged record' "
        "guard: proposed changes make a delete a conflict).",
    )


def no_entity() -> Case:
    text = "The weather was lovely and the train was on time."
    return Case(
        id="no_entity",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[],
        must_not=[MustNot.insert(), MustNot.delete()],
        tags=("people", "negative"),
        world=lambda: FakeWorld(text),
        note="No person in the text: no record, and (offline) no extraction request.",
    )


def pet_not_person() -> Case:
    text = "My neighbour's dog Rex is five years old. Rex loves fetch."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.non_entity("Rex", schema=Member, mentions=["Rex"])
        return w

    return Case(
        id="pet_not_person",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[],
        must_not=[MustNot.insert("Rex")],
        tags=("people", "negative"),
        world=world,
        note="A dog is found as a name and refused by qualification (not a Person).",
    )


def brand_not_person() -> Case:
    text = (
        "Ford announced a recall on Monday. Jaguar unveiled a new model, and I "
        "bought an Apple laptop."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        for brand in ("Ford", "Jaguar", "Apple"):
            w.non_entity(brand, schema=Member, mentions=[brand])
        return w

    return Case(
        id="brand_not_person",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[],
        must_not=[
            MustNot.insert("Ford"),
            MustNot.insert("Jaguar"),
            MustNot.insert("Apple"),
        ],
        tags=("people", "negative", "qualify"),
        world=world,
        note="Names that are also people's names (Ford, Jaguar, Apple) used as "
        "companies or products must not become people: measures how far the "
        "qualify question (kind named, refuse at 0.4) was loosened.",
    )


def only_phone() -> Case:
    text = "Dana's new mobile is +49 30 5555 0198."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Contact",
            schema=Contact,
            mentions=["Dana"],
            fields={"name": "Dana", "phone": "+49 30 5555 0198"},
        )
        return w

    return Case(
        id="only_phone",
        text=text,
        schemas=lambda: [One(Contact)],
        existing=[("crm-1", "Contact", CONTACT_EXISTING)],
        expect_ops=[Op("replace", "/phone", "+49 30 5555 0198", who="crm-1")],
        must_not=[
            MustNot.op("remove"),
            MustNot.op("replace", "/name"),
            MustNot.op("replace", "/email"),
            MustNot.op("replace", "/city"),
        ],
        tags=("contact", "update"),
        world=world,
        note="Only what the text states changes; 'Dana' is a short form, not a rename.",
    )


def silence_is_not_deletion() -> Case:
    text = "Dana Whitfield called about the invoice."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "the Contact",
            schema=Contact,
            mentions=["Dana Whitfield"],
            fields={"name": "Dana Whitfield"},
        )
        return w

    return Case(
        id="silence_is_not_deletion",
        text=text,
        schemas=lambda: [One(Contact)],
        existing=[("crm-1", "Contact", CONTACT_EXISTING)],
        expect_ops=[],
        # live 5: "called about the invoice" is a stated fact (attribution 0.97) and
        # may be appended as a note; no field may change or go
        allow_ops=[
            Op("add", "/notes/-", contains="called about the invoice", who="crm-1")
        ],
        must_not=[MustNot.op("remove"), MustNot.op("replace")],
        tags=("contact", "update", "negative"),
        world=world,
        note="A text that says nothing about the other fields changes none of them.",
    )


def bleed_update() -> Case:
    text = "Maria is moving to Lisbon next month. Her colleague Jon stays in Oslo, as before."
    existing = [
        ("0", "Member", member("Maria Costa", city="Porto")[1]),
        ("1", "Member", member("Jon Berg", city="Oslo")[1]),
    ]

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Maria",
            schema=Member,
            mentions=["Maria"],
            fields={"name": "Maria", "city": "Lisbon"},
            matches="Maria Costa",
        )
        w.entity(
            "Jon",
            schema=Member,
            mentions=["Jon"],
            fields={"name": "Jon", "city": "Oslo"},
            matches="Jon Berg",
        )
        w.bleed("Jon", "city", "Lisbon")
        return w

    return Case(
        id="bleed_update",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        existing=existing,
        expect_ops=[Op("replace", "/city", "Lisbon", who="0")],
        must_not=[
            MustNot.op("replace", "/city", who="1"),
            MustNot.value_at("1", "/city", contains="Lisbon"),
        ],
        tags=("people", "update", "bleed"),
        world=world,
        note="Maria's city is offered for Jon (who is mentioned right after): the "
        "attribution check must refuse it.",
    )


NOTE_FOR_DANA = "She also asked to be introduced to our logistics team."


def notes_update_contact() -> Case:
    def world() -> FakeWorld:
        w = FakeWorld(CONTACT_NOTE)
        entity = w.entity(
            "the Contact",
            schema=Contact,
            mentions=["Dana"],
            fields={
                "name": "Dana",
                "phone": "+49 30 5555 0198",
                "city": "Berlin",
                "company": "Northwind",
                "title": "Head of Ops",
                "notes": [NOTE_FOR_DANA],
            },
        )
        # Clairic's list type, asked for an open-ended field, answers with
        # confidence ~1e-11 (live): the list answer is not trusted
        w.low_confidence(entity, "notes", 1e-9, persistent=True)
        return w

    return Case(
        id="notes_update_contact",
        text=CONTACT_NOTE,
        schemas=lambda: [One(Contact)],
        existing=[("crm-4711", "Contact", CONTACT_EXISTING)],
        expect_ops=[
            Op("replace", "/phone", "+49 30 5555 0198", who="crm-4711"),
            Op("replace", "/city", "Berlin", who="crm-4711"),
            Op("replace", "/company", "Northwind", who="crm-4711"),
            Op("replace", "/title", "Head of Ops", who="crm-4711"),
            Op(
                "add",
                "/notes/-",
                contains="introduced to our logistics team",
                who="crm-4711",
            ),
        ],
        must_not=[
            MustNot.op("replace", "/name"),
            MustNot.op("replace", "/email"),
            MustNot.op("remove"),
            MustNot.op("replace", "/notes"),
        ],
        tags=("contact", "update", "notes"),
        world=world,
        note="Examples/03. Live (05207aa): four replaces right, but the note was never "
        "appended (list confidence 3.2e-11).",
    )


def notes_insert_contact() -> Case:
    text = (
        "Dana Ruiz called from Lyon. She wants to meet in May. Her mobile died. "
        "Dr. Wu referred her to us."
    )
    # about Dana (a first live run's "Dr. Wu will follow up." was about Dr. Wu:
    # Clef rightly refused it as a note of hers); "Dr." must not split a sentence
    meet, follow = "She wants to meet in May.", "Dr. Wu referred her to us."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        entity = w.entity(
            "the Contact",
            schema=Contact,
            mentions=["Dana Ruiz"],
            fields={"name": "Dana Ruiz", "city": "Lyon", "notes": [meet, follow]},
        )
        w.low_confidence(entity, "notes", 1e-9, persistent=True)
        return w

    return Case(
        id="notes_insert_contact",
        text=text,
        schemas=lambda: [One(Contact)],
        expect_ops=[Op("insert")],
        expect_values=[
            Val("Contact", "/name", "Dana Ruiz"),
            Val("Contact", "/city", "Lyon"),
            Val("Contact", "/notes", contains="meet in May"),
            Val("Contact", "/notes", contains="referred her"),
        ],
        tags=("contact", "insert", "notes"),
        world=world,
        note="An open-ended notes list on an insert; 'Dr. Wu' must not split a sentence.",
    )


def notes_people_insert() -> Case:
    text = (
        "Friend: I wanted to introduce you to my cousin. Her name is Olivia Davis, "
        "she's a 27-year-old graphic designer. Olivia loves art and is always "
        "sketching in her free time. She also volunteers at the local animal shelter "
        "on weekends. Priya Raman is her flatmate."
    )

    def world() -> FakeWorld:
        w = FakeWorld(text)
        w.entity(
            "Olivia Davis",
            schema=Member,
            mentions=["Olivia Davis", "Olivia"],
            fields={
                "name": "Olivia Davis",
                "notes": [OLIVIA_DESIGNER, OLIVIA_ART, OLIVIA_SHELTER],
            },
        )
        w.entity(
            "Priya Raman",
            schema=Member,
            mentions=["Priya Raman"],
            fields={"name": "Priya Raman"},
        )
        return w

    return Case(
        id="notes_people_insert",
        text=text,
        schemas=lambda: [Many(Member, identity="name")],
        options={"enable_inserts": True},
        expect_ops=[Op("insert", who="Olivia"), Op("insert", who="Priya")],
        expect_values=[
            # must-have; her other notes (art, the shelter) are fine but optional
            Val("Olivia", "/notes", contains="graphic designer"),
        ],
        must_not=[MustNot.value_at("Priya", "/notes", contains="sketching")],
        tags=("people", "insert", "notes", "scoped"),
        world=world,
        note="Notes scoped to a person: the sentences that mention her and the ones "
        "after; Priya's record must not take Olivia's notes.",
    )


def notes_near_duplicate() -> Case:
    text = "Dana Whitfield called today. She really prefers email over phone."
    sentence = "She really prefers email over phone."

    def world() -> Overridden:
        w = FakeWorld(text)
        w.entity(
            "the Contact",
            schema=Contact,
            mentions=["Dana Whitfield"],
            fields={"name": "Dana Whitfield", "notes": [sentence]},
        )
        return Overridden(w, list_dedup=near_duplicates(0.95))

    return Case(
        id="notes_near_duplicate",
        text=text,
        schemas=lambda: [One(Contact)],
        existing=[("crm-1", "Contact", CONTACT_EXISTING)],
        expect_ops=[],
        expect_rejected=[Rejected("duplicate", "/notes/-", "crm-1")],
        # live 6: "Dana Whitfield called today." is grounded (attribution 0.92) and
        # new; the duplicate must still be rejected and nothing removed
        allow_ops=[Op("add", "/notes/-", contains="called today", who="crm-1")],
        must_not=[
            MustNot.op("add", "/notes/-", contains="prefers email"),
            MustNot.op("remove"),
        ],
        tags=("contact", "update", "notes"),
        world=world,
        note="The record already says she prefers email over phone: a paraphrase is "
        "the same fact and is not appended; the existing notes are untouched.",
    )


def note_beside_applied_change() -> Case:
    text = "Dana Whitfield called. She moved to Berlin to lead the Ops team."
    sentence = "She moved to Berlin to lead the Ops team."

    def world() -> FakeWorld:
        w = FakeWorld(text)
        entity = w.entity(
            "the Contact",
            schema=Contact,
            mentions=["Dana Whitfield"],
            fields={"name": "Dana Whitfield", "city": "Berlin", "notes": [sentence]},
        )
        w.low_confidence(entity, "notes", 1e-9, persistent=True)
        return w

    return Case(
        id="note_beside_applied_change",
        text=text,
        schemas=lambda: [One(Contact)],
        existing=[("crm-1", "Contact", CONTACT_EXISTING)],
        expect_ops=[
            Op("replace", "/city", "Berlin", who="crm-1"),
            Op("add", "/notes/-", contains="lead the Ops team", who="crm-1"),
        ],
        # live 6: "Dana Whitfield called." is a grounded extra note (0.97)
        allow_ops=[
            Op("add", "/notes/-", contains="Dana Whitfield called", who="crm-1")
        ],
        must_not=[MustNot.op("replace", "/name"), MustNot.op("remove")],
        tags=("contact", "update", "notes"),
        world=world,
        note="City=Berlin is applied; the sentence also says she leads the Ops team, "
        "which the record does not hold, so the note survives the novelty check "
        "(judged against the record after the ACCEPTED changes only).",
    )


CASES: list[Case] = [
    readme_people(),
    colleague_bleed(),
    computed_total(),
    invoice_contacts(),
    contacts_update_add(),
    other_invoice_total(),
    total_correction(),
    bill_insert(),
    rename(),
    short_form(),
    different_emma(),
    new_hire_same_first_name(),
    order_line_items(),
    order_remove_item(),
    relative_edit(),
    relative_mixed(),
    delete_request(),
    change_not_delete(),
    no_entity(),
    pet_not_person(),
    brand_not_person(),
    only_phone(),
    silence_is_not_deletion(),
    bleed_update(),
    notes_update_contact(),
    notes_insert_contact(),
    notes_people_insert(),
    notes_near_duplicate(),
    note_beside_applied_change(),
]
"""Every case, in the order reports list them."""

_UNIQUE = {c.id for c in CASES}
assert len(_UNIQUE) == len(CASES), "case ids must be unique"
