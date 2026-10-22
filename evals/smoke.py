"""The live smoke scenarios: what ``make smoke`` runs and ``make record`` records.

Each scenario states its input, its schemas and the invariants that must hold
whatever the live models answer. The same scenarios are replayed offline from the
cassettes ``make record`` writes into ``tests/recorded`` (``tests/unit/
test_recorded.py``), so a recorded run can be re-checked on every ``make test``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evals import invariants as inv
from evals.scenarios import (
    CONTACT_EXISTING,
    CONTACT_NOTE,
    CONVERSATION,
    EXISTING_PEOPLE,
    INVOICE_TEXT,
    ORDER_EMAIL,
    ORDER_EXISTING,
    Contact,
    Invoice,
    Order,
    Person,
)
from trustcall_system_one import ExtractionResult, Many, One

__all__ = ["SCENARIOS", "Scenario"]


@dataclass(frozen=True)
class Scenario:
    """One smoke scenario.

    Attributes:
        name: Names the cassette (``smoke_<name>.jsonl``).
        schemas: Builds the schemas for ``create_extractor``.
        text: The input.
        check: Asserts the invariants of a result (raises ``AssertionError``).
        existing: The ``existing`` argument.
        options: Extra ``create_extractor`` keyword arguments.
    """

    name: str
    schemas: Callable[[], list[Any]]
    text: str
    check: Callable[[ExtractionResult], None]
    existing: Any = None
    options: dict[str, Any] = field(default_factory=dict)


def _common(result: ExtractionResult, text: str, existing_ids: set[str]) -> None:
    assert result.usage.requests == len(result.calls) > 0
    assert all(c.error is None for c in result.calls), [c.error for c in result.calls]
    inv.every_patch_applies(result)
    inv.unmentioned_fields_are_untouched(result)
    inv.ids_are_distinct_and_existing(result, existing_ids)
    inv.spans_quote_the_input(result, text)


def _people(result: ExtractionResult) -> None:
    _common(result, CONVERSATION, {"0", "1", "2"})
    inv.only_kinds(result, {"add", "replace"})
    for record in inv.updates(result):
        assert record.before is not None and record.after is not None
        # a short form ("Sarah") never replaces the record's full name
        assert record.after["name"] == record.before["name"], record.json_doc_id
        # the existing notes stay, in place; new ones are appended
        held = record.before["notes"]
        assert record.after["notes"][: len(held)] == held, record.json_doc_id
    inserted = [r for r in result.records if r.kind == "insert"]
    names = {r.entity.label for r in inserted if r.entity is not None}
    assert not any("sunny" in n.casefold() for n in names), "the dog is not a Person"
    assert not result.deleted


def _invoice(result: ExtractionResult) -> None:
    _common(result, INVOICE_TEXT, set())
    (record,) = result.records
    assert record.kind == "insert"
    assert record.valid and record.value is not None, record.errors
    number: Any = record.value.invoice_number  # type: ignore[attr-defined]
    assert number in INVOICE_TEXT


def _contact(result: ExtractionResult) -> None:
    _common(result, CONTACT_NOTE, {"crm-4711"})
    inv.only_kinds(result, {"add", "replace"})
    (record,) = result.records
    assert record.kind == "update" and record.json_doc_id == "crm-4711"
    assert record.valid
    assert record.after is not None
    # the note says nothing about the e-mail address, and "Dana" is a short form
    assert record.after["email"] == CONTACT_EXISTING["email"]
    assert record.after["name"] == CONTACT_EXISTING["name"]
    assert record.after["notes"][:2] == CONTACT_EXISTING["notes"]


def _order(result: ExtractionResult) -> None:
    _common(result, ORDER_EMAIL, {"SO-10442"})
    inv.only_kinds(result, {"add", "replace"})
    # the steel kettle and the wooden spoon set are not in the email
    inv.no_op_inside(
        result, "SO-10442", ["/items/0", "/items/2", "/customer", "/order_id"]
    )
    (record,) = result.records
    assert record.valid and record.after is not None
    assert record.after["items"][0] == ORDER_EXISTING["items"][0]
    assert record.after["items"][2] == ORDER_EXISTING["items"][2]


SCENARIOS: list[Scenario] = [
    Scenario(
        "people",
        lambda: [Many(Person, identity="name")],
        CONVERSATION,
        _people,
        existing=EXISTING_PEOPLE,
        options={"enable_inserts": True},
    ),
    Scenario("invoice_insert", lambda: [One(Invoice)], INVOICE_TEXT, _invoice),
    Scenario(
        "contact_update",
        lambda: [One(Contact)],
        CONTACT_NOTE,
        _contact,
        existing=[("crm-4711", "Contact", CONTACT_EXISTING)],
    ),
    Scenario(
        "order_update",
        lambda: [One(Order)],
        ORDER_EMAIL,
        _order,
        existing=[("SO-10442", "Order", ORDER_EXISTING)],
        options={"enable_inserts": True},
    ),
]
"""The scenarios, by name: people, invoice_insert, contact_update, order_update."""
