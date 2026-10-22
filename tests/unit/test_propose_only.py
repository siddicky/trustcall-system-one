"""Propose-only updates (a flagged record with updates off): S1 and the two nits.

* S1: the items of a flagged parent count as proposals, so a parent whose items
  the input changes is a ``conflict_update_delete``, not a deletion.
* (a) a refused answer that only repeats the record's value proposes nothing.
* (b) a list element that repeats the record is dropped by the near-duplicate
  check before it counts as proposed.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Annotated, Any, Optional

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import Script, ca, pick
from trustcall_system_one import Many, create_extractor
from trustcall_system_one._compile import compile_schema
from trustcall_system_one._entity import update_one
from trustcall_system_one._existing import ExistingRecord
from trustcall_system_one._run import Run
from trustcall_system_one.testing import FakeWorld


class Pet(BaseModel):
    """A pet."""

    name: str = Field(description="the pet's name")
    age: Optional[int] = Field(None, description="age in years")


class Person(BaseModel):
    """A person."""

    name: str = Field(description="full name")
    pets: Annotated[list[Pet], Many(identity="name")] = Field(default_factory=list)


TEXT = "Emma Thompson's dog Rex is now 5."
EXISTING = [
    ("e", "Person", {"name": "Emma Thompson", "pets": [{"name": "Rex", "age": 3}]})
]


def flagged_emma(updates: bool, *, rex_age: int = 5) -> Any:
    world = FakeWorld(TEXT)
    emma = world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Rex",
        schema=Pet,
        parent=emma,
        mentions=["Rex"],
        fields={"name": "Rex", "age": rex_age},
        matches={"name": "Rex"},
    )
    world.delete("e", 0.97)
    extractor = create_extractor(
        [Many(Person)],
        transport=world.transport(),
        enable_updates=updates,
        enable_deletes=True,
        enable_inserts=True,
    )
    return extractor


@pytest.mark.parametrize("updates", [True, False])
async def test_a_flagged_parent_whose_items_change_is_a_conflict_not_a_delete(
    updates: bool,
) -> None:
    result = await flagged_emma(updates).ainvoke(TEXT, existing=EXISTING)
    assert result.deleted == ()
    assert result.records == ()
    (skip,) = result.skipped
    assert skip.reason == "conflict_update_delete"
    assert "replace /pets/0/age" in skip.detail


async def test_a_flagged_parent_whose_items_the_input_leaves_alone_is_deleted() -> None:
    result = await flagged_emma(False, rex_age=3).ainvoke(TEXT, existing=EXISTING)
    assert result.deleted == ("e",)
    assert result.skipped == ()


# ---- (a) a refused answer that repeats the record ------------------------ #


class Contact(BaseModel):
    """A contact."""

    name: str
    city: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


CONTACT_TEXT = "Dana lived in Paris once. Then she moved back to Austin."
CONTACT = {"name": "Dana Whitfield", "city": "Austin", "notes": ["Likes jazz"]}


async def propose(script: Script, text: str, record: dict[str, Any]) -> Any:
    run = Run(transport=script.transport)
    plan = compile_schema(Contact)
    return await update_one(
        run,
        plan,
        text,
        ExistingRecord("c", "Contact", record),
        label="the Contact",
        skip_presence=True,
        propose_only=True,
    )


def contact_script() -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", None, p_null=0.98),
        "city": ca("text", "Paris", find="Paris", confidence=0.9),
        "notes": ca("list", None, p_null=0.98),
    }
    return world


async def test_a_refused_answer_is_a_proposal_when_it_differs_from_the_record() -> None:
    world = contact_script()
    world.attribution = {"attr.city": 0.02}
    world.reask = {"city": ca("text", "Paris", find="Paris", confidence=0.9)}
    outcome = await propose(world, CONTACT_TEXT, CONTACT)
    assert outcome.proposed == ("city (attribution)",)


async def test_a_refused_answer_equal_to_the_record_proposes_nothing() -> None:
    world = contact_script()
    # the input-only pass reads the record's own city and the adjudication takes it
    # over; it then fails attribution: refused, but it repeats the record
    world.cross = {"city": ca("text", "Austin", find="Austin", confidence=0.9)}
    world.adjudicate = {"city": pick("b", ["a", "b", "neither"], 0.9)}
    world.attribution = {"attr.city": 0.02}
    outcome = await propose(world, CONTACT_TEXT, CONTACT)
    assert world.calls("adjudicate")
    assert outcome.proposed == ()


# ---- (b) the near-duplicate check ---------------------------------------- #


def notes_script(dedup: float) -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", None, p_null=0.98),
        "city": ca("text", None, p_null=0.98),
        "notes": ca(
            "list",
            ["She really likes jazz"],
            find_all=["She really likes jazz"],
            p_null=0.05,
            confidence=0.95,
        ),
    }
    world.dedup_default = dedup
    return world


JAZZ_TEXT = "Dana called. She really likes jazz."


async def test_a_repeat_of_a_record_element_is_not_proposed() -> None:
    world = notes_script(0.95)
    outcome = await propose(world, JAZZ_TEXT, {**CONTACT, "notes": ["She likes jazz"]})
    assert world.calls("list_dedup")
    assert outcome.proposed == ()


async def test_a_new_element_is_proposed_after_the_duplicate_check() -> None:
    world = notes_script(0.05)
    outcome = await propose(world, JAZZ_TEXT, {**CONTACT, "notes": ["She likes jazz"]})
    assert world.calls("list_dedup")
    assert outcome.proposed == ("add /notes/-",)


@pytest.mark.parametrize("updates", [True, False])
@pytest.mark.parametrize("inserts", [True, False])
async def test_a_new_item_of_a_flagged_parent_is_proposed_even_without_inserts(
    updates: bool, inserts: bool
) -> None:
    text = "Emma Thompson got a new dog, Max, who is 2."
    world = FakeWorld(text)
    emma = world.entity(
        "Emma Thompson",
        schema=Person,
        mentions=["Emma Thompson"],
        fields={"name": "Emma Thompson"},
        matches={"name": "Emma Thompson"},
    )
    world.entity(
        "Max",
        schema=Pet,
        parent=emma,
        mentions=["Max"],
        fields={"name": "Max", "age": 2},
    )
    world.delete("e", 0.97)
    extractor = create_extractor(
        [Many(Person)],
        transport=world.transport(),
        enable_updates=updates,
        enable_deletes=True,
        enable_inserts=inserts,
    )
    result = await extractor.ainvoke(text, existing=EXISTING)
    assert result.deleted == () and result.records == ()
    (skip,) = result.skipped
    assert skip.reason == "conflict_update_delete"
    assert "add /pets/-" in skip.detail
