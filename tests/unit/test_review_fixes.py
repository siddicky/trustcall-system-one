"""Regressions of the review of the integrated fixes (probes turned into tests)."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import Script, ca, make_extractor
from trustcall_system_one import CallMeta
from trustcall_system_one._sentences import ITEMS_VETO, _hedged_veto
from trustcall_system_one._wire import ChoiceAnswer
from trustcall_system_one.testing import choice, clef_response

__all__: list[str] = []

# --------------------------------------------------------------------------- #
# S3: the items veto is for hedging readers only (probe_items)
# --------------------------------------------------------------------------- #


class Contact(BaseModel):
    """A contact."""

    name: str = Field(description="the person's full name")
    phones: list[str] = Field(default_factory=list, description="phone numbers")


PHONES = "Dana Ruiz is our new lead. Call her on 555-0101 or 555-0199 any time."


class _Kind(Script):
    p_items = 0.7

    def _list_kind(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {
                q: choice(
                    "items", {"items": self.p_items, "statements": 1 - self.p_items}
                )
                for q in payload["questions"]
            }
        )


@pytest.mark.parametrize("p_items", [0.7, 0.6, 0.55])
async def test_phone_numbers_stay_items_when_clef_grades_them_as_items(
    p_items: float,
) -> None:
    world = _Kind(
        clairic={
            "name": ca("text", "Dana Ruiz", find="Dana Ruiz", confidence=0.99),
            "phones": ca(
                "list", None, find_all=["555-0101", "555-0199"], confidence=0.3
            ),
        },
        reask={
            "phones": ca(
                "list", None, find_all=["555-0101", "555-0199"], confidence=0.3
            )
        },
        sentences={"555-0101": 0.9},
    )
    world.p_items = p_items
    (record,) = (await make_extractor([Contact], world).ainvoke(PHONES)).records
    # the graded answer of a provider that is not hedging keeps the plain rule: the
    # list is NOT read by sentences, so the phones are never the sentence
    phones = (record.after or {}).get("phones") or []
    assert "Call her on 555-0101 or 555-0199 any time." not in phones
    assert not any("sentences" in w for w in record.fields["phones"].warnings)


def _answer(items: float, *, hard: bool, confidence: float) -> ChoiceAnswer:
    body: dict[str, Any] = {
        "type": "choice",
        "choice": "items",
        "probabilities": {"items": items, "statements": 1 - items},
        "confidence": confidence,
    }
    if hard:
        body["hard"] = True
    return ChoiceAnswer.model_validate(body)


def test_the_veto_is_clamped_to_the_hard_cap() -> None:
    from types import SimpleNamespace

    ctx: Any = SimpleNamespace(run=SimpleNamespace(calls=[]))
    spec: Any = SimpleNamespace(wire_name="notes")
    assert (
        _hedged_veto(ctx, spec, _answer(0.8, hard=True, confidence=0.8)) == ITEMS_VETO
    )
    # a lower cap: a hard "items" holds the cap, which is the veto, so it vetoes
    low = _hedged_veto(ctx, spec, _answer(0.6, hard=True, confidence=0.6))
    assert low == 0.6
    assert low is not None and low <= 0.6
    # a graded answer with no Decisions provider on record does not hedge
    assert _hedged_veto(ctx, spec, _answer(0.7, hard=False, confidence=0.4)) is None
    decisions = SimpleNamespace(kind="list_kind", field="notes", provider="decisions")
    ctx.run.calls.append(decisions)
    assert (
        _hedged_veto(ctx, spec, _answer(0.7, hard=False, confidence=0.4)) == ITEMS_VETO
    )
    ctx.run.calls[:] = [
        SimpleNamespace(kind="list_kind", field="notes", provider="cloudflare")
    ]
    assert _hedged_veto(ctx, spec, _answer(0.7, hard=False, confidence=0.4)) is None


# --------------------------------------------------------------------------- #
# NIT: the code screen counts entities, keeps short names, and gives back
# --------------------------------------------------------------------------- #


class Pet(BaseModel):
    """A pet."""

    name: str = Field(description="the pet's name")


class Row(BaseModel):
    """A row of a table."""

    remarks: str | None = None


def _screened(
    model: type[BaseModel],
    texts: list[str],
    count: int | None,
    *,
    identity: str | None,
    parent: bool = True,
) -> tuple[list[str], list[str]]:
    from types import SimpleNamespace

    from trustcall_system_one._cardinality import Many
    from trustcall_system_one._collection import _prefilter
    from trustcall_system_one._compile import compile_schema
    from trustcall_system_one._types import Mention

    plan = compile_schema(model)
    many = Many(model, identity=identity) if identity else Many(model)
    warnings: list[str] = []
    scope: Any = SimpleNamespace(
        plan=plan,
        cardinality=many,
        parent=object() if parent else None,
        noun=model.__name__.lower(),
        kind=model.__name__.lower(),
        name=model.__name__,
        scoped=lambda text: text,
    )
    run: Any = SimpleNamespace(warn=warnings.append)
    mentions = [Mention(t, (i * 10, i * 10 + len(t))) for i, t in enumerate(texts)]
    kept = _prefilter(run, scope, mentions, count)
    return [m.text for m in kept], warnings


def test_the_count_is_of_entities_not_of_mentions() -> None:
    # three mentions of ONE pet and two dropped codes; Clairic counted 3 entities: the
    # survivors are one distinct entity, so the best of the dropped come back
    kept, warnings = _screened(Pet, ["Rex", "Rex", "Rex", "L", "7"], 3, identity="name")
    assert "L" in kept and "7" in kept
    assert any("came back" in w for w in warnings)
    # three distinct entities survive: nothing comes back
    kept, _ = _screened(Pet, ["Rex", "Fido", "Bella", "L", "7"], 3, identity="name")
    assert kept == ["Rex", "Fido", "Bella"]


def test_a_two_letter_name_is_kept_in_a_nested_named_list() -> None:
    kept, _ = _screened(Pet, ["Jo", "Rex", "L"], 2, identity="name")
    assert "Jo" in kept
    # not a name when the list has no identity (a column of codes)
    kept, _ = _screened(Row, ["Jo", "Rex", "A12"], 2, identity=None)
    assert "Jo" not in kept
    # and not in a top level collection, which is not screened by this rank
    kept, _ = _screened(Pet, ["Jo", "Rex", "A12"], 2, identity="name", parent=False)
    assert "Jo" not in kept


def test_without_a_count_a_screen_that_leaves_nothing_gives_back() -> None:
    kept, warnings = _screened(Row, ["L", "7", "LH"], None, identity=None)
    assert sorted(kept) == ["7", "L", "LH"]
    assert any("nothing else survived" in w for w in warnings)
    # something survives and no count: the screen stands
    kept, _ = _screened(Row, ["L", "7", "Left engine"], None, identity=None)
    assert kept == ["Left engine"]


# --------------------------------------------------------------------------- #
# The attribution question: a clause description, and an amount added to a new item
# --------------------------------------------------------------------------- #


def test_a_clause_description_is_not_hung_on_its_last_noun() -> None:
    from trustcall_system_one._verify import _attribution_question

    clause = _attribution_question(
        "how the guest knows the host", "colleague", element=False
    ).instructions
    # "the host of <entity>" would say whose host; the entity is the guest
    assert "state that for `entity.label`, how the guest knows the host is" in clause
    assert "host of `entity.label`" not in clause
    for noun in ("product name", "the number of units"):
        plain = _attribution_question(noun, "x", element=False).instructions
        assert f"state that {noun} of `entity.label` is" in plain
    for word in ("whether", "when", "where", "which", "what", "who"):
        asked = _attribution_question(f"{word} it is", "x", element=False)
        assert "for `entity.label`, " in asked.instructions
    # a noun that merely starts like a wh-word is a noun
    howl = _attribution_question("howler count", "x", element=False)
    assert "howler count of `entity.label`" in howl.instructions


class Seat(BaseModel):
    """One line of a booking."""

    name: str = Field(description="the name of the line")
    seats: int = Field(description="number of seats to book")


class Booking(BaseModel):
    """A booking."""

    ref: str
    lines: list[Seat] = Field(default_factory=list)


def test_the_ask_for_question_of_a_new_entity_says_it_is_new() -> None:
    from trustcall_system_one._verify import _FRESH_NOTE, _attribution_question

    wording = "number of seats to book"
    new = _attribution_question(wording, 2, element=False, ask_for=True, fresh=True)
    held = _attribution_question(wording, 2, element=False, ask_for=True)
    assert new.instructions == str(held.instructions) + _FRESH_NOTE
    assert "is the whole amount" in str(new.instructions)
    # a value that is only stated is never said to be an amount to add
    stated = _attribution_question("the amount due", 2, element=False, fresh=True)
    assert "whole amount" not in str(stated.instructions)
    assert "whole amount" not in str(held.instructions)


async def test_only_an_insert_is_told_that_its_entity_is_new() -> None:
    from tests.unit.extract_helpers import Script, ca, make_extractor
    from trustcall_system_one import One

    def asked(world: Script) -> list[str]:
        return [
            q["instructions"]
            for call in world.calls("attribution")
            for qid, q in call[1]["questions"].items()
            if qid == "attr.seats"
        ]

    text = "Add two more seats in the stalls."

    def script() -> Script:
        world = Script()
        world.clairic = {
            "name": ca("text", "stalls", find="stalls"),
            "seats": ca("number", 2, find="two"),
        }
        return world

    # a record that is created: the entity is new
    world = script()
    await make_extractor([One(Seat)], world).ainvoke(text)
    assert asked(world)
    assert all("is new, nothing of it was held" in a for a in asked(world))

    # a record that is updated: it holds seats already, "two more" is relative, and
    # the question does not say the entity is new
    world = script()
    world.relative = {"two more": 0.95}  # Clef reads "two more" as a relative edit
    result = await make_extractor([One(Seat)], world).ainvoke(
        text, existing=[("s-1", "Seat", {"name": "stalls", "seats": 4})]
    )
    assert asked(world)
    assert not [a for a in asked(world) if "is new" in a]
    (record,) = result.records
    assert record.after is not None and record.after["seats"] == 4  # never written as 2
    assert [r.reason for r in record.rejected] == ["relative_edit_unsupported"]
