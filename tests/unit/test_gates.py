"""The change gate and the near-duplicate check, on a fake transport."""

import copy
from typing import Any

import pytest
from pydantic import BaseModel

from trustcall_system_one import CallMeta, GateConfig
from trustcall_system_one._diff import Change
from trustcall_system_one._gates import (
    GATE_QUESTION,
    dedup_elements,
    gate_changes,
    gate_state,
    threshold,
)
from trustcall_system_one._run import Run
from trustcall_system_one._schemas import coerce_schema
from trustcall_system_one._verify import Ctx
from trustcall_system_one._wire import CLEF, CLEF_FLASH
from trustcall_system_one.testing import FakeTransport, clef_response, noul

CLEF_MODELS = {CLEF: "clef", CLEF_FLASH: "clef-flash"}

RECORD = {"name": "Dana Whitfield", "city": "Austin", "notes": ["Prefers email"]}
TEXT = "Dana moved to Berlin. She likes tea in the morning."


class Contact(BaseModel):
    name: str
    city: str | None = None
    notes: list[str] = []


def make(
    p: Any = 0.9, **run_kwargs: Any
) -> tuple[Ctx, FakeTransport, list[tuple[str, dict[str, Any], CallMeta]]]:
    """A context whose Clef answers ``p`` (a float, or a function of the payload)."""
    from trustcall_system_one import compile_schema

    def handler(model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        value = p(payload) if callable(p) else p
        return clef_response(
            {qid: noul(value) for qid in payload["questions"]},
            model=CLEF_MODELS[model],  # type: ignore[arg-type]
        )

    transport = FakeTransport(handler)
    run = Run(transport=transport, **run_kwargs)
    ctx = Ctx(
        run=run,
        plan=compile_schema(coerce_schema(Contact)),
        text=TEXT,
        label="the Contact",
    )
    return ctx, transport, transport.calls


def replace_city() -> Change:
    return Change(
        "replace", "/city", "Berlin", "Austin", "city", ("city",), ((14, 20),)
    )


def add_note() -> Change:
    return Change("add", "/notes/-", "likes tea", None, "notes", ("notes",), ())


def set_null() -> Change:
    return Change("replace", "/city", None, "Austin", "city", ("city",), ())


def remove_city() -> Change:
    return Change("remove", "/city", None, "Austin", "city", ("city",), ())


# --------------------------------------------------------------------------- #
# One request per operation
# --------------------------------------------------------------------------- #


async def test_every_operation_gets_its_own_request_with_its_own_state() -> None:
    ctx, _, calls = make()
    verdicts = await gate_changes(ctx, RECORD, [replace_city(), add_note()])
    assert len(calls) == 2 and all(meta.kind == "gate" for _, _, meta in calls)
    states = [payload["state"] for _, payload, _ in calls]
    assert states[0] != states[1]
    assert {s["proposed_change"]["path"] for s in states} == {"/city", "/notes/-"}
    assert all(len(payload["questions"]) == 1 for _, payload, _ in calls)
    assert [v.passed for v in verdicts] == [True, True]


async def test_the_state_carries_input_entity_record_and_the_proposed_change() -> None:
    ctx, _, calls = make()
    await gate_changes(ctx, RECORD, [replace_city()])
    ((_, payload, meta),) = calls
    state = payload["state"]
    assert set(state) == {"input", "entity", "record", "proposed_change"}
    assert state["input"] == TEXT
    assert state["entity"] == {"label": "the Contact", "mentions": []}
    assert state["record"] == RECORD
    assert state["proposed_change"] == {
        "op": "replace",
        "path": "/city",
        "field": "city",
        "from": "Austin",
        "to": "Berlin",
    }
    assert meta.field == "city" and meta.schema == "Contact"


async def test_an_add_has_no_from_value() -> None:
    ctx, _, calls = make()
    await gate_changes(ctx, RECORD, [add_note()])
    change = calls[0][1]["state"]["proposed_change"]
    assert change == {
        "op": "add",
        "path": "/notes/-",
        "field": "notes",
        "from": None,
        "to": "likes tea",
    }


async def test_the_question_is_the_documented_one() -> None:
    ctx, _, calls = make()
    await gate_changes(ctx, RECORD, [replace_city()])
    ((_, payload, _),) = calls
    (question,) = payload["questions"].values()
    assert question["type"] == "noul"
    assert question["instructions"] == GATE_QUESTION
    assert "`proposed_change.to`" in question["instructions"]
    assert "`entity.label`" in question["instructions"]
    assert "`record`" in question["instructions"]
    assert set(question["criteria"]) == {"true", "false"}
    assert "`proposed_change.to`" in question["criteria"]["true"]
    assert "shorter or partial form" in question["criteria"]["false"]


async def test_a_removal_is_asked_as_removed_or_no_longer_applies() -> None:
    ctx, _, calls = make()
    await gate_changes(ctx, RECORD, [remove_city(), set_null()])
    for _, payload, _ in calls:
        (question,) = payload["questions"].values()
        assert "was removed or no longer applies" in question["instructions"]
        assert "`city`" in question["instructions"]
    assert calls[0][1]["state"]["proposed_change"]["to"] is None


async def test_the_state_does_not_alias_the_record() -> None:
    ctx, _, _ = make()
    record = copy.deepcopy(RECORD)
    await gate_changes(ctx, record, [replace_city()])
    assert record == RECORD


async def test_the_gate_model_is_used() -> None:
    ctx, _, calls = make(gate_model=CLEF_FLASH)
    await gate_changes(ctx, RECORD, [replace_city()])
    assert calls[0][0] == CLEF_FLASH
    ctx, _, calls = make()
    await gate_changes(ctx, RECORD, [replace_city()])
    assert calls[0][0] == CLEF  # the default


async def test_no_changes_make_no_request() -> None:
    ctx, _, calls = make()
    assert await gate_changes(ctx, RECORD, []) == []
    assert calls == []


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("make_change", "needed"),
    [
        (add_note, 0.55),
        (replace_city, 0.6),
        (remove_city, 0.85),
        (set_null, 0.85),  # a set-to-null is a removal
    ],
)
def test_the_threshold_depends_on_the_kind_of_operation(
    make_change: Any, needed: float
) -> None:
    assert threshold(GateConfig(), make_change()) == needed


@pytest.mark.parametrize(
    ("make_change", "needed"),
    [(add_note, 0.55), (replace_city, 0.6), (remove_city, 0.85)],
)
async def test_an_operation_passes_exactly_at_its_threshold(
    make_change: Any, needed: float
) -> None:
    ctx, _, _ = make(needed)
    (verdict,) = await gate_changes(ctx, RECORD, [make_change()])
    assert verdict.passed and verdict.p == needed and verdict.needed == needed
    ctx, _, _ = make(round(needed - 0.01, 2))
    (verdict,) = await gate_changes(ctx, RECORD, [make_change()])
    assert not verdict.passed


async def test_thresholds_come_from_the_gate_config() -> None:
    config = GateConfig(add_threshold=0.9, replace_threshold=0.8, remove_threshold=0.99)
    ctx, _, _ = make(0.85, gates=config)
    verdicts = await gate_changes(
        ctx, RECORD, [add_note(), replace_city(), remove_city()]
    )
    assert [(v.passed, v.needed) for v in verdicts] == [
        (False, 0.9),
        (True, 0.8),
        (False, 0.99),
    ]


async def test_each_operation_is_judged_on_its_own_answer() -> None:
    ctx, _, _ = make(
        lambda payload: (
            0.1 if payload["state"]["proposed_change"]["path"] == "/city" else 0.9
        )
    )
    verdicts = await gate_changes(ctx, RECORD, [replace_city(), add_note()])
    assert [v.passed for v in verdicts] == [False, True]
    assert [v.p for v in verdicts] == [0.1, 0.9]


async def test_a_missing_answer_is_never_a_pass() -> None:
    ctx, _, _ = make()

    async def nothing(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    ctx.run.call_clef = nothing  # type: ignore[method-assign]
    (verdict,) = await gate_changes(ctx, RECORD, [add_note()])
    assert not verdict.passed and verdict.p is None


# --------------------------------------------------------------------------- #
# Big inputs
# --------------------------------------------------------------------------- #


def test_a_big_input_is_narrowed_around_the_spans_and_the_record_is_kept() -> None:
    ctx, _, _ = make()
    filler = "The weather was mild. " * 6000
    ctx.text = filler + "Dana moved to Berlin. " + filler
    start = ctx.text.index("Berlin")
    change = Change(
        "replace", "/city", "Berlin", "Austin", "city", ("city",), ((start, start + 6),)
    )
    state = gate_state(ctx, RECORD, change)
    assert len(state["input"]) < 3000 and "Berlin" in state["input"]
    assert state["record"] == RECORD
    assert state["proposed_change"]["to"] == "Berlin"
    assert any("clef_state_narrowed" in w for w in ctx.run.warnings)


# --------------------------------------------------------------------------- #
# Near-duplicates
# --------------------------------------------------------------------------- #


def near_note(text: str, old: str, ratio: float = 0.8) -> Change:
    return Change(
        "add", "/notes/-", text, None, "notes", ("notes",), (), near=[(old, ratio)]
    )


async def test_one_question_per_near_duplicate_in_one_request() -> None:
    ctx, _, calls = make(0.9)
    changes = [
        near_note("Prefers email to phone calls", "Prefers email over phone"),
        add_note(),  # nothing resembles it: not asked
        near_note("Likes tea in the morning", "Likes tea"),
    ]
    verdicts = await dedup_elements(ctx, RECORD, changes)
    ((_, payload, meta),) = calls
    assert meta.kind == "list_dedup" and set(payload["questions"]) == {
        "dedup.0",
        "dedup.2",
    }
    first = payload["questions"]["dedup.0"]["instructions"]
    assert '"Prefers email to phone calls"' in first
    assert '"Prefers email over phone"' in first
    assert "same fact" in first
    assert set(payload["state"]) == {"input", "entity", "record"}
    assert set(verdicts) == {0, 2}
    assert all(v.duplicate and v.p == 0.9 for v in verdicts.values())
    assert verdicts[0].other == "Prefers email over phone"


@pytest.mark.parametrize(("p", "duplicate"), [(0.59, False), (0.6, True), (0.95, True)])
async def test_an_element_is_a_duplicate_from_0_6(p: float, duplicate: bool) -> None:
    ctx, _, _ = make(p)
    verdicts = await dedup_elements(ctx, RECORD, [near_note("x y z", "x y")])
    assert verdicts[0].duplicate is duplicate


async def test_nothing_near_means_no_request() -> None:
    ctx, _, calls = make()
    assert await dedup_elements(ctx, RECORD, [add_note(), replace_city()]) == {}
    assert calls == []


async def test_the_most_similar_old_element_is_the_one_compared() -> None:
    ctx, _, calls = make()
    change = Change(
        "add",
        "/notes/-",
        "new fact",
        None,
        "notes",
        ("notes",),
        (),
        near=[("closest", 0.85), ("further", 0.65)],
    )
    await dedup_elements(ctx, RECORD, [change])
    (question,) = calls[0][1]["questions"].values()
    assert '"closest"' in question["instructions"]
    assert '"further"' not in question["instructions"]
