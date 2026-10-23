"""Decisions as the PRIMARY Clef provider: what live runs found, with their shapes.

The shapes come from the recordings of a live run (``--clef-provider decisions``)
and are written out here by hand, small and synthetic:

* ``kind`` of a notes field: ``items`` 0.65 to 0.67 (confidence 0.3 to 0.34), graded,
  in 15 of 15 asks, where Clef gave ``statements`` 0.95 to 0.97; no note was ever
  added;
* most other answers are saturated (``{yes: 1.0, no: 0.0}``), which the translation
  turns into a capped "hard pick" (0.8 for yes, 0.2 for no);
* one answer came back as ``{"type": "refusal", "name": "plan"}`` (no ``choice``) and
  the whole case errored.
"""

from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from evals.cases import CASES
from evals.configs import build_options
from evals.scoring import actual_ops
from tests.unit.test_decisions import decisions
from trustcall_system_one import (
    CallMeta,
    TransportResponse,
    compile_schema,
    create_extractor,
)
from trustcall_system_one._decisions import (
    from_decisions_response,
    readable_state,
    to_decisions_request,
)
from trustcall_system_one._run import FALLBACK_WARNING, HARD_PICKS_WARNING, Run
from trustcall_system_one._sentences import list_kind
from trustcall_system_one._verify import Ctx
from trustcall_system_one._wire import ClefRequest

# --------------------------------------------------------------------------- #
# A Decisions server that answers with the shapes the live API sent
# --------------------------------------------------------------------------- #


class Notes(BaseModel):
    """A contact."""

    name: str
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


KIND_QUESTION = {
    "model": "clef",
    "state": {"field": {"name": "notes", "description": "things worth remembering"}},
    "questions": {
        "kind": {
            "type": "choice",
            "instructions": "Are the items of `field.name` statements?",
            "criteria": {"items": "Short named values.", "statements": "Statements."},
        }
    },
}


def decisions_body(*answers: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": "gpt-6-luna-decisions",
        "answers": list(answers),
        "usage": {"input_tokens": 10, "output_tokens": 0, "cost": 0.0},
    }


def choice_answer(
    name: str, choice: str, probabilities: Mapping[str, float], confidence: float
) -> dict[str, Any]:
    return {
        "type": "choice",
        "name": name,
        "choice": choice,
        "probabilities": [
            {"value": v, "probability": p} for v, p in probabilities.items()
        ],
        "confidence": confidence,
    }


def a_kind_run(answer: Mapping[str, Any]) -> tuple[Run, Ctx, Any]:
    """A run whose Clef is a Decisions server that gives ``answer`` to ``kind``."""
    transport, _ = decisions(httpx.Response(200, json=decisions_body(answer)))
    run = Run(transport=transport)
    plan = compile_schema(Notes)
    return run, Ctx(run=run, plan=plan, text="x", label="the Notes"), plan


async def kind_of(answer: Mapping[str, Any]) -> str:
    _, ctx, plan = a_kind_run(answer)
    return await list_kind(ctx, plan.spec(("notes",)))


# --------------------------------------------------------------------------- #
# 1. Notes: the kind of a notes field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("items", "confidence"), [(0.67, 0.34), (0.65, 0.3), (0.5, 0.0), (0.74, 0.48)]
)
async def test_a_hedged_items_answer_does_not_keep_notes_out(
    items: float, confidence: float
) -> None:
    # recorded: items 0.67 / statements 0.33 (confidence 0.34) and 0.65 / 0.35 (0.3)
    answer = choice_answer(
        "kind",
        "items",
        {"items": items, "statements": round(1 - items, 2)},
        confidence,
    )
    assert await kind_of(answer) == "statements"


async def test_a_clear_items_answer_still_keeps_a_list_of_names_out() -> None:
    graded = choice_answer("kind", "items", {"items": 0.9, "statements": 0.1}, 0.8)
    assert await kind_of(graded) == "items"
    saturated = choice_answer("kind", "items", {"items": 1.0, "statements": 0.0}, 1.0)
    assert await kind_of(saturated) == "items"  # a hard pick: 0.8 for items


async def test_a_statements_answer_is_statements_hard_or_graded() -> None:
    saturated = choice_answer(
        "kind", "statements", {"items": 0.0, "statements": 1.0}, 1.0
    )
    assert await kind_of(saturated) == "statements"
    graded = choice_answer(
        "kind", "statements", {"items": 0.03, "statements": 0.97}, 0.9
    )
    assert await kind_of(graded) == "statements"


async def test_a_refused_kind_question_stays_items() -> None:
    # no answer is no evidence: the field keeps its low-confidence rejection
    assert await kind_of({"type": "refusal", "name": "kind"}) == "items"


# --------------------------------------------------------------------------- #
# The dev suite's scripted worlds, answered the way Decisions answered
# --------------------------------------------------------------------------- #


class DecisionsShaped:
    """A transport that gives a scripted world's Clef answers in Decisions' shapes.

    Every answer is saturated (yes/no, 1.0/0.0), except ``list_kind``, which is the
    recorded 0.67/0.33 in favour of ``items``; the kinds in ``refuse`` come back as
    refusals. Each answer goes through the real translation both ways. Clairic calls
    go to the world as they are.
    """

    provider = "decisions"

    def __init__(self, world: Any, refuse: frozenset[str] = frozenset()) -> None:
        self._inner = world.transport()
        self._refuse = refuse

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        if payload.get("model") not in ("clef", "clef-flash"):
            relayed: TransportResponse = await self._inner.run(
                model, payload, meta=meta, timeout=timeout
            )
            return relayed
        body = self._inner._handler(model, dict(payload), meta)
        _, plans = to_decisions_request(payload)
        answers: list[dict[str, Any]] = []
        for plan in plans:
            given = body["answers"][plan.qid]
            if meta.kind in self._refuse:
                answers.append({"type": "refusal", "name": plan.qid})
            elif meta.kind == "list_kind":
                graded = {"items": 0.67, "statements": 0.33}
                answers.append(choice_answer(plan.qid, "items", graded, 0.34))
            else:
                if plan.kind == "noul":
                    picked = "yes" if given["noul"] >= 0.5 else "no"
                elif plan.kind == "choice":
                    picked = given["choice"]
                else:
                    levels = given["probabilities"]
                    picked = max(plan.values, key=lambda v: levels.get(v, 0.0))
                saturated = {v: 1.0 if v == picked else 0.0 for v in plan.values}
                answers.append(choice_answer(plan.qid, picked, saturated, 1.0))
        clef = from_decisions_response(
            decisions_body(*answers), plans, clef_model=str(payload["model"])
        )
        return TransportResponse(body=clef, status_code=200, provider="decisions")


async def run_case(case_id: str, *, refuse: frozenset[str] = frozenset()) -> Any:
    (case,) = [c for c in CASES if c.id == case_id]
    assert case.world is not None
    extractor = create_extractor(
        case.schemas(),
        transport=DecisionsShaped(case.world(), refuse),
        **build_options({}, case.options),
    )
    return await extractor.ainvoke(case.text, existing=case.existing)


@pytest.mark.parametrize(
    "case_id",
    [
        "notes_update_contact",
        "notes_insert_contact",
        "note_beside_applied_change",
        "notes_people_insert",
        "readme_people",
    ],
)
async def test_the_notes_cases_add_their_notes_under_decisions_shaped_answers(
    case_id: str,
) -> None:
    (case,) = [c for c in CASES if c.id == case_id]
    result = await run_case(case_id)
    from evals.scoring import score

    scored = score(case, result)
    assert scored.n_notes > 0
    assert scored.notes_found == scored.n_notes, scored.missing


@pytest.mark.parametrize("case_id", ["delete_request", "order_remove_item"])
async def test_a_hard_yes_still_cannot_delete(case_id: str) -> None:
    # the cap (0.8) is below the delete gate (0.85) on purpose: a saturated "yes" from
    # a reader that says nothing about how sure it is cannot remove a record
    result = await run_case(case_id)
    assert not [op for op in actual_ops(result) if op.kind in ("delete", "remove")]
    assert any(
        w.startswith(HARD_PICKS_WARNING) and "delete" in w for w in result.warnings
    )


# --------------------------------------------------------------------------- #
# 2. A refusal is no answer
# --------------------------------------------------------------------------- #

REFUSAL = {"type": "refusal", "name": "plan"}  # the shape the live API sent


def two_questions() -> tuple[dict[str, Any], Any]:
    payload = {
        "model": "clef",
        "state": "Orbit Freight is on the business plan.",
        "questions": {
            "plan": {
                "type": "choice",
                "instructions": "Which plan?",
                "criteria": {"starter": "Starter", "business": "Business"},
            },
            "seats": {
                "type": "noul",
                "instructions": "Are seats stated?",
                "criteria": {"true": "Yes", "false": "No"},
            },
        },
    }
    return payload, to_decisions_request(payload)[1]


def test_a_refused_question_is_left_unanswered_and_named() -> None:
    payload, plans = two_questions()
    body = decisions_body(
        REFUSAL,
        choice_answer("seats", "yes", {"yes": 0.9, "no": 0.1}, 0.8),
    )
    clef = from_decisions_response(body, plans, clef_model="clef")
    assert list(clef["answers"]) == ["seats"]  # not 'plan': unanswered
    (note,) = clef["decisions_notes"]
    assert note.startswith("decisions_refusal") and "plan" in note
    assert ClefRequest.model_validate(payload)  # and the reply still parses as Clef


def test_a_response_of_only_refusals_is_no_answers_not_an_error() -> None:
    _, plans = two_questions()
    body = decisions_body(REFUSAL, {"type": "refusal", "name": "seats"})
    clef = from_decisions_response(body, plans, clef_model="clef")
    assert clef["answers"] == {}


def test_an_answer_that_is_not_a_refusal_needs_its_choice() -> None:
    from trustcall_system_one import ModelResponseError

    _, plans = two_questions()
    body = decisions_body(
        {"type": "choice", "name": "plan"},
        choice_answer("seats", "yes", {"yes": 1.0, "no": 0.0}, 1.0),
    )
    with pytest.raises(ModelResponseError, match="has no choice"):
        from_decisions_response(body, plans, clef_model="clef")


def test_a_refusal_of_a_question_that_was_not_asked_is_still_an_error() -> None:
    from trustcall_system_one import ModelResponseError

    _, plans = two_questions()
    body = decisions_body(
        REFUSAL,
        {"type": "refusal", "name": "nope"},
    )
    with pytest.raises(ModelResponseError, match="do not match the questions"):
        from_decisions_response(body, plans, clef_model="clef")


async def test_a_refusal_over_the_wire_is_a_warning_and_the_run_goes_on() -> None:
    payload, _ = two_questions()
    body = decisions_body(
        REFUSAL, choice_answer("seats", "yes", {"yes": 0.9, "no": 0.1}, 0.8)
    )
    transport, server = decisions(httpx.Response(200, json=body))
    run = Run(transport=transport)
    request = ClefRequest.model_validate(payload)
    answers = await run.call_clef(
        request.questions, request.state, meta=CallMeta(kind="clef_fields", schema="A")
    )
    assert set(answers) == {"seats"}
    assert any(w.startswith("decisions_refusal") for w in run.warnings)
    assert len(server.requests) == 1  # the refused question is not asked again
    again = await run.call_clef(
        request.questions, request.state, meta=CallMeta(kind="clef_fields", schema="A")
    )
    assert set(again) == {"seats"} and len(server.requests) == 1


async def test_a_refused_gate_is_never_a_pass() -> None:
    # every change gate of an update comes back as a refusal: nothing is written
    control = await run_case("contacts_update_add")
    assert [op for op in actual_ops(control) if op.kind == "add"]  # it does pass
    result = await run_case("contacts_update_add", refuse=frozenset({"gate"}))
    assert not [op for op in actual_ops(result) if op.kind in ("add", "replace")]
    assert any(w.startswith("decisions_refusal") for w in result.warnings)


# --------------------------------------------------------------------------- #
# 4. The warning of a primary provider's hard picks
# --------------------------------------------------------------------------- #


async def test_a_primary_providers_hard_picks_are_not_called_a_fallback() -> None:
    payload, _ = two_questions()
    only_noul = {
        "model": "clef",
        "state": "x",
        "questions": {"seats": payload["questions"]["seats"]},
    }
    body = decisions_body(choice_answer("seats", "yes", {"yes": 1.0, "no": 0.0}, 1.0))
    transport, _ = decisions(httpx.Response(200, json=body))
    run = Run(transport=transport)
    request = ClefRequest.model_validate(only_noul)
    answers = await run.call_clef(
        request.questions, request.state, meta=CallMeta(kind="attribution", schema="A")
    )
    assert answers["seats"].noul == 0.8  # type: ignore[union-attr]
    assert run.hard_picks == 1 and run.fallback_hard == 0 and run.fallback_used == 0
    (warning,) = run.warnings
    assert warning.startswith(f"{HARD_PICKS_WARNING}: 1 Clef answer(s) from decisions")
    assert "attribution 1" in warning
    assert FALLBACK_WARNING not in warning and "fallback" not in warning

    # the same request again is answered from the memo: no fallback "use" either
    await run.call_clef(
        request.questions, request.state, meta=CallMeta(kind="gate", schema="A")
    )
    assert run.fallback_fields == {}
    (warning,) = run.warnings
    assert warning.startswith(HARD_PICKS_WARNING)


async def test_a_graded_primary_makes_no_warning_at_all() -> None:
    body = decisions_body(choice_answer("seats", "yes", {"yes": 0.9, "no": 0.1}, 0.8))
    transport, _ = decisions(httpx.Response(200, json=body))
    run = Run(transport=transport)
    request = ClefRequest.model_validate(
        {
            "model": "clef",
            "state": "x",
            "questions": {"seats": two_questions()[0]["questions"]["seats"]},
        }
    )
    await run.call_clef(
        request.questions, request.state, meta=CallMeta(kind="gate", schema="A")
    )
    assert run.warnings == []


async def test_a_full_run_on_decisions_says_hard_picks_not_fallback() -> None:
    result = await run_case("notes_insert_contact")
    assert any(w.startswith(HARD_PICKS_WARNING) for w in result.warnings)
    assert not any(w.startswith(FALLBACK_WARNING) for w in result.warnings)


# --------------------------------------------------------------------------- #
# 5. What the reader sees of an entity that has no mentions
# --------------------------------------------------------------------------- #

BILL = {
    "input": "Invoice INV-2300 from Fabrikam totals $500.00, due on receipt.",
    "entity": {"label": "the Bill", "mentions": []},
}


def test_an_entity_without_mentions_says_what_its_label_is() -> None:
    # recorded: the label "the Bill" is a description, the input never says it, and
    # "mentions": [] anchors it to nothing; Decisions answered 0.4 and 0.27 for the
    # invoice number and the total that the one line states
    payload = {
        "model": "clef",
        "state": BILL,
        "questions": {
            "attr.total": {
                "type": "noul",
                "instructions": "Does `input` state the total of `entity.label`?",
                "criteria": {"true": "yes", "false": "no"},
            }
        },
    }
    body, _ = to_decisions_request(payload)
    text = body["input"]
    assert '"mentions": []' not in text
    assert "the input need not contain those words" in text
    assert '"label": "the Bill"' in text and "Invoice INV-2300" in text


def test_an_entity_with_mentions_is_shown_as_it_is() -> None:
    state = {
        "input": "Olivia Davis is 27.",
        "entity": {
            "label": "Olivia Davis",
            "mentions": [{"text": "Olivia Davis", "span": [0, 12]}],
        },
    }
    assert readable_state(state) is state
    assert readable_state("plain text") == "plain text"
    assert readable_state({"input": "x"}) == {"input": "x"}
    shown = readable_state(BILL)
    assert BILL["entity"] == {"label": "the Bill", "mentions": []}  # not mutated
    assert shown["entity"]["label"] == "the Bill" and "mentions" not in shown["entity"]
