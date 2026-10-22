"""Lists of text, sums and clauses on Jev: what the live benchmark found.

The shapes are those of the recordings of the first live run (Jev for both models):
the notes of an update that took "Berlin" and a phone number as notes of their own, a
discovery list that took a whole clause as a person's name, a total that Jev answered
with ``none`` because the sum was offered under the text of the sentence, and a
clause splitter that cut "$1,125.00" at its comma. Jev is scripted; nothing is sent.
"""

import re
from collections.abc import Callable
from typing import Any

import pytest

from tests.unit.test_typesafe_transport import (
    Jev,
    choice,
    clairic,
    clairic_payload,
    make,
)
from trustcall_system_one import TypeSafeConfig
from trustcall_system_one._structure import build_structure, to_lines

SENTENCE = re.compile(r"\[(S\d+)\] ((?:(?!\[S\d+\]).)*)")
NO_VERIFY = TypeSafeConfig(verify_selection=False)

NOTES_FIELD = {
    "notes": {
        "type": "list",
        "items": "text",
        "instructions": "Things worth remembering about the person",
    }
}


def members(
    accepted: set[str], kind: str = "statements", kind_p: float = 0.93
) -> Callable[..., dict[str, Any] | None]:
    """A Jev that says yes to the candidates whose text is in ``accepted``.

    A sentence is named by its id in the question, its text is in the state.
    """

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        _, _, rest = qid.partition("::")
        if rest == "kind":
            return choice(question, kind, kind_p)
        if rest.startswith("in"):
            sentences = {
                sid: text.strip()
                for sid, text in SENTENCE.findall(body["state"]["input"])
            }
            cand = question["instructions"]["candidate"]
            text = cand.get("text") or sentences.get(cand.get("id", ""), "")
            return {"type": "noul", "noul": 0.95 if text in accepted else 0.03}
        return None

    return script


def answer_of(
    text: str,
    field: dict[str, Any],
    accepted: set[str],
    kind: str,
    kind_p: float = 0.93,
) -> dict[str, Any]:
    jev = Jev(members(accepted, kind, kind_p))
    sent = clairic(make(jev, config=NO_VERIFY), clairic_payload(field, text))
    name = next(iter(field))
    answer: dict[str, Any] = sent.body["answers"][name]
    return answer


# --------------------------------------------------------------------------- #
# A list of statements is made of statements
# --------------------------------------------------------------------------- #

MOVE = (
    "Quick note: Dana moved to Berlin last month. Her mobile is now +49 30 5555 0198."
)
MOVE_SENTENCE = "Quick note: Dana moved to Berlin last month."


def test_a_name_inside_a_sentence_that_was_not_taken_is_not_a_note() -> None:
    """note_beside_applied_change: "Berlin" came back as a note beside the change."""
    answer = answer_of(
        MOVE,
        NOTES_FIELD,
        {"Berlin", "Her mobile is now +49 30 5555 0198."},
        "statements",
    )
    assert answer["value"] == ["Her mobile is now +49 30 5555 0198."]


def test_a_number_inside_an_accepted_sentence_is_not_a_second_note() -> None:
    """notes_update_contact / only_phone: the sentence and the number, both as notes."""
    answer = answer_of(
        MOVE,
        NOTES_FIELD,
        {
            MOVE_SENTENCE,
            "Berlin",
            "Her mobile is now +49 30 5555 0198.",
            "+49 30 5555 0198",
        },
        "statements",
    )
    assert answer["value"] == [MOVE_SENTENCE, "Her mobile is now +49 30 5555 0198."]
    state = clairic_payload(NOTES_FIELD, MOVE)["state"]
    assert [state[a:b] for a, b in answer["spans"]] == answer["value"]


def test_a_noun_phrase_is_not_a_note() -> None:
    text = "She got promoted to Senior Marketing Manager last month."
    answer = answer_of(
        text, NOTES_FIELD, {"Senior Marketing Manager", "rock climbing"}, "statements"
    )
    assert answer["value"] is None  # nothing but a phrase was taken


# --------------------------------------------------------------------------- #
# A list of short values keeps its values, and its names beat the clause around them
# --------------------------------------------------------------------------- #

MENTIONS = {
    "mentions": {
        "type": "list",
        "items": "text",
        "instructions": "How the input refers to each person",
    }
}
INTRO = "Her name is Olivia Davis, she's a 27-year-old graphic designer."
CLAUSE = "Olivia Davis, she's a 27-year-old graphic designer"


def test_a_name_beats_the_clause_that_contains_it() -> None:
    """readme_people: the person's label was the whole clause, so `name` became it."""
    answer = answer_of(INTRO, MENTIONS, {CLAUSE, "Olivia Davis", "Olivia"}, "items")
    assert CLAUSE not in answer["value"]
    assert "Olivia Davis" in answer["value"]


def test_a_clause_is_kept_when_no_value_inside_it_was_taken() -> None:
    """Mentions that are not names ("the tea towels at 5") stay phrases."""
    text = "Please add two more of the blue mugs and drop the tea towels at 5."
    answer = answer_of(text, MENTIONS, {"drop the tea towels at 5"}, "items")
    assert answer["value"] == ["drop the tea towels at 5"]


def test_phone_numbers_and_names_in_a_list_of_values_are_kept() -> None:
    field = {
        "phones": {"type": "list", "items": "text", "instructions": "Phone numbers"}
    }
    text = "Call 415-555-0100 or 415-555-0199 today."
    answer = answer_of(text, field, {"415-555-0100", "415-555-0199"}, "items")
    assert answer["value"] == ["415-555-0100", "415-555-0199"]
    names = {"people": {"type": "list", "items": "text", "instructions": "Contacts"}}
    answer = answer_of(
        "Contact: Dana Ruiz, Priya Shah.", names, {"Dana Ruiz", "Priya Shah"}, "items"
    )
    assert answer["value"] == ["Dana Ruiz", "Priya Shah"]


def test_a_sentence_over_the_names_it_holds_is_not_the_list() -> None:
    names = {"people": {"type": "list", "items": "text", "instructions": "Contacts"}}
    text = "Contact: Dana Ruiz, Priya Shah."
    answer = answer_of(text, names, {text, "Dana Ruiz", "Priya Shah"}, "items")
    assert answer["value"] == ["Dana Ruiz", "Priya Shah"]


def test_the_kind_is_items_unless_jev_is_sure_it_is_statements() -> None:
    accepted = {MOVE_SENTENCE, "Berlin"}
    unsure = answer_of(MOVE, NOTES_FIELD, accepted, "statements", kind_p=0.45)
    assert "Berlin" in unsure["value"]  # the default is items: nothing is dropped
    sure = answer_of(MOVE, NOTES_FIELD, accepted, "statements", kind_p=0.9)
    assert sure["value"] == [MOVE_SENTENCE]


def test_the_kind_is_one_more_question_in_the_same_request() -> None:
    jev = Jev(members({MOVE_SENTENCE}))
    clairic(make(jev, config=NO_VERIFY), clairic_payload(NOTES_FIELD, MOVE))
    (select,) = [r for r in jev.requests if "notes::in0" in r["questions"]]
    question = select["questions"]["notes::kind"]
    assert set(question["criteria"]) == {"items", "statements"}
    assert (
        "notes" in question["instructions"].lower()
        or "worth" in question["instructions"]
    )
    assert sum(1 for r in jev.requests if "notes::kind" in r["questions"]) == 1


def test_a_list_of_numbers_has_no_kind_question() -> None:
    field = {"n": {"type": "list", "items": "number", "instructions": "Quantities"}}
    jev = Jev(members(set()))
    clairic(make(jev, config=NO_VERIFY), clairic_payload(field, "We need 3 and 4."))
    assert not any("n::kind" in r["questions"] for r in jev.requests)


# --------------------------------------------------------------------------- #
# A number is not cut at its comma; a sum says what it is
# --------------------------------------------------------------------------- #

CORRECTION = "Correction: the total of INV-2291 is $1,125.00, not $1,080.00."


def test_a_comma_inside_a_number_is_not_a_clause_boundary() -> None:
    structure = build_structure(CORRECTION, to_lines(CORRECTION), {})
    texts = {c.text for c in structure.candidates}
    for fragment in ("125.00", "not $1", "080.00", "the total of INV-2291 is $1"):
        assert fragment not in texts
    assert "the total of INV-2291 is $1,125.00" in texts
    assert "not $1,080.00" in texts


def test_a_time_is_not_cut_at_its_colon() -> None:
    text = "Dana arrives at 2:30 pm, then leaves."
    texts = {c.text for c in build_structure(text, to_lines(text), {}).candidates}
    assert not any(t in ("30 pm", "Dana arrives at 2") for t in texts)


INVOICE = (
    "Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30. "
    "4 pallets at $250.00 each, shipping $80.00. Contact: Dana Ruiz, Priya Shah."
)
TOTAL_FIELD = {
    "total": {
        "type": "number",
        "instructions": "The Invoice's Amount due including shipping, in USD",
    }
}
SUM_ID = "1080 (the sum of 1000 + 80)"


def test_the_total_is_offered_as_the_number_it_is() -> None:
    """invoice_insert: the sum was offered under the sentence, Jev chose none."""

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid == "total::pick":
            return choice(
                question, SUM_ID if SUM_ID in question["criteria"] else "none"
            )
        return None

    jev = Jev(script)
    sent = clairic(make(jev, config=NO_VERIFY), clairic_payload(TOTAL_FIELD, INVOICE))
    (select,) = [r for r in jev.requests if "total::pick" in r["questions"]]
    question = select["questions"]["total::pick"]
    assert SUM_ID in question["criteria"]
    assert "the sum of" in question["instructions"]
    answer = sent.body["answers"]["total"]
    assert answer["value"] == 1080
    assert isinstance(answer["value"], int | float)  # a number, never the text
    state = clairic_payload(TOTAL_FIELD, INVOICE)["state"]
    start, end = answer["span"]
    assert state[start:end] == "4 pallets at $250.00 each, shipping $80.00"


@pytest.mark.parametrize("field", ["total", "amount"])
def test_a_number_field_never_returns_text(field: str) -> None:
    spec = {field: {"type": "number", "instructions": "Total amount due"}}

    def script(
        qid: str, question: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any] | None:
        if qid.endswith("::pick"):
            opts = list(question["criteria"])
            return choice(question, next(o for o in opts if o != "none"))
        return None

    jev = Jev(script)
    sent = clairic(make(jev, config=NO_VERIFY), clairic_payload(spec, INVOICE))
    value = sent.body["answers"][field]["value"]
    assert value is None or isinstance(value, int | float)
