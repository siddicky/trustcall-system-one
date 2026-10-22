"""End-to-end tests of the update path on a fake transport (single ``One`` schema)."""

# ruff: noqa: UP007, UP045 - tests deliberately spell Optional and Union

import datetime as dt
from typing import Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from tests.unit.extract_helpers import (
    Script,
    ca,
    kind_counts,
    kinds,
    make_extractor,
    pick,
)
from trustcall_system_one import (
    ExtractionResult,
    ExtractionValidationError,
    GateConfig,
    One,
    ResultRecord,
    SchemaInstance,
    VerifyConfig,
)
from trustcall_system_one.testing import choice

TEXT = (
    "Dana moved to Berlin last month. Her mobile is now +49 30 5555 0198. "
    "She is 41 now and asked for an intro to the logistics team."
)


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    age: Optional[int] = None
    notes: list[str] = Field(default_factory=list)


RECORD: dict[str, Any] = {
    "name": "Dana Whitfield",
    "email": "dana@fabrikam.example",
    "phone": "+1 415 555 0134",
    "city": "Austin",
    "age": 40,
    "notes": ["Prefers email over phone"],
}


def script(**changes: Any) -> Script:
    """A world where the note states a new city, phone, age and note."""
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana", find="Dana", p_null=0.05, confidence=0.9),
        "email": ca("text", None, p_null=0.98, confidence=0.9),
        "phone": ca("text", "+49 30 5555 0198", find="+49 30 5555 0198"),
        "city": ca("text", "Berlin", find="Berlin", confidence=0.95),
        "age": ca("number", 41, find="41", confidence=0.95),
        "notes": ca(
            "list",
            ["asked for an intro to the logistics team"],
            find_all=["asked for an intro to the logistics team"],
            p_null=0.05,
            confidence=0.95,
        ),
    }
    for name, value in changes.items():
        setattr(world, name, value)
    return world


async def update(
    world: Script,
    text: str = TEXT,
    record: Any = None,
    *,
    schema: Any = None,
    **kwargs: Any,
) -> ExtractionResult:
    extractor = make_extractor(schema or [One(Contact)], world, **kwargs)
    existing = [("rec-1", "Contact", RECORD if record is None else record)]
    return await extractor.ainvoke(text, existing=existing)


def only(result: ExtractionResult) -> ResultRecord:
    (record,) = result.records
    return record


def ops(record: ResultRecord) -> list[tuple[str, str]]:
    return [(p.op, p.path) for p in record.patches]


def reasons(record: ResultRecord) -> dict[str, str]:
    return {r.path: r.reason for r in record.rejected}


def calls_of(world: Script, kind: str) -> list[Any]:
    return world.calls(kind)


# --------------------------------------------------------------------------- #
# The central guarantee
# --------------------------------------------------------------------------- #


async def test_a_field_the_input_never_mentions_keeps_its_value() -> None:
    world = script()
    result = await update(world)
    record = only(result)
    assert (record.kind, record.json_doc_id, record.valid) == (
        "update",
        "rec-1",
        True,
    )
    assert record.after is not None and record.before == RECORD
    # the email is not in the text: unchanged, no operation, one explained rejection
    assert record.after["email"] == RECORD["email"]
    assert all(p.path != "/email" for p in record.patches)
    email = [r for r in record.rejected if r.path == "/email"]
    assert [(r.op, r.reason) for r in email] == [(None, "not_mentioned")]
    # and nobody asked the gate about it
    gated = [c[1]["state"]["proposed_change"]["path"] for c in calls_of(world, "gate")]
    assert "/email" not in gated
    assert sorted(gated) == ["/age", "/city", "/notes/-", "/phone"]


async def test_the_patch_is_computed_in_code_in_a_fixed_order() -> None:
    result = await update(script())
    record = only(result)
    # replaces (in field order) before the append
    assert ops(record) == [
        ("replace", "/phone"),
        ("replace", "/city"),
        ("replace", "/age"),
        ("add", "/notes/-"),
    ]
    assert record.after == {
        **RECORD,
        "phone": "+49 30 5555 0198",
        "city": "Berlin",
        "age": 41,
        "notes": [
            "Prefers email over phone",
            "asked for an intro to the logistics team",
        ],
    }
    assert record.value is not None and record.value.age == 41  # type: ignore[attr-defined]
    assert result.responses == (record.value,)
    assert result.response_metadata[0]["json_doc_id"] == "rec-1"
    assert result.untouched == ()


async def test_the_name_short_form_is_an_identity_subset_not_a_rename() -> None:
    world = script()
    record = only(await update(world))
    assert record.after is not None and record.after["name"] == "Dana Whitfield"
    (rejected,) = [r for r in record.rejected if r.path == "/name"]
    assert (rejected.op, rejected.reason) == ("replace", "identity_subset")
    gated = [c[1]["state"]["proposed_change"]["path"] for c in calls_of(world, "gate")]
    assert "/name" not in gated


async def test_a_run_with_no_change_is_still_an_update_record() -> None:
    world = script()
    world.clairic = {
        "name": ca("text", "Dana Whitfield", find="Dana"),
        "email": ca("text", None, p_null=0.98),
        "phone": ca("text", "+1 415 555 0134", find="+49"),
        "city": ca("text", "Austin", find="Berlin"),
        "age": ca("number", 40, find="41"),
        "notes": ca("list", None, p_null=0.98),
    }
    result = await update(world)
    record = only(result)
    assert record.kind == "update" and record.valid
    assert record.patches == ()
    assert record.after == record.before == RECORD
    assert record.value is not None and result.responses == (record.value,)
    assert kind_counts(world.transport)["gate"] == 0  # nothing to judge


# --------------------------------------------------------------------------- #
# The record is shown after the input, and not in the cross-check
# --------------------------------------------------------------------------- #


async def test_the_record_follows_the_input_and_the_cross_check_lacks_it() -> None:
    world = script()
    await update(world)
    extract = calls_of(world, "extract")
    cross = calls_of(world, "crosscheck")
    assert len(extract) == len(cross) == 1
    state = extract[0][1]["state"]
    assert state.index("</input>") < state.index(
        '<existing id="rec-1" schema="Contact">'
    )
    assert '"name": "Dana Whitfield"' in state[state.index("<existing") :]
    assert state.endswith("</existing>")
    assert "<existing" not in cross[0][1]["state"]
    # the same fields are asked both times
    assert sorted(extract[0][1]["fields"]) == sorted(cross[0][1]["fields"])


async def test_the_pipeline_runs_in_this_order() -> None:
    world = script()
    await update(world)
    seen = kinds(world.transport)
    assert seen[0] == "presence"
    assert sorted(seen[1:3]) == ["crosscheck", "extract"]
    assert seen[3] == "attribution"
    # a changed age is not an ordered quantity: no relative-edit request for it
    assert seen[4:] == ["gate"] * 4


# --------------------------------------------------------------------------- #
# Echoes: the answer was read from the record
# --------------------------------------------------------------------------- #


async def test_an_answer_whose_span_lies_in_the_record_is_rejected() -> None:
    world = script()
    # "Austin" is only in the <existing> block: the span points there
    world.clairic["city"] = ca("text", "Austin", find="Austin", p_null=0.02)
    world.cross["city"] = ca("text", None, p_null=0.97)  # the input alone has none
    record = only(await update(world))
    assert reasons(record)["/city"] == "echo_span"
    assert record.after is not None and record.after["city"] == "Austin"
    assert record.fields["city"].rejected_reason == "echo_span"
    assert ("replace", "/city") not in ops(record)


async def test_a_list_element_read_from_the_record_is_dropped_as_an_echo() -> None:
    world = script()
    world.clairic["notes"] = ca(
        "list",
        ["Prefers email over phone", "asked for an intro to the logistics team"],
        find_all=[
            "Prefers email over phone",
            "asked for an intro to the logistics team",
        ],
        p_null=0.05,
        confidence=0.95,
    )
    record = only(await update(world))
    assert ops(record)[-1] == ("add", "/notes/-")
    echoed = [r for r in record.rejected if r.reason == "echo_span"]
    assert [r.path for r in echoed] == ["/notes/0"]
    assert record.after is not None
    assert record.after["notes"] == [
        "Prefers email over phone",
        "asked for an intro to the logistics team",
    ]


async def test_a_number_without_a_span_is_caught_by_the_cross_check() -> None:
    world = script()
    # the model echoes the record's age; there is no span to give it away
    world.clairic["age"] = ca("number", 77, confidence=0.95, p_null=0.02)
    world.cross["age"] = ca("number", None, p_null=0.97, confidence=0.9)
    record = only(await update(world))
    assert reasons(record)["/age"] == "echo_crosscheck"
    assert record.after is not None and record.after["age"] == 40
    assert record.fields["age"].crosscheck_agree is False
    assert all(
        c[1]["state"]["proposed_change"]["path"] != "/age"
        for c in calls_of(world, "gate")
    )


async def test_the_cross_check_agreeing_lets_the_change_through() -> None:
    world = script()
    record = only(await update(world))
    assert record.fields["city"].crosscheck_agree is True
    assert record.fields["age"].crosscheck_agree is True
    assert record.fields["email"].crosscheck_agree is None  # nothing to check


@pytest.mark.parametrize(
    ("choice", "p", "expected_city", "reason"),
    [
        ("a", 0.9, "Berlin", None),
        ("b", 0.9, "Hamburg", None),
        ("neither", 0.9, "Austin", "crosscheck_disagree"),
        ("a", 0.4, "Austin", "crosscheck_disagree"),
    ],
)
async def test_two_different_readings_go_to_a_clef_choice(
    choice: str, p: float, expected_city: str, reason: str | None
) -> None:
    text = TEXT + " Hamburg is nice."
    world = script()
    world.clairic["city"] = ca("text", "Berlin", find="Berlin")
    world.cross["city"] = ca("text", "Hamburg", find="Hamburg")
    options = ["a", "b", "neither"]
    world.adjudicate["city"] = pick(choice, options, p=p)
    record = only(await update(world, text))
    assert record.after is not None and record.after["city"] == expected_city
    assert reasons(record).get("/city") == reason
    (call,) = calls_of(world, "adjudicate")
    question = next(iter(call[1]["questions"].values()))
    assert question["criteria"] == {
        "a": "Berlin",
        "b": "Hamburg",
        "neither": "The input states neither",
    }


async def test_the_cross_check_can_be_switched_off() -> None:
    world = script()
    result = await update(world, verify=VerifyConfig(input_only_crosscheck=False))
    assert kind_counts(world.transport)["crosscheck"] == 0
    assert ("replace", "/city") in ops(only(result))


async def test_a_new_list_element_needs_the_cross_check_too() -> None:
    world = script()
    world.cross["notes"] = ca("list", None, p_null=0.97)
    record = only(await update(world))
    assert ("add", "/notes/-") not in ops(record)
    dropped = [r for r in record.rejected if r.reason == "echo_crosscheck"]
    assert dropped and dropped[0].path.startswith("/notes")


# --------------------------------------------------------------------------- #
# Attribution and the gate are both needed
# --------------------------------------------------------------------------- #


async def test_a_failed_attribution_blocks_an_operation_the_gate_would_pass() -> None:
    world = script()
    world.attribution = {"attr.city": 0.05}
    world.gate_default = 0.99
    record = only(await update(world))
    assert reasons(record)["/city"] == "attribution"
    assert ("replace", "/city") not in ops(record)
    gated = [c[1]["state"]["proposed_change"]["path"] for c in calls_of(world, "gate")]
    assert "/city" not in gated  # never asked: attribution came first


async def test_a_failed_gate_blocks_an_operation_attribution_passed() -> None:
    world = script()
    world.attribution_default = 0.99
    world.gate = {"/city": 0.1}
    record = only(await update(world))
    (rejected,) = [r for r in record.rejected if r.path == "/city"]
    assert (rejected.op, rejected.reason, rejected.p) == ("replace", "gate_noul", 0.1)
    assert record.after is not None and record.after["city"] == "Austin"
    assert record.fields["city"].gate_p == 0.1
    assert record.fields["city"].rejected_reason == "gate_noul"
    assert record.fields["city"].value is None
    assert record.fields["phone"].gate_p == 0.95
    assert ("replace", "/phone") in ops(record)


@pytest.mark.parametrize(
    ("path", "p", "applied"),
    [
        ("/city", 0.59, False),  # replace needs 0.6
        ("/city", 0.6, True),
        ("/notes/-", 0.54, False),  # add needs 0.55
        ("/notes/-", 0.55, True),
    ],
)
async def test_the_gate_thresholds_depend_on_the_kind_of_operation(
    path: str, p: float, applied: bool
) -> None:
    world = script()
    world.gate = {path: p}
    record = only(await update(world))
    assert any(op.path == path for op in record.patches) is applied


async def test_the_gate_thresholds_are_configurable() -> None:
    world = script()
    world.gate = {"/city": 0.7}
    gates = GateConfig(replace_threshold=0.8)
    record = only(await update(world, gates=gates))
    assert reasons(record)["/city"] == "gate_noul"


async def test_the_gate_uses_the_gate_model_and_one_request_per_operation() -> None:
    world = script()
    extractor = make_extractor(
        [One(Contact)], world, gate_model="@cf/cloudflare/clef-flash"
    )
    await extractor.ainvoke(TEXT, existing=[("rec-1", "Contact", RECORD)])
    gates = calls_of(world, "gate")
    assert len(gates) == 4
    assert {c[0] for c in gates} == {"@cf/cloudflare/clef-flash"}
    assert all(len(c[1]["questions"]) == 1 for c in gates)
    state = gates[0][1]["state"]
    assert set(state) == {"input", "entity", "record", "proposed_change"}
    assert state["record"] == RECORD
    # a single record is named by what it is called (its identity), not only by type
    assert state["entity"] == {"label": "the Contact Dana Whitfield", "mentions": []}


# --------------------------------------------------------------------------- #
# Presence, updates off, opaque records
# --------------------------------------------------------------------------- #


async def test_a_record_the_input_does_not_mention_is_untouched() -> None:
    world = script(presence=0.03)
    result = await update(world)
    assert result.records == () and result.responses == ()
    assert result.untouched == ("rec-1",)
    assert kinds(world.transport) == ["presence"]
    assert result.attempts == 0


async def test_disabled_updates_skip_the_record_after_the_presence_gate() -> None:
    world = script()
    result = await update(world, enable_updates=False, enable_inserts=True)
    assert result.records == ()
    (skipped,) = result.skipped
    assert skipped.reason == "updates_disabled" and skipped.schema_name == "Contact"
    assert "rec-1" in skipped.detail
    assert kinds(world.transport) == ["presence"]
    assert result.untouched == ()  # it was found, just not updated


async def test_disabled_updates_still_leave_an_absent_record_untouched() -> None:
    world = script(presence=0.03)
    result = await update(world, enable_updates=False, enable_inserts=True)
    assert result.skipped == () and result.untouched == ("rec-1",)


class Ticket(BaseModel):
    """A support ticket."""

    subject: str


async def test_records_of_unknown_schemas_are_opaque_context() -> None:
    world = script()
    extractor = make_extractor([One(Contact)], world, existing_schema_policy=False)
    existing = [
        SchemaInstance("rec-1", "Contact", RECORD),
        SchemaInstance("t-9", "Ticket", {"subject": "printer"}),
    ]
    result = await extractor.ainvoke(TEXT, existing=existing)
    assert only(result).json_doc_id == "rec-1"
    assert result.untouched == ("t-9",)
    assert any("t-9" in w and "context only" in w for w in result.warnings)
    # never shown to a model, never patched
    for call in world.calls("extract"):
        assert "printer" not in call[1]["state"]


async def test_an_unknown_schema_raises_by_default_and_is_dropped_on_request() -> None:
    world = script()
    existing = [("t-9", "Ticket", {"subject": "printer"})]
    extractor = make_extractor([One(Contact)], world)
    with pytest.raises(ValueError, match="Ticket"):
        await extractor.ainvoke(TEXT, existing=existing)
    quiet = make_extractor([One(Contact)], world, existing_schema_policy="ignore")
    result = await quiet.ainvoke(TEXT, existing=existing)
    assert result.untouched == ()  # dropped, so not even listed
    assert only(result).kind == "insert"  # no record of Contact: an insert


async def test_a_schema_without_a_record_is_inserted() -> None:
    world = script()
    world.clairic = {"subject": ca("text", "printer", find="printer")}
    extractor = make_extractor([One(Ticket)], world, existing_schema_policy=False)
    result = await extractor.ainvoke(
        "the printer is broken", existing=[("c-1", "Contact", RECORD)]
    )
    assert only(result).kind == "insert"
    assert result.untouched == ("c-1",)


async def test_any_applies_to_the_extractors_schema() -> None:
    world = script()
    extractor = make_extractor([Contact], world)
    result = await extractor.ainvoke(TEXT, existing={"__any__": RECORD})
    record = only(result)
    assert record.kind == "update" and record.json_doc_id == "__any__"


# --------------------------------------------------------------------------- #
# Cardinality
# --------------------------------------------------------------------------- #


async def test_dict_form_implies_one() -> None:
    world = script()
    extractor = make_extractor([Contact], world)
    result = await extractor.ainvoke(TEXT, existing={"Contact": RECORD})
    assert only(result).kind == "update"


async def test_one_with_several_records_suggests_many() -> None:
    world = script()
    extractor = make_extractor([One(Contact)], world)
    existing = [("a", "Contact", RECORD), ("b", "Contact", RECORD)]
    with pytest.raises(ValueError, match="Many"):
        await extractor.ainvoke(TEXT, existing=existing)


async def test_an_explicit_one_takes_the_single_list_form_record() -> None:
    world = script()
    result = await update(world)
    assert only(result).json_doc_id == "rec-1"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class Account(BaseModel):
    """An account."""

    owner: str
    balance: Optional[int] = Field(default=None, ge=0)
    note: Optional[str] = None


async def test_an_operation_that_breaks_validation_is_dropped() -> None:
    world = Script()
    world.clairic = {
        "owner": ca("text", "Ada Lovelace", find="Ada Lovelace"),
        "balance": ca("number", -5, find="-5"),
        "note": ca("text", "overdrawn", find="overdrawn"),
    }
    extractor = make_extractor([One(Account)], world)
    record = only(
        await extractor.ainvoke(
            "Ada Lovelace: balance -5, overdrawn",
            existing=[("a-1", "Account", {"owner": "Ada", "balance": 10})],
        )
    )
    assert record.valid, record.errors
    assert record.after == {"owner": "Ada Lovelace", "balance": 10, "note": "overdrawn"}
    (bad,) = [r for r in record.rejected if r.reason == "validation"]
    assert (bad.op, bad.path) == ("replace", "/balance")
    assert "balance" in bad.detail
    assert ("replace", "/balance") not in ops(record)
    assert ("replace", "/owner") in ops(record)
    assert record.fields["balance"].rejected_reason == "validation"


async def test_a_record_that_stays_invalid_is_reported_not_raised() -> None:
    world = Script()
    world.clairic = {
        "owner": ca("text", None, p_null=0.99),
        "balance": ca("number", 5, find="5"),
        "note": ca("text", None, p_null=0.99),
    }
    extractor = make_extractor([One(Account)], world)
    # the stored record lacks its required owner
    result = await extractor.ainvoke(
        "balance 5", existing=[("a-1", "Account", {"balance": 10})]
    )
    record = only(result)
    assert not record.valid and record.value is None
    assert record.errors and record.errors[0].startswith("owner:")
    assert result.responses == ()
    assert record.after == {"balance": 5}  # the operation itself was fine


async def test_a_record_that_stays_invalid_raises_when_strict() -> None:
    world = Script()
    world.clairic = {
        "owner": ca("text", None, p_null=0.99),
        "balance": ca("number", 5, find="5"),
        "note": ca("text", None, p_null=0.99),
    }
    extractor = make_extractor([One(Account)], world, strict=True)
    with pytest.raises(ExtractionValidationError) as info:
        await extractor.ainvoke(
            "balance 5", existing=[("a-1", "Account", {"balance": 10})]
        )
    assert info.value.record is not None and info.value.record.kind == "update"


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #


class Aliased(BaseModel):
    full_name: str = Field(alias="fullName")
    home_city: Optional[str] = Field(default=None, alias="homeCity")


async def test_operations_use_the_keys_the_schema_validates() -> None:
    world = Script()
    world.clairic = {
        "full_name": ca("text", "Dana Whitfield", find="Dana Whitfield"),
        "home_city": ca("text", "Berlin", find="Berlin"),
    }
    extractor = make_extractor([One(Aliased)], world)
    # the record uses field names (as a model dumped without by_alias does)
    record = only(
        await extractor.ainvoke(
            "Dana Whitfield lives in Berlin",
            existing=[
                ("x", "Aliased", {"full_name": "Dana Whitfield", "home_city": "Rome"})
            ],
        )
    )
    assert record.before == {"fullName": "Dana Whitfield", "homeCity": "Rome"}
    assert ops(record) == [("replace", "/homeCity")]
    assert record.after == {"fullName": "Dana Whitfield", "homeCity": "Berlin"}
    state = world.calls("extract")[0][1]["state"]
    assert '"homeCity": "Rome"' in state  # the block shows the normalised record


# --------------------------------------------------------------------------- #
# Lists
# --------------------------------------------------------------------------- #


async def test_a_near_duplicate_list_element_is_checked_with_clef() -> None:
    world = script()
    world.clairic["notes"] = ca(
        "list",
        ["prefers email to phone calls"],
        find_all=["asked for an intro to the logistics team"],
    )
    # the scripted element text is not in the input: use a text that is
    text = TEXT + " Prefers email to phone calls."
    world.clairic["notes"] = ca(
        "list",
        ["Prefers email to phone calls"],
        find_all=["Prefers email to phone calls"],
    )
    world.dedup = {"Prefers email to phone calls": 0.9}
    record = only(await update(world, text))
    assert ("add", "/notes/-") not in ops(record)
    (dup,) = [r for r in record.rejected if r.reason == "duplicate"]
    assert dup.op == "add" and dup.path == "/notes/-" and dup.p == 0.9
    (call,) = calls_of(world, "list_dedup")
    (question,) = call[1]["questions"].values()
    assert '"Prefers email to phone calls"' in question["instructions"]
    assert '"Prefers email over phone"' in question["instructions"]


async def test_a_near_duplicate_that_is_not_one_is_appended() -> None:
    world = script()
    text = TEXT + " Prefers email to phone calls."
    world.clairic["notes"] = ca(
        "list",
        ["Prefers email to phone calls"],
        find_all=["Prefers email to phone calls"],
    )
    world.dedup = {"Prefers email to phone calls": 0.1}
    record = only(await update(world, text))
    assert ("add", "/notes/-") in ops(record)


# --------------------------------------------------------------------------- #
# Removals
# --------------------------------------------------------------------------- #


async def test_a_silent_field_is_never_removed_by_default() -> None:
    world = script()
    result = await update(world)
    assert not [p for p in only(result).patches if p.op == "remove"]


async def test_removals_are_opt_in_and_gated_with_the_removal_threshold() -> None:
    world = script()
    world.gate = {"/email": 0.9}
    record = only(await update(world, gates=GateConfig(allow_null_removals=True)))
    # email is nullable: set to null (replace with null) when the gate says so
    assert ("replace", "/email") in ops(record)
    assert record.after is not None and record.after["email"] is None
    (call,) = [
        c
        for c in calls_of(world, "gate")
        if c[1]["state"]["proposed_change"]["path"] == "/email"
    ]
    (question,) = call[1]["questions"].values()
    assert "removed or no longer applies" in question["instructions"]
    assert call[1]["state"]["proposed_change"]["to"] is None


async def test_a_removal_below_the_removal_threshold_is_refused() -> None:
    world = script()
    world.gate = {"/email": 0.8}  # add/replace would pass; removal needs 0.85
    record = only(await update(world, gates=GateConfig(allow_null_removals=True)))
    assert reasons(record)["/email"] == "gate_noul"
    assert record.after is not None and record.after["email"] == RECORD["email"]


# --------------------------------------------------------------------------- #
# Nested sub-models and unions
# --------------------------------------------------------------------------- #


class Address(BaseModel):
    """An address."""

    city: str
    zip: Optional[str] = None


class Author(BaseModel):
    name: str
    address: Optional[Address] = None


class Book(BaseModel):
    title: str
    author: Author
    isbn: Optional[str] = None


BOOK_TEXT = "The book is now titled Dune Messiah. Its author lives in Portland 97201."


def book_script() -> Script:
    world = Script()
    world.clairic = {
        "title": ca("text", "Dune Messiah", find="Dune Messiah"),
        "isbn": ca("text", None, p_null=0.98),
        "author.name": ca("text", None, p_null=0.98),
        "author.address.city": ca("text", "Portland", find="Portland"),
        "author.address.zip": ca("text", "97201", find="97201"),
    }
    return world


async def test_a_sub_model_the_record_lacks_is_added_as_one_operation() -> None:
    world = book_script()
    extractor = make_extractor([One(Book)], world)
    record = only(
        await extractor.ainvoke(
            BOOK_TEXT,
            existing=[("b-1", "Book", {"title": "Dune", "author": {"name": "Frank"}})],
        )
    )
    assert ops(record) == [("replace", "/title"), ("add", "/author/address")]
    add = record.patches[1]
    assert add.value == {"city": "Portland", "zip": "97201"}
    assert record.after == {
        "title": "Dune Messiah",
        "author": {"name": "Frank", "address": {"city": "Portland", "zip": "97201"}},
    }
    # gated once, as a whole
    gated = [c[1]["state"]["proposed_change"] for c in world.calls("gate")]
    (change,) = [g for g in gated if g["path"] == "/author/address"]
    assert change["op"] == "add" and change["from"] is None
    assert change["to"] == {"city": "Portland", "zip": "97201"}
    assert change["field"] == "author.address"
    assert reasons(record).get("/author/name") == "not_mentioned"


async def test_a_sub_model_the_record_has_is_diffed_field_by_field() -> None:
    world = book_script()
    extractor = make_extractor([One(Book)], world)
    record = only(
        await extractor.ainvoke(
            BOOK_TEXT,
            existing=[
                (
                    "b-1",
                    "Book",
                    {
                        "title": "Dune Messiah",
                        "author": {
                            "name": "Frank",
                            "address": {"city": "Reno", "zip": "97201"},
                        },
                    },
                )
            ],
        )
    )
    assert ops(record) == [("replace", "/author/address/city")]
    assert record.after is not None
    assert record.after["author"]["address"] == {"city": "Portland", "zip": "97201"}


async def test_an_optional_sub_model_the_input_does_not_state_is_left_alone() -> None:
    world = book_script()
    world.group_presence = {"author.address": 0.02}
    extractor = make_extractor([One(Book)], world)
    existing = {
        "title": "Dune",
        "author": {"name": "Frank", "address": {"city": "Reno"}},
    }
    record = only(
        await extractor.ainvoke(BOOK_TEXT, existing=[("b", "Book", existing)])
    )
    assert ops(record) == [("replace", "/title")]
    assert record.after is not None
    assert record.after["author"] == existing["author"]
    assert reasons(record)["/author/address"] == "not_mentioned"
    assert not [
        c for c in world.calls("extract") if "author.address.city" in c[1]["fields"]
    ]


class Cat(BaseModel):
    kind: Literal["cat"] = "cat"
    name: str
    lives: Optional[int] = None


class Dog(BaseModel):
    kind: Literal["dog"] = "dog"
    name: str
    good: Optional[bool] = None


class Owner(BaseModel):
    name: str
    pet: Union[Cat, Dog]


OWNER_TEXT = "Sam now has a dog called Rex."


def owner_script(member: str) -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Sam", find="Sam"),
        "pet.Cat.name": ca("text", "Rex", find="Rex"),
        "pet.Cat.lives": ca("number", None, p_null=0.98),
        "pet.Dog.name": ca("text", "Rex", find="Rex"),
        "pet.Dog.good": ca("text", None, p_null=0.98),
    }
    world.clef = {"pet": pick(member, ["Cat", "Dog"])}
    world.clef["pet.Dog.good"] = pick("unstated", ["yes", "no", "unstated"])
    return world


async def test_a_union_member_switch_is_one_replace_of_the_whole_value() -> None:
    world = owner_script("Dog")
    extractor = make_extractor([One(Owner)], world)
    old = {"name": "Sam", "pet": {"kind": "cat", "name": "Tom", "lives": 9}}
    record = only(await extractor.ainvoke(OWNER_TEXT, existing=[("o", "Owner", old)]))
    assert ops(record) == [("replace", "/pet")]
    assert record.patches[0].value["name"] == "Rex"
    assert record.after is not None and record.after["pet"]["name"] == "Rex"
    assert record.after["pet"].get("lives") is None  # the cat is gone
    assert record.after["pet"]["kind"] == "dog"  # the tag is written out
    assert isinstance(record.value, Owner) and isinstance(record.value.pet, Dog)
    (change,) = [
        c[1]["state"]["proposed_change"]
        for c in world.calls("gate")
        if c[1]["state"]["proposed_change"]["path"] == "/pet"
    ]
    assert change["op"] == "replace" and change["from"] == old["pet"]


async def test_the_same_union_member_is_diffed_field_by_field() -> None:
    world = owner_script("Dog")
    extractor = make_extractor([One(Owner)], world)
    old = {"name": "Sam", "pet": {"kind": "dog", "name": "Max"}}
    record = only(await extractor.ainvoke(OWNER_TEXT, existing=[("o", "Owner", old)]))
    assert ops(record) == [("replace", "/pet/name")]
    assert record.after == {"name": "Sam", "pet": {"kind": "dog", "name": "Rex"}}


async def test_a_union_the_record_lacks_is_added_whole() -> None:
    class Loose(BaseModel):
        name: str
        pet: Optional[Union[Cat, Dog]] = None

    world = owner_script("Dog")
    world.clef = {
        "pet": pick("Dog", ["Cat", "Dog", "unstated"]),
        "pet.Dog.good": pick("unstated", ["yes", "no", "unstated"]),
    }
    extractor = make_extractor([One(Loose)], world)
    record = only(
        await extractor.ainvoke(OWNER_TEXT, existing=[("o", "Loose", {"name": "Sam"})])
    )
    assert ops(record) == [("add", "/pet")]
    assert record.after is not None and record.after["pet"]["name"] == "Rex"


# --------------------------------------------------------------------------- #
# Clef-typed fields
# --------------------------------------------------------------------------- #


class Plan(BaseModel):
    customer: str
    tier: Literal["free", "pro"] = "free"
    active: bool = True


async def test_clef_typed_fields_are_updated_through_the_gate() -> None:
    world = Script()
    world.clairic = {"customer": ca("text", "Acme", find="Acme")}
    world.clef = {
        "tier": pick("pro", ["free", "pro", "unstated"]),
        "active": 0.9,  # type: ignore[dict-item]
    }
    world.clef["active"] = noul_answer(0.9)
    extractor = make_extractor([One(Plan)], world)
    record = only(
        await extractor.ainvoke(
            "Acme upgraded to pro",
            existing=[
                ("p", "Plan", {"customer": "Acme", "tier": "free", "active": True})
            ],
        )
    )
    assert ops(record) == [("replace", "/tier")]
    assert record.after == {"customer": "Acme", "tier": "pro", "active": True}


def noul_answer(p: float) -> dict[str, Any]:
    from trustcall_system_one.testing import noul

    return noul(p)


# --------------------------------------------------------------------------- #
# Long inputs: only the input is chunked, the record is never cut
# --------------------------------------------------------------------------- #


async def test_a_truncated_state_chunks_the_input_and_keeps_the_record_whole() -> None:
    filler = "The weather was mild and nothing else happened. "
    text = filler * 40 + "Dana moved to Berlin last month. " + filler * 40
    world = script()
    world.clairic = {
        "name": ca("text", None, p_null=0.98),
        "email": ca("text", None, p_null=0.98),
        "phone": ca("text", None, p_null=0.98),
        "city": ca("text", "Berlin", find="Berlin", confidence=0.95),
        "age": ca("number", None, p_null=0.98),
        "notes": ca("list", None, p_null=0.98),
    }
    world.limit = lambda state, n: 1500 if len(state) > 1500 else None
    result = await update(world, text, verify=VerifyConfig(window_chars=100))
    record = only(result)
    assert ops(record) == [("replace", "/city")]
    assert any("clairic_chunked" in w for w in result.warnings)
    record_json = '{"name": "Dana Whitfield"'
    chunked = [c for c in world.calls("extract") if len(c[1]["state"]) < 1500]
    assert len(chunked) >= 2
    for _, payload, _ in chunked:
        state = payload["state"]
        assert state.endswith("</existing>")
        assert record_json in state[state.index("<existing") :]
        assert '"notes": ["Prefers email over phone"]' in state
    # the cross-check chunks the same way, and its chunks never carry the record
    cross = [c for c in world.calls("crosscheck") if len(c[1]["state"]) < 1500]
    assert len(cross) >= 2
    assert all("<existing" not in c[1]["state"] for c in cross)
    assert record.fields["city"].crosscheck_agree is True


async def test_a_chunked_answer_read_from_the_record_is_rejected_as_an_echo() -> None:
    filler = "The weather was mild and nothing else happened. "
    text = filler * 40 + "Dana moved. " + filler * 40
    world = script()
    world.clairic = {
        "name": ca("text", None, p_null=0.98),
        "email": ca("text", None, p_null=0.98),
        "phone": ca("text", None, p_null=0.98),
        "city": ca("text", "Austin", find="Austin", confidence=0.95, p_null=0.02),
        "age": ca("number", None, p_null=0.98),
        "notes": ca("list", None, p_null=0.98),
    }
    # the chunks that carry the record answer from it
    world.limit = lambda state, n: 1500 if len(state) > 1500 else None
    record = only(await update(world, text, verify=VerifyConfig(window_chars=100)))
    assert ops(record) == []
    assert reasons(record)["/city"] == "echo_span"
    assert record.after == RECORD


# --------------------------------------------------------------------------- #
# The cross-check compares like with like
# --------------------------------------------------------------------------- #


class Event(BaseModel):
    title: str
    day: Optional[dt.date] = None
    place: Optional[str] = None


async def test_dates_must_agree_exactly_and_are_never_fuzzy() -> None:
    world = Script()
    world.clairic = {
        "title": ca("text", None, p_null=0.98),
        "day": ca("date", "2025-03-04", find="March 4", confidence=0.95),
        "place": ca("text", None, p_null=0.98),
    }
    world.cross = {"day": ca("date", "2025-03-05", find="March 4", confidence=0.95)}
    world.adjudicate = {"day": pick("neither", ["a", "b", "neither"], p=0.9)}
    extractor = make_extractor([One(Event)], world)
    record = only(
        await extractor.ainvoke(
            "The party is on March 4.",
            existing=[("e", "Event", {"title": "Party", "day": "2025-01-01"})],
        )
    )
    # one digit apart is a different date, not "similar text": adjudicated
    assert len(calls_of(world, "adjudicate")) == 1
    assert reasons(record)["/day"] == "crosscheck_disagree"
    assert record.after == {"title": "Party", "day": "2025-01-01"}


async def test_free_text_may_agree_on_similarity() -> None:
    world = Script()
    world.clairic = {
        "title": ca("text", None, p_null=0.98),
        "day": ca("date", None, p_null=0.98),
        "place": ca("text", "Frankfurt am Main", find="Frankfurt am Main"),
    }
    world.cross = {"place": ca("text", "Frankfurt a. Main", find="Frankfurt am Main")}
    extractor = make_extractor([One(Event)], world)
    record = only(
        await extractor.ainvoke(
            "Held in Frankfurt am Main.",
            existing=[("e", "Event", {"title": "Party", "place": "Rome"})],
        )
    )
    assert calls_of(world, "adjudicate") == []
    assert ops(record) == [("replace", "/place")]
    assert record.fields["place"].crosscheck_agree is True


async def test_a_hesitant_cross_check_that_agrees_still_supports_the_change() -> None:
    world = script()
    # the second pass found the same city but is unsure (probably null): it still
    # shows that the input gives the value
    world.cross["city"] = ca("text", "Berlin", find="Berlin", p_null=0.9)
    record = only(await update(world))
    assert ("replace", "/city") in ops(record)
    assert "/city" not in reasons(record)
    assert record.fields["city"].crosscheck_agree is True


async def test_a_hesitant_cross_check_is_not_taken_as_the_value() -> None:
    text = TEXT + " Hamburg is nice."
    world = script()
    world.clairic["city"] = ca("text", "Berlin", find="Berlin")
    world.cross["city"] = ca("text", "Hamburg", find="Hamburg", p_null=0.9)
    world.adjudicate["city"] = pick("b", ["a", "b", "neither"], p=0.9)
    record = only(await update(world, text))
    # the adjudication prefers the second reading, which is too unsure to take
    assert reasons(record)["/city"] == "crosscheck_disagree"
    assert record.after is not None and record.after["city"] == "Austin"


async def test_nothing_is_cross_checked_when_nothing_would_change() -> None:
    world = script()
    world.clairic = {
        "name": ca("text", "Dana Whitfield", find="Dana"),
        "email": ca("text", "dana@fabrikam.example", find="Dana"),
        "phone": ca("text", None, p_null=0.98),
        "city": ca("text", None, p_null=0.98),
        "age": ca("number", None, p_null=0.98),
        "notes": ca("list", None, p_null=0.98),
    }
    world.cross = {"name": ca("text", None, p_null=0.99)}  # would reject if checked
    record = only(await update(world))
    assert record.patches == () and not [
        r for r in record.rejected if r.reason == "echo_crosscheck"
    ]
    assert kind_counts(world.transport)["attribution"] == 0


# --------------------------------------------------------------------------- #
# The cross-check reading is taken when the primary reading gave nothing new
# --------------------------------------------------------------------------- #


class Plain(BaseModel):
    name: str
    city: Optional[str] = None
    age: Optional[int] = None


PLAIN_TEXT = "Dana moved to Berlin. She is 41 now."
PLAIN_RECORD = {"name": "Dana", "city": "Austin", "age": 40}


def plain_script() -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Dana", find="Dana"),
        "city": ca("text", "Austin", find='"Austin"'),  # span in the existing block
        "age": ca("number", 40),  # equals the record, no span
    }
    world.cross = {
        "city": ca("text", "Berlin", find="Berlin"),
        "age": ca("number", 41, find="41"),
    }
    return world


async def run_plain(world: Script, **kwargs: Any) -> ExtractionResult:
    extractor = make_extractor([One(Plain)], world, **kwargs)
    return await extractor.ainvoke(PLAIN_TEXT, existing={"Plain": PLAIN_RECORD})


async def test_an_echoing_primary_reading_gives_way_to_the_input_only_one() -> None:
    world = plain_script()
    record = only(await run_plain(world))
    # the primary pass echoed the record (a span in <existing>; a value equal to
    # the record): the input-only pass read the new values, and they are taken
    assert ops(record) == [("replace", "/city"), ("replace", "/age")]
    assert record.after == {"name": "Dana", "city": "Berlin", "age": 41}
    assert not [r for r in record.rejected if r.path in ("/city", "/age")]
    # taken values still go through attribution and the gate
    assert kind_counts(world.transport)["gate"] == 2
    assert kind_counts(world.transport)["attribution"] == 1
    gated = {c[1]["state"]["proposed_change"]["path"] for c in world.calls("gate")}
    assert gated == {"/city", "/age"}
    city = record.fields["city"]
    assert city.crosscheck_agree is False
    assert "adopted_from_crosscheck" in city.warnings
    assert "echo_span: replaced by the input-only reading" in city.warnings
    assert city.span is not None  # a span in the input, not in the record
    assert PLAIN_TEXT[city.span[0] : city.span[1]] == "Berlin"


async def test_a_null_primary_reading_gives_way_to_the_input_only_one() -> None:
    world = plain_script()
    world.clairic["city"] = ca("text", None, p_null=0.98)
    record = only(await run_plain(world))
    assert ("replace", "/city") in ops(record)
    assert record.after is not None and record.after["city"] == "Berlin"


async def test_a_list_the_primary_reading_missed_is_taken_from_the_input_only_one() -> (
    None
):
    world = script()
    world.cross["notes"] = world.clairic["notes"]
    world.clairic["notes"] = ca("list", None, p_null=0.98)
    record = only(await update(world))
    assert ("add", "/notes/-") in ops(record)


async def test_an_unusable_input_only_reading_is_not_taken() -> None:
    world = plain_script()
    world.cross["city"] = ca("text", "Berlin", find="Berlin", p_null=0.9)
    record = only(await run_plain(world))
    assert ops(record) == [("replace", "/age")]
    assert record.after is not None and record.after["city"] == "Austin"


async def test_a_taken_input_only_reading_must_still_be_attributed() -> None:
    world = plain_script()
    world.attribution = {"attr.city": 0.05}
    record = only(await run_plain(world))
    assert reasons(record)["/city"] == "attribution"
    assert ops(record) == [("replace", "/age")]


async def test_an_input_only_reading_equal_to_the_record_changes_nothing() -> None:
    world = plain_script()
    world.cross = {"city": ca("text", "Austin", find="Berlin"), "age": ca("number", 40)}
    record = only(await run_plain(world))
    assert record.patches == ()
    assert "adopted_from_crosscheck" not in record.fields["city"].warnings
    assert not world.calls("gate")


CONTESTED_TEXT = "Dana lived in Austin once, and moved to Berlin. She is 41 now."


def contested_script(verdict: str) -> Script:
    """The primary reading is the record's city, WITH a span in the input."""
    world = plain_script()
    world.clairic["city"] = ca("text", "Austin", find="Austin")
    world.adjudicate = {"city": pick(verdict, ["a", "b", "neither"], 0.9)}
    return world


async def run_contested(world: Script) -> ExtractionResult:
    extractor = make_extractor([One(Plain)], world)
    return await extractor.ainvoke(CONTESTED_TEXT, existing={"Plain": PLAIN_RECORD})


async def test_a_genuine_primary_reading_is_adjudicated_not_overruled() -> None:
    # "Austin" equals the record AND points at the input: a genuine reading. The
    # input-only pass says "Berlin": the two are adjudicated, and here Clef says b
    world = contested_script("b")
    record = only(await run_contested(world))
    (call,) = world.calls("adjudicate")
    assert call[2].field == "city"
    criteria = call[1]["questions"]["adjudicate"]["criteria"]
    assert (criteria["a"], criteria["b"]) == ("Austin", "Berlin")
    assert ("replace", "/city") in ops(record)
    assert record.after is not None and record.after["city"] == "Berlin"
    assert "adjudicated_to_crosscheck" in record.fields["city"].warnings
    # the age has no span at all (a number): the input-only reading is taken as is
    assert [c[2].field for c in world.calls("adjudicate")] == ["city"]
    assert ("replace", "/age") in ops(record)
    # the taken value still goes through attribution and the gate
    gated = {c[1]["state"]["proposed_change"]["path"] for c in world.calls("gate")}
    assert gated == {"/city", "/age"}


@pytest.mark.parametrize("verdict", ["a", "neither"])
async def test_a_genuine_primary_reading_stays_unless_the_other_wins(
    verdict: str,
) -> None:
    world = contested_script(verdict)
    record = only(await run_contested(world))
    assert ops(record) == [("replace", "/age")]
    assert record.after is not None and record.after["city"] == "Austin"
    assert any(
        w.startswith("crosscheck_overruled") for w in record.fields["city"].warnings
    )
    gated = {c[1]["state"]["proposed_change"]["path"] for c in world.calls("gate")}
    assert gated == {"/age"}


async def test_an_indecisive_adjudication_leaves_the_record_alone() -> None:
    world = contested_script("b")
    world.adjudicate = {"city": choice("b", {"a": 0.35, "b": 0.4, "neither": 0.25})}
    record = only(await run_contested(world))
    assert record.after is not None and record.after["city"] == "Austin"


# --------------------------------------------------------------------------- #
# An "__any__" record belongs to one schema
# --------------------------------------------------------------------------- #


class Owner2(BaseModel):
    name: str
    pet: str


class Drifter(BaseModel):
    name: str
    city: Optional[str] = None


class Sailor(BaseModel):
    name: str
    ship: Optional[str] = None


def any_script() -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Sam", find="Sam"),
        "city": ca("text", "Oslo", find="Oslo"),
        "pet": ca("text", None, p_null=0.98),
        "ship": ca("text", "Maud", find="Maud"),
    }
    return world


async def test_an_any_record_is_updated_by_the_schema_that_validates_it() -> None:
    world = any_script()
    extractor = make_extractor([One(Drifter), One(Owner2)], world)
    result = await extractor.ainvoke(
        "Sam moved to Oslo.", existing={"__any__": {"name": "Sam", "city": "Rome"}}
    )
    by_schema = {r.schema_name: r for r in result.records}
    assert by_schema["Drifter"].kind == "update"
    assert by_schema["Drifter"].json_doc_id == "__any__"
    assert ops(by_schema["Drifter"]) == [("replace", "/city")]
    assert (
        by_schema["Owner2"].kind == "insert" and by_schema["Owner2"].json_doc_id is None
    )
    assert [r.json_doc_id for r in result.records if r.kind == "update"] == ["__any__"]
    warning = next(w for w in result.warnings if "'__any__'" in w)
    assert "Drifter" in warning and "Owner2" in warning and "validates" in warning
    assert result.untouched == ()


async def test_an_any_record_goes_to_the_most_present_schema_when_several_fit() -> None:
    world = any_script()
    world.presence_by_schema = {"Drifter": 0.4, "Sailor": 0.9}
    extractor = make_extractor([One(Drifter), One(Sailor)], world)
    result = await extractor.ainvoke(
        "Sam is on the Maud.", existing={"__any__": {"name": "Sam"}}
    )
    by_schema = {r.schema_name: r for r in result.records}
    assert by_schema["Sailor"].kind == "update"
    assert by_schema["Drifter"].kind == "insert"
    presences = [c for c in result.calls if c.kind == "presence"]
    assert len(presences) == 2  # asked once for each, not again by the passes
    assert any("Sailor" in w and "most present" in w for w in result.warnings)


async def test_a_schema_with_its_own_record_does_not_claim_an_any_record() -> None:
    world = any_script()
    extractor = make_extractor([One(Drifter), One(Sailor)], world)
    result = await extractor.ainvoke(
        "Sam moved to Oslo, aboard the Maud.",
        existing=[
            ("d-1", "Drifter", {"name": "Sam", "city": "Rome"}),
            ("any-1", "__any__", {"name": "Sam"}),
        ],
    )
    by_schema = {r.schema_name: r for r in result.records}
    assert by_schema["Drifter"].json_doc_id == "d-1"
    assert by_schema["Sailor"].json_doc_id == "any-1"
    assert not [w for w in result.warnings if "'__any__'" in w]  # one candidate


async def test_untouched_is_keyed_by_schema_and_id() -> None:
    world = any_script()
    world.presence_by_schema = {"Drifter": 0.95, "Sailor": 0.03}
    extractor = make_extractor([One(Drifter), One(Sailor)], world)
    result = await extractor.ainvoke(
        "Sam moved to Oslo.",
        existing=[
            ("same", "Drifter", {"name": "Sam", "city": "Rome"}),
            ("same", "Sailor", {"name": "Sam"}),
        ],
    )
    assert [r.schema_name for r in result.records] == ["Drifter"]
    assert result.untouched == ("same",)  # the Sailor's record, same id


async def test_two_any_records_are_claimed_one_to_one() -> None:
    world = any_script()
    world.presence_by_schema = {"Drifter": 0.9, "Sailor": 0.6}
    extractor = make_extractor([One(Drifter), One(Sailor)], world)
    result = await extractor.ainvoke(
        "Sam moved to Oslo, aboard the Maud.",
        existing=[
            ("any-1", "__any__", {"name": "Sam", "city": "Rome"}),
            ("any-2", "__any__", {"name": "Sam"}),
        ],
    )  # both records fit both schemas: this used to raise mid-invoke
    by_schema = {r.schema_name: r for r in result.records}
    assert (by_schema["Drifter"].kind, by_schema["Drifter"].json_doc_id) == (
        "update",
        "any-1",
    )
    assert (by_schema["Sailor"].kind, by_schema["Sailor"].json_doc_id) == (
        "update",
        "any-2",
    )
    assert result.untouched == ()
    assert len([w for w in result.warnings if "'__any__'" in w]) == 2
    assert len([c for c in result.calls if c.kind == "presence"]) == 2  # asked once


async def test_an_any_record_nobody_is_left_to_claim_stays_untouched() -> None:
    world = any_script()
    extractor = make_extractor([One(Drifter), One(Sailor)], world)
    result = await extractor.ainvoke(
        "Sam moved to Oslo, aboard the Maud.",
        existing=[
            ("any-1", "__any__", {"name": "Sam"}),
            ("any-2", "__any__", {"name": "Sam"}),
            ("any-3", "__any__", {"name": "Sam"}),
        ],
    )
    assert sorted(
        r.json_doc_id or "" for r in result.records if r.kind == "update"
    ) == [
        "any-1",
        "any-2",
    ]
    assert result.untouched == ("any-3",)
    assert any("any-3" in w and "no schema" in w for w in result.warnings)


async def test_one_schema_takes_one_any_record_and_leaves_the_rest() -> None:
    world = any_script()
    extractor = make_extractor([One(Drifter)], world)
    result = await extractor.ainvoke(
        "Sam moved to Oslo.",
        existing=[
            ("any-1", "__any__", {"name": "Sam", "city": "Rome"}),
            ("any-2", "__any__", {"name": "Sam", "city": "Rome"}),
        ],
    )
    (record,) = result.records
    assert (record.kind, record.json_doc_id) == ("update", "any-1")
    assert result.untouched == ("any-2",)


async def test_validating_a_record_does_not_bypass_presence() -> None:
    # Drifter is all-optional, so it "validates" almost any record; Owner2 needs a
    # pet the record does not hold yet. The text is about Owner2's pet, not about
    # drifting: Drifter is absent, so it must not take the record.
    world = any_script()
    world.clairic["pet"] = ca("text", "Rex", find="Rex")
    world.presence_by_schema = {"Drifter": 0.02, "Owner2": 0.9}
    extractor = make_extractor([One(Drifter), One(Owner2)], world)
    result = await extractor.ainvoke(
        "Sam got a dog called Rex.", existing={"__any__": {"name": "Sam"}}
    )
    (record,) = result.records
    assert (record.schema_name, record.kind, record.json_doc_id) == (
        "Owner2",
        "update",
        "__any__",
    )
    assert ops(record) == [("add", "/pet")]
    assert any("Owner2" in w and "most present" in w for w in result.warnings)
    assert result.untouched == ()


async def test_an_any_record_no_present_schema_can_take_stays_untouched() -> None:
    world = any_script()
    world.presence_by_schema = {"Drifter": 0.02, "Sailor": 0.05}
    extractor = make_extractor([One(Drifter), One(Sailor)], world)
    result = await extractor.ainvoke(
        "Nothing about Sam here.", existing={"__any__": {"name": "Sam"}}
    )
    assert result.records == ()
    assert result.untouched == ("__any__",)
    assert any("none of the candidates is present" in w for w in result.warnings)
    assert not world.calls("extract")


async def test_a_schema_that_loses_an_any_claim_takes_the_insert_path() -> None:
    world = any_script()
    world.presence_by_schema = {"Drifter": 0.9, "Sailor": 0.8}
    extractor = make_extractor([One(Drifter), One(Sailor)], world)
    result = await extractor.ainvoke(
        "Sam moved to Oslo, aboard the Maud.",
        existing={"__any__": {"name": "Sam", "city": "Rome"}},
    )
    by_schema = {r.schema_name: r for r in result.records}
    assert by_schema["Drifter"].kind == "update"  # the same text updates one...
    assert by_schema["Sailor"].kind == "insert"  # ...and inserts the other's record


# --------------------------------------------------------------------------- #
# Validation errors below a union are mapped to the document
# --------------------------------------------------------------------------- #


class StrictCat(BaseModel):
    kind: Literal["cat"]
    lives: int = Field(ge=1)


class StrictDog(BaseModel):
    kind: Literal["dog"]
    tricks: int = Field(ge=0)


class StrictOwner(BaseModel):
    name: str
    pet: Union[StrictCat, StrictDog]


async def test_a_failing_operation_inside_a_union_member_is_dropped() -> None:
    world = Script()
    world.clairic = {
        "name": ca("text", "Sam", find="Sam"),
        "pet.StrictDog.tricks": ca("number", -2, find="-2"),
        "pet.StrictCat.lives": ca("number", None, p_null=0.99),
    }
    world.clef = {"pet": pick("dog", ["cat", "dog"])}
    extractor = make_extractor([One(StrictOwner)], world)
    record = only(
        await extractor.ainvoke(
            "Sam's dog now knows -2 tricks.",
            existing={
                "StrictOwner": {"name": "Sam", "pet": {"kind": "dog", "tricks": 3}}
            },
        )
    )
    assert record.valid, record.errors
    assert record.patches == ()
    assert record.after == {"name": "Sam", "pet": {"kind": "dog", "tricks": 3}}
    (bad,) = [r for r in record.rejected if r.reason == "validation"]
    assert (bad.op, bad.path) == ("replace", "/pet/tricks")
    assert "pet.tricks" in bad.detail  # the member is not part of the path


async def test_the_errors_of_a_record_that_stays_invalid_are_reported_per_path() -> (
    None
):
    world = Script()
    world.clairic = {
        "name": ca("text", None, p_null=0.99),
        "pet.StrictDog.tricks": ca("number", 5, find="5"),
        "pet.StrictCat.lives": ca("number", None, p_null=0.99),
    }
    world.clef = {"pet": pick("dog", ["cat", "dog"])}
    extractor = make_extractor([One(StrictOwner)], world)
    # the stored record lacks the required name
    result = await extractor.ainvoke(
        "the dog knows 5 tricks",
        existing={"StrictOwner": {"pet": {"kind": "dog", "tricks": 1}}},
    )
    record = only(result)
    assert not record.valid
    assert record.errors == ("name: Field required",)  # no noise of the cat member
    assert (None, "/name", "validation") in [
        (r.op, r.path, r.reason) for r in record.rejected
    ]
    assert ("replace", "/pet/tricks") in ops(record)


# --------------------------------------------------------------------------- #
# Clef answers are not operations unless the input states them
# --------------------------------------------------------------------------- #


class Vip(BaseModel):
    name: str
    vip: bool = False
    city: Optional[str] = None


def vip_script(vip: float = 0.05) -> Script:
    world = Script()
    world.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "city": ca("text", "Oslo", find="Oslo"),
    }
    world.clef = {"vip": noul_answer(vip)}
    return world


async def run_vip(
    world: Script, record: dict[str, Any], **kwargs: Any
) -> ExtractionResult:
    extractor = make_extractor([One(Vip)], world, **kwargs)
    return await extractor.ainvoke("Ann moved to Oslo.", existing={"Vip": record})


async def test_a_silent_bool_does_not_flip_the_record() -> None:
    world = vip_script()
    world.stated = {"stated.vip": 0.1}
    world.gate_default = 0.99  # the gate would pass it
    record = only(await run_vip(world, {"name": "Ann", "vip": True, "city": "Rome"}))
    assert ops(record) == [("replace", "/city")]
    assert record.after is not None and record.after["vip"] is True
    assert (None, "/vip", "not_mentioned") in [
        (r.op, r.path, r.reason) for r in record.rejected
    ]
    gated = [c[1]["state"]["proposed_change"]["path"] for c in world.calls("gate")]
    assert "/vip" not in gated
    assert "not_stated" in record.fields["vip"].warnings
    (call,) = world.calls("stated")
    assert call[2].kind == "stated" and list(call[1]["questions"]) == ["stated.vip"]
    assert "state" in call[1]["questions"]["stated.vip"]["instructions"]


async def test_a_stated_bool_becomes_an_operation_and_goes_to_the_gate() -> None:
    world = vip_script(vip=0.05)
    world.stated = {"stated.vip": 0.9}
    record = only(await run_vip(world, {"name": "Ann", "vip": True, "city": "Rome"}))
    assert ("replace", "/vip") in ops(record)
    assert record.after is not None and record.after["vip"] is False


async def test_the_stated_threshold_is_configurable() -> None:
    world = vip_script()
    world.stated = {"stated.vip": 0.75}
    record = only(
        await run_vip(
            world,
            {"name": "Ann", "vip": True, "city": "Rome"},
            verify=VerifyConfig(clef_stated_min=0.8),
        )
    )
    assert ("replace", "/vip") not in ops(record)


async def test_a_default_is_never_added_to_a_record_that_lacks_the_field() -> None:
    world = vip_script(vip=0.05)  # answers False: the default
    record = only(await run_vip(world, {"name": "Ann", "city": "Rome"}))
    assert ops(record) == [("replace", "/city")]
    assert not world.calls("stated")  # nothing would change, so nobody asks


async def test_a_clef_answer_equal_to_the_record_is_not_asked_about() -> None:
    world = vip_script(vip=0.95)  # True, and the record says True
    record = only(await run_vip(world, {"name": "Ann", "vip": True, "city": "Rome"}))
    assert ops(record) == [("replace", "/city")]
    assert not world.calls("stated")


class Tiered(BaseModel):
    customer: str
    tier: Literal["gold", "silver"]
    active: bool = True


async def test_a_silent_required_literal_is_not_changed_by_a_weak_choice() -> None:
    world = Script()
    world.clairic = {"customer": ca("text", "Acme", find="Acme")}
    world.clef = {
        "tier": pick("silver", ["gold", "silver"], 0.6),
        "active": noul_answer(0.05),
    }
    world.stated = {"stated.tier": 0.2, "stated.active": 0.2}
    extractor = make_extractor([One(Tiered)], world)
    record = only(
        await extractor.ainvoke(
            "Acme called.",
            existing={"Tiered": {"customer": "Acme", "tier": "gold", "active": True}},
        )
    )
    assert record.patches == ()
    # one batched request for the group, one question per Clef field
    (call,) = world.calls("stated")
    assert sorted(call[1]["questions"]) == ["stated.active", "stated.tier"]


class Addr(BaseModel):
    city: str
    verified: bool = False


class Resident(BaseModel):
    name: str
    addr: Optional[Addr] = None


async def run_resident(world: Script) -> ResultRecord:
    world.clairic = {
        "name": ca("text", "Ann", find="Ann"),
        "addr.city": ca("text", "Oslo", find="Oslo"),
    }
    world.group_presence = {"addr": 0.9}
    extractor = make_extractor([One(Resident)], world)
    return only(
        await extractor.ainvoke(
            "Ann lives in Oslo.", existing={"Resident": {"name": "Ann"}}
        )
    )


async def test_a_new_sub_document_leaves_out_unstated_and_default_values() -> None:
    world = Script()
    world.clef = {"addr.verified": noul_answer(0.1)}  # False: the default
    record = await run_resident(world)
    assert ops(record) == [("add", "/addr")]
    assert record.patches[0].value == {"city": "Oslo"}
    (change,) = [c[1]["state"]["proposed_change"] for c in world.calls("gate")]
    assert change["to"] == {"city": "Oslo"}


async def test_a_new_sub_document_carries_a_stated_non_default_clef_value() -> None:
    world = Script()
    world.clef = {"addr.verified": noul_answer(0.95)}
    record = await run_resident(world)
    assert record.patches[0].value == {"city": "Oslo", "verified": True}


async def test_a_new_sub_document_drops_a_clef_value_the_input_does_not_state() -> None:
    world = Script()
    world.clef = {"addr.verified": noul_answer(0.95)}  # True, but...
    world.stated = {"stated.addr.verified": 0.1}  # ...not stated
    record = await run_resident(world)
    assert record.patches[0].value == {"city": "Oslo"}


# --------------------------------------------------------------------------- #
# A record that is too big for the request is compacted, not the input chunked
# --------------------------------------------------------------------------- #

HISTORY = "history-" * 400  # not a field of the schema, 3200 characters


def blob_world() -> Script:
    world = script()
    world.clairic = {
        "name": ca("text", None, p_null=0.98),
        "email": ca("text", None, p_null=0.98),
        "phone": ca("text", None, p_null=0.98),
        "city": ca("text", "Berlin", find="Berlin", confidence=0.95),
        "age": ca("number", None, p_null=0.98),
        "notes": ca("list", None, p_null=0.98),
    }
    world.limit = lambda state, n: 2500 if len(state) > 2500 else None
    return world


async def test_a_cut_inside_the_record_compacts_it_and_does_not_chunk_the_input() -> (
    None
):
    world = blob_world()
    record_with_history = {**RECORD, "history": HISTORY}
    result = await update(world, TEXT, record_with_history)
    record = only(result)
    assert ops(record) == [("replace", "/city")]
    assert any("record_compacted" in w for w in result.warnings)
    assert not any("clairic_chunked" in w for w in result.warnings)
    extracts = world.calls("extract")
    assert len(extracts) == 2  # the cut one, and the same request again
    full, compact = (c[1]["state"] for c in extracts)
    assert "history-history" in full
    assert "history-" not in compact and compact.endswith("</existing>")
    assert '"name": "Dana Whitfield"' in compact  # the identity is kept
    assert '"city": "Austin"' in compact  # and the fields that are asked
    assert record.after is not None and record.after["history"] == HISTORY  # untouched


async def test_a_big_record_does_not_shrink_the_chunks_to_the_minimum() -> None:
    filler = "The weather was mild and nothing else happened. "
    text = filler * 45 + "Dana moved to Berlin last month. " + filler * 45
    world = blob_world()
    result = await update(
        world,
        text,
        {**RECORD, "history": HISTORY},
        verify=VerifyConfig(window_chars=100),
    )
    record = only(result)
    assert ops(record) == [("replace", "/city")]
    assert any("record_compacted" in w for w in result.warnings)
    assert any("clairic_chunked" in w for w in result.warnings)
    chunked = [
        c[1]["state"] for c in world.calls("extract") if len(c[1]["state"]) < 2500
    ]
    assert chunked and all(len(state) > 1000 for state in chunked)
    assert all("history-" not in state for state in chunked)
    assert all(state.endswith("</existing>") for state in chunked)


async def test_a_record_with_nothing_to_compact_is_still_sent_whole() -> None:
    world = blob_world()
    world.limit = lambda state, n: 150 if len(state) > 150 else None  # cuts anything
    result = await update(world, TEXT, RECORD)
    # nothing can be dropped (every field is asked), so the input is chunked as before
    assert not any("record_compacted" in w for w in result.warnings)
