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
    assert seen[4:] == ["gate"] * 4


# --------------------------------------------------------------------------- #
# Echoes: the answer was read from the record
# --------------------------------------------------------------------------- #


async def test_an_answer_whose_span_lies_in_the_record_is_rejected() -> None:
    world = script()
    # "Austin" is only in the <existing> block: the span points there
    world.clairic["city"] = ca("text", "Austin", find="Austin", p_null=0.02)
    world.cross["city"] = ca("text", "Berlin", find="Berlin")
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
    assert state["entity"] == {"label": "the Contact", "mentions": []}


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


async def test_list_form_records_without_a_wrapper_imply_many() -> None:
    world = script()
    extractor = make_extractor([Contact], world)
    with pytest.raises(NotImplementedError, match="M7-M8"):
        await extractor.ainvoke(TEXT, existing=[("rec-1", "Contact", RECORD)])
    assert world.transport.calls == []


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


async def test_a_cross_check_answer_that_is_too_null_counts_as_nothing() -> None:
    world = script()
    # the second pass found a city but says it is probably null: not usable
    world.cross["city"] = ca("text", "Berlin", find="Berlin", p_null=0.9)
    record = only(await update(world))
    assert reasons(record)["/city"] == "echo_crosscheck"


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
