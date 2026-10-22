"""The curated examples run offline and show what they claim to show.

Every example has ``main(offline: bool)``; here it is always called with
``offline=True`` and without Cloudflare credentials, so nothing leaves the process
(the unit suite also blocks sockets). Examples that need a milestone that is not
built yet are marked ``xfail(strict=True)``: the milestone that builds the feature
makes the test pass, strict mode then reports that as a failure, and whoever built
it removes the marker. That is the point: the example is the milestone's demo.
"""

import importlib.util
import sys
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

NAMES = [
    "01_invoice",
    "02_support_ticket",
    "03_update_contact",
    "04_repair",
    "05_people_memory",
    "06_attribution_guard",
    "07_order_line_items",
    "08_testing_your_extractor",
]

BANNER = "OFFLINE DEMO — answers come from a scripted transport, not the models"


@pytest.fixture(autouse=True)
def _offline_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """No credentials (so live mode cannot switch on) and the examples' helpers."""
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", [str(EXAMPLES), *sys.path])


def load(name: str) -> ModuleType:
    """Import ``examples/<name>.py`` (the names start with digits)."""
    spec = importlib.util.spec_from_file_location(
        f"examples_{name}", EXAMPLES / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(
    name: str, capsys: pytest.CaptureFixture[str], *, keep_output: bool = False
) -> Any:
    """Run an example offline; check the banner and that it printed something.

    With ``keep_output`` the output is printed again so the caller can read it
    from ``capsys`` too.
    """
    result = load(name).main(offline=True)
    out = capsys.readouterr().out
    assert BANNER in out
    assert len(out.splitlines()) > 10
    if keep_output:
        print(out, end="")
    return result


# --------------------------------------------------------------------------- #
# All examples
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
def test_every_example_imports_and_has_a_documented_main(name: str) -> None:
    module = load(name)
    assert callable(module.main)
    assert module.main.__code__.co_varnames[0] == "offline"
    doc = module.__doc__ or ""
    assert len(doc.splitlines()) > 5  # what it demonstrates and what to look at
    assert "uv run python examples/" in doc


def test_the_readme_lists_every_example() -> None:
    readme = (EXAMPLES / "README.md").read_text()
    for name in NAMES:
        assert f"{name}.py" in readme


def test_mode_selection() -> None:
    common = importlib.import_module("_common")
    live_available, resolve_offline = common.live_available, common.resolve_offline

    both = {"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_API_TOKEN": "token"}
    assert live_available(both) is True
    assert live_available({"CLOUDFLARE_ACCOUNT_ID": "acct"}) is False
    assert live_available({}) is False
    assert resolve_offline(False, both) is False  # live: credentials, no flag
    assert resolve_offline(True, both) is True  # --offline wins
    assert resolve_offline(False, {}) is True  # no credentials: offline


def test_offline_locate_never_guesses_between_occurrences() -> None:
    offline = importlib.import_module("_offline")
    source = "Berlin is big. Then we flew to Berlin again. " + "x" * 200
    wanted = source.rfind("Berlin")
    window = source[wanted - 30 : wanted + 30]
    state = f"<input>\n{window}\n</input>"
    # found by its surrounding text, so it is the second occurrence that is meant
    span = offline.locate(state, "Berlin", 1, source)
    assert span is not None and state[span[0] : span[1]] == "Berlin"
    assert span[0] == state.index("flew to Berlin") + len("flew to ")
    # a state that holds only the OTHER occurrence must not be answered with it
    other = "<input>\nBerlin is big.\n</input>"
    with pytest.raises(offline.ScriptError, match="another occurrence"):
        offline.locate(other, "Berlin", 1, source)
    # a state that holds neither is simply a miss
    assert (
        offline.locate("<input>\nsomething else\n</input>", "Berlin", 1, source) is None
    )
    # a needle that occurs once in the text needs no context
    once = "We met in Paris. " + "y" * 100
    assert offline.locate("<input>\nin Paris.\n</input>", "Paris", 0, once) is not None


# --------------------------------------------------------------------------- #
# 01: works today
# --------------------------------------------------------------------------- #


def test_01_invoice(capsys: pytest.CaptureFixture[str]) -> None:
    result = run("01_invoice", capsys)
    (record,) = result.records
    assert record.valid and record.kind == "insert"
    invoice = record.value
    assert invoice.invoice_number == "INV-2024-0457"
    assert invoice.vendor == "Brightwave Supplies Ltd"
    assert invoice.issue_date == date(2024, 3, 14)
    assert invoice.terms_days == 30
    assert invoice.total == Decimal("1080.00")
    assert invoice.currency == "USD"
    assert invoice.paid is False
    # "do the maths in code"
    assert invoice.due_date == date(2024, 4, 13)
    # absent from the email: a confident null, the guess is metadata only
    assert invoice.po_number is None
    po = record.fields["po_number"]
    assert po.value is None and po.p_null is not None and po.p_null > 0.9
    assert po.guess == "INV-2024-0412"
    # verification events: the low-confidence vendor was re-asked and agreed; the
    # terms were re-asked, the second answer disagreed, and Clef adjudicated
    assert record.fields["vendor"].reask is True
    assert record.fields["terms_days"].reask is True
    assert sorted(c.field or "" for c in result.calls if c.kind == "reask") == [
        "terms_days",
        "vendor",
    ]
    assert [c.field for c in result.calls if c.kind == "adjudicate"] == ["terms_days"]
    # a value the checks reject: a name that is not the approver
    assert invoice.approved_by is None
    assert {r.path: r.reason for r in record.rejected} == {
        "/approved_by": "attribution"
    }
    # text values are verbatim spans of the email
    text = load("01_invoice").EMAIL
    for field in ("invoice_number", "vendor"):
        answer = record.fields[field]
        assert answer.span is not None
        assert text[answer.span[0] : answer.span[1]] == answer.value
    assert result.usage.requests == len(result.calls) == 7


# --------------------------------------------------------------------------- #
# 02-07: written against the planned API; each flips with its milestone
# --------------------------------------------------------------------------- #


def test_02_support_ticket(capsys: pytest.CaptureFixture[str]) -> None:
    result = run("02_support_ticket", capsys, keep_output=True)
    (record,) = result.records
    assert record.valid, record.errors
    ticket = record.value
    assert ticket.customer.name == "Tomás Herrera"
    assert ticket.customer.email == "tomas.herrera@lumenlabs.example"
    assert ticket.customer.plan == "pro"
    assert type(ticket.issue).__name__ == "BillingIssue"
    assert ticket.issue.kind == "billing"
    assert ticket.issue.invoice_number == "LL-88213"
    assert ticket.issue.amount_disputed == Decimal("249.00")
    assert ticket.urgency.name == "HIGH"
    assert ticket.refund_requested is True
    # never mentioned: the presence gate skips the sub-model entirely
    assert ticket.shipping_address is None
    gate = record.fields["shipping_address"]
    assert (gate.engine, gate.rejected_reason) == ("submodel", "absent")
    assert not [path for path in record.fields if path.startswith("shipping_address.")]
    # one presence request for it, and no extraction request about it at all
    assert [c.field for c in result.calls if c.kind == "presence"] == [
        None,
        "shipping_address",
    ]
    assert not [
        c for c in result.calls if c.kind == "extract" and c.field == "shipping_address"
    ]
    out = capsys.readouterr().out
    assert "requests that mention 'shipping_address': presence" in out


def test_03_update_contact(capsys: pytest.CaptureFixture[str]) -> None:
    module = load("03_update_contact")
    result = run("03_update_contact", capsys)
    (record,) = result.records
    assert (record.kind, record.json_doc_id) == ("update", "crm-4711")
    assert record.valid and result.untouched == ()
    assert {(p.op, p.path) for p in record.patches} == {
        ("replace", "/city"),
        ("replace", "/company"),
        ("replace", "/title"),
        ("replace", "/phone"),
        ("add", "/notes/-"),
    }
    # replaces first (in the order of the fields), then the append
    assert [p.op for p in record.patches] == ["replace"] * 4 + ["add"]
    before = module.EXISTING.model_dump()
    assert record.before == before
    after = record.after
    assert after["city"] == "Berlin"
    assert after["company"] == "Northwind"
    assert after["title"] == "Head of Ops"
    assert after["phone"] == "+49 30 5555 0198"
    assert after["notes"][:2] == before["notes"]  # existing notes kept
    assert after["notes"][2:] == ["asked to be introduced to our logistics team"]
    # untouched fields are untouched, and each says why no operation was made:
    # the email is never mentioned; "Dana" is a token subset of the identity
    assert (after["name"], after["email"]) == (before["name"], before["email"])
    assert {r.path: r.reason for r in record.rejected} == {
        "/name": "identity_subset",
        "/email": "not_mentioned",
    }
    assert {r.path: r.op for r in record.rejected} == {
        "/name": "replace",
        "/email": None,
    }


def test_03_update_contact_calls_follow_the_update_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run("03_update_contact", capsys)
    kinds = [call.kind for call in result.calls]
    assert kinds[0] == "presence"
    # the record is read next to the note, and again without it
    assert sorted(kinds[1:3]) == ["crosscheck", "extract"]
    # one change-gate request per applied operation, none for the refused name
    assert kinds.count("gate") == 5 and kinds.count("attribution") == 1
    assert kinds[-5:] == ["gate"] * 5
    assert len(kinds) == 9


def test_04_repair(capsys: pytest.CaptureFixture[str]) -> None:
    """Two fields fail validation; one repair round (two re-asks) fixes both."""
    result = run("04_repair", capsys, keep_output=True)
    (record,) = result.records
    assert record.valid, record.errors
    applicant = record.value
    assert (applicant.name, applicant.postcode, applicant.age) == (
        "Katarina Vogel",
        "10115",
        34,
    )
    # one extraction round and ONE repair round, however many fields it re-asked
    assert result.attempts == 2
    repairs = [call for call in result.calls if call.kind == "repair"]
    assert sorted(call.field or "" for call in repairs) == ["age", "postcode"]
    assert [call.kind for call in result.calls] == [
        "presence",
        "extract",
        "attribution",  # the age 1990 is refused here, so the age is missing
        "repair",
        "repair",
        "attribution",  # the repaired values are attributed like any answer
    ]
    # the repaired fields say so, with their history; the others do not
    assert record.fields["postcode"].repaired and record.fields["age"].repaired
    assert not record.fields["name"].repaired
    assert record.fields["postcode"].warnings == (
        "repair[1] (format): 'Berlin 10115' -> '10115'",
    )
    assert record.fields["age"].warnings == (
        "repair[1] (missing): None -> 34",
        # the wrong age (1990) never became a value: attribution refused it first
        "the first answer had been rejected (attribution) before the repair",
    )
    assert result.skipped == () and record.rejected == ()
    # a repair request as the library sent it
    out = capsys.readouterr().out
    for block in ("<previous_attempt>", "<validation_errors>", "<expected_schema>"):
        assert block in out


def test_05_people_memory(capsys: pytest.CaptureFixture[str]) -> None:
    module = load("05_people_memory")
    result = run("05_people_memory", capsys)
    assert [(r.json_doc_id, r.kind) for r in result.records] == [
        ("0", "update"),
        ("1", "update"),
        ("2", "update"),
        (None, "insert"),
    ]
    emma, michael, sarah, olivia = result.responses
    assert [p.name for p in (emma, michael, olivia)] == [
        "Emma Thompson",
        "Michael Chen",
        "Olivia Davis",
    ]
    assert sarah.name == "Sarah Johnson"  # "Sarah" is not a rename
    existing = module.EXISTING
    for old, new in zip(existing, (emma, michael, sarah), strict=True):
        assert new.notes[:3] == old.notes  # existing notes untouched, in place
        assert new.relationship == old.relationship
    assert emma.notes[3:] == [
        "walking her new puppy, a golden retriever named Sunny",
        "got promoted to Senior Marketing Manager last month",
        "taken up rock climbing",
    ]
    # Michael: Emma's rock climbing was refused by the attribution check
    assert "rock climbing" not in " ".join(michael.notes)
    michael_record = result.records[1]
    assert [
        r.reason for r in michael_record.rejected if r.path.startswith("/notes")
    ] == ["attribution"]
    # Olivia inserted, spans are verbatim text: "my cousin", not a paraphrase
    assert olivia.relationship == "my cousin"
    assert len(olivia.notes) == 4
    # Sunny is a dog
    assert [(s.reason, s.entity.first_mention.text) for s in result.skipped] == [
        ("not_qualified", "Sunny")
    ]
    assert result.untouched == () and result.deleted == ()
    # the patches: notes are appended, nothing else is touched
    assert [[(p.op, p.path) for p in r.patches] for r in result.records] == [
        [("add", "/notes/-")] * 3,
        [("add", "/notes/-")] * 2,
        [("add", "/notes/-")] * 3,
        [],
    ]
    assert [m["json_doc_id"] for m in result.response_metadata] == [
        "0",
        "1",
        "2",
        None,
    ]
    # Sarah: the short form never overwrites the full name, and the case-only
    # relationship change is refused by the gate
    sarah_record = result.records[2]
    assert {(r.path, r.reason) for r in sarah_record.rejected} >= {
        ("/name", "identity_subset"),
        ("/relationship", "gate_noul"),
    }
    # the match probabilities are reported per entity: the record, or __new__
    chosen = []
    for record in result.records:
        assert record.entity is not None
        probs = record.entity.match_probs
        chosen.append(max(probs, key=lambda k: probs[k]))
    assert chosen == ["0", "1", "2", "__new__"]
    # every mention carries its span in the conversation
    text = module.CONVERSATION
    for record in result.records:
        assert record.entity is not None and record.entity.mentions
        for mention in record.entity.mentions:
            assert mention.span is not None
            assert text[mention.span[0] : mention.span[1]] == mention.text


def test_05_calls_follow_the_matching_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run("05_people_memory", capsys, keep_output=True)
    counts = Counter(call.kind for call in result.calls)
    assert counts["match"] == 4  # one choice per person that qualifies; not Sunny
    assert counts["qualify"] == 1 and counts["gate"] == 9
    # a request as the library sent it: the options are the records as JSON
    out = capsys.readouterr().out
    assert "match 4" in out and "29 requests" in out


def test_06_attribution_guard(capsys: pytest.CaptureFixture[str]) -> None:
    result = run("06_attribution_guard", capsys)
    priya, tom = result.responses
    assert (priya.name, priya.city, priya.role) == ("Priya Raman", "Lisbon", None)
    assert (tom.name, tom.city, tom.role) == ("Tom Becker", None, "finance lead")
    rejected = {
        r.value.name: {(x.path, x.reason) for x in r.rejected} for r in result.records
    }
    assert rejected == {
        "Priya Raman": {("/role", "attribution")},
        "Tom Becker": {("/city", "attribution")},
    }


def test_06_calls_follow_the_collection_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run("06_attribution_guard", capsys, keep_output=True)
    kinds = [call.kind for call in result.calls]
    assert sorted(kinds[:3]) == ["count", "discover", "presence"]  # in parallel
    # the one pair of mentions is asked about (every pair is, up to 24 mentions);
    # both entities are qualified in one request
    assert kinds[3:] == ["coref", "qualify", "extract", "extract"] + ["attribution"] * 2
    # the call log the example prints
    out = capsys.readouterr().out
    assert "9 requests" in out
    assert "attribution 2" in out and "qualify 1" in out
    colleague = result.records[0].entity
    assert colleague is not None and colleague.label == "Priya Raman"
    assert [m.text for m in colleague.mentions] == ["Priya Raman"]


async def test_05_offline_script_answers_the_real_discovery_shapes() -> None:
    """The discovery half of 05, on its own: the shapes M7 sends."""
    from trustcall_system_one import Many, compile_schema
    from trustcall_system_one._collection import Scope, collect
    from trustcall_system_one._run import Run

    module = load("05_people_memory")
    script = module.offline_script()
    scope = Scope(
        compile_schema(module.Person),
        module.CONVERSATION,
        Many(module.Person, identity="name"),
    )
    outcome = await collect(scope, Run(transport=script.transport))
    assert [e.label for e in outcome.entities] == [
        "Emma Thompson",
        "Michael Chen",
        "Sarah",
        "Olivia Davis",
    ]
    assert [(s.reason, s.entity.label) for s in outcome.skipped] == [
        ("not_qualified", "Sunny")
    ]
    assert outcome.count == 4 and outcome.warnings == ()
    for entity in outcome.entities:
        for mention in entity.mentions:
            assert mention.span is not None
            assert module.CONVERSATION[slice(*mention.span)] == mention.text


@pytest.mark.xfail(strict=True, reason="needs M9")
def test_07_order_line_items(capsys: pytest.CaptureFixture[str]) -> None:
    module = load("07_order_line_items")
    result = run("07_order_line_items", capsys)
    (record,) = result.records
    assert (record.kind, record.json_doc_id) == ("update", "SO-10442")
    assert {(p.op, p.path) for p in record.patches} == {
        ("replace", "/items/1/quantity"),
        ("add", "/items/-"),
    }
    before = module.EXISTING.model_dump()
    items = record.after["items"]
    assert len(items) == 4
    assert items[0] == before["items"][0]  # the kettle: untouched
    assert items[2] == before["items"][2]  # the spoons: untouched
    assert items[1] == {**before["items"][1], "quantity": 1}
    assert items[3]["name"] == "blue mugs" and items[3]["quantity"] == 2
    assert (record.after["order_id"], record.after["customer"]) == (
        before["order_id"],
        before["customer"],
    )


def test_07_update_path_reports_the_collection_it_cannot_extract_yet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Until M9 the line items are left as the record has them, and it says so."""
    result = run("07_order_line_items", capsys)
    (record,) = result.records
    assert record.kind == "update" and record.patches == ()
    assert any(
        "collection fields are not extracted until M9" in w for w in result.warnings
    )
    assert any("left as the record has it" in w for w in result.warnings)
    unsupported = [r for r in record.rejected if r.reason == "unsupported"]
    assert [(r.op, r.path) for r in unsupported] == [(None, "/items")]
    assert record.after == record.before


# --------------------------------------------------------------------------- #
# 08: FakeTransport + route, and FakeWorld for a multi-entity run
# --------------------------------------------------------------------------- #


def test_08_testing_your_extractor(capsys: pytest.CaptureFixture[str]) -> None:
    module = load("08_testing_your_extractor")
    outcome = module.main(offline=True)
    out = capsys.readouterr().out
    assert BANNER in out
    assert outcome == {test.__name__: True for test in module.TESTS}
    assert len(outcome) == 8
