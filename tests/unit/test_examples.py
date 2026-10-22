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


def run(name: str, capsys: pytest.CaptureFixture[str]) -> Any:
    """Run an example offline; check the banner and that it printed something."""
    result = load(name).main(offline=True)
    out = capsys.readouterr().out
    assert BANNER in out
    assert len(out.splitlines()) > 10
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
    # a verification event: the low-confidence vendor was re-asked
    assert record.fields["vendor"].reask is True
    assert [c.field for c in result.calls if c.kind == "reask"] == ["vendor"]
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
    assert result.usage.requests == len(result.calls) == 5


# --------------------------------------------------------------------------- #
# 02-07: written against the planned API; each flips with its milestone
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="needs M4")
def test_02_support_ticket(capsys: pytest.CaptureFixture[str]) -> None:
    module = load("02_support_ticket")
    result = run("02_support_ticket", capsys)
    (record,) = result.records
    assert record.valid, record.errors
    ticket = record.value
    assert ticket.customer.name == "Tomás Herrera"
    assert ticket.customer.email == "tomas.herrera@lumenlabs.example"
    assert ticket.customer.plan == "pro"
    assert isinstance(ticket.issue, module.BillingIssue)
    assert ticket.issue.kind == "billing"
    assert ticket.issue.invoice_number == "LL-88213"
    assert ticket.issue.amount_disputed == Decimal("249.00")
    assert ticket.urgency is module.Urgency.HIGH
    assert ticket.refund_requested is True
    # never mentioned: the presence gate skips the sub-model entirely
    assert ticket.shipping_address is None
    assert not [path for path in record.fields if path.startswith("shipping_address")]


@pytest.mark.xfail(strict=True, reason="needs M5")
def test_03_update_contact(capsys: pytest.CaptureFixture[str]) -> None:
    module = load("03_update_contact")
    result = run("03_update_contact", capsys)
    (record,) = result.records
    assert (record.kind, record.json_doc_id) == ("update", "crm-4711")
    assert {(p.op, p.path) for p in record.patches} == {
        ("replace", "/city"),
        ("replace", "/company"),
        ("replace", "/title"),
        ("replace", "/phone"),
        ("add", "/notes/-"),
    }
    before = module.EXISTING.model_dump()
    assert record.before == before
    after = record.after
    assert after["city"] == "Berlin"
    assert after["company"] == "Northwind"
    assert after["title"] == "Head of Ops"
    assert after["phone"] == "+49 30 5555 0198"
    assert after["notes"][:2] == before["notes"]  # existing notes kept
    assert after["notes"][2:] == ["asked to be introduced to our logistics team"]
    # untouched fields are untouched, and each says why no operation was made
    assert (after["name"], after["email"]) == (before["name"], before["email"])
    assert {r.path: r.reason for r in record.rejected} == {
        "/name": "gate_noul",
        "/email": "p_null",
    }


@pytest.mark.xfail(strict=True, reason="needs M6")
def test_04_repair(capsys: pytest.CaptureFixture[str]) -> None:
    result = run("04_repair", capsys)
    (record,) = result.records
    assert record.valid, record.errors
    assert (record.value.name, record.value.country, record.value.age) == (
        "Katarina Vogel",
        "DE",
        34,
    )
    assert result.attempts == 2  # the extraction, then one repair round
    repairs = [call for call in result.calls if call.kind == "repair"]
    assert sorted(call.field or "" for call in repairs) == ["age", "country"]


@pytest.mark.xfail(strict=True, reason="needs M8")
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
    assert result.untouched == ()
    # every mention carries its span in the conversation
    text = module.CONVERSATION
    for record in result.records:
        assert record.entity is not None and record.entity.mentions
        for mention in record.entity.mentions:
            assert mention.span is not None
            assert text[mention.span[0] : mention.span[1]] == mention.text


@pytest.mark.xfail(strict=True, reason="needs M7")
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


# --------------------------------------------------------------------------- #
# 08: runs today on FakeTransport + route (FakeWorld arrives with M7)
# --------------------------------------------------------------------------- #


def test_08_testing_your_extractor(capsys: pytest.CaptureFixture[str]) -> None:
    module = load("08_testing_your_extractor")
    outcome = module.main(offline=True)
    out = capsys.readouterr().out
    assert BANNER in out
    assert outcome == {test.__name__: True for test in module.TESTS}
    assert len(outcome) == 5
