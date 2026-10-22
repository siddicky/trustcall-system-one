"""A support email in, a nested ``Ticket`` out: sub-models, a union, an optional gate.

Needs milestone M4 (nested sub-models and unions); until it lands this example
raises or returns an incomplete ticket (``tests/unit/test_examples.py`` expects the
failure and flips when M4 does). ``uv run python examples/02_support_ticket.py``.

What it demonstrates:

* Nested models. ``Ticket.customer`` is a ``Customer`` sub-model; its fields show up
  below as ``customer.name``, ``customer.email``, ``customer.plan``. Free text goes
  to Clairic, the ``Literal`` plan to Clef.
* A union of models. ``issue`` is ``BillingIssue | OutageIssue``, discriminated by a
  ``kind`` Literal. Clef first decides which member the email describes; only that
  member's fields are then asked, and ``kind`` itself is filled in code (a
  single-option Literal needs no request).
* An optional sub-model that is never mentioned. The email has no postal address, so
  the presence gate (Clef) answers "no" and none of ``shipping_address``'s four
  fields is sent to Clairic. ``shipping_address`` is simply ``None``.
* An ordered enum. ``urgency`` is an ``Enum`` marked ``s1_field(clef="score")``:
  Clef rates the email on the four described levels and the rating maps back to a
  member, instead of Clairic extracting a word that is not in the text.
* ``refund_requested: bool`` is a Clef yes/no.

Look at the output for: the resolved union member, the absence of any
``shipping_address.*`` line, and the call counts (no Clairic request mentions the
address). Live mode runs on the real models; offline mode scripts the answers.
"""

import enum
import sys
from decimal import Decimal
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    build_extractor,
    cli,
    heading,
    print_banner,
    resolve_offline,
    show_record,
    show_source,
    show_summary,
    show_verification,
)
from _offline import OfflineScript, choose, pick, says, scale
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult, s1_field

EMAIL = """\
Subject: Charged twice for March, please refund

Hi,

I'm Tomás Herrera (tomas.herrera@lumenlabs.example) and we are on the Pro plan. Invoice LL-88213 for March was charged to our card twice, $249.00 each time, and I would like the duplicate payment refunded.

Our finance team closes the books on Friday, so this is fairly pressing, but nothing is broken on our side.

Thanks,
Tomás"""  # noqa: E501


class Urgency(enum.Enum):
    """How soon the customer needs an answer, lowest first."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Customer(BaseModel):
    """The person writing in."""

    name: str
    email: str
    plan: Literal["free", "pro", "enterprise"]


class BillingIssue(BaseModel):
    """A problem with a charge or an invoice."""

    kind: Literal["billing"]
    invoice_number: str | None = None
    amount_disputed: Decimal | None = None


class OutageIssue(BaseModel):
    """A service that is down or degraded."""

    kind: Literal["outage"]
    service: str
    region: str | None = None


class Address(BaseModel):
    """A postal address."""

    street: str
    city: str
    postcode: str
    country: str


class Ticket(BaseModel):
    """A customer support ticket."""

    customer: Customer
    issue: BillingIssue | OutageIssue = Field(discriminator="kind")
    shipping_address: Address | None = Field(
        None, description="postal address the customer wants goods sent to"
    )
    urgency: Urgency = Field(
        Urgency.NORMAL,
        json_schema_extra=s1_field(
            clef="score",
            criteria=[
                "No time pressure; a reply within a week is fine.",
                "An ordinary request; a reply within a couple of days is fine.",
                "Pressing: the customer needs a reply today or tomorrow.",
                "Critical: the customer's business is stopped right now.",
            ],
        ),
    )
    refund_requested: bool = Field(description="the customer asks for money back")


def offline_script() -> OfflineScript:
    """What Clairic and Clef would plausibly answer for ``EMAIL``."""
    return OfflineScript(
        presence=0.99,
        absent=("address",),  # the optional sub-model's presence gate says no
        clairic={
            "name": says("text", "Tomás Herrera", find="Tomás Herrera", p_null=0.01),
            "email": says(
                "text",
                "tomas.herrera@lumenlabs.example",
                find="tomas.herrera@lumenlabs.example",
                p_null=0.01,
            ),
            "invoice_number": says("text", "LL-88213", find="LL-88213", p_null=0.02),
            "amount_disputed": says("number", 249.0, p_null=0.03, confidence=0.93),
        },
        clef={
            "plan": pick("pro", ["free", "pro", "enterprise"], 0.97),
            "issue": choose("billing"),  # which member of the union
            "urgency": scale(2),  # "pressing" = HIGH
            "refund_requested": 0.98,
        },
    )


def main(offline: bool = False) -> ExtractionResult:
    """Extract the ticket and print what happened; return the result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    with build_extractor([Ticket], offline=offline, script=script) as extractor:
        result = extractor.invoke(EMAIL)

    (record,) = result.records
    show_source("The email, with the spans the model pointed at", EMAIL, record)
    show_record(record)
    ticket = record.value
    if isinstance(ticket, Ticket):
        heading("What the schema's shape did")
        issue = type(ticket.issue).__name__
        print(
            f"  issue             {issue} (union member chosen by Clef, "
            f"kind={ticket.issue.kind!r})"
        )
        asked = [path for path in record.fields if path.startswith("shipping_address")]
        print(
            "  shipping_address  "
            + (
                "None; the presence gate found no address, no field of it was asked"
                if ticket.shipping_address is None and not asked
                else repr(ticket.shipping_address)
            )
        )
        print(
            f"  urgency           {ticket.urgency} (a Clef score on a described scale)"
        )
        print()
    show_verification(result)
    show_summary(result)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
