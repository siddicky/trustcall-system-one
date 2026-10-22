"""A support email in, a nested ``Ticket`` out: sub-models, a union, an optional gate.

Runs today (M4). ``uv run python examples/02_support_ticket.py [--offline]``

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

Look at the output for: the resolved union member, the one ``shipping_address``
line that says it is absent, and "Every request that was made", which ends by listing
the requests that mention ``shipping_address``: only the presence question. No
Clairic request ever asks for a street or a postcode. Live mode runs on the real
models; offline mode scripts the answers.
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
    only_record,
    print_banner,
    resolve_offline,
    show_call_log,
    show_record,
    show_source,
    show_summary,
    show_verification,
)
from _offline import OfflineScript, pick, says, scale
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult, ResultRecord, s1_field

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
        source=EMAIL,
        presence=0.99,
        # The optional sub-model's gate is a Clef noul with id "<path>.__presence".
        absent=("shipping_address.__presence",),
        clairic={
            "customer.name": says(
                "text", "Tomás Herrera", find="Tomás Herrera", p_null=0.01
            ),
            "customer.email": says(
                "text",
                "tomas.herrera@lumenlabs.example",
                find="tomas.herrera@lumenlabs.example",
                p_null=0.01,
            ),
            # Only the chosen union member is asked (its wire names carry the
            # member label); keys match by wire-name suffix, not by wording.
            "invoice_number": says("text", "LL-88213", find="LL-88213", p_null=0.02),
            "amount_disputed": says("number", 249, p_null=0.03, confidence=0.93),
        },
        clef={
            "customer.plan": pick("pro", ["free", "pro", "enterprise"], 0.97),
            # The union: the options are the discriminator's values.
            "issue": pick("billing", ["billing", "outage"], 0.96),
            "urgency": scale(2),  # "pressing" = HIGH
            "refund_requested": 0.98,
        },
    )


def show_shape(record: ResultRecord, ticket: Ticket) -> None:
    """Print what the schema's shape did, from the record's own field log."""
    heading("What the schema's shape did")
    union = record.fields.get("issue")
    chosen = ""
    if union is not None and isinstance(union.raw, dict):
        pick = union.raw.get("choice")
        chosen = f"  (Clef choice {pick!r}, confidence {union.confidence})"
    print(f"  issue             {type(ticket.issue).__name__}{chosen}")
    print(
        f"                    kind={ticket.issue.kind!r} was filled in code, not asked"
    )
    gate = record.fields.get("shipping_address")
    leaves = [p for p in record.fields if p.startswith("shipping_address.")]
    if ticket.shipping_address is None and gate is not None:
        yes = gate.raw.get("noul") if isinstance(gate.raw, dict) else None
        print(f"  shipping_address  None  (presence gate: P(yes) {yes})")
        print(f"                    fields asked about it: {len(leaves)}")
    else:
        print(f"  shipping_address  {ticket.shipping_address!r}")
    print(f"  urgency           {ticket.urgency} (a Clef score on the described scale)")
    print()


def main(offline: bool = False) -> ExtractionResult:
    """Extract the ticket and print what happened; return the result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    with build_extractor([Ticket], offline=offline, script=script) as extractor:
        result = extractor.invoke(EMAIL)

    record = only_record(result, "ticket")
    if record is None:
        show_call_log(result, script)
        return result
    show_source("The email, with the spans the model pointed at", EMAIL, record)
    show_record(record)
    ticket = record.value
    if isinstance(ticket, Ticket):
        show_shape(record, ticket)
    show_call_log(result, script, mentioning="shipping_address")
    show_verification(result)
    show_summary(result, offline=offline)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
