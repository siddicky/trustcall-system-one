"""An invoice email in, a validated ``Invoice`` out.

Runs today (M3). ``uv run python examples/01_invoice.py [--offline]``

What it demonstrates, in the order you will see it:

* Typed extraction from messy prose. ``str`` and ``date``/``int``/``Decimal`` fields
  go to Clairic; the ``Literal`` currency (a choice) and ``paid: bool`` (a yes/no)
  go to Clef. The email deliberately holds decoys: a subtotal next to the total, an
  earlier invoice that *was* paid, a person who is not the approver.
* Spans: Clairic's text values are verbatim pieces of the email. Each is bracketed
  in the printed email with the field it supports. Dates and numbers carry no span.
* A confident null. Nothing in the email is a purchase order number, so
  ``po_number`` is ``None`` with a high ``p_null``; Clairic's ``guess`` is shown for
  transparency and is never used as a value.
* The verification ladder at work. Look at the "Verification" section: a vendor
  answer below 0.85 confidence is re-asked in isolation, and ``approved_by`` was
  answered with a name that the attribution check (Clef) refused to attribute to
  this field, so it was rejected rather than guessed.
* "Do the maths in code." Clairic is not a calculator, so the due date is not
  extracted: a pydantic ``computed_field`` adds the extracted terms to the issue date.

Live mode runs on the real models; its confidences and verification events will
differ. Offline mode (no credentials, or ``--offline``) scripts plausible answers.
"""

import sys
from datetime import date, timedelta
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
from _offline import OfflineScript, nothing, pick, says
from pydantic import BaseModel, Field, computed_field

from trustcall_system_one import ExtractionResult

EMAIL = """\
From: billing@brightwave-supplies.example
To: priya.nair@northwind-traders.example
Subject: Invoice INV-2024-0457 from Brightwave Supplies Ltd

Hi Priya,

Please find attached invoice INV-2024-0457 from Brightwave Supplies Ltd, issued on 14 March 2024 for the standing desks delivered last week.

Subtotal: $1,000.00
Sales tax (8%): $80.00
Total due: $1,080.00 USD

Payment terms are net 30 days. Your previous invoice, INV-2024-0412, was settled on 2 March, thank you. Please send any questions to Dana Whitlock on our accounts team.

Kind regards,
Brightwave Supplies Accounts"""  # noqa: E501


class Invoice(BaseModel):
    """An invoice received from a vendor."""

    invoice_number: str = Field(description="invoice number of this invoice")
    vendor: str = Field(description="company that issued the invoice")
    issue_date: date = Field(description="date the invoice was issued")
    terms_days: int = Field(ge=0, description="payment terms in days")
    total: Decimal = Field(description="amount due on the invoice, tax included")
    currency: Literal["USD", "EUR", "GBP"]
    paid: bool = Field(description="this invoice has already been paid")
    po_number: str | None = Field(None, description="purchase order number")
    approved_by: str | None = Field(
        None, description="person at the customer who approved paying the invoice"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def due_date(self) -> date:
        """Issue date plus the payment terms (arithmetic belongs in code)."""
        return self.issue_date + timedelta(days=self.terms_days)


def offline_script() -> OfflineScript:
    """What Clairic and Clef would plausibly answer for ``EMAIL``."""
    currency = ["USD", "EUR", "GBP"]
    return OfflineScript(
        presence=0.99,
        clairic={
            "invoice_number": says(
                "text",
                "INV-2024-0457",
                find="INV-2024-0457",
                p_null=0.002,
                confidence=0.99,
            ),
            # Below 0.85 confidence: the ladder will re-ask it.
            "vendor": says(
                "text",
                "Brightwave Supplies Ltd",
                find="Brightwave Supplies Ltd",
                occurrence=1,
                p_null=0.05,
                confidence=0.74,
            ),
            "issue_date": says("date", "2024-03-14", p_null=0.01, confidence=0.96),
            "terms_days": says("number", 30, p_null=0.01, confidence=0.95),
            "total": says("number", 1080.0, p_null=0.01, confidence=0.97),
            # Not in the email. Clairic's best guess is the *other* invoice number.
            "po_number": nothing("text", p_null=0.97, guess="INV-2024-0412"),
            # A person's name is the only candidate; she is the vendor's contact.
            "approved_by": says(
                "text",
                "Dana Whitlock",
                find="Dana Whitlock",
                p_null=0.12,
                confidence=0.88,
            ),
        },
        clef={
            "currency": pick("USD", currency, 0.98),
            "paid": 0.03,
        },
        reask={
            "vendor": says(
                "text",
                "Brightwave Supplies Ltd",
                find="Brightwave Supplies Ltd",
                occurrence=1,
                p_null=0.02,
                confidence=0.97,
            )
        },
        attribution={
            "attr.invoice_number": 0.99,
            "attr.vendor": 0.97,
            "attr.issue_date": 0.96,
            "attr.terms_days": 0.93,
            "attr.total": 0.98,
            "attr.approved_by": 0.12,  # the email never says she approved anything
        },
    )


def main(offline: bool = False) -> ExtractionResult:
    """Extract the invoice and print what happened; return the result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    with build_extractor([Invoice], offline=offline, script=script) as extractor:
        result = extractor.invoke(EMAIL)

    (record,) = result.records
    show_source("The email, with the spans the model pointed at", EMAIL, record)
    show_record(record)
    invoice = record.value
    if isinstance(invoice, Invoice):
        heading("Maths in code")
        print(f"  issue_date {invoice.issue_date} + terms_days {invoice.terms_days}")
        print(
            f"  -> due_date {invoice.due_date}"
            "   (a computed_field, never asked of a model)"
        )
        print()
    show_verification(result)
    show_summary(result)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
