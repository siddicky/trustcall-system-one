"""Edit one line of an order and add another, leaving every other line alone.

``uv run python examples/07_order_line_items.py [--offline]``

``Order.items`` is a ``list[LineItem]`` marked ``Many(identity="name")``. The same
collection engine that finds people in a conversation runs *inside* the order,
scoped to it: it discovers which line items the email talks about, matches each
one to an existing line (or decides it is new), extracts only that line's fields
and turns the difference into patch operations whose paths point inside the order.

The email: "please add two more of the blue mugs and drop the quantity of the tea
towels to 1".

What to look at:

* The patch has exactly two operations: ``replace /items/1/quantity`` (the tea towels,
  4 -> 1) and ``add /items/-`` (the blue mugs, a line the order did not have). Both
  were gated by Clef; the first against the tea towels' own record.
* The steel kettle and the wooden spoon set are not mentioned, so they are not
  asked about and appear unchanged in the "Before -> after" view: no operation
  touches them.
* "Not applied": the tea towels' name came back as "tea towels" (the email's wording)
  for a line called "Tea towels", so no rename is made (``unchanged``); fields the
  email is silent about are null (``p_null``) and never delete anything.
* Relative edits are not supported (a settled decision: ``docs/DESIGN.md``, "Design
  decisions settled from the examples review"). The library reads what the text
  *states* and never computes from the old value. "Two more of the blue mugs" is
  the new line with the quantity the text states, 2, here only because the order
  has no blue mugs line. Had it one with 3, the extraction would still read 2 (the
  text's number, not 3 + 2), so the change gate would be judging "quantity 2" against
  a record that says 3: say "make it five" when you mean five. That is why the
  tea towels edit is absolute ("to 1").

Live mode runs on the real models; offline mode scripts the answers.
"""

import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    build_extractor,
    cli,
    only_record,
    print_banner,
    resolve_offline,
    show_before_after,
    show_call_log,
    show_patch_lines,
    show_rejected,
    show_source,
    show_summary,
    show_verification,
)
from _offline import (
    OfflineScript,
    change_gate,
    entity_count,
    mention_discovery,
    nothing,
    pair_coreference,
    qualifier,
    record_matcher,
    says,
)
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult, Many, One

EMAIL = (
    "Hi Sam, it's Ines about order SO-10442. Please add two more of the blue mugs "
    "and drop the quantity of the tea towels to 1. Everything else stays as it is. "
    "Thanks!"
)


class LineItem(BaseModel):
    """One product line on an order."""

    name: str = Field(description="product name")
    quantity: int = Field(ge=1, description="number of units ordered")
    sku: str | None = Field(None, description="product code")


class Order(BaseModel):
    """A customer order."""

    order_id: str
    customer: str
    items: Annotated[list[LineItem], Many(identity="name")]


EXISTING = Order(
    order_id="SO-10442",
    customer="Ines Romero",
    items=[
        LineItem(name="Steel kettle", quantity=1, sku="KTL-STL"),
        LineItem(name="Tea towels", quantity=4, sku="TWL-LIN"),
        LineItem(name="Wooden spoon set", quantity=2, sku="SPN-WD"),
    ],
)


def offline_script() -> OfflineScript:
    """What Clairic and Clef would plausibly answer for ``EMAIL``."""
    script = OfflineScript(
        source=EMAIL,
        presence=0.98,
        clairic={
            # The order itself: the id is restated, the customer is not mentioned.
            "order_id": says("text", "SO-10442", find="SO-10442", confidence=0.98),
            "customer": nothing("text", p_null=0.96),
            # The line items, one entity each.
            "tea towels::name": says("text", "tea towels", find="tea towels"),
            "tea towels::quantity": says("number", 1, p_null=0.02, confidence=0.94),
            "tea towels::sku": nothing("text", p_null=0.97),
            "blue mugs::name": says("text", "blue mugs", find="blue mugs"),
            "blue mugs::quantity": says("number", 2, p_null=0.03, confidence=0.92),
            "blue mugs::sku": nothing("text", p_null=0.96),
        },
    )
    script.handlers["discover"] = mention_discovery(["blue mugs", "tea towels"])
    script.handlers["count"] = entity_count(2)
    script.handlers["coref"] = pair_coreference({})
    script.handlers["qualify"] = qualifier({})
    script.handlers["match"] = record_matcher(
        {"tea towels": "Tea towels", "blue mugs": None}
    )
    script.handlers["gate"] = change_gate(0.94)
    return script


def main(offline: bool = False) -> ExtractionResult:
    """Apply the email to the order and print the patch; return the result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    existing = [("SO-10442", "Order", EXISTING.model_dump())]
    extractor = build_extractor(
        [One(Order)],
        offline=offline,
        script=script,
        enable_inserts=True,
    )
    with extractor:
        result = extractor.invoke(EMAIL, existing=existing)

    record = only_record(result, "order")
    if record is None:
        show_call_log(result, script)
        return result
    show_source("The email, with the spans the model pointed at", EMAIL, record)
    show_before_after(record)
    show_patch_lines(record, "Patch (inside the order)", EMAIL)
    show_rejected(record, "Not applied (and why)")
    show_verification(result)
    show_summary(result, offline=offline)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
