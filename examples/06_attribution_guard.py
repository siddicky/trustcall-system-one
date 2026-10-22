"""Stop one person's fact from landing on the other person's record.

Needs milestone M7 (multi-entity extraction with the attribution check); until it
lands ``Many(...)`` raises ``NotImplementedError`` (``tests/unit/test_examples.py``
expects that and flips when M7 does).
``uv run python examples/06_attribution_guard.py [--offline]``

The failure this guards against is the quiet one. Extraction with several people in a
passage is asked per person ("answer only about Priya Raman"), but a short passage
puts "Lisbon" and "finance lead" a few words from both names, and a reader focused on
one person can pick up the other's fact with perfectly good confidence. The record then
validates and looks fine, and is wrong.

Every non-null Clairic value is therefore checked by Clef with a yes/no question that
names the entity: "Does ``input`` state that the role of ``Priya Raman`` is 'finance
lead'?". At least 0.6 keeps the value, below 0.3 drops it (``attribution``), and in
between the field is re-asked in isolation and checked once more.

What to look at: the "Naive vs guarded" table (what taking every Clairic answer at
face value would have written, next to what was kept) and the two ``rejected`` lines.
Priya ends up with no role and Tom with no city, because the passage states neither.

Offline, the bleed is injected on purpose: the scripted Clairic hands Priya the finance
role and Tom the Lisbon office, and the scripted Clef refuses both. Live, whatever the
real models do is printed: with luck the table shows no difference, which is also a
pass; the guard only matters when the bleed happens.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    build_extractor,
    cli,
    fmt,
    heading,
    print_banner,
    resolve_offline,
    show_overview,
    show_record,
    show_source,
    show_summary,
    show_verification,
)
from _offline import (
    OfflineScript,
    entity_count,
    mention_discovery,
    pair_coreference,
    qualifier,
    says,
)
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult, Many

PASSAGE = (
    "Priya Raman is moving to our Lisbon office next month. "
    "Tom Becker joins the same week as finance lead."
)


class Colleague(BaseModel):
    """A colleague named in an announcement."""

    name: str = Field(description="full name")
    city: str | None = Field(None, description="city of the office the person works in")
    role: str | None = Field(None, description="the person's job role")


def offline_script() -> OfflineScript:
    """Clairic bleeds each person's fact onto the other; Clef refuses it."""
    script = OfflineScript(
        presence=0.99,
        clairic={
            "Priya::name": says("text", "Priya Raman", find="Priya Raman"),
            "Priya::city": says("text", "Lisbon", find="Lisbon"),
            # Bleed: stated about Tom, picked up while focused on Priya.
            "Priya::role": says(
                "text",
                "finance lead",
                find="finance lead",
                p_null=0.12,
                confidence=0.88,
            ),
            "Tom::name": says("text", "Tom Becker", find="Tom Becker"),
            # Bleed: "the same week" is not "the same office".
            "Tom::city": says(
                "text", "Lisbon", find="Lisbon", p_null=0.14, confidence=0.87
            ),
            "Tom::role": says("text", "finance lead", find="finance lead"),
        },
        bleed=[("Priya", "finance lead", 0.06), ("Tom", "Lisbon", 0.09)],
    )
    script.handlers["discover"] = mention_discovery(
        ["Priya Raman", "Priya", "Tom Becker", "Tom"]
    )
    script.handlers["count"] = entity_count(2)
    script.handlers["coref"] = pair_coreference(
        {"Priya Raman": "priya", "Priya": "priya", "Tom Becker": "tom", "Tom": "tom"}
    )
    script.handlers["qualify"] = qualifier({})
    return script


def show_naive_vs_guarded(result: ExtractionResult) -> None:
    """For every field the guard changed, the raw Clairic answer and what stayed."""
    heading("Naive vs guarded (only the fields where they differ)")
    differences = 0
    for record in result.records:
        who = record.entity.label if record.entity else record.schema_name
        for answer in record.fields.values():
            naive = answer.raw if answer.engine == "clairic" else answer.value
            if answer.rejected_reason and naive is not None:
                differences += 1
                print(
                    f"  {who}.{answer.pretty}: naive {fmt(naive)}  ->  "
                    f"kept {fmt(answer.value)}"
                    f"   ({answer.rejected_reason}, attributed {answer.attribution_p})"
                )
    if not differences:
        print("  no difference: nothing was attributed to the wrong person this time")
    print()


def main(offline: bool = False) -> ExtractionResult:
    """Extract both colleagues and show what the attribution check refused."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    extractor = build_extractor(
        [Many(Colleague, identity="name")], offline=offline, script=script
    )
    with extractor:
        result = extractor.invoke(PASSAGE)

    show_source("The passage", PASSAGE)
    show_overview(result)
    for record in result.records:
        show_record(record, PASSAGE, snippets=True)
    show_naive_vs_guarded(result)
    show_verification(result)
    show_summary(result)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
