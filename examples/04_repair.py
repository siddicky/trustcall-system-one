"""When validation fails, repair only the failing fields, with the error in hand.

Needs milestone M6 (the repair loop); until it lands the record comes back invalid
(``tests/unit/test_examples.py`` expects that and flips when M6 does).
``uv run python examples/04_repair.py [--offline]``

trustcall feeds validation errors back to the model so it can fix its tool call.
Here the same idea is narrower and cheaper: after the first extraction the record is
validated with pydantic; every failing field gets ONE small Clairic re-ask whose state
holds the previous attempt, the validation error and the field's schema excerpt, and
the answer goes through the normal verification ladder again. Fields that were fine
are not asked again.

The schema has two constraints a first extraction easily breaks:

* ``country`` must be an ISO code (``^[A-Z]{2}$``) but the text says "Germany", and
  Clairic returns verbatim text;
* ``age`` must be 0..130 but the text says "born in 1990" before it says "34 years
  old", and 1990 is a perfectly good number.

Look at: ``attempts`` (2: the extraction and one repair round), the two ``repair``
calls (one per failing field, nothing else re-asked) and, offline, the repair request
exactly as the library sent it, with ``<validation_errors>`` and the schema excerpt in
its state. The valid values at the end are the repaired ones.

Offline is the main mode here: the answers are scripted so that the first attempt
fails. Live, the models may well get both fields right first time, in which case no
repair is needed and ``attempts`` is 1; that is the feature working too.
"""

import sys
from pathlib import Path

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
from _offline import OfflineScript, says
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult

APPLICATION = """\
Application for the Data Analyst role

Name: Katarina Vogel

I was born in 1990 in Leipzig, so I am 34 years old, and I have lived in Germany all my life. I currently work as a BI developer in Berlin and would like to move into analytics."""  # noqa: E501


class Applicant(BaseModel):
    """A job applicant."""

    name: str
    country: str = Field(
        pattern=r"^[A-Z]{2}$",
        description="ISO 3166-1 alpha-2 code of the country the applicant lives in",
    )
    age: int = Field(ge=0, le=130, description="age in years")


def offline_script() -> OfflineScript:
    """First answers that fail validation, then the repaired ones."""
    return OfflineScript(
        presence=0.98,
        clairic={
            "name": says("text", "Katarina Vogel", find="Katarina Vogel", p_null=0.01),
            # Verbatim text, so not an ISO code.
            "country": says("text", "Germany", find="Germany", confidence=0.93),
            # A number from the text, but the birth year rather than the age.
            "age": says("number", 1990, p_null=0.06, confidence=0.9),
        },
        attribution={"attr.age": 0.8},  # "born in 1990": a lenient yes
        repair={
            # The narrow re-asks answer from the previous attempt and the error.
            "country": says("text", "DE", p_null=0.02, confidence=0.9),
            "age": says("number", 34, p_null=0.02, confidence=0.96),
        },
    )


def show_repair_requests(script: OfflineScript) -> None:
    """Print the repair requests as the library sent them (offline only)."""
    sent = [
        (payload, meta) for _, payload, meta in script.calls if meta.kind == "repair"
    ]
    if not sent:
        return
    heading("A repair request, as sent (Clairic)")
    for payload, meta in sent[:1]:
        print(f"  field: {meta.field}")
        for name, spec in payload["fields"].items():
            print(f"  instructions for {name}: {spec['instructions']}")
        print("  state:")
        for line in str(payload["state"]).splitlines():
            print("    " + line)
    print()


def main(offline: bool = False) -> ExtractionResult:
    """Extract the applicant, let validation fail and repair; return the result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    with build_extractor([Applicant], offline=offline, script=script) as extractor:
        result = extractor.invoke(APPLICATION)

    (record,) = result.records
    show_source(
        "The application, with the spans the model pointed at", APPLICATION, record
    )
    print(f"attempts: {result.attempts}  (1 = the first extraction was valid)\n")
    repairs = [call for call in result.calls if call.kind == "repair"]
    for call in repairs:
        print(f"  repair call for field {call.field!r}")
    if repairs:
        print()
    show_record(record)
    if script is not None:
        show_repair_requests(script)
    show_verification(result, kinds=("repair", "reask", "adjudicate"))
    show_summary(result)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
