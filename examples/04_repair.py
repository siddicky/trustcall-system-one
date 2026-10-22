"""When validation fails, repair only the failing fields, with the error in hand.

Needs milestone M6 (the repair loop). Until it lands the record simply comes back
invalid and no repair request is made; the example prints that honestly, and
``tests/unit/test_examples.py`` pins the gap so M6 has to update it.
``uv run python examples/04_repair.py [--offline]``

trustcall feeds validation errors back to the model so it can fix its tool call. Here
the same idea is narrower and cheaper: after the first extraction the record is
validated with pydantic; every failing field gets ONE small Clairic re-ask whose state
holds the previous attempt, the validation error and the field's schema excerpt, and
the answer goes through the normal verification ladder again. Fields that were fine
are not asked again.

The schema has two constraints that a first extraction breaks, each for a different
reason:

* ``postcode`` must be five digits (``^\\d{5}$``) but the first answer is the whole
  address fragment "Berlin 10115", which is a verbatim span of the text and fails the
  pattern. The re-ask sees the error and answers "10115": also a verbatim span.
* ``age`` must be 0..130. The text says "born in 1990" before it says "34 years old",
  and Clairic first answers 1990. The attribution check (Clef: "does the input say the
  age is 1990?") refuses it, so the required field is missing and validation fails with
  "Field required". The re-ask answers 34.

Repair is for fixing *extraction* mistakes. Clairic is extractive, so a repaired value
must be somewhere in the text; repair cannot turn "Germany" into an ISO code like "DE".
To normalise to a vocabulary, make the field a ``Literal`` (or an ``Enum``): Clef then
chooses among the allowed values and no validation error arises in the first place.

Look at: the final valid record, the two ``repair`` calls (one per failing field,
nothing else re-asked) and, offline, a repair request exactly as the library sent it,
with the validation error in its state. Offline is the main mode: the answers are
scripted so that the first attempt fails. Live, the models may well get both fields
right first time, in which case no repair is needed and there is one extraction round;
that is the feature working too.
"""

import sys
from pathlib import Path

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
from _offline import OfflineScript, says
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult

APPLICATION = """\
Application for the Data Analyst role

Name: Katarina Vogel
Address: Friedrichstraße 12, Berlin 10115

I was born in 1990 in Leipzig, so I am 34 years old. I currently work as a BI developer and would like to move into analytics."""  # noqa: E501


class Applicant(BaseModel):
    """A job applicant."""

    name: str
    postcode: str = Field(
        pattern=r"^\d{5}$",
        description="five-digit German postcode of the applicant's address",
    )
    age: int = Field(ge=0, le=130, description="age in years")


def offline_script() -> OfflineScript:
    """First answers that fail validation, then the repaired ones."""
    return OfflineScript(
        source=APPLICATION,
        presence=0.98,
        clairic={
            "name": says("text", "Katarina Vogel", find="Katarina Vogel", p_null=0.01),
            # A verbatim span, but the city comes along: fails the pattern.
            "postcode": says(
                "text", "Berlin 10115", find="Berlin 10115", confidence=0.9
            ),
            # A number from the text, but the birth year rather than the age.
            "age": says("number", 1990, p_null=0.06, confidence=0.9),
        },
        attribution={
            "attr.postcode": 0.9,
            # First check: the text does not say the age is 1990, so it is dropped.
            # Second check, of the repaired answer: it says 34.
            "attr.age": [0.12, 0.97],
        },
        repair={
            # The narrow re-asks see the error and answer from the text.
            "postcode": says(
                "text", "10115", find="10115", p_null=0.02, confidence=0.95
            ),
            "age": says("number", 34, p_null=0.02, confidence=0.96),
        },
    )


def show_repair_requests(script: OfflineScript) -> None:
    """Print a repair request as the library sent it (offline only)."""
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

    record = only_record(result, "applicant")
    if record is None:
        show_call_log(result, script)
        return result
    show_source(
        "The application, with the spans the model pointed at", APPLICATION, record
    )
    repairs = [call for call in result.calls if call.kind == "repair"]
    outcome = "valid" if record.valid else "INVALID (no repair was attempted)"
    print(f"extraction rounds: {result.attempts}   final record: {outcome}")
    for call in repairs:
        print(f"  repair call for field {call.field!r}")
    print()
    show_record(record)
    if script is not None:
        show_repair_requests(script)
    show_verification(result, kinds=("repair", "reask", "adjudicate"))
    show_call_log(result, script)
    show_summary(result, offline=offline)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
