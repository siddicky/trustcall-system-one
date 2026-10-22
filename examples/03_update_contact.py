"""Update a CRM record from a short note, as a JSON Patch, without clobbering it.

``uv run python examples/03_update_contact.py [--offline]``

The idea this library borrows from trustcall: do not ask a model to regenerate a
record, ask what the new text *says*, and turn the difference into RFC 6902 patch
operations in code. Silence is never deletion.

What to look at in the output:

* The patch. Exactly four ``replace`` operations (phone, city, company, title) and
  one ``add /notes/-``: the facts the note actually states. Existing notes stay and
  the new one is appended (a list has set semantics: only elements the record does
  not hold are added).
* "Not applied". ``email`` is absent from the note: Clairic answers null, so there
  is no operation and the field is reported as ``not_mentioned``. ``name`` is the
  interesting one: Clairic returns the verbatim span "Dana", a short form, so a
  naive diff would rename the contact from "Dana Whitfield" to "Dana". The diff
  never replaces an identity value with a token subset of the existing one
  (``identity_subset``), before any model is asked about it.
* Each applied operation passed three checks: the value is stated about this record
  (attribution), a second Clairic pass over the note alone, without the record,
  read the same value (the cross-check: it catches a value merely echoed from the
  record), and the change gate (one Clef yes/no per operation, comparing the
  proposed change against the record) said the note supports the change.

``json_doc_id`` is the id you gave the record, so patches can be written back to
your store. Live mode runs on the real models; offline mode scripts the answers.
"""

import sys
from pathlib import Path

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
    show_patches,
    show_rejected,
    show_source,
    show_summary,
)
from _offline import OfflineScript, change_gate, nothing, says
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult, One

NOTE = (
    "Quick note after catching up with Dana: she moved to Berlin last month and "
    "started as Head of Ops at Northwind. Her mobile is now +49 30 5555 0198. "
    "She also asked to be introduced to our logistics team."
)


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    email: str
    phone: str | None = None
    city: str | None = None
    company: str | None = None
    title: str | None = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


EXISTING = Contact(
    name="Dana Whitfield",
    email="dana.whitfield@fabrikam.example",
    phone="+1 415 555 0134",
    city="Austin",
    company="Fabrikam",
    title="Operations Manager",
    notes=["Prefers email over phone", "Met at the 2023 logistics summit"],
)


def offline_script() -> OfflineScript:
    """What Clairic and Clef would plausibly answer for ``NOTE`` and the record."""
    script = OfflineScript(
        source=NOTE,
        presence=0.97,
        clairic={
            # A short form: a verbatim span of the text, but not a new name.
            "name": says("text", "Dana", find="Dana", p_null=0.05, confidence=0.9),
            "email": nothing("text", p_null=0.98),
            "phone": says("text", "+49 30 5555 0198", find="+49 30 5555 0198"),
            "city": says("text", "Berlin", find="Berlin", confidence=0.97),
            "company": says("text", "Northwind", find="Northwind", confidence=0.96),
            "title": says("text", "Head of Ops", find="Head of Ops", confidence=0.95),
            "notes": says(
                "list",
                None,
                find_all=["asked to be introduced to our logistics team"],
                p_null=0.04,
                confidence=0.9,
            ),
        },
    )
    # The change gate: every proposed operation is checked against the record. (The
    # short name never gets that far: the diff refuses it as an identity subset.)
    script.handlers["gate"] = change_gate(default=0.94)
    return script


def main(offline: bool = False) -> ExtractionResult:
    """Update the contact and print the patch; return the result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    existing = [("crm-4711", "Contact", EXISTING.model_dump())]
    with build_extractor([One(Contact)], offline=offline, script=script) as extractor:
        result = extractor.invoke(NOTE, existing=existing)

    record = only_record(result, "contact")
    if record is None:
        show_call_log(result, script)
        return result
    show_source("The note, with the spans the model pointed at", NOTE, record)
    print(f"record {record.json_doc_id!r}: {record.kind}\n")
    show_before_after(record)
    show_patch_lines(record, "Each applied operation, next to the text behind it", NOTE)
    show_patches(record)
    show_rejected(record, "Not applied (and why)")
    show_summary(result, offline=offline)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
