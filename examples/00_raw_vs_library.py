"""The CRM update of example 03 twice: by hand over REST, then with the library.

``uv run python examples/00_raw_vs_library.py [--offline]``

Part 1 makes the raw Clairic call with ``httpx``, the way the README's "Without the
library" section does, prints what comes back, and then does the obvious thing with
it: copy every non-null ``value`` over the record. Look at what that does to the
contact: the name shrinks to "Dana", and the open-ended ``notes`` list is replaced by
whatever fragments the model returned. The response carries typed answers, not a
patch; nothing in it knows about the existing record, checks who a value is about,
or validates the result.

Part 2 is the same task through ``create_extractor(...).invoke(...)``: a JSON Patch,
each operation checked, and every refusal named.

Live mode needs ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN``. Offline mode
answers part 1 with ``httpx.MockTransport`` over the scripted models of example 03,
so the offline notes are tidier than the live ones (live, Clairic returns junk
fragments for ``notes``); the library's code in part 2 is real either way.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from _common import (
    build_extractor,
    cli,
    heading,
    only_record,
    print_banner,
    resolve_offline,
    show_patches,
    show_rejected,
    show_summary,
)
from _offline import OfflineScript

from trustcall_system_one import CLAIRIC, CallMeta, ExtractionResult, One

contact = importlib.import_module("03_update_contact")
NOTE: str = contact.NOTE

FOCUS = '<focus entity="the Contact">\nAnswer only about the Contact.\n</focus>\n'
URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/" + CLAIRIC


def request_body(note: str) -> dict[str, Any]:
    """The Clairic request: the note as ``state`` and one typed spec per field."""

    def text(what: str, null_threshold: float) -> dict[str, Any]:
        spec = f"the Contact's {what}"
        return {"type": "text", "instructions": spec, "null_threshold": null_threshold}

    return {
        "model": "clairic",
        "state": FOCUS + f"<input>\n{note}\n</input>",
        "fields": {
            "name": text("name", 0.7),
            "email": text("email", 0.7),
            "phone": text("phone", 0.3),
            "city": text("city", 0.3),
            "company": text("company", 0.3),
            "title": text("title", 0.3),
            "notes": {
                "type": "list",
                "instructions": "the Contact's things worth remembering",
                "items": "text",
            },
        },
        "options": {"return_state_text": True},
    }


def scripted_clairic(script: OfflineScript) -> httpx.MockTransport:
    """Answer the raw request from the scripted models, in the REST envelope."""

    def handler(request: httpx.Request) -> httpx.Response:
        meta = CallMeta(kind="extract", schema="Contact")
        sent = json.loads(request.content)
        reply = asyncio.run(script.transport.run(CLAIRIC, sent, meta=meta))
        envelope = {"success": True, "errors": [], "messages": [], "result": reply.body}
        return httpx.Response(200, json=envelope)

    return httpx.MockTransport(handler)


def raw_call(offline: bool) -> dict[str, Any]:
    """POST the request and unwrap ``{success, result}``; return ``result``."""
    if offline:
        client = httpx.Client(transport=scripted_clairic(contact.offline_script()))
        account = "scripted"
    else:
        account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        token = os.environ["CLOUDFLARE_API_TOKEN"]
        client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"}, timeout=60.0
        )
    with client:
        response = client.post(URL.format(account=account), json=request_body(NOTE))
    response.raise_for_status()
    envelope = response.json()
    if not envelope["success"]:
        raise RuntimeError(envelope["errors"])
    result: dict[str, Any] = envelope["result"]
    return result


def show_raw(answers: dict[str, Any]) -> None:
    """One line per answer: what the model says, and how sure it claims to be."""
    heading("Part 1. The raw answers (you threshold them yourself)")
    for name, answer in answers.items():
        value = json.dumps(answer["value"], ensure_ascii=False)
        p_null = answer.get("p_null")
        shown = f"p_null={p_null:.3g}  " if p_null is not None else ""
        print(f"  {name:<8} {value}")
        print(f"           {shown}confidence={answer['confidence']:.3g}")
    print()


def show_naive_merge(answers: dict[str, Any]) -> None:
    """Trust every non-null ``value`` and write it over the record."""
    heading("Part 1. Trusting every non-null value, over the existing record")
    record = contact.EXISTING.model_dump()
    for name, answer in answers.items():
        if answer["value"] is not None and answer["value"] != record[name]:
            print(
                f"  {name}: {json.dumps(record[name])} -> {json.dumps(answer['value'])}"
            )
    print("  (no patch, no check against the record, no validation)\n")


def main(offline: bool = False) -> ExtractionResult:
    """Make the raw call, merge it naively, then run the library; return its result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    answers = raw_call(offline)["answers"]
    show_raw(answers)
    show_naive_merge(answers)

    heading("Part 2. The same task with the library")
    script = contact.offline_script() if offline else None
    existing = [("crm-4711", "Contact", contact.EXISTING.model_dump())]
    with build_extractor([One(contact.Contact)], offline=offline, script=script) as ex:
        result = ex.invoke(NOTE, existing=existing)
    print()
    record = only_record(result, "contact")
    if record is not None:
        show_patches(record)
        show_rejected(record, "Not applied (and why)")
    show_summary(result, offline=offline)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
