# trustcall-system-one

[![CI](https://github.com/siddicky/trustcall-system-one/actions/workflows/ci.yml/badge.svg)](https://github.com/siddicky/trustcall-system-one/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trustcall-system-one.svg)](https://pypi.org/project/trustcall-system-one/)

Extract typed data from text, and update the records you already have, on Cloudflare's
**System One** models. Patch, don't regenerate. Verify, don't generate.

System One models emit no tokens, messages or tool calls. **Clairic**
(`@cf/cloudflare/clairic`) reads text and returns typed answers with probabilities and
spans. **Clef** (`@cf/cloudflare/clef`) answers yes/no, choice and score questions.
This library turns those probabilities into validated pydantic objects, and turns
"what does this note say?" into RFC 6902 JSON Patch operations that are **computed in
code and checked one by one**, so a field the text never mentions is never touched.

It is a reimplementation of the ideas of [trustcall](https://github.com/hinthornw/trustcall)
(see [Credits](#credits)) with no LangChain or LangGraph. Status: alpha (`0.1.0a1`),
the API may still move.

On this page: [quickstart](#quickstart) · [raw calls vs the library](#without-the-library-vs-with-the-library) ·
[install](#install) · [insert](#extract-one-record) · [many people](#extract-and-update-many-records) ·
[never clobbered](#fields-the-text-does-not-mention-are-never-touched) ·
[types](#type-mapping) · [cardinality](#cardinality) · [verification](#verification-and-thresholds) ·
[cost](#requests-latency-and-cost) · [`existing`](#the-five-existing-shapes) ·
[testing](#testing-your-code) · [examples](#examples) · [limitations](#limitations)

## Quickstart

```bash
uv add --prerelease=allow trustcall-system-one
export CLOUDFLARE_ACCOUNT_ID=...   # your Cloudflare account id
export CLOUDFLARE_API_TOKEN=...    # a token with Workers AI access
```

Update an existing record from a note (`Contact`, `NOTE` and `RECORD` are defined in the
[next section](#without-the-library-vs-with-the-library)):

```python
from trustcall_system_one import One, create_extractor

with create_extractor([One(Contact)]) as extractor:
    result = extractor.invoke(NOTE, existing=[("crm-4711", "Contact", RECORD)])

record = result.records[0]
record.patches  # the JSON Patch operations that were applied
record.rejected  # everything computed and then refused, with a reason
record.after  # the patched record
```

Full setup, extraction without an existing record, and the details are [below](#install).

## Without the library vs with the library

One task, twice: **update a CRM contact from a short note** (this is
[`examples/03_update_contact.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/03_update_contact.py); the raw call of step 1, a naive merge of its values and the library run
below are [`examples/00_raw_vs_library.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/00_raw_vs_library.py)).

The note, the record you already hold, and your schema:

```python
from pydantic import BaseModel, Field

NOTE = (
    "Quick note after catching up with Dana: she moved to Berlin last month and "
    "started as Head of Ops at Northwind. Her mobile is now +49 30 5555 0198. "
    "She also asked to be introduced to our logistics team."
)
RECORD = {
    "name": "Dana Whitfield",
    "email": "dana.whitfield@fabrikam.example",
    "phone": "+1 415 555 0134",
    "city": "Austin",
    "company": "Fabrikam",
    "title": "Operations Manager",
    "notes": ["Prefers email over phone", "Met at the 2023 logistics summit"],
}


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
```

Every request and response below was captured from the wire during a **live run** on the
real models (2026-10-22, library commit `05207aa`; a traced rerun of example 03 whose printed
output is identical to the output shown further down). Bodies are verbatim. Where a field
is long I replaced it with `…` and say so. Account id and token are placeholders.

### Without the library

**1. Call Clairic.** Raw REST: one JSON spec per field.

```bash
curl -sS "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clairic" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "model": "clairic",
  "state": "<focus entity=\"the Contact\">\nAnswer only about the Contact.\n</focus>\n<input>\nQuick note after catching up with Dana: she moved to Berlin last month and started as Head of Ops at Northwind. Her mobile is now +49 30 5555 0198. She also asked to be introduced to our logistics team.\n</input>",
  "fields": {
    "name": {"type": "text", "instructions": "the Contact's name", "null_threshold": 0.7},
    "email": {"type": "text", "instructions": "the Contact's email", "null_threshold": 0.7},
    "phone": {"type": "text", "instructions": "the Contact's phone", "null_threshold": 0.3},
    "city": {"type": "text", "instructions": "the Contact's city", "null_threshold": 0.3},
    "company": {"type": "text", "instructions": "the Contact's company", "null_threshold": 0.3},
    "title": {"type": "text", "instructions": "the Contact's title", "null_threshold": 0.3},
    "notes": {"type": "list", "instructions": "the Contact's things worth remembering", "items": "text"}
  },
  "options": {"return_state_text": true}
}
JSON
```

The same call in Python with `httpx`:

```python
import os

import httpx

# NOTE is the note above
URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    f"{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/run/@cf/cloudflare/clairic"
)
HEADERS = {"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}


def text(what: str, null_threshold: float) -> dict:
    return {
        "type": "text",
        "instructions": f"the Contact's {what}",
        "null_threshold": null_threshold,
    }


body = {
    "model": "clairic",
    "state": f'<focus entity="the Contact">\nAnswer only about the Contact.\n</focus>\n<input>\n{NOTE}\n</input>',
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
envelope = httpx.post(URL, headers=HEADERS, json=body, timeout=60).json()
answers = envelope["result"]["answers"]
```

(Clairic reads `state` as plain text, so you could send the bare note. The `<focus>` and
`<input>` tags are this library's framing, kept so that the body is the one captured on
the wire. The Python above builds exactly that body: it is the library's own input-only
pass, request kind `crosscheck`.)

**2. The real response** (HTTP 200; `state_text`, an echo of the `state` you sent, trimmed to `…`):

<details>
<summary>The real Clairic response: typed answers with <code>p_null</code>, <code>confidence</code>, <code>guess</code> and <code>span</code>, and a <code>notes</code> list of fragments</summary>

```json
{
  "result": {
    "model": "clairic",
    "answers": {
      "name": {
        "type": "text",
        "p_null": 0.0012319437228143215,
        "span": [111, 115],
        "guess": "Dana",
        "confidence": 0.9999885559082031,
        "value": "Dana"
      },
      "email": {
        "type": "text",
        "p_null": 0.9999866485595703,
        "span": [111, 115],
        "guess": "Dana",
        "confidence": 0.536175549030304,
        "value": null
      },
      "phone": {
        "type": "text",
        "p_null": 3.4659529774216935e-05,
        "span": [207, 223],
        "guess": "+49 30 5555 0198",
        "confidence": 0.9999884366989136,
        "value": "+49 30 5555 0198"
      },
      "city": {
        "type": "text",
        "p_null": 0.00025216868380084634,
        "span": [130, 136],
        "guess": "Berlin",
        "confidence": 0.9999841451644897,
        "value": "Berlin"
      },
      "company": {
        "type": "text",
        "p_null": 0.013622269965708256,
        "span": [178, 187],
        "guess": "Northwind",
        "confidence": 0.9998990297317505,
        "value": "Northwind"
      },
      "title": {
        "type": "text",
        "p_null": 0.0015616711461916566,
        "span": [163, 174],
        "guess": "Head of Ops",
        "confidence": 0.987583577632904,
        "value": "Head of Ops"
      },
      "notes": {
        "type": "list",
        "value": [
          "started",
          "Head of Ops at",
          "Northwind",
          "mobile",
          "+49 30 5"
        ],
        "spans": [
          [152, 159],
          [163, 177],
          [178, 187],
          [193, 199],
          [207, 215]
        ],
        "confidence": 1.9324552606715345e-06
      }
    },
    "truncated": false,
    "usage": {
      "state_tokens": 82,
      "image_tokens": 0,
      "schema_tokens": 834,
      "input_tokens": 1408,
      "output_tokens": 0,
      "passes": 7
    },
    "state_text": "…"
  },
  "success": true,
  "errors": [],
  "messages": []
}
```

</details>

**3. What this response does not give you.** All of it is visible in the response above (expand it).

- **An envelope.** The answers sit under `result`, next to `success`, `errors` and
  `messages`. Unwrapping it, checking `success`, mapping errors and retrying 429s is yours.
- **Typed answers you must threshold yourself.** Each carries `value`, `p_null`,
  `confidence`, `guess` and `span`, and they can disagree. `email` has `"value": null`
  and `p_null` 0.99999, but also `"guess": "Dana"` and `"span": [111, 115]`: read the span
  and "Dana" becomes the email. `name` is `"Dana"` at confidence 0.99999, a first name
  where the record holds "Dana Whitfield". `notes` is five fragments (`"started"`,
  `"Head of Ops at"`, `"+49 30 5"`, ...) at `confidence` 1.9e-06; a list has no `p_null`, so the
  confidence is the only tell.
- **No booleans or enums.** Clairic has no such types; those need a second model with a
  different request shape (step 5).
- **No JSON Patch.** You get values, not changes. Diffing against the record, ordering
  the operations, `/notes/-` for lists and applying them is yours. Copying every
  non-null `value` over the record would turn "Dana Whitfield" into "Dana" and replace
  both existing notes with the five fragments.
- **No check that a value is about the right person.** A span says where a value was
  read, not whom it describes. This note has one person, so nothing goes wrong here;
  [`examples/06`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/06_attribution_guard.py) injects the case where it does.
- **No validation or repair.** Nothing checks the values against your schema, and no
  request is made again when one fails.
- **No multi-entity matching.** One request answers about one entity. With several
  people in the text, finding them, grouping their mentions and matching each to a
  record is yours.

**4. The model has no notion of "the existing record".** The obvious move is to put the
record in `state` as context. This is the `state` the library sends for that (kind
`extract`; its `fields` are the same seven; the JSON string is shown with its newlines
and quotes unescaped):

```text
<focus entity="the Contact">
Answer only about the Contact.
The existing record after the input is context only: answer from the input text, never from the existing record.
</focus>
<input>
Quick note after catching up with Dana: she moved to Berlin last month and started as Head of Ops at Northwind. Her mobile is now +49 30 5555 0198. She also asked to be introduced to our logistics team.
</input>
<existing id="crm-4711" schema="Contact">
{"name": "Dana Whitfield", "email": "dana.whitfield@fabrikam.example", "phone": "+1 415 555 0134", "city": "Austin", "company": "Fabrikam", "title": "Operations Manager", "notes": ["Prefers email over phone", "Met at the 2023 logistics summit"]}
</existing>
```

`phone`, `city` and `company` of the response (the other four answers trimmed to `…`; `state_text` trimmed):

<details>
<summary>The real response when the record is in <code>state</code>: <code>phone</code> and <code>company</code> are the record's old values, read from inside the <code>&lt;existing&gt;</code> block</summary>

```json
{
  "result": {
    "model": "clairic",
    "answers": {
      "name": "…",
      "email": "…",
      "phone": {
        "type": "text",
        "p_null": 0.005261575803160667,
        "span": [525, 540],
        "guess": "+1 415 555 0134",
        "confidence": 0.9957774877548218,
        "value": "+1 415 555 0134"
      },
      "city": {
        "type": "text",
        "p_null": 0.0015457599656656384,
        "span": [243, 249],
        "guess": "Berlin",
        "confidence": 0.602903425693512,
        "value": "Berlin"
      },
      "company": {
        "type": "text",
        "p_null": 0.01499671209603548,
        "span": [573, 581],
        "guess": "Fabrikam",
        "confidence": 0.8630199432373047,
        "value": "Fabrikam"
      },
      "title": "…",
      "notes": "…"
    },
    "truncated": false,
    "usage": {
      "state_tokens": 211,
      "image_tokens": 0,
      "schema_tokens": 834,
      "input_tokens": 2311,
      "output_tokens": 0,
      "passes": 7
    },
    "state_text": "…"
  },
  "success": true,
  "errors": [],
  "messages": []
}
```

</details>

Five of the six text answers are the record's own old values, read from the
`<existing>` block (it starts at character 402; the spans 525 to 540 and 573 to 581
lie inside it): the old phone `+1 415 555 0134` at confidence 0.996, `Fabrikam`,
and likewise the name, the email and the title. Only `city` came from the note, at
0.60. This is despite the line "answer from the input text, never from the existing
record" in the `<focus>` header. To tell a changed value from an echoed one you have to
find where the `<existing>` block starts and compare every span against it yourself, and a
list answer (`notes`) mixes both: its spans fall on `mobile` in the note and on
`Prefers email over phone` in the record.

**5. Booleans and enums go to Clef**, a different model with a different request: yes/no
questions are `noul` (probability of yes), options are `choice` (a `criteria` map of
option to description), ordered scales are `score`. This is a real Clef request and response
from the same run (the library's "is there a Contact in this text?" question; `input`
trimmed to `…`). A `bool` field would be asked in exactly this shape:

<details>
<summary>A real Clef request and response: the presence question, a <code>noul</code> with its own request shape</summary>

```json
{
  "model": "clef",
  "state": {
    "input": "Quick note after catching up with Dana: … introduced to our logistics team."
  },
  "questions": {
    "presence": {
      "type": "noul",
      "instructions": "Does `input` contain a Contact: A contact in the CRM?",
      "criteria": {
        "true": "The input describes a Contact.",
        "false": "The input does not describe a Contact."
      }
    }
  }
}
```

```json
{
  "result": {
    "model": "clef",
    "answers": {
      "presence": {
        "type": "noul",
        "noul": 0.9351
      }
    },
    "usage": {
      "input_tokens": 204,
      "output_tokens": 0
    }
  },
  "success": true,
  "errors": [],
  "messages": []
}
```

</details>

**6. Then your code** has to do everything in the table below.

| What you'd have to build | What the library does |
| --- | --- |
| Unwrap `{success, result, errors, messages}`, map HTTP and Cloudflare errors, retry 408/429/5xx, stay under rate limits | `CloudflareTransport` unwraps the envelope, raises typed errors (`AuthError`, `BadRequestError`, `RateLimitError`, `ServerError`, `ModelResponseError`), retries 408/429/5xx honouring `Retry-After`, and halves its concurrency on a 429 |
| Pick thresholds for `p_null` and `confidence`; decide what to do with `guess` and a span on a null | Per-field `null_threshold` (0.7 required, 0.3 optional), minimum confidence 0.5, a re-ask of answers under 0.85, a Clef check of borderline `p_null` (0.25 to 0.75); a `guess` is metadata and is never used as a value |
| Ask booleans and enums on a second model, with its own request shape, and map the answers back | The schema compiler: `bool` becomes a Clef `noul`, `Literal` and `Enum` a Clef `choice` (with an `unstated` option when the field is nullable or defaulted); request and mapping back are generated |
| Keep the model from echoing the record you gave it | Spans inside the `<existing>` block are rejected (`echo_span`); a second Clairic pass reads the input alone and a changed value it does not reproduce is rejected (`echo_crosscheck`); disagreements go to a Clef adjudication |
| Turn answers into a JSON Patch | `add` and `replace` computed from the field plan (`remove` only when you opt in: `GateConfig(allow_null_removals=True)` for a field, `enable_item_removals` for items of a nested list; `enable_deletes` deletes whole records, returned as `kind="delete"` results), list elements by set semantics (`add /notes/-`), a fixed operation order, applied with `jsonpatch` on a copy; `before`, `after` and `patches` on the result |
| Know that a change is real and about this person | One Clef question per value ("does the input state this about the entity?"), kept at 0.75 or more, dropped under 0.4, re-asked in between; one Clef change gate per operation (add 0.55, replace 0.6); a short form of an identity value ("Dana" for "Dana Whitfield") is refused as `identity_subset` |
| Validate the result and fix what fails | The patched record is validated by pydantic; each failing field is re-asked alone with the error in the request (the repair loop), and operations on a failing path are dropped; `strict=True` raises instead |
| Find several entities, group mentions, match them to records | `Many(Schema)`: discovery, coreference (Clef), qualification ("is Sunny a Person?"), one-to-one matching to existing records or `__new__`, opt-in deletes |
| Track what every call cost | `result.usage` (by model and by kind), `result.cost`, and one `result.calls` entry per request |

### With the library

The same task, with `Contact`, `NOTE` and `RECORD` from above:

```python
from trustcall_system_one import One, create_extractor

existing = [("crm-4711", "Contact", RECORD)]  # (record id, schema name, record)

# credentials are read from CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
with create_extractor([One(Contact)]) as extractor:
    result = extractor.invoke(NOTE, existing=existing)

record = result.records[0]
record.patches  # the applied JSON Patch operations
record.rejected  # everything computed and then refused, each with a reason
record.after  # the patched record, validated as a Contact
```

**Live output** of `uv run python examples/03_update_contact.py`. It is identical to the
traced run whose wire bodies are above. Left out: the note with its spans bracketed, the
`record 'crm-4711': update` line, and each operation printed next to its text:

```text
LIVE — requests go to Cloudflare (Clairic and Clef)

Before -> after
---------------
  phone: "+1 415 555 0134" -> "+49 30 5555 0198"
  city: "Austin" -> "Berlin"
  company: "Fabrikam" -> "Northwind"
  title: "Operations Manager" -> "Head of Ops"
  unchanged: name, email, notes

JSON Patch (RFC 6902)
---------------------
[
  {
    "op": "replace",
    "path": "/phone",
    "value": "+49 30 5555 0198"
  },
  {
    "op": "replace",
    "path": "/city",
    "value": "Berlin"
  },
  {
    "op": "replace",
    "path": "/company",
    "value": "Northwind"
  },
  {
    "op": "replace",
    "path": "/title",
    "value": "Head of Ops"
  }
]

Not applied (and why)
---------------------
  /email  echo_span
      the answer's span lies in the existing record, not the input
  /notes  low_confidence (p=0.00)
      confidence 0.00 < 0.5
  /notes/2  echo_span
      '"+' was read from the existing record
  /notes/3  echo_span
      'Prefers email over phone' was read from the existing record
  /notes/4  echo_span
      'Met at the 2023 logistics summit' was read from the existing record
  replace /name  identity_subset
      'Dana' is only a short form of the record's 'Dana Whitfield'

9 requests (attribution 1 · crosscheck 1 · extract 1 · gate 4 · presence 1 · reask 1)
6,543 input tokens (clairic 3,937 · clef 2,606) · $0.000394 + clef unpriced
```

What to read in it:

- The four `replace` operations are the four facts the note states. Each one passed
  attribution, the input-only cross-check and its own Clef change gate.
- `replace /name` was **refused in code before any change gate was asked**: "Dana" is a
  short form of "Dana Whitfield" (`identity_subset`). `/email` is not mentioned; its answer was the
  record's own value, so it was rejected as an echo (`echo_span`).
- **The note is not appended, live.** The open-ended `notes` list is where Clairic fails: it
  returned fragments at confidence 3.2e-11 (with the record) and 1.9e-06 (without it).
  The library refuses them (`/notes` `low_confidence`, and the fragments read from the
  existing record `echo_span`), so `add /notes/-` never happens and "she asked to be
  introduced to our logistics team" is lost. The offline run of the same example *does*
  append it, because the script is told to; do not read that as live behaviour (see
  [Limitations](#limitations)).
- The price of checking: the raw call above is **1 request and 1,408 input tokens**. The
  library made **9 requests and 6,543 input tokens in about 4 seconds**: a presence check, the
  extraction with the record, the input-only cross-check (that request is the raw one
  above), one re-ask (`city` came back at 0.60 in step 4, and 0.9998 when asked alone), one
  attribution request for the five values, and four gates, one per operation.

<details>
<summary>One of the four change-gate requests behind those operations (real, <code>input</code> trimmed to …)</summary>

The library gives Clef the note, the record and the proposed change, and asks one yes/no question:

```json
{
  "model": "clef",
  "state": {
    "input": "Quick note after catching up with Dana: … introduced to our logistics team.",
    "entity": {
      "label": "the Contact",
      "mentions": []
    },
    "record": {
      "name": "Dana Whitfield",
      "email": "dana.whitfield@fabrikam.example",
      "phone": "+1 415 555 0134",
      "city": "Austin",
      "company": "Fabrikam",
      "title": "Operations Manager",
      "notes": [
        "Prefers email over phone",
        "Met at the 2023 logistics summit"
      ]
    },
    "proposed_change": {
      "op": "replace",
      "path": "/company",
      "field": "company",
      "from": "Fabrikam",
      "to": "Northwind"
    }
  },
  "questions": {
    "gate": {
      "type": "noul",
      "instructions": "Does `input` state the information in `proposed_change.to` about `entity.label`, as new or changed compared to `record`?",
      "criteria": {
        "true": "The input states the information in `proposed_change.to` about `entity.label`, and `record` does not already hold it.",
        "false": "The input does not state it about `entity.label`: it only gives a shorter or partial form, another value, something about somebody else, or it repeats what `record` already holds."
      }
    }
  }
}
```

```json
{
  "result": {
    "model": "clef",
    "answers": {
      "gate": {
        "type": "noul",
        "noul": 0.9827
      }
    },
    "usage": {
      "input_tokens": 379,
      "output_tokens": 0
    }
  },
  "success": true,
  "errors": [],
  "messages": []
}
```

0.9827 is above the replace threshold of 0.6, so `replace /company` stays.

</details>

## Install

Python 3.10 or newer. The package is a pre-release (alpha), so the `--pre` or
`--prerelease=allow` flag is needed to install it:

```bash
pip install --pre trustcall-system-one
# or, with uv:
uv add --prerelease=allow trustcall-system-one
```

To install the latest code from the repository instead, use [uv](https://docs.astral.sh/uv/):

```bash
uv add "git+https://github.com/siddicky/trustcall-system-one"
```

It depends on `pydantic>=2.7`, `httpx`, `jsonpatch`, `jsonpointer`, `typing-extensions`
and `annotated-types`. An optional extra, `jsonschema`, installs `dydantic` for JSON
Schema inputs.

Credentials come from the environment (or `CloudflareTransport(account_id=..., api_token=...)`):

```bash
export CLOUDFLARE_ACCOUNT_ID=...   # your Cloudflare account id
export CLOUDFLARE_API_TOKEN=...    # a token with Workers AI access
```

`CLOUDFLARE_AUTH_TOKEN` is accepted in place of `CLOUDFLARE_API_TOKEN`. Credentials are
read on the first request, so building an extractor needs none (tests build them with
`FakeTransport`). To go through AI Gateway pass `CloudflareTransport(gateway="<id>")`.

For development: `make install`, `make format`, `make lint`, `make test` (`make help` lists every
target). See [`CONTRIBUTING.md`](https://github.com/siddicky/trustcall-system-one/blob/main/CONTRIBUTING.md) for releases.

### Using Clef from Experiential Labs

Clef can be served by Experiential Labs while Clairic stays on Cloudflare. This sends your text to
another company, so it is always an explicit choice: set `EXPERIENTIAL_API_KEY` **and**
`TRUSTCALL_S1_CLEF_PROVIDER=experiential` (the key alone changes nothing), or pass a transport:

```python
from trustcall_system_one import (
    CloudflareTransport,
    ExperientialTransport,
    One,
    RoutingTransport,
    create_extractor,
)

transport = RoutingTransport(
    {"clef": ExperientialTransport()},  # key from EXPERIENTIAL_API_KEY
    default=CloudflareTransport(),  # Clairic
)
extractor = create_extractor([One(Contact)], transport=transport)
# or: system_one_transport("experiential")
```

`ExperientialTransport` sends each request once. After a timeout, a dropped connection or a 5xx it
may already have run, and been charged, so it raises `UncertainOutcomeError` instead of retrying;
only a failed connection and a 429 with `Retry-After` are retried. Experiential reports the price
of each call: see `result.usage.reported_cost_usd` and `result.cost.reported_usd`.

An Experiential request takes at most 32 questions (Cloudflare takes 64): the library splits larger
sets by the limit of the transport that serves the model. **Cloudflare stays the default, and the
thresholds were tuned on Cloudflare's Clef.** Clef on Experiential calibrates differently, so the
same input can score differently and results differ between providers. In live run 8 the presence
of a line item for "two more of the blue mugs" scored 0.21 and its qualify questions 0.07 and
0.016, and the attribution of "5 of the tea towels" 0.53 (0.83 on Cloudflare). A choice's
`confidence` is the top probability itself there, so a `__new__` is judged by its probability
(`match.new_final`) and not by that confidence. The thresholds have not been retuned for it.

As a fallback for Clef you can also opt in to Experiential's Decisions API
(`TRUSTCALL_S1_CLEF_FALLBACK=decisions`, or `system_one_transport(clef_fallback="decisions")`).
**The Decisions API is served by OpenAI through Experiential Labs, so opting in sends your text
to OpenAI as well.** A Clef request that its provider certainly did not process (refused with a
429, or the connection failed) is sent once to it, as a different model; the answers are marked
(`FieldAnswer.provider`) and the result carries the warning `clef_fallback_used`. After a
timeout or a 5xx the primary may already have run and been charged, so the same question is
*not* sent again unless you also opt in to that
(`TRUSTCALL_S1_CLEF_FALLBACK_ON=unavailable,uncertain`): then it can be paid for twice.
With Clef on Cloudflare the default falls back only for a 429 or a failed connection, and not
when any earlier attempt may have run. The Decisions API sends bare picks (probabilities of 0
and 1): they are capped at `DecisionsTransport(decisions_hard_cap=0.8)`, below the removal
gate (0.85), so a fallback answer can add or replace a value but cannot remove one by itself.
Hard picks of a PRIMARY Decisions Clef are reported as `clef_hard_picks: N Clef answer(s) from
decisions were hard picks ...; by call kind: ...` (never as `clef_fallback_used`, which is only
for requests a fallback answered). An answer the model refuses (`{"type": "refusal"}`, no
`choice`) leaves its question unanswered, with a `decisions_refusal` warning: no gate reads it as
a pass, and the run goes on.

### Clef from the Decisions API

`TRUSTCALL_S1_CLEF_PROVIDER=decisions` (or `system_one_transport("decisions")`, or `--clef-provider
decisions` in the evals) makes Experiential's Decisions API (`gpt-6-luna-decisions`, served by OpenAI)
answer **every** Clef request, not only after a failure, while Clairic stays on Cloudflare. It is an
opt-in like the others (an `EXPERIENTIAL_API_KEY` alone changes nothing), takes at most 32 questions
per request, and a bare pick is capped at 0.8 so it cannot authorise a removal by itself. That
guarantee has a price: a delete needs `match.delete_threshold` (0.85) and a saturated "yes" is 0.8,
so a record is deleted only when Decisions sends a graded answer at 0.85 or more (see
Limitations).

### TypeSafe-only: Jev for extraction and decisions

TypeSafe's **Jev** can replace both Cloudflare models, so one key and one provider serve the whole
pipeline. Opt in with `TRUSTCALL_S1_PROVIDER=typesafe` **and** `TYPESAFE_API_KEY` (the key alone changes
nothing), or pass the transport:

```python
from trustcall_system_one import One, TypeSafeTransport, create_extractor

# no Cloudflare credentials; the key is read from TYPESAFE_API_KEY
extractor = create_extractor([One(Contact)], transport=TypeSafeTransport())
# or: system_one_transport(provider="typesafe")
# optional: TypeSafeTransport(model="jev-1.14.0", escalate=my_reasoning_model_hook)
```

Jev answers closed questions and generates nothing, so the library turns Clairic into **selection**
(TypeSafe's extraction cookbooks): code finds the candidate spans (emails, phones, amounts, ids, names,
sentences, clauses), Jev picks one or `none`, and the answer carries the exact text and span. Dates,
times and durations are read as parts (month, day, year, anchor, weekday) that code resolves against
`reference_date`; free text is read as blocks after Jev stitches hard-wrapped lines; every selected value is
verified by two yes/no questions and a doubted one comes back as a low-confidence or null answer for the
usual ladder (or goes to your `escalate` hook). **Jev can only select a value that is among the
candidates**, so a value no generator finds is never returned, and a Clairic call is a few requests
(stitch, select, verify), not one. A date the text does not fully state (no year, a bare weekday) is
resolved in code but flagged `needs_review` with its confidence capped at 0.8, so the ladder re-checks it;
two-digit years, day-first against month-first dates and time zones are not read (see the design doc's
limits). Jev reads at most 32k tokens of state: the library narrows what it sends to that. The
thresholds are the cookbooks' and have not been tuned for this library. See [`docs/DESIGN.md`](https://github.com/siddicky/trustcall-system-one/blob/main/docs/DESIGN.md)
("TypeSafe-only provider") for the design and its limits.

To compare the three combinations on the evals (live; see the runbook): Cloudflare Clairic and Clef
(`--provider cloudflare`), Cloudflare Clairic with the Decisions API for Clef (`--clef-provider
decisions`), and Jev for both (`--provider typesafe`, which refuses `--clef-provider` and
`--clef-fallback`: Jev answers Clef itself). `results.json` records the combination as `extraction` and
`decisions` labels; a replay of cassettes sends nothing and says what they were recorded on
(`replay (recorded on typesafe jev)`).

### Mixing providers per call kind and field type

To find which provider is best for what, name each one in a **combination spec**: the extraction
provider per field type, the decision provider per call kind, and optional calibration shifts and
extractor knobs. Naming a provider is the opt-in to send text to it (a key in the environment never
selects one), and every provider is only given what it serves:

```python
from trustcall_system_one import create_extractor
from trustcall_system_one.combos import ComboSpec, build_combo_transport

spec = ComboSpec.model_validate(
    {
        "name": "jev-dates",
        "extraction": "cloudflare",  # Clairic: cloudflare | typesafe
        "extraction_by_type": {"date": "typesafe", "datetime": "typesafe"},
        "decision": "cloudflare",  # Clef: any of the four providers
        "by_kind": {"attribution": "decisions", "gate": "typesafe"},
        "calibration": {
            "decisions": {"attribution": 0.5}
        },  # logit shift of its probabilities
        "options": {"verify.attribution_keep": 0.7},
    }
)
extractor = create_extractor(
    [Person], transport=build_combo_transport(spec), **spec.extractor_options()
)
```

The providers are `cloudflare`, `experiential-clef`, `decisions` (the Decisions API, which is OpenAI behind
Experiential Labs) and `typesafe`; `spec.companies()` says who will see the text. The same spec in a
JSON file (YAML too if PyYAML is installed) runs on the evals, and a greedy search finds the best one:

```bash
uv run python -m evals combo --spec spec.json --live --record cassettes --tag jev-dates
uv run python -m evals combo --spec spec.json --suite holdout       # evals/holdout_cases.py
uv run python -m evals search --base spec.json --live --providers cloudflare,typesafe,decisions \
    --price cloudflare=IN:OUT --price typesafe=IN:OUT --price decisions=IN:OUT --budget-usd 5 --calibrate
```

`--price` is USD per million input and output tokens of a provider (a `--budget-usd` needs one for each).
`--replay` is for the calibration search alone (`--calibrate-only`), and a recording answers only the
provider that made it; `--resume` reuses a trial only in the mode (offline, live, replay) it ran in; a gain
must beat the incumbent by `--min-margin` (default one case or 0.02 F1); `gate`, `null_check` and `delete`
can authorise a removal, so they are not calibrated upwards unless asked twice.
`search` walks the decision kinds in groups and the field types, keeps a provider when more cases pass
(then op F1, then value accuracy, then fewer bleed and wrong deletes; a rise in either is a veto), never
retries a failed case, and prints a leaderboard of every spec it tried (dev suite only; check the winner
on the holdout). The building blocks are public too: `KindRoutingTransport({"attribution": ...,
"gate": ...}, default=...)`, `FieldTypeRoutingTransport({"date": ...}, default=...)` and
`CalibratedTransport(inner, {"attribution": 0.5})`. See [`docs/DESIGN.md`](https://github.com/siddicky/trustcall-system-one/blob/main/docs/DESIGN.md)
("Mixing providers per call kind and field type") for the rules.

## Extract one record

Schemas are plain pydantic models (also `TypedDict`s, JSON Schemas or annotated
functions). With no `existing` records, every schema is an insert.

```python
from pydantic import BaseModel, Field

from trustcall_system_one import create_extractor


class Invoice(BaseModel):
    """An invoice."""

    invoice_number: str
    total: float = Field(description="Amount due including shipping, in USD")
    contacts: list[str]


with create_extractor([Invoice]) as extractor:
    result = extractor.invoke(
        "Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30. "
        "4 pallets at $250.00 each, shipping $80.00. Contact: Dana Ruiz, Priya Shah."
    )

print(result.responses)
```

Live output (same run, 3 requests: presence, extract, attribution; 1,379 input tokens):

```text
(Invoice(invoice_number='INV-2291', total=1080.0, contacts=['Dana Ruiz', 'Priya Shah']),)
```

`result.responses` holds the validated models. `result.records[i]` is the audit
trail for each: `fields[path]` has the `FieldAnswer` (value, `p_null`, `confidence`,
`span`, attribution probability), `rejected` lists answers that were refused and why,
`valid` and `errors` say whether it validated. `total` was never written in the text;
Clairic worked out 4 x 250 + 80 and the attribution check accepted it (0.94), because the
question allows "simple arithmetic from stated figures". Examples
[`01`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/01_invoice.py), [`02`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/02_support_ticket.py) and
[`04`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/04_repair.py) go deeper (spans, nested models, repair).

`invoke` is synchronous (it runs on a private event-loop thread, so it also works inside
Jupyter or a running loop); `ainvoke`, `batch` and `abatch` are the async and batch forms.
Build the extractor once and reuse it: it owns a connection pool. `strict=True` raises
`ExtractionValidationError` instead of returning an invalid record.

## Extract and update many records

The scenario from trustcall's README: you keep a list of people, and a chat mentions
several of them, one stranger and a dog. One call should update the right records,
insert the new person and ignore the dog.

```python
from pydantic import BaseModel, Field

from trustcall_system_one import Many, create_extractor


class Person(BaseModel):
    """Someone the speaker knows or is being introduced to."""

    name: str = Field(description="full name")
    relationship: str = Field(description="how the person relates to the speaker")
    notes: list[str] = Field(description="things worth remembering about the person")


CONVERSATION = "..."  # the chat, see examples/05_people_memory.py
people = [  # (record id, schema name, record)
    (
        "0",
        "Person",
        {
            "name": "Emma Thompson",
            "relationship": "College friend",
            "notes": ["Loves hiking", "Works in marketing", "Has a dog named Max"],
        },
    ),
    (
        "1",
        "Person",
        {
            "name": "Michael Chen",
            "relationship": "Coworker",
            "notes": ["Great at problem-solving", "Vegetarian", "Plays guitar"],
        },
    ),
    (
        "2",
        "Person",
        {
            "name": "Sarah Johnson",
            "relationship": "Neighbor",
            "notes": ["Has two kids", "Loves gardening", "Makes amazing cookies"],
        },
    ),
]
extractor = create_extractor([Many(Person, identity="name")], enable_inserts=True)
with extractor:
    result = extractor.invoke(CONVERSATION, existing=people)

for record in result.records:  # in order of first mention
    print(record.json_doc_id, record.kind, record.entity.label, len(record.patches))
print([(s.entity.label, s.reason) for s in result.skipped], result.untouched)
```

`Many` finds every distinct person (discovery, then coreference so "Emma Thompson" and
"Emma" are one), checks that each is a `Person` (the dog is not), matches each to
one existing record or to none (one-to-one), and runs the full single-entity pipeline per
person. `enable_inserts=True` lets the unmatched person become a new record; deletes
(`enable_deletes`) and removal of nested items (`enable_item_removals`) are off unless you
ask. **Scripted output (offline)** of
`uv run python examples/05_people_memory.py --offline` (the answers come from a scripted
transport; the matching, checks and patching are the real code):

```text
Records
-------
  id  kind    entity         changes  rejected
  0   update  Emma Thompson  3 ops    1
  1   update  Michael Chen   2 ops    2
  2   update  Sarah          3 ops    1
  -   insert  Olivia Davis   new      0
  skipped 'Sunny': not_qualified 'Sunny' is not a Person (p=0.04)
  untouched existing records: none
```

Still the scripted run (offline): Michael Chen's applied operations, then the verification
log and the totals of the run (the token counts are estimates):

```text
Each applied operation, next to the text behind it
--------------------------------------------------
  add       /notes/-  "switched jobs and is now working as a Data Scientist at a startup"
  add       /notes/-  "thinking of going vegan"
```

```text
Verification
------------
  crosscheck  second pass over the input alone, without the record
  crosscheck [e2]  second pass over the input alone, without the record
  crosscheck [e3]  second pass over the input alone, without the record
  rejected   /relationship  not_mentioned
             the input does not state it
  rejected   /notes/2  attribution p=0.05
             rock climbing
  rejected   /relationship  not_mentioned
             the input does not state it
  rejected   /name  identity_subset
             'Sarah' is only a short form of the record's 'Sarah Johnson'

28 requests (attribution 4 · coref 1 · count 1 · crosscheck 3 · discover 1 · extract 4 · gate 8 · match 4 · presence 1 · qualify 1)
22,662 input tokens (clairic 5,110 · clef 17,552) · $0.000511 + clef unpriced (estimated offline: no model counted these)
```

In the scripted run, rock climbing was Emma's and sits right before Michael's name; the
attribution check dropped it (`attribution p=0.05`). "Sarah" would have overwritten "Sarah
Johnson" and was refused (`identity_subset`).

**Honest status of the live run.** The same scenario on the real models is still weak
(2026-10-22, commit `05207aa`, 20 requests, 5 s): Michael and Sarah were matched but got no
operations, because every `notes` answer was fragments and was rejected (the open-ended list
problem below); Emma was skipped as `ambiguous_match` (the best record had p=0.51 against an
acceptance bar of 0.6); Sunny was correctly excluded as `not_qualified`; Olivia was found but
her record was invalid (`relationship` refused by the attribution check, `notes` rejected).
The offline output shows what the library does with plausible answers, not how well the
models answer.

## Fields the text does not mention are never touched

A null answer is not an operation. If the text says nothing about a field, nothing is
written, and the field is reported:

```python
for r in result.records[0].rejected:
    print(r.op, r.path, r.reason)
```

In example 03 that is `None /email not_mentioned` offline (live the model echoed the
record's email, so it shows as `echo_span`; either way, no operation), and
`replace /name identity_subset`. Removing a value needs `GateConfig(allow_null_removals=True)`
and a Clef check at 0.85. Removing a whole record (`enable_deletes`) or a nested item
(`enable_item_removals`) is a Clef question too, and it applies the same silence rule: a
matched record or item that the input flags for deletion is removed only when the input
proposes no change to it. If the text both changes and removes it, nothing is done and it is
reported as skipped with `conflict_update_delete`, even if the change gate would have
refused the change. A record the text does not mention at all is returned in
`result.untouched`. An update that applies nothing is still an `update` record with
`patches == ()` and `after == before`.

## Type mapping

The schema compiler decides which model is asked, and how, from the annotation:

| Annotation | Asked as |
| --- | --- |
| `str` | Clairic `text` |
| `int`, `float`, `Decimal` | Clairic `number` |
| `date`, `time`, `datetime` | Clairic date, time, datetime (`reference_date=` anchors relative dates) |
| `timedelta` | Clairic `duration` |
| `list[str \| int \| float \| date]`, `tuple[X, ...]`, `set`, `frozenset` | Clairic `list`; spans stay aligned to the surviving elements |
| `bool` | Clef `noul`, abstains within 0.5 +/- 0.15; `Optional[bool]` is a `choice` of yes, no, unstated |
| `Literal[...]`, `Enum` | Clef `choice` over 2 to 255 options; an `unstated` option is added when the field is nullable or has a default; `s1_field(clef="score")` for ordered scales |
| `list[Literal \| Enum]` | Clairic `list` of text, then one Clef `choice` per element |
| nested `BaseModel` / `TypedDict` | dotted leaf fields; an optional sub-model is gated by a Clef presence question and not asked at all if the text does not state it |
| union of models (`A \| B`) | one Clef `choice` of the member, then only the chosen member is extracted |
| `list[BaseModel]` | a nested collection: each item is an entity of the engine (below) |
| `dict[str, X]` | a collection keyed by the dict key |
| `computed_field` | computed in code, never asked |
| `Any`, `dict[str, Any]` | skipped (the pydantic default applies) |

Per-field options go through `Field(json_schema_extra=s1_field(...))`: `instructions`,
`null_threshold`, `min_confidence`, `identity`, `skip`, `clef`, `criteria`, `unit`,
`decimals`. Clairic is **extractive**: a `str` answer is a verbatim span of the text. If you
want a normalised vocabulary, make the field a `Literal` (a Clef choice), not free text.

## Cardinality

- `One(Schema)`: exactly one entity of that schema. `Many(Schema, identity="name")`: any number.
  Both wrap a schema in `create_extractor([...])`.
- Nested lists use a marker: `pets: Annotated[list[Pet], Many(identity="name")]`.
- The identity may be a dotted path to a scalar inside a sub-model:
  `Many(AircraftSpec, identity="identification.msn")` (also in a compound identity or a nested
  list marker). It is used for matching against existing records and to protect the identity from
  being replaced by a short form of itself. Discovery does **not** ask for the identifier itself
  (a serial number is not what a sheet calls an aircraft: asked for "mentions by their msn" the
  model listed table cells, and a bare `12345` was refused as an aircraft): it asks for the words
  that name or describe each entity (its model, type, name or heading; a table cell that does so
  counts), and the qualify question reads "is X an aircraft, or the msn of one?". If the words find
  too few, a second pass asks by the msn; an entity called by a bare number that is not its msn (a
  weight) is dropped (`bare_number_entity_dropped`); an existing record whose msn is written in the
  input is found without a presence request. Table and emphasis markup around a mention
  (`8547 |`, `**Airbus**`) is stripped. A path that does not exist or does not end at a plain field
  is a `ValueError`.
- With a table of cells (more than `discovery.screen_above` = 12 mentions found), the mentions of
  a nested list that has an identity (`checks: list[Check]` named by `check_type`) or of a
  top-level `Many` with a dotted identity are screened like identity-less items: bare figures, one
  or two letters, markup, headings and the names of the item's own fields are dropped in code,
  and each remaining mention is qualified on its own before they are clustered (warnings
  `junk_mentions_dropped`, `mentions_screened_out`). Never below the count the model gave; 0
  switches it off; a top-level `Many` with a plain identity (a name) is never screened.
- Items with no identity (`engines: list[EngineInfo]`, rows with a position and serial numbers) are
  found by what an item is, the mentions are screened in code and one by one before clustering
  (warnings `junk_mentions_dropped`, `mentions_screened_out`) and the result is cut to the count.
  Name the items with `Many(identity=...)` whenever they have something to be named by.
- A sub-model's class docstring is sent as context: it follows the name of a field that holds the
  model (`billing address (a postal address)`; the name is never replaced) and the instructions of
  the sub-model's own fields ("hours times 60 plus the minutes").
- `Many(identity="identification")` (a sub-model as the identity) is still accepted, with an
  `identity_not_scalar` warning that names the dotted path to use instead.
- Without a wrapper: list-form `existing` with records of a schema implies `Many(Schema)`;
  dict-form `{"Schema": record}` implies `One(Schema)`; otherwise `One(Schema)`. An explicit
  wrapper always wins. `One` given several list-form records raises `ValueError` ("use Many").
- `Many` without existing records is a multi-entity insert; with them, each entity is
  matched to one record or to `__new__` (`enable_inserts` decides whether a new one is
  inserted). `MatchConfig(accept=0.6, on_ambiguous="skip" | "insert" | "raise")` governs
  uncertain matches.
- Several schemas in one extractor (`create_extractor([Invoice, Contact])`) run in
  parallel, each behind a presence check, so a schema the text is not about is not extracted.
  `ainvoke(..., schema=Invoice)` forces one, like trustcall's `tool_choice`.

## Verification and thresholds

Every non-null answer goes through a ladder before it becomes a value or an operation. All
of it is on by default, and every number is a knob.

| Check | Default | Configure |
| --- | --- | --- |
| Null threshold sent to Clairic | 0.7 required, 0.3 optional | `GateConfig(required_null_threshold, optional_null_threshold)`, `s1_field(null_threshold=)` |
| Minimum answer confidence | 0.5 | `GateConfig(min_confidence)` |
| Re-ask a low-confidence answer, alone | below 0.85 | `VerifyConfig(low_confidence, reask_low_confidence)` |
| Borderline `p_null` double-checked by Clef | 0.25 to 0.75 | `VerifyConfig(null_band, borderline_null)` |
| Attribution (is it about this entity?) | keep 0.75, drop 0.4, re-ask between | `VerifyConfig(attribution, attribution_keep, attribution_drop)` |
| Input-only cross-check (updates) | on; disagreements adjudicated at 0.6 | `VerifyConfig(input_only_crosscheck, adjudicate_min)` |
| Clef choice acceptance | probability 0.55, confidence 0.5 | `GateConfig(min_choice_prob, min_choice_confidence)` |
| Change gate, one Clef request per operation | add 0.55, replace 0.6, remove 0.85 | `GateConfig(add_threshold, replace_threshold, remove_threshold)` |
| Presence gate | absent below 0.2 | `PresenceConfig(threshold_absent, threshold_present)` |
| Repair loop | on, at most 5 rounds | `RepairConfig(enabled, max_attempts, ...)` |
| An uncertain attribution after the repair keeps a required text value | auto: on when Jev answers the attribution and the re-ask, off otherwise | `RepairConfig(keep_uncertain=None \| True \| False)` |

```python
from trustcall_system_one import (
    GateConfig,
    RepairConfig,
    VerifyConfig,
    create_extractor,
)

extractor = create_extractor(  # Contact: the model from the first example
    [Contact],
    gates=GateConfig(replace_threshold=0.7),
    verify=VerifyConfig(attribution_keep=0.8),
    repair=RepairConfig(max_attempts=2),
)
```

`keep_uncertain` is `None` ("auto") unless you set it: on only when TypeSafe (Jev) answers the
attribution question AND the repair re-ask (with a `RoutingTransport`, `KindRoutingTransport` or
`FieldTypeRoutingTransport`, the transports those calls are routed to), off on Cloudflare,
Experiential Labs and the Decisions API, and off when Jev reads the text but another provider answers
the attribution. Jev's attribution of a name or an id is uncalibrated and can land between the two bars
on a value the input does state; with it on, a repaired text value of a REQUIRED field that the re-ask
gave again, and that is not nearer to another entity, stands (noted `attribution_uncertain`) instead of
the record being lost; a value the check refutes never does, and an optional field is never filled
this way. `True` or `False` always wins.

**These numbers are starting points, not tuned values.** They have been exercised on two small
live runs of the examples, not measured on a data set; see [Limitations](#limitations).

## Requests, latency and cost

The library asks more than a hand-rolled call, on purpose: accuracy first, one request per
distinct piece of state, run in parallel. Measured on the real models on 2026-10-22
(commit `05207aa`; no rate limits, no retries, no HTTP errors in the run):

| Example | Requests | Input tokens | Wall time |
| --- | --- | --- | --- |
| `01_invoice` (insert, 9 fields) | 4 | 3,699 | 3 s |
| `02_support_ticket` (nested, union) | 10 | 4,184 | 3 s |
| `03_update_contact` (7 fields, 1 record) | 9 | 6,543 | 4 s |
| `04_repair` | 3 | 1,514 | 2 s |
| `05_people_memory` (`Many`, 3 records + 1 new) | 20 | 21,063 | 5 s |
| `06_attribution_guard` (2 people) | 9 | 3,562 | 2 s |
| `Invoice` snippet above | 3 | 1,379 | |

Rule of thumb from these runs: 2 to 5 seconds end to end, 3 to 4 requests for a flat
insert and 9 to 10 for a single-record update or a nested insert, more for each extra
entity (the offline run of example 05 makes 28 requests; live it made 20 because fewer
entities got through). Clairic is $0.10 per million input tokens, which is why the
examples print costs in fractions of a cent. Clef's price is not published, so
`result.cost` sums the priced models and reports the rest as unpriced
(`result.cost.complete` is `False` until `prices=Prices(per_million_input={...})` covers every model).
Every Clef request is estimated before it is sent (state and questions, calibrated on real
table-heavy text) and kept under `Limits(clef_state_token_budget=56_000)`: split by questions, or its
input narrowed to windows around the mentions. A transport can ask for less: on `TypeSafeTransport`
the effective budget is `min(56_000, 24_000)`, the 24k of what Jev reads, in the same unit, so Jev
receives exactly the state the library built and never narrows it again. A request the provider
still refuses for its size (HTTP 413, Cloudflare's "Request requires N tokens", Jev's 400
`max_tokens_exceeded`) or that `TypeSafeTransport` refuses locally before sending anything (a typed
`RequestTooLargeError`: a question as large as Jev's context, a state with nothing to narrow) does not
abort the invoke: that request's answers are missing and `request_rejected` says which (a local
refusal, or the library's own budget refusal, of one oversized question leaves its sibling
questions to be asked). When the refusal arrives
beside parts of the same call that failed or timed out, which may have been charged, it is raised
instead and nothing is sent again. Any other 4xx (a malformed field), auth
errors, rate limits, anything that may have been charged (an uncertain outcome, also behind a
fallback) and a refusal that repeats more than 8 times still raise, and `strict=True` raises on a
size refusal too.
`result.usage`, `result.calls` (one entry per request, with kind, status and duration)
and `result.warnings` tell you what happened; `passes=PassesConfig(split="entity" |
"submodel" | "field")` trades request count for narrower questions.

## The five `existing` shapes

All of trustcall's shapes are accepted, and `result.records[i].json_doc_id` is the id you gave:

```python
from trustcall_system_one import SchemaInstance

# Contact is the model from the first example
rec = {"name": "Dana Whitfield", "email": "dana@fabrikam.example"}

existing = {"Contact": rec}  # 1. {schema name: record}; the id is the schema name
existing = [SchemaInstance("crm-4711", "Contact", rec)]  # 2. list[SchemaInstance]
existing = [("crm-4711", "Contact", rec)]  # 3. (record id, schema name, record)
existing = [
    ("crm-4711", Contact(**rec))
]  # 4. (record id, BaseModel); the schema name is the class name
existing = [Contact(**rec)]  # 5. bare models; ids are minted (uuid4)
```

A record is a dict or a model. `"__any__"` as the schema name applies a record to any schema.
`existing_schema_policy` decides about records of schemas the extractor does not know:
`True` (default) raises `ValueError`, `False` keeps them as context only and lists them in
`result.untouched`, `"ignore"` drops them.

## Testing your code

Tests never need a network or keys. `FakeTransport` takes a handler
`(model, payload, meta) -> response body` and validates both sides against the published
Clairic and Clef schemas, stricter than the real API: a request the library should not send,
or a fake answer that does not fit the request, fails your test. `FakeWorld` answers every kind
of call of a multi-entity run from one declaration of the text:

```python
from pydantic import BaseModel, Field

from trustcall_system_one import Many, create_extractor
from trustcall_system_one.testing import FakeWorld


class Attendee(BaseModel):
    """Someone who will attend a meeting."""

    name: str = Field(description="full name")
    role: str | None = Field(None, description="the person's role")


TEXT = "Maria Alvarez will present. Tom Becker, the finance lead, will take notes."
world = FakeWorld(TEXT)
world.entity(
    "Maria Alvarez",
    schema=Attendee,
    mentions=["Maria Alvarez"],
    fields={"name": "Maria Alvarez"},
)
world.entity(
    "Tom Becker",
    schema=Attendee,
    mentions=["Tom Becker"],
    fields={"name": "Tom Becker", "role": "finance lead"},
)
# a fault: Tom's role is offered for Maria too
world.bleed("Maria Alvarez", "role", "finance lead")

transport = world.transport()
with create_extractor([Many(Attendee, identity="name")], transport=transport) as ex:
    result = ex.invoke(TEXT)

# the attribution check caught it
maria, tom = result.responses
assert maria.role is None
assert tom.role == "finance lead"
```

For a single entity, `FakeTransport(route(presence=..., extract=..., clef_fields=...,
attribution=...))` with the helpers `clairic_answer`, `clairic_response`, `noul`, `choice` and
`clef_response` is all you need; `transport.calls` records every request so a test can also
assert what was **not** asked. See [`examples/08_testing_your_extractor.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/08_testing_your_extractor.py),
which is also a runnable pytest module. This repository's own suite is
`make test` (`pytest --disable-socket`).

## Examples

Each example prints what was extracted, the span behind each value, what the checks did and
what it cost. They run **live** when `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` are set,
and **offline** (a banner says so; answers come from a scripted transport) otherwise or with
`--offline`: `uv run python examples/03_update_contact.py [--offline]`.

| Example | Shows | Live run of 2026-10-22 |
| --- | --- | --- |
| [`00_raw_vs_library.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/00_raw_vs_library.py) | The raw `httpx` call and a naive merge of its values, then the library, on the CRM note | offline only so far; the wire bodies in this README are from the live run of 03 |
| [`01_invoice.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/01_invoice.py) | Invoice email to a typed model: spans, a confident null, re-asks, the due date computed in code | correct |
| [`02_support_ticket.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/02_support_ticket.py) | Nested models, a union, an optional sub-model skipped by the presence gate, an ordered enum | correct |
| [`03_update_contact.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/03_update_contact.py) | A CRM record updated as a JSON Patch; every refusal with its reason | 4 of 5 expected operations; note not appended |
| [`04_repair.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/04_repair.py) | Validation failures fixed by narrow re-asks | correct, but the repair loop was not needed |
| [`05_people_memory.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/05_people_memory.py) | trustcall's relationships scenario with `Many` | weak (see above) |
| [`06_attribution_guard.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/06_attribution_guard.py) | One person's fact landing on another, caught by the attribution check | names correct; Tom's city kept (attributed 0.77 against a keep bar of 0.75) |
| [`07_order_line_items.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/07_order_line_items.py) | Edits inside a nested list of an existing order | not in the live runs so far |
| [`08_testing_your_extractor.py`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/08_testing_your_extractor.py) | Unit-testing your extractor with `FakeTransport` and `FakeWorld` | offline only |

[`examples/README.md`](https://github.com/siddicky/trustcall-system-one/blob/main/examples/README.md) has the details of examples 01 to 08.

## Limitations

Read these before relying on it.

- **Relative edits are not supported.** The update path reads what the text *states* and
  never computes from the old value: "two more of the blue mugs" yields 2, not the old
  quantity plus 2.
- **Open-ended list fields are weak on the live models.** `notes: list[str]` described as
  "things worth remembering" does not work: Clairic returns word fragments at confidence
  below 2e-6 (as low as 6e-23), the library rightly rejects them, and nothing is appended (examples
  03 and 05). Lists of well-defined items (names, ids) work: `contacts: list[str]` above came
  back right at confidence 0.999. A narrower field definition or a Clef yes/no per candidate is the
  likely fix, untested.
- **Multi-entity extraction is not yet reliable live** (example 05 above): matching
  confidence, qualification and the notes problem combine.
- **`list[A | B]`, a list of a union of models, is not extracted** (warning `unsupported_list_item`).
  `dict[int, X]` mappings, `Any` and `dict[str, Any]` are skipped.
- **Thresholds are not tuned.** They are hypotheses with documented defaults. Two live margins are
  thin: a computed total attributed 0.7695 against a keep bar of 0.75, and a bleed
  ("Lisbon" onto Tom) kept at 0.7736.
- **Clairic is extractive.** Text values are verbatim spans (notes read like quotes, and a
  relationship can come back in the speaker's voice, "my cousin"); use `Literal` for a
  vocabulary.
- **Cost and speed.** Verification costs requests: expect 3 to 10 per record and 2 to 5 seconds
  live. Clef's price is unpublished and Clairic/Clef rate limits are undocumented; the
  transport adapts to 429s but you may meet them in large batches.
- **Long inputs.** Clef truncates long state silently and refuses a request over its maximum, so
  the library estimates every request (state and questions, calibrated on recorded table-heavy
  requests: dense tables are 2.4 to 2.9 characters per token, and each option of a choice
  question costs about 18 tokens per question), splits it or narrows `input` to windows around
  the mentions to stay under `clef_state_token_budget`, contains a refusal as a warning, and
  warns past 97% of the limit. On a table-heavy sheet with hundreds of table cells the
  entities of identity-less lists are still the weak part (see DESIGN, Risks 10).
- **Decisions as the primary Clef is much weaker than Clef on Cloudflare** (live, same thresholds:
  13/29 dev and 6/30 holdout against 26/29 and 16/30; precision stays 100%, recall does not).
  Fixed on our side: the notes classification (a hedged `items` 0.65 no longer keeps notes out,
  `ITEMS_VETO`), a refusal answer, the warning name, and the context of an entity without mentions.
  What remains is the model: it hedges on identifiers and numbers (0.4 to 0.65 where Clef says
  0.97: a normalised value like `09:00:00` against the input's `09:00`, serial numbers, e-mail
  addresses), a delete needs a graded 0.85 (a saturated yes is capped at 0.8 by design), and an
  input that both removes a record and changes it was read as a plain change (0.1 for the delete,
  so no `conflict_update_delete`). Compare with the default run before relying on it.
- **Placeholder text is null.** A plain text field whose answer is `TBA`, `TBD`, `N/A`, `n.a.` or
  a double or long dash is null (`GateConfig.placeholders`; `GateConfig(placeholders=())` keeps
  them as values, `s1_field(placeholders=False)` does for one field). `TBC`, a lone `-` and a
  lone `?` are not on the list (ambiguous: a result or a status can read `-`); add them to
  `placeholders` if your data uses them as "no value".
  List elements and Clef-typed fields are not touched. What a schema makes of the *kind* of a value
  (check types that are interval strings) is the schema's business.
- **Duplicates across spellings.** Two names for one entity that no identity value ties together
  (`AIRBUS A320-251 NEO` and `A320-200 NEO` of one aircraft whose MSN is `TBA`) can stay two
  records: the merge check needs every pair across two clusters to pass (live: 0.59 and 0.26,
  the second with junk mentions inside the cluster, which the screening now keeps out). Two
  mentions with the same words are asked about, not assumed to be one (two people called Emma).
- **Alpha.** Public names are in `trustcall_system_one/__init__.py`; anything with a leading
  underscore is internal and may change.

The design, with the full pipeline and every rule above, is in [`docs/DESIGN.md`](https://github.com/siddicky/trustcall-system-one/blob/main/docs/DESIGN.md);
the Clairic and Clef request and response schemas it implements are in [`docs/schemas/`](https://github.com/siddicky/trustcall-system-one/tree/main/docs/schemas).

## Credits

The idea (let the model emit RFC 6902 JSON Patch instead of regenerating records, validate
every result, narrow the repair) and the `existing` conventions come from
[trustcall](https://github.com/hinthornw/trustcall) by William Fu-Hinthorn (LangChain), which is
MIT-licensed. This is a clean reimplementation of its behaviour on different models, not a port of
its code.
Clairic and Clef are Cloudflare's System One models; Jev is TypeSafe's.

This repository is released into the public domain under the Unlicense; see [`LICENSE`](https://github.com/siddicky/trustcall-system-one/blob/main/LICENSE).
