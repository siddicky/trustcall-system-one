# Examples

Short, runnable stories on realistic text. Each prints what the library extracted,
which span of the text supports each value, what the verification checks did, and
what it cost in requests and tokens.

```bash
uv run python examples/01_invoice.py            # live if credentials are set, else offline
uv run python examples/01_invoice.py --offline  # force the scripted transport
```

## Live and offline

* **Live** runs on the real models. Set both `CLOUDFLARE_ACCOUNT_ID` and
  `CLOUDFLARE_API_TOKEN` (a token with Workers AI access) and do not pass `--offline`.
  Confidences, spans and verification events then come from Clairic and Clef, and will
  differ from the offline output.
* **TypeSafe** (Jev for both models) is live too: set `TRUSTCALL_S1_PROVIDER=typesafe` and
  `TYPESAFE_API_KEY` (a key alone changes nothing, and no Cloudflare variable is needed).
* **Offline** is used when either variable is missing or `--offline` is passed. A
  banner says `OFFLINE DEMO — answers come from a scripted transport, not the models`.
  The answers are scripted to be what the models would plausibly return, with spans
  computed against the text the library really sends and at least one answer that the
  checks reject; the library code that checks, patches and validates is the real thing.
  Token counts and cost are estimates and are labelled as such. Offline output shows
  the behaviour, never the models' accuracy.

Every example is a module with `main(offline: bool)` returning its result, so
`tests/unit/test_examples.py` runs them all offline. An example that needs a milestone
that is not built yet is `xfail(strict=True)` there, and flips when it lands (all of
them work now).

## The examples

| example | shows | runs | needs |
| --- | --- | --- | --- |
| [`00_raw_vs_library.py`](00_raw_vs_library.py) | The CRM update of example 03 twice: the raw Clairic call with `httpx` and a naive merge of its values (the name shrinks to "Dana", the notes are replaced), then `create_extractor(...).invoke(...)` on the same note. Mirrors the "Without the library / With the library" section of the main README. | live or offline (offline answers the raw call through `httpx.MockTransport`) | M5 (works now) |
| [`01_invoice.py`](01_invoice.py) | An invoice email into a typed `Invoice`: spans, a confident null that a `guess` never fills, a low-confidence answer re-asked, a disagreeing re-ask settled by Clef adjudication, a value refused by the attribution check, the due date computed in code. | live or offline | M3 (works now) |
| [`02_support_ticket.py`](02_support_ticket.py) | Nested sub-models, a discriminated union of models, an optional sub-model skipped by the presence gate (the request log shows nothing asked about it), an ordered enum scored by Clef. | live or offline | M4 (works now) |
| [`03_update_contact.py`](03_update_contact.py) | Updating a CRM record as a JSON Patch: four `replace` ops and an `add /notes/-`, each printed next to the text it came from; every other field rejected with its reason (`identity_subset` for the short name "Dana", `not_mentioned` for the email). | live or offline | M5 |
| [`04_repair.py`](04_repair.py) | Validation failures (`"Berlin 10115"` for a five-digit postcode, a dropped birth year for an age) fixed by narrow re-asks that carry the error. Repair fixes extraction mistakes; vocabularies belong in a `Literal`. | offline (live only repairs if the models slip) | M6 |
| [`05_people_memory.py`](05_people_memory.py) | trustcall's relationships scenario: three records updated, one inserted, a dog skipped, the mention clusters, notes as verbatim quotes. | live or offline | M7-M8 (works now) |
| [`06_attribution_guard.py`](06_attribution_guard.py) | One person's fact bleeding onto another, caught by the attribution check. | live or offline (bleed injected offline) | M7 (works now) |
| [`07_order_line_items.py`](07_order_line_items.py) | Edits inside a nested list of an existing order: `replace /items/1/quantity` and `add /items/-`, other lines untouched. | live or offline | M9 (works now) |
| [`08_testing_your_extractor.py`](08_testing_your_extractor.py) | A miniature pytest module: unit-testing your own extractor with `FakeTransport` and `route`, no network. Also runs under `pytest`. | offline only | M1-M3, `FakeWorld` since M7 (works now) |

`_common.py` (mode selection, printing) and `_offline.py` (the scripted models) are
shared helpers, not examples.
