# trustcall-system-one demo: backend

FastAPI backend of the demo UI (`../frontend`). It reads documents, turns pydantic
source into JSON Schemas, and runs
[`trustcall-system-one`](../../README.md) extractions and updates, returning every
detail of the result (patches, rejected operations, per-field answers, usage, the call
log) as JSON.

## Run it locally

```bash
cd demo/backend
uv sync
uv run fastapi dev            # http://127.0.0.1:8000, docs at /docs
```

Without credentials everything works except `POST /api/extract`, which answers 503
`no_provider`. To extract for real, set one of

```bash
export CLOUDFLARE_ACCOUNT_ID=...  CLOUDFLARE_API_TOKEN=...      # Clairic and Clef
export TRUSTCALL_S1_PROVIDER=typesafe  TYPESAFE_API_KEY=...      # TypeSafe's Jev
```

The server must be restarted to see changed variables.

### Develop against the local library

`pyproject.toml` pins the published pre-release, `trustcall-system-one[jsonschema]==0.1.0a2`,
because that is what FastAPI Cloud installs. Its documentation says that `uv.lock` is
used to install dependencies, and says nothing about `[tool.uv.sources]`; a path source
such as `../..` points outside the directory that is uploaded, so it would break the
build. The pin therefore stays and the checkout is used per command:

```bash
uv run --with-editable ../.. fastapi dev
uv run --with-editable ../.. pytest --disable-socket --allow-unix-socket
```

This does not touch `pyproject.toml` or `uv.lock`. The checkout and 0.1.0a2 have the same
library code today; use the flag whenever the library API moves ahead of the release,
and bump the pin (`uv add "trustcall-system-one[jsonschema]==<version>"`) after a release.

## Tests

No test touches the network (a `FakeTransport` answers extractions).

```bash
uv run pytest --disable-socket --allow-unix-socket
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Deploy to FastAPI Cloud

```bash
uv run fastapi login
uv run fastapi deploy        # from demo/backend
```

The entrypoint is `[tool.fastapi] entrypoint = "app.main:app"` in `pyproject.toml`.
`.fastapicloudignore` keeps `tests/` and caches out of the upload. Set these environment
variables in the FastAPI Cloud app (dashboard, or `fastapi cloud env set`; secrets
as secrets):

| Variable | Value |
| --- | --- |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | Cloudflare credentials for Clairic and Clef; **or** |
| `TRUSTCALL_S1_PROVIDER=typesafe`, `TYPESAFE_API_KEY` | TypeSafe's Jev answers both models (a key alone never switches provider) |
| `DEMO_ALLOWED_ORIGINS` | Comma-separated origins of the frontend, for CORS. Default `http://localhost:5173,http://127.0.0.1:5173` |

Whoever can reach the API spends your model credits: put the demo behind something
(rate limits, an access gate) if the URL is public.

## API

All errors are `{"detail": {"code": str, "message": str}}` (`/api/schema/parse` errors
also carry `"line": int | null`). Request bodies that do not fit the models answer
FastAPI's default 422.

### `GET /api/health`

```json
{"ok": true, "provider": "cloudflare", "version": "0.1.0a2"}
```

`provider` is `"cloudflare"` (`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` set),
`"typesafe"` (`TRUSTCALL_S1_PROVIDER=typesafe` and `TYPESAFE_API_KEY` set) or `null`.
`version` is the library's `__version__`.

### `POST /api/document` (multipart field `file`)

Reads `.md`, `.markdown`, `.txt` (UTF-8, undecodable bytes replaced) and `.pdf`
(`pypdf`, pages joined with a blank line). At most 10 MB and 200,000 characters.

```json
{"filename": "note.pdf", "kind": "pdf", "text": "...", "pages": 2, "chars": 1234}
```

`kind` is `markdown`, `pdf` or `text`; `pages` is `null` unless the file is a PDF.
Errors: 415 `unsupported_type`; 413 `too_large`; 422 `pdf_no_text` (a scan with no text
layer); 422 `pdf_parse_error`.

### `POST /api/schema/parse`

```json
{"source": "from pydantic import BaseModel\nclass Contact(BaseModel):\n    name: str\n"}
```

returns `{"schemas": [{"name": "Contact", "json_schema": {...}}]}`: one entry per
top-level `BaseModel` subclass, in source order (`model_json_schema()`; enums and
helper classes are not listed, models they use appear in `$defs`). Errors, all 400:
`unsafe_source`, `syntax_error`, `no_models`, `invalid_model`, each with `message`
and `line`.

**Security.** This endpoint executes code that strangers send. It is guarded, not
sandboxed:

* The source (at most 20,000 characters) is parsed and checked against an `ast`
  allowlist *before* anything runs: imports of named pydantic, typing, datetime, enum,
  decimal, uuid and annotated_types names only; class definitions without decorators
  or keywords; annotated and plain assignments; expressions made of names, attributes
  (no leading underscore), subscripts, constants, tuples, lists, dicts, `|`, signed
  numbers and calls of `Field` and the imported names. Everything else is
  rejected: dunder names and attributes, `lambda`, `def`, comprehensions, decorators,
  `global`/`nonlocal`, `with`/`try`/`while`/`for`, starred expressions, f-strings, other
  operators, any call of a name that was not imported. Imported names that take
  callables or import by string (`import_string`, validators, `TypeAdapter`, ...) are not
  on the list.
* String annotations (`x: "Foo"`) are refused, because pydantic `eval`s them. Define
  a model before the models that use it.
* The code runs with a curated `__builtins__` (`__build_class__`, `__name__`, `str`, `int`,
  `float`, `bool`, `list`, `dict`, `set`, `tuple`, `bytes`) and an `__import__` that knows only the
  allowlisted modules, in a throwaway module that is dropped afterwards.

It is a defence in depth for a demo, not a security boundary for hostile multi-tenant
use: Python in the same process can never be made fully safe. For a public deployment
keep the deployment disposable, give it no secrets other than the model credentials, and
rate-limit it.

### `POST /api/extract`

```json
{
  "text": "Dana moved to Berlin.",
  "schemas": [
    {"name": "Contact", "json_schema": {"type": "object", "properties": {"...": {}}},
     "cardinality": "one"}
  ],
  "existing": [{"id": "crm-4711", "schema": "Contact", "record": {"name": "Dana", "city": "Austin"}}],
  "options": {"enable_inserts": false, "enable_deletes": false, "reference_date": null}
}
```

* `text` is at most 200,000 characters (413 `too_large`).
* `schemas[].name` becomes the JSON Schema `title`, **replacing** any `title`: it is the
  name the library knows the schema by, and what `existing[].schema` refers to.
  `cardinality` is `one` (default) or `many` (`Many(...)`). The JSON Schemas are built by
  the library with `dydantic`, which does not read `"type": ["string", "null"]`; use
  `anyOf` (what pydantic emits).
* `existing[]` become the library's `(id, schema_name, record)` tuples. A record that names
  an unknown schema is a 400.
* `options.reference_date` is an ISO date that relative dates ("last month") resolve against.

The response is `serialize_result(result)` plus `elapsed_ms`, with the real names of the
library's dataclasses:

| Key | Content |
| --- | --- |
| `records[]` | `schema_name`, `json_doc_id`, `kind` (`insert`, `update`, `delete`), `value`, `before`, `after`, `patches[]`, `rejected[]`, `fields`, `entity`, `valid`, `errors`, `delete_p` |
| `records[].patches[]` | `op`, `path` (JSON Pointer), `value` (not for `remove`), `repaired_from` |
| `records[].rejected[]` | `op`, `path`, `reason`, `detail`, `p` |
| `records[].fields` | keyed by dotted path; each has `path` (list), `pretty`, `wire_name`, `engine`, `value`, `raw`, `p_null`, `confidence`, `guess`, `span`, `spans`, `source`, `attribution_p`, `gate_p`, `crosscheck_agree`, `provider`, `reask`, `repaired`, `rejected_reason`, `warnings` |
| `records[].entity` | `key`, `label`, `mentions[]` (`text`, `span`), `first_mention`, `qualify_p`, `match_probs`, `parent` |
| `skipped[]` | `schema_name`, `entity`, `reason`, `detail`, `candidates` |
| `untouched`, `deleted` | lists of existing record ids |
| `attempts` | number of extraction rounds |
| `usage` | token and request totals with `by_model` and `by_kind` |
| `cost` | `usd`, `unpriced_models`, `complete`, `reported_usd` |
| `calls[]` | `model`, `kind`, `schema`, `entity`, `field`, `scope`, `n_items`, `usage`, `duration_ms`, `retries`, `error`, `status_code`, `request_id`, `cost_usd`, `provider`, `fallback_from` |
| `warnings` | list of strings |
| `elapsed_ms` | wall-clock time of the extraction |

`responses` (the validated models) is left out: it repeats `records[].value`.

Errors: 503 `no_provider`; 502 `auth_error`; 429 `rate_limited`; 400 `bad_request` /
`request_too_large`; 422 `validation_failed`; 502 `upstream_error` (transport or model
failure) and `extraction_error` (other library errors); 400 `invalid_schema` (a schema
the library cannot build, duplicate names) and `invalid_request` (the library refused the
`existing` records).

### CORS

Origins come from `DEMO_ALLOWED_ORIGINS`, read at start-up.
