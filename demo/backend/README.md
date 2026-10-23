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

No test touches the network (a `FakeTransport` answers extractions). The sandbox tests
start Python subprocesses, which `--disable-socket` allows.

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
`.fastapicloudignore` keeps `tests/`, `.venv/` and the caches (`__pycache__/`, `*.py[cod]`, ruff, mypy, pytest) out of the upload. Set these environment
variables in the FastAPI Cloud app (dashboard, or `fastapi cloud env set`; secrets
as secrets):

| Variable | Value |
| --- | --- |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | Cloudflare credentials for Clairic and Clef; **or** |
| `TRUSTCALL_S1_PROVIDER=typesafe`, `TYPESAFE_API_KEY` | TypeSafe's Jev answers both models (a key alone never switches provider) |
| `DEMO_ALLOWED_ORIGINS` | Comma-separated origins of the frontend, for CORS. Default `http://localhost:5173,http://127.0.0.1:5173` |
| `DEMO_TOKEN_SECRET` | Shared with the Vercel frontend (secret). Every route but `GET /api/health` then needs `Authorization: Bearer <token>`, minted by the frontend's `/api/auth/token` for signed-in users |

Whoever can reach the API spends your model credits, so deployments set
`DEMO_TOKEN_SECRET`. Without it the API is open, which suits local development, and
it logs a warning at start-up. A token is `base64url(json).hex(hmac-sha256)` with
payload `{"email", "aud": "trustcall-demo-api", "exp"}` and lives 10 minutes; see
`app/auth.py`. A missing, forged, expired or wrong-audience token answers 401
`unauthorized`.

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
(`pypdf`, pages joined with a blank line). At most 10 MB, 300 PDF pages and 200,000
characters.

```json
{"filename": "note.pdf", "kind": "pdf", "text": "...", "pages": 2, "chars": 1234}
```

`kind` is `markdown`, `pdf` or `text`; `pages` is `null` unless the file is a PDF.
Errors: 415 `unsupported_type`; 413 `too_large` (over 10 MB, over 300 pages, or over
200,000 characters of text: the message says which); 422 `pdf_no_text` (a scan with no
text layer); 422 `pdf_parse_error` (damaged or password protected, or the PDF took more
than 20 s or too much memory to read); 503 `sandbox_unavailable` (the worker process
could not be started).

The endpoint is a plain `def`, so it runs in FastAPI's thread pool and a slow PDF does
not stall other requests. PDFs are parsed in a worker process, never in the server
(see "Security" below). The body size is checked before the upload is parsed (see
"Request size limits").

### `POST /api/schema/parse`

```json
{"source": "from pydantic import BaseModel\nclass Contact(BaseModel):\n    name: str\n"}
```

returns `{"schemas": [{"name": "Contact", "json_schema": {...}}]}`: one entry per
top-level `BaseModel` subclass, in source order (`model_json_schema()`; enums and
helper classes are not listed, models they use appear in `$defs`).

Errors, all 400 except the last, each with `message` and `line`:

| `code` | Meaning |
| --- | --- |
| `unsafe_source` | The source uses something outside the allowlist (see "Security"), or is over 20,000 characters. Nothing was run. |
| `syntax_error` | The source does not parse. |
| `no_models` | The source defines no `BaseModel` subclass. |
| `invalid_model` | The source ran and failed (pydantic or Python error), or the worker crashed or ran out of memory. |
| `timeout` | Building the models took more than 5 s of wall-clock time or 3 s of CPU time and the worker was killed. |
| 503 `sandbox_unavailable` | The worker process could not be started. |

#### What the source may contain

Imports of the listed names from `pydantic`, `typing` (and `typing_extensions`),
`datetime`, `enum`, `decimal`, `uuid` and `annotated_types`, as `from m import n`
or `import m` (also `import m as alias`); class definitions with plain or
`module.Name` bases; annotated fields; `Field(...)` and the other imported callables;
enum members; and top-level constants. `date` and `datetime.date` both work, as does
`pydantic.Field`. Attribute access is allowed only as `module.member` (a listed
member of an imported module) and `EnumClass.MEMBER` (an enum class defined earlier in
the source). Not allowed: chains such as `uuid.os.system`, a rebinding of any
imported name, allowlisted name or builtin (`Field = ...`, `class str`), top-level
assignments of anything but constants and simple literals (so no type aliases:
write the type in the field), `Field(default_factory=...)` with anything but `list`,
`dict` or `set`, validators, decorators, `def`, `lambda`, comprehensions, loops,
`try`, `with`, f-strings, string annotations (forward references: define a model before
the models that use it).

A field may be named like an allowlisted type (`date: str | None = None`) unless
the source imported that name; then the field is a rebinding and is refused.

#### Security

This endpoint executes code that strangers send, so the server process never does.
Three layers, each of which must hold on its own:

1. **Syntax tree allowlist** (`app/_schema_guard.py`). The source is parsed and every
   node is checked before anything runs: an explicit list of qualified names
   (`MODULE_MEMBERS`), no attribute chains, attribute access only on imported module
   names and earlier enum classes, no name bound twice, no rebinding of imports,
   allowlisted names or builtins at the top level or in a class body, top-level
   assignments only of constants. This rejects the known escapes (`uuid.os`,
   `datetime.sys`, `Field = uuid.os.system`, ...) with `unsafe_source` without
   starting a process. The allowlist is a filter on attacker-controlled text; it is
   not trusted to be complete, which is why layers 2 and 3 exist.
2. **A separate, locked-down process** (`app/_schema_worker.py`, started by
   `app/_sandbox.py`). The exec and `model_json_schema()` run in
   `python -I` (isolated mode), with an environment of `PATH` and a few variables
   Python needs plus `PYTHONHASHSEED=0` (so no `CLOUDFLARE_*`, no `TYPESAFE_*`, no
   other secret), in a throwaway working directory, with a 5 s wall-clock timeout
   enforced by the server (the worker is killed) and `resource` limits the worker sets
   on itself before it reads the source: 3 s of CPU time, 512 MB of address space,
   file size 0, 64 open files, no core dumps, and `RLIMIT_NPROC` 0 (no new
   processes; the kernel does not enforce this one for root, and it counts all
   processes of the user). The worker checks the allowlist again before it
   executes anything and answers with JSON on stdout. A timeout answers
   `timeout`, a crash `invalid_model`. Tests prove this layer alone, with the allowlist
   switched off in the worker: no `CLOUDFLARE_API_TOKEN` in `os.environ`, an endless
   loop is killed, memory and file writes are limited.
3. **Curated builtins** in the worker: the source runs with `__builtins__` of
   `__build_class__`, `str`, `int`, `float`, `bool`, `list`, `dict`, `set`, `tuple`,
   `bytes` and an `__import__` that knows only the allowlisted modules. String
   annotations are refused, because pydantic `eval`s them.

What this is not: a container. The worker runs as the same user as the server, can read
files that user can read (an allowlist bypass could reach them) and shares the
network namespace. Python code in a Python process cannot be made fully safe, so for a
public deployment keep the deployment disposable, give it no secrets other than the
model credentials (which the worker does not see), and rate-limit it. Each request
costs one Python start-up with pydantic (about 0.15 s).

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

### Request size limits

`app/limits.py` is a small ASGI middleware that judges a request before its body is
parsed or buffered: a `Content-Length` above the limit is a 413 `too_large` at once,
and bodies without one (chunked) are counted as they arrive and stop at the limit.
The limits are 10 MB plus 64 KB of multipart framing for `/api/document`, 1 MB for
`/api/schema/parse` and 2 MB for `/api/extract` (200,000 characters of text, which
JSON can escape to six bytes each, plus schemas and existing records). An invalid
`Content-Length` is a 400 `bad_request`. The 413 carries the CORS headers. The
file-size and character checks of `/api/document` and `/api/extract` stay as the exact
limits.

### CORS

Origins come from `DEMO_ALLOWED_ORIGINS`, read at start-up.
