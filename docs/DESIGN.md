# trustcall-system-one — Clairic/Clef-based extraction + JSON-Patch updates

## Context

`trustcall` (reference impl at `/home/user/trustcall`) solves two problems with tool-calling
LLMs: they fail at populating large nested schemas, and they clobber existing data when asked
to update it. Its fix is to make the LLM emit **RFC-6902 JSON Patch** tool calls
(`PatchDoc`, `PatchFunctionErrors`) instead of whole documents, validate every call against
its pydantic schema, and feed validation errors back as a narrowed repair task.

`trustcall-system-one` (this repo — currently empty: README, Unlicense, Python `.gitignore`)
rebuilds that behaviour on Cloudflare's **System One** models, which emit no tokens, no
messages and no tool calls:

- **Clairic** (`@cf/cloudflare/clairic`) — extraction. `state` (text) + flat `fields` map of
  typed specs → one typed answer per field with `value`, `p_null`, `confidence`, `guess`,
  `span(s)`. Text answers are verbatim spans of `state`. $0.10/M input tokens.
- **Clef** (`@cf/cloudflare/clef`, `clef-flash`) — decisions. `state` (string **or structured
  JSON**) + 1–64 `questions` (`noul` → P(yes); `choice` over 2–255 options →
  `choice/probabilities/confidence`; `score` over 2–10 levels).

Both are reached through `POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}`.
Exact request/response JSON schemas were fetched from developers.cloudflare.com and are the
spec for `_wire.py` (section 9).

Outcome: a dependency-light Python package that takes pydantic schemas and extracts one or
many instances from text, or updates existing records by computing JSON Patch ops **in code**
from Clairic answers and gating every op with Clef. A field the input never mentions is never
touched, and every change is visible as a patch op.

## Decisions (settled with the user)

1. **Native library, no LangChain/LangGraph.** pydantic v2 + httpx + jsonpatch.
2. **Clef fills Clairic's type gaps and makes routing decisions.** `bool` → `noul`,
   `Literal`/`Enum` → `choice`; also presence, coreference, record matching, deletes.
3. **Patches are computed in code, then gated per op by Clef** (always on).
4. **Full nested support including dynamic cardinality** (`list[Model]`).
5. **Request count and cost are not a constraint.** Optimise for accuracy, then simplicity:
   one request per distinct `state`, fired in parallel; verification passes on by default;
   caps only where correctness or a real API limit demands one.
6. **Multi-entity extraction is first-class.** Many top-level instances of one schema and a
   nested `list[Model]` share one collection engine.

## How trustcall's patch tool calls map onto System One

| trustcall | here |
| --- | --- |
| LLM emits `PatchDoc{json_doc_id, planned_edits, patches}` | Clairic answers fields with the record in `state`; `_diff.py` builds the ops; Clef gates each op |
| `planned_edits` prose ordering (replace → remove high-index-first → add `/-`) | deterministic sort key in `_diff.py` |
| `PatchFunctionErrors` repair tool + error `ToolMessage` | narrow per-field Clairic re-ask with error text + schema excerpt in `state`; field spec adapted to the error class |
| `PatchFunctionName`, `RemoveDoc` id validation | gone: we choose schemas and ids, the model never names one |
| several tool calls of one schema (`tool_choice="any"`) | collection engine: discover → corefer → extract per entity → attribute → match |
| fan-out of repairs = one attempt | kept |
| `_apply_patch` + `/-` string-concat fix | reimplemented (not copied: MIT → Unlicense) + `_fix_missing_parents` |
| teardown erases `PatchDoc` | inverted: applied ops stay on `ResultRecord.patches` as the audit trail |
| `existing` in 5 shapes, `existing_schema_policy` | kept as-is |

## Pipeline

```
invoke(text, existing)
 └─ per schema S (parallel), cardinality One | Many
     ├─ collect(Scope(S, input, parent=None, existing_S))           [collection engine]
     │    presence (Clef) ║ discovery (Clairic list) ║ count (Clairic number)
     │    → coreference (Clef pairwise noul, average linkage) → entities by first mention
     │    → qualification (Clef noul per entity)
     │    → matching (Clef choice per entity over existing + __new__; Hungarian assignment)
     │    → deletes (Clef noul per existing record, if enable_deletes)
     └─ per entity (parallel): extract_entity
          Clairic per field group (focus header + input + matched record)
          ║ Clairic input-only cross-check (updates) ║ Clef typed fields
          nested list[Model] → collect(Scope(Item, parent=entity, existing=record.items))
          → verification ladder → diff → per-op Clef change gate → jsonpatch apply
          → model_validate → repair loop
```
A `One` schema runs the same path with discovery/coref/count skipped; its single entity is
"the <Schema>".

## Package layout (`trustcall_system_one/`)

| module | owns |
| --- | --- |
| `__init__.py` | public exports only |
| `_errors.py`, `_types.py`, `_config.py` | exceptions; result types, `SchemaInstance`, `Mention`/`Entity`; all config dataclasses |
| `_cardinality.py` | `Many`/`One` wrappers + `Annotated` markers; default resolution |
| `transport.py` (public) | `Transport` protocol, `CloudflareTransport`, adaptive per-model limiter |
| `_wire.py` | exact Clairic/Clef request+response pydantic models |
| `_schemas.py` | coerce BaseModel / TypedDict / JSON Schema / callable → BaseModel |
| `_names.py` | path ↔ wire-name map, deterministic blake2s shortening |
| `_compile.py` | **schema compiler** → `SchemaPlan` |
| `_flatten.py`, `_coerce.py` | flatten/unflatten/prune_nulls; answer → python value |
| `_render.py` | Clairic text state (focus/input/existing blocks, offsets); Clef structured state |
| `_collection.py`, `_coref.py`, `_match.py` | **collection engine**; mention clustering; record matching + assignment |
| `_entity.py`, `_verify.py` | per-entity passes; attribution, cross-check, re-ask, borderline-null |
| `_diff.py`, `_gates.py`, `_repair.py` | ops + apply; thresholds + change gate; repair loop |
| `_run.py`, `_extractor.py` | per-invoke context (usage, call log, semaphores); factory + sync bridge |
| `testing.py` (public) | `FakeTransport`, `FakeWorld`, recording/cassette transports |

### Public API

```python
def create_extractor(
    schemas: Sequence[SchemaLike | Many | One], *,
    transport: Transport | None = None,            # default CloudflareTransport()
    clairic_model: str = CLAIRIC, clef_model: str = CLEF, gate_model: str = CLEF,
    enable_inserts=False, enable_updates=True, enable_deletes=False, enable_item_removals=False,
    existing_schema_policy: bool | Literal["ignore"] = True,
    discovery=None, coref=None, match=None, verify=None, gates=None, repair=None,
    limits=None, concurrency=None, prices=None,
    reference_date: date | str | None = None,
    strict: bool = False,                          # True: raise on terminal validation failure
) -> Extractor

class Extractor:  # frozen; per-invoke state lives in _Run
    async def ainvoke(self, input: str | ExtractionInput, *, existing=None,
                      schema=None,                 # tool_choice analogue: force one schema
                      images=None, reference_date=None) -> ExtractionResult
    def invoke(...)                                # runs ainvoke on a private loop thread
    async def abatch(...); def batch(...); def with_options(**kw); def plan_for(schema)
```

- **Cardinality:** `create_extractor([Many(Person, identity="name"), Event])`; nested via
  `Annotated[list[Pet], Many(identity="name")]`. Resolution: explicit wrapper wins →
  list-form `existing` with records of S implies `Many(S)` → dict-form `{S: record}` implies
  `One(S)` → default `One(S)`. `One` with several list-form records raises `ValueError`.
- **`existing`:** trustcall's five shapes (`{name: rec}`, `list[SchemaInstance]`,
  `(id, name, rec)`, `(id, BaseModel)`, bare `list[BaseModel]` with minted ids), `"__any__"`,
  and `existing_schema_policy` True/False/"ignore" — behavioural spec is
  `_validate_existing` in `/home/user/trustcall/trustcall/_base.py:850-990`.
- **`s1_field(...)`** → `Field(json_schema_extra={"trustcall_system_one": {...}})` with
  `instructions, null_threshold, min_confidence, clairic_type, unit, decimals, integer,
  identity, skip, clef, criteria`.
- **Result:** `ExtractionResult{records, responses, skipped, untouched, deleted, attempts,
  usage, cost_usd, calls, warnings}` + a trustcall-compatible `response_metadata` property.
  `ResultRecord{schema_name, json_doc_id (None for inserts), kind insert|update|delete,
  value, before, after, patches, rejected, fields: {path: FieldAnswer}, entity: EntityInfo,
  valid, errors}`. `EntityInfo{key, label, mentions (text+span), first_mention, qualify_p,
  match_probs, parent}`. `SkippedEntity{schema_name, entity, reason, detail, candidates}` with
  reasons `inserts_disabled | updates_disabled | ambiguous_match | not_qualified |
  conflict_update_delete | validation_failed | empty_after_verification`. Records ordered by
  first mention, deletes last.

## Schema compiler (`_compile.py`)

Pure, `lru_cache`d. `SchemaPlan{specs (one FieldSpec per leaf path), groups, collections,
identity, names, preamble, warnings}`.

| annotation | engine / emitted |
| --- | --- |
| `str` | Clairic `text` (length/pattern constraints in prose, enforced by pydantic) |
| `int` / `float` / `Decimal` | Clairic `number` (+`integer`; `decimals` only when `decimal_places` or `s1_field(decimals=)` is set) |
| `date` / `time` / `datetime` | Clairic same-named type; `reference_date` at request level |
| `timedelta` | Clairic `duration` with `unit` (default seconds) |
| `list[str\|int\|float\|date]`, `tuple[X, ...]`, `set`, `frozenset` | Clairic `list` with `items`; answers parse per element and spans stay aligned with the surviving elements |
| `list[datetime\|time\|timedelta]` | `items:"text"` + per-element parse (flagged lossy) |
| `list[Literal\|Enum]` | `items:"text"` then one Clef `choice` per element |
| `bool` | Clef `noul` with `criteria{true,false}`; abstain inside `0.5 ± 0.15` |
| `Optional[bool]` | Clef `choice{yes,no,unstated}`; `unstated` maps to the `UNSTATED` sentinel (nullable field → `None`) |
| `Literal` / `Enum` | Clef `choice` (2–255 options), `key_map` back to values; option ids are the sanitised value (Enum: the member name) and the criteria text keeps the original value/name and member description; `s1_field(clef="score")` for ordered enums |
| `Literal` / `Enum` / bool-as-choice that is nullable or has a default | the same `choice` plus an `unstated` option (`unstated_2`, … on a collision) mapping to the `UNSTATED` sentinel, so an unmentioned field is not forced to a real value: `coerce` gives `None` for a nullable field and `ok=False, reason="unstated"` for a defaulted one (the default stays). A required non-nullable field gets no `unstated` option |
| single-option `Literal` / `Enum` | required field: engine `const`, filled in code, no request (discriminator tags); nullable or defaulted: skipped with a `single_option` warning |
| nested `BaseModel` / `TypedDict` (also self-referencing) | dotted paths; each sub-model is its own field group; recursion is recorded and bounded by the expander (`limits.max_depth`) |
| `list[BaseModel]`, `dict[str, prim]` | handed to the collection engine |
| `Union[prim, prim]` | Clairic `text` + Clef `choice` over the member types (`<field>.__type`); `coerce_union` parses the text as the chosen member |
| `Union` of models | Clef `choice` over members/discriminator, then the chosen member's fields |
| `dict[str, Any]`, `Any` | skipped (pydantic default), opt-in `limits.any_policy="text"` |
| required / optional | `null_threshold` 0.7 / 0.3 |
| recursion | recorded as a deferred `submodel`/`collection`; the expander stops at `limits.max_depth = 8` |

`ge/le/gt/lt` go into prose (Clairic `min`/`max` are estimate/forecast-only); opt-in
`limits.estimate_for_bounded_numbers`. Instructions = parent context (" → "-joined ancestor
descriptions; mandatory when a name was shortened) + description (`Field.description` →
`typing_extensions.Doc` → title → attribute docstring → humanised name) + type/constraint hints, templated per entity
("{label}'s {desc}"). A lint warns on aggregation-style descriptions ("total of…").

Names: Clairic `^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`, Clef ids `[A-Za-z0-9_.\-]{1,100}`;
over-length or colliding → `head_tail_<blake2s 6 hex>` (every path in a collision group gets the digest form, so names do not depend on declaration order), widened on a digest collision, pinned by a test. Documents are keyed by field name; `FieldSpec.alias` records the key `model_validate` expects and `unflatten(plan=...)`/`prune_nulls` emit it. Per-entity
requests never put list indices in names.

## Collection engine (`_collection.py`) — multi-entity and nested lists

`collect(Scope(plan, source, parent, pointer, existing, cardinality)) -> CollectionOutcome`.
Top-level and nested differ only in: discovery wording ("…belonging to {parent}"), existing
items (records with ids vs list elements with indices), output (one `ResultRecord` per entity
vs ops inside the parent's patch list such as `add /pets/-`), and removal flag
(`enable_deletes` vs `enable_item_removals`).

1. **Identity field:** `Many(identity=...)` (compound allowed) → `s1_field(identity=True)` →
   a `str` field named name/full_name/title/label/id/key/slug/email/handle/sku → first
   required `str` → none (docstring-driven discovery, whole-record matching, warning).
2. **Discovery, in parallel:** Clef presence `noul` (< 0.2 → zero entities; 0.2–0.5 →
   proceed only if mentions found); Clairic `list` of every identity mention incl. short
   forms, no pronouns (stoplist + determiner stripping in code); Clairic integer count.
3. **Coreference (`_coref.py`):** unit is the mention, not the surface form. One Clef `noul`
   per mention pair over one structured state `{input, mentions:{m0:{text,span,context}}}`,
   64 per request; average-linkage clustering at 0.5 resolves inconsistent triangles;
   deterministic tie-break by first mention. Above 150 mentions switch to incremental
   assignment (Clef `choice` over clusters + `__new__`). Canonical label = most tokens →
   most frequent → earliest; the label is only framing, the stored identity value is
   extracted like any field.
4. **Qualification:** Clef `noul` per entity "is `entity.label` a {S}?" — filters "Sunny"
   (a dog) out of `Person`.
5. **Count reconciliation:** count > k → one broadened re-discovery, else warning; count < k →
   cluster-level merge checks; above 10 a mismatch only warns (counting is a Clairic weakness).
6. **Per-entity extraction:** one Clairic request per entity per field group, with a focus
   header naming the entity and its mention spans, the input, and only that entity's matched
   record. Chosen over packing `person.0.x, person.1.x` because disambiguation lives in the
   state not an index, server-side pass splitting would separate an entity's fields, records
   stay isolated (no cross-record echo), and failures stay per-entity.
7. **Attribution (default on):** Clef `noul` per non-null value and per list element: "Does
   `input` state that {desc} of `entity.label` is '{value}'?" ≥ 0.6 keep, < 0.3 drop
   (`attribution`), between → isolated re-ask then one more check. On updates an op must pass
   **both** attribution (right entity) and the change gate (new vs. record).
8. **Matching (`_match.py`):** Clef `choice` per entity with criteria = each existing record
   as JSON plus `__new__`. > 254 records → prefilter by identity similarity (warning);
   over the Clef state budget → compact rendering (warning). One-to-one **Hungarian
   assignment** over `-log p` with a `__new__` column per entity; collisions where both
   claims ≥ 0.5 raise warning `match_collision` (opt-in `match.collision="merge"`). Apply
   only if p ≥ 0.6 and confidence ≥ 0.5, else `SkippedEntity(ambiguous_match)`.
9. **Flags per entity:** matched → update (if `enable_updates`); `__new__` → insert (if
   `enable_inserts` or no existing records of S were given); deletes via Clef `noul` per
   existing record ≥ 0.85; update+delete on one record → `conflict_update_delete`.
   Unmentioned records → `untouched`, no ops.

## Verification ladder (`_verify.py`) and update path (`_diff.py`, `_gates.py`)

| check | default | rule |
| --- | --- | --- |
| Answer-level thresholds | on | reject if `p_null > null_threshold`, `confidence < 0.5`, span inside `<existing>`, Clef choice < 0.55 |
| Attribution | on | above |
| Input-only cross-check (updates) | on | second Clairic pass without `<existing>`; null there → reject as echo; disagreement → Clef `choice{a, b, neither}` ≥ 0.6. Covers number/date answers that carry no span |
| Low-confidence re-ask | on (< 0.85) | isolated single-field request on mention windows; disagreement → Clef adjudication |
| Borderline-null | on (`p_null` 0.25–0.75) | Clef `noul` "does the input state it?" → re-ask or null |
| List near-duplicate dedup | on | difflib 0.6–0.9 vs record elements → Clef `noul` "same fact?" |
| **Change gate** | **always** | one Clef request per op, structured state `{input, entity, record, proposed_change}`; add ≥ 0.55, replace ≥ 0.6, remove/set-null ≥ 0.85 |
| State-variant self-consistency | off (`k=1`) | opt-in |
| `guess` as a value | never | metadata only |

Diff walks the field plan, never the document: absent → `add` (parents first); differs
(normalised) → `replace`; null answer → **no op** (silence is not deletion; opt-in
`gates.allow_null_removals`); scalar lists use set semantics → `add /notes/-` for new
elements only. Order: replace → remove (highest index first) → add. Apply with
`jsonpatch.apply_patch` on a deep copy; on conflict try `_fix_string_concat` then
`_fix_missing_parents`, recording `repaired_from`. Clairic requests always send
`options.return_state_text=true` so span offsets are re-anchored on sentinel markers;
`truncated: true` → retry on mention windows, then warn/raise `StateTruncatedError`.

## Repair loop (`_repair.py`)

`model_validate` → map each error `loc` to the longest known path → **one narrow Clairic
re-ask per failing field**, in parallel, with `<previous_attempt>`, `<validation_errors>` and
a `$defs`-resolved schema excerpt in `state`, and the spec adapted to the error (parsing →
`number`/`date`; `enum`/`literal_error` → Clef `choice`; `bool_parsing` → Clef `noul`;
collection length → re-run discovery with the limit stated). Re-asked values go through the
full verification ladder. One round = one attempt; `repair.max_attempts = 5` (termination
only); stop early when the error signature repeats. Terminal: drop still-failing ops and
re-validate; then `strict` raises `ExtractionValidationError`, otherwise the record is
`valid=False` / `SkippedEntity(validation_failed)` and kept out of `responses`.

## Transport and wire (`transport.py`, `_wire.py`)

- `Transport` protocol: `async run(model, payload, *, meta: CallMeta, timeout) ->
  TransportResponse`. `CallMeta{kind, schema, entity, field, scope}` routes fakes and becomes
  AI Gateway `cf-aig-metadata` (optional via `send_metadata`; `entity` can be a person's
  name). `TransportResponse{body, retries, duration_ms, status_code}` is a frozen dataclass:
  `body` is the bare response object and the rest feeds `CallRecord` (`retries`,
  `duration_ms`, `status_code`). A failed call raises a `TransportError` that carries
  `status_code`, `retries` and `retry_after`; the engine records those plus the message in
  `CallRecord.error`.
- `CloudflareTransport`: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` (fallback
  `CLOUDFLARE_AUTH_TOKEN`), optional `base_url` / `gateway=`; httpx async client; timeout
  60 s (connect 10 s); retries on 408/429/5xx/transport errors honouring `Retry-After`, else
  decorrelated jitter; unwraps the REST `{success, result, errors, messages}` envelope and
  accepts bare objects; errors `AuthError`, `BadRequestError` (with CF `errors[]`),
  `RateLimitError`, `ServerError`, `ModelResponseError`.
- Concurrency: 64 in flight per transport per event loop (shared across invokes and batch),
  32 per model, adaptive (halve on 429, at most once per generation so a burst of 429s halves
  once, +1 per 20 successes). The slot is released while a retry backs off. Limiter state and
  the httpx client are kept per event loop. Sync `invoke` runs `ainvoke` on a private
  event-loop thread.
- Wire models mirror the published schemas exactly (requests `extra="forbid"`, responses
  `extra="allow"`). Requests forbid unknown keys deliberately, which is stricter than the
  published schemas, to catch our own typos; they also enforce the documented image and body
  size limits before sending. Clairic answer requires `type`, `value`; response requires `answers`,
  `truncated`, `usage`. Clef answers: `{type:"noul", noul}`,
  `{type:"choice", choice, probabilities, confidence}`,
  `{type:"score", score, legend, probabilities, confidence}`; Clef body `model` is
  `clef`/`clef-flash` while the URL uses `@cf/cloudflare/...`.
- Clef truncates long state **silently**: estimate size before sending (budget 56K of 64K
  tokens), narrow `input` to mention windows when over, and warn when
  `usage.input_tokens ≥ 0.97 × 65,536`.
- Usage per call into `CallRecord`s and a `Usage` total by model and by kind. `Prices`
  defaults to Clairic $0.10/M only; Clef's price isn't published, so `Prices.cost(usage)`
  returns a frozen `CostReport{usd, unpriced_models, complete}`: `usd` sums the priced models
  only, and `complete` is false (with the models listed) until the caller configures a price
  for every model that used input tokens.
- Experimental, off: Workers AI queue mode (`queueRequest` + `request_id`) behind a live
  probe — the Clairic schema has no batch properties, so it may be unsupported.

## Testing

- `testing.FakeTransport(handler)` routed on `CallMeta`, and **`FakeWorld`**: declare text,
  entities (mentions, field values, matching record), non-entities, injected bleed and
  low-confidence faults; it answers every request kind consistently and computes spans with
  `str.find` over the rendered state.
- Unit tests (`pytest --disable-socket --allow-unix-socket`): `test_wire`, `test_compile` (one
  case per type-table row), `test_names` (regex property, pinned hash), `test_flatten`,
  `test_coerce`, `test_diff` (ordering, `/-` repair, list set semantics, echo spans),
  `test_gates` (one gate request per op), `test_verify`, `test_repair`, `test_existing`
  (5 shapes × 3 policies, ported by behaviour from trustcall's `test_strict_existing.py`),
  `test_cardinality`, `test_transport` (`httpx.MockTransport`: retries, `Retry-After`,
  limiter, envelope).
- `test_collection.py` on `FakeWorld`: (1) the trustcall README Emma/Michael/Sarah/Olivia
  scenario — three updates with ids 0/1/2 and appended notes, Olivia inserted, existing notes
  untouched, Sunny `not_qualified`, `enable_inserts=False` variant skips Olivia;
  (2) coreference merge incl. inconsistent triangle and two distinct Emmas; (3) bleed
  rejection; (4) two entities claiming one record; (5) zero entities makes no extraction
  calls; (6) 300 existing records → prefilter; (7) count reconciliation both directions;
  (8) nested `Person.pets` through the same engine; (9) ambiguous match; (10) update/delete
  conflict.
- `tests/live/test_smoke.py` (`@pytest.mark.live`, needs `TRUSTCALL_S1_LIVE` + creds) runs the
  README scenario and records cassettes for `tests/recorded/`; `tests/evals/` (opt-in) tunes
  thresholds and `passes.split`.

## Request and latency profile (information, not a constraint)

README scenario (Many(Person), 3 existing, 4 people + Sunny, ~9 mentions): ~39 requests in
~8 sequential hops; Clef median 209 ms (published), Clairic latency unpublished (assume
0.3–0.8 s) → roughly 2–3.5 s end to end. Flat `One` insert: ~7 requests, ~1–2 s.

## Delivery

### Execution model: Sonnet and Haiku subagents only

All execution — file edits, `uv`/`make` commands, tests, commits, pushes, PR creation — is done
by subagents launched with `Agent(subagent_type="general-purpose", model=...)`. The main
session only orchestrates and reviews (read-only). The model is chosen per task: **Haiku**
for mechanical work with no design judgement, **Sonnet** wherever the code carries the
library's logic.

| task | model | why |
| --- | --- | --- |
| M0 skeleton: pyproject, Makefile, package dirs, `_errors`, `_types` stubs, `docs/DESIGN.md` copy, `AGENTS.md` | Haiku | boilerplate fully specified by this plan |
| M1 transport + exact wire models + `FakeTransport` | Sonnet | retry/limiter/envelope logic, strict schema fidelity |
| M2 compiler, names, flatten, coerce | Sonnet | the type mapping is the core of the library |
| M3 `One` insert pipeline + verification ladder + sync bridge | Sonnet | orchestration and thresholds |
| M4 nested sub-models / field groups | Sonnet | compiler and unflatten interplay |
| M5 update path: diff, change gate, cross-check, apply | Sonnet | correctness-critical patch generation |
| M6 repair loop | Sonnet | error-to-field mapping and spec adaptation |
| M7 collection engine discovery + coreference + `FakeWorld` | Sonnet | clustering and multi-entity logic |
| M8 matching, assignment, deletes, flags | Sonnet | Hungarian assignment and the README scenario |
| M9 nested collections through the engine | Sonnet | scope handling |
| M10a cassette/recording transports, live smoke + evals harness | Sonnet | test infrastructure with behaviour to get right |
| M10b README, CI workflow, Makefile polish | Haiku | prose and config from this plan's outline |
| Push branch + open draft PR | Haiku | mechanical |
| CI fixes on the PR | Haiku for lint/format/lockfile/config failures; Sonnet for failing tests or type errors in logic | match the fix to its difficulty |

1. **One subagent per milestone, run sequentially** (each milestone builds on the previous
   one's code). M0's subagent copies this plan into the repo as `docs/DESIGN.md`, so every later
   brief points at that file plus the milestone's section rather than restating it.
2. **Each brief states:** the milestone's scope and exit criteria from the table below; the
   modules and tests to write; the reference code in `/home/user/trustcall` to reimplement by
   behaviour, never copy; the exact Clairic/Clef schemas in `_wire.py`; and that the subagent
   must run `make lint` and `make test`, include their real output in its report, and commit
   only when both pass, with the session's commit attribution lines.
3. **Review between milestones (main session, read-only):** read the subagent's diff
   (`git diff`/`git show` via read-only commands) against this plan, look for silently
   narrowed scope, copied MIT code, and tests that assert nothing. Check the reported lint and
   test output. If something is wrong, send the findings back to the same subagent with
   `SendMessage` for a fix before starting the next milestone.
4. **Push and PR:** after M10, a Haiku subagent pushes the branch and opens the draft PR;
   CI fixes go to Haiku or Sonnet per the table above. If a Haiku subagent's work fails
   review or its fix doesn't hold, the task is re-run on Sonnet.

### Branch, packaging, tooling

- Branch `feat/system-one-extractor` off `main` in `/home/user/trustcall-system-one`; one
  commit per milestone, each leaving `make lint test` green; push and open a draft PR.
- `pyproject.toml`: hatchling; deps `pydantic>=2.7,<3`, `httpx>=0.27,<1`,
  `jsonpatch>=1.33,<2`, `jsonpointer>=2.4`, `typing-extensions>=4.9`; optional
  `jsonschema = ["dydantic"]`; dev group pytest, pytest-asyncio, pytest-socket, pytest-xdist,
  pytest-cov, hypothesis, ruff, mypy (strict, pydantic plugin); ruff google docstrings.
  `uv` only, committed `uv.lock`. `Makefile`: install/test/cov/smoke/record/evals/lint/
  format/build. CI: lint + test on 3.10–3.13; live jobs manual only. Add `AGENTS.md`.

| # | milestone |
| --- | --- |
| M0 | skeleton: pyproject, Makefile, package, errors, types |
| M1 | transport + exact wire models + `FakeTransport` |
| M2 | compiler (flat types), names, flatten, coerce |
| M3 | `One` insert end to end incl. Clef typed fields + verification ladder; sync bridge |
| M4 | nested sub-models / field groups |
| M5 | update path: existing coercion, diff, change gate, cross-check, apply |
| M6 | repair loop |
| M7 | collection engine discovery side + `Many` + `FakeWorld` (multi-entity insert) |
| M8 | matching side: assignment, prefilter, deletes, flags → README scenario green |
| M9 | nested `list[Model]` / `dict[str, prim]` through the engine |
| M10a | cassette/recording transports, live smoke, evals harness |
| M10b | README, CI workflow, Makefile polish |

README: pitch (patch, don't regenerate; verify, don't generate), install/env, `One` insert,
the multi-entity scenario printing ids/kinds/patches/spans/skipped/untouched, "unmentioned
fields never clobbered", type table, cardinality rules, verification knobs, request profile,
`existing` shapes, testing with `FakeWorld`, limitations vs trustcall, credits.

## Risks

1. **Clairic is extractive**: text values are verbatim spans, so notes read "got promoted to
   Senior Marketing Manager last month" rather than a paraphrase, and Olivia's relationship
   comes back as "my cousin" (speaker's voice). Normalised vocabularies should use `Literal`
   (→ Clef `choice`). README shows real output.
2. Clef silently truncates long state — pre-send budget + narrowing + post-hoc warning.
3. Clairic/Clef rate limits are unpublished; ~40 requests per invoke × `batch` may hit 429s —
   adaptive limiter.
4. Span offsets could drift if Clairic normalises state — always request `state_text`.
5. Thresholds and "narrower is more accurate" are hypotheses — every threshold is a knob and
   the evals harness measures them.
6. Clef option-id pattern isn't explicitly published — reuse the question-id pattern; the
   live smoke test checks it.
7. trustcall is MIT, this repo is Unlicense — reimplement by behaviour and credit.

## Verification

1. `make lint` (ruff + mypy strict) and `make test` (socket-disabled) green after every
   milestone, run by the milestone's subagent and checked in its report, with the main
   session reviewing the diff before the next milestone starts.
2. The README Emma/Michael/Sarah/Olivia scenario passes on `FakeWorld`, including bleed,
   coreference, collision and zero-entity cases.
3. If `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN` are available in the session, run
   `make smoke` against the live API and record cassettes; if not, report that live
   verification was not possible and leave `make smoke` as the documented check.
4. A Haiku subagent pushes and opens the draft PR; the main session watches CI and sends
   fixes to Haiku or Sonnet subagents (per the execution table) until it is green.
