# trustcall-system-one — Clairic/Clef-based extraction + JSON-Patch updates

## Context

`trustcall` (reference impl at https://github.com/hinthornw/trustcall) solves two problems with tool-calling
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

**Groups (single entity, insert).** The root group's Clairic request, the Clef questions of its
Clef-typed fields and the choices of its unions go out together; its sub-models run concurrently.
`passes.split`: `submodel` (default) = one Clairic request per group, `entity` = one request for
every group of the unit (the root and its sub-models; each chosen union member is a unit of its
own), `field` = one per field.
* *Optional sub-model:* one Clef `noul` per optional group (call kind `presence`, "Does `input`
  state the Book's author's address?"); the optional siblings of one group share a request
  (`meta.field` lists their dotted paths, comma-joined), asked while the parent's own request is
  in flight. A missing answer counts as a borderline probability, never as present. `p < presence.threshold_absent` → the group is not asked at all (`None` if
  nullable, else the default; `prune_nulls` applies a default to a `None`); between the two
  thresholds → asked, and if every leaf then comes back null (after the verification ladder) the
  sub-model is treated as absent; at or above `threshold_present` it stands. Only verified non-null
  Clairic values count as evidence: a `bool` or a required `Literal` always resolves and says
  nothing about the group. Required sub-models
  are always asked, and a group below an absent one is never reached.
* *Union of models:* the choice (call kind `clef_fields`) is answered with the parent group's
  Clef questions; then, as soon as it is in (the parent's Clairic pass may still be running),
  only the chosen member's subtree is extracted, and a tag that the choice
  fixes is filled in from it (a single-valued tag is a `const`; a multi-valued one is not
  asked). A choice below `gates.min_choice_prob`/`min_choice_confidence`: an optional union is
  treated as unstated; a required one extracts every member and keeps the one with the most
  verified non-null Clairic values (ties to the earlier member; none → unset), with a warning.
* *Validation is bottom-up.* An optional sub-model and the chosen union member are validated as
  their own model (`member.model.model_validate`, aliases applied) and the *instance* goes into
  the parent document, so pydantic's smart-union cannot hand the value to another member whose
  keys overlap; required sub-models stay documents validated with the root (`TypedDict`s stay
  dicts). An optional sub-model that fails validation is treated as absent (`None`/default) with
  a warning and `RejectedOp(path=<group>, reason="invalid_optional_group")`, and the leaf
  answers it held are reported as dropped; a required union member that fails makes the record
  invalid with its errors under the union's path. What is wrong inside a dropped group goes
  with it: its errors do not invalidate the record. An optional sub-model that can never
  validate (a required field of it was skipped by `max_depth`/`max_recursion`) is skipped at
  compile time (one `unviable_group` warning, no presence question, no extraction).
  Known limits: (1) a parent's `model_validator(mode="before")` now receives model *instances*
  for optional sub-models and union members, not dicts; (2) a union of `TypedDict` members is
  validated as plain dicts by the parent, so pydantic's smart-union can still pick another
  member whose keys overlap; (3) `CallMeta.field` of a batched request (sibling presence
  questions, chunk adjudication/attribution of several fields) holds comma-joined paths or
  names.
* *Evidence.* A Clairic value counts when verified and non-null. A Clef answer counts only when
  it resolved ok, was not `unstated` and is not what the field would have anyway: a field with
  a default whose answer differs from it, or a nullable field that was answered. A required
  Clef field (`bool`, `Literal`) and a filled-in const never count. The same rule decides the
  borderline-absent check and the required-union fallback.
* *Errors* raised in any concurrent step propagate as themselves: `_run.gather` re-raises the
  first real exception (never a sibling's `CancelledError`), and a flow never cancels a future
  that another flow awaits.
* All leaves of all groups share one verification ladder run, so the attribution check is one
  request over every leaf. Ladder questions carry `FieldSpec.scoped` (`city (author → address)`).
  Assembly builds the nested document level by level (`unflatten(..., plan=, members=)` renames
  each level's keys to their aliases, `prune_nulls` prunes per level) and validates the root;
  `ResultRecord.fields` is keyed by full dotted path, and holds an extra entry for each presence
  question and union choice with the container's own engine (`submodel` / `union_models`; the
  question's answer is in `raw`, `rejected_reason` is `absent`, `invalid_optional_group` or
  `low_confidence_fallback`), so a consumer that walks leaf answers skips them by engine.

## Package layout (`trustcall_system_one/`)

| module | owns |
| --- | --- |
| `__init__.py` | public exports only |
| `_errors.py`, `_types.py`, `_config.py` | exceptions; result types, `SchemaInstance`, `Mention`/`Entity`; all config dataclasses |
| `_cardinality.py` | `Many`/`One` wrappers + `Annotated` markers; default resolution |
| `transport.py` (public) | `Transport` protocol, `CloudflareTransport`, `ExperientialTransport`, `DecisionsTransport`, `TypeSafeTransport`, `FallbackTransport`, `RoutingTransport`, `system_one_transport`, adaptive per-model limiter |
| `_typesafe.py`, `_candidates.py`, `_structure.py`, `_dateparts.py` | Jev as a stand-in for Clef (translation) and Clairic (selection): orchestration and verification; pattern candidates; lines, stitching, blocks and sentences; date, time and duration parts |
| `_decisions.py` | Clef body <-> Decisions API request/response translation (the Clef fallback) |
| `_wire.py` | exact Clairic/Clef request+response pydantic models |
| `_schemas.py` | coerce BaseModel / TypedDict / JSON Schema / callable → BaseModel |
| `_names.py` | path ↔ wire-name map, deterministic blake2s shortening |
| `_compile.py` | **schema compiler** → `SchemaPlan` |
| `_flatten.py`, `_coerce.py` | flatten/unflatten/prune_nulls; answer → python value |
| `_render.py` | Clairic text state (focus/input/existing blocks, offsets); Clef structured state |
| `_collection.py`, `_coref.py`, `_match.py` | **collection engine**; mention clustering; record matching + assignment |
| `_existing.py` | `existing` coercion (five shapes, policy, `__any__`), key normalisation, target selection |
| `_entity.py`, `_verify.py` | per-entity passes (insert, update); attribution, cross-check, re-ask, borderline-null |
| `_nested.py` | nested collections: the guard (`max_depth`/`max_recursion`), an insert's items as the parent's document and answers, an update's item records as operations of the parent |
| `_sentences.py` | sentence splitter, entity scoping and the Clef pass of the `sentences` list strategy |
| `openai_transport.py` (public), `_escalate.py`, `_relative.py` | the opt-in escalation tier: `OpenAIResponsesTransport` (`gpt-6-luna`, Responses API); its four kinds and their verification; relative dates read in code |
| `_diff.py`, `_gates.py`, `_repair.py` | ops + apply; thresholds + change gate; repair loop |
| `_run.py`, `_extractor.py` | per-invoke context (usage, call log, semaphores); factory + sync bridge |
| `testing.py` (public) | `FakeTransport`, `FakeWorld`, `RecordingTransport`, `CassetteTransport` (+ `CassetteMissError`, `request_key`, `trace_to_cassette`) |
| `evals/` (repo top level, not packaged) | eval dataset, scorer, runner and sweeps (`python -m evals`), smoke scenarios and invariants |

### Public API

```python
def create_extractor(
    schemas: Sequence[SchemaLike | Many | One], *,
    transport: Transport | None = None,            # default: system_one_transport() (env-driven)
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
  `_validate_existing` in trustcall's `trustcall/_base.py` (https://github.com/hinthornw/trustcall).
- **Matching:** `match=MatchConfig(accept, min_confidence, render_fields, collision, on_ambiguous,
  delete_threshold)` configures how a `Many` schema's entities are matched to its existing records
  (see "Collection engine", steps 8-9); `on_ambiguous="raise"` raises `AmbiguousMatchError`
  (`schema_name`, `label`, `candidates`).
- **`s1_field(...)`** → `Field(json_schema_extra={"trustcall_system_one": {...}})` with
  `instructions, null_threshold, min_confidence, clairic_type, unit, decimals, integer,
  identity, skip, clef, criteria, list_strategy` (`"list"`, `"sentences"` or `"auto"` for a
  `list[str]` of free text; unset follows `Limits.list_strategy`, default `"auto"`; see "Open-ended
  lists").
- **Result:** `ExtractionResult{records, responses, skipped, untouched, deleted, attempts (the
  extraction round plus the repair rounds),
  usage, cost_usd, calls, warnings}` + a trustcall-compatible `response_metadata` property.
  `ResultRecord{schema_name, json_doc_id (None for inserts), kind insert|update|delete,
  value, before, after, patches, rejected, fields: {path: FieldAnswer}, entity: EntityInfo,
  valid, errors}`. `EntityInfo{key, label, mentions (text+span), first_mention, qualify_p,
  match_probs, parent}`. `SkippedEntity{schema_name, entity, reason, detail, candidates}` with
  reasons `inserts_disabled | updates_disabled | ambiguous_match | not_qualified |
  conflict_update_delete | validation_failed | empty_after_verification`. Records ordered by
  first mention, deletes last.
  A run whose records are all invalid (so `responses` is empty) adds a `no_valid_record` warning
  per invalid record that names it and its first errors.

## Schema compiler (`_compile.py`)

Pure, `lru_cache`d. `SchemaPlan{specs, groups, unions, member_groups, member_specs, identity,
clairic_names, clef_names, warnings}`. `specs` holds one `FieldSpec` per field, depth first in
declaration order: a sub-model or union field (the *container*, engine `submodel` /
`union_models`) precedes the leaves of its group. `groups` is the root `FieldGroup` plus one per
nested sub-model; `FieldGroup{path, model, optional, nullable, has_default, description, alias,
spec (the container; None for the root), specs (leaves directly in the group), children, unions,
identity}`. A union field is a `UnionGroup{path, spec, members, discriminator, option_tags,
optional, ...}`; each `UnionMember{label, model, group, options}` is a compiled subtree whose fields
live in `member_specs`/`member_groups`, not in `specs`, because they are only asked once the choice
resolves. Members may share field names, so wire names below a union are allocated under
`FieldSpec.wire_path` (the member label follows the union field: `pet.Cat.name`); `spec.path` stays
the document path; `plan.spec(path, member=...)` and `plan.spec_by_wire_path(...)` look a member's field
up, and a bare `spec(path)` that several members share raises instead of guessing.

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
| `Literal` / `Enum` / bool-as-choice that is nullable or has a default | the same `choice` plus an `unstated` option (`unstated_2`, … on a collision) mapping to the `UNSTATED` sentinel, so an unmentioned field is not forced to a real value: `coerce` gives `ok=True, value=None, reason="unstated"` for both (like a Clairic null; `prune_nulls` applies a default). A required non-nullable field gets no `unstated` option |
| single-option `Literal` / `Enum` | required field: engine `const`, filled in code, no request (discriminator tags); nullable or defaulted: skipped with a `single_option` warning |
| nested `BaseModel` / `TypedDict` (also self-referencing) | dotted leaf paths (`author.address.city`); each sub-model is its own `FieldGroup` and, when optional (nullable or defaulted), carries a Clef `noul` presence question (`<path>.__presence`). `limits.max_depth` is the longest leaf path and `limits.max_recursion` (default 2) how many times a model may reappear on its own ancestor chain: past either the field is `skip` with one `max_depth`/`max_recursion` warning per (model, field), so a self-referencing `Optional[Node]` unrolls to the limits and never further. `limits.max_groups` (default 128, union members included) turns a runaway schema into a compile-time `ValueError` naming the model, never a silent truncation. A union member is left out when it would reappear too often, or when a required field of it was skipped by a limit (it could never validate); a union with no member left is skipped, and one with a single option left asks nothing |
| `list[BaseModel]` | `engine="collection"`: the items are entities of the item model, found and extracted by the collection engine scoped to the parent entity (see "Nested collections"). Item identity: `Annotated[list[Pet], Many(identity="name", label=...)]`, else the usual rules. A list of a union of models is not extracted (warning `unsupported_list_item`) |
| `dict[str, X]` (`X` a primitive, a list of primitives, a `Literal`, a model, ...) | `engine="mapping"`: a collection keyed by the dict key; each entry is an entity of a synthetic model with a `key` text field (the identity: the key is extracted like a name) and a `value` field (the value type; a model is a sub-model `value.*`). The entry model is `<Parent><Field>Entry` (`StudentScoresEntry`), or the value model's own name for a model. `dict[int, X]` and a value the compiler cannot ask for are skipped with `unsupported_mapping_key` / that type's tag |
| `Union[prim, prim]` | Clairic `text` + Clef `choice` over the member types (`<field>.__type`); `coerce_union` parses the text as the chosen member |
| `Union` of models (also `Optional[A \| B]`) | one Clef `choice` (`UnionGroup.spec`): options are the discriminator's literal values (`Field(discriminator=...)`, else a required `Literal`/`Enum` field every member declares with values no other member shares), else the member names; an option's criteria is the member docstring, else its field list; an optional union adds an `unstated` option (the 255-option cap counts options, not models). Each member is compiled as a subtree (const tags included) and asked only after the choice resolves |
| `dict[str, Any]`, `Any` | skipped (pydantic default), opt-in `limits.any_policy="text"` |
| required / optional | `null_threshold` 0.7 / 0.3 |
| recursion | sub-models and unions expand down to `limits.max_depth = 8`; collections (`list[Model]`) and `dict[str, X]` are recorded (`plan.collection_specs()`, at any depth) and their item plan is compiled on first use (`plan.collection(spec)`, memoised `compile_schema` of the item model or the synthetic entry model with the same gates and limits), never eagerly, so a model that holds a list of itself terminates. `limits.max_depth` and `limits.max_recursion` apply to the whole document: the engine carries the depth and the chain of item models (`Nest`) and skips a collection past either with one `max_depth` / `max_recursion` warning |

`ge/le/gt/lt` go into prose (Clairic `min`/`max` are estimate/forecast-only); opt-in
`limits.estimate_for_bounded_numbers`. Instructions = parent context (" → "-joined model title and ancestor field names, with the member after a union field: `Book → author → address`, `Owner → pet (Dog)`; always present for a nested field, and mandatory when a name was shortened) + description, templated through the owner chain (`the Book's author's address's city`; a leading article of the description is dropped so the template does not double it) (`Field.description` →
`typing_extensions.Doc` → title → humanised name, which carries the class docstring of the model the field holds in parentheses (its first sentence, for a sub-model or the items of a `list[Model]`: `billing address (a postal address)`; pydantic's own default docstring is never used, nor one that only repeats the names, and a parametrised generic uses its generic's docstring). The docstring is added to the field's own name, never put in its place, so two fields of one model (a billing and a shipping address, six weights) keep distinct descriptions and presence questions) + type/constraint hints, templated per entity
("{label}'s {desc}").
**Class docstrings reach the leaves.** The parent context of a field of a nested model also carries the first sentences (at most 160 characters) of the docstring of the model that declares it, after the last ancestor: `Aircraft → airframe → total airframe hours (A duration as printed on a sheet. The total in minutes is hours times 60 plus the minutes): total minutes`. So what `HhMm` says about converting hours:minutes to `total_minutes` is part of the Clairic instruction and of every Clef question about that leaf (`FieldSpec.context` / `clef_context`, which the engine reuses when it rebuilds a question). Not added below a union member (the member's tag already carries its docstring), nor when the docstring only repeats the model or field name ("An address."), nor beyond `limits.max_instruction_chars` (the description is never the part that is cut). A lint warns on aggregation-style descriptions ("total of…").

Names: Clairic `^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`, Clef ids `[A-Za-z0-9_.\-]{1,100}`;
over-length or colliding → `head_tail_<blake2s 6 hex>` (every path in a collision group gets the digest form, so names do not depend on declaration order), widened on a digest collision, pinned by a test. Documents are keyed by field name; `FieldSpec.alias` records the key `model_validate` expects and `unflatten(plan=...)`/`prune_nulls` emit it. Per-entity
requests never put list indices in names.

## Collection engine (`_collection.py`) — multi-entity and nested lists

`collect(scope, run) -> CollectionOutcome{entities, skipped, warnings, presence, count}`
with `Scope{plan, source, cardinality (the Many), parent, pointer, existing}`; at the top level
`parent is None` and `pointer == ""`. `insert_collection(run, scope)` runs `collect` and then
step 6. Top-level and nested differ only in: discovery wording ("…belonging to {parent}"),
existing items (records with ids vs list elements with indices, or mapping entries with their
key), output (one `ResultRecord` per entity vs ops inside the parent's patch list such as
`add /pets/-`), and removal flag (`enable_deletes` vs `enable_item_removals`). See "Nested
collections" below. `collect` builds the discovery side (steps 1-5);
`insert_collection` inserts every entity (step 6) and `apply_collection` is what a `Many(S)` WITH
existing records of S runs (list-form `existing` of a schema without a wrapper implies `Many`, so
it does too): `collect`, then steps 8 and 9 (matching, updates, inserts, deletes). An `"__any__"`
record takes part in the one-to-one claim like for a `One` (see "Update path"): a `Many` that
claims one has it among its existing records. Settings: `DiscoveryConfig(presence_absent=0.2, presence_present=0.5,
count_trust_max=10 (1 to 30), broaden_on_count_mismatch=True)` and `CorefConfig(strategy="auto",
merge_threshold=0.5, max_pairwise_mentions=60, context_chars=120,
all_pairs_below=24)`, passed to
`create_extractor(discovery=, coref=)`. `Many.label` is what prompts call an entity (default the
schema title: "Person").

1. **Identity field:** `Many(identity=...)` (compound allowed; a name the schema lacks is a
   `ValueError` at `create_extractor`; a name may be a **dotted path through sub-models to a scalar
   leaf**, `Many(AircraftSpec, identity="identification.msn")`, also in a compound identity and in a
   nested-list marker `Annotated[list[AircraftSpec], Many(identity="identification.msn")]`. A dotted path
   that does not exist, passes through something that is not a sub-model, or ends at a sub-model, a
   list of models or a mapping is a `ValueError` naming the step that fails. A plain top-level name is
   accepted whatever it holds, as before (`Many(identity="identification")`, a sub-model, was the
   live workaround and still works, if poorly: matching ignores a dict value), with an
   `identity_not_scalar` warning that suggests the dotted path to its first text field. The same resolved
   path is used everywhere an identity is: the discovery question (`by their msn (manufacturer
   serial number)`: the last segment and the leaf's description), the existing records' identity
   values for matching and the prefilter (`record_value` reads the aliased keys level by level:
   `identification.serial`), the compact record shown to Clef (the sub-model keeps only its
   identity leaves), the identity-conflict guard, and the identity-subset protection of the diff: a
   named identity is never replaced by a token subset of itself (`identity_subset`), whoever
   named it (`Ctx.identity`). `s1_field(identity=True)` stays per level: it marks the identity of the
   model that declares the field, not of the model above it) → `SchemaPlan.identity` (`s1_field(identity=True)`, else
   a `str` field named name/full_name/title/label/id/key/slug/email/handle/sku, else the first
   required `str`) → none: discovery then follows the schema docstring ("Every mention of a
   Reading (A temperature reading…)") and the run warns `no_identity_field`. The identity reads
   in the discovery question as `by their name (full name)`, compound fields joined with "and".
   **A dotted identity is not asked for by value**: a serial number nested in a sub-model is not
   what a sheet calls the entity (live, `identification.msn` on a two-aircraft sheet: discovery
   by msn listed the table cells `8547 |` and `12345 |`, and "is `12345 |` an aircraft?" got 0.14,
   so the B737 was lost; the A320's own msn, asked "of `8547 |`", scored 0.485 and was dropped).
   With a dotted identity discovery asks for "every mention of an aircraft: the exact words the
   input uses to name or describe it (its model, type, name or heading), not the msn or another
   number or code on its own. A table cell is a mention when it names or describes the aircraft: its
   model or type, such as the cell next to a label like 'aircraft', 'model' or 'type'" (a sheet
   that names its aircraft only in a table, `| Aircraft | Airbus A320-214 |`, would lose them under
   "do not list table cells"), the entity is labelled by those words (`Airbus A320-200`), and
   qualification reads "is `entities.e0.label` an aircraft, or the msn of an aircraft?" (true:
   "...or the msn that identifies one"). A plain identity (a name) is asked as before. A mention
   loses the table and emphasis markup at its edges (`clean_mention`: `|`, `*`, `_`, `~`, `\`,
   en and em dashes), so `8547 |` is `8547`. **The words can fail, so there are three nets.** (1) The
   broader second pass (`count` above the entities found) asks by the identity, `by their msn`: the
   cells that hold one, which qualification accepts in context ("or the msn of an aircraft"). (2) An
   insert entity called by a bare number (an MTOW `78000`, an hours figure, a year) is dropped when
   its extracted identity is not that number (`not_qualified`, warning `bare_number_entity_dropped`,
   which names the kind and never a value), however qualification judged it; and `_rank` does not
   offer a figure that does not match the identity field's regex `pattern` (read from the field's
   hints) to qualification when it has one. (3) An existing record whose identifier (four
   characters or more, one a digit) is written verbatim in the input needs no `record_presence`
   request: a qualified entity with a mention in the same line or sentence covers it, else the group
   qualification refused there is brought back (from `REVIVE_MIN`), else the identifier's own words
   become the entity's mention (`_adopt_by_identifier`); matching still decides which record each
   entity is. Before, discovery by identifier gave that for free and the words made a request per
   record.
   **Screening a table's worth of mentions** (`discovery.screen_above`, default 12; 0 switches it
   off): the code screen (`_prefilter`) and the one-by-one qualification before coreference
   (`_group(screen=True)`) of identity-less items also run, when discovery returned more than
   that many mentions, for a NESTED list that has an identity (live: 1,663 of the 1,705 refused
   entities of a 591-line sheet were `MaintenanceCheck`, named by `check_type`; 513 coreference
   requests) and for a top-level `Many` with a dotted identity (38 refused aircraft: `Qu`, `A`,
   `77`, `CF`, `ATA`, `CN21.61`, `\*`; their junk mentions also sat inside the real aircraft's
   clusters and spans). It never takes the candidates below Clairic's count, which is a count of
   ENTITIES: for a collection with an identity the survivors are counted by distinct (normalised)
   mention, not one per mention (three mentions of one pet are one entity), and without a count at
   least one entity is expected, so a screen that leaves nothing gives back the best of what it
   dropped (up to ten per expected entity). A two-letter name (`Jo`, `Al`) is a name, not a code, in
   a NESTED list that has a plain identity. A top-level collection with a plain identity is never
   screened; the cap-to-count of identity-less items stays theirs alone.
2. **Discovery, three requests in parallel** (a forced `schema=` skips the presence question
   only):
   * presence, Clef `noul` (kind `presence`, id `presence`): "Does `input` mention any {Person}
     ({description})?"; `p < presence_absent` → zero entities and no further call (the other two
     requests are already in flight), unless the question had to be cut to the Clef budget (it then
     reads only the start of the input) and discovery found mentions: the answer is ignored with a
     `presence_unreliable` warning; `presence_absent ≤ p < presence_present` → proceed only if
     discovery found mentions;
   * discovery, Clairic `list` of `text` (kind `discover`, field `mentions`; state
     `<focus schema="Person" task="discover">…</focus><input>…</input>`): "Every mention of a
     Person by their {identity}, in the order they appear in the input. Include repeated
     mentions and shortened ones (a first name alone). Do not list pronouns. Each item is the
     exact words the input uses." A truncated answer goes through the usual chunking
     (`ask_clairic`). In code each item is located in the ORIGINAL input (the span mapped out of
     the rendered state, else a search after the previous mention; an item that is not in the
     input is dropped with a `mention_not_in_input` warning; an item without a span takes the
     next unused occurrence of its text, so repeated mentions survive), pronoun-only items are
     dropped (English closed-class stoplist `_collection.PRONOUNS`; an all-caps word such as "US"
     or "IT" is an acronym and stays, as does a name made of function words, "The Who"), a trailing
     possessive (`'s`) is stripped with the span, and so is a leading lower-case article or
     demonstrative (the/a/an/this/that/these/those), or a capitalised one when what follows is a
     common phrase ("The cousin"): before a capitalised name it is part of it ("The Hague", "An
     Nguyen", "This Is Us"); a descriptive "my cousin" stays. Of two overlapping mentions the
     longer wins when it contains exactly one of them, or when it is ONE name that several of them
     quote: at most one of the parts is itself a multi-token name, and only whitespace, hyphens
     and initials lie between them ("Emma Thompson" over "Emma" and "Thompson"), or capitalised
     words too when every part is a single token ("Mary Ann Smith" over "Mary" and "Smith").
     Separators (`, ; : & /`, a line break, "and"), lower-case words and possessives never join:
     "Attendees: Emma Thompson Michael Chen", "Emma, Michael", "Emma & Michael" and "Emma's
     brother Tom" stay separate mentions (up to `all_pairs_below` mentions coreference still
     joins a name that was wrongly kept apart, as every pair is asked). Result: `Mention(text, span)` in input
     order;
   * count, Clairic `number` with `integer: true` (kind `count`, field `count`, same state): "How
     many distinct {Person}s does the input mention?". A truncated or null answer means "no
     count" (`count_unreliable` warning when truncated).
3. **Coreference (`_coref.py`):** unit is the mention, not the surface form. Strategy
   `pairwise`: one Clef `noul` per mention pair (kind `coref`, id `pair.m{i}.m{j}`, i < j,
   "Do `mentions.m{i}` and `mentions.m{j}` refer to the same {Person}?", criteria true/false). Up to
   `all_pairs_below` (24) mentions EVERY pair is asked (at most 276 pairs, 5 requests). Above that
   only plausible pairs are: the mentions share a name token, one token starts another ("Ann",
   "Anne"), two tokens are forms of one given name (a small built-in table: Bob/Robert,
   Liz/Elizabeth, Bill/William, Peggy/Margaret, Tom/Thomas …), one is the initials of the other
   ("JS", "J. Smith"), a single name stands within ~200 characters (or the same sentence) of a
   longer one, or either side is a description with no name ("my cousin"). A pair that is not asked
   is "not asked", never a "no": it stays out of the pair cache (a debug log counts them), the
   clustering treats it as 0, and a count-driven merge can still check it (step 5). At most 64
   questions per request, the requests concurrent, each with its own structured state
   `{"input", "mentions": {"m0": {"text", "span", "context"}, …}}` that holds only the mentions its
   questions name (context = ±`context_chars` of the input; the input is narrowed to windows
   around the mentions when over the Clef budget). Average-linkage agglomerative clustering at `merge_threshold` (inclusive):
   the two clusters with the highest average pairwise probability merge first, ties go to the
   lowest first-mention index, so A~B 0.9, B~C 0.9, A~C 0.05 gives {A, B}, {C}. Strategy
   `incremental`: the mentions after the first are walked in order in batches of 16; a batch is ONE
   request of Clef `choice` questions (id `assign.m{k}`, state `{"input", "mentions": {"m{k}": …},
   "clusters": {"c{n}": {"label", "mentions": [up to 3 with spans]}, …}}`: each earlier entity is
   described once, in `clusters`, and the options are only `c{n}: <label>` plus `__new__`), over the
   clusters as they were when the batch started, all questions with the same options; a mention joins
   a cluster only if its option has probability >= `merge_threshold`, else it starts one, and the
   mentions of one batch that started clusters are then compared with each other pairwise (the same
   rules for which pairs are asked; only these intra-batch pairs reach the pair cache, mentions of
   different batches are never compared, so a count-driven merge check may ask about them). More
   than 254 clusters show the 254 most recently
   MENTIONED (`coref_options_capped` warning). `strategy="auto"` is pairwise up to
   `max_pairwise_mentions` (60) mentions and incremental above; `"pairwise"` and `"incremental"`
   force it. Canonical label: a name beats a description (`name_part`: a capitalised start; a
   mention led by a possessive is a description, and "her cousin Olivia Davis" is called "Olivia
   Davis"), then the most tokens, the most frequent normalised form, the earliest; the label is only
   framing, the stored identity value is extracted like any field. Pair probabilities are cached, so
   the broadened re-discovery asks only about pairs that involve a new mention.
4. **Qualification:** one Clef `noul` per entity (kind `qualify`, id `qualify.e{n}`), all entities in
   ONE request per schema up to 64 questions (beyond that one request per 64 entities, each state
   holding only its own entities) over the state `{"input", "entities": {"e0": {"label",
   "mentions": [{"text", "span", "context"}]}, …}}` (a state whose input was narrowed carries
   `{"text", "context"}` without spans; `context` only for the first 8 and last 2 mentions); the call carries no entity key, since the numbering changes
   with a broadened discovery or a merge: "Is `entities.e{n}.label` a {kind}?" where the KIND is what the schema is, not its
   title: `Many(kind=...)`, else the leading noun phrase of the schema description (the `label` is how the
   entity is displayed and never the kind): cut at the first blank line, parenthesis, abbreviation-safe
   sentence end, preposition, relative word, participle or determiner, at most four words ("A person in an
   address book" is a "person", "One product line on an order" a "product line", "Someone ..." a "person").
   A META noun (information, details, data, record, schema, model, entry, profile) is not what the schema
   is: "Details of a person" and "Schema for a customer" give "person" and "customer", "Customer record"
   gives "customer"; a possessive owner is dropped ("The user's pet" is a "pet"), a plural made singular
   ("Customers"); "An entry in the address book", "Something we keep" or no description give nothing, and
   the kind is then the title, with the warning `kind_fallback` (name it with `Many(kind=...)`). "a" and
   "an" follow how the noun is said ("a user", "an LLM"). The title fallback is the class name as
   words ("PersonModel" is "person", "HTTPRequest" "http request", "Item_v2" "item"), and leading
   intensifiers are dropped ("Very important customer record" is a "customer"). Live, "Is Emma a Member?" scored 0.24, Maria 0.189, Jon 0.226 and the
   presence question 0.21 (live run 5), against 0.97 for "is Emma a person?", so a terse title made bare
   names unqualifiable; the presence question ("Does `input` mention any {kind}?") names the kind as well.
   The criteria refuse only a clear non-match ("The entity is clearly something else, not a {kind}.").
   `< discovery.qualify_min` (0.4) → `SkippedEntity(reason="not_qualified")` ("Sunny", a dog, 0.02, for
   `Person`); a missing answer qualifies. Qualification runs BEFORE the count comparison, so "k" is the number of entities
   that qualify.
   **Record-driven presence (updates, `discovery.record_presence`, default 0.5).** Discovery and
   qualification do not see the existing records. For every existing record that no qualified entity
   already fits (`identity_compatible` with one of its identity values) and whose identity has a word of
   three letters or more in the input, one Clef `noul` of its own (kind `record_presence`, id `record.{k}`,
   `CallMeta.field` the record id) asks "Does `input` refer to the same {Person} as `record`?" over
   `{input, record (compacted), parent?}`. Only a name word counts as "in the input": three letters or
   more, not a stopword, a proper name (capitalised in the record) in its own case, so "pay the bill" does
   not ask for "Bill Gates". More than `record_presence_max` = 20 such records skip the check for the
   collection, with the warning `record_presence_capped`. A yes (≥ the knob) brings the entity back: the
   group discovery found and qualification refused, only if qualification put it at 0.3 or more (a dog
   "Sunny" at 0.02 stays refused for the record "Sunny Patel"), keeping its own `qualify_p`; else a new
   group whose mention is where the input names the record (the whole identity, else its first name word,
   in its case; none found, none made), unless that overlaps a mention already found. A record an entity already fits is never asked about, so no entity is made twice;
   matching still decides which record each entity is, and a first name that Clef calls a NEW person is not
   forced onto a record (the match's top option decides, see step 8). 0 switches it off.
5. **Count reconciliation** (count `n` against `k` qualifying entities; acted on only when
   `broaden_on_count_mismatch` and `max(n, k) ≤ count_trust_max`, else the mismatch only warns;
   `count_trust_max` is validated to at most 30, so a merge asks at most 30·29/2 = 435 pair checks):
   * `n > k`: unless the DISTINCT names of the QUALIFIED entities already cover `n` (case-folded, a short form
     is its full name: "Sarah" ~ "Sarah Johnson", so "Sarah Johnson called. Sarah and my cousin went hiking"
     has one name and a count of 2 and the cousin is still missing; a refused non-person such as a dog is
     no name at all; "Sarah Johnson" and "Sarah Miller", the model counting two names of one person that
     coreference joined, are two: nothing was missed, no warning; when in doubt, broaden), ONE broadened re-discovery (the same question plus "Also include descriptive
     references such as 'my cousin' when no name is given."), whose mentions are only ADDED to the
     first list where they overlap none of its mentions (a broadened "Emma's brother Tom" never
     takes the place of, or joins, "Emma" and "Tom"), re-clustered (cached pairs) and re-qualified
     for the new clusters; if `n > k` still, warning `count_exceeds_discovered` (an entity is never
     invented);
   * `n < k`: cluster-level merge checks for every pair of qualifying clusters whose best
     ASKED pairwise probability is ≥ 0.2 or that were never asked about (an unasked pair is not a
     "no"; needs the pairwise or incremental strategy): one Clef `noul` each (kind
     `coref`, id `merge`, state `{"input", "a": {"label", "mentions"}, "b": {…}}`, "Do `a` and
     `b` refer to the same {Person}?"), run together; pairs at ≥ `merge_threshold` are approved, and
     the highest probability first two sets of clusters are joined, until `k == n`, only if EVERY
     pair of clusters across the two sets has passed a check (A~B and B~C never join A and C by
     themselves; a pair not yet checked is asked for before the join); if `n < k` still, warning
     `count_below_discovered`. A count of 0 while entities qualify is not acted on (never merge
     toward 0), it only warns.
   Entities are then ordered by first mention and keyed `e0, e1, …` over ALL entities, skipped
   ones included, so a skipped entity keeps its number (Sunny is `e1`, Michael `e2`).
6. **Per-entity extraction:** every qualifying entity runs the single-entity insert pipeline
   (`insert_one`: groups, unions, Clef typed fields, verification ladder incl. attribution,
   repair loop) concurrently, without asking presence again. The Clairic focus header names the
   schema, the entity and its mention spans:
   `<focus schema="Person" entity="Emma Thompson">` / "Answer each field for the entity named
   above only. Ignore facts about any other person." / `The entity is mentioned in the input at
   characters 11-24, 131-135.` The header names the entity ONCE and never quotes a mention: the
   first live run showed Clairic starting a text span at a quoted name in the header and ending
   it in the input, so the value held header text and the input tag. As a second line of defence
   a text or list answer whose span touches the header or the blocks around the input of the
   state Clairic says it read is rejected (`span_outside_input`; per element for a list) and
   never kept. A span that merely cannot be mapped back (`state_text` agrees with what was sent
   only up to some point: CRLF turned into LF, so everything after the first difference has no
   anchor) says nothing against the value: if the value is found in the original input it
   stays and its span is relocated there (the occurrence nearest to where the span would be,
   else the first; note `span_relocated`), and only a value that the input does not contain
   is rejected. Instructions are templated with
   the label ("Emma Thompson's full name"). The offsets are those of the whole input, so a request
   whose state holds only windows of it has no mentions line. Every Clef state about
   the entity (attribution, null check, typed fields, sub-model presence, repair) is `{"input",
   "entity": {"label", "mentions": [{"text", "span", "context"}]}}`, mentions with spans and a short
   context so that two entities with the same label stay apart. An over-budget input is narrowed to
   windows around the entity's own mentions (and the values being checked), never cut to its start,
   and the spans, which would point at nothing in a narrowed input, are then left out. A run
   warning about one entity starts with its label and key (`Emma Thompson (e0): …`). Chosen over packing `person.0.x, person.1.x` because disambiguation lives in the
   state not an index, server-side pass splitting would separate an entity's fields, records
   stay isolated (no cross-record echo), and failures stay per-entity.
   **Result:** one `ResultRecord(kind="insert", json_doc_id=None, entity=EntityInfo(key, label,
   mentions, first_mention, qualify_p, parent=None))` per qualifying entity in first-mention
   order (`match_probs` stays empty without existing records, see step 8), the skipped entities (`not_qualified` and
   `validation_failed`, in first-mention order) and the warnings; `responses` holds the valid
   models in the same order.
7. **Attribution (default on):** Clef `noul` per non-null value and per list element. A scalar: "Does
   `input` state that {desc} of `entity.label` is '{value}', or does it follow by simple
   arithmetic or date math from figures the input states about `entity.label`?" An ORDERED QUANTITY (a number field named
   `quantity` or `qty`, or `units`, `count`, `pieces`, `seats`, `tickets`, `copies`, `boxes`, `num_*` on a line of
   an order, booking or the like, or whose description is a number of units, items or copies ordered, wanted
   or to order, or "how many ... to book/order/attend/reserve/buy" (a stated "how many children she has" is not); never a price, cost, id, "number of the invoice", a count to date such as "copies sold to date",
   or a total) is asked as "Does `input` ask for
   {value} of `entity.label` (as the {desc})?": "add two more of the blue mugs" asks for a quantity of 2
   that no sentence states (live 0.34 for the "states" wording; this one 0.78 to 0.86 for 2, 0.02 to 0.04
   for 3, 0.92 to 0.98 for a stated 1). A total is not asked this way: "state or ask for" took a derived
   total from 0.88 to 0.52. **For an INSERT the question adds one sentence**: "`entity.label` is new, nothing of it was
   held before: an amount the input asks to add ("two more") is the whole amount of it." The engine knows the line is
   not in the record (the match said `__new__`); the model reading the input does not, and with "two more" it
   suspects a quantity already there (live: 0.48 for 2, a tiebreak by gpt-6-luna said "no" at every effort).
   With the sentence 0.84 at the first look. It is never added to an update (`Ctx.fresh` is False on the
   input-only copy of a record's context): there "two more" stays a relative edit and is refused as
   `relative_edit_unsupported`. **Nor to a new item that resembles a held one**: the match says `__new__` but its
   words are those of an item the list holds ("mugs" and a held "blue mug": the stems of one's words, plural "s"
   taken off, include the other's, `_collection._held_item`); `Ctx.held` is that item's record, `fresh` is False,
   and after the ladder the relative-edit question (kind `relative_edit`, as for a replacement) is asked of the add
   with the held amount as the current value: at `verify.relative_edit` or above the field is rejected as
   `relative_edit_unsupported` (warning of that name; the required quantity is then missing, so nothing is
   written). A new item that matches nothing held ("two more notebooks" next to a mug line, "red mugs" next to
   "blue mug") is still the whole amount, and so is any new item of an empty list. The rule: refuse "N more X"
   when a held item matches X; treat it as new when none does and the match said `__new__`. A description that is a CLAUSE (it starts with how, whether, when, where, why,
   what, who or which as a word of its own followed by more words: "how the person relates to the speaker";
   "Which-way sign text" and "Who's Who entry title" are noun phrases; the clause is put into the sentence with its
   first letter lowercase, and the element and "ask for" questions take it too: "an answer, for the entity, to
   ..." and "(as how many ...)") does not take "of `entity.label`", which hung
   the entity on the clause's last noun ("the speaker of Olivia Davis"; live 0.69 for "cousin", the escalation
   tiebreak answered "no" or "yes" by effort): it reads "state that for `entity.label`, {desc} is '{value}'"
   (0.86). A list element names its field: "According to `input`, is '{value}' one of
   the {desc} of `entity.label`? Something the input asks to include counts." with the plain criteria
   "Yes."/"No." (without the field a non-contact scored higher than a contact). The scalar criteria:
   true: the input states it about the entity, or it follows by simple arithmetic or date math from
   figures stated about the entity; false: the input does not state it about `entity.label`, or it
   concerns another entity. A computed value such as a total stated only as its parts is a legitimate
   answer, a figure of somebody else is not; an integral float is shown as `1080`, not `1080.0`.
   `entity.label` of a single record being updated carries what it is called ("the Bill INV-2291": its
   identity values, else its id when short and with a letter): a figure the input gives for another invoice
   is then not attributed to this one (0.88 became 0.015). Clairic's instructions keep the plain label.
   ≥ 0.75 keep, < 0.4 drop (`attribution`), between → isolated re-ask then one more check. The second
   check is a DIFFERENT request: on a short input the re-ask changes neither the value nor the state (the
   whole input fits the window), so a repeat would be byte-identical, the run's memo would return the same
   middle-band answer and the value would always be dropped. The second look quotes the clause the re-ask
   found the value in ("Reading the passage "Please add two more of the blue mugs" of `input`: ..."; with
   no span, "Reading `input` once more, and only what it says about `entity.label`: ..."). Under the
   defaults (isolated decisions) a middle-band value is therefore kept if the second look says yes at
   0.75, not dropped by construction. (Live 5: the quantity of "two more of the blue mugs" scored 0.4793
   next to the clause about the tea towels; the clause quote is the fix, not yet checked live.) The
   0.75/0.4 defaults come from one live observation (a bleed kept at 0.62); they are guesses to be
   tuned by the evals harness (M10a). On updates an op must pass
   **both** attribution (right entity) and the change gate (new vs. record).
8. **Matching (`_match.py`, `apply_collection`)**, for a `Many(S)` with existing records of S (the
   `ExistingRecord`s of S, plus a claimed `"__any__"` one). `MatchConfig(accept=0.6,
   min_confidence=0.5, render_fields=4, collision="warn", on_ambiguous="skip",
   delete_threshold=0.85)` (frozen, hashable, validated; `create_extractor(match=)`):
   * **Request.** One Clef `choice` per qualifying entity (call kind `match`, question id `match`,
     `CallMeta.entity` the entity key): "Which listed record is `entity.label`?" over the
     entity-anchored state `{input, entity: {label, mentions}}` (`Ctx.fitted`: mentions with spans
     and context; an over-budget input is narrowed to windows around them), with criteria
     `{"rec_<i>": <the record, normalised keys, as a JSON object>, …, "__new__": "A <Title> not
     among the listed records"}` (`i` is the record's place in the list, so ids stay stable when
     the list is cut). Entities have distinct states, so every entity is its own request, all
     concurrent. A missing answer gives no probabilities (the entity is ambiguous).
   * **Prefilter.** More than 254 records (Clef allows 255 options, `__new__` is one): each entity
     is offered the 254 records most similar to it, by identity: the best difflib token-set ratio
     between the record's identity value(s) (`Many(identity=)`, else the schema's, else its first
     text fields) and the entity's label and each of its mention forms, a diminutive
     (`_coref.nicknames`: Bob/Robert) counting 0.9; ties go to the earlier record. One run warning
     `match_prefiltered` carries the number of records. Records that were not offered have no
     probability.
   * **Budget.** If the state plus the options estimate over `limits.clef_state_token_budget`, the
     records are rendered compactly (identity fields, up to `render_fields` further non-null
     scalar fields, the first 3 elements of every list, long texts clipped to 160 characters); if
     that is still too much only as many of the most similar records as fit are offered (at least
     one); if even one does not fit beside the whole state the input is narrowed further.
     One warning `match_compacted` says so (and how few records an entity was offered).
   * **Assignment.** The probabilities of all entities go into ONE one-to-one assignment: the
     Hungarian algorithm (pure Python, `hungarian`; O(rows² · columns)) over the cost
     `-log(max(p, 0.001))` (every probability below 0.001 costs the same, so a stronger claim
     always beats a weaker one above the floor, and the result is the exact optimum of that
     clipped matrix, cross-checked against brute force), with the records as columns plus one
     `__new__` column per entity, so any number of entities can be new and the loser of a
     contested record falls to its best feasible alternative. A record that every entity gives
     no more than the floor is no column (it can never beat the entity's own `__new__` column;
     this keeps the matrix small with hundreds of records). Exact ties go to the earlier
     entity and the earlier record (a perturbation of 1e-12 per column index, not dictionary
     order). **Collisions:** two entities that both claim one record, each with its most likely
     option being that record at `p >= 0.5`, warn `match_collision` (both entities, the record, the
     probabilities, who got it and where the other fell back). With `match.collision="merge"` the
     claimants are first checked pairwise with the coreference cluster-level merge check (Clef
     `noul`, kind `coref`, id `merge`); two sets of claimants join only if every pair across
     them passed `coref.merge_threshold`; a merged entity (the earliest key, the union of the
     mentions, relabelled) is matched again (`match_merged` warning) and a collision that
     survives warns as above.
   * **Thresholds.** An assignment stands only if its option (a record or `__new__`) has `p >=
     accept` AND the answer's confidence is `>= min_confidence`, or if the pick is a record that is
     Clef's own top option (above every other option, `__new__` included), the ONLY record any of the
     entity's mentions fits (`identity_compatible`) and has `p >= accept_compatible` (0.5) and beats every
     other option (`__new__` and the runner-up) by `compatible_margin` (0.1; 0.51 against 0.49 is a coin
     flip), and the fit stands on its own (`strong_compatible`: a mention of two words or more, or a single
     word that is a word or diminutive of the record's identity; "Dan" is only the start of "Daniela Cruz",
     "Emma Thomson" for "Emma Thompson" is fine). A blank identity value fits nothing. Clef's
     choice `confidence` is not the probability of the chosen option (live: p 0.84 with confidence 0.47).
     The assignment and the identity-conflict guard are unchanged. **Pairwise confirmation
     (`match.confirm_accept`, default 0.7, 0 off; call kind `match_confirm`).** Clef's record choice
     answered `__new__` at 0.83 for "Emma" against the record "Emma Thompson" and 0.84 for "Maria" against
     "Maria Costa" while the record-presence check agreed with the records at 0.87, and gave the README's Emma
     0.506 (a coin flip for the margin). The confirmation settles only an UNSETTLED outcome. A record picked with `p >= accept` and
     `confidence >= min_confidence` is final; a `__new__` is final at `p >= match.new_final` (default 0.9) BY
     ITS PROBABILITY ALONE, since the choice's `confidence` is calibrated differently by every provider
     (Cloudflare: 0.44 for "Emma" `__new__` 0.831; Experiential: the top probability itself, 0.8176). Below
     0.9 the pairwise answer may settle it, and it must beat `max(confirm_accept, p(__new__))`. Live numbers:
     `different_emma` (0.984) and `new_hire_same_first_name` (0.929, a sure 0.92) stay new whatever a
     gullible pairwise says; `short_form` (0.8176 or 0.831), the README Emma (0.468) and "Maria" (0.84 or
     0.696) may be confirmed ("A new hire, Emma, joined the Oslo office" is never written onto Emma
     Thompson by a gullible pairwise 0.75). And EXACTLY ONE existing
     record fits the entity's mentions at all (`identity_compatible`) and strongly (`strong_compatible`),
     one Clef `noul` in a request of its own asks "In `input`, is `entity.label` the same {kind} as the
     existing record `record` ({identity})?" over the input windows around the mentions, the entity and the
     record. At the knob the entity is matched to the record (warning `match_confirmed`); otherwise the
     choice's outcome stands. Never with two or more compatible records ("Emma" with Emma Thompson and Emma
     Watson), never for "Emma Watson" against "Emma Thompson" (not a strong fit), never when a mention of two
     words or more contradicts the record ("Emma Davis, ... Emma has a cat" for Emma Thompson), never for a
     record another entity holds. The identity-conflict guard still refuses a confirmed match whose extracted
     name contradicts the record, and then the entity is treated as NEW (`match_confirmed_reverted`: an
     insert, or `inserts_disabled` when inserts are off), never dropped. Else the entity is ambiguous:
     `SkippedEntity(reason="ambiguous_match", candidates=((option, p), …))` (the options with p ≥
     0.01, the most likely first; the records it might have been are NOT handled, so they stay
     untouched), or, with `on_ambiguous="insert"`, it is treated as new when its `__new__`
     probability is at least `accept` (still subject to the insert rule of step 9; else it is
     skipped as ambiguous: "insert" never duplicates a record the entity most likely is), or
     `"raise"` raises `AmbiguousMatchError` (before any extraction). A collision loser whose
     fallback is below `accept` is ambiguous too.
   * **Identity guard** (`update_one(identity=)`). A match is a model's guess, so after the
     entity's identity has been extracted (and verified) it is compared with the record's:
     they must be forms of one name (`_coref.identity_compatible`: every word of the one with
     fewer words is a form of a different word of the other, i.e. the same word, a diminutive,
     an initial, a prefix or a small typo; or the initials of the whole name; words with digits
     must be equal). "Liz" fits "Elizabeth Smith", "Sarah" fits "Sarah Johnson", "Emma Watson"
     does NOT fit "Emma Thompson" (a shared first name is not enough). The check is made for
     the extracted value AND for each of the entity's mentions: if ANY of them is compatible
     with the record's identity the match stands (coreference linked the names), so "Sarah
     Johnson got married and is now Sarah Miller" is the record "Sarah Johnson" and the new
     name goes to the change gate like any other edit; an entity none of whose mentions fits
     ("Emma Watson" onto "Emma Thompson") is still refused. With several identity
     fields the match stands as soon as one fits; a field the input gave no value for says
     nothing; a schema without an identity field has no guard. On a conflict the WHOLE update
     is refused: nothing is gated or applied, every proposed operation is
     `RejectedOp(op, path, "identity_conflict", detail)`, the record is an `update` with no
     patches (`after == before`) and the entity is also reported as `SkippedEntity(reason=
     "ambiguous_match", detail="identity conflict: …", candidates=…)`.
   * **Result.** `EntityInfo.match_probs` holds `{record id: p, …, "__new__": p}` for the options
     it was offered; `ExtractionResult.records` are ordered by first mention (updates and inserts
     mixed), then the deletes in record order; `skipped` is in entity order; `json_doc_id` is
     the record id for updates and deletes and `None` for inserts; `response_metadata` and
     `responses` list the valid inserts and updates (a delete has no model).
9. **Flags per entity:**
   * matched to record r → the full update path (`update_one(entity=, focus=)`: presence is NOT
     asked again, the Clairic focus header and every Clef state name the entity and its mention
     spans, so attribution and the change gate are about that entity only; per-operation gate; the
     identity token-subset rule, so "Sarah" never overwrites "Sarah Johnson"; scalar-list appends
     as `add /notes/-`; `not_mentioned` for fields it did not state), if `enable_updates`, else
     `SkippedEntity(reason="updates_disabled")` (the record counts as dealt with, not untouched);
   * `__new__` → insert (`insert_one`) if `enable_inserts`, else
     `SkippedEntity(reason="inserts_disabled")` (an extractor given existing records of S is the
     only case here, so "or no existing records of S were given" is the plain insert path of
     step 6);
   * **deletes** (`enable_deletes`; skipped when the presence question found no mention of S at
     all): the records go into Clef requests of up to 64 (fewer when the state would be over the
     budget; a record too big even alone is compacted as above), one `noul` each (call kind
     `delete`), over the structured state `{input, records: {r0: <record>, r1: …}}` with the
     question `delete.r<k>` ("Does `input` state that `records.r<k>` should be deleted — e.g. the
     person/item was removed, has died, or should no longer be tracked? A change to its details
     is NOT deletion."; `k` is the record's place in the list). `CallMeta.entity` is unset and
     `CallMeta.field` holds the ids of the request's records, comma-joined, in question order.
     They run concurrently with matching. At `>= delete_threshold` the record is
     `ResultRecord(kind="delete", json_doc_id=<id>, before=<record>, after=None, value=None,
     entity=None, delete_p=p)` and `ExtractionResult.deleted` lists its id;
   * a record both matched (accepted) and flagged for delete (the input MENTIONS it, as in "please
     remove X"): the extraction runs first and the decision is made on what the input PROPOSES
     for the record (`UpdateOutcome.proposed`: the operations that were applied, the ones the gate,
     the duplicate check or validation refused, and the answers the verification ladder refused;
     a short form of the record's name, a repeat of what it holds and a field the input is silent
     about are not changes). Nothing proposed (nothing but "remove it" is said about it) → the
     record is deleted (`kind="delete"`). Anything proposed, passed or refused ("drop the quantity
     of X to 1" flagged by mistake, even if the gate then refuses the change) → the input both
     changes and removes it: `SkippedEntity(reason="conflict_update_delete", candidates=((id,
     match p),))`, neither an update nor a delete, with the proposed changes listed in its
     `detail`. The rule does not depend on `enable_updates`: with updates off the extraction and
     verification still run, to see what is proposed, but the record's own changes are not gated
     or applied. The record's ITEMS are looked at either way (M10a, S1): the nested pipelines run
     as for an update and what they would change or add counts as proposed, so a flagged parent
     whose items the input changes ("Emma Thompson's dog Rex is now 5", Emma flagged) is a
     `conflict_update_delete` whatever `enable_updates` says, not a deletion. Propose-only runs
     the same near-duplicate check as an update before it counts a list element as proposed (a
     repeat of an element the record holds is not a change), and an answer the ladder refused
     counts only when it differs from the record's value. A NEW item counts as proposed whether or
     not `enable_inserts` is on ("Emma got a new dog, Max" with Emma flagged is a conflict, not a
     deletion): the flagged record's extraction runs with the nested inserts forced on, and what it
     produces is only read, never kept. An
     identity-conflict refusal (a wrong match) is reported as such and holds the delete. Cost: every
     flagged and matched entity costs one extraction (Clairic per group plus the cross-check and
     the ladder; with updates on also the gate for its changes), and, if it holds collections, their
     discovery and item pipelines as for any updated entity;
   * a record flagged for delete that an `ambiguous_match` entity most likely is (its best record
     candidate, at 0.25 or more, including an identity-conflict refusal) is not deleted either:
     run warning `match_delete_held`;
   * records nothing happened to (not matched, not flagged, held back, or matched ambiguously) →
     `ExtractionResult.untouched`, keyed by (schema, id) so that an `"__any__"` record is told
     apart; no ops. With no entities at all (presence below `discovery.presence_absent`, or
     nothing found) nothing but the discovery requests is made.

## Nested collections (M9, `_collection.py`, `_nested.py`, `_entity.py`)

`list[Model]` and `dict[str, X]` fields go through the same engine as a top-level `Many`, scoped
to the entity that holds them. Every entity (a `One` schema's single entity, each entity of a
`Many`, an item of an outer list, at any depth within `limits.max_depth` / `max_recursion`) runs its
own fields first (the groups, unions and Clef fields as before) and then, for every collection
field of a group that stands (an optional sub-model that was absent, or a union member that was
not chosen, has none), `collect(Scope(item_plan, source, cardinality, parent=<its EntityInfo>,
pointer="/pets", existing=...))`. The item plan is `plan.collection(spec)`; the cardinality is the
field's `Many` marker (identity, label), a mapping's is `Many(identity="key", label=<singular of
the field name>)`.

**What is scoped to the parent.** Discovery: "Every mention of a Pet belonging to Emma Thompson by
their name ...", the focus header `<focus schema="Pet" task="discover" parent="Emma Thompson">` ("Look
for every pet of the parent named above ... ignore any pet that belongs to someone else", with the
parent's offsets); presence, one Clef `noul` per (parent, collection), "Does `input` mention any Pet
... belonging to `parent.label`?" over `{input, parent: {label, mentions}}`; the count ("How many
distinct Pets belonging to ..."); qualification ("Is `entities.e0.label` a Pet belonging to
`parent.label`?", the state carries `parent`), which is what keeps Emma's dog out of Michael's
record when discovery offers it for Michael (`not_qualified`, `EntityInfo.parent` = Michael's
key); coreference and merge checks as at the top level. Item extraction: `<focus schema="Pet"
entity="Rex" parent="Emma Thompson">` and every Clef state about the item is `{input, entity:
{label, mentions, parent: {label, mentions}}, ...}`, so attribution, the change gate, null checks and
repair are about that item of that parent. The label of a `One` parent is "the Order". Call
metadata: `CallMeta.scope` is `"<parent key>/<pointer>"` (`e0/pets`) on every call of the
collection and of its items, and run warnings start with the ancestors: `Emma Thompson (e0) ›
pets: count_exceeds_discovered ...`, `Emma Thompson (e0) › Rex (e1): ...`.

**Items without an identity (rows).** A nested list whose item model has no text field to name an
item by (`list[EngineInfo]` with a `Literal` position and serial numbers, `list[LandingGear]`
with an enum) used to be discovered like a list of people: the model listed the words of the
table (live: 40 to 70 per list and aircraft, among them the class name, `1`, `Honeywell`,
`TSN:**<br`), each was clustered (up to 29 coreference requests on a small sheet; on a real
table the incremental assignment that overflowed the Clef window) and qualified, 134 to 208
requests for one datasheet. For a nested collection with no identity (`Scope.parent` set and no
identity names; a top-level `Many` keeps its behaviour, since `Many(Reading)` over bare numbers
is legitimate) the engine now:

1. asks by what an item *is*: "Every mention of a {kind} belonging to {parent} ({docstring}), in
   the order they appear. Give one entry per {kind}: the exact words of the input that label it or
   tell it apart from the others, such as its position, number or model (a {kind} has {fields}). Do
   not list headings, units, bare numbers or the values of those fields on their own, and list a
   {kind} once for each row or place that describes it." {kind} is the title-derived kind
   (`EngineInfo` is "engine") unless `Many(label=)` says otherwise;
2. **screens the mentions in code** (`_prefilter`, warning `junk_mentions_dropped`), ranking each
   mention: markup (`**`, `|`, `<br>`) is stripped, and a mention is dropped when nothing is left or
   when it is only a heading (rank 0: the stoplist `model`, `type`, `total`, `cycles`, `status`, ..., what the
   collection and its item are called, or the label of one of the item's own fields or a word of
   it, `tsn` for `tsn_hours`). It is kept (rank 3 or more) when it is a word or phrase, a serial or
   part number (four digits or more, or letters with a figure), a figure when the item model has
   a *required* numeric field (`Reading.value` keeps `21.5`), or an **option of one of the item's
   own choice fields** (`1` for `position: Literal["1", "2"]`, `Nose` for an enum). Bare figures and
   one or two letters (`A`, `MA`, `L`, `LH`) rank 1 or 2 and are dropped, **but the screen never
   takes the candidates below the count Clairic gave**: when fewer survive than the input holds
   items (a landing gear table whose rows are `L`, `R`, `LH`), the best of the dropped (a capitalised
   code before a bare figure, never a heading) come back, up to ten per item, and go on to step 3,
   which is what decides about them (the warning says how many came back);
3. **qualifies each remaining mention on its own before coreference** (`_group(screen=True)`): one
   `noul` per mention, 64 to a request, so a table of 100 cells costs two requests, and only the
   mentions that are not refused are clustered. The refused ones are not entities; one warning
   (`mentions_screened_out`) counts them and quotes the first five, instead of 100 `SkippedEntity`
   rows that would bury the real skips. Qualification answers are cached by mention, so a
   cluster that is a single mention is not asked again;
4. on a **count below the candidates** runs the merge checks as for any `Many`, then cuts only what
   qualified *weakly* (below 0.8; `qualify_p=None` is never cut) down to the count, worst first:
   those are skipped as `capped_to_count` with `items_capped_to_count` warning. A confidently
   qualified item is never dropped because Clairic undercounted; the run says
   `count_below_discovered` and keeps it.

On a synthetic engine, gear and component table (two engines, three gear legs, 120 table cells that a
model might list) this takes 34 requests where the old path took 61 (coreference 29 → 2), and no cell
is reported as a skipped entity (the one-by-one qualification asks about 141 mentions in 4
requests; serial numbers and option values are kept for it, which is the price of not losing rows). Left out: a plural phrase that covers several items ("2 x V2527-A5",
"CFM56-7B26 engines") is still one mention, and so one item; splitting it by the count is a
positional extraction that needs the model, not a rule, and is a known limit (see Risks).

**Insert.** Each item runs `insert_one` (full pipeline, own repair loop); the valid ones are
put into the parent's document in first-mention order (a list), or keyed by the extracted key
(a mapping: `parent[key] = entry.value`; an entry without a usable key or repeating one is
rejected, the first wins). A collection with nothing in it takes its default; a required one is
`[]` / `{}`. The items' answers are answers of the parent, keyed by path (`pets.0.name`,
`scores.math`, `stock.mugs.quantity`; `FieldAnswer.path` is `("pets", 0, "name")`), their
rejections are the parent's, re-pointed (`/pets/0/weight`), and an item that fails validation
after its own repair is left out (`RejectedOp(reason="invalid_item")` at the collection's
pointer) and is a `SkippedEntity(reason="validation_failed")` whose `entity.parent` is the
parent's key. `ExtractionResult.skipped` carries the skipped items (`not_qualified`,
`validation_failed`, `inserts_disabled`, `ambiguous_match`, `conflict_update_delete`) in order
after their parent's own.

**Update.** The parent's existing items are the existing records of the scope: a list element
has the id of its index (`"0"`, `"1"`), a mapping entry its key and the record `{"key", "value"}`;
keys are normalised like any record's. `apply_collection` then does what it does for a `Many`
with records (steps 8-9), about the items of this parent: Clef `choice` per discovered item over
the parent's items plus `__new__` (the state carries the parent), one-to-one Hungarian
assignment, thresholds, `on_ambiguous`, `match_collision` etc.; then
* *matched item* → the full update path on the item against its element (`update_one`: Clairic
  with the element in `<existing>`, input-only cross-check, ladder, per-op change gate against the
  ELEMENT, identity guard per item, repair loop on the item), and the operations it applied are
  rebased into the parent's patch list (`replace /items/1/quantity`; a mapping entry's `/value/qty` is
  `/stock/mugs/qty` and its `/value` is `/stock/mugs`; an operation on the entry's key is dropped as
  `key_fixed`); they were attributed and gated already and stand as they are. An item that is
  invalid after its update contributes no operation (`invalid_item`);
* *new item* (`__new__`) → `add /items/-` (a mapping: `add /stock/<key>`, escaped, a key the
  record has is a `duplicate`) of the new item's document (values equal to a default left out),
  if `enable_inserts` or the parent record held no items; else `SkippedEntity(inserts_disabled)`.
  The parent judges it like any other addition: the near-duplicate check and one change gate
  request over `{input, entity: <the parent>, record: <the parent>, proposed_change}`;
* *existing item the input does not mention* → untouched, no request about it beyond the match
  options and the removal question;
* *removal* only with `enable_item_removals` (`create_extractor`): one Clef `noul` per existing
  item, over `{input, records: {r0: <item>, ...}, parent}` ("Does `input` state that `records.r0`
  should be deleted from `parent.label` ..."), at `match.delete_threshold` → `remove /items/<i>`
  (a mapping: `remove /stock/<key>`); an item both matched and flagged follows the same rule as a
  top-level record (step 9): nothing proposed for the item (the text only says to remove it) →
  `remove`, any proposed change, passed or refused → `SkippedEntity(conflict_update_delete)` with
  no operation.

All operations join the parent's list and follow its rules (see "Operation order" below);
`apply_ops` knows the nested list pointers (`_fix_missing_parents` creates a list the record lacks,
and treats one the record holds as `null` as missing: the `add` of the list replaces the null); the patched parent is validated, an
operation on the line of a validation error is dropped (`validation`, its path inside the item),
and an error about an element the record already held is not the update's (`_held_element`: no
request, the update's operations stay). The items' answers are `fields` of the parent
(`items.1.quantity` with its `gate_p`); a position is the element's index in the RECORD (a new item
is numbered after the existing ones), so a removal does not renumber the answers.

**Relative edits are not supported** (settled). The library reads what the text *states* and never
computes from the old value: "two more of the blue mugs" is extracted as 2, whether or not the order
has a blue mugs line. For a line the order lacks that is the right quantity; for an existing line of
3 it is the stated 2 and not 5, and it is the change gate's decision (and the author's: write "make
it five") whether `replace quantity 2` stands. There is no arithmetic on the record anywhere in
the update path. `examples/07_order_line_items.py` and `tests/unit/test_nested_collections.py`
pin this.

**Textual fallbacks of discovery.** A discovered value no returned span covers (or listed more
often than spans cover it) takes occurrences of its text by search: all of them when no span covers
it, else only as many as are missing, standalone ones first (an occurrence followed by a capitalised
word is probably a longer name); never an occurrence inside the text of a longer listed value;
and in a nested scope only those near the parent's mentions (the verification windows around them)
when there are any, else anywhere.

**Required collections** that were not extracted (past `max_depth` / `max_recursion`, or a list of
a union of models) are `[]` / `{}` in an insert, like one the input says nothing about. For a mapping
of models `limits.max_recursion` counts the VALUE model on the chain (each level builds its own
entry class).

**Known limits.** A list of a union of models is not extracted (`unsupported_list_item`). A mapping
key is its identity and is not renamed. The parent label that scopes a request is the label of
the parent entity (the canonical label of its mentions; "the Order" for a `One`), and its mention
spans settle which of two parents with one label is meant. A collection is extracted after its
parent's own fields (one more sequential hop), and the items of an inserted parent are not
re-extracted by the parent's repair loop (each item has its own).
10. Known limits, measured live. Olivia's "cousin" (a relationship in the speaker's voice) scores 0.6875
    on attribution, under keep 0.75: the speaker-relative value is dropped, and a REQUIRED
    `relationship` then makes her record invalid (a `Literal` or an optional field avoids it). The
    quantity of a NEW line, "add two more of the blue mugs" when no such line exists, is genuinely
    ambiguous ("2 of blue mugs" scored 0.48): it is asked as a requested quantity and the second look
    quotes the clause, but it can still be dropped. `computed_total` (0.7308) and a city (0.7124) sit
    just under keep 0.75; the threshold is not lowered for them (a bleed was kept at 0.7139). A single
    capitalised name that only starts a sentence ("Alice called.") is not made up from the mention.
    **The `new_final` band is a known risk.** With p(`__new__`) between 0.7 and 0.9 and a pairwise
    answer above it, a gullible pairwise answer (0.9 for a new "Emma" next to "Emma Thompson") still
    writes the new person's facts onto the wrong record: that is the shape of the correct live matches
    too (short_form 0.83 then 0.87, Maria 0.84 then 0.87), so no margin separates them; a test pins it
    (`__new__` 0.85, pairwise 0.9: matched). Opt out with `match.new_final=0.7` (from 0.7 a `__new__` is
    final) or `match.confirm_accept=0` (confirmation off; 1.0 never confirms).

## Verification ladder (`_verify.py`) and update path (`_diff.py`, `_gates.py`)

| check | default | rule |
| --- | --- | --- |
| Answer-level thresholds | on | reject if `p_null > null_threshold`, `confidence < 0.5`, span inside `<existing>`, Clef choice < 0.55 |
| Attribution | on | above |
| Input-only cross-check (updates) | on (`verify.input_only_crosscheck`) | second Clairic pass per group (call kind `crosscheck`) with the same fields and NO `<existing>` block. For a change the primary pass proposed: the raw cross-check value agrees (normalised equality; text similarity ≥ 0.9 only for free text; its own confidence does not matter) → proceed; null there → reject `echo_crosscheck`; two different values → Clef `choice{a: primary, b: cross-check, neither}` (kind `adjudicate`) at ≥ `verify.adjudicate_min`: `a` keeps, `b` takes the input-only value (if it passes the answer thresholds), anything else rejects `crosscheck_disagree`; a list element must have an agreeing cross-check element or is dropped `echo_crosscheck`. Where the primary pass gave nothing new and no span in the input (null, an `echo_span`, or the record's own value without a span) and the cross-check reads a usable different value, that value is taken and goes on through the ladder and the gate (covers number/date answers, which carry no span); where the primary pass read the record's own value WITH a span in the input, the two are adjudicated and only a decisive win of the cross-check takes over |
| Low-confidence re-ask | on (< 0.85) | isolated single-field request on mention windows; disagreement → Clef adjudication |
| Borderline-null | on (`p_null` 0.25–0.75) | Clef `noul` "does the input state it?" → re-ask or null |
| List near-duplicate dedup | on | difflib 0.6–0.9 vs record elements → Clef `noul` "same fact?" |
| **Change gate** | **always** | one Clef request per op, structured state `{input, entity, record, proposed_change}`; add ≥ 0.55, replace ≥ 0.6, remove/set-null ≥ 0.85 |
| State-variant self-consistency | off (`k=1`) | opt-in |
| `guess` as a value | never | metadata only |

Diff walks the field plan, never the document (`_diff.py`): absent → `add`; differs
(normalised: ISO dates, `math.isclose` numbers, stripped text, enum values; free-text `str` fields
and their list elements also case-folded with internal whitespace collapsed, so "neighbor" for
"Neighbor" is no change, while a `Literal`/`Enum`, a union or a `pattern`-constrained string
compares exactly) → `replace`;
equal → nothing; null answer → **no op**, recorded as `RejectedOp(op=None,
reason="not_mentioned")` (silence is not deletion; opt-in `gates.allow_null_removals`);
scalar lists use set semantics → `add /notes/-` for new elements only. Order (see "Operation
order"): in place → removals (highest array index first) → appends, a deterministic sort key. Apply with
`jsonpatch.apply_patch` on a deep copy, op by op; on conflict try `_fix_string_concat`
then `_fix_missing_parents`, recording `repaired_from`. The full rules are in "Update path".

Clairic's list `value` and `spans` are NOT index-aligned (the first live run returned a span for
a second "Sarah" that the values left out, so the spans after it belonged to the wrong
elements): a text list answer is therefore re-paired by text right after the response
(`_verify.align_list_spans`; the spans as returned stay in `ClairicAnswer._raw_spans`): each
element gets the first unused span whose covered text equals it (case and whitespace aside),
else the first free occurrence of its text in the returned input block, and an element with
none leaves the answer without spans. Discovery uses ALL the returned spans: every span whose
text equals a listed value is a mention of it (so a repeated mention that the values list
once is kept), a value no span covers (or listed more often than spans cover it) takes every
whole-word occurrence in the input, and `mention_not_in_input` is warned only for a value the
input does not contain at all.

A required `list[...]` field the input says nothing about (null answer, or every element
rejected) is assembled as `[]` on an insert, not as `null` (which fails validation: "Input
should be a valid list"); a nullable one keeps `None` and a defaulted one its default. An update
makes no operation for it.

**Open-ended lists: the sentences strategy (`_sentences.py`, M10a).** Clairic's `list` type is a span
picker. For defined items (names, contacts) it is confident (0.9995 live); for an open-ended field
("things worth remembering") it returns word fragments at confidence 1e-6 to 1e-23, so the answer is
rejected and no note is ever added (live runs 2: examples 03 and 05). A `list[str]` field of free
text can therefore be read another way, chosen per field with `s1_field(list_strategy=...)` or for
every field with `limits.list_strategy`:

| strategy | what happens |
| --- | --- |
| `"list"` | Clairic's `list` type only (the behaviour before M10a) |
| `"sentences"` | Clairic is not asked about the field at all. The input is split into sentences in code; Clef is asked one `noul` per candidate sentence (call kind `sentences`, `CallMeta.field` the wire name): "Is this sentence one of the <field description>, about `entity.label`: "<sentence>"?", with criteria. A sentence at or above `verify.sentence_keep` (0.6) is an element, verbatim, with its span |
| `"auto"` (default) | Clairic's list first (the request and the cross-check are the same as `"list"`). If its answer holds elements but its confidence is below `limits.list_auto_below` (0.5; a null or empty answer is the model saying there is nothing, not distrust), or the answer is missing, the existing low-confidence re-ask runs first (one isolated Clairic question over the input alone, kind `reask`). A re-ask that is trusted replaces the answer and the field stays a Clairic list. Otherwise Clef is asked once per schema and field and run (kind `list_kind`, memoised; entities that run at the same time share the one request) a `choice` between `items` ("short named values: names, languages, tags, ids") and `statements` ("free-form statements or facts"). `statements` → the field is read by sentences; `items` → the old low-confidence rejection stands, so a list of languages Clairic is merely unsure of is never turned into sentences. `statements` needs `statements` CHOSEN with `p(statements) >= gates.min_choice_prob`; for a HEDGING reader only (the Decisions API, or a hard pick: `_hedged_veto`) `p(items) < ITEMS_VETO` (0.75, at most the hard cap, so a hard `items` always vetoes) is enough: only a CLEAR `items` keeps a list out of the sentence path, because a wrong `items` drops the field while a wrong `statements` still faces the per-sentence question, attribution, novelty and the change gate. A graded answer of any other provider keeps the plain rule: a list of phone numbers that Cloudflare's Clef calls `items` at 0.70 must not become the sentence that holds them. (Live, Decisions as the primary gave `items` 0.65 to 0.67, graded, 15 times out of 15 for a notes field, where Cloudflare's Clef gave `statements` 0.95 to 0.97; under the old rule no note was ever added: notes recall 0%.) A missing answer (no Clef answer, a refusal) is `items` |

An explicit `s1_field(list_strategy="sentences")` (or `limits.list_strategy="sentences"`) skips Clairic and the classification altogether and starts together with the field's other requests; the `auto` fallback needs Clairic's answer first, so it cannot.

Splitting (`split_sentences`, no dependency): a boundary is `.`, `!` or `?` (with closing quotes and
brackets) followed by whitespace and a capital, digit, quote or bracket, or a CJK `。！？`; not after
an abbreviation ("Dr.", "Mrs.", "Inc.", "St.", "e.g.", "i.e.", "U.S."), not after an initial (a
single capital before a capitalised word and after a capitalised word or another initial: "J. R. R.
Tolkien" is one name, "So did I. We left." and "She got a B. Then" are two sentences), not inside a
number or an address ("3.5", "$1.5M", "example.com"), not before a lower-case word; "a.m./p.m." end a
sentence before a capital. A line break always splits, a leading bullet is dropped and `;` splits
clauses. A word that is also a common word ("no", "am", "sat") ends a sentence, except in front of a
number ("No. 5", "Oct. 5"). Offsets are exact: `text[start:end] == sentence`. Known limits: text
written entirely in lower case has no capital to start a sentence and stays one; "Inc." or "St."
followed by a capital keeps its sentence together.

A transcript is text where at least two lines start with the same `Name:` label ("Me: ...",
"Sarah: ..."). Only there is the label not part of the sentences; it is kept as `Sentence.speaker`
and `label_span`, and still counts for scoping (Sarah's own lines are candidates for Sarah). A label
that starts a single line ("Her email: a@b.co.", "Plan A: go home.") is ordinary text.

Scoping: a `One` (no mentions) asks about every sentence. An entity of a `Many` is asked only about
the sentences that mention it and the next `limits.sentence_follow` sentences (4: conversational
filler sits between a name and the facts, "have you heard from Michael Chen? Actually, yes. We had a
video call last week. He's switched jobs..."). A mention is found by its span when it has one (two
people called Emma are told apart); mention texts are matched as whole words only for a mention
without a span. The reach of a mention stops before a sentence that names another entity of the same
collection (their mention spans travel with the entity: `Scope.siblings`, `Placement.others`,
`Ctx.others`). More than `limits.max_sentences` (200) candidates are cut with a warning. The questions
share the Clef window with the state, and the SUM fits `limits.clef_state_token_budget`: a request
starts with as many questions as fit in half of it (at most 64), is given its own state (the input
narrowed to windows around the sentences its questions quote, into what the questions leave) and, if
that still does not fit, the chunk is halved until it does (one question and the narrowest state that
do not fit warn). Token counts have a floor of one token per CJK character or emoji. An element needs `verify.sentence_keep` and the field's `min_confidence`; one below either is
dropped, the list is kept.

The elements then take the usual path. The ladder runs on them (attribution per element; an element
in the middle band is dropped, since Clairic's list type cannot re-read it; no low-confidence
re-ask; no input-only cross-check, the sentences come from the input alone and the field is marked
`crosscheck_agree=True`); on an update `_diff` keeps only the elements the record lacks, a
near-duplicate of an existing note goes to the `list_dedup` question and is rejected as `duplicate`
when it is the same fact, then (updates, `verify.note_novelty` = 0.5, 0 off) one Clef `noul` of its own per
sentence (kind `novelty`): "Does the sentence 'S' say anything about `entity.label` that `record` does not
already hold?" (true: the sentence adds information the record does not hold; false: everything it says is
already held by `record`: nothing else drops a note) over the record AFTER the update's other operations
that were ACCEPTED, so the other fields' gates run first (a sentence that only repeats a value written to
another field, "Her mobile is now ...", is `not_novel`; live 0.04 to 0.22 against 0.90 to 0.97 for the
notes that matter; but with the phone change refused by its gate the sentence is news, and "She moved to
Berlin to lead the Ops team" next to an applied city survives because "lead the Ops team" is new), and each remaining `add /notes/-` passes the change gate, judged against the
same record (the gate compared "Bob Roe's new phone number is 555-0199" with the record's OLD phone and
passed it at 0.97). Only sentences read by the `sentences` strategy are checked: a list Clairic reads holds
short named values, not sentences copied from the input that can repeat another field. Existing notes
are never touched. Repair never sends such a field to Clairic's `list`: a list over its `max_length`
keeps the sentences Clef was surest of, as many as the limit allows (minus the elements the record
holds, on an update), in input order, and is validated again at no cost in requests; any other error
ends the field (an insert's optional field goes back to its default, an update drops the operations
that break validation). `FieldAnswer.warnings` says how the field was read (`list_strategy:sentences`,
`list_strategy:auto->sentences (the list answer had confidence ...)`, `list_strategy:auto re-ask`,
`sentences:k/n`) and `FieldAnswer.raw` lists every candidate with its probability. `list_strategy` on
a field that is not a `list[str]` of free text is a compile-time `ValueError`.

The change is not free: elements are whole sentences, not tidy paraphrases (Clairic is extractive
and so is this), and an entity's notes depend on the scoping window. `python -m evals run --config
list|sentences|auto --tag notes` compares the strategies; `limits.sentence_follow` and
`verify.sentence_keep` are the knobs to sweep.

Clairic requests always send
`options.return_state_text=true` so span offsets are re-anchored on sentinel markers;
`truncated: true` → re-ask the same fields on overlapping chunks that cover the whole input (size from the returned `state_text`, overlap ≥ `verify.window_chars`), concurrently, and merge per field: every non-null scalar is kept as a candidate, ranked by confidence; candidates that agree (normalised equality / text ratio ≥ 0.9 / equal numbers) are one value, disagreeing ones go to a Clef `adjudicate` over the distinct candidates (≥ `verify.adjudicate_min`; one request for all conflicting fields, and a decisive `neither` makes the field null), then to attribution in confidence order (one request, first to pass wins; its probability is handed to the ladder so it is not asked twice), then the most confident stands for the ladder to judge; dates, times, durations, numbers and digit-heavy strings agree only when equal after normalisation, never on text similarity; lists are united in input order and an overlapping span replaces a kept element only if it contains exactly that one (confidence breaks ties for equal spans); an all-null merge reports the lowest `p_null` so the borderline-null check still fires; a chunk that is still truncated is split smaller once, then warn/raise `StateTruncatedError`. Any part of the input that is not sent always produces a warning.

**Operation order (`_diff.sort_changes`).** Operations name array elements by their index in the
RECORD, and only a removal shifts indexes, so: (1) *in place* first: replacements, then every other
operation whose path lies inside an existing array element at any level (`add /items/1/sku`,
`add /items/1/tags/-`, a set-to-null), because `add /items/1/sku` sorted after `remove /items/0`
would land on the wrong element; (2) *removals*, the highest array index first, and what lies inside
an element before the element; (3) *appends* (`/-`, a new key), in plan order. Test:
`test_diff.py` (two levels) and `test_nested_collections.py`.

**Decisive questions are asked one per request (`verify.isolate_decisions`, default on) and a
byte-identical Clef request is sent once per run.** Live, Clef's answer to a question depends on the
requests it sits in: the same attribution question gave 0.5597 alone and 0.2800 beside another one,
Tom's bleed 0.6743 in a batch of three and 0.7139 alone, and the question id and the criteria text move
it too (0.4762 with the long criteria, 0.8581 with "Yes."/"No."), so a threshold margin of 0.02 means
nothing: the levers are the wording, the criteria and the state, not the threshold. Attribution, the
note novelty check and the record-presence check (the questions that decide whether a value is kept)
therefore get a request each (`Run.call_clef(..., decisive=True)`); the other kinds stay batched. Clef is
deterministic for an identical request, so the run keeps the answers of the requests it has sent
(`Run.clef_memo`, finished requests only: a cancelled call is cancelled) and a re-check that would send
the same bytes again (a re-ask that returned the same value over the same state) is answered from there.
The memo is keyed on the SHA-256 of the canonical JSON of the request (keys sorted within objects, the
questions kept in the order asked: their order is part of the request), not on the request, which holds
the whole state and the images.
Clef's answer to "state that X" cannot be checked against arithmetic (live: a wrong total 1100 scored
0.89 and the right 1080 0.26 on a "follows by arithmetic" question), so there is no second arithmetic
question; and no value-independent field-level question for derived values (one positive case, and it
cannot tell a right derived number from a wrong one).

**A new entity's name falls back to its mention.** When extraction gives the NAME field of a new entity
of a collection nothing usable (null, or a low confidence on the very words discovery quoted: "blue mugs"
at 0.33 on the first pass and the repair, or refused by the null check or the attribution check: live 8,
"A new hire, Emma, joined the Oslo office" had "Emma" extracted at 0.997 and then dropped because Clef
doubted that the input states Emma's "full name" (null check 0.02, attribution 0.45), which left the
new person invalid and dropped), the field takes a proper-noun mention (label and span), with
the warning `identity_from_mention: <the reason>` on the field and in `result.warnings`; never for an
update. The field's span is the value's own words in the input (`_own_span`): the entity's label can be shorter
than the mention that was found for it (discovery on Jev listed the clause `Her name is Olivia Davis`, the
longer mention replaced `Olivia Davis`, and the label `Olivia Davis` was stored with the CLAUSE's span, so
the smoke's `spans_quote_the_input` failed intermittently), so the span is the occurrence of the value
inside the mention, else the closest one, and none when the input does not hold the words. Only a
name-like identity field qualifies (named `name`, `full_name`, `title`, `label` or
`display_name`, or marked `s1_field(identity=True)` with a description about a name, title or label), so a
Subscriber's `email` or a Ticket's `id` is never made up from "Dana Ruiz" or "printer ticket"; and only a
capitalised mention that is not a descriptive reference ("my cousin", "the client", "The printer ticket")
counts as a proper noun. A pronoun, a role or a kin word ("Client", "Manager", "Mom", "CEO", "He", "I",
"Someone") is never a name, and a single capitalised word that only starts a sentence ("Client called
...") is a name only if it is written with its capital inside a sentence too; so "Alice called." alone
does not name Alice when extraction gave nothing (the record is skipped, never invented).

**Nested presence on updates.** The presence question of a nested list is the weakest one ("Does `input`
mention any product line of `parent.label`?" scored 0.1991 for "two more of the blue mugs" next to an
order, under the 0.2 floor, and the line was never looked at). On an update, before the collection
concludes that there are no items, the parent's existing items are asked about with the record-presence
check (only those with TWO name words in the input, or one capitalised word of an input that has case;
everyday verbs such as "set" are no name word: "set the delivery date" does not revive "Wooden spoon set"),
and the items they bring back go on to matching and
the usual per-item update, where the relative-edit refusal can run.

**A name that is a sibling's is refused (`identity_bleed`).** The entity "blue mugs" came back from
extraction named "tea towels", the item next to it in the same sentence (live 6, example 07). A name
that fits none of the entity's own mentions but fits the mention of another entity of the same list and
scope is thrown away (`RejectedOp(reason="identity_bleed")`, a run warning) and the mention fallback names
the entity. For the items of a NESTED list, which are named by what they are, a noun phrase counts as a
name there ("the blue mugs" gives "blue mugs"); at the top level a descriptive reference still does not.

**A relative edit is refused (`verify.relative_edit`, default 0.5, 0 off).** "Please add two more of the
blue mugs" says how a quantity changes; the library does no arithmetic on a record's values, and writing
the stated 2 over the line's 3 loses data. For every `replace` of a whole number by another in a number field that is not only ever STATED (not
a price, cost, total, id, count to date, year or age, whatever the field is called: `units`, `count`,
`num_items` carried the data loss too when the check relied on the name), one Clef `noul` (kind `relative_edit`, the gate's model and state) asks it about THAT change:
"Does `input` only describe how the current value `proposed_change.from` of `proposed_change.field` of
`entity.label` changes (e.g. 'two more', 'one fewer', 'double it'), instead of giving `proposed_change.to`
as its new value?" ("two more of the blue mugs and the tea towels at 5" refuses the mugs, 3 to 2, and
writes the towels, 4 to 5); at the knob or above the operation is refused as
`relative_edit_unsupported` and nothing is written. The offline stand-ins answer from the change, not from
words anywhere in the input (`testing.states_relative_change`). It is never asked for an
`add` (a new line's quantity is attributed as a requested quantity above). The README case expects no
operation, and an operation that writes the 2 is a `must_not` violation.

## Update path (`One` schema with an existing record)

`ainvoke(text, existing=...)`: `_existing.py` coerces `existing` (five shapes, `"__any__"`,
`existing_schema_policy`), then each `One` schema that has a record runs `update_one`
(`_entity.py`); a schema without a record is extracted as an insert (`enable_inserts` decides about
inserts next to existing records of the same schema, which needs `Many` and matching, step 8).

**`existing` (`_existing.py`).** Shapes: `{schema_name: record}` (the id is the schema name);
`list[SchemaInstance]`; `list[(record_id, schema_name, record)]`; `list[(record_id, BaseModel)]`
(schema name = class name); bare `list[BaseModel]` (ids minted with `uuid4`). A record is a
dict or a model (dumped by alias, JSON mode, so `before`/`after`/patch values are
JSON-compatible); ids are non-empty strings, unique per schema. `existing_schema_policy`:
`True` raises `ValueError` for an unknown schema name; `False` keeps the record as opaque
context (never shown to a model, never patched, a warning, listed in
`ExtractionResult.untouched`); `"ignore"` drops it (debug log). `"__any__"` is always known; claims are ONE-TO-ONE. Among the
schemas without a record of their own (and without a claim yet) the presence of every candidate is
asked first, in parallel, and not asked again; only a schema whose presence reaches
`presence.threshold_absent` can claim (validating is never a way around presence: an all-optional
schema "validates" almost any record, and a record meant for a schema whose required field is not
held yet must not go to one the input is not about). Of the present candidates the only one whose
model validates the record claims it, else the most present one, with a warning naming the
claimant. Once a schema has claimed a record it leaves the candidate set (a `One` takes one
record; a `Many` takes one too, which then joins its existing records), a record nobody is left to claim stays untouched (listed in `untouched`, with a warning;
nothing raises after calls were spent), and a schema that loses a claim takes the insert path as if
no record existed, so the same text can update one schema's record and insert another schema's
record. Record keys are normalised to the keys the schema
validates (alias where there is one) at every level of sub-models and unions, so diffs,
patches and validation talk about the same keys. Cardinality: dict form → `One(S)`; an explicit
`One(S)` with exactly one list-form record of S → that record; several → `ValueError`
("use Many"); list-form records of S with no explicit wrapper imply `Many(S)`, which matches
its entities against them (steps 8-9 of "Collection engine").

**Pipeline** (`update_one`):
1. Presence gate as for inserts (skipped for a forced schema). Absent → no operation, no
   further call, the record id is in `untouched`.
2. `enable_updates=False` → `SkippedEntity(reason="updates_disabled")` (after presence, no
   extraction call); the record is neither updated nor untouched.
3. Clairic per group with the record rendered after the input as
   `<existing id="…" schema="…">{json}</existing>` (`RenderedState.existing_span`; only the
   input is ever narrowed or chunked, the record always follows whole). An answer whose span
   lies in the block is rejected `echo_span` (a list element individually; for chunked asks the
   chunk's echo is nulled before the merge). A second, input-only pass per group runs
   concurrently (kind `crosscheck`). If Clairic cuts the state inside the record while the whole
   input was read (or the record is at least half of what was read), the input is not chunked:
   the record is compacted to its identity and the fields being asked, with a `record_compacted`
   warning, and the request is repeated once.
4. Candidate changes (a non-null answer that differs from the record) are cross-checked;
   answers equal to the record are not. Where the primary answer has no span in the input (null, an echo
   of the record, or a span-less value equal to the record: numbers, dates) and the cross-check
   reads a usable different value, the cross-check value is taken (`adopted_from_crosscheck` in
   the field's warnings). Where the primary answer equals the record WITH a span in the input (a
   genuine reading, not an echo) the two are adjudicated instead (Clef `choice{a: primary, b:
   cross-check, neither}`, kind `adjudicate`): only a decisive `b` takes the cross-check value
   (`adjudicated_to_crosscheck`), anything else leaves the field as the record has it
   (`crosscheck_overruled`). In parallel, every Clef-typed answer
   (`bool`, `Literal`, `Enum`) that would change or add to the record is asked about with one
   Clef `noul` per group (call kind `stated`, "Does `input` state <field>?"): below
   `verify.clef_stated_min` (0.7) it is made a null (`not_stated`, reported `not_mentioned`),
   because such an answer always resolves, stated or not.
5. The verification ladder (low-confidence re-ask, borderline-null, attribution) runs on the
   *input only*: its re-asks never see the record. Fields equal to the record are skipped.
6. `diff_group` → operations; near-duplicate list elements → `list_dedup`; one gate request
   per remaining operation; sort; apply; validate.

**Operation shapes.** Leaf the record lacks (key absent or null) → `add /field`; leaf that
differs → `replace /field`; paths use the schema's validation keys. An *identity* field
(`plan.identity`, `s1_field(identity=True)`) is never replaced by a short form of the
existing value: a proper token subset ("Dana" for "Dana Whitfield"), or a diminutive, initials,
a prefix or a typo of it with no more words ("Liz" for "Elizabeth Smith"); a fuller name or a
reordering is not one: `identity_subset`, before any model is asked;
other identity edits go to the gate. A scalar list the record lacks → one `add /list` of the
deduplicated list; otherwise one `add /list/-` per new element (elements of the record that the
answer omits are left alone). A new element equal to a record element after folding case, punctuation and spacing
is a `duplicate` without asking. One that resembles a record element (difflib ratio ≥ 0.6)
goes to one Clef `noul` (kind `list_dedup`, "Is "new" the same fact as "old"?"), p ≥ 0.6 drops
it as `duplicate`; but a changed number or a toggled negation (not/no/never/n't/without ...)
makes a different fact: such an element is never dropped and not asked about. An answer
equal to the field's default is never *added* (the default already says it), and a new
sub-document leaves out default-equal values and Clef answers that failed the `stated` check. An optional
(or required) sub-model the input states while the record lacks it → ONE `add` of the whole
sub-document (gated once); one the record has is diffed leaf by leaf; one the input does not
state → no operation. A union whose chosen member is not the record's member (found by its
discriminator, else by validating) → ONE `replace` of the whole union value (an `add` when the
record has none); the member's single-value `Literal` tags are written out even when only
defaulted, so the value cannot validate as another member; the same member → leaf by leaf.
Removals exist only with `gates.allow_null_removals=True`: a silent field the record holds
becomes `replace /f null` (nullable), `remove /f` (has a default) or a `required_field`
rejection, and the gate asks "Does `input` state that `f` was removed or no longer applies?"
at `gates.remove_threshold`.

**Change gate (`_gates.py`, always on).** One Clef request per operation (kind `gate`, model
`gate_model`) with its own state `{"input", "entity": {"label", "mentions"}, "record",
"proposed_change": {"op", "path", "field", "from", "to"}}`; question "Does `input` state the
information in `proposed_change.to` about `entity.label`, as new or changed compared to
`record`?" (true: states it and the record does not hold it; false: only a shorter or partial
form, another value, somebody else, or a repeat of the record). Thresholds: add
`gates.add_threshold` 0.55, replace 0.6, remove or set-null 0.85. An operation applies only if
it survived attribution (the ladder) AND the gate; an attribution failure leaves no operation,
so the gate is not even asked. A narrowed Clef state cuts `input` only (never the record).

**Apply and validate.** `apply_ops` applies the operations one by one on a deep copy;
`_fix_string_concat` (an `add /f/-` whose `/f` is a string becomes a `replace` of the
concatenation) then `_fix_missing_parents` (missing ancestors added outermost first, `[]` before
an `-`/index step, `{}` otherwise; the parents carry `repaired_from`); an operation that still
fails is `apply_failed`. Which parents are lists comes from the plan (`list_paths`), not from digits in a key. The patched document is validated (alias-aware); each error location is mapped to the document
(`_diff.document_path(plan, loc, chosen)`: pydantic puts the union member between a union field and the
field below it, which is dropped, and errors of members the update did not choose are ignored); on failure the
operations whose paths lie on the line of an error (one path a prefix of the other;
`/list/-` stands for `/list`; a location-less model-wide error implicates none) are dropped
(`validation`) and the record is validated once more; still failing → `valid=False`,
`value=None` (`strict` raises `ExtractionValidationError`); each remaining error is also a
`RejectedOp(op=None, path, reason="validation")`. Before any operation is dropped the repair loop
(below) gets its turn: a failing patched record first re-asks the fields its own operations touched.

**Result.** `ResultRecord(kind="update", json_doc_id=<record id>, before=<normalised record>,
after=<patched document>, patches=<applied operations in applied order>, rejected, fields,
valid)`. A run that applies nothing is still an `update` record: `patches == ()`, `after ==
before`, `value` the validated record. `fields` carries `crosscheck_agree` and `gate_p`.
`untouched` lists existing ids that got neither a record nor a skipped entity (plus opaque
ones).

**`RejectedOp.reason` values.** Answers that never became an operation (`op=None`):
`not_mentioned`; answer level `p_null`, `low_confidence`, `low_choice_prob`, `abstain`,
`unstated`, `type_uncertain`, `missing_answer`, `unparseable`, `invalid`; ladder
`adjudication_failed`, `null_check`, `attribution`, `attribution_missing`, `none_of_these`
(list element); update checks `echo_span`, `echo_crosscheck`, `crosscheck_disagree`;
sub-models `invalid_optional_group`; nested collections `invalid_item` (an item that failed validation,
left out of the record; its path is the collection's), `missing_key` (a mapping entry with no
key; `duplicate` when it repeats one of the record or of an earlier new entry, the first is kept) and
`key_fixed` (an operation on the key of a mapping entry, which is its identity and is never renamed); `validation` for an error left after the drops. Computed operations that were thrown away (`op` is the
operation): `identity_subset`, `duplicate`, `required_field`, `gate_noul`, `gate_missing`
(no gate answer), `apply_failed`, `validation`; `identity_conflict` (a collection entity whose
extracted identity shares no name with the record it was matched to: every operation of the
update, see "Collection engine" step 8). Paths are JSON Pointers of the document.

## Repair loop (`_repair.py`)

Replaces trustcall's `PatchFunctionErrors`. It runs for inserts and updates of `One` schemas (the
multi-entity engine reuses it per entity) and is configured by `RepairConfig(enabled=True,
max_attempts=5, stop_on_repeat=True, allow_pattern_reask=True, keep_uncertain=None)`.

**Loop.** After assembly the candidate (insert) or the patched record (update) is validated with
`model_validate`. On success the record is done. On a `ValidationError`:

1. Each error `loc` becomes a document path (`_diff.document_path`: the union member segment is
   dropped, errors of members that were not chosen are ignored) and then the field of the plan it
   implicates, the longest known prefix (a list index stays with its list). A model-level error
   (`loc == ()`, a `model_validator`) implicates every leaf of that (sub-)model, or only the fields
   its message names (by name, key, spaced name or title); a `missing` error,
   and the type error that a null gets (`int_type`, `string_type`, ...: a required leaf with no
   value), implicates that leaf. Errors inside an optional sub-model or union member that was
   dropped as absent are repaired too, before the group is given up. An update re-asks only a
   field that one of its own operations touched, or a required value that is missing from the record
   (that one is re-asked input-only, attributed and gated like the rest). Any other error the
   record already had is not the update's to repair: that includes an error about a list element
   the record held before the update, which costs no request and does not drop the update's own
   operations.
2. ONE narrow request per implicated field, all in parallel (call kind `repair`, `CallMeta.field`
   the wire name): a Clairic request with that one field, whose `state` is the focus header, the
   `<input>` (input only: never the existing record, so a repair cannot echo it) and the blocks
   `<previous_attempt>` (just that field's rejected answer, never the patched document),
   `<validation_errors>` (pydantic message and type; on an insert also the input, only where it is
   the field's own scalar value at a leaf. For a model-level or container error pydantic's input is
   the whole document, and on an update even a leaf's input may be a value of the record, so an
   update shows no input at all) and `<expected_schema>` (the `$defs`-resolved JSON-schema excerpt of
   the path), or a Clef question when the field is adapted to Clef (its structured state carries
   the same three things as keys). The instructions are the field's rendered instructions plus
   "Must satisfy: {constraint}. The previous answer {old!r} was rejected because: {message}". An
   answer whose span lands in one of the three blocks (a copy of the rejected attempt) is not used.
3. What comes back goes through the full verification ladder (`run_ladder`: low-confidence re-ask,
   borderline null, attribution, one request for all fields of the round); a value the ladder
   throws away is not used and the field keeps its old (failing) value. On an update the repaired
   value then goes through the diff, the near-duplicate check and the change gate again (only the
   operations the new value changed are judged anew; a Clef answer also gets the `stated` check).
   The input-only cross-check is not needed: a repair state never contains the record.
4. One repair ROUND (every failing field, in parallel) is ONE attempt, like trustcall's fan-out of
   repairs. `ExtractionResult.attempts` is 1 plus the rounds of the record that needed the most,
   so up to `1 + max_attempts`. The error signature normalises list indexes (`/notes/2` and
   `/notes/3` are `/notes/*`).
5. Model-level errors repair minimally. A model-level error implicates every field of the model,
   or only the fields its message names (by name, key, spaced name or title; pydantic's "Value
   error, " / "Assertion failed, " prefix and quoted values are not part of the message for this).
   Answers to such an error are TENTATIVE: they stay only if the error that caused them went away
   (a direct fix of another field does not count). The answers to the fields a message NAMES are
   judged as a SET: all are kept when together they make the error disappear, else all are taken
   back, and never one by one (each was asked because the validator showed the old value wrong,
   so a set whose parts happen to validate separately is not split). Answers to an error that
   names no field are carried over to the next round, and when the record validates each is taken
   back in turn and stays back while the record still validates, until no single one can go; if
   the loop ends invalid they are all taken back. A revert never brings back a value that a message
   named and a repair replaced. Named answers are therefore never
   taken back one by one, even when the record would still validate without one: that would
   reinstate a value the validator showed wrong (an `end < start` error that reverts one date
   lands on 10 May to 10 May). Known, accepted trade-off: a named field that merely rode along
   (the caption of a range whose ends were wrong) keeps its re-asked value when that differs.
   On an update, a repair that comes back to the RECORD's value for a
   field whose operation was meant to change it is a failed repair: the field keeps the answer
   the operation came from and the operation is reported `RejectedOp(op, path, "validation")`.
   An error about a list element counts as one the record already held (no request, the update's
   operations are not dropped for it) only when the patched document's element at that path equals
   the record's and no applied operation wrote an ancestor of the path: after `replace /pet` the
   new pet's `toys/0` is the update's own, even though the record had some `pet/toys/0`.

| error type | re-asked as |
| --- | --- |
| `missing`, or a null that got a type error (a required field with no value) | the same field with `null_threshold=0.95` and "The input states this; quote it." |
| `int_parsing`, `float_parsing`, `decimal_parsing` (a field that was asked as text) | Clairic `number` (+`integer` for an `int`) |
| `date_parsing`, `datetime_parsing`, `time_parsing` | Clairic `date`, `datetime`, `time` |
| `enum`, `literal_error` | Clef `choice` over the allowed values (no Clairic; an `unstated` option when the field is optional) |
| `bool_parsing` | Clef `noul` |
| `greater_than*`, `less_than*`, `multiple_of` | the same type, the bounds in prose |
| `string_too_short/long`, `too_short/long` (lists) | the same type, the length budget in prose |
| `string_pattern_mismatch` and the other format errors (`url_*`, `uuid_*`, `ip_*`) | the pattern rule below |
| a model-level error | the same spec of every leaf of the model, the error in the state |
| anything else | the same spec, the error in the state |

**Pattern rule.** Clairic only quotes the input, so it cannot turn "Berlin 10115" into "10115" by
reformatting; it can only find another quote. A field that failed a `pattern` (or another format
check) is therefore re-asked ONCE, with the pattern in prose, and only if `allow_pattern_reask`
and a verbatim piece of the input could satisfy it: for a regex, any match of the pattern in the
input (its `^`/`$` anchors aside); for another format, a whitespace-separated token of the input
that the field's own validation accepts. The answer is accepted only if its span is in the input
and it meets the format. Otherwise the field fails fast: no request is made (or no second one),
the reason is in the field's warnings, and the other failing fields go on without it. Vocabularies
belong in a `Literal`, where Clef chooses among the allowed values and no error arises.

**An uncertain attribution does not lose the record (`keep_uncertain`).** A repaired answer goes
through the whole ladder, and the attribution check that threw the first answer away judges the second
one by the same question about the same value: a model that answers the same pick twice (Jev does: it is
deterministic over one request) gets the same 0.44, and the repair can never change the outcome. The
cost of that veto is not one field: a required field that is null leaves the record invalid, with every
other field it held. So, for an INSERT, a TEXT value for a field that was null stands when its
attribution, after the second look, is neither confirmed (`>= verify.attribution_keep`) nor refuted
(`< verify.attribution_drop`): it is kept with the field warning `attribution_uncertain: p=0.46 (kept: a
required field, the same value twice)`. Nothing is retuned (both bars are as they were); a refuted value
(below the drop bar) never stands. **Only a value that is the field's own** (`_may_settle` in
`_repair.py`): (1) the field is REQUIRED and not nullable (`spec.required`): an optional field that a
model validator implicated ("email or phone is required") is not a lost record, and the value found for
it is whoever's it happens to be (probe: "Dana Ruiz joined the call with Mark Lee. Mark's email is
mark@lee.example." gave Dana Mark's address); (2) the re-ask gave the SAME value as the first answer
(`Target.first`), not a new guess that nobody judged twice; (3) in a collection, the value's span is not
inside, or nearer to, a mention of another entity (`Ctx.others`) than to the entity's own mentions, and
a value without a span cannot be told. **The default depends on the provider (`keep_uncertain=None`,
"auto").** The setting is a TypeSafe/Jev behaviour: the doubt it answers is Jev's uncalibrated
attribution of a name or an id, which the other providers do not show, and the repair needs Jev to read
the same value again. So `Run.keeps_uncertain` (`_run.py`) is `repair.keep_uncertain` when it is a bool
(an explicit `True` or `False` always wins) and otherwise ON only when the provider that answers the
ATTRIBUTION (`clef_model`, kind `attribution`) AND the one that answers the repair and the isolated
re-ask (`clairic_model`, kinds `repair` and `reask`) all say yes: `prefers_keep_uncertain(transport,
model, kind)` (`transport.py`), a bool attribute or a `(model, kind)` method read defensively like
`max_questions`. `TypeSafeTransport` has it `True`, every other transport says nothing (`False`). The
transports that hold others answer for the call: `RoutingTransport` for the transport serving the model,
`KindRoutingTransport` for the route of `(kind, model)`, `FieldTypeRoutingTransport` for its `text` route
when the model is Clairic and for its default for Clef (a Clef request is never split),
`FallbackTransport` for its primary, and `CalibratedTransport`, the combos label and `RecordingTransport`
for what they wrap. So Jev reading the text while Cloudflare answers the attribution (or the other way
round) leaves it OFF, and so does a `by_kind` route that sends only the re-ask or only the attribution
elsewhere. A recording writes the decision itself on every call line (`keep_uncertain`, what the wrapped
transport said of that model and kind) and `CassetteTransport` replays it per model family and kind (the
provider it stands for, else the `inner` transport of a miss, else off, when nothing was recorded: never
a guess over every recorded provider). `RepairConfig(keep_uncertain=False)` restores the strict repair on
Jev. Text only: a quoted
span is certainly in the input, what is uncertain is whose it is; a number, a date or a time is derived
(a sum, "two more", a relative date, `09:30` read as `09:30:00`) and its attribution is the guard of
relative edits and table rows, so it is never kept this way. Updates are never kept this way either (the
old value stands there, nothing is lost). Live (78c647b, uncalibrated Jev): the patient name (0.44, then 0.46),
the product name (0.41), an invoice number (0.59, 0.57) and a contact name (0.65, 0.60) were values of exactly
this kind; an event name (0.38) and a lead's name (0.16) were refuted and are not kept.

**Terminal failure.** Updates: the operations on the paths that still fail are dropped
(`RejectedOp(reason="validation")`) and the record is validated once more (the M5 rule); a
model-level error whose message names fields drops the operations on just those fields
(`detail` is the message). A model-level error that names nothing drops nothing, the record is
invalid, and every operation of the failing document is reported `RejectedOp(op, path,
"validation")` next to the error of the record as a whole. Inserts:
an optional field that still fails goes back to its default (or null) with `RejectedOp(op=None,
path, reason="validation")` and the record is validated again; if a required field still fails the
record is `valid=False` (`value=None`, kept out of `responses`; it still appears in `records` as an
invalid `ResultRecord` AND as a skipped entity) with `SkippedEntity(reason=
"validation_failed", detail=<the errors>)`; `strict=True` raises `ExtractionValidationError` (with
the record and the call log, repair calls included). An optional sub-model or union member that
cannot be repaired is dropped as absent, as before (`invalid_optional_group`). With
`repair.enabled=False` none of the insert rules applies: the record is validated once and is
simply invalid when it fails, as before M6 (no optional drop, no `SkippedEntity`).

**Result.** Repair calls are in `calls` with kind `repair`. A field the loop replaced has
`FieldAnswer.repaired=True`; the history of every attempt on a field (`repair[1] (number): '34
years' -> 34`, or why it failed) is in its `warnings`, and an earlier rejection of the first answer
is noted there too. A repaired field that was a Clairic field and became a Clef choice reports
`engine="clef"`.

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
  decorrelated jitter, within a total `call_deadline` (default 300 s) per `run()` that also
  caps each attempt's timeout and raises a `TransportError` when the next wait or attempt would not fit; unwraps the REST `{success, result, errors, messages}` envelope and
  accepts bare objects; errors `AuthError`, `BadRequestError` (with CF `errors[]`),
  `RateLimitError`, `ServerError`, `ModelResponseError`.
- **Second Clef provider: `ExperientialTransport`** (Experiential Labs, `POST
  https://api.experientiallabs.ai/v1/systemone`, configurable `base_url`). Same Clef request
  body as for Cloudflare (`model` `clef` or `clef-flash`, `state`, `questions`; `images` is part of
  the same schema), `Authorization: Bearer <key>`, key from `api_key=` or
  `EXPERIENTIAL_API_KEY` (read on each `run`; missing is `AuthError`). It serves Clef only (any
  other model is a `BadRequestError` before anything is sent) and never sends `CallMeta`. A 2xx body
  is unwrapped (a bare `{model, answers, usage}`, or the same inside a `{success, result}` /
  `{data}` envelope; `success: false` or an `error` object is a `BadRequestError`), validated with
  `parse_clef` (`ModelResponseError` on a mismatch) and returned as the bare dict the Clef parsing
  already expects. Answers are read by question id (`answers["review"]`). **Observed
  live:** a 200 is the *bare* Clef response with no `success`/`result` envelope: `{id, model,
  answers, usage, provider}`, where `answers.<question id>` is the usual Clef answer
  (`{type: "choice", choice, confidence, probabilities}`), `usage` is `{input_tokens,
  output_tokens, cost, is_byok}` (`cost` is in USD, presumably) and `id` / `provider` are extra
  keys that the response models tolerate and that are passed through. Responses carry
  `x-request-id` and `x-correlation-id` headers (the latter is put in error messages so a charge
  can be disputed) and several `x-gateway-*` headers. The scrubbed body is
  `docs/schemas/experiential-clef-response.example.json`, which the tests use. An error is
  `{"error": {"message", "type", "param", "code"}}` (OpenAI style; a 401 is `invalid_key`), shown
  in the exception message. The envelope unwrapping stays as a defensive fallback.
- **Question limit per transport (`transport.max_questions`).** Experiential rejects more than 32
  questions per request (`HTTP 400: questions must map 1 through 32 ids to typed questions`, live 8: the
  36 pairs of a 9-mention pairwise coref failed the README scenario); Cloudflare takes 64. A transport
  says its limit with an attribute or a method `max_questions(model)` (Cloudflare 64, Experiential 32,
  the Decisions API 32 because it documents none, a `RoutingTransport` that of the transport serving the
  model, a `FallbackTransport` the smaller of primary and fallback when the fallback serves the model, a
  transport that says nothing 64: `transport.max_questions(transport, model)`). `Run.clef_batch(model)`
  is that number, and `Run.call_clef`, coreference (36 pairs go as 32 + 4), qualification, the sentence
  requests and the delete questions group their questions by it instead of 64. The published
  `clef-input.json` schema says 1 to 64 and stays the validator of what is sent.
- **Provider calibration.** Every threshold (attribution 0.75/0.4, gates, presence 0.2/0.5, qualify
  0.4, match accept 0.6, confirm 0.7) was tuned on Cloudflare's Clef, which stays the default. Clef on
  Experiential Labs calibrates differently (live 8): the presence of a line item for "two more of the
  blue mugs" scored 0.21 and its qualify questions 0.07 and 0.016 (so the line was never found), the
  attribution of "5 of the tea towels" 0.53 where Cloudflare gave 0.83, and a choice's `confidence`
  equals its top probability (Cloudflare gave 0.44 for a `__new__` at 0.83). Results therefore differ
  between providers, and a threshold that is right for one can be wrong for the other. Nothing is
  retuned for Experiential yet; where a rule depended on a provider-specific number (the sureness of
  `__new__`) it now depends on a probability (`match.new_final`).
- **Retry policy of `ExperientialTransport` (deliberately unlike Cloudflare's).** A request that
  may have reached the server is never retried: a timeout can leave a call that was run and
  charged with the result lost. Timeouts (read, write), read/write/protocol errors, a connection
  dropped after sending, 408 and every 5xx raise `UncertainOutcomeError(TransportError)` after
  exactly one request; its message says the outcome is unknown and may have been charged, and
  `status_code`/`body` are kept. The only retries are the ones that cannot double a charge:
  `ConnectError`/`ConnectTimeout` (nothing was sent; jittered backoff) and 429 *with*
  `Retry-After` (refused unprocessed; waits that long, at most `max_retry_after`), both bounded by
  `max_retries` (6) and `call_deadline` (300 s) and releasing the limiter slot while waiting. A
  re-send is also never made with less time left than an attempt normally gets
  (`min_attempt_budget`, default the attempt's read timeout, `0` turns it off): after 429 waits the
  budget can be nearly gone, and a request cut off after about 5 s instead of 60 is likely to have
  been run, and charged, with its result lost. The check runs before the wait (no point sleeping
  for a send that will not happen) and again before the send, and fails with a deadline
  `TransportError` that says the request was not sent. The first attempt always goes out. A
  429 without `Retry-After`, 401/403 (`AuthError`), other 4xx and 3xx (`BadRequestError`, with the
  error object in `cf_errors`), a pool timeout and a proxy error are not retried. The engine does
  not retry a failed call either, so an `UncertainOutcomeError` ends the invoke with the usage so
  far attached; whether to ask again is the caller's decision.
- **Routing.** `RoutingTransport(routes, default)` picks a transport by model: keys are `clairic`,
  `clef`, `clef-flash` or full ids; `clef` covers `clef-flash` unless that has its own key;
  `run` passes `model`, `payload`, `meta` and `timeout` through unchanged, so `CallMeta` still
  reaches fakes, recorders and AI Gateway metadata. It aggregates `sensitive_values()` and closes
  its members. `system_one_transport(clef=None | "cloudflare" | "experiential", cloudflare=,
  experiential=, concurrency=)` builds the usual setup: `"cloudflare"` is a plain
  `CloudflareTransport`, `"experiential"` is `RoutingTransport({"clef": ExperientialTransport()},
  default=CloudflareTransport())`. `None` reads the environment through `clef_provider_from_env()`:
  `TRUSTCALL_S1_CLEF_PROVIDER=cloudflare|experiential|decisions`, else `cloudflare` (an unknown value is a
  `ValueError`, raised when the extractor is created). **Opt-in only:** an `EXPERIENTIAL_API_KEY`
  in the environment does not move Clef, because that would send a user's text to another company
  because of a variable they set for something else; choose it with the provider variable, with
  `--clef-provider`, or by passing a transport. The same selection is the extractor's default
  transport, `examples/_common.py`, `tests/live/conftest.py` and the evals (`--clef-provider
  cloudflare|experiential`, recorded in `results.json` as `clef_provider`; shown in the report of a
  live run and of a replay, and a note is printed when the flag is given without `--live`).
  Cassettes are keyed by model id and payload, so they replay under either provider.
- **Decisions API as a fallback for Clef (`DecisionsTransport`, `FallbackTransport`).**
  `POST https://api.experientiallabs.ai/v1/decisions` (same key) takes `{model, input,
  questions: [{type, name, instructions, choices: [{value, description}]}]}` and answers `{model,
  answers: [{type, name, choice, probabilities: [{value, probability}], confidence}], usage
  {input_tokens, output_tokens, cost, ...}, provider}` (a live probe is saved as
  `docs/schemas/experiential-decisions-response.example.json`; `model` is
  `gpt-6-luna-decisions`, configurable). **Disclosure: the Decisions API is served by OpenAI
  through Experiential Labs (the probe's response says `"provider": "openai"`), so opting in
  sends the user's text to OpenAI as well as to Experiential Labs.** Only the `choice` question
  type has been observed, so `DecisionsTransport` (a subclass of `ExperientialTransport`: same
  key, limiter, deadline and **same no-retry policy**) sends every Clef question as one and
  translates the answer back to a Clef-shaped response that `parse_clef` accepts
  (`_decisions.py`, validated both ways: the Clef request, the Decisions request, the Decisions
  response and the Clef result).
  - `state` becomes `input`: a string as it is, an object as one **fenced** section per key
    (`<<<TOKEN "key">>>`, the value, `<<<END TOKEN>>>`; strings verbatim, other values as
    indented JSON; the key JSON-encoded). The questions refer to the keys (`input`, `entity`,
    `record`, `proposed_change`) and the user's text should stay unescaped, but a plain
    `## record` header could be forged by that text (a note that contains `\n## record\n...`
    would pass for the record); the fence token is a hash of the whole state, so it is
    reproducible, no text of the state contains it and its author cannot know it. Structured
    `instructions` are fenced the same way. An `entity` whose `mentions` is `[]` (the one record of
    a `One(Bill)`, labelled "the Bill", a description the input never uses) is shown with a note
    instead (`readable_state`): "the input is about this one entity and `label` only describes
    it", because a reader with only the text has nothing to read the label against (live: 0.4 and
    0.27 for an invoice number and total that the line states, Clef said 0.95).
  - `questions` become a list in the same order with `name` = the question id; `choice` criteria
    become `choices` (id kept, description a string or rendered JSON, none when null);
    **`noul`** becomes two choices `yes`/`no` and `noul = P(yes)`; **`score`** becomes one choice
    per level (`"0"`...) and `score` is the probability-weighted level with the legend from the
    rubric. Every answer carries `provider: "decisions"`.
  - Probabilities are checked: they must be non-negative and sum to 1 within 0.02, else they
    are normalised and the run gets a `decisions_probabilities_normalised` warning (all zero is a
    `ModelResponseError`), and `choice` must be the most probable option (a tie is fine), else
    `ModelResponseError`; the confidence reported cannot exceed the most probable option's
    probability (after normalising). **An answer without probabilities, or whose probabilities
    are all within 0.001 of 0 and 1 (the shape the live API sends; {yes: 0.9999, no: 0.0001} counts), is a
    hard pick**: it carries `"hard": true`
    and its probability and confidence are capped at `decisions_hard_cap` (default **0.8**,
    range (0.5, 1]; 1.0 gives it full weight), the rest sharing what is left. 0.8 sits below
    `gates.remove_threshold` (0.85) and above the add (0.55) and replace (0.6) gates and the
    attribution keep (0.75): **a bare GPT yes can never authorise a removal or a set-null by
    itself, but it still passes an add or a replace**, and a bare no is 0.2 (below the 0.4
    attribution drop). The cap protects removals only while `gates.remove_threshold` stays above it: a user
    who lowers `remove_threshold` below `decisions_hard_cap` (0.8) loses that protection. The warning counts
    hard picks. The same cap bounds every other gate above 0.8: `match.delete_threshold` (0.85) means
    a record is deleted only on a GRADED answer of 0.85 or more (live, Decisions primary: a graded
    0.98 deleted, saturated and graded 0.8 yes answers did not), by design. The hard picks of a
    PRIMARY Decisions Clef are the warning `clef_hard_picks: N Clef answer(s) from decisions were
    hard picks ...; by call kind: attribution 25, presence 12 ...` (`Run.hard_picks`,
    `Run.hard_kinds`), not `clef_fallback_used`, which says that a fallback answered a request
    and counts only those; a memoised answer is a fallback's "use" only when a fallback answered
    something in the run. **A `refusal` answer** (`{"type": "refusal", "name": "plan"}`, no
    `choice`; live on a `clef_fields` question) is no answer: the question is left out of the Clef
    response (every gate reads a missing answer as "no", never as a pass), `decisions_notes` carries
    `decisions_refusal: the model refused N question(s)...`, which becomes a run warning, and the
    request is memoised like any other. `decisions_refused` lists the ids, and a refused ATTRIBUTION is
    final (`Run.was_refused`): no isolated re-check and no second request, the value is rejected as
    `attribution_missing`. A refusal of a question that was not asked, or a choice-less
    answer that is not a refusal, is still a `ModelResponseError`. Graded probabilities (0.1/0.9, say) are
    the API's own numbers and are left alone; thresholds written for graded Clef answers (a
    re-ask band between 0.4 and 0.75) behave like yes/no on a fallback that mostly sends picks.
  - `images` are refused (`BadRequestError`) before sending, and `serves(model, payload)` is
    false for them (and for a payload that does not translate), so the primary's error is what
    comes out. Response keys: `provider: "decisions"`, `decisions_model`, `usage.cost` (the
    reported price, 0.0 in the probe).
- `FallbackTransport(primary, fallback, *, on={"unavailable"})`: when the primary raises an
  error that `on` selects, the same request goes to the fallback **once**, and the response comes
  back with `provider`, `fallback_from`, `fallback_error` and `served_model` set.
  - **The default is `unavailable` only: the primary certainly ran nothing.** The fallback is a
    second request to the same vendor, which is what "a timeout can leave a charged, unknown
    outcome; do not auto-retry" forbids when the outcome is unknown. `uncertain` and
    `bad_response` are explicit opt-ins: in code (`on=...`, `system_one_transport(
    clef_fallback_on=...)`), `TRUSTCALL_S1_CLEF_FALLBACK_ON=unavailable,uncertain,bad_response`
    (`clef_fallback_on_from_env()`, validated when the extractor is created) and `--clef-fallback-on`
    in the evals. (An earlier revision defaulted to all three; it no longer does.)
  - Categories, which **the primary classifies for its own errors** (`classify_failure(exc)`; the
    module function `classify_failure` is the default): `unavailable` = a refused request
    (`RateLimitError` after its bounded retries), a failed connection, a request that was not
    sent; `uncertain` = the request may have been run and charged (`UncertainOutcomeError`, a
    timeout, a dropped connection after sending, a 408, a 5xx); `bad_response` =
    `ModelResponseError`: the primary certainly ran and charged and its answer was unusable.
    `ExperientialTransport` raises the right type for each. `CloudflareTransport` retries
    timeouts, dropped connections, 408 and 5xx and raises the last failure, so after its retries
    a plain `TransportError` can be either a failed connection (`unavailable`) or a request that
    was sent and lost (`uncertain`). Only the last attempt is visible in the last error, so
    `CloudflareTransport.run` keeps a sticky `maybe_sent`: once any attempt failed after the
    request may have reached the server (a post-send httpx error, a 408, a 5xx) the final error,
    including a deadline or slot-timeout error, carries `maybe_sent=True` and classifies as
    `uncertain` whatever the last attempt was (a timeout followed by a refused connection, or by
    a 429, is a request that may have run); its `classify_failure` also looks at the error and
    its causes. So
    `system_one_transport(clef="cloudflare", clef_fallback="decisions")` with the default only
    falls back when nothing could have run (a 429, a failed connection), not after a read
    timeout or a 5xx. An unknown transport is classified by type (`ServerError` is `uncertain`).
    Exception classes may be listed in `on` too. **With `uncertain` the question can be paid for
    twice** (the primary may have charged and the fallback is charged too).
  - `AuthError` and `BadRequestError` never fall back (listing them, or `Exception`, is a
    `ValueError`), nor does a cancellation; `FallbackTransport(p, p)` is a `ValueError`. The
    primary and the fallback each see at most one request (plus a failed connection or a 429
    with `Retry-After` inside it). A fallback that cannot serve the request (`serves()`: not
    Clef, images, a payload that does not translate) is not tried and the primary's error comes
    out.
  - A fallback that fails too raises *its own* error (its `__cause__` is untouched) with the
    primary's in `primary_error`; a fallback cancelled in flight carries it too, and a fallback
    that answers returns the primary's error on `TransportResponse.primary_error`, so the
    engine logs the primary's call with its cost, request id, status and retries and adds a
    charged-but-unusable primary answer to `reported_cost_usd` (primary $0.003 plus fallback
    $0.001 is $0.004, each record with its request id). Transports
    mark the errors they raise with their `provider`. The engine logs **both calls**, each with
    its provider, whatever the outcome: the primary's failure and the fallback's answer, error
    or cancellation. So `unknown_outcomes` counts a primary timeout followed by a fallback
    timeout as 2, a primary timeout followed by a bad (charged) response as 1, and a primary
    timeout followed by a cancelled fallback as 2.
- **Accounting by the model that answered.** A fallback's calls are filed under the model that
  answered (`TransportResponse.served_model`: `gpt-6-luna-decisions`, also on the `CallRecord`
  and in `Usage.by_model`), never under Clef's: Clef's token price (`Prices`) is not applied to
  GPT tokens, the Decisions cost shows up in `reported_usd` (a model whose every call reported
  one counts as priced), and the tokens of the fallback do not feed the Clef truncation warning.
  A response that was charged (`usage.cost`) but could not be used is a failed call with
  `cost_usd` set, and its cost is added to `Usage.reported_cost_usd` (`Usage.add_spend`).
- **Provenance of the answers.** Every answer of the fallback carries `provider: "decisions"`
  (and `hard`) as extra keys, so the memo replays them with their provenance. `FieldAnswer.provider`
  is `decisions` for a field the fallback answered, with the notes `answered_by:decisions` and
  `hard_answer` in `warnings`; a Clairic field whose attribution check came from the fallback has
  `attribution_by:decisions` in its `warnings`. The change gates and the isolated re-asks are not
  marked per decision; the call log (`CallRecord.provider`, `fallback_from`) and the warning cover
  them. The warning `clef_fallback_used: N Clef request(s) were answered by the fallback (...)
  because <primary> could not answer them; H hard pick(s) (no or saturated probabilities); used for:
  <kind>:<Schema>.<field>, ...` lists up to 8 uses (also uses of a memoised answer), counts hard
  answers and never names an entity.
- Setup: `system_one_transport(clef=..., clef_fallback="decisions"|"none"|None,
  clef_fallback_on=None)` builds `RoutingTransport({"clef": FallbackTransport(<Experiential or
  Cloudflare>, DecisionsTransport(), on=...)}, default=CloudflareTransport())`; `None` reads
  `TRUSTCALL_S1_CLEF_FALLBACK=decisions|none` (`clef_fallback_from_env()`; unset is none, a bad
  value is a `ValueError` when the extractor is created). **Opt-in only**, like the provider: the
  fallback sends the text to Experiential Labs and OpenAI, so a key in the environment never
  enables it; that also applies when Clef itself stays on Cloudflare. The evals take
  `--clef-fallback decisions|none` and `--clef-fallback-on`, recorded as `clef_fallback`
  (`decisions (on unavailable)`).
- Two transports mean two limiters: `RoutingTransport` does not share one, so the in-flight caps
  (64 per transport, 32 per model) apply to Cloudflare and to Experiential separately, and a 429
  from one lowers only that provider's limit. Pass `concurrency=` to lower both.
- **Every call that may have been charged is on record.** A call that is cancelled (a sibling
  failed, or the caller gave up) after it may have been sent is logged by `Run._send` as a
  `CallRecord` with `error="cancelled after send; outcome unknown"`, for every transport, and then
  the cancellation goes on; a call cancelled while it still waited for a limiter slot is logged
  the same way (the engine cannot tell the difference, so it errs towards "may have been
  charged"). `CallRecord.request_id` and `TransportResponse.request_id` carry the server's
  correlation (else request) id, which is also put in the message of an error raised from a 2xx
  (`ModelResponseError`, `BadRequestError`) and of an HTTP error. The evals count
  `unknown_outcomes` (uncertain and cancelled calls) per case, also for a case that failed.
- **Reported cost.** `ClefUsage.cost` (optional) is the price Experiential reports per call.
  `Usage.reported_cost_usd` sums it (with `reported_requests`, the calls that reported one, per
  model and kind), `CallRecord.cost_usd` has it per call and `CostReport.reported_usd` sums it over
  the models all of whose calls reported a cost, which are then counted as priced (`complete`).
  It stays apart from `CostReport.usd`, which is estimated from `Prices`; where a model has both
  they are two measures of the same spend, not amounts to add. A cost that is negative, not
  finite or not a number counts as not reported.
- **Where the key can leak.** The key sits in the headers of the httpx request, which the httpx
  exception chained to an `UncertainOutcomeError` (`__cause__`) holds. The library's messages never
  contain it, but an error tracker that records chained exceptions or locals can send it: scrub
  `Authorization` headers and `xpl_` keys in its `before_send` hook (or use
  `trustcall_system_one.testing.scrub` on text).
- Concurrency: 64 in flight per transport per event loop (shared across invokes and batch),
  32 per model (each transport has its own limiter: with a `RoutingTransport` or a
  `FallbackTransport` the caps apply to every provider separately, so up to 64 per provider in
  all), adaptive (halve on 429, at most once per generation so a burst of 429s halves
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
  `clef`/`clef-flash` while the URL uses `@cf/cloudflare/...`. **Observed constraint, stricter
  than the published schema:** a Clairic `list` field takes no `null_threshold` (HTTP 422, error
  3030, "list fields don't take 'null_threshold'"; a list answer has no `p_null` either). The
  compiler never emits it for a list, `Ctx.clairic_field` leaves it out of re-asks, and
  `ClairicField` rejects it, so `FakeTransport` catches it offline. Every other type accepts it.
- Clef truncates long state **silently**, and a request over the provider's maximum is refused
  (Cloudflare: HTTP 422 `Request requires 69639 tokens; the maximum is 65529`). Every Clef
  request is therefore estimated before it is sent and kept within
  `Limits.clef_state_token_budget` (default 56,000 of 64K estimated tokens; the one knob), in ONE
  place, `Run.call_clef`, so no caller can forget it:
  * **The estimate** (`_render.estimate_request_tokens`) covers the state AND the questions, and
    counts word pieces, digits, symbols and collapsed runs of spaces or rules instead of
    characters over 3.5. It was calibrated on 524 recorded Clef requests of a live run over
    table-heavy spec sheets (`usage.input_tokens` against the estimate). Findings: dense
    markdown tables and prose are 2.4-2.9 characters per token, a space-padded table 5-6, so no
    single ratio fits both (the old `len(json)/3.5` was a median 0.82 of the real count on that
    run); and a choice question costs about 18 tokens per option **per question**, which a
    character ratio does not see at all: 16 questions over 250 clusters are 72,000 tokens of
    options. The old estimate saw 10,573 tokens in the request the provider counted as 65,227.
    The calibrated one was never below the provider's count over the 524 requests (min 1.02
    times it, median 1.24, max 1.52; includes a 1.1 safety factor and 10 extra tokens per option
    and question). The padded table is the 1.52: the estimate is deliberately conservative where
    it cannot tell, and a long dense document is narrowed a little earlier than it had to be.
  * **The guard** (`Run._fit_request`): a request over budget is first *split* by questions (up to
    3 requests, the state shared) when its questions are what is large, else the `input` is
    narrowed to windows around the spans the state cites (mentions, clusters, parent), with those
    spans re-based into the narrowed text (`rebase_spans`; spans outside every window are
    dropped, text and context stay). Only the keys that cite offsets into the input are read or
    rewritten (`OFFSET_KEYS`: `mentions`, `entity`, `entities`, `parent`, `clusters`, `a`, `b`,
    `candidates`); a `record`, `records` or `proposed_change` that happens to contain a key called
    `span` is never touched and the questions split to fit what is left; if even that does
    not fit, the request is dropped without being sent (`RequestTooLargeError`, treated like a
    refusal below). `Run.clef_state` (what the callers use to build a state) narrows with the same
    estimate, and `Run.clef_budget()` is the effective budget, the single place that combines the
    knob with the `state_token_budget` a transport may advertise (an attribute or a method, read
    through `transport.state_token_budget`, like `max_questions`; TypeSafe's 24,000 makes it
    `min(56k, 24k)`). Every call site uses it, none computes its own minimum.
  * **Coreference** (`_incremental`) builds its requests to the budget: the input is held to half
    of it (windows around the 16 mentions of the batch beyond that), the earlier entities offered
    are the longest *most recent* prefix that fits (`coref_options_capped` now also says "within
    the Clef request budget"), the batch is halved while fewer than 24 entities would fit, and
    with no room at all the batch's mentions simply start entities of their own. The pairwise
    questions keep a quarter of the budget for themselves.
  * **A request refused for its SIZE does not abort the invoke.** Contained are: HTTP 413, a
    `RequestTooLargeError` (a local refusal before anything is sent: the engine's own, or
    `TypeSafeTransport`'s when Jev's context cannot hold the request), and a 400/422 whose message is a recognised size refusal
    (Cloudflare's `Request requires N tokens; the maximum is M`, the Decisions API's
    `max_tokens_exceeded`, "maximum context length"). Every OTHER 400 or 422 raises: a malformed
    field (a 422 code 3030) is a bug that every request shares, and swallowing it would turn a
    run into a string of warnings. (There is no switch to contain all 4xx errors; if one is
    wanted it is a one-line change of `_errors.is_request_rejection`.) A contained request is
    logged, a `request_rejected` warning names the call (kind, scope, schema, entity, field) and
    the provider's message, and the call returns no answers, which every caller already handles
    as an unanswered question: a Clairic field resolves as `missing_answer`, a presence or
    qualify question as "unknown", a coreference pair or assignment as "not the same entity"
    unless the two mentions are the same words (they are then one entity, so a refused
    coreference does not insert `PN-0003` twice), a gate as "not a pass". The identical request is never
    sent again in the run (`Run.rejected`); a refused request of several questions is sent once
    more as two *smaller* requests (`clef_request_resplit`; only for these recognised size
    refusals); with `strict=True` the error is raised; and a **circuit breaker** raises the
    refusal once the same one (its message with the figures taken out) has been contained more
    than 8 times in a run (the halves of a resplit do not count), because a refusal that every
    request meets is not a one-off.
  * **What is never contained or resent**: anything that may have been run and charged. An
    `UncertainOutcomeError`, a `ServerError`, an error with `maybe_sent`, or one with a
    `partial_usage` raises, and so does an error that merely *carries* one: a
    `FallbackTransport` whose primary failed uncertain and whose fallback then answered 413 raises
    that 413 with the primary's error as `primary_error` (`_errors.may_have_run` looks through
    `primary_error`, `__cause__` and `__context__`, and the transport's own `classify_failure`
    is asked too), so neither half of a split is sent to the primary again. Also raised:
    `AuthError` (401/403), rate limits, other server errors, a 404/405.
  * The post-hoc check stays: a warning when `usage.input_tokens ≥ 0.97 × 65,536`.
- Usage per call into `CallRecord`s and a `Usage` total by model and by kind. `Prices`
  defaults to Clairic $0.10/M only; Clef's price isn't published, so `Prices.cost(usage)`
  returns a frozen `CostReport{usd, unpriced_models, complete}`: `usd` sums the priced models
  only, and `complete` is false (with the models listed) until the caller configures a price
  for every model that used input tokens.
- Recording and replay (`testing.py`, public). `RecordingTransport(inner, path, append=True)` wraps
  any transport and appends one JSON line per call: `{key, model, kind, scope, schema, field,
  request, response, status, duration_ms}`. A failed call is re-raised and, only with
  `record_errors=True`, also written with `response: null` and an `error` object; the default is
  not to, since a 429 recorded once would otherwise fail every later run of the cassette (the evals
  and smoke recorders use the default). `key = request_key(model, payload)` is a hash (sha256, 24 hex digits)
  of the model id and the canonical JSON of the payload (sorted keys, no whitespace), so equal
  requests always share a key and any change to a request is a new one. The transport never sees
  headers; as a second line of defence `testing.scrub` replaces by `<redacted>` the values of
  `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_AUTH_TOKEN` and
  `EXPERIENTIAL_API_KEY`, the account id, token and key given explicitly to the wrapped
  `CloudflareTransport` / `ExperientialTransport` / `RoutingTransport` (`sensitive_values()`), any
  `Bearer` token, any `xpl_[A-Za-z0-9_-]{16,}` key, and the id in an `accounts/<id>` URL path or a gateway `/v1/<id>/` path (an error message
  quotes the request URL). The evals scrub the case errors and warnings and the reports they write
  with the same function (tests: `test_recording.py` and `test_evals_harness.py` set the ids and
  tokens in the environment and explicitly, with and without a gateway, and grep the files). `CassetteTransport(path, *, strict=True, inner=None)` loads a file, a
  directory of `*.jsonl` or a list of them and replays by key; duplicate keys replay in recorded
  order, and a key with no recording left raises `CassetteMissError` (an `AssertionError`) naming
  the call kind, the field, the schema and the key. `strict=False` with an `inner` sends only the
  misses to `inner` (wrap it in a `RecordingTransport` to keep what comes back), and skips a
  recorded failure instead of replaying it. `hits`,
  `misses`, `forwarded`, `unused()` and `reset()` support tests and sweeps. `trace_to_cassette`
  converts a wire trace (`{url, http, request, response}`, bodies only) to a cassette: the model
  is the end of the URL, the REST envelope is removed, the kind is left empty (lookups use the key
  only). A cassette is a snapshot of the request bodies of one commit: a change to the library
  that alters a request makes its key miss. That is the point (it never replays a stale
  request as if it were current), and the fix is to record again (`make record`), not to loosen
  the match.
- Experimental, off: Workers AI queue mode (`queueRequest` + `request_id`) behind a live
  probe — the Clairic schema has no batch properties, so it may be unsupported.

### Mixing providers per call kind and field type

`RoutingTransport` routes by MODEL only, and `system_one_transport` / `TRUSTCALL_S1_PROVIDER`,
`TRUSTCALL_S1_CLEF_PROVIDER` and `TRUSTCALL_S1_CLEF_FALLBACK` choose whole providers. To find which provider
is best for which decision, three wrappers and a declarative spec (`trustcall_system_one.combos`) route
finer, and the evals sweep them. Everything here is explicit: naming a provider in a routing config or a spec
is the opt-in; no key in the environment ever routes text anywhere, and nothing reads
`TRUSTCALL_S1_PROVIDER` and the like when a spec decides.

- **`KindRoutingTransport(routes, default)`.** Every engine call carries `CallMeta.kind`. A key is a call
  kind (`attribution`, `gate`, `extract`...), a model (`clairic`, `clef`, `clef-flash`, full ids; `clef`
  covers `clef-flash` unless it has its own key) or `kind@model` (`repair@clairic`: `repair` and `raw` are
  asked of both models). **The most specific key wins: `kind@model`, then `kind`, then `model`, then
  `default`.** The call is passed on unchanged. A request the routed transport cannot serve (a Clairic
  call on the Decisions API, a `forecast` field on Jev) is refused by it locally, as with
  `RoutingTransport`; only a request with images moves, and only to a default that can serve it
  (`rerouted_from`, and the engine warns). Failures are never retried or resent: the error comes out
  tagged with the `provider` of the transport that raised it (`_routed_through`), and
  `classify_failure(exc)` asks THAT transport, so a `FallbackTransport` around the router, and the engine's
  containment of a size refusal (`Run._contains`), see the right category. The engine fits a Clef request
  before it knows its kind (`Run.clef_batch(model)`, `Run.clef_budget`), so **`max_questions` and
  `state_token_budget` are the smallest limits of the transports that may be asked the model** (a route of
  a Clairic-only kind does not narrow Clef and the other way round; a transport that does not serve the
  model is not counted). `serves(model, payload)` is true when every transport that may be picked can serve
  it; without a payload, when at least one can. `aclose`, `close` and `sensitive_values` cover every member
  once. `provider`/`served_model` provenance and `partial_usage` pass through untouched (the same error
  object).
- **`FieldTypeRoutingTransport(routes, default)`** splits a Clairic request by the `type` of its fields (the
  `ClairicField.type` literals, plus `list:text|number|date`, which beats `list`) and sends each partition to
  its routed transport; two types on one transport share a request, and a request whose fields all go to one
  transport is passed on unchanged (the same payload object). Every partition carries **the same `state`**
  (and images, `reference_date`, `options`), so each provider's spans index the same text and stay exact.
  The answers are merged **in field order**, `usage` is summed (token fields, `passes`, a reported `cost`),
  `truncated` is true when any partition's is, and `state_text` is the text the partitions share (the
  shortest when one read only the beginning: `reanchor` then maps the spans inside that prefix and drops
  the rest, and `truncated` makes the engine ask again on chunks; texts that are not prefixes of one another
  lose the spans of the partitions that disagree with the first, with a `field_type_routing` warning).
  **Accounting:** the merged body carries `routing.partitions` (provider, served model, field count, usage,
  retries, duration, status, request id of each part), and `Run.call_clairic` logs one `CallRecord` per
  partition under ITS provider and model (so `Usage.by_model` and the evals' cost estimate are per provider)
  instead of one for the sum; a replay of a recording carries it too, since it is in the body. Nothing is
  sent unless every partition `serves` its part; images that one partition cannot take send the request
  whole to the default (if it can). **A failure in one partition after another answered, or while another
  was in flight, is uncertain**, the same pattern as TypeSafe's split Clef request: the partitions run under
  `gather_all`, the error goes through `_blame` (a sibling that may have run is the error that is raised),
  carries `maybe_sent`, and, when a partition had answered, `partial_usage` = what the answered partitions
  used plus the failed one's own, so the engine neither contains it (`is_size_rejection` is false for a
  charged request) nor resends the siblings, and the engine logs the siblings' errors and cancellations.
  The error is a COPY of the failing one (an error object that another call may share, the stitch memo, is
  never annotated). **Classification** (`_classify_through`, used by both routers, mirroring
  `TypeSafeTransport.classify_failure`): `None` for `AuthError` and `BadRequestError` (they never fall
  back), `uncertain` for anything with `maybe_sent` or `partial_usage` whatever its type, so a
  `ModelResponseError` of one partition after another answered is not `bad_response` (a
  `FallbackTransport(router, backup, on={"bad_response"})` does not resend), and only then the transport
  that raised it. A merge that fails (a partition's body without answers) is annotated the same way with
  the sum of what every partition used, the reported cost included (it travels in `partial_usage.cost`).
  The failed call is logged as one record with the summed usage, under the failed partition's provider (a
  per-partition breakdown of a failure is not kept). A partition that returns no `state_text` while another
  does read the state as it was sent. `aclose` and `close` of every router and wrapper reach the
  innermost transports and close each once (`_innermost`: a transport shared by the routers of a nested
  stack is not closed twice).
- **`CalibratedTransport(inner, shifts, *, by_provider=None)`** applies a **monotone logit shift** to the
  probabilities of Clef answers of the listed call kinds, so a provider can be calibrated to the thresholds
  the engine was tuned on without changing them (see "Provider calibration" above: Experiential scored a
  line item at 0.21 where Cloudflare would not). `noul`: `sigmoid(logit(p) + shift)` (positive: yes is
  likelier). `choice` and `score`: the shift moves the logit of the leading option, the other options keep
  their proportions and share the rest (so the probabilities sum to 1 again; a positive shift makes the
  model surer of its pick, a negative one less sure), `choice` becomes the most probable option again (ties
  keep the original pick; a leader pushed below the runner-up hands the pick over), `confidence` follows (it
  stays equal to the top probability when it was, else it is shifted itself) and a score is recomputed as
  the probability-weighted level. **Left alone:** probabilities of exactly 0 or 1, an answer flagged `hard`
  (a bare pick of the Decisions API is capped at 0.8 on purpose so that it cannot authorise a removal; a
  calibration must not lift it past `gates.remove_threshold`), Clairic answers, anything malformed. The
  default is no shift: a kind without one, or with 0, returns the very same response object. A calibrated
  response gets the warning `calibration_applied: <kind> logit shift <+x>` (once per run, no text); a
  `by_provider` shift that cannot be looked up because a response names no provider says
  `calibration_skipped` instead of silently doing nothing. Shifts are limited to **+-3** (+3 turns 0.6 into
  0.95), keys are checked against `CallKind`. **A positive shift on a kind that can authorise a removal
  makes the engine remove more:** `gate` (a removal or set-null needs `gates.remove_threshold`),
  `null_check` (`verify.null_confirm_yes` sets a value to null) and `delete` (`match.delete_threshold`) are
  `combos.REMOVAL_KINDS`. The search leaves them out of the default `--calibrate` kinds, they have to be
  named in `--calibrate-kinds`, and even then only shifts of 0 or less are tried unless
  `--allow-positive-removal-shift` is given; `ComboSpec.removal_shifts()` lists such shifts, the leaderboard
  flags every combination that has one (`[removal shift: ...]` and a section), and `combo` says so before it
  runs. `by_provider` (response `provider` to kind to
  shift) lets one wrapper calibrate each provider of a router, and **the wrapper can sit outside a
  `RecordingTransport` or `CassetteTransport`** (they keep each provider's raw answer and its `provider`):
  a recording made under one calibration replays under another, which is what makes the calibration search
  cheap.
- **Combination specs (`ComboSpec`, `build_combo_transport`, `load_combo`).** A pydantic model (a dict, JSON,
  or YAML when PyYAML is installed): `extraction` (Clairic default provider), `extraction_by_type`,
  `decision` (Clef default provider), `by_kind` (kind or `kind@model` to provider, any kind), `calibration`
  (provider to kind to shift), `options` (the `create_extractor` knobs as `group.field` overrides, the same
  parser as the evals: `combos.extractor_options`), `only` and free `labels`. Providers are `cloudflare`,
  `experiential-clef`, `decisions` and `typesafe`. The stack, outside in: `CalibratedTransport` (only if the
  spec calibrates), `KindRoutingTransport` (`by_kind`, plus `clairic` and `clef` as the two defaults, and
  Cloudflare, if the spec names it, as the `default` that takes pictured requests), and
  `FieldTypeRoutingTransport` for Clairic (if `extraction_by_type` is not empty). One provider is the
  transport itself. Only the providers the spec names are created. `spec.canonical()` drops what the
  defaults already say (so `spec.key()` is equal for specs that route alike: the search relies on it).
  **Validation** refuses unknown providers, call kinds, field types, options and `kind@model` pairs that are
  never asked (`attribution@clairic`), and every combination in which a provider would be given what it does
  not serve: a Clef-only provider (`experiential-clef`, `decisions`) as `extraction`, for a field type, or
  for a Clairic kind (`extract`, `crosscheck`, `reask`, `discover`, `count`; `repair` and `raw` are both
  and must say `repair@clef`), Jev for `forecast` fields, and a calibration of a Clairic kind or of a
  provider the spec never uses. **The rule about mixing:** `system_one_transport(provider="typesafe")`
  refuses another Clef provider and a Clef fallback, because there they would send text somewhere the user
  did not name; a spec names every provider, so mixing is allowed and those two rules are not kept (and a
  spec has no fallback: nothing in it can send a request twice). What is kept is about safety and
  capability, plus a latch for the TypeSafe-only guarantee: with `"only": "typesafe"` (any provider) a spec
  that names another provider anywhere is refused, so a search over candidates can be made unable to send
  text elsewhere. `spec.companies()` lists who will see the text (`decisions` is OpenAI behind Experiential
  Labs), and the evals print it before a live run. Jev as the default `extraction` with `forecast` fields
  unmapped fails locally on a forecast request, as TypeSafe alone does.
- **Evals.** `python -m evals combo --spec spec.json [--live] [--record DIR] [--tag NAME] [--cases ...]
  [--case-tag ...] [--suite default|holdout] [--replay DIR]`: the scoring is unchanged, `results.json` gets
  `combo` (`spec` as given, `key`, `describe`, `labels` = the spec's labels plus the `extraction` /
  `decisions` labels, `companies`), `suite` and `tag`; `--record DIR` writes the cassettes to `DIR/NAME`
  (the tag, else the spec's name). Offline each provider is the case's scripted world (`Settings.members`
  lets a test make a provider better at one kind); live needs the credentials of the providers the spec
  names (`CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN`, `EXPERIENTIAL_API_KEY`, `TYPESAFE_API_KEY`; the
  error names the variables, never values). The calibration wraps the recorder and the cassette, as above.
  `--suite holdout` reads `evals/holdout_cases.py` (`CASES`); a missing file is a one-line error, not a
  traceback. `CaseResult` gained `output_tokens` and `by_provider` (requests, tokens and reported cost per
  provider, from the call log, failed calls included, and what the calls answered from a cassette used:
  `replayed_*`) and `aggregate` gained `op_f1` and `output_tokens`.
  **A replay is provider-aware.** `request_key` hashes the model and the payload only, so a cassette of one
  combination would answer the SAME request routed to ANOTHER provider (an attribution routed to the
  Decisions API answered with the recorded Cloudflare probabilities). In a combination every provider's
  transport is wrapped on its own (`build_combo_transport(wrap=...)`): a `RecordingTransport` that writes
  what THAT provider answered, then a `CassetteTransport(provider=...)` that replays only what that provider
  recorded (a recording by another provider, or by none, is a miss: `cassette miss ... made by another
  provider`). A miss fails its case, and goes to the provider only with `--live` (`results.json` has
  `replay`, the `mode` is `replay+live` and `forwarded` counts them). The calibration stays outermost, over
  the raw answers. `RecordingTransport` and `CassetteTransport` forward `serves`, `classify_failure` and
  `sensitive_values` to the transport they wrap (a router that asks what a provider can serve, for images,
  must get the provider's answer through the recorder); a replay with nothing behind it is wrapped by the
  runner in an adapter that answers those for the member it stands for. `Run`'s `gather` reads every
  task's error, not only the first, so no "Task exception was never retrieved" is logged when several
  chunks fail together. Cassettes recorded before this (one recorder over the whole stack) hold the merged
  Clairic responses and replay nothing for a combination: record again. The limits a replay uses
  (`max_questions`, `state_token_budget`) are the smallest recorded.
- **`python -m evals search`** (`evals/search.py`): a greedy coordinate search from `--base spec.json` over
  the dimensions of `--order` (default: presence/qualify/count, coref, match/match_confirm,
  attribution/stated/null_check, gate/novelty/list_dedup/relative_edit, clef_fields, verify
  (crosscheck/reask/adjudicate), then `extract:<type>` for each field type; `extract` and `other` can be
  named too). For each dimension it tries each candidate of `--providers` (applied to the kinds it can
  serve) from the CURRENT best and keeps the one that improves the objective: **more cases passed, then
  op F1, then value accuracy, then fewer bleed + wrong deletes; bleed or wrong deletes going up is a veto**
  whatever else improved; and **an improvement must beat the incumbent by a margin** (`--min-margin
  CASES[:F1]`, default one passed case, or, with as many passed, 0.02 of op F1 or of value accuracy; fewer
  bleed and wrong deletes with no loss needs none), so that a one-case swing within the noise of two live
  runs is not kept; `--repeats N` evaluates a candidate that wins narrowly (under two cases and 0.05 of F1)
  N times in all and keeps it only if every run beats the incumbent (each run is paid for).
  **Dev suite only** (`--suite holdout` is refused for `search`, and for `sweep`: check the winner with `combo
  --spec <out>/best.json --suite holdout`). `--max-runs` (new evaluations) and `--budget-usd` stop BETWEEN
  trials (the budget compares spent + the dearest trial so far; the base always runs; with `--live` it
  needs a `--price provider=IN:OUT` per candidate, USD per million tokens, since a budget that cannot be
  priced cannot bind). Cost, per provider: its tokens x its price when it has one, else the cost it
  reported (Experiential does), and `~` in the table whenever a provider that spent tokens has neither;
  calls answered from a cassette are not spend. **Nothing is retried:** a case
  that fails is a failed case with its error in the table, a trial that raises is an `error` row and the
  search goes on (the base's failure ends it: there is nothing to compare with), and a combination
  (`spec.key()`) is run once, also across `--resume DIR`, which reloads `search.json` (a different suite or
  case list is refused) and replays the same decisions from the cache. **A trial is reused only in the
  mode it ran in** (`offline`, `live`, `replay:<dir>`, `live+replay:<dir>`, stored on each trial): an
  offline search resumed with `--live` reuses nothing and runs the base for real. A trial that errored is
  reported again, not run again, unless `--retry-errors`. `--live` needs `--providers`
  (naming them is the opt-in). Output: `search.json`, `leaderboard.md` (every spec tried: result, pass,
  op F1, value accuracy, bleed, wrong deletes, requests, tokens, cost, seconds per case; the failures of the
  best; what was not run and why), `best.json` (a spec file), and `trials/NN-<change>/results.json`.
  **`--calibrate`** then grid-searches a logit shift per (provider, kind) (`--grid=-1.5:1.5:0.5`,
  `--calibrate-kinds`, the removal kinds as above), the provider being the one that answers the kind in
  the best spec; the first shift that improves the objective by the margin is kept. `--calibrate-only`
  (or `--order none --calibrate`) skips the routing dimensions (`--order ""` is an error, not "all").
  **`--replay` is accepted by the calibration search alone**: a routing search runs each combination for
  real, because a different routing is a different question. Replay the recording of the combination that
  is calibrated (`combo --record DIR --tag NAME`, then `--base <that spec> --calibrate-only --replay
  DIR/NAME`): the raw answers are replayed and only the requests that a shift changes go to a provider
  (with `--live`; otherwise they miss).
- **Left out on purpose:** a `FallbackTransport` in a spec; per-kind (not per-model) `max_questions` (the
  engine sizes a request before it knows its kind, so the smallest limit among candidates applies to all
  Clef requests of the stack: Decisions in the mix limits every Clef request to 32 questions and Jev to
  24k tokens of state); a per-partition breakdown of a failed split call; calibrating Clairic's `p_null`;
  a search over `options`; a PyYAML dependency (JSON needs nothing).
- **Gold that needs a report.** `Case.expect_skipped=[(reason, record id)]` (scored: `n_skipped`,
  `skipped_found`) makes a case whose right answer is "do nothing, and say why" fail for a combination that
  does nothing: `update_and_delete_conflict` expects `conflict_update_delete` for `acct-2`.
- **Tests:** `test_kind_routing.py`, `test_field_type_routing.py`, `test_calibration.py`, `test_combos.py`,
  `test_evals_combo.py` and `test_evals_review.py`. The search picks, from scripted providers that are each
  wrong about different kinds, the provider that is right for each; the veto, the margin and its repeats,
  the budget, `--max-runs`, resume by mode, no retry, the calibration grid and its removal kinds, the
  suite and its missing file; and the regressions of the review: a replay answers only from the provider
  it was routed to, classification and annotation of a partial failure, a failed merge, the closing of
  nested stacks, the cost of a replay.

## TypeSafe-only provider (`TypeSafeTransport`, `_typesafe.py`)

TypeSafe's **Jev** (`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`)
can replace BOTH Cloudflare models: Clef (decisions) and Clairic (extraction). It is an explicit
opt-in like every other provider (a `TYPESAFE_API_KEY` alone changes nothing): `TRUSTCALL_S1_PROVIDER=typesafe`
or `system_one_transport(provider="typesafe")`. No Cloudflare credential is needed, and nothing is
sent to any other company (a Clef provider other than Cloudflare's, or a Clef fallback, together
with the TypeSafe provider is a `ValueError`).

### What the live probes showed (scrubbed bodies in `docs/schemas/typesafe-*.example.json`)

* `GET /v1/models`: 200 `{models: [{name, description, release_date}]}` with `jev-latest` and
  `jev-preview` (the `release_date` is a full ISO timestamp, the docs say `YYYY-MM-DD`).
* One request with a `noul`, a `choice` and a `score` (`typesafe-systemone-response.example.json`):
  200, `{model, answers, usage: {input_tokens, output_tokens}}`. `model` answered `jev-1.14.0` for
  `jev-latest` (the models page says `jev-1.13.0`: the alias moves, log the answered model). Shapes
  are Clef's exactly: `{type: "noul", noul}`, `{type: "choice", choice, confidence, probabilities}`,
  `{type: "score", score, confidence, legend: {"0": ..}, probabilities: {"0": ..}}`. A score level is
  its position in `criteria`, `score` is the probability-weighted level. Response headers: only
  `x-typesafe-request-id` is TypeSafe's (no rate-limit headers, no `Retry-After` seen); about 0.2 to
  0.4 s per request.
* **The shipped selection scheme, probed last** (`typesafe-span-ids-request.example.json` and
  `typesafe-span-ids-response.example.json`, one request, 820 input and 252 output tokens, about
  0.4 s): a choice whose option ids are the spans themselves, `"(415) 555-0177"`,
  `"+1 415 555 0142"`, `"Dana Whit <dana.whit@acme-corp.com>"` (spaces, brackets, a plus sign and
  punctuation), one sentence id with a description (`"S001"`), an amount with a reading, and `none` last.
  For "the Person's mobile phone number to call first" Jev answered with the span, verbatim:
  `(415) 555-0177` **0.65**, `none` **0.28**, the other phone number 0.04, `S001` 0.02, every other
  option 0.00, `confidence` 0.60. For a field the input does not state (the fax number, in the same
  request) `none` was **1.0**, `confidence` 1.0. So a span with spaces and punctuation works as an option
  id, an answer comes back as the id itself (no escaping, no `candidate_N` indirection), and a missing
  value is found by selecting `none`. The word "Call" was among the options and got 0.00 (it is an
  instruction, not a name; the generators now drop such words). An earlier probe of a draft scheme
  (`candidate_N` ids and a `__none__` escape; yes/no questions per candidate answered 0.95 and 0.04)
  led to this one; its bodies are not kept because that scheme is not shipped.
* A malformed question is `422 {"detail": [{type, loc, msg, input, ctx}]}`
  (`typesafe-error-422.example.json`); `input` echoes the offending value, which may be the user's
  text, so the library's error message shows `loc` and `msg` only.
* Not probed: a question-count limit. The docs say there is none ("any number of questions is allowed as
  long as the request fits both limits": 64k tokens per request and 32k per question; 262k with
  `TypeSafe-Danger-Zone-Long-Context: true`). `TypeSafeTransport.max_questions` is 64, the bound of the
  Clef request schema the library builds (it chunks the Clef requests the engine makes); the transport
  itself splits by tokens alone (see below), since TypeSafe states no count.
* The OpenAPI document is saved as `docs/schemas/typesafe-openapi.json`; the tests validate every
  request and response of the scripted Jev against it.

### Clef: almost the identity

Jev's `noul`, `choice` and `score` are Clef's, so `to_jev_clef` keeps the state, the question ids
and their order and the questions as they are, and changes only `model` (`jev-latest`; `model=` or
`TYPESAFE_DEFAULT_MODEL` pins one, which matters because the thresholds depend on calibration).
`images` are refused (Jev reads text), and so a Clef request with images is not Jev's to answer
(`serves()` says so; see the Decisions routing below). The response is validated against the Clef
response schema (`parse_clef`) and against the questions: an unanswered question, an answer of another
type, an option or a level that was not offered is a `ModelResponseError` (the message names the
question, never the option: it can be the user's text).

*Jev's context is 64k tokens for the state and all the questions of a request and 32k for the state and
one question* (262k with `long_context=True`). Clef cut a long state silently; Jev answers `400
max_tokens_exceeded`. So the transport meets the limits itself, with the engine's own estimate
(`_render.estimate_request_tokens`, calibrated and conservative: the unit the budget it advertises
is in, so the engine and the transport never disagree about how large a request is; the limits are
60,000 for a request and 30,000 for the state and its widest question):

* it **advertises the state it takes**: `TypeSafeTransport.state_token_budget` (24,000; a
  `long_context` transport takes more; `None` for every other transport, which says nothing).
  `transport.state_token_budget(transport, model)` reads it (an int attribute or a method, like
  `max_questions`); `RoutingTransport` answers for the transport that serves the model,
  `FallbackTransport` with the smaller of its two (a fallback that does not `serve` the model is not
  counted, as for `max_questions`), and `RecordingTransport` and `CassetteTransport` forward it like
  `max_questions` (a cassette is replayed with the budget each model was recorded with, so the
  request keys still match). The engine has ONE effective budget, `Run.clef_budget(model)`:
  `min(Limits.clef_state_token_budget, transport budget)` (56,000 and 24,000 on TypeSafe, so 24,000),
  read through `transport.state_token_budget(...)` for the model the request goes to (a state
  built before the model is known fits the smaller budget of `clef_model` and `gate_model`). Every
  place that sizes a Clef request asks it (`Run.clef_state`, the request guard `Run._fit_request`,
  coreference, matching, sentences). The budget bounds the state AND the questions of one request,
  and 24,000 is below both Jev limits (30,000 for the state and its widest question, 60,000 for the
  request) in the SAME unit, so **a request the engine fitted passes the transport's checks
  untouched: Jev reads exactly the state the engine built**, spans and records included;
* **`to_jev_clef` splits and narrows as the last line of defence** (for a request that did not
  come through the engine's budget, or a smaller Jev context): questions go into as few requests
  as hold them (each with the whole state; the answers are merged in the request's order, the usage
  summed), and a state still too large for even one question has its `input` narrowed, with the
  engine's own machinery (`fit_clef_state`: windows around the spans the state cites and around the
  passages the questions quote, spans re-based with `rebase_spans` under `OFFSET_KEYS`; text nobody
  cites is cut from its end), so no request exceeds 32k per question or 64k per request. **Only
  `input` is ever narrowed**: a `record`, `records`, `proposed_change` or `existing` is the user's
  data and is never cut, and a narrowing is never silent (`TransportResponse.warnings`, recorded by the
  engine as a `typesafe_state_narrowed` warning; a cassette records the warnings and replays them,
  and one recorded before the field existed replays without any). A request that cannot be made to fit is refused
  locally with a `RequestTooLargeError` (a `BadRequestError`) and nothing is sent. The engine
  recognises it as a size refusal (`is_size_rejection`) and treats it as a rejected request: a
  `request_rejected` warning, the request's questions unanswered, the invoke goes on (see the
  Clef budget item above). Nothing was sent, so it is safe to contain; an error that says
  `maybe_sent` or carries a `partial_usage` is never contained, whatever its class. Refused so: a
  question that alone is as large as the context (30k estimated tokens), or a state that has
  nothing left to cut (no `input`, or the rest of the state is too large). When the refusal names
  ONE question that cannot fit (`question_id`), the engine refuses that question alone and sends its
  siblings together in one request, fitted again (`clef_request_isolated`), instead of halving the
  request over and over. That covers the engine's own refusal too, which is the one met at the real
  limits: a question estimated over `Run.clef_budget` (24,000 on TypeSafe) with nothing else in the
  request (`Run._fit_request`) never reaches the transport, so a question between 24k and Jev's 30k is
  isolated by the engine. Only a refusal made before anything was sent is treated so (nothing was
  charged), never one the server made (those are sent again in halves). A request that fits is one request
  with the questions as they are.
  If some parts of a split request answered and another failed, the error carries what the answered
  parts cost (`partial_usage`) and `maybe_sent`, so it is `uncertain` and a `FallbackTransport` does
  not re-ask the questions that were already charged. **Concurrent siblings.** The parts of a split
  Clef request and the batches of a Clairic stage go out together, so one may be refused (a 400
  `max_tokens_exceeded`) in the same moment that another times out, or is still in flight when the
  first failure is raised. `gather_all` therefore says what became of the others
  (`cancelled_siblings`: still running and cancelled; `sibling_errors`: failed too, even in the same
  tick), and `_blame` makes the raised error uncontainable whenever there is any: if a sibling's
  error says it may have run (`may_have_run`: an uncertain outcome, a 5xx, `maybe_sent`,
  `partial_usage`) THAT error is raised, with the refusal logged beside it; otherwise the error
  itself is raised with `maybe_sent`. The engine then never contains the refusal as a size
  rejection or resends the siblings' questions, and `Run._record_siblings` logs every sibling (the
  timed-out ones with their error, the cancelled ones as `cancelled after send; outcome unknown`),
  so a possible charge is never invisible. Only requests that were SENT count (`gather_all(in_flight=)`,
  fed by `_post(on_send=)` the moment a request leaves its limiter slot): a batch still queued for a
  slot sent nothing and is neither counted nor logged, so a call log has at most one row per request
  put on the wire. A caller's cancellation (or any non-`Exception`) is never turned into an
  `UncertainOutcomeError` by `_blame`: it is annotated and re-raised as it is. The sibling error that
  is raised is a copy (an error object is never annotated in place), and `_own_copy` drops these
  notes from the copy a waiter of a shared stitch gets. A call of a single request is unaffected:
  its 400 is still contained.

### Clairic: selection, following TypeSafe's extraction cookbooks

Jev answers closed questions and generates nothing ("select instead of generate"). So Clairic's
`fields` are served by code that finds candidates and Jev that picks, in four ways. All questions over
the same state go out in **one request** (the fan-out pattern), split **by a token budget only**
(`TypeSafeConfig.request_token_budget`, 40k estimated tokens): TypeSafe states no limit on the number of
questions, so the internal stitch, selection and verification requests carry as many as fit (hundreds of
list candidates, a stitch question per line) and are not capped at the 64 of a Clef request.

| Clairic field | cookbook | what happens |
| --- | --- | --- |
| `text`, `number`, `estimate` | Pre-parsed value extraction | code finds candidate spans (`_candidates.py`); one `choice` per field whose options are the spans themselves (`criteria = {span: None} + {"none": "None of these is the requested value."}`) |
| free `text`, notes (`list` of text) | Structure recovery | lines with ids, one request stitches hard-wrapped lines, options are block, sentence and clause ids of tagged blocks in the state (`_structure.py`) |
| `date`, `time`, `datetime`, `duration` | Date extraction | Jev picks the PARTS the text names; code resolves them against `reference_date` (`_dateparts.py`) |
| `list` | both | one `noul` per candidate: "is the `candidate` one of the items described by `field`?"; elements at or above 0.5 |
| every selected scalar | SDE cascade | verified by two `noul`s; a doubted field escalates |

**Pre-parsed value extraction** (cookbook: `find` + `pick`, "a regex finds the candidate values,
TypeSafe picks the one the question asks for, and code copies it verbatim"). Candidates, by field
type, always deduplicated by text (a repeated text is one option with up to three places) and in
document order (Jev leans toward the first option, `none` is last):

* `text`: email, URL, telephone (7 to 15 digits, not a date; no pattern runs across a line break, so
  a telephone number, a date or a time never joins two lines), `Label: value` values, quoted strings,
  identifiers with a letter (`INV-2291`), capitalised name sequences (with the parts of a name:
  "Emma Thompson", "Emma", "Thompson"; `of` and particles inside a name; sentence starters, the
  verbs that open an instruction ("Call", "Please", "Contact", "Send", ...) and weekday and month names
  dropped), amounts and numbers written as they are, and dates and times
  written as they are (not resolved);
* `number`/`estimate` (`integer`, `decimals` and, for an estimate, `min`/`max` constrain the values):
  numeric literals (grouping, `.5`, European `1.315,50`, signs), currency (`$`, `€`, `£`, `¥`, `₹`
  or a code, the reading shows `1315.5 USD`), percentages (`15%`, value 15), `5k`, `2.5 million`, number
  words (`twenty-one`, `two hundred and five`, `a dozen`, `half a dozen`); digits inside emails, URLs,
  telephone numbers, dates, times and identifiers are not quantities. **Sums**: only for a field
  whose instructions ask for a total ("total", "amount due", "including", ...), the sum of the
  products (`4 x 250`, `4 pallets at $250.00`) and amounts of one sentence or chained by `+`,
  offered as one option whose span runs from the first term to the last and whose description reads
  `1080 (the sum of 1000 + 80)`. That is the only arithmetic, and it is the one the Clairic
  examples in this repo need (the invoice total); the library does no other arithmetic, and a value
  that needs one (a difference, a percentage of a figure) is not offered. The cookbook's other
  advice, that Jev is bad at arithmetic, is why the sum is computed in code and offered.
* A span longer than 100 characters is not an option id by itself (a sentence is offered by its id).
* More than 254 candidates (a choice takes 255 options, one is `none`): chunks of 254 are asked in the same
  request (`<field>::pick0`, `<field>::pick1`, ...), the best candidate of each chunk goes to a final
  choice (`<field>::final`, a second request), which is the cookbook's "two stages: pick the section
  first, then the span inside it". Beyond 254 x 254 candidates the lowest-probability chunk winners are dropped.

The answer maps back as `value` (the span's text, or the number), `span` (the candidate's exact
offsets in the state that was sent: `state_text[s:e] == text` for a pattern candidate), `p_null` =
P(`none`), `confidence` = Jev's and `guess` = the best non-`none` candidate. **Options that say the
same thing share their odds**: the input offers `Hearthside Maple Granola` as a span, as the sentence
`S000` and as the block `B000`, and Jev spreads its probability over the three, so the best single option
holds a third of what the VALUE holds and Jev's `confidence` (the lead of the chosen option over the next
one, which is the same value) is low although nothing else was in doubt (live: 0.42 for a product name,
0.39 for a surname: under `min_confidence`, so a stated value was rejected and its record invalid). The
value is chosen among values: the odds of the options with the same value (spaces and case aside) are
added up, the best value's likeliest option gives the span, and `confidence` is the value's lead (its
pooled odds over the best DIFFERENT answer, `none` included, never below 0); Jev's own confidence is
about ITS choice, so it is a floor (`max(Jev's, the lead)`) only when the pooled value IS the value of
Jev's choice. A value Jev did not choose (a rival option, or `none`, was the single likeliest) has its
lead only: a number chosen at 0.35 with confidence 0.7 and a product name pooling 0.42 give the name
0.07, not 0.7. When the pooled odds of the best value exceed those of `none`, the value was chosen even
though `none` is the single likeliest option. **When Jev's choice is `none` the value is null, whatever the other
probabilities are** (unless the shapes of one value outweigh it, as above) (`none` 0.45 over a phone number at
0.40 is "not stated", and `p_null` is 0.45, so the ladder sees a doubt, not a decision), and the
best candidate is only the `guess` (so a null `value` whose `p_null` is *below* the
field's threshold means Jev chose `none` narrowly; `p_null` stays P(`none`) and is not raised to hide it).
When a candidate was chosen, `value` is null if `p_null` is above
the field's `null_threshold` (default 0.5), as Clairic's. No candidate at all is a confident null without
a request. The option ids are the spans (probe 6 above); the ids of structure (`B003`, `S007`) and the
description of an amount or a sum are the only non-span options.

**Structure recovery** (cookbook: two passes, "the model never generates text: it answers narrow
questions about the document, and code does the rendering"). Free text is read as blocks, not
as the lines it happened to be wrapped at:

1. *Lines* (code): non-blank lines with ids `L000`, exact offsets and a flag for a blank line before.
2. *Stitch* (one request, thresholds from the cookbook): one `noul` per adjacent pair without a blank
   line between, "Does line L015 pick up mid-sentence, continuing a sentence left unfinished at the end of
   line L014?", with the cookbook's criteria. The cookbook found that asking "same paragraph" merges
   every unmarked list (17 blocks against 12), so the question names the narrowest fact. A join
   probability of **0.2 or more merges after a dangling line** (no `.` `!` `?` `:` `;` at its end) and
   **0.5 or more after terminal punctuation** (the cookbook: true continuations score as low as 0.39,
   a list after a colon scores 0.22, so no single cutoff works). Both are `TypeSafeConfig` defaults to
   evaluate (they were tuned on one memo). The answers are remembered by the hash of the input, so the
   engine's many calls over one input stitch once, **also when they are concurrent** (the first call over an
   input asks and the others await the same future: single flight; a failure clears the entry so the next
   call asks again, and the calls that were waiting on it fail with it instead of sending the request
   again); a single-line input needs no request.
3. *Blocks and sentences* (code): merged lines are blocks `B003`; a block is split into sentences
   `S007` by the library's own splitter (`split_sentences`, pure code, now applied to a stitched
   block, where a hard wrap no longer splits a sentence), and a sentence into clauses `S007.1` and noun
   phrases `S007.p1` (determiner phrases, and phrases after a linking word: "as finance lead").
   Every id maps to an exact span of the original input (through the lines, so a sentence across a hard
   wrap still has the original start and end).
4. The state is `{focus, input}` with `input` the tagged blocks, one copy of the text:
   `B003| [S007] The old make targets keep working. [S008] The new entrypoint ...`. The options of
   a text field are the pattern candidates (their text is their id) and the block, sentence and
   clause ids with **short descriptions** (60 characters), never the long text again; a
   yes/no about a sentence names it by id (`{"id": "S007"}`), a clause or a name by its words and a snippet.
   A value that is a block or a sentence is the joined text (`value` has no line breaks, the span
   is the original text with them).

The cookbook's type-confidence review flag (underline a block whose type confidence is under 0.55)
has no counterpart: the library's own thresholds (`min_confidence`, the low-confidence re-ask)
play that part. This stitching replaces the line-based reading of the input in the TypeSafe path.
The library's own `sentences` list strategy (a Clef `noul` per sentence of its splitter) is not
triggered, because Jev's list answers are confident (>= 0.5 by construction) and so the
`auto` fallback does not fire; the sentence boundaries inside a block still come from the
splitter, since the cookbook recovers blocks only, not sentence ends.

**What a list of text holds** (found by the first live run of the TypeSafe-only provider: notes that
held "Berlin" and a phone number next to the sentence they came from, and a person whose label was a
whole clause). A `list` of text asks one more question in the same request as its members,
`<field>::kind` (the question the library asks of Clef for `list_strategy: auto`): are the items short
named values (names, tags, ids) or free-form statements? The answer has three states, by
`TypeSafeConfig.kind_min_prob` (0.5): Jev chose `statements` or `items` with at least that probability,
or the answer is **unclear** (a lower probability, a missing answer, an answer of the wrong type or with an
option that was not offered; none of these fails the call). A list of fixed choices (`list[Literal]`, "Each
item must be one of: ...") is not asked at all. What the accepted members may be:

* **statements** (notes): only whole sentences, clauses and blocks. A name, a number or a noun phrase is
  not a note, whether or not the sentence around it was accepted; a clause inside an accepted
  sentence is dropped as before.
* **items** (names, tags, mentions): a sentence or block that contains an accepted value is dropped, and so
  is a clause that is an **apposition** of one: it starts at the value and the value is followed by a comma, a
  copula or a pronoun ("Olivia Davis, she's a 27-year-old graphic designer", "Olivia Davis is a pilot"), so
  the label of an entity found by discovery is the name. A clause that names something through a value
  is kept ("Olivia Davis's husband", "the CEO of Acme", "Dana's manager"): those are other
  people. Noun phrases are never dropped.
* **unclear**: nothing is filtered, as before the question existed, and the answer carries
  `note: "list kind unclear: the members were not filtered"`.

Smaller repairs from the same run: a comma, colon or dash between digits (`$1,125.00`, `2:30`, `10-12`)
is not a clause boundary, and a **sum** is offered under an option id that says what it is
(`1080 (the sum of 1000 + 80)`, described as "the total of the amounts written in '4 pallets at $250.00
each, shipping $80.00'") with a sentence in the instructions that such an option is the total, where it had
been offered under the text of the sentence and Jev chose `none` for the total of an invoice. A candidate
with an id is not bound by the 100-character limit of an option text, so a total over several lines
is offered too (its span runs from the first amount to the last).

**Date extraction** (cookbook: "TypeSafe answers `Choice` questions about the date in one call,
code turns those answers into a `date`. The model reads what the text says and never does the
calendar math"). For each date field, nine choices go in the same request: the cookbook's seven, `mode` (`absolute`,
`relative`, `none`), `month`, `day`, `year` (1900 to 2050 plus `none` and `out_of_range`),
`day_anchor`, `weekday`, `week_offset`, in the cookbook's wording, and two of this library's for a date given as a
distance from today, `offset_count` (1 to 60, `none`, or `out_of_range` for a number above 60 or one that is not whole: flagged, never guessed) and `offset_unit` (`days_ago`, `days_ahead`, `weeks_ago`, ... `years_ahead`, `none`; "last week" and "next month" have no number and are `none`, not "ago, 1"),
read when `day_anchor` is `offset`. Code reads only the parts `mode`
calls for: a year-less date takes the reference year, or the next one when it is more than 31 days
past; "next Thursday" is the following calendar week, "this Thursday" this week, **"last Thursday" the
previous one (`week_offset` has `previous`), "yesterday" the day before (`day_anchor` has
`yesterday`)**, a bare weekday the next on or after the reference date; February 30 or a missing month is
"no date" (flagged, never guessed). A year or a week that the text did **not** state is inferred by
that code, and the answer says so: `needs_review` is true, `note` says "inferred year, not stated" (or
week) and the confidence is capped at 0.8 (`INFERRED_CONFIDENCE`), which is under the library's
low-confidence bar (0.85), so the ladder re-asks and cross-checks it instead of accepting it. A
distance from today ("3 days ago", "in two weeks") is resolved by code from the count and the unit; "last week"
and "next month" (no count) are not parts, so such a date is not resolved (flagged). **The confidence of the answer is the weakest confidence among the parts used**, `p_null`
is P(`mode` = `none`) when a date was resolved (for a `datetime` the larger of the date's and the
time's, since it is not stated if either part is not) and `max(P(none), 1 - confidence)` when it was not,
and the cookbook's review flag (no date, or confidence under **0.60**) is returned as `needs_review`
(and `note`) next to the answer. The same scheme reads `time` (`mode`, `hour` on the 24-hour clock,
`minute`), `datetime` (date parts and time parts; a date without a time is no datetime) and
`duration` (`mode` and a count per unit from years to seconds; the counts are every whole number from 1
to 120 and then 150, 180, 240, 300, 360, 480 and 600, `0.5` and `out_of_range`, so 61 to 89 are
there (an uncommon count above 120 is `out_of_range`: flagged, not guessed); the answer is the sum in the
field's `unit`, `integer` and `decimals` honoured). The reference date is the request's
`reference_date`, else the current UTC date. Dates carry no span, as Clairic's.
A `list` of dates finds the mentions by pattern, asks a yes/no for each, and asks the date
parts of every accepted mention in a second request.

**SDE cascade** (cookbook: extract with a cheap model, verify with narrow Jev `noul`s, escalate to a
reasoning model if a signal fires). The stages map onto the library's ladder like this:

* Stage 1, the cheap extractor, is the selection above.
* Stage 2, the verifier: for each selected scalar, two `noul`s in the cookbook's shape
  (`instructions = {field_spec, extracted_field, main_question}`, `criteria = {true, false}`, "bad =
  true"): `hallucinated` ("is the extracted_field unsupported by, or absent from, the source
  text?") and `off_target` ("does the source text fail to genuinely report the thing the field_spec
  describes, so the value was pulled from incidental text?"), two of the cookbook's seven heads, the
  two it gates on,
  in one request over `{source_text, extraction}`. P(wrong) is the max of the two (the cookbook's
  `any_flag` gate: one confident red flag is enough) and is returned as `p_wrong`. This is the
  adapter's own cheap check; the library's attribution (Clef `noul` per value), null check, cross-check
  and re-ask still run on top of it as for any provider, and nothing here replaces them. The cookbook's
  `absence_wrong` head for an empty field is the library's `stated`/null check, so it is not duplicated.
  The other heads are not asked: each is one more `noul` per field on every selection, and the
  library's own checks cover what they would. They could be added to the same request as further
  `noul`s under the same `any_flag` rule if the evals show they pay.
* Stage 3, escalation: P(wrong) above **0.7** (`TypeSafeConfig.fire_threshold`, the cookbook's
  `FIRE_T`) escalates the field. There is no reasoning model in TypeSafe-only mode, so by default
  the answer comes back as the library's ladder understands doubt: `p_null = max(p_null, P(wrong))`,
  `confidence = min(confidence, 1 - P(wrong))`, `value` null if `p_null` passes the null threshold, the
  selection kept as `guess`, and `escalated: "unavailable"`: the ladder then rejects it, or re-asks it
  in an isolated request on mention windows (low-confidence re-ask) and cross-checks. A user-supplied
  hook is optional: `TypeSafeTransport(escalate=callable)` gets an `EscalationRequest` (field, spec, input,
  focus, selected value, `p_wrong`, signals), may be sync or async, and returns `{"value": ..., "span":
  [start, end] (offsets in the input), "confidence", "p_null"}` or `None` (decline: the default above
  applies). Its answer is marked `escalated: "hook"` and is **validated**: a value of the wrong type for the field,
  a span that is not two integers inside the input (`0 <= start <= end <= len(input)`) or a
  confidence or `p_null` outside 0..1 is not used (`escalated: "hook_invalid"`), and a hook that raises
  is tagged `escalated: "hook_error"` and does not fail the call (the doubted field is returned as a
  null for the ladder, as when there is no hook). The hook is the place to call a reasoning
  model; it is called once per doubted field and its cost is the caller's. `verify_selection=False`
  turns stage 2 off.

### Known limits of selection

* **Jev can only select a value that is among the candidates.** A value that is not a span code
  finds is never returned: a bare noun ("engineer") that no phrase rule produces, a value that needs
  rewriting (a phone number to E.164, "Dana" expanded to a full name, a total that is not a sum of what
  the sentence states), a part of a word. The recall of the generators is the ceiling of the
  extraction, and they are English-oriented (number words, months, weekdays, stop words). Dates are
  the exception: Jev reads their parts, so a date written in another way or language can still be
  read.
* **Dates**: the parts are read in English and resolved in code, with limits worth knowing. A two-digit
  year is not resolved (the year choices are 1900 to 2050; "14/8/26" asks Jev to name a year and
  it may name one wrongly, the confidence is then the weakest part's). Day-first against month-first
  (`3/4/2026`) is Jev's reading of the text: nothing in code decides it, so a numeric date with both
  numbers up to 12 is the least reliable. Time zones are not read: a time is the clock time written,
  with no offset. The year-less date is biased to the future (the reference year, or the next if the
  date is more than 31 days past); that and a bare weekday are flagged as inferred
  (`needs_review`, confidence at most 0.8), but "August 14" in a document about last year still resolves
  to the coming one. "Last Thursday", "yesterday", "three days ago", "in two weeks" and "a month from
  now" are read (`day_anchor` `offset` with `offset_count` 1 to 60 and `offset_unit` `days_ago`,
  `weeks_ahead`, ... : the unit and the direction are one choice; a month or a year that lands past
  the end of a month stops on its last day); "last week", "next month", "a couple of days ago", "the end
  of the month" and "a week from Tuesday" are not. The attribution of a date field says what day it is
  (`Today is Monday, 2026-03-23.`): the day is `Run.today`, the `reference_date` or, without one, the UTC
  date pinned when the invoke began, and that one day is also what TypeSafe resolves a relative date
  against (`CallMeta.today`, never put in the Clairic request: that would change every Clairic request
  and cassette key). **Compatibility:** the attribution requests of date fields on EVERY provider now
  carry the note, so their request keys changed: a cassette recorded before this does not replay those
  requests strictly, and an invoke without a `reference_date` asks a different question each day, so a
  strict replay needs a `reference_date` (or a new recording).
  Live (78c647b): a date that was right and had been resolved from "yesterday" or "last Tuesday" scored
  0.03 to 0.13 on "does the input state this date?" because the question never said what day it was.
* Two entities whose names are the same text share candidates (dedup by text; the span is the first
  occurrence), and the focus can quote the entity's mentions but cannot tell which occurrence it means.
* **Tables.** A row of a table (`| PT-1042 | 09:30 | Room 3 | vaccination |`) and a tab-separated line
  are sentences whose cells are clauses (`S004.3`), so a cell is an option; before, only the whole row
  was (live: the room of a visit was `| PT-1041 | 09:00 | Room 2 | checkup |`). A cell of one character
  is not an option (a number cell is a numeric candidate anyway), a sentence offers at most 12 pieces
  (a table of more than 12 columns loses the last), and columns aligned with spaces only are not cells.
* **Initials.** `J. Okafor` is one name: the period that the mention cleaning takes off the initial is
  not what separates two mentions, and the period of an initial or a title (`J.`, `Dr.`) does not end a
  sentence (so `Okafor` after `J.` does not "only start a sentence"). A clause that continues a name
  after a spaced dash, a colon or a semicolon (`J. Okafor - buyer`) is an apposition of the name, like one
  after a comma, and is not a mention of its own. Before, discovery of `J. Okafor - buyer` left four
  fragments (`J`, `Okafor`, `M`, `Brandt`), the entity was labelled `J`, and no fallback could name it.
* **What this does not fix.** (a) Jev answers the library's attribution question about the NAME or id
  that identifies a record (the patient's full name, the event's name, an invoice number, the lead's
  name) at 0.16 to 0.65, against 0.9 for a stated amount or id of the same sentence; a name that is
  rejected below `attribution_drop` (0.4: an event name at 0.38) is lost, whatever the repair does. The
  question's wording was not changed, because it cannot be tested without calls; a different wording,
  or Jev's own verification (`p_wrong`) standing in for the attribution of a quoted text, is the
  experiment to run live. (b) A number, a date or a time of a table row whose attribution is in the
  middle band is still rejected (the row's identity is what that question guards). (c) A field whose
  description asks for more than the input states (`Amount due including shipping` for an invoice that
  says `totals $500.00`) is Jev's `none` at 0.46: a model choice. (d) A relative date that Jev reads but
  the attribution still doubts with the day given (not measured: there is no live run of this change).
* `count` fields ("how many Persons does the input mention?") find a numeral only if one is written:
  a count nobody states is a null (the count reconciliation of `Many` is then simply off). Lists of
  more than `max_list_candidates` (400) candidates drop the structural candidates first.
* Cost: every Clairic call is the stitch request (once per input), the selection request(s) and the
  verification request, each carrying the state again, plus the options; Clef calls are one request
  each. Output tokens are free and input is $0.042 per Mtok, so tokens are cheap and round trips (about
  0.2 to 0.4 s each) are the latency. `usage.passes` of a Clairic response is its number of Jev requests
  and `jev.stages` the count by stage (`stitch`, `select`, `refine`, `verify`).
* Calibration: the library's thresholds (attribution 0.75/0.4, the gates, `min_confidence` 0.5, presence,
  qualify, match) were tuned on Cloudflare's Clairic and Clef. Jev calibrates differently; a choice's
  `confidence` over 255 options is not Clairic's, and the cookbooks' numbers (0.2/0.5, 0.60, 0.7) were
  tuned on TypeSafe's own documents. Nothing is retuned: treat them as defaults to evaluate
  (`TypeSafeConfig`, and the sweeps in the runbook).
* Not served: `images` (BadRequestError), `forecast` fields (BadRequestError; `serves()` is false), an
  input of more than `max_input_chars` (60,000) is cut at a word, the response says `truncated: true`
  and `state_text` is the prefix that was read, so the library re-asks on windows as it does for
  Clairic; `long_context=True` raises the limits.
* **Accounting.** A call is accounted under the model that answered
  (`TransportResponse.served_model`, as for a fallback), here `jev-1.14.0`, so `Usage.by_model` shows
  Jev and its tokens are not priced as Clairic's. An error is logged under the served model when the
  failing request got one (an earlier request of the call did) and else under the alias (`jev-latest`):
  that is the model that was asked. A Clairic call that sent no Jev request (no candidate in the input)
  is recorded under the TypeSafe model (`jev-latest`) with 0 tokens and 0 passes. `Prices` has no
  default for Jev ($0.042 per Mtok input, output free): give `Prices({"jev-latest": 0.042})`, which
  prices every `jev-*` version that has no price of its own (the alias is the fallback for any
  `jev-1.x.y`), or a price per version; until then `cost.complete` is false. TypeSafe reports no cost
  field.
* **A call of several requests that fails part-way still cost something.** The error carries the tokens
  of the requests that had answered (`partial_usage`, a `ClairicUsage`) and the engine records them in
  the call log and the totals; requests that were cancelled while in flight may have been sent and are
  counted (`cancelled_siblings`, named in the message); and once any request went out the failure is
  classified `uncertain` whatever its last error was (a rate limit on the third request is not "nothing
  ran"). Each request is counted the moment it answers, not when its siblings are all done, so when one of
  26 parallel requests fails the 25 that answered are on the error. The module-level `classify_failure`
  honours `maybe_sent` / `partial_usage` on any error, so `RecordingTransport` or `RoutingTransport` around
  the TypeSafe transport do not hide it from a `FallbackTransport`, and the engine keeps a failed
  primary's `partial_usage` in the call log. `call_deadline` (300 s) bounds the whole Clairic call, not
  each request: a call that runs out of time is an `UncertainOutcomeError` (something may have gone out),
  with the tokens of the requests that had answered. Calls that share one stitch request each get their
  own copy of its failure, without the leader's charge or cancelled-sibling note.

### Transport, errors and retries

`TypeSafeTransport` is an `ExperientialTransport` with TypeSafe's endpoint, key (`TYPESAFE_API_KEY`;
`TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL` are the SDK's variables and are read too) and
errors: one HTTP client per loop, the adaptive limiter (one limit for both models, key `typesafe`), a
300 s call deadline. **A request that may have reached the server is never resent**: a timeout, a read
or write error, a dropped connection, a 408 or **any 5xx, 529 Overloaded included**, raises
`UncertainOutcomeError` (classified `uncertain` for `FallbackTransport`). A 529 means the server
was overloaded, not that the request was not processed: it may have run the model and may have charged, so
it is exactly as uncertain as a 503, never resent, and `max_retries` does not apply to it; it does
tell the adaptive limiter to slow down, as a 429 does. **Retried**: a
connect error (nothing was sent) with jittered backoff, and **`429 Too Many Requests` only**, which the
API documents as "retry after a short delay" and which means the request was refused before any
work: it honours `Retry-After` and `retry-after-ms`, otherwise backs off exponentially up to
`max_retries` (6), within the deadline; a wait longer than `max_retry_after` (60 s) fails at once; it
halves the limiter. Unlike the Experiential transport a 429 **without** `Retry-After` is retried too,
since TypeSafe says to back off and retry (`_retry_needs_hint = False`). After the retries a 429 is a
`RateLimitError` (`unavailable`: nothing ran).
401 and 403 are `AuthError`, 400, 413, 415, 422 and other 4xx `BadRequestError` (a 422 shows each
`detail` as `loc: msg`; **no user text is in an error message or in the body kept on the error**: `input` and
`ctx` are dropped, a `loc` is cut after `criteria` (the next key is an option id, which can be the
user's text), and `cf_errors` is the same cleaned list), a 2xx that is not a valid answer to its questions a
`ModelResponseError`. A Clairic call is several requests; the error is the failing request's, the
others are cancelled, and what the requests that had answered cost is on the error (see Accounting
above). The messages of `from_jev`, of an invalid request and of a response that does not parse
(`parse_clef`, `parse_clairic`: pydantic's text with its `input_value` is replaced by `loc: message`, the
location cut after `criteria` and `probabilities`, whose next keys are options) name the question and the
field, never an option or a value. `x-typesafe-request-id` is the request id.

### Selection

`system_one_transport(clef=None, *, provider=None, typesafe=None, ...)`: `provider="typesafe"` returns a
`TypeSafeTransport`; `provider_from_env()` reads `TRUSTCALL_S1_PROVIDER` (`cloudflare` default,
`typesafe`; anything else is a `ValueError` when the extractor is created); the extractor's default
transport, `examples/_common.py`, `tests/live/conftest.py` and the evals (`--provider typesafe`) use it. A
key in the environment alone never selects it. Cassettes are keyed by the engine's Clairic or Clef
request, so a recording made on TypeSafe replays under any provider and the other way round (the
Clairic answers are the adapter's).

**Repair and existing records.** A repair request (the library's `<previous_attempt>` and
`<validation_errors>` blocks after the input; the errors are the engine's `{field, message, type, input}`
objects) is served as one: the errors, shortened (`field: message`, at most three; the `input` an error may
show is never read, it can be the user's text), go into the pick instructions ("An earlier answer was
rejected (email: not allowed on this domain): it is not an option."), and the rejected answer is removed
from the options when an error names **this field** (a nested path such as `contacts.0.email` names
`email`), since offering it again would repeat the answer that failed. A **record-level** error (the
field is `(record)`: a model validator such as "discount must be below price") does not say which field
was wrong, so every candidate stays, including the previous answer, and the instructions only say that a
check on the whole record failed and the earlier answer may still be right. If nothing else is left (the only candidate was the rejected
one) the field is a null with no request, rather than an identical paid call. The `<existing>` record is
**not sent to Jev**: selection reads the input only (an echo of the stored value is caught by the
input-only cross-check, as for Clairic), so Jev cannot "confirm" a value from the record.

**The Decisions API as the primary for Clef.** `system_one_transport("decisions")` (or
`TRUSTCALL_S1_CLEF_PROVIDER=decisions`, or `--clef-provider decisions` in the evals) is
`RoutingTransport({"clef": DecisionsTransport()}, default=CloudflareTransport())`: the Decisions
API (`gpt-6-luna-decisions` via Experiential Labs, served by OpenAI) answers every Clef request, not
only after a failure. The same opt-in rule applies (an `EXPERIENTIAL_API_KEY` alone changes nothing), the
hard-pick cap of 0.8 applies (a bare yes cannot authorise a removal), the question limit is
`DecisionsTransport`'s 32, and it is a `ValueError` to also ask for the Decisions fallback. The Decisions API takes text only, so
**a Clef request with images goes to the default (Cloudflare)**: `RoutingTransport` asks the routed
transport's `serves(model, payload)` and hands such a request, and only a request with images, to the
default when it is another transport that can serve it (the Decisions API never sees the image). The
response says so (`TransportResponse.rerouted_from`) and the engine records a `request_rerouted` warning.
Any other request the routed transport cannot serve (a `forecast` field on Jev, an invalid
Decisions payload) is not moved: the routed transport refuses it locally, so the text never goes to
another company unasked.
So three combinations can be benchmarked: Cloudflare Clairic + Clef (default), Cloudflare Clairic + the
Decisions API for Clef, and TypeSafe Jev for both.

**Evals record the combination.** `SuiteResult` and `results.json` / `summary.md` carry `provider`,
`clef_provider`, `clef_fallback` and two labels for tabulating: `extraction` (`cloudflare clairic`,
`typesafe jev`, `scripted world`) and `decisions` (`cloudflare clef`, `experiential clef`,
`experiential decisions (gpt-6-luna-decisions, hard cap 0.8)`, `typesafe jev`, plus ` + fallback
decisions`). **A pure replay says what was recorded, not what would answer**: when nothing is sent the
labels are `replay (recorded on <label>)`, from the `provider` each recording carries (`unknown` for
a cassette from before the field existed). `--provider typesafe` together with `--clef-provider` or
`--clef-fallback` (or the `TRUSTCALL_S1_CLEF_PROVIDER` / `TRUSTCALL_S1_CLEF_FALLBACK` variables) is
refused, since Jev answers Clef itself; the same goes for `system_one_transport(provider="typesafe")` with an
explicit `cloudflare=`, `experiential=` or `decisions=` object.

**A TypeSafe object needs the provider.** `system_one_transport(typesafe=TypeSafeTransport(...))`
with the provider resolving to anything but `typesafe` is a `ValueError` (it would have returned a
`CloudflareTransport` and sent the text to Cloudflare while the caller thought it was on Jev): say
`provider="typesafe"` or `TRUSTCALL_S1_PROVIDER=typesafe`. The provider is never switched for you.

**Cassettes keep provenance.** A recording carries `provider` and `served_model` (and, for Clef, the
`state_token_budget` its states were narrowed to, as it carries `max_questions`) when the transport knows
them, and a replay returns them, so a replayed call is accounted under the model that answered. A
cassette recorded before these fields replays exactly as it did. Cassettes are still keyed by the
engine's request, so one recorded on any provider replays under any other.

### Scrubbing

`testing.scrub` replaces the exact value of `TYPESAFE_API_KEY` (like the other credentials), any
`apikey_` + 16 or more characters of `[A-Za-z0-9_-]`, and a Bearer header; `RecordingTransport` writes
no call whose request or response holds the key (an explicit `api_key=` too, through
`sensitive_values()`), so a cassette cannot hold it. Like the Experiential key it is inside the httpx
exception chained to an `UncertainOutcomeError`: scrub `Authorization` headers and `apikey_` keys in an
error tracker's `before_send`.

### Tests

`test_typesafe_candidates.py` (candidates per type with tables, exact spans, the structure and its
thresholds, the date parts), `test_typesafe_transport.py` (Clef both ways against `clef-*.json`, Clairic
answers against `clairic-output.json` and `parse_clairic`, errors, retries, the limits, verification and
the escalation hook), `test_typesafe_select.py` (selection by argument and environment, the contradictions,
the Decisions provider, scrubbing, the evals and their labels) and `test_typesafe_review.py` (the corrections of the review: 529, none-means-null, the state budget and a
Clef request split to Jev's limits, no user text in errors, cassette provenance, single-flight stitching,
repair, Decisions with images, partial usage and the deadline, the hook, the evals' refusals) and
`test_typesafe_world.py` (every eval
case end to end: `tests/unit/jev_mock.py` plays Jev from the case's `FakeWorld`, validating each request
against the OpenAPI schema, with the real transport and engine in between: `One` inserts, updates and
`Many` scenarios). All offline (`httpx.MockTransport`).

## Escalation tier (gpt-6-luna)

An **opt-in** tier that asks OpenAI's `gpt-6-luna` (Responses API) about the few cases in which the
default setup (Cloudflare Clairic for extraction, Clef for decisions) was observed to fail. **The text
of those cases goes to OpenAI.** It is off unless it is configured explicitly, and a key in the
environment never turns it on (the rule of every other provider here: an `OPENAI_API_KEY` set for
something else must not move a user's text to another company).

**What failed (a live run of Clairic + Clef on a 30-case held-out set, 14 cases):**

| group | symptom | handled by |
| --- | --- | --- |
| relative dates | "yesterday", "three days ago", "last Tuesday", "in two weeks" make a required date invalid, so the record is rejected | the coercion (code) and `relative_dates` |
| nested `list[Model]` rows, often markdown tables | the count says 1 for 3 rows and one row gets values; rows named by position codes (`ENG-1`, `NLG`) come out empty; an update to one row's fields is not written | `list_rows` |
| mid-band attribution | "does the input state that X of Y is V?" answers 0.4 to 0.75 and drops correct values (a phone number next to a name, a commitment spoken by a labelled speaker, a relationship in the speaker's voice) | `attribution_tiebreak` |
| dotted identity inserts | `Many(X, identity="identification.serial")` inserts nothing | `dotted_discovery` |

**Relative dates: the root cause, and what is fixed without the tier.** Clairic documents
`reference_date` as the "ISO date that anchors dates written *without a year*"
(`docs/schemas/clairic-input.json`); nothing in its contract says that it resolves a relative phrase,
and the date answer's `value` is untyped (`{}`) with no span. No recorded Clairic response in
`tests/recorded` has a date field, so what Clairic returns for "three days ago" cannot be read from the
repository: the fix covers what is certain. `_coerce._parse_date` accepts an ISO date or datetime
*string* and nothing else (a number is refused on purpose: pydantic would read it as a Unix
timestamp), so any other shape is `ok=False` ("is not an ISO date") and a required date field then
fails validation, whatever the repair re-ask says. The README and the docstrings claimed that
`reference_date` "anchors relative dates"; it does not, as far as the API says. **Fixed in code, no
model involved (`_relative.py`, `_coerce._relative`):** a date or datetime answer that is a phrase
naming ONE date whatever the convention (`today`, `tomorrow`, `yesterday`, the day after tomorrow
or before yesterday, `N days|weeks|fortnights|months|years ago|from now`, and `in N ...` where the
sentence is in the future tense; numbers as digits or words) is resolved against the invoke's `reference_date` (a datetime is that day at
midnight, noted `relative_date: resolved in code (midnight)`); without a reference date the answer stays
invalid (the code never uses today's date). A weekday phrase ("last Tuesday", "next Friday") is NOT
resolved in code: on a Friday "last Tuesday" can mean either of two dates, so the code does not pick.
`N days later` / `earlier` count from ANOTHER date the text names ("arrived on 1 March and was inspected two days
later"), not from today, so they are neither found nor resolved (`plausible` does not judge them either); and
`in N weeks` is a duration as often as a distance from today ("done in two weeks"), so it counts only where its
sentence says it is ahead (will, due, scheduled, expected, opens, ... or `from now`; `sentence_around`), in
`find_phrases` and in `resolve_phrase(context=...)`; the coercion has only the phrase, so a bare "in two weeks"
handed back by Clairic is not read. What is left (a null answer, a weekday phrase, any other shape) is the tier's.

**Never on its own word.** Every escalation result is a *proposal*. It goes through the same
verification ladder (attribution by Clef, and for updates the input-only pass is skipped for it, since
the model that failed is the Clairic pass) and the same change gates as an answer of the default setup;
a quoted value must occur verbatim in the input; and the tier never writes, deletes or overrides a
confident Clef answer. `prob_cap` (default 0.8, validated to lie in (0.5, 0.85) and below
`gates.remove_threshold` and `match.delete_threshold` when the extractor is made) is the most
probability an answer of the tier stands for: it can keep a value (the attribution keep bar is 0.75) or
pass an add or a replace gate, never authorise a removal, a set-null or a delete. A state the tier
proposed is never sent to the tier again (the tie-break would confirm itself).

### The transport: `OpenAIResponsesTransport` (`openai_transport.py`, public)

`POST {base_url}/v1/responses`, `Authorization: Bearer <key>` (key: `api_key=` or `OPENAI_API_KEY`, read
on each call; missing is `AuthError` before anything is sent; `base_url=` or `OPENAI_BASE_URL`, default
`https://api.openai.com`, a value ending `/v1` or `/v1/responses` is taken as it is). Body: `model`
(default `gpt-6-luna`), `reasoning: {"effort": ...}`, `input`, optional `instructions`, `text.format` a
strict `json_schema`, and `store: false` (the live probe did not send it and the response said
`store: true`: OpenAI keeps a response unless told not to, so the text is not left behind).

* **Reasoning effort** is a first-class setting: `none` (default), `low`, `medium`, `high`, `xhigh`,
  `max`, on the transport (`reasoning_effort=`), on `EscalationConfig.reasoning_effort` (it rides on the
  payload, so one transport serves several efforts), on `ComboSpec.escalate`, on the evals
  (`--escalate-effort`) and as a search dimension (`escalate.effort`). A higher effort thinks longer
  before the first byte, so the per-attempt read timeout scales with it (`EFFORT_TIMEOUTS`: 60, 120,
  180, 300, 600, 900 s; the effort is the EFFECTIVE one: a call's own, else the transport's; a `timeout=` given
  to the transport or to the call is never scaled or raised; the "auto" call deadline is 300 s or 2.5 times the
  read timeout of that effective effort, whichever is more, passed per call to `_post(call_deadline=...)`, so
  an `xhigh` or `max` call on a transport built at `none` is not cut at 300 s). **The retry rules do not change with the effort: a timeout after
  the request was sent is an uncertain outcome and is never retried.**
* **Response.** The answer is found by item type: the `message` item's `output_text` content (a
  `reasoning` item before it, present when the effort is above `none`, is ignored and its summary is
  never kept; the live probe showed `output[0]` as the message at effort `none`, `phase:
  "final_answer"`). The reply body is normalised to `{model, status, parsed, text, refusal, no_answer,
  logprobs, usage}`. A `refusal` content item, a `status` other than `completed` (`incomplete`,
  `failed`, or none stated) and a response without an `output_text` are **no answer** (`parsed: None`,
  `no_answer` says why): the response arrived and was charged, so it is not an uncertain outcome, nothing
  is retried, and the engine only warns (`escalation_no_answer`). Text that is not a JSON object under a
  strict schema is a `ModelResponseError` (its message never holds the text).
* **Usage.** `input_tokens`, `output_tokens` and `output_tokens_details.reasoning_tokens` become the
  call's usage with `provider="openai"` and the served model. OpenAI's `output_tokens` already include the
  reasoning tokens, so a higher effort is more output tokens, in `Usage.output_tokens`, in
  `Usage.reasoning_tokens` (a new counter, also in `CallRecord.usage`) and in the cost. **No default
  price**: without one `CallRecord.cost_usd` is `None` and the model is in `CostReport.unpriced_models`
  (`complete` is false); the caller passes `Prices` with an input price and, new, `per_million_output`
  (an output price needs the same model's input price) to get a cost; the evals' `--price
  openai=IN:OUT` does the same for the search.
* **Retries and errors (the library's rules, as `ExperientialTransport`).** A connect error (nothing was
  sent) and a 429 are retried, a 429 honouring `Retry-After`, else the `x-ratelimit-reset-*` hints
  (`6m0s`, `20ms`, `1.5s`; the reset of an exhausted resource per the `x-ratelimit-remaining-*`, else the
  shortest), else a jittered backoff; a wait over `max_retry_after` (60 s) fails at once with
  `RateLimitError`, and a 429 with `insufficient_quota` is not retried (waiting does not refill a quota).
  A timeout, a dropped connection, a 408 and any 5xx (a 529 too) raise `UncertainOutcomeError` after
  exactly ONE request. 401/403 are `AuthError`, other 4xx `BadRequestError`.
* **No user text in an error.** The message and the kept body hold the HTTP status, OpenAI's
  `error.type`, `error.code` and `error.param` (each only if it is a short identifier) and the request
  id, never `error.message`, which can quote the request (and a 401's quotes part of the key).
* **Limits.** `max_questions` is 8 (the transport serves no Clef; the number keeps what it advertises
  small), `state_token_budget` 100,000 tokens (the escalation skips a request whose input is longer, with
  a warning), `serves` is true for any model but Clairic and Clef, `classify_failure` is the library's,
  `sensitive_values()` holds an explicit key.
* **Logprobs** are opt-in (`top_logprobs` in the payload sends `include:
  ["message.output_text.logprobs"]` and returns the entries untouched) and the library does not use them:
  the live probe showed that under a strict schema constrained decoding masks the alternatives
  (`top_logprobs` holds only the chosen token), so P(yes) against P(no) cannot be computed. The tie-break
  uses the hard mapping below.

**Scrubbing.** `testing.scrub` replaces the value of `OPENAI_API_KEY` and any `sk-` key
(`sk-proj-...` too); `RecordingTransport` writes no call whose request or response holds an exact
credential (the key passed explicitly is in `sensitive_values()`), and a recording never holds a header.
The key sits in the httpx request that an `UncertainOutcomeError` chains (`__cause__`): scrub
`Authorization` headers and `sk-` keys in an error tracker.

### Configuration and wiring

`EscalationConfig(enabled=False, transport=None, kinds=<all four>, reasoning_effort=None, limits={...},
max_calls_per_invoke=20, prob_cap=0.8, mode="on_failure")` (`_config.py`): `limits` are per kind and invoke (defaults 8
dates, 4 row sets, 12 tie-breaks, 2 discoveries; 0 switches a kind off), `max_calls_per_invoke` counts
all kinds, a limit that is reached warns once (`escalation_limit`) and the call is not made. An enabled
config without a transport gets an `OpenAIResponsesTransport` from `OPENAI_API_KEY` on first use (owned,
closed with the extractor).

The opt-in, in order of precedence: `create_extractor(escalation=EscalationConfig(enabled=True))`; the
tier a transport carries (`system_one_transport(escalate="openai")` returns an `EscalatingTransport`, a thin
wrapper around the transport it builds or was given that holds `.escalation`; a transport that was passed in is
NEVER changed, so the opt-in cannot stick on the caller's object and reach a later extractor; `escalate="none"`
adds nothing; every wrapper (`RecordingTransport`, `CassetteTransport`, `FallbackTransport`, the routers,
`CalibratedTransport`) forwards `.escalation` of what it wraps, `transport.escalation_of`; it is a `ValueError`
with `provider="typesafe"` (TypeSafe only) and an `openai=` transport without `escalate=` is one too); for the
default transport only, `TRUSTCALL_S1_ESCALATE=openai` (and `TRUSTCALL_S1_ESCALATE_EFFORT`; a bad value fails when
the extractor is made; **with `TRUSTCALL_S1_PROVIDER=typesafe` it is a `ValueError` too**, as is an explicit
`create_extractor(escalation=...)` on the default transport, and the evals' `run_suite` with provider typesafe and
the tier on: TypeSafe only means nothing goes to OpenAI); `ComboSpec.escalate`
(`"openai"` or `{"provider": "openai", "model", "reasoning_effort", "kinds", "limits",
"max_calls_per_invoke", "prob_cap"}`; part of `spec.key()` and `describe()`, `spec.companies()` adds
OpenAI, `only` refuses it, `by_kind` and `calibration` refuse the `escalate` kind); evals `--escalate
openai --escalate-effort EFFORT`.

**Mode.** `EscalationConfig.mode` says WHEN each kind is asked. For one kind it is `"on_failure"` (default:
everything above) or `"route"`: the three classes in `ROUTED_KINDS` go to the tier FIRST, whatever the default
setup would have answered. (1) A date or datetime field whose mention window holds a relative phrase (needs a
`reference_date`) is asked of the tier (`_escalate._relative_date`). The phrase must lie in the entity's windows,
and where it occurs more than once the occurrence nearest the entity's mentions is the one checked (`_check_date`).
A verified date that differs from Clairic's takes its place **with Clairic's answer kept as a backup**
(`FieldState.backup`): if the ladder refuses the proposal, `run_ladder` (`_restore_backups`) restores the backup
and judges it normally, so asking first can never leave null a date Clairic read ("Completed on 2026-03-01; the
part was ordered three days ago": the tier's date for the part is refused, the 1 March stands). (2) Every nested
`list[Model]`: `collect` asks the rows before Clairic's discovery (`route_discovery`; the quotes become the mentions
and are not code-screened, since a row code such as `A1` is a verified quote), in the parent's part of the input
(windows around the parent's mentions; the whole input for a `One`), and `_fill_row` puts the tier's reading of a
row's fields in place of Clairic's where they differ (with the same backup). Clairic's COUNT is still asked: if it
exceeds the rows the tier gave (a partial answer), Clairic's discovery runs too and its mentions join the tier's
(`escalation_routed_partial`), as a broadened discovery does. (3) The discovery of a top-level `Many` with a dotted
identity, the same way (count; partial answers completed), with the routed labels put through the junk screen
(`_prefilter`: a label such as "Aircraft" names nothing and is dropped; if nothing survives Clairic discovers).
Unchanged: every quote occurs verbatim, the value is coerced, the ladder and the change gates judge it (Clef has to
agree), an answer stands for at most `prob_cap`, nothing is deleted or removed on its word. An answer that is off,
over a limit, unverifiable or failed leaves the normal path exactly as `on_failure` would run it (Clairic discovers
and counts).

**Verification of quotes (`_escalate.locate`, `_verify_rows`, `_pair_entities`).** A quote must occur as WHOLE
WORDS (white space folded): `ENG-1` is not in `ENG-10`, `2` is not in `B2` or `SN-1002`; no two rows (or entities)
share an occurrence of their identity. A row's identity is the occurrence whose line or sentence (`segment_around`)
holds most of its values; a value must lie in that line or sentence. Only an identity that stands alone on its line
(a heading) may have values on the lines after it, within its block of lines (no blank line) and `window_chars`, and
never in the line of another row's identity: a value quoted from another row is not this row's. `_row_of` prefers
the row whose span EQUALS a mention's over one that overlaps it (then the longest overlap). For dotted discovery the
identity is located first (one entity per identity value), and the label is the free occurrence nearest to it in its
line, else its block of lines, else within `window_chars`; pairs that cross (entities appear in the input in one
order, so a label paired with another entity's identity breaks it) are refused. The entity's mention is the label (the
verified identity pairs it but is not a second mention: coreference and qualification would take a bare serial for
another entity).

*Mode by kind.* `mode` is a string or a mapping (`_config.resolve_escalation_mode`; stored on the config as a
frozen mapping of all four kinds, read with `EscalationConfig.mode_for(kind)`, `routes(kind)` is `runs(kind)`
and `mode_for(kind) == "route"`). A plain string means the same mode for every kind: `"on_failure"` is all
`on_failure`, `"route"` routes every kind in `ROUTED_KINDS`. A mapping names kinds
(`{"relative_dates": "route", "list_rows": "route", "dotted_discovery": "on_failure"}`); a kind it leaves out is
`on_failure`, an unknown kind or mode is a `ValueError`. `attribution_tiebreak` is asked after the second look by
nature and can only be `on_failure`: `route` for it is refused, in the mapping, the spec, the text forms and the
CLI. The preset `"recommended"` (`ESCALATION_MODE_PRESETS`) is exactly
`{relative_dates: route, list_rows: route, dotted_discovery: on_failure, attribution_tiebreak: on_failure}`: dates
and table rows are asked of the tier first, dotted discovery only where Clairic finds nothing.
**The recommended mapping was chosen from the holdout2 results**, so holdout2 is no longer a clean
measurement for that choice: its score under `recommended` is a selected number, and a fresh blind suite is
needed to say how well the choice generalises. The tie-break is asked in every mode.

Spelling: `EscalateSpec.mode` (the combo spec's `escalate`) takes the string or the mapping
(`{"escalate": {"mode": "recommended"}}`, `{"escalate": {"mode": {"list_rows": "route"}}}`; `describe()` names it,
and an equal mapping is labelled as the preset); evals `--escalate-mode on_failure|route|recommended|kind=mode,kind=mode`
and `TRUSTCALL_S1_ESCALATE_MODE` take the same text (`_config.parse_escalation_mode`: case and blanks do not
matter, a kind left out is `on_failure`, a malformed pair, a repeated kind or `attribution_tiebreak=route` is an
error naming the variable or flag). The default stays `"on_failure"`, and the tier itself stays opt-in: a mode
alone (the variable, the flag, the spec) enables nothing and nothing goes to OpenAI unless the tier is turned on.
Tests: `test_escalation_route.py`.

Every call is a `CallRecord` of the new `CallKind` `"escalate"` (never routed through the Clairic/Clef
stack: `ESCALATE_KINDS`, outside `CLEF_KINDS`), `CallRecord.field` starts with the sub-kind
(`relative_dates:done_on`, `list_rows`, `attribution_tiebreak:phone`, `dotted_discovery`), `provider`
is `openai`, `Usage.by_kind["escalate"]` and `by_model["gpt-6-luna"]` carry its tokens, and each use is a
warning `escalation:<kind>: ...` (field paths and counts, no text). **Failure policy:** the tier is an
optional improvement, so a call that fails in a way that says nothing about the request (a timeout, a
rate limit that did not clear, an uncertain outcome, a refusal) leaves the default setup's result, with
`escalation_failed` / `escalation_no_answer` in the warnings and the failed call in the log (never sent
again); a rejected key (`AuthError`), a malformed request and every failure under `strict=True` raise.

### The four kinds

| kind | trigger | prompt | verification | tests |
| --- | --- | --- | --- | --- |
| `relative_dates` | a date/datetime field (`_verify.FieldState`) with no usable value (null, unparseable, missing answer) whose entity's mention windows (the whole input for a `One`) hold a relative phrase (`_relative.find_phrases`), with a `reference_date` | the text of the windows, the reference date with its weekday, the field's description; strict `{phrase, iso_date, time}`; `time` only for a datetime | the phrase occurs verbatim in a window and holds a relative phrase; `iso_date` is `YYYY-MM-DD` and a real date; `_relative.plausible`: a phrase that names one date must equal it, a weekday phrase must fall on that weekday on the right side of today within 14 days; a datetime needs `HH:MM`; then the state is marked `escalated`, goes through attribution (with the "Today is ..." note) and the gates | `test_escalation.py` (reproduced off, fixed on, 7 rejections, no call without a phrase or a reference date, no override of a confident answer, datetime) |
| `list_rows` | (1) a nested list with no mention while presence says there are some, or a count above the rows found (`collect`): the verified identities join the mentions (a broadened discovery); (2) a nested row that came back empty, or without a required field (insert), or for which nothing at all was proposed (update) (`escalate_states`) | the text, the item fields with their descriptions; strict `{rows: [{identity, fields: {name: quote or null}}]}`, ONE call per list and invoke (single-flight) | the identity occurs verbatim; every value occurs verbatim (near the identity), and is coerced like a Clairic answer (`_coerce.coerce`); only fields that have no usable value are filled, and only for the row whose identity lies on the entity's mention; the rows then go through qualification, attribution, the gates and the diff. No row, no op and no delete is ever created from the model's list of rows | rows empty (position codes), count above the rows found, update not written, invented rows and values rejected, no removal with `enable_item_removals` |
| `attribution_tiebreak` | an attribution still in `[verify.attribution_drop, verify.attribution_keep)` after the existing second look (scalar, and each list element) | the text (windows around the value), the entity label and the attribution question as Clef was asked it; strict `{answer: yes|no}` | yes counts as `prob_cap` (0.8), no as `1 - prob_cap`; only adopts the value when that reaches the keep bar; never asked for a gate, a delete or a removal, never about a value the tier itself proposed, never outside the band | `test_escalation.py` (two synthetic domains, yes/no, confident answers untouched, `prob_cap` 0.7 cannot keep, delete not escalated) |
| `dotted_discovery` | a top-level `Many` with a dotted identity whose discovery found no mention while presence says there are some (or presence was not asked) | the text; strict `{entities: [{label, identity}]}` | both quotes occur verbatim; the entity becomes a mention (the label quote), then the normal qualify, extract and match path | reproduced off, fixed on, an entity with an invented label or identity rejected, no call when discovery works |

### Evals, combinations, search

`python -m evals run --escalate openai [--escalate-effort low]` (also `TRUSTCALL_S1_ESCALATE`): live it sends
the text of failing cases to OpenAI (needs `OPENAI_API_KEY`; the error names the variable, never a value);
offline the tier is answered by a stand-in that never answers (`FakeEscalation`; `Settings.escalation`
scripts it), so a run with the tier on is the same run and the offline evals stay green. `results.json`
has `escalation` (`openai gpt-6-luna effort=low`) and `escalate_effort`, the aggregate has
`escalation_calls`, `_failed`, `_input_tokens`, `_output_tokens`, `_reasoning_tokens`,
`_latency_ms` and `_mean_latency_ms`, a case has `escalation` (and `by_provider.openai` with
`latency_ms`: every provider row now has it), a combination records `labels.escalation` and OpenAI in
`companies`. Record and replay treat the tier as the provider `openai` (a cassette replays only what
`openai` recorded). `evals search --order escalate.effort` (alone or with other dimensions) sweeps the
base spec's `escalate.reasoning_effort` over `none, low, medium, high` with the usual margin and veto, on
the dev suite only; the leaderboard gains `esc calls`, `esc reasoning tok` and `esc ms/call`, and with
`--price openai=IN:OUT` the cost counts the output (reasoning) tokens, so the table shows the cost and the
latency of each effort.

**Left out on purpose:** probabilities from logprobs (the strict schema masks them); a fallback for the
tier; the tier for `strict`-schema features the Responses API does not take (a JSON schema that is not an
object); `xhigh` and `max` in the search's default candidates (a spec can name them); retrying an
uncertain escalation; escalating `gate`, `null_check` and `delete` (they can authorise a removal).

## Testing

- `testing.FakeTransport(handler)` routed on `CallMeta`, and **`FakeWorld`**: declare text,
  entities (mentions, field values, matching record), non-entities, injected bleed and
  low-confidence faults; it answers every request kind consistently and computes spans with
  `str.find` over the rendered state. `FakeWorld(text)`, `.entity(label, schema=, mentions=,
  fields=, matches=, descriptive=)`, `.non_entity(label, schema=, mentions=)`, `.bleed(entity,
  field, value)`, `.low_confidence(entity, field, confidence)`, `.coref(a, b, p)`,
  `.count(schema, n)`, `.noise(pronouns=, determiners=, possessives=, missing_span=, shuffle=)`,
  `.handler` (to wrap one kind of answer) and `.transport()`. Nested items:
  `world.entity("Rex", schema="Pet", parent="Emma Thompson", ...)` (and `non_entity(..., parent=)`)
  declares an item of that parent's list: requests carry the parent (the focus header's `parent=`
  attribute, or `state.entity.parent` / `state.parent` of a Clef state; a leading "the" does not
  matter) and the world answers discovery, count, presence, qualification, matching and extraction
  for the items of THAT parent only; `.bleed_item(parent, item)` makes discovery for `parent` also
  offer another parent's item (qualification then refuses it); `.delete(record_id, p, parent=)`
  declares the removal answer for an item (its id is its index, or the key of a mapping entry).
  The parent is named as the library names it: a `One` is "the Order", an entity the canonical
  label of its mentions. A mapping entry is declared with `fields={"key": ..., "value": ...}`
  under the schema `<Parent><Field>Entry` (the value model's own name for a model value). The focused
  entity is read from the `<focus … entity="…">` header (Clairic) or `state.entity` (Clef): the
  label first, the mention spans (or contexts) settling which of several entities share it, then
  "the only entity of the schema" (the `One` path). The world fails loudly on a label that names
  one entity while the spans belong to another, and (`check_state`, on by default) on a Clef state
  whose input holds none of the entity's mentions.
  Every request shape it relies on is listed in the `FakeWorld` docstring. `matches=` (a
  model, a dict whose keys must equal the record's, or a string that occurs in the record's JSON)
  picks the offered record a `match` call answers at 0.92 (the rest share 0.08; no `matches=` is
  `__new__` at 0.92; one that fits no offered record, or several, is an `AssertionError`);
  `match_odds={"Emma Thompson": 0.55, "__new__": 0.4}` (keys: a string occurring in exactly one
  offered record, or `__new__`; the rest share what is left) and `match_confidence=` declare the
  probabilities and confidence instead, for the ambiguous paths; `.delete(record_id, p)`
  declares the `delete` answer (default 0.05).
- Unit tests (`pytest --disable-socket --allow-unix-socket`): `test_wire`, `test_compile` (one
  case per type-table row), `test_names` (regex property, pinned hash), `test_flatten`,
  `test_coerce`, `test_diff` (ordering, `/-` repair, list set semantics, echo spans),
  `test_gates` (one gate request per op), `test_verify`, `test_repair`, `test_existing`
  (5 shapes × 3 policies, ported by behaviour from trustcall's `test_strict_existing.py`),
  `test_cardinality`, `test_transport` (`httpx.MockTransport`: retries, `Retry-After`,
  limiter, envelope).
- `test_match.py` (Hungarian vs brute force, `assign`, collisions, similarity, the prefilter,
  compaction, thresholds, `on_ambiguous`, merge, delete questions) and `test_collection_update.py`
  (the scenarios below, end to end on `FakeWorld`);
  `test_collection.py` on `FakeWorld` (M7 covers the insert side: the README people without
  existing records, Sunny `not_qualified`, coreference, count reconciliation, presence,
  attribution bleed, mixed `[Many, One]`; scenarios 1, 4, 5, 6, 9 and 10 below are M8, 8 is M9,
  `tests/unit/test_nested_collections.py`):
  (1) the trustcall README Emma/Michael/Sarah/Olivia
  scenario — three updates with ids 0/1/2 and appended notes, Olivia inserted, existing notes
  untouched, Sunny `not_qualified`, `enable_inserts=False` variant skips Olivia;
  (2) coreference merge incl. inconsistent triangle and two distinct Emmas; (3) bleed
  rejection; (4) two entities claiming one record; (5) zero entities makes no extraction
  calls; (6) 300 existing records → prefilter; (7) count reconciliation both directions;
  (8) nested `Person.pets` through the same engine; (9) ambiguous match; (10) update/delete
  conflict.
- `tests/live/test_smoke.py` (`@pytest.mark.live`, skipped unless `TRUSTCALL_S1_LIVE=1` and the
  credentials are set; `make smoke`) runs the scenarios of `evals/smoke.py` against the real
  models: the README people conversation, a `One` insert (an invoice), a `One` update (the CRM
  contact note) and a nested order update (the order email). It asserts invariants only, from
  `evals/invariants.py`: no exception, every patch applies to `before` and gives `after`, a field
  no operation touches is as it was, updates and deletes name existing records once each, spans
  lie in the input and an inserted text value is its span verbatim, no operation reaches another entity's
  record (the unmentioned order lines, the existing notes stay in place, a short form never
  renames, the dog is not a Person). Exact values belong to the evals. With
  `TRUSTCALL_S1_RECORD=<dir>` each scenario is written to `<dir>/smoke_<name>.jsonl` (`make
  record` writes into `tests/recorded/`); `tests/unit/test_recorded.py` replays whichever of those
  files exist, strictly and offline, through the same invariants on every `make test` (a scenario
  not recorded yet is skipped). `make test` runs `tests/unit` only, with sockets disabled; live
  tests are never collected by it.
- Legacy cassettes: `tests/recorded/legacy_05207aa_{03_update_contact,06_attribution_guard}.jsonl`
  are the wire traces of the second live run (commit 05207aa) converted by `trace_to_cassette`.
  Replayed at HEAD, 8 of 9 (03) and 7 of 9 (06) requests hit; the misses are exactly the
  attribution requests, whose wording M9 changed (arithmetic and date math moved into the
  question), so `test_recorded.py` answers those from the model's recorded answer to the same
  question ids about the same entity (the one deliberate loosening, in the test, never in
  `CassetteTransport`) and asserts the live outcomes: four `replace` operations and no note for
  03 (with `limits.list_strategy="list"`, as that commit had no other), Tom's Lisbon kept at
  attribution 0.7736 for 06. Example 05 was not converted into a test: since 05207aa discovery
  realigns Clairic's list values and spans (M9), which changes the mentions (9 instead of 7) and
  with them the key of every later request (12 of 36 requests hit); `make record` replaces it.
- Evals (`evals/`, `make evals`, `uv run python -m evals ...`): a dataset of 24 small cases
  (`evals/cases.py`), each with the input, schemas, existing records, expected operations (`Op`:
  kind, path, value or a text it must contain, the record), expected final values (`Val`) and
  `must_not` checks (`MustNot`: an operation, a final value, an insert, a delete), optional
  `allow_ops` (legitimate extras, such as a true extra note, count neither for nor against
  precision) and `expect_rejected` (a rejection the result must report), plus a
  `FakeWorld` that answers it offline like an ideal model (with the odd injected fault: a fact
  offered for the wrong person, an open-ended list Clairic is unsure of). The runner scores each
  case: op precision and recall (one-to-one matching of produced and expected operations), field
  value accuracy, `must_not` violations by kind (bleed, wrong deletes, wrong inserts), valid-record
  rate, notes recall (the expected notes, apart from the other operations), requests, input
  tokens and wall time, and writes `results.json` and `summary.md`.
  `python -m evals run [--cases a,b] [--tag notes] [--config default|list|sentences|auto|
  no_verify|strict|lenient] [--set group.field=value] [--live] [--record DIR] [--replay DIR]
  [--out DIR]`; `python -m evals sweep --param group.field=v1,v2,... [--param ...]` runs the grid
  and tabulates one row per point; `python -m evals list` names the cases. Default is offline
  (each case answered by its world; a regression test of the library and the harness at once,
  `tests/unit/test_evals_harness.py`: 24/24 pass, and the strategies `list`, `sentences` and `auto`
  can be compared on the notes cases). `--live` needs the Cloudflare credentials in the
  environment. `--record DIR` writes `<case>.jsonl` cassettes; `--replay DIR` answers from them
  and, with `--live`, sends only the requests that are not recorded (`CassetteTransport(strict=
  False, inner=live)`), so a sweep over thresholds replays the first-pass requests. Which knobs
  change requests is tabulated in `evals/configs.py`: `REQUEST_LIGHT` (act on answers already
  received: `verify.attribution_keep/drop`, `verify.low_confidence`, `verify.sentence_keep`,
  `gates.add/replace/remove_threshold`, `gates.min_*`, `limits.list_auto_below`, `match.accept`,
  ...; only follow-up requests that depend on the decision are new) and `REQUEST_CHANGING`
  (`limits.list_strategy`, `limits.sentence_follow`, `gates.*_null_threshold` (sent as Clairic
  `null_threshold`), `verify.instruction_template`, `verify.window_chars`,
  `verify.input_only_crosscheck`, `passes.split`, the presence and discovery thresholds, `repair.*`:
  every point needs its requests live). The sweep prints a note for each request-changing knob and
  the number of requests it had to send live per point. `docs/LIVE_RUNBOOK.md` has the commands
  of a live run.

## Request and latency profile (information, not a constraint)

README scenario (Many(Person), 3 existing, 4 people + Sunny, ~9 mentions): ~39 requests in
~8 sequential hops; Clef median 209 ms (published), Clairic latency unpublished (assume
0.3–0.8 s) → roughly 2–3.5 s end to end. Flat `One` insert: ~7 requests, ~1–2 s.

Table-heavy spec sheets (live aircraft run, `AircraftSpec` with engine, gear and check lists): 134 requests
and 229 k input tokens for one sheet of one aircraft before the item screening, 206 and 588 k for two,
about two thirds of the tokens Clef's; about half of the requests are the per-field attribution
checks. The screening of identity-less items (see "Nested collections") and the request budget
exist to cut the junk part of that and to keep every request under the provider's maximum.

## Delivery

### Packaging, CI and release

- `pyproject.toml`: hatchling build. Runtime dependencies `pydantic>=2.7,<3`, `httpx>=0.27,<1`,
  `jsonpatch>=1.33,<2`, `jsonpointer>=2.4`, `typing-extensions>=4.9`, `annotated-types>=0.4`;
  optional extra `jsonschema` (`dydantic`). Dev group: pytest, pytest-asyncio, pytest-socket,
  pytest-xdist, pytest-cov, hypothesis, ruff, mypy (strict, pydantic plugin). Ruff enforces
  Google-style docstrings.
- `uv` only, with `uv.lock` committed; `make lint` fails if the lock is stale.
- The wheel ships `trustcall_system_one/` (with `py.typed`) and nothing else. The sdist also
  carries `tests/`, `examples/`, `evals/` and `docs/`.
- `Makefile`: `help`, `install`, `test`, `cov`, `smoke`, `record`, `evals`, `lint`, `format`,
  `build`, `clean`.
- CI (`.github/workflows/ci.yml`): on pushes to `main`, on every pull request and on manual
  dispatch. `make lint` (Python 3.13) and `make test` on Python 3.10, 3.11, 3.12 and 3.13 run on
  pushes and pull requests. A new push cancels the older run of the same pull request. The live
  smoke test (`make smoke`) runs only from a manual dispatch, in the `cloudflare-live`
  environment, which holds `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` as secrets and can
  require reviewers. Jobs are capped at 20 minutes, and the workflow's `GITHUB_TOKEN` is
  read-only.
- Release (`.github/workflows/release.yml`): a tag `v*` checks that the tag (without the `v`)
  equals both `pyproject.toml` `version` and `trustcall_system_one/_version.py` `__version__`,
  runs the tests, builds the sdist and wheel, and publishes to PyPI with trusted publishing
  (environment `pypi`, `id-token: write`, no stored API token). The trusted publisher on pypi.org
  (Manage > Publishing) is: owner `siddicky`, repository `trustcall-system-one`, workflow
  `release.yml`, environment `pypi`.
- Version `0.1.0a1` (alpha) was uploaded by hand with an API token, before the workflow existed.
  Its install commands are `pip install --pre trustcall-system-one` and
  `uv add --prerelease=allow trustcall-system-one`. The first workflow-driven release is `0.1.0a2`.

### Design decisions settled from the examples review

* A field the input does not state is never an operation and is reported as
  `RejectedOp(op=None, reason="not_mentioned")`; the older `p_null`/`unchanged` reasons for
  silent or equal fields are gone (an equal answer is simply no operation).
* The "Dana" / "Dana Whitfield" case is a deterministic rule in the diff (`identity_subset`),
  not left to the gate; the gate stays as the backstop for other identity edits.
* An update run that applies nothing is still an `update` record with empty `patches`.
* Gate state keys are `input`, `entity`, `record`, `proposed_change`; paths are JSON Pointers
  with the schema's validation keys (`/name`, `/notes/-`).
* The input-only cross-check, not the span, is what catches an echoed number or date.
* M6, pattern and format errors: Clairic only quotes the input, so a field that failed a regex
  `pattern` (or another format check) is re-asked ONCE, with the pattern stated in prose, and only
  if `repair.allow_pattern_reask` and a verbatim piece of the input could satisfy it (a regex match
  of the pattern in the input); the answer is accepted only with a span in the input that meets the
  format, else the field fails fast. Repair fixes extraction mistakes, it does not normalise.
* M9, relative edits are not supported: the update path reads what the text states and never
  computes from the old value ("two more of the blue mugs" is 2, not "existing + 2"); see
  "Nested collections".
* M9, a nested item is an entity of the collection engine scoped to its parent: the item's field
  changes are attributed and gated by the item's own update (against the element), the parent
  only judges the addition of a new item; an item removal needs `enable_item_removals`.
* M9, carry-overs of the M8 re-review: a span that cannot be mapped is rejected only when it lies
  in the frame of the state Clairic read or its value is not in the input; the identity guard looks
  at every mention of the entity; the attribution question separates "states it about the entity"
  from "follows by simple arithmetic from figures stated about the entity"; list answers are
  re-paired by text; a required list is `[]` when nothing is said.
* M6, attempts: one repair round (every failing field in parallel) is one attempt;
  `repair.max_attempts` is a termination guard, not a cost cap; `ExtractionResult.attempts` counts
  the extraction round too.

### Milestones

Each milestone landed as one commit that left `make lint` and `make test` green.

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
| M11 | aircraft-run fixes: Clef request budget (calibrated estimate, central guard, coreference to the budget), containment of a refused request, dotted identity paths, class docstrings of nested models, screening of identity-less items |
| M12 | live-run fixes (Decisions as the primary, the aircraft sheets): `list_kind` veto, refusal answers, `clef_hard_picks`, entities without mentions for Decisions, dotted-identity discovery and qualification, markup in mentions, screening of tables for collections with an identity (`discovery.screen_above`), placeholder text is null (`gates.placeholders`) |
| M13 | integration of the TypeSafe-path fixes (pooled equivalent options, date offsets, table cells, initials, `_own_span`, `keep_uncertain`) with the Decisions-primary fixes; `keep_uncertain` is "auto" (on only where Jev answers the attribution and the re-ask) |

### README

The README opens with a **without vs with** comparison: the raw Clairic/Clef calls a developer
would make by hand (curl and httpx, with real response bodies from the live traces) and
everything they would still have to build, next to the same task done with
`create_extractor` (real live output). Then: pitch (patch, don't regenerate; verify, don't
generate), quickstart and install/env, `One` insert, the multi-entity scenario printing
ids/kinds/patches/spans/skipped/untouched, "unmentioned fields never clobbered", type table,
cardinality rules, verification knobs, request profile, `existing` shapes, testing with
`FakeWorld`, examples, limitations vs trustcall, credits (trustcall, MIT, by William Fu-Hinthorn
(LangChain): https://github.com/hinthornw/trustcall).

### Curated examples

Each example is a runnable story on realistic text. It runs live when `CLOUDFLARE_ACCOUNT_ID`
and `CLOUDFLARE_API_TOKEN` are set; otherwise it runs with `--offline` against a scripted
transport and prints a clear "offline demo, scripted answers" banner.
`tests/unit/test_examples.py` runs every example offline and asserts its key outputs. An
example whose feature belonged to an unbuilt milestone was marked
`xfail(strict=True, reason="needs Mn")`, so the milestone that built the feature had to turn it
green. The examples were the acceptance tests for M5 to M9.

| example | shows | needs |
| --- | --- | --- |
| `00_raw_vs_library.py` | the raw httpx call and a naive merge of its values, then the library, on the CRM note | M3 |
| `01_invoice.py` | typed flat extraction: dates, Decimal, Clef-typed Literal/bool, spans highlighted in the source, confidence, verification notes, maths done in code via `computed_field` | M3 |
| `02_support_ticket.py` | nested sub-models, an optional sub-model the text never mentions, a discriminated union of issue types, an ordered score enum | M4 |
| `03_update_contact.py` | updating a CRM record from a short note: only the mentioned fields change, the patch list is the audit trail, the rejected ops say why each other field was left alone | M5 |
| `04_repair.py` | a constrained schema whose first answer fails validation and is fixed by a narrow re-ask | M6 |
| `05_people_memory.py` | trustcall's Emma/Michael/Sarah/Olivia scenario: `Many(Person)`, coreference, updates plus inserts, Sunny filtered out, untouched records, notes appended with `/-` | M7–M8 |
| `06_attribution_guard.py` | two people in one passage; a value that belongs to one is rejected on the other | M7 |
| `07_order_line_items.py` | nested `list[LineItem]`: add an item and change a quantity through the collection engine | M9 |
| `08_testing_your_extractor.py` | unit-testing your own extractor with `FakeTransport`/`FakeWorld` | M7 |

## Risks

1. **Clairic is extractive**: text values are verbatim spans, so notes read "got promoted to
   Senior Marketing Manager last month" rather than a paraphrase, and Olivia's relationship
   comes back as "my cousin" (speaker's voice). Normalised vocabularies should use `Literal`
   (→ Clef `choice`). README shows real output.
2. Clef silently truncates long state, and a request over the maximum is refused — pre-send budget
   on the calibrated estimate of the whole request (state and questions), split/narrow in
   `Run.call_clef`, containment of a refusal, post-hoc warning. The estimate is calibrated on one
   live run of one kind of document (aircraft spec tables on Cloudflare); another tokenizer, a
   dense non-Latin text or images (not counted) can still exceed it, in which case the provider's
   422 is contained (that request's answers are missing, with a `request_rejected` warning) rather
   than fatal. Containment (size refusals only) means an invoke can now *finish* with some fields
   `missing_answer` or some mentions unclustered where it used to raise: read `warnings`, or use
   `strict=True`. Non-size 4xx errors, anything uncertain and a refusal that repeats more than 8
   times still raise.
3. Clairic/Clef rate limits are unpublished; ~40 requests per invoke × `batch` may hit 429s —
   adaptive limiter.
4. Span offsets could drift if Clairic normalises state — always request `state_text`.
5. Thresholds and "narrower is more accurate" are hypotheses — every threshold is a knob and
   the evals harness measures them.
6. Clef option-id pattern isn't explicitly published — reuse the question-id pattern; the
   live smoke test checks it.
7. trustcall is MIT, this repo is Unlicense — reimplement by behaviour and credit.
8. Clairic's `list` type cannot read open-ended fields (confidence 1e-6 to 1e-23 live). Mitigated
   by the sentences strategy (see "Open-ended lists"); the live evals decide whether `"auto"` is
   the right default and what `sentence_follow` and `sentence_keep` should be.
9. Cassettes are snapshots of request bodies: any change to the wording of a question, the
   rendering of a state or a threshold that is sent to a model (`null_threshold`) changes the key.
   A replay that misses names the call; record again rather than loosening the match.
10. Identity-less nested items are still the weakest path. The screening cuts the junk, not the
    granularity: one mention that covers several items ("CFM56-7B26 engines") is one entity, so
    an engine row can still be lost (live, two-aircraft run: the A320's engine row was proposed as
    `1` / `CFM56-5B4/3` and refused, the B737's two engines became one entity that failed
    validation on `position`); the count cap keeps the best qualified but cannot split a phrase.
    A required `Literal` that the input does not state for an item still drops the whole item. The
    junk-word stoplist and the field-label rule are English/table oriented heuristics.
11. A nested `s1_field(identity=True)` identifies the sub-model, not the record above it (per-level
    identity is documented and tested), so an identity inside a sub-model of a `Many` item is
    named with `Many(identity="a.b")`, not marked on the field.
12. Decisions as the PRIMARY Clef (live on commit `353b33b`, same thresholds as Cloudflare's: 20/29 dev, 12/30
    holdout2 against 28/29 and 16/30; precision 90% and 97%, recall 79% and 80%; see the README's Performance
    section; the earlier 13/29 and 6/30 were measured before the fixes below). Our part is fixed (notes: the
    `list_kind` veto; a `refusal` answer; the warning name; an entity without mentions). The rest is
    the model, measured on identical requests: attribution of text and identifier values 0.4 to
    0.65 where Clef says 0.97 (a normalised `09:00:00` against the input's `09:00`, serial numbers,
    e-mail, addresses), a graded 0.8 for a clear delete (below the 0.85 delete gate), and an
    input that both removes and changes a record answered "no" for the delete (0.1), so the
    update proceeds instead of the documented `conflict_update_delete`: with an unconfirmed delete
    the update is the right outcome, and that gold expectation is provider-specific. The
    chance-normalised `confidence` it reports (0.3 for 0.65) is lower than Clef's for the same
    probability and is used as reported. Re-measured after these fixes (the figures above).
13. Placeholder text (`TBA`, `TBD`, `N/A`, a double or long dash) is null in a plain text field
    (`GateConfig.placeholders`; the opt-outs are `()` for the run and `s1_field(placeholders=False)`
    for one field; `TBC`, a lone `-` and a lone `?` are not defaults: ambiguous, and a result or
    status can read `-`): the live sheet's `apu.status = "TBA"` and
    its three MSNs `TBA` (which also made three identical, different "identity values"). List
    elements and Clef-typed fields are untouched; `none`, `unknown` and `na` are deliberately not
    on the list. What the schema calls a check type (an interval string) is the schema's.
14. Two names for one entity that no identity value joins can stay two records (live, real sheet:
    `AIRBUS A320-251 NEO` and `A320-200 NEO`, one 2024 aircraft; count 3, four found): the count
    driven merge joins two clusters only if EVERY pair across them passes the merge check (0.5):
    0.87 for `Airbus A320-251N` with the first, then 0.59 and 0.26 against the second (a cluster
    that held `CF` and `NEO) -` junk). The screening removes the junk from clusters; the rule
    stays, and two mentions with the same words are asked about, not assumed to be one.

15. The holdout is contaminated for TypeSafe. Commit `7decebc` (pooled equivalent
    options, date offsets and the "today" note, `keep_uncertain`, table cells, initials, the
    identity span) fixed failures that were read from live TypeSafe runs of the holdout suite, and
    the cases they fixed then passed. **Every TypeSafe holdout result from `7decebc` on is
    diagnosed-on-holdout** (`evals/README.md`): it is a dev number, not a blind one, and it does
    not support choosing a combination. The same holds for any live holdout number that was read
    while a fix was written. A fresh blind suite (`holdout2`) is written separately; the offline
    holdout (scripted worlds) measures the engine, not a provider.

## Verification

1. `make lint` (ruff + mypy strict) and `make test` (socket-disabled) green after every
   milestone.
2. The README Emma/Michael/Sarah/Olivia scenario passes on `FakeWorld`, including bleed,
   coreference, collision and zero-entity cases.
3. If `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN` are available in the environment, run
   `make smoke` against the live API and record cassettes (`make record`); if not, report that live
   verification was not possible and leave `make smoke` as the documented check. The commands of a
   live run (smoke, record, evals baseline, threshold and strategy sweeps, and which numbers to
   report) are in `docs/LIVE_RUNBOOK.md`.
4. CI must be green on the pull request before merge.
