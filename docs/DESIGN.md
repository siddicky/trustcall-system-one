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
| `transport.py` (public) | `Transport` protocol, `CloudflareTransport`, adaptive per-model limiter |
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
- **Matching:** `match=MatchConfig(accept, min_confidence, render_fields, collision, on_ambiguous,
  delete_threshold)` configures how a `Many` schema's entities are matched to its existing records
  (see "Collection engine", steps 8-9); `on_ambiguous="raise"` raises `AmbiguousMatchError`
  (`schema_name`, `label`, `candidates`).
- **`s1_field(...)`** → `Field(json_schema_extra={"trustcall_system_one": {...}})` with
  `instructions, null_threshold, min_confidence, clairic_type, unit, decimals, integer,
  identity, skip, clef, criteria`.
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
`typing_extensions.Doc` → title → attribute docstring → humanised name) + type/constraint hints, templated per entity
("{label}'s {desc}"). A lint warns on aggregation-style descriptions ("total of…").

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
   `ValueError` at `create_extractor`) → `SchemaPlan.identity` (`s1_field(identity=True)`, else
   a `str` field named name/full_name/title/label/id/key/slug/email/handle/sku, else the first
   required `str`) → none: discovery then follows the schema docstring ("Every mention of a
   Reading (A temperature reading…)") and the run warns `no_identity_field`. The identity reads
   in the discovery question as `by their name (full name)`, compound fields joined with "and".
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
   with a broadened discovery or a merge: "Is `entities.e{n}.label` a {Person} ({description})?". `< 0.5` →
   `SkippedEntity(reason="not_qualified")` ("Sunny", a dog, for `Person`); a missing answer
   qualifies. Qualification runs BEFORE the count comparison, so "k" is the number of entities
   that qualify.
5. **Count reconciliation** (count `n` against `k` qualifying entities; acted on only when
   `broaden_on_count_mismatch` and `max(n, k) ≤ count_trust_max`, else the mismatch only warns;
   `count_trust_max` is validated to at most 30, so a merge asks at most 30·29/2 = 435 pair checks):
   * `n > k`: ONE broadened re-discovery (the same question plus "Also include descriptive
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
7. **Attribution (default on):** Clef `noul` per non-null value and per list element: "Does
   `input` state that {desc} of `entity.label` is '{value}', or does it follow by simple
   arithmetic or date math from figures the input states about `entity.label`?" (true: the
   input states it about the entity, or it follows by simple arithmetic or date math from
   figures stated about the entity; false: the input does not state it about `entity.label`,
   or it concerns another entity. A computed value such as a total stated only as its parts is
   a legitimate answer, a figure of somebody else is not; an integral float is shown as `1080`,
   not `1080.0`)
   ≥ 0.75 keep, < 0.4 drop (`attribution`), between → isolated re-ask then one more check. The
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
     accept` AND the answer's confidence is `>= min_confidence`. Else the entity is ambiguous:
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
     remove X"): the update runs first and decides. If no operation survives the gate (nothing but
     "remove it" is said about it) the record is deleted (`kind="delete"`); if at least one gated
     operation exists ("drop the quantity of X to 1" flagged by mistake) the input both changes
     and removes it, which is `SkippedEntity(reason="conflict_update_delete", candidates=((id,
     match p),))`: neither an update nor a delete, its operations are not in the result. With
     `enable_updates=False` nothing could change, so the removal stands without a request; an
     identity-conflict refusal (a wrong match) is reported as such and holds the delete;
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
  top-level record (step 9): no operation through the gate (the text only says to remove it) →
  `remove`, otherwise `SkippedEntity(conflict_update_delete)` with no operation.

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
max_attempts=5, stop_on_repeat=True, allow_pattern_reask=True)`.

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
  `clef`/`clef-flash` while the URL uses `@cf/cloudflare/...`. **Observed constraint, stricter
  than the published schema:** a Clairic `list` field takes no `null_threshold` (HTTP 422, error
  3030, "list fields don't take 'null_threshold'"; a list answer has no `p_null` either). The
  compiler never emits it for a list, `Ctx.clairic_field` leaves it out of re-asks, and
  `ClairicField` rejects it, so `FakeTransport` catches it offline. Every other type accepts it.
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
