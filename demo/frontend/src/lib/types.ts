/** Types of the demo backend's JSON API (`demo/backend/app`). */

export type Provider = "cloudflare" | "typesafe"

export interface Health {
  ok: boolean
  provider: Provider | null
  version: string
}

export type DocumentKind = "markdown" | "pdf" | "text"

export interface DocumentInfo {
  filename: string
  kind: DocumentKind
  text: string
  /** `null` unless the file is a PDF. */
  pages: number | null
  chars: number
}

// ---------------------------------------------------------------------------
// JSON Schema
// ---------------------------------------------------------------------------

export type JsonObject = { [key: string]: JsonValue }
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | JsonObject

/** A JSON Schema document or sub-schema. Loosely typed on purpose. */
export type JsonSchema = JsonObject

export interface ParsedSchema {
  name: string
  json_schema: JsonSchema
}

export interface SchemaParseResponse {
  schemas: ParsedSchema[]
}

// ---------------------------------------------------------------------------
// Extract request
// ---------------------------------------------------------------------------

export type Cardinality = "one" | "many"

export interface SchemaIn {
  name: string
  json_schema: JsonSchema
  cardinality: Cardinality
}

export interface ExistingIn {
  id: string
  schema: string
  record: JsonObject
}

export interface ExtractOptions {
  enable_inserts: boolean
  enable_deletes: boolean
  /** ISO date (YYYY-MM-DD) that relative dates resolve against. */
  reference_date: string | null
}

export interface ExtractRequest {
  text: string
  schemas: SchemaIn[]
  existing: ExistingIn[]
  options: ExtractOptions
}

// ---------------------------------------------------------------------------
// Extract response
// ---------------------------------------------------------------------------

/** `[start, end)` offsets. In a result they index the input text. */
export type Span = [number, number]

export type RecordKind = "insert" | "update" | "delete"

export interface PatchOp {
  op: "add" | "remove" | "replace" | "move" | "copy" | "test" | string
  path: string
  /** Absent for `remove`. */
  value?: JsonValue
  repaired_from: string | null
}

export interface RejectedOp {
  op: string | null
  path: string
  reason: string
  detail: string
  p: number | null
}

export interface FieldAnswer {
  /** Position of the field in the document: keys and list indices. */
  path: Array<string | number>
  /** `path` joined with dots; also the key in `fields`. */
  pretty: string
  wire_name: string
  engine: string
  value: JsonValue
  raw: JsonValue
  p_null: number | null
  confidence: number | null
  guess: JsonValue
  /** Supporting span in the original input. */
  span: Span | null
  /** Supporting spans of a list answer, aligned with `value`. */
  spans: Span[] | null
  source: "clairic" | "clef" | "const" | "default"
  /** A float for a scalar, one entry per element for a list. */
  attribution_p: number | Array<number | null> | null
  gate_p: number | null
  crosscheck_agree: boolean | null
  provider: string | null
  reask: boolean
  repaired: boolean
  rejected_reason: string | null
  warnings: string[]
}

export interface Mention {
  text: string
  span: Span | null
}

export interface EntityInfo {
  key: string
  label: string
  mentions: Mention[]
  first_mention: Mention | null
  qualify_p: number | null
  match_probs: Record<string, number>
  parent: string | null
}

export interface ResultRecord {
  schema_name: string
  json_doc_id: string | null
  kind: RecordKind
  value: JsonObject | null
  before: JsonObject | null
  after: JsonObject | null
  patches: PatchOp[]
  rejected: RejectedOp[]
  fields: Record<string, FieldAnswer>
  entity: EntityInfo | null
  valid: boolean
  errors: string[]
  delete_p: number | null
}

export interface SkippedEntity {
  schema_name: string
  entity: EntityInfo
  reason: string
  detail: string
  /** `[existing record id, probability]` pairs. */
  candidates: Array<[string, number]>
}

export interface Usage {
  state_tokens: number
  image_tokens: number
  schema_tokens: number
  input_tokens: number
  output_tokens: number
  reasoning_tokens: number
  passes: number
  requests: number
  reported_requests: number
  reported_cost_usd: number
  by_model: Record<string, Usage>
  by_kind: Record<string, Usage>
}

export interface CostReport {
  usd: number
  unpriced_models: string[]
  complete: boolean
  reported_usd: number
}

export interface CallRecord {
  model: string
  kind: string
  schema: string
  entity: string | null
  field: string | null
  scope: string
  n_items: number
  usage: Record<string, number>
  duration_ms: number
  retries: number
  error: string | null
  status_code: number | null
  request_id: string | null
  cost_usd: number | null
  provider: string | null
  fallback_from: string | null
}

export interface ExtractResponse {
  records: ResultRecord[]
  skipped: SkippedEntity[]
  untouched: string[]
  deleted: string[]
  attempts: number
  usage: Usage
  cost: CostReport
  calls: CallRecord[]
  warnings: string[]
  elapsed_ms: number
}

/** Codes of `{detail: {code, message}}` errors, for reference. */
export type ApiErrorCode =
  | "no_provider"
  | "invalid_schema"
  | "invalid_request"
  | "bad_request"
  | "request_too_large"
  | "validation_failed"
  | "rate_limited"
  | "auth_error"
  | "upstream_error"
  | "extraction_error"
  | "pdf_no_text"
  | "pdf_parse_error"
  | "unsupported_type"
  | "too_large"
  | "unsafe_source"
  | "syntax_error"
  | "no_models"
  | "invalid_model"
