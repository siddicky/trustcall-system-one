/**
 * A schema as the editor holds it: the form model plus the three views
 * (form, Pydantic source, JSON Schema text) and their sync state.
 */
import {
  fieldsToJsonSchema,
  fieldsToPydantic,
  jsonSchemaMeta,
  jsonSchemaToFields,
  newId,
  type FormField,
} from "./schema"
import type { Cardinality, JsonSchema, ParsedSchema } from "./types"

export interface SchemaDraft {
  id: string
  name: string
  description: string
  cardinality: Cardinality
  fields: FormField[]
  /** What is sent to the backend. Equals the form's schema unless JSON or Pydantic set it. */
  jsonSchema: JsonSchema
  /** Text of the JSON Schema editor (may be invalid while typing). */
  jsonText: string
  jsonError: string | null
  /** Text of the Pydantic editor. */
  pydantic: string
  /** False while the Pydantic text has edits that were not synced to the form. */
  pydanticSynced: boolean
}

export function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

interface DraftInit {
  name: string
  description?: string
  cardinality?: Cardinality
  fields: FormField[]
}

function regenerate(
  d: Pick<SchemaDraft, "name" | "description" | "fields">
): Pick<SchemaDraft, "jsonSchema" | "jsonText" | "jsonError" | "pydantic" | "pydanticSynced"> {
  const jsonSchema = fieldsToJsonSchema(d.name, d.fields, d.description)
  return {
    jsonSchema,
    jsonText: pretty(jsonSchema),
    jsonError: null,
    pydantic: fieldsToPydantic(d.name, d.fields, d.description),
    pydanticSynced: true,
  }
}

export function createDraft(init: DraftInit): SchemaDraft {
  const base = {
    name: init.name,
    description: init.description ?? "",
    fields: init.fields,
  }
  return {
    id: newId(),
    cardinality: init.cardinality ?? "one",
    ...base,
    ...regenerate(base),
  }
}

/** The form changed: everything else is derived again. */
export function withFields(d: SchemaDraft, fields: FormField[]): SchemaDraft {
  return { ...d, fields, ...regenerate({ ...d, fields }) }
}

/** The name or description changed. */
export function withMeta(
  d: SchemaDraft,
  meta: { name?: string; description?: string }
): SchemaDraft {
  const next = { ...d, ...meta }
  return { ...next, ...regenerate(next) }
}

/** The JSON Schema text was edited. Valid object schemas replace the form. */
export function withJsonText(d: SchemaDraft, text: string): SchemaDraft {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (err) {
    const message = err instanceof Error ? err.message : "Invalid JSON"
    return { ...d, jsonText: text, jsonError: message }
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ...d, jsonText: text, jsonError: "The root must be a JSON object." }
  }
  const schema = parsed as JsonSchema
  if (schema.type !== undefined && schema.type !== "object") {
    return {
      ...d,
      jsonText: text,
      jsonError: 'The root schema must have "type": "object".',
    }
  }
  const fields = jsonSchemaToFields(schema)
  const description = jsonSchemaMeta(schema).description
  return {
    ...d,
    description,
    fields,
    jsonSchema: schema,
    jsonText: text,
    jsonError: null,
    pydantic: fieldsToPydantic(d.name, fields, description),
    pydanticSynced: true,
  }
}

export function withPydanticText(d: SchemaDraft, text: string): SchemaDraft {
  return { ...d, pydantic: text, pydanticSynced: false }
}

/** Replace a draft with a model parsed from Pydantic source by the backend. */
export function fromParsed(
  d: SchemaDraft,
  parsed: ParsedSchema,
  source: string
): SchemaDraft {
  const fields = jsonSchemaToFields(parsed.json_schema)
  return {
    ...d,
    name: parsed.name,
    description: jsonSchemaMeta(parsed.json_schema).description,
    fields,
    jsonSchema: parsed.json_schema,
    jsonText: pretty(parsed.json_schema),
    jsonError: null,
    pydantic: source,
    pydanticSynced: true,
  }
}

/** A short caption: `7 FIELDS`. */
export function fieldCount(d: SchemaDraft): number {
  const props = d.jsonSchema.properties
  return typeof props === "object" && props !== null && !Array.isArray(props)
    ? Object.keys(props).length
    : 0
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

export interface StoredSchemas {
  v: 1
  selectedId: string
  drafts: SchemaDraft[]
}

function isFormField(value: unknown): value is FormField {
  if (typeof value !== "object" || value === null) return false
  const f = value as Record<string, unknown>
  return (
    typeof f.id === "string" &&
    typeof f.name === "string" &&
    typeof f.type === "string" &&
    typeof f.required === "boolean" &&
    Array.isArray(f.children) &&
    f.children.every(isFormField) &&
    Array.isArray(f.enumValues)
  )
}

/** Validate what was read from storage; `null` when it is not usable. */
export function restoreSchemas(value: unknown): StoredSchemas | null {
  if (typeof value !== "object" || value === null) return null
  const stored = value as Partial<StoredSchemas>
  if (stored.v !== 1 || !Array.isArray(stored.drafts)) return null
  const drafts = stored.drafts.filter(
    (d): d is SchemaDraft =>
      typeof d === "object" &&
      d !== null &&
      typeof d.id === "string" &&
      typeof d.name === "string" &&
      typeof d.pydantic === "string" &&
      typeof d.jsonText === "string" &&
      (d.cardinality === "one" || d.cardinality === "many") &&
      typeof d.jsonSchema === "object" &&
      d.jsonSchema !== null &&
      Array.isArray(d.fields) &&
      d.fields.every(isFormField)
  )
  if (drafts.length === 0) return null
  const selectedId = drafts.some((d) => d.id === stored.selectedId)
    ? (stored.selectedId as string)
    : drafts[0].id
  return { v: 1, selectedId, drafts }
}
