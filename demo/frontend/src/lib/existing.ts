import type { SchemaDraft } from "./draft"
import { newId } from "./schema"
import type { ExistingIn, JsonObject } from "./types"

/** An existing record as the form holds it. */
export interface ExistingDraft {
  key: string
  id: string
  schema: string
  /** JSON text of the record. */
  text: string
}

export interface ExistingCheck {
  /** Blocks the run. */
  error: string | null
  /** Worth a look, does not block. */
  warning: string | null
  record: JsonObject | null
}

export function newExisting(schema: string, id = ""): ExistingDraft {
  return { key: newId(), id, schema, text: "{\n  \n}" }
}

/** Check every entry: JSON validity, ids, schema names and unknown keys. */
export function checkExisting(
  entries: ExistingDraft[],
  drafts: SchemaDraft[]
): ExistingCheck[] {
  const seen = new Map<string, number>()
  for (const e of entries) {
    const id = e.id.trim()
    seen.set(id, (seen.get(id) ?? 0) + 1)
  }
  return entries.map((e) => {
    const id = e.id.trim()
    if (!id) return { error: "needs an id", warning: null, record: null }
    if ((seen.get(id) ?? 0) > 1) {
      return { error: "id is used twice", warning: null, record: null }
    }
    const draft = drafts.find((d) => d.name === e.schema)
    if (!draft) {
      return { error: "pick one of the schemas", warning: null, record: null }
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(e.text)
    } catch (err) {
      const message = err instanceof Error ? err.message : "invalid JSON"
      return { error: `invalid JSON (${message})`, warning: null, record: null }
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return {
        error: "the record must be a JSON object",
        warning: null,
        record: null,
      }
    }
    const record = parsed as JsonObject
    const props = draft.jsonSchema.properties
    let warning: string | null = null
    if (typeof props === "object" && props !== null && !Array.isArray(props)) {
      const unknown = Object.keys(record).filter((k) => !(k in props))
      if (unknown.length > 0) {
        warning = `keys not in ${draft.name}: ${unknown.join(", ")}`
      }
    }
    return { error: null, warning, record }
  })
}

export function toExistingIn(
  entries: ExistingDraft[],
  checks: ExistingCheck[]
): ExistingIn[] {
  const out: ExistingIn[] = []
  entries.forEach((e, i) => {
    const record = checks[i]?.record
    if (record) out.push({ id: e.id.trim(), schema: e.schema, record })
  })
  return out
}
