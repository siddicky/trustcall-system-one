/**
 * Pure helpers for the schema editor: the form model, its conversion to and
 * from JSON Schema, and Pydantic source generation.
 *
 * The JSON Schema written here follows what the backend accepts: optional
 * fields are `anyOf: [{...}, {type: "null"}]` with `default: null` (never
 * `type: [x, "null"]`), nested objects live in `$defs` behind `$ref`, and
 * required fields are listed in `required`. Optional lists carry `default: []`.
 */
import type { JsonObject, JsonSchema, JsonValue } from "./types"

export const FIELD_TYPES = [
  "string",
  "integer",
  "number",
  "boolean",
  "date",
  "datetime",
  "enum",
  "object",
  "array",
] as const

export type FieldType = (typeof FIELD_TYPES)[number]

/** The item type of an array: any field type but another array. */
export type ItemType = Exclude<FieldType, "array">

export const ITEM_TYPES: readonly ItemType[] = FIELD_TYPES.filter(
  (t): t is ItemType => t !== "array"
)

export interface FormField {
  /** Stable identity for React keys and tree updates; not part of the schema. */
  id: string
  name: string
  type: FieldType
  description: string
  required: boolean
  /** Values of an `enum` field, or of an array whose items are `enum`. */
  enumValues: string[]
  /** Child fields of an `object` field, or of an array whose items are `object`. */
  children: FormField[]
  /** Item type of an `array` field. */
  itemType: ItemType
}

// ---------------------------------------------------------------------------
// Construction and tree updates
// ---------------------------------------------------------------------------

let counter = 0

/** A fresh id for a form field. */
export function newId(): string {
  counter += 1
  return `f${counter.toString(36)}${Math.random().toString(36).slice(2, 6)}`
}

/** Build a field, filling in the defaults. */
export function makeField(partial: Partial<FormField> = {}): FormField {
  return {
    id: newId(),
    name: "",
    type: "string",
    description: "",
    required: true,
    enumValues: [],
    children: [],
    itemType: "string",
    ...partial,
  }
}

/** Whether a field has child rows (an object, or an array of objects). */
export function hasChildren(field: FormField): boolean {
  return (
    field.type === "object" ||
    (field.type === "array" && field.itemType === "object")
  )
}

/** Whether a field has enum values (an enum, or an array of enums). */
export function hasEnum(field: FormField): boolean {
  return (
    field.type === "enum" ||
    (field.type === "array" && field.itemType === "enum")
  )
}

/** Apply `fn` to the field with the given id, anywhere in the tree. */
export function updateField(
  fields: FormField[],
  id: string,
  fn: (field: FormField) => FormField
): FormField[] {
  return fields.map((f) => {
    if (f.id === id) return fn(f)
    if (f.children.length === 0) return f
    return { ...f, children: updateField(f.children, id, fn) }
  })
}

/** Remove the field with the given id, anywhere in the tree. */
export function removeField(fields: FormField[], id: string): FormField[] {
  return fields
    .filter((f) => f.id !== id)
    .map((f) =>
      f.children.length === 0
        ? f
        : { ...f, children: removeField(f.children, id) }
    )
}

/** Append `child` to the children of `parentId`, or to the top level for `null`. */
export function addField(
  fields: FormField[],
  parentId: string | null,
  child: FormField
): FormField[] {
  if (parentId === null) return [...fields, child]
  return updateField(fields, parentId, (f) => ({
    ...f,
    children: [...f.children, child],
  }))
}

/** Change the type of a field, keeping what still applies. */
export function retype(field: FormField, type: FieldType): FormField {
  const next: FormField = { ...field, type }
  if (type === "array" && !ITEM_TYPES.includes(next.itemType)) {
    next.itemType = "string"
  }
  return next
}

/** Number of leaf paths a field list compiles to. */
export function leafCount(fields: FormField[]): number {
  let n = 0
  for (const f of fields) {
    if (hasChildren(f)) n += Math.max(1, leafCount(f.children))
    else n += 1
  }
  return n
}

/** Problems that would make the schema unusable; empty when it is fine. */
export function validateFields(fields: FormField[], prefix = ""): string[] {
  const issues: string[] = []
  const seen = new Set<string>()
  for (const f of fields) {
    const name = f.name.trim()
    const where = prefix ? `${prefix}.` : ""
    if (!name) {
      issues.push(`${where}(unnamed field) needs a name`)
      continue
    }
    if (seen.has(name)) issues.push(`${where}${name} is defined twice`)
    seen.add(name)
    if (hasEnum(f) && f.enumValues.length === 0) {
      issues.push(`${where}${name} needs at least one enum value`)
    }
    if (hasChildren(f)) {
      if (f.children.length === 0) {
        issues.push(`${where}${name} needs at least one child field`)
      }
      issues.push(...validateFields(f.children, `${where}${name}`))
    }
  }
  return issues
}

/** Names that appear more than once among siblings, for inline highlighting. */
export function duplicateNames(fields: FormField[]): Set<string> {
  const seen = new Set<string>()
  const dup = new Set<string>()
  for (const f of fields) {
    const name = f.name.trim()
    if (!name) continue
    if (seen.has(name)) dup.add(name)
    seen.add(name)
  }
  return dup
}

// ---------------------------------------------------------------------------
// Names of nested models
// ---------------------------------------------------------------------------

function pascal(text: string): string {
  const words = text
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
  const joined = words.map((w) => w[0].toUpperCase() + w.slice(1)).join("")
  return /^[0-9]/.test(joined) ? `M${joined}` : joined
}

function singular(text: string): string {
  if (/ies$/i.test(text) && text.length > 3) return text.slice(0, -3) + "y"
  if (/(ss|us|is)$/i.test(text)) return text
  if (/s$/i.test(text) && text.length > 1) return text.slice(0, -1)
  return text
}

/** A valid Python class name for a schema name. */
export function classIdent(name: string): string {
  const id = name.replace(/[^A-Za-z0-9_]/g, "_").replace(/^([0-9])/, "_$1")
  return id || "Model"
}

/** Class names of the nested models, keyed by field id. */
export function defNames(
  rootName: string,
  fields: FormField[]
): Map<string, string> {
  const used = new Set<string>([classIdent(rootName)])
  const names = new Map<string, string>()
  const visit = (list: FormField[]) => {
    for (const f of list) {
      if (!hasChildren(f)) continue
      const base =
        (f.type === "array"
          ? pascal(singular(f.name))
          : pascal(f.name)) || "Model"
      let candidate = base
      for (let i = 2; used.has(candidate); i += 1) candidate = `${base}${i}`
      used.add(candidate)
      names.set(f.id, candidate)
      visit(f.children)
    }
  }
  visit(fields)
  return names
}

// ---------------------------------------------------------------------------
// Fields to JSON Schema
// ---------------------------------------------------------------------------

function scalarSchema(type: ItemType, f: FormField): JsonObject {
  switch (type) {
    case "string":
      return { type: "string" }
    case "integer":
      return { type: "integer" }
    case "number":
      return { type: "number" }
    case "boolean":
      return { type: "boolean" }
    case "date":
      return { type: "string", format: "date" }
    case "datetime":
      return { type: "string", format: "date-time" }
    case "enum":
      return { type: "string", enum: [...f.enumValues] }
    case "object":
      return { type: "object" }
  }
}

/**
 * Build the JSON Schema of a form.
 *
 * @param name - Schema name; becomes the `title`.
 * @param fields - The top-level fields.
 * @param description - Optional description of the whole record.
 */
export function fieldsToJsonSchema(
  name: string,
  fields: FormField[],
  description = ""
): JsonSchema {
  const names = defNames(name, fields)
  const defs: Record<string, JsonObject> = {}

  const refFor = (f: FormField): JsonObject => {
    const defName = names.get(f.id) ?? "Model"
    defs[defName] = {
      title: defName,
      ...objectBody(f.children),
    }
    return { $ref: `#/$defs/${defName}` }
  }

  const baseOf = (f: FormField): JsonObject => {
    if (f.type === "object") return refFor(f)
    if (f.type === "array") {
      const items =
        f.itemType === "object" ? refFor(f) : scalarSchema(f.itemType, f)
      return { type: "array", items }
    }
    return scalarSchema(f.type, f)
  }

  const propertyOf = (f: FormField): JsonObject => {
    const base = baseOf(f)
    const desc = f.description.trim()
    const out: JsonObject = {}
    if (f.type === "array") {
      Object.assign(out, base)
      if (desc) out.description = desc
      if (!f.required) out.default = []
      return out
    }
    if (f.required) {
      Object.assign(out, base)
      if (desc) out.description = desc
      return out
    }
    out.anyOf = [base, { type: "null" }]
    out.default = null
    if (desc) out.description = desc
    return out
  }

  function objectBody(list: FormField[]): JsonObject {
    const properties: JsonObject = {}
    const required: string[] = []
    for (const f of list) {
      const key = f.name.trim()
      if (!key) continue
      properties[key] = propertyOf(f)
      if (f.required) required.push(key)
    }
    const body: JsonObject = { type: "object", properties }
    if (required.length > 0) body.required = required
    return body
  }

  const body = objectBody(fields)
  const out: JsonSchema = { title: name }
  const desc = description.trim()
  if (desc) out.description = desc
  Object.assign(out, body)
  if (Object.keys(defs).length > 0) out.$defs = defs
  return out
}

// ---------------------------------------------------------------------------
// JSON Schema to fields
// ---------------------------------------------------------------------------

function isObject(value: JsonValue | undefined): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function lookupRef(root: JsonSchema, ref: string): JsonValue | undefined {
  if (!ref.startsWith("#/")) return undefined
  let node: JsonValue | undefined = root
  for (const part of ref.slice(2).split("/")) {
    const key = part.replace(/~1/g, "/").replace(/~0/g, "~")
    if (!isObject(node)) return undefined
    node = node[key]
  }
  return node
}

function isNullSchema(node: JsonValue): boolean {
  return isObject(node) && node.type === "null"
}

interface Resolved {
  schema: JsonObject
  nullable: boolean
  description: string | undefined
  /** Refs followed so far on this path, to stop cycles. */
  seen: ReadonlySet<string>
}

function resolve(
  node: JsonValue | undefined,
  root: JsonSchema,
  seen: ReadonlySet<string>
): Resolved {
  if (!isObject(node)) {
    return { schema: {}, nullable: false, description: undefined, seen }
  }
  const own =
    typeof node.description === "string" ? node.description : undefined

  if (typeof node.$ref === "string") {
    const ref = node.$ref
    const target = lookupRef(root, ref)
    if (seen.has(ref) || target === undefined) {
      return {
        schema: { type: "object" },
        nullable: false,
        description: own,
        seen,
      }
    }
    const inner = resolve(target, root, new Set(seen).add(ref))
    return { ...inner, description: own ?? inner.description }
  }

  if (Array.isArray(node.allOf) && node.allOf.length > 0) {
    const inner = resolve(node.allOf[0], root, seen)
    return { ...inner, description: own ?? inner.description }
  }

  const variants = Array.isArray(node.anyOf)
    ? node.anyOf
    : Array.isArray(node.oneOf)
      ? node.oneOf
      : null
  if (variants && variants.length > 0) {
    const nonNull = variants.filter((v) => !isNullSchema(v))
    const nullable = nonNull.length < variants.length
    if (nonNull.length === 0) {
      return { schema: {}, nullable: true, description: own, seen }
    }
    const inner = resolve(nonNull[0], root, seen)
    return {
      ...inner,
      nullable: nullable || inner.nullable,
      description: own ?? inner.description,
    }
  }

  if (Array.isArray(node.type)) {
    const types = node.type.filter((t) => t !== "null")
    return {
      schema: { ...node, type: (types[0] ?? "string") as JsonValue },
      nullable: types.length < node.type.length,
      description: own,
      seen,
    }
  }

  return { schema: node, nullable: false, description: own, seen }
}

function typeOf(schema: JsonObject): FieldType {
  if (Array.isArray(schema.enum) || "const" in schema) return "enum"
  switch (schema.type) {
    case "integer":
      return "integer"
    case "number":
      return "number"
    case "boolean":
      return "boolean"
    case "array":
      return "array"
    case "object":
      return "object"
    case "string":
      if (schema.format === "date") return "date"
      if (schema.format === "date-time") return "datetime"
      return "string"
  }
  if (isObject(schema.properties)) return "object"
  return "string"
}

function enumValuesOf(schema: JsonObject): string[] {
  if (Array.isArray(schema.enum)) return schema.enum.map((v) => String(v))
  if ("const" in schema) return [String(schema.const)]
  return []
}

function objectFields(
  schema: JsonObject,
  root: JsonSchema,
  seen: ReadonlySet<string>
): FormField[] {
  const props = isObject(schema.properties) ? schema.properties : {}
  const required = new Set(
    Array.isArray(schema.required)
      ? schema.required.filter((r): r is string => typeof r === "string")
      : []
  )
  return Object.entries(props).map(([name, node]) =>
    toField(name, node, required.has(name), root, seen)
  )
}

function toField(
  name: string,
  node: JsonValue,
  isRequired: boolean,
  root: JsonSchema,
  seen: ReadonlySet<string>
): FormField {
  const r = resolve(node, root, seen)
  const type = typeOf(r.schema)
  const field = makeField({
    name,
    type,
    description: r.description ?? "",
    required: isRequired && !r.nullable,
  })
  if (type === "enum") field.enumValues = enumValuesOf(r.schema)
  if (type === "object") {
    field.children = objectFields(r.schema, root, r.seen)
  }
  if (type === "array") {
    const item = resolve(r.schema.items, root, r.seen)
    const itemType = typeOf(item.schema)
    field.itemType = itemType === "array" ? "string" : itemType
    if (field.itemType === "enum") field.enumValues = enumValuesOf(item.schema)
    if (field.itemType === "object") {
      field.children = objectFields(item.schema, root, item.seen)
    }
    // An array is never "required" in the form unless the schema says so.
    field.required = isRequired && !r.nullable
  }
  return field
}

/**
 * Read the fields of an object JSON Schema, including pydantic's
 * `model_json_schema()` output: `$defs`/`$ref`, `anyOf` with null, `allOf`
 * with a single ref, enum definitions, `title` and `description`.
 *
 * Constructs the form cannot show (unions of several types, extra keywords)
 * are simplified to their first variant.
 */
export function jsonSchemaToFields(js: JsonSchema): FormField[] {
  const root = resolve(js, js, new Set())
  return objectFields(root.schema, js, root.seen)
}

/** The `title` and `description` of a JSON Schema (the description may sit on a `$ref` target). */
export function jsonSchemaMeta(js: JsonSchema): {
  title: string
  description: string
} {
  const root = resolve(js, js, new Set())
  return {
    title: typeof js.title === "string" ? js.title : "",
    description: root.description ?? "",
  }
}

// ---------------------------------------------------------------------------
// Fields to Pydantic
// ---------------------------------------------------------------------------

const PY_KEYWORDS = new Set([
  "False",
  "None",
  "True",
  "and",
  "as",
  "assert",
  "async",
  "await",
  "break",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "try",
  "while",
  "with",
  "yield",
  "match",
  "case",
  "type",
  "model_config",
])

function pyString(text: string): string {
  return JSON.stringify(text)
}

function fieldIdent(name: string): string {
  let id = name.replace(/[^A-Za-z0-9_]/g, "_")
  if (!id || /^[0-9]/.test(id)) id = `_${id}`
  if (id.startsWith("__") || PY_KEYWORDS.has(id)) id = `${id}_`
  if (id.startsWith("_")) id = `f${id}`
  return id
}

const MAX_LINE = 88

/**
 * Pydantic source for a form.
 *
 * @param name - Class name of the record.
 * @param fields - The top-level fields.
 * @param description - Becomes the class docstring.
 */
export function fieldsToPydantic(
  name: string,
  fields: FormField[],
  description = ""
): string {
  const names = defNames(name, fields)
  const used = { date: false, datetime: false, literal: false, field: false }

  const scalar = (type: ItemType, f: FormField): string => {
    switch (type) {
      case "string":
        return "str"
      case "integer":
        return "int"
      case "number":
        return "float"
      case "boolean":
        return "bool"
      case "date":
        used.date = true
        return "date"
      case "datetime":
        used.datetime = true
        return "datetime"
      case "enum":
        if (f.enumValues.length === 0) return "str"
        used.literal = true
        return `Literal[${f.enumValues.map(pyString).join(", ")}]`
      case "object":
        return names.get(f.id) ?? "dict"
    }
  }

  const annotation = (f: FormField): string =>
    f.type === "array"
      ? `list[${scalar(f.itemType, f)}]`
      : scalar(f.type, f)

  const classes: string[] = []

  const emitClass = (
    className: string,
    list: FormField[],
    doc: string
  ): void => {
    // Nested models first, so that every name is defined before it is used.
    for (const f of list) {
      if (hasChildren(f) && f.name.trim()) {
        emitClass(names.get(f.id) ?? "Model", f.children, "")
      }
    }
    const lines: string[] = [`class ${className}(BaseModel):`]
    if (doc.trim()) {
      const body = doc.trim().replace(/\\/g, "\\\\").replace(/"""/g, '\\"\\"\\"')
      lines.push(`    """${body}"""`, "")
    }
    let count = 0
    for (const f of list) {
      const key = f.name.trim()
      if (!key) continue
      count += 1
      const ident = fieldIdent(key)
      const args: string[] = []
      const optionalList = !f.required && f.type === "array"
      const optionalScalar = !f.required && f.type !== "array"
      if (optionalList) args.push("default_factory=list")
      if (optionalScalar) args.push("None")
      if (f.description.trim()) {
        args.push(`description=${pyString(f.description.trim())}`)
      }
      if (ident !== key) args.push(`alias=${pyString(key)}`)
      let ann = annotation(f)
      if (optionalScalar) ann += " | None"
      const head = `    ${ident}: ${ann}`
      if (args.length === 0) {
        lines.push(head)
      } else if (args.length === 1 && args[0] === "None") {
        lines.push(`${head} = None`)
      } else {
        const oneLine = `${head} = Field(${args.join(", ")})`
        if (oneLine.length <= MAX_LINE) {
          lines.push(oneLine)
        } else {
          const inner = `        ${args.join(", ")}`
          if (inner.length <= MAX_LINE) {
            lines.push(`${head} = Field(`, inner, "    )")
          } else {
            lines.push(`${head} = Field(`)
            for (const a of args) lines.push(`        ${a},`)
            lines.push("    )")
          }
        }
        used.field = true
      }
    }
    if (count === 0 && !doc.trim()) lines.push("    pass")
    classes.push(lines.join("\n"))
  }

  emitClass(classIdent(name), fields, description)

  const imports: string[] = []
  const dt = [used.date ? "date" : "", used.datetime ? "datetime" : ""]
    .filter(Boolean)
    .join(", ")
  if (dt) imports.push(`from datetime import ${dt}`)
  if (used.literal) imports.push("from typing import Literal")
  if (imports.length > 0) imports.push("")
  imports.push(
    `from pydantic import ${used.field ? "BaseModel, Field" : "BaseModel"}`
  )
  return `${imports.join("\n")}\n\n\n${classes.join("\n\n\n")}\n`
}

// ---------------------------------------------------------------------------
// Pydantic parse results
// ---------------------------------------------------------------------------

/**
 * Drop the helper models from a list of parsed Pydantic models.
 *
 * `/api/schema/parse` returns every top-level `BaseModel`, including the
 * classes only used inside another model (they appear in its `$defs`). The
 * ones the others refer to are helpers, not schemas to extract.
 */
export function topLevelModels<T extends { name: string; json_schema: JsonSchema }>(
  models: T[]
): T[] {
  const referenced = new Set<string>()
  for (const m of models) {
    for (const key of ["$defs", "definitions"]) {
      const defs = m.json_schema[key]
      if (isObject(defs)) {
        for (const defName of Object.keys(defs)) referenced.add(defName)
      }
    }
  }
  const roots = models.filter((m) => !referenced.has(m.name))
  return roots.length > 0 ? roots : models
}
