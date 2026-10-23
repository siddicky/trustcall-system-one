import { describe, expect, it } from "vitest"

import pydanticInvoice from "./__fixtures__/pydantic-invoice.json"
import {
  addField,
  fieldsToJsonSchema,
  fieldsToPydantic,
  jsonSchemaMeta,
  jsonSchemaToFields,
  leafCount,
  makeField,
  removeField,
  retype,
  topLevelModels,
  updateField,
  validateFields,
  type FormField,
} from "./schema"
import type { JsonSchema } from "./types"

const FIXTURE = pydanticInvoice as unknown as JsonSchema

/** Compare forms without their ids. */
function strip(fields: FormField[]): unknown[] {
  return fields.map((f) => ({
    name: f.name,
    type: f.type,
    description: f.description,
    required: f.required,
    enumValues: f.enumValues,
    itemType: f.type === "array" ? f.itemType : undefined,
    children: strip(f.children),
  }))
}

function invoiceFields(): FormField[] {
  return [
    makeField({ name: "invoice_number", description: "Vendor id" }),
    makeField({ name: "issue_date", type: "date" }),
    makeField({
      name: "vendor",
      type: "object",
      children: [
        makeField({ name: "name" }),
        makeField({ name: "address", required: false }),
      ],
    }),
    makeField({
      name: "line_items",
      type: "array",
      itemType: "object",
      description: "One per billed line",
      children: [
        makeField({ name: "description" }),
        makeField({ name: "quantity", type: "integer" }),
        makeField({ name: "unit_price", type: "number" }),
      ],
    }),
    makeField({
      name: "currency",
      type: "enum",
      enumValues: ["USD", "EUR", "GBP"],
    }),
    makeField({ name: "paid", type: "boolean", required: false }),
    makeField({ name: "sent_at", type: "datetime", required: false }),
    makeField({ name: "notes", type: "array", required: false }),
  ]
}

describe("fieldsToJsonSchema", () => {
  const schema = fieldsToJsonSchema("Invoice", invoiceFields(), "An invoice.")
  const props = schema.properties as Record<string, JsonSchema>

  it("sets title, description and type", () => {
    expect(schema.title).toBe("Invoice")
    expect(schema.description).toBe("An invoice.")
    expect(schema.type).toBe("object")
  })

  it("lists required fields", () => {
    expect(schema.required).toEqual([
      "invoice_number",
      "issue_date",
      "vendor",
      "line_items",
      "currency",
    ])
  })

  it("maps date, datetime and enum", () => {
    expect(props.issue_date).toEqual({ type: "string", format: "date" })
    expect(props.currency).toEqual({
      type: "string",
      enum: ["USD", "EUR", "GBP"],
    })
    expect(props.sent_at).toEqual({
      anyOf: [{ type: "string", format: "date-time" }, { type: "null" }],
      default: null,
    })
  })

  it("emits optionals as anyOf with null, never a type list", () => {
    expect(props.paid).toEqual({
      anyOf: [{ type: "boolean" }, { type: "null" }],
      default: null,
    })
    expect(JSON.stringify(schema)).not.toMatch(/"type":\[/)
  })

  it("puts nested objects in $defs behind $ref", () => {
    expect(props.vendor).toEqual({ $ref: "#/$defs/Vendor" })
    expect(props.line_items).toEqual({
      type: "array",
      items: { $ref: "#/$defs/LineItem" },
      description: "One per billed line",
    })
    const defs = schema.$defs as Record<string, JsonSchema>
    expect(Object.keys(defs).sort()).toEqual(["LineItem", "Vendor"])
    expect(defs.Vendor.required).toEqual(["name"])
  })

  it("gives an optional list default []", () => {
    expect(props.notes).toEqual({ type: "array", items: { type: "string" }, default: [] })
  })

  it("omits required and $defs when empty", () => {
    const s = fieldsToJsonSchema("X", [makeField({ name: "a", required: false })])
    expect(s.required).toBeUndefined()
    expect(s.$defs).toBeUndefined()
  })

  it("skips unnamed fields", () => {
    const s = fieldsToJsonSchema("X", [makeField({ name: "  " })])
    expect(s.properties).toEqual({})
  })

  it("names optional objects with anyOf and a $ref", () => {
    const s = fieldsToJsonSchema("X", [
      makeField({
        name: "address",
        type: "object",
        required: false,
        children: [makeField({ name: "street" })],
      }),
    ])
    expect((s.properties as Record<string, JsonSchema>).address).toEqual({
      anyOf: [{ $ref: "#/$defs/Address" }, { type: "null" }],
      default: null,
    })
  })
})

describe("round trip", () => {
  it("fields -> JSON Schema -> fields", () => {
    const fields = invoiceFields()
    const back = jsonSchemaToFields(fieldsToJsonSchema("Invoice", fields))
    expect(strip(back)).toEqual(strip(fields))
  })

  it("keeps description and title", () => {
    const js = fieldsToJsonSchema("Invoice", [], "An invoice.")
    expect(jsonSchemaMeta(js)).toEqual({
      title: "Invoice",
      description: "An invoice.",
    })
  })

  it("handles a name collision between nested models", () => {
    const fields = [
      makeField({
        name: "billing",
        type: "object",
        children: [makeField({ name: "x" })],
      }),
      makeField({
        name: "billing",
        type: "object",
        children: [makeField({ name: "y" })],
      }),
    ]
    const js = fieldsToJsonSchema("T", fields)
    expect(Object.keys(js.$defs as object)).toEqual(["Billing", "Billing2"])
  })
})

describe("jsonSchemaToFields on pydantic output", () => {
  const fields = jsonSchemaToFields(FIXTURE)
  const byName = Object.fromEntries(fields.map((f) => [f.name, f]))

  it("reads every property in order", () => {
    expect(fields.map((f) => f.name)).toEqual([
      "invoice_number",
      "issue_date",
      "sent_at",
      "vendor",
      "ship_to",
      "line_items",
      "notes",
      "currency",
      "priority",
      "paid",
      "tags",
    ])
  })

  it("reads the root title and description", () => {
    expect(jsonSchemaMeta(FIXTURE)).toEqual({
      title: "Invoice",
      description: "A supplier invoice.",
    })
  })

  it("reads required, descriptions and scalar types", () => {
    expect(byName.invoice_number).toMatchObject({
      type: "string",
      required: true,
      description: "Vendor-assigned id",
    })
    expect(byName.issue_date).toMatchObject({ type: "date", required: true })
    expect(byName.paid).toMatchObject({ type: "boolean", required: false })
  })

  it("reads anyOf with null as optional", () => {
    expect(byName.sent_at).toMatchObject({ type: "datetime", required: false })
    expect(byName.tags).toMatchObject({
      type: "array",
      itemType: "string",
      required: false,
    })
  })

  it("resolves $ref to nested objects", () => {
    expect(byName.vendor.type).toBe("object")
    expect(byName.vendor.required).toBe(true)
    expect(byName.vendor.description).toBe("Postal address.")
    expect(byName.vendor.children.map((c) => [c.name, c.required])).toEqual([
      ["street", true],
      ["city", false],
    ])
    expect(byName.ship_to).toMatchObject({ type: "object", required: false })
  })

  it("resolves arrays of objects", () => {
    expect(byName.line_items).toMatchObject({
      type: "array",
      itemType: "object",
      required: true,
    })
    expect(byName.line_items.children.map((c) => c.type)).toEqual([
      "string",
      "integer",
      "number",
    ])
    expect(byName.line_items.children[0].description).toBe("What was billed")
  })

  it("reads Literal and Enum definitions as enum", () => {
    expect(byName.currency).toMatchObject({
      type: "enum",
      enumValues: ["USD", "EUR"],
    })
    expect(byName.priority).toMatchObject({
      type: "enum",
      enumValues: ["low", "high"],
      required: false,
    })
  })

  it("reads allOf with a single ref (older pydantic)", () => {
    const js: JsonSchema = {
      $defs: { A: { type: "object", properties: { x: { type: "string" } }, required: ["x"] } },
      type: "object",
      properties: { a: { allOf: [{ $ref: "#/$defs/A" }], description: "the a" } },
      required: ["a"],
    }
    const [a] = jsonSchemaToFields(js)
    expect(a).toMatchObject({ name: "a", type: "object", description: "the a" })
    expect(a.children[0]).toMatchObject({ name: "x", required: true })
  })

  it("reads type lists with null", () => {
    const js: JsonSchema = {
      type: "object",
      properties: { a: { type: ["string", "null"] } },
    }
    expect(jsonSchemaToFields(js)[0]).toMatchObject({ type: "string", required: false })
  })

  it("does not loop on recursive models", () => {
    const js: JsonSchema = {
      $defs: {
        Node: {
          type: "object",
          properties: { child: { $ref: "#/$defs/Node" } },
        },
      },
      $ref: "#/$defs/Node",
    }
    const fields = jsonSchemaToFields(js)
    expect(fields[0].name).toBe("child")
  })

  it("survives a second round through fieldsToJsonSchema", () => {
    const js = fieldsToJsonSchema("Invoice", fields)
    expect(strip(jsonSchemaToFields(js))).toEqual(strip(fields))
  })
})

describe("fieldsToPydantic", () => {
  it("renders the README Contact model", () => {
    const code = fieldsToPydantic(
      "Contact",
      [
        makeField({ name: "name" }),
        makeField({ name: "email" }),
        makeField({ name: "phone", required: false }),
        makeField({ name: "city", required: false }),
        makeField({ name: "company", required: false }),
        makeField({ name: "title", required: false }),
        makeField({
          name: "notes",
          type: "array",
          required: false,
          description: "things worth remembering",
        }),
      ],
      "A contact in the CRM."
    )
    expect(code).toBe(`from pydantic import BaseModel, Field


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    email: str
    phone: str | None = None
    city: str | None = None
    company: str | None = None
    title: str | None = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )
`)
  })

  it("renders nested models first, imports and Literal", () => {
    const code = fieldsToPydantic("Invoice", invoiceFields(), "An invoice.")
    expect(code).toContain("from datetime import date, datetime")
    expect(code).toContain("from typing import Literal")
    expect(code.indexOf("class Vendor")).toBeLessThan(code.indexOf("class Invoice"))
    expect(code.indexOf("class LineItem")).toBeLessThan(code.indexOf("class Invoice"))
    expect(code).toContain("    vendor: Vendor")
    expect(code).toContain("    line_items: list[LineItem] = Field(")
    expect(code).toContain('    currency: Literal["USD", "EUR", "GBP"]')
    expect(code).toContain("    sent_at: datetime | None = None")
    expect(code).toContain("    notes: list[str] = Field(default_factory=list)")
  })

  it("aliases names that are not Python identifiers", () => {
    const code = fieldsToPydantic("X", [makeField({ name: "class" }), makeField({ name: "due-date" })])
    expect(code).toContain('class_: str = Field(alias="class")')
    expect(code).toContain('due_date: str = Field(alias="due-date")')
  })

  it("writes pass for an empty model", () => {
    expect(fieldsToPydantic("X", [])).toContain("    pass")
  })
})

describe("tree helpers", () => {
  it("updates, adds and removes nested fields", () => {
    const child = makeField({ name: "c" })
    const parent = makeField({ name: "p", type: "object", children: [child] })
    let fields = [parent]
    fields = updateField(fields, child.id, (f) => ({ ...f, name: "renamed" }))
    expect(fields[0].children[0].name).toBe("renamed")
    const extra = makeField({ name: "d" })
    fields = addField(fields, parent.id, extra)
    expect(fields[0].children).toHaveLength(2)
    fields = removeField(fields, child.id)
    expect(fields[0].children.map((c) => c.name)).toEqual(["d"])
    expect(addField(fields, null, makeField({ name: "top" }))).toHaveLength(2)
  })

  it("counts leaf paths", () => {
    expect(leafCount(invoiceFields())).toBe(1 + 1 + 2 + 3 + 1 + 1 + 1 + 1)
  })

  it("validates names and required children", () => {
    expect(validateFields([makeField({ name: "" })])).toHaveLength(1)
    expect(validateFields([makeField({ name: "a" }), makeField({ name: "a" })])).toHaveLength(1)
    expect(validateFields([makeField({ name: "e", type: "enum" })])).toHaveLength(1)
    expect(validateFields([makeField({ name: "o", type: "object" })])).toHaveLength(1)
    expect(validateFields(invoiceFields())).toEqual([])
  })

  it("retypes an array's item type safely", () => {
    const f = retype(makeField({ name: "x" }), "array")
    expect(f.itemType).toBe("string")
  })
})

describe("topLevelModels", () => {
  it("drops models used by others", () => {
    const models = [
      { name: "Vendor", json_schema: { type: "object" } as JsonSchema },
      {
        name: "Invoice",
        json_schema: { $defs: { Vendor: {} } } as JsonSchema,
      },
    ]
    expect(topLevelModels(models).map((m) => m.name)).toEqual(["Invoice"])
  })

  it("keeps independent models", () => {
    const models = [
      { name: "A", json_schema: {} as JsonSchema },
      { name: "B", json_schema: {} as JsonSchema },
    ]
    expect(topLevelModels(models)).toHaveLength(2)
  })
})
