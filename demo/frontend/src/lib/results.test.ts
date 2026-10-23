import { describe, expect, it } from "vitest"

import live from "./__fixtures__/contact-result.json"
import { codePointIndex, cpToUtf16 } from "./offsets"
import { buildMarks, buildRows, quoteParts, recordProbability, recordTitle } from "./results"
import type { ExtractResponse } from "./types"

const { text, result } = live as unknown as { text: string; result: ExtractResponse }
const record = result.records[0]
const rows = buildRows(record, text)
const byPointer = Object.fromEntries(rows.map((r) => [r.pointer, r]))

describe("buildRows on a live Contact update", () => {
  it("lists changed rows first", () => {
    const changed = rows.map((r) => r.changed)
    expect(changed.indexOf(false)).toBeGreaterThan(changed.lastIndexOf(true))
    expect(rows.filter((r) => r.changed).map((r) => r.pointer).sort()).toEqual(
      ["/city", "/company", "/notes", "/phone", "/title"]
    )
  })

  it("maps spans to text of the input", () => {
    const quote = (p: string) =>
      byPointer[p].evidence.map((e) => text.slice(e.start, e.end))
    expect(quote("/phone")).toEqual(["+49 30 5555 0198"])
    expect(quote("/city")).toEqual(["Berlin"])
    expect(quote("/company")).toEqual(["Northwind"])
    expect(quote("/title")).toEqual(["Head of Ops"])
  })

  it("picks the sentence of the added list item only", () => {
    expect(byPointer["/notes"].evidence.map((e) => text.slice(e.start, e.end))).toEqual([
      "She also asked to be introduced to our logistics team.",
    ])
  })

  it("gives no evidence to a rejected answer", () => {
    expect(byPointer["/email"].evidence).toEqual([])
    expect(byPointer["/email"].flags).toContain("rejected: echo_span")
  })

  it("numbers evidence in text order", () => {
    const numbered = rows.flatMap((r) => r.evidence).sort((a, b) => a.start - b.start)
    expect(numbered.map((e) => e.n)).toEqual(numbered.map((_, i) => i + 1))
  })

  it("uses the change gate as the probability when there is one", () => {
    expect(byPointer["/phone"].p).toEqual({ value: 0.9818, label: "change gate" })
  })
})

describe("record helpers", () => {
  it("titles, marks and probability", () => {
    expect(recordTitle(record)).toBe("Dana Whitfield")
    const marks = buildMarks(record, rows, text)
    expect(marks.every((m) => m.end > m.start && m.end <= text.length)).toBe(true)
    const p = recordProbability(rows)
    expect(p?.count).toBe(5)
    expect(p?.mean).toBeGreaterThan(0.9)
  })

  it("quotes with context cut at line breaks", () => {
    const q = quoteParts("one\ntwo three four", { start: 8, end: 13 }, 3)
    expect(q).toEqual({ pre: "...wo ", hit: "three", post: " fo..." })
  })

  it("locates a value when the span is missing", () => {
    const rec = structuredClone(record)
    rec.fields.city.span = null
    const r = buildRows(rec, text).find((x) => x.pointer === "/city")!
    expect(r.evidence[0].located).toBe(true)
    expect(text.slice(r.evidence[0].start, r.evidence[0].end)).toBe("Berlin")
  })
})

describe("code point offsets from the backend", () => {
  // Python counts "😀" as one character, JavaScript as two UTF-16 code units.
  const emojiText = "😀 Dana"

  it("converts code point offsets to UTF-16 indices", () => {
    expect(cpToUtf16(emojiText, 0)).toBe(0)
    expect(cpToUtf16(emojiText, 1)).toBe(2)
    expect(cpToUtf16(emojiText, 2)).toBe(3)
    expect(cpToUtf16(emojiText, 6)).toBe(emojiText.length)
    expect(cpToUtf16(emojiText, 99)).toBe(emojiText.length)
    expect(codePointIndex(emojiText).length).toBe(6)
    // Text without astral characters is the identity.
    expect(cpToUtf16("Dana", 3)).toBe(3)
    expect(codePointIndex("Dana").length).toBe(4)
  })

  function recordFor(span: [number, number], mention: [number, number] | null) {
    const rec = structuredClone(record)
    rec.before = { name: null }
    rec.after = { name: "Dana" }
    rec.value = { name: "Dana" }
    rec.fields = {
      name: {
        ...structuredClone(record.fields.city),
        path: ["name"],
        pretty: "name",
        wire_name: "name",
        value: "Dana",
        raw: "Dana",
        span,
        spans: null,
      },
    }
    rec.entity = {
      ...record.entity!,
      mentions: mention ? [{ text: "Dana", span: mention }] : [],
    }
    return rec
  }

  it("slices the evidence of a span that follows an emoji", () => {
    const [row] = buildRows(recordFor([2, 6], null), emojiText)
    const [ev] = row.evidence
    expect(emojiText.slice(ev.start, ev.end)).toBe("Dana")
    expect(ev.located).toBe(false)
  })

  it("marks the mention of an entity after an emoji", () => {
    const rec = recordFor([2, 6], [2, 6])
    const marks = buildMarks(rec, buildRows(rec, emojiText), emojiText)
    expect(marks.map((m) => emojiText.slice(m.start, m.end))).toEqual(["Dana", "Dana"])
  })

  it("rejects a span that is out of range in code points", () => {
    // [2, 7] is inside the 7 UTF-16 units but past the 6 code points.
    const [row] = buildRows(recordFor([2, 7], null), emojiText)
    const [ev] = row.evidence
    expect(ev.located).toBe(true) // found by searching for the value instead
    expect(emojiText.slice(ev.start, ev.end)).toBe("Dana")
  })
})
