/**
 * Turns an extraction result into the rows, evidence spans and highlights the
 * results view shows.
 *
 * Spans in a result are offsets into the text that was sent to `/api/extract`
 * (the library maps them from the model's state text to the input before it
 * returns them). They count Unicode code points, like Python string indices;
 * everything this module returns (`Evidence`, `Mark`) counts UTF-16 code units,
 * the indices of a JavaScript string, converted with `cpToUtf16`. Two cases
 * still need care and are handled here:
 *
 * - A list answer made of sentences has one span per sentence, while the
 *   record only gains some of them. The span of an added item is found by
 *   matching the item against the spans' text.
 * - A span can be missing or wrong for an answer. Then the value is located in
 *   the text by a case-insensitive search, and the evidence is marked
 *   `located`.
 */
import { jsonEqual, pointerOf } from "./format"
import { codePointIndex, cpToUtf16 } from "./offsets"
import type {
  FieldAnswer,
  JsonObject,
  JsonValue,
  ResultRecord,
  Span,
} from "./types"

export interface Evidence {
  /** UTF-16 index into the text. */
  start: number
  /** UTF-16 index into the text, exclusive. */
  end: number
  /** True when found by searching for the value instead of from a span. */
  located: boolean
  /** Number shown on the highlight, in text order across the record. */
  n: number
}

export interface Probability {
  value: number
  /** Which figure it is, for the tooltip. */
  label: string
}

export interface FieldRow {
  /** Key in `record.fields`. */
  key: string
  answer: FieldAnswer
  pointer: string
  before: JsonValue | undefined
  after: JsonValue | undefined
  changed: boolean
  p: Probability | null
  evidence: Evidence[]
  /** Short badges: RE-ASKED, REPAIRED, and so on. */
  flags: string[]
}

export interface Mark {
  /** UTF-16 index into the text. */
  start: number
  /** UTF-16 index into the text, exclusive. */
  end: number
  /** Row key for field marks; `mention` for entity mentions. */
  key: string
  kind: "field" | "mention"
  n?: number
}

export function getAtPath(
  doc: JsonObject | null | undefined,
  path: ReadonlyArray<string | number>
): JsonValue | undefined {
  let node: JsonValue | undefined = doc ?? undefined
  for (const part of path) {
    if (node === null || node === undefined || typeof node !== "object") {
      return undefined
    }
    node = Array.isArray(node)
      ? node[Number(part)]
      : (node as JsonObject)[String(part)]
  }
  return node
}

function mean(values: Array<number | null>): number | null {
  const nums = values.filter((v): v is number => typeof v === "number")
  if (nums.length === 0) return null
  return nums.reduce((a, b) => a + b, 0) / nums.length
}

export function probabilityOf(a: FieldAnswer): Probability | null {
  if (typeof a.gate_p === "number") {
    return { value: a.gate_p, label: "change gate" }
  }
  const attribution = Array.isArray(a.attribution_p)
    ? mean(a.attribution_p)
    : a.attribution_p
  if (typeof attribution === "number") {
    return { value: attribution, label: "attribution check" }
  }
  if (typeof a.confidence === "number") {
    return { value: a.confidence, label: "model confidence" }
  }
  if (typeof a.p_null === "number") {
    return { value: 1 - a.p_null, label: "1 minus P(null)" }
  }
  return null
}

function normalise(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim()
}

function looseMatch(slice: string, target: string): boolean {
  const a = normalise(slice)
  const b = normalise(target)
  if (!a || !b) return false
  return a.includes(b) || b.includes(a)
}

/** Whether `span` is a code point range inside a text of `length` code points. */
function validSpan(span: Span | null | undefined, length: number): span is Span {
  return (
    Array.isArray(span) &&
    Number.isInteger(span[0]) &&
    Number.isInteger(span[1]) &&
    span[0] >= 0 &&
    span[1] > span[0] &&
    span[1] <= length
  )
}

function locate(text: string, value: string): Span | null {
  const needle = value.trim()
  if (needle.length < 3) return null
  const at = text.toLowerCase().indexOf(needle.toLowerCase())
  return at < 0 ? null : [at, at + needle.length]
}

function isNil(v: JsonValue | undefined): boolean {
  return v === undefined || v === null
}

function addedItems(
  after: JsonValue[],
  before: JsonValue | undefined
): JsonValue[] {
  const old = Array.isArray(before) ? before : []
  return after.filter((item) => !old.some((o) => jsonEqual(item, o)))
}

type RawEvidence = Omit<Evidence, "n">

function evidenceOf(
  answer: FieldAnswer,
  before: JsonValue | undefined,
  after: JsonValue | undefined,
  changed: boolean,
  text: string
): RawEvidence[] {
  // A rejected answer keeps no evidence, unless the field still changed (a list
  // can gain one item and have others refused).
  if ((answer.rejected_reason && !changed) || isNil(after)) return []
  const spans: Span[] = (
    answer.spans && answer.spans.length > 0
      ? answer.spans
      : answer.span
        ? [answer.span]
        : []
  )
    .filter((s): s is Span => validSpan(s, codePointIndex(text).length))
    // From code points (the backend) to UTF-16 indices (the string).
    .map(([start, end]): Span => [cpToUtf16(text, start), cpToUtf16(text, end)])

  // What the evidence has to support.
  let targets: JsonValue[]
  if (Array.isArray(after)) {
    targets = addedItems(after, before)
    if (targets.length === 0) return []
  } else {
    targets = [after as JsonValue]
  }
  const strings = targets.filter((t): t is string => typeof t === "string")

  if (strings.length === 0) {
    // Numbers, booleans and dates: trust the span when the value changed.
    return changed
      ? spans.map(([start, end]) => ({ start, end, located: false }))
      : []
  }

  const out: RawEvidence[] = []
  const used = new Set<number>()
  for (const target of strings) {
    const at = spans.findIndex(
      (s, i) => !used.has(i) && looseMatch(text.slice(s[0], s[1]), target)
    )
    if (at >= 0) {
      used.add(at)
      out.push({ start: spans[at][0], end: spans[at][1], located: false })
      continue
    }
    if (answer.source === "default" || answer.source === "const") continue
    const found = locate(text, target)
    if (found) out.push({ start: found[0], end: found[1], located: true })
  }
  return out
}

function flagsOf(a: FieldAnswer, changed: boolean): string[] {
  const flags: string[] = []
  if (a.rejected_reason) {
    flags.push(
      `${changed ? "some refused" : "rejected"}: ${a.rejected_reason}`
    )
  }
  if (a.reask) flags.push("re-asked")
  if (a.repaired) flags.push("repaired")
  if (a.crosscheck_agree === false) flags.push("cross-check differed")
  if (a.provider) flags.push(`via ${a.provider}`)
  return flags
}

/** The rows of a record's field table, with evidence located in `text`. */
export function buildRows(record: ResultRecord, text: string): FieldRow[] {
  const rows: FieldRow[] = Object.entries(record.fields).map(([key, answer]) => {
    const before = getAtPath(record.before, answer.path)
    const after = getAtPath(record.after ?? record.value, answer.path)
    const changed = !jsonEqual(before ?? null, after ?? null)
    return {
      key,
      answer,
      pointer: pointerOf(answer.path),
      before,
      after,
      changed,
      p: probabilityOf(answer),
      evidence: [],
      flags: flagsOf(answer, changed),
    }
  })
  const rawByRow = rows.map((row) =>
    evidenceOf(row.answer, row.before, row.after, row.changed, text)
  )
  // Number the evidence in text order.
  const order = rawByRow
    .flatMap((list, r) => list.map((ev, i) => ({ r, i, start: ev.start })))
    .sort((a, b) => a.start - b.start)
  const numbers = new Map<string, number>()
  order.forEach((o, idx) => numbers.set(`${o.r}:${o.i}`, idx + 1))
  rows.forEach((row, r) => {
    row.evidence = rawByRow[r].map((ev, i) => ({
      ...ev,
      n: numbers.get(`${r}:${i}`) ?? 0,
    }))
  })
  // Changed rows first; the order is otherwise the field order.
  return [...rows].sort((a, b) => Number(b.changed) - Number(a.changed))
}

/** Highlights for a record: the evidence of its fields and its entity mentions. */
export function buildMarks(
  record: ResultRecord,
  rows: FieldRow[],
  text: string
): Mark[] {
  const marks: Mark[] = []
  for (const mention of record.entity?.mentions ?? []) {
    if (validSpan(mention.span, codePointIndex(text).length)) {
      marks.push({
        start: cpToUtf16(text, mention.span[0]),
        end: cpToUtf16(text, mention.span[1]),
        key: "mention",
        kind: "mention",
      })
    }
  }
  for (const row of rows) {
    for (const ev of row.evidence) {
      marks.push({
        start: ev.start,
        end: ev.end,
        key: row.key,
        kind: "field",
        n: ev.n,
      })
    }
  }
  return marks
}

/** The evidence with some context around it, cut at line breaks. */
export function quoteParts(
  text: string,
  ev: { start: number; end: number },
  window = 40
): { pre: string; hit: string; post: string } {
  const lineStart = text.lastIndexOf("\n", ev.start - 1) + 1
  const nextBreak = text.indexOf("\n", ev.end)
  const lineEnd = nextBreak < 0 ? text.length : nextBreak
  const from = Math.max(lineStart, ev.start - window)
  const to = Math.min(lineEnd, ev.end + window)
  let hit = text.slice(ev.start, ev.end)
  if (hit.length > 160) hit = `${hit.slice(0, 157)}...`
  return {
    pre: `${from > lineStart ? "..." : ""}${text.slice(from, ev.start)}`,
    hit,
    post: `${text.slice(ev.end, to)}${to < lineEnd ? "..." : ""}`,
  }
}

/** The best name for a record. */
export function recordTitle(record: ResultRecord): string {
  const doc = record.after ?? record.value ?? record.before
  if (doc) {
    for (const key of ["name", "title", "invoice_number", "id", "label"]) {
      const v = doc[key]
      if (typeof v === "string" && v.trim()) return v
    }
    for (const v of Object.values(doc)) {
      if (typeof v === "string" && v.trim()) return v
    }
  }
  const label = record.entity?.label
  if (label) return label
  return record.json_doc_id ?? record.schema_name
}

/** Mean probability of a record's changed fields and the weakest one. */
export function recordProbability(rows: FieldRow[]): {
  mean: number
  count: number
  weakest: { pointer: string; p: number } | null
} | null {
  const scored = rows.filter((r) => r.changed && r.p)
  if (scored.length === 0) return null
  const values = scored.map((r) => r.p!.value)
  let weakest = scored[0]
  for (const r of scored) if (r.p!.value < weakest.p!.value) weakest = r
  return {
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    count: scored.length,
    weakest: { pointer: weakest.pointer, p: weakest.p!.value },
  }
}
