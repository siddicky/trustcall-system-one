/**
 * Span offsets from the backend count Unicode code points (Python `str`
 * indices); JavaScript strings are indexed by UTF-16 code units. They differ
 * after the first character outside the Basic Multilingual Plane (an emoji, for
 * one), so every offset must be converted before it slices a string.
 */

export interface CodePointIndex {
  /** Number of code points in the text. */
  length: number
  /** The UTF-16 index of the code point at `cp`; past the end, the text length. */
  toUtf16: (cp: number) => number
}

const ASTRAL = /[\ud800-\udbff][\udc00-\udfff]/

/** Build the index of a text: one pass, then each conversion is a lookup. */
function build(text: string): CodePointIndex {
  if (!ASTRAL.test(text)) {
    // Every code point is one code unit: the offsets are the same.
    return {
      length: text.length,
      toUtf16: (cp) => (cp >= text.length ? text.length : cp),
    }
  }
  const starts: number[] = []
  let unit = 0
  for (const point of text) {
    starts.push(unit)
    unit += point.length
  }
  return {
    length: starts.length,
    toUtf16: (cp) => (cp >= starts.length ? text.length : starts[cp]),
  }
}

let lastText: string | null = null
let lastIndex: CodePointIndex | null = null

/**
 * The code point index of `text`, built once: asking again for the same text
 * (the usual case, one document per result) returns the cached index.
 */
export function codePointIndex(text: string): CodePointIndex {
  if (lastIndex === null || lastText !== text) {
    lastIndex = build(text)
    lastText = text
  }
  return lastIndex
}

/**
 * Convert a code point offset into a UTF-16 index of `text`.
 *
 * `cpToUtf16("😀 Dana", 2)` is `3`: the emoji is one code point but two code
 * units.
 */
export function cpToUtf16(text: string, cp: number): number {
  return codePointIndex(text).toUtf16(cp)
}
