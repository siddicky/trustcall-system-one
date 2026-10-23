import type { JsonValue } from "./types"

export function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`
}

export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`
}

export function formatUsd(usd: number): string {
  if (usd === 0) return "$0"
  if (usd < 0.01) return `$${usd.toFixed(4)}`
  return `$${usd.toFixed(usd < 1 ? 3 : 2)}`
}

export function formatInt(n: number): string {
  return n.toLocaleString("en-US")
}

export function formatP(p: number): string {
  return p.toFixed(2)
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`
}

/** Compact one-line JSON for values shown inline. */
export function compactJson(value: JsonValue | undefined): string {
  if (value === undefined) return ""
  return JSON.stringify(value)
}

export function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

/** Structural equality of JSON values. */
export function jsonEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== typeof b || a === null || b === null) return false
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false
    }
    return a.every((v, i) => jsonEqual(v, b[i]))
  }
  if (typeof a === "object") {
    const ao = a as Record<string, unknown>
    const bo = b as Record<string, unknown>
    const keys = Object.keys(ao)
    if (keys.length !== Object.keys(bo).length) return false
    return keys.every((k) => k in bo && jsonEqual(ao[k], bo[k]))
  }
  return false
}

/** Escape a key for a JSON pointer (RFC 6901). */
export function pointerOf(path: ReadonlyArray<string | number>): string {
  return (
    "/" +
    path.map((p) => String(p).replace(/~/g, "~0").replace(/\//g, "~1")).join("/")
  )
}
