// Google sign-in session for the demo, shared by the Routing Middleware
// (`middleware.ts`) and the auth functions under `api/auth/`. A session is a
// `base64url(json).hex(hmac-sha256)` cookie signed with AUTH_SECRET, the same
// scheme as the Siddicky client portal. WebCrypto only, so it runs anywhere.

export const SESSION_COOKIE = "tc_session"
export const STATE_COOKIE = "tc_oauth_state"
export const SESSION_MAX_AGE = 7 * 24 * 3600 // 7 days
export const STATE_MAX_AGE = 600 // 10 minutes

export type Session = { email: string; name: string; exp: number }

type Env = Record<string, string | undefined>

const DEV_SECRET = "insecure-dev-secret"

function isProduction(env: Env): boolean {
  return env.VERCEL_ENV === "production" || env.NODE_ENV === "production"
}

/**
 * The HMAC key. Fails closed on a deployment: without AUTH_SECRET, sessions
 * would be signed with a publicly known constant.
 */
export function authSecret(env: Env = process.env): string {
  const secret = env.AUTH_SECRET
  if (secret) return secret
  if (env.VERCEL || isProduction(env)) {
    throw new Error("AUTH_SECRET must be set on a deployment.")
  }
  return DEV_SECRET
}

function toHex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (b) =>
    b.toString(16).padStart(2, "0")
  ).join("")
}

function toBase64Url(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ""
  for (const b of bytes) binary += String.fromCharCode(b)
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
}

function fromBase64Url(text: string): string {
  const b64 = text.replace(/-/g, "+").replace(/_/g, "/")
  const binary = atob(b64 + "=".repeat((4 - (b64.length % 4)) % 4))
  return new TextDecoder().decode(
    Uint8Array.from(binary, (c) => c.charCodeAt(0))
  )
}

async function sign(payload: string, secret: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  )
  return toHex(
    await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload))
  )
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false
  let diff = 0
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i)
  return diff === 0
}

function nowSeconds(): number {
  return Math.floor(Date.now() / 1000)
}

/** Sign a session for `email`, valid for SESSION_MAX_AGE seconds. */
export async function createSessionToken(
  email: string,
  name: string,
  secret: string
): Promise<string> {
  const session: Session = {
    email,
    name: name || email,
    exp: nowSeconds() + SESSION_MAX_AGE,
  }
  const payload = toBase64Url(JSON.stringify(session))
  return `${payload}.${await sign(payload, secret)}`
}

/** Return the session in `token`, or null when it is forged, malformed or expired. */
export async function verifySessionToken(
  token: string | null | undefined,
  secret: string
): Promise<Session | null> {
  if (!token) return null
  const dot = token.lastIndexOf(".")
  if (dot < 1) return null
  const payload = token.slice(0, dot)
  const signature = token.slice(dot + 1)
  if (!constantTimeEqual(signature, await sign(payload, secret))) return null
  try {
    const data = JSON.parse(fromBase64Url(payload)) as Partial<Session>
    if (typeof data.email !== "string" || typeof data.exp !== "number") {
      return null
    }
    if (data.exp < nowSeconds()) return null
    return { email: data.email, name: data.name ?? data.email, exp: data.exp }
  } catch {
    return null
  }
}

/**
 * Parse ALLOWED_EMAILS: comma-separated addresses (`me@example.com`) or whole
 * domains (`@example.com`), case-insensitive.
 */
export function parseAllowlist(raw: string | undefined): string[] {
  return (raw ?? "")
    .split(",")
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean)
}

/**
 * Whether `email` may use the demo. An empty allowlist admits nobody: the demo
 * spends model credits, so a missing setting must not open it to every Google
 * account.
 */
export function isAllowed(email: string, allowlist: string[]): boolean {
  const lower = email.toLowerCase()
  const domain = lower.slice(lower.lastIndexOf("@"))
  return allowlist.some((entry) =>
    entry.startsWith("@") ? entry === domain : entry === lower
  )
}

/** A random hex string for the OAuth `state` parameter. */
export function randomState(): string {
  return toHex(crypto.getRandomValues(new Uint8Array(24)).buffer)
}

/**
 * Keep a post-sign-in redirect on this site. The URL parser drops tabs and
 * newlines and reads `\` as `/`, so `/\t/host` would become `//host`: reject
 * control characters and backslashes, then confirm the path stays same-origin.
 */
export function safeNextPath(raw: string | null | undefined): string {
  // eslint-disable-next-line no-control-regex
  if (!raw || !raw.startsWith("/") || /[\u0000-\u001f\u007f\\]/.test(raw)) {
    return "/"
  }
  const base = "https://same-origin.invalid"
  try {
    const url = new URL(raw, base)
    if (url.origin !== base) return "/"
    return url.pathname + url.search + url.hash
  } catch {
    return "/"
  }
}

/** Read one cookie from a `Cookie` request header. */
export function readCookie(request: Request, name: string): string | null {
  const header = request.headers.get("cookie")
  if (!header) return null
  for (const part of header.split(";")) {
    const eq = part.indexOf("=")
    if (eq < 0) continue
    if (part.slice(0, eq).trim() === name) {
      try {
        return decodeURIComponent(part.slice(eq + 1).trim())
      } catch {
        return null // Malformed percent-encoding: treat as absent.
      }
    }
  }
  return null
}

/** Serialise an HttpOnly, SameSite=Lax cookie; `maxAge` 0 deletes it. */
export function serializeCookie(
  name: string,
  value: string,
  { maxAge, secure }: { maxAge: number; secure: boolean }
): string {
  const parts = [
    `${name}=${encodeURIComponent(value)}`,
    "Path=/",
    `Max-Age=${maxAge}`,
    "HttpOnly",
    "SameSite=Lax",
  ]
  if (secure) parts.push("Secure")
  return parts.join("; ")
}

/** Whether the request reached us over HTTPS (directly or via the proxy). */
export function isHttps(request: Request): boolean {
  return (
    request.headers.get("x-forwarded-proto") === "https" ||
    new URL(request.url).protocol === "https:"
  )
}
