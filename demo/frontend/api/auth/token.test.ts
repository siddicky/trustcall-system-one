import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { createSessionToken, SESSION_COOKIE } from "../../server/session.js"
import { GET } from "./token.js"

const AUTH = "auth-secret"

async function call(email?: string) {
  const headers = new Headers()
  if (email) {
    headers.set(
      "cookie",
      `${SESSION_COOKIE}=${await createSessionToken(email, "", AUTH)}`
    )
  }
  return GET(new Request("https://t.test/api/auth/token", { headers }))
}

beforeEach(() => {
  vi.stubEnv("AUTH_SECRET", AUTH)
  vi.stubEnv("DEMO_TOKEN_SECRET", "api-secret")
  vi.stubEnv("ALLOWED_EMAILS", "ada@example.com")
})

afterEach(() => {
  vi.unstubAllEnvs()
})

describe("GET /api/auth/token", () => {
  it("mints a token for an allowed session", async () => {
    const res = await call("ada@example.com")
    expect(res.status).toBe(200)
    const body = (await res.json()) as { token: string; exp: number }
    const payload = JSON.parse(
      Buffer.from(body.token.split(".")[0], "base64url").toString()
    )
    expect(payload).toMatchObject({
      email: "ada@example.com",
      aud: "trustcall-demo-api",
      exp: body.exp,
    })
  })

  it("refuses signed-out and non-allowlisted callers", async () => {
    expect((await call()).status).toBe(401)
    expect((await call("eve@example.com")).status).toBe(401)
  })

  it("answers 404 when the backend is not gated", async () => {
    vi.stubEnv("DEMO_TOKEN_SECRET", "")
    expect((await call("ada@example.com")).status).toBe(404)
  })
})
