import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import middleware from "./middleware.js"
import { createSessionToken, SESSION_COOKIE } from "./server/session.js"

const SECRET = "test-secret"
const ORIGIN = "https://trustcall.test"

function get(path: string, init: { cookie?: string; html?: boolean } = {}) {
  const headers = new Headers()
  if (init.cookie) headers.set("cookie", init.cookie)
  if (init.html !== false) headers.set("accept", "text/html")
  return new Request(`${ORIGIN}${path}`, { headers })
}

async function sessionCookie(email: string, secret = SECRET) {
  return `${SESSION_COOKIE}=${await createSessionToken(email, "", secret)}`
}

/** `next()` from @vercel/functions marks pass-through with this header. */
function passedThrough(res: Response) {
  return res.headers.get("x-middleware-next") === "1"
}

beforeEach(() => {
  vi.stubEnv("AUTH_SECRET", SECRET)
  vi.stubEnv("ALLOWED_EMAILS", "ada@example.com,@team.dev")
  vi.stubEnv("VERCEL", "1")
  vi.stubEnv("VERCEL_ENV", "production")
})

afterEach(() => {
  vi.unstubAllEnvs()
})

describe("middleware", () => {
  it("redirects a signed-out page view to /signin with next", async () => {
    const res = await middleware(get("/?tab=results"))
    expect(res.status).toBe(302)
    const location = new URL(res.headers.get("location")!)
    expect(location.pathname).toBe("/signin")
    expect(location.searchParams.get("next")).toBe("/?tab=results")
  })

  it("answers 401 to a signed-out non-page request", async () => {
    const res = await middleware(get("/data.json", { html: false }))
    expect(res.status).toBe(401)
  })

  it("lets an allowed session through", async () => {
    const cookie = await sessionCookie("ada@example.com")
    expect(passedThrough(await middleware(get("/", { cookie })))).toBe(true)
    const domain = await sessionCookie("bob@team.dev")
    expect(passedThrough(await middleware(get("/", { cookie: domain })))).toBe(
      true
    )
  })

  it("rejects a session signed with another secret", async () => {
    const cookie = await sessionCookie("ada@example.com", "other")
    const res = await middleware(get("/", { cookie }))
    expect(res.status).toBe(302)
  })

  it("signs out a session whose address left the allowlist", async () => {
    const cookie = await sessionCookie("eve@example.com")
    const res = await middleware(get("/", { cookie }))
    expect(res.status).toBe(302)
    expect(res.headers.get("location")).toContain("error=not_allowed")
    expect(res.headers.get("set-cookie")).toContain(`${SESSION_COOKIE}=;`)
  })

  it("keeps the sign-in page, auth functions and assets public", async () => {
    for (const path of [
      "/signin",
      "/signin.html",
      "/api/auth/google/start",
      "/assets/main-abc.js",
    ]) {
      expect(passedThrough(await middleware(get(path))), path).toBe(true)
    }
  })

  it("does not treat look-alike paths as public", async () => {
    for (const path of ["/signin-x", "/api/authx", "/assetsx/a.js"]) {
      expect((await middleware(get(path))).status, path).toBe(302)
    }
  })

  it("gates Vite dev-server paths on a deployment", async () => {
    for (const path of ["/src/main.tsx", "/@vite/client"]) {
      expect((await middleware(get(path))).status, path).toBe(302)
    }
  })

  it("opens Vite dev-server paths under vercel dev", async () => {
    vi.stubEnv("__VERCEL_DEV_RUNNING", "1")
    for (const path of ["/src/signin.tsx", "/@vite/client"]) {
      expect(passedThrough(await middleware(get(path))), path).toBe(true)
    }
  })

  it("gates Vite dev-server paths when no environment is set", async () => {
    vi.stubEnv("VERCEL_ENV", "")
    vi.stubEnv("VERCEL", "")
    expect((await middleware(get("/src/main.tsx"))).status).toBe(302)
  })

  it("does not send a signed-in visitor off-site via next", async () => {
    const cookie = await sessionCookie("ada@example.com")
    const res = await middleware(get("/signin?next=/%09/evil.com", { cookie }))
    expect(new URL(res.headers.get("location")!).origin).toBe(ORIGIN)
  })

  it("sends a signed-in visitor from /signin to next", async () => {
    const cookie = await sessionCookie("ada@example.com")
    const res = await middleware(get("/signin?next=/x", { cookie }))
    expect(res.status).toBe(302)
    expect(res.headers.get("location")).toBe(`${ORIGIN}/x`)
  })

  it("fails closed when AUTH_SECRET is missing on a deployment", async () => {
    vi.stubEnv("AUTH_SECRET", "")
    const res = await middleware(get("/"))
    expect(res.status).toBe(503)
  })
})
