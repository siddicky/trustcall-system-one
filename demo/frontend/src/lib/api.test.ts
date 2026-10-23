import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type Call = { url: string; auth: string | null }

let calls: Call[]

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  })
}

/** Load a fresh copy of the module, so its token cache starts empty. */
async function load() {
  vi.resetModules()
  return import("./api")
}

beforeEach(() => {
  calls = []
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function stubFetch(
  handler: (url: string, auth: string | null) => Response | Promise<Response>
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string, init?: RequestInit) => {
      const auth = new Headers(init?.headers).get("authorization")
      calls.push({ url: input, auth })
      return handler(input, auth)
    })
  )
}

describe("apiFetch auth", () => {
  it("sends the site's token as a bearer token and caches it", async () => {
    const exp = Date.now() / 1000 + 600
    stubFetch((url) =>
      url === "/api/auth/token" ? json({ token: "t1", exp }) : json({ ok: 1 })
    )
    const { apiFetch } = await load()
    await apiFetch("/api/health")
    await apiFetch("/api/health")
    expect(calls.filter((c) => c.url === "/api/auth/token")).toHaveLength(1)
    expect(calls.filter((c) => c.url.endsWith("/api/health"))).toEqual([
      expect.objectContaining({ auth: "Bearer t1" }),
      expect.objectContaining({ auth: "Bearer t1" }),
    ])
  })

  it("retries once with a fresh token after a 401", async () => {
    const exp = Date.now() / 1000 + 600
    let minted = 0
    stubFetch((url, auth) => {
      if (url === "/api/auth/token") return json({ token: `t${++minted}`, exp })
      return auth === "Bearer t2" ? json({ ok: 1 }) : json({}, 401)
    })
    const { apiFetch } = await load()
    await expect(apiFetch("/api/extract")).resolves.toEqual({ ok: 1 })
    expect(minted).toBe(2)
  })

  it("sends no token where the site mints none", async () => {
    stubFetch((url) =>
      url === "/api/auth/token"
        ? new Response("<!doctype html>", { status: 200 })
        : json({ ok: 1 })
    )
    const { apiFetch } = await load()
    await apiFetch("/api/health")
    await apiFetch("/api/health")
    expect(calls.filter((c) => c.url === "/api/auth/token")).toHaveLength(1)
    expect(calls.every((c) => c.auth === null)).toBe(true)
  })

  it("reports an expired sign-in", async () => {
    stubFetch((url) =>
      url === "/api/auth/token" ? json({}, 401) : json({ ok: 1 })
    )
    const { apiFetch } = await load()
    await expect(apiFetch("/api/extract")).rejects.toMatchObject({
      status: 401,
      code: "signed_out",
    })
  })
})
