import { afterEach, describe, expect, it, vi } from "vitest"

import {
  authSecret,
  createApiToken,
  createSessionToken,
  isAllowed,
  parseAllowlist,
  readCookie,
  safeNextPath,
  serializeCookie,
  verifySessionToken,
} from "./session.js"

const SECRET = "test-secret"

afterEach(() => {
  vi.useRealTimers()
})

describe("session tokens", () => {
  it("round-trips a signed session", async () => {
    const token = await createSessionToken("Ada@Example.com", "Ada", SECRET)
    const session = await verifySessionToken(token, SECRET)
    expect(session).toMatchObject({ email: "Ada@Example.com", name: "Ada" })
  })

  it("rejects a token signed with another secret", async () => {
    const token = await createSessionToken("ada@example.com", "Ada", "other")
    expect(await verifySessionToken(token, SECRET)).toBeNull()
  })

  it("rejects a payload swapped under a valid signature", async () => {
    const token = await createSessionToken("ada@example.com", "Ada", SECRET)
    const signature = token.slice(token.lastIndexOf(".") + 1)
    const forged = btoa(
      JSON.stringify({ email: "eve@example.com", name: "Eve", exp: 2e9 })
    )
      .replace(/=+$/, "")
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
    expect(
      await verifySessionToken(`${forged}.${signature}`, SECRET)
    ).toBeNull()
  })

  it("rejects an expired session", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"))
    const token = await createSessionToken("ada@example.com", "Ada", SECRET)
    vi.setSystemTime(new Date("2026-01-09T00:00:00Z"))
    expect(await verifySessionToken(token, SECRET)).toBeNull()
  })

  it("rejects missing and malformed tokens", async () => {
    expect(await verifySessionToken(undefined, SECRET)).toBeNull()
    expect(await verifySessionToken("", SECRET)).toBeNull()
    expect(await verifySessionToken("no-dot", SECRET)).toBeNull()
    expect(await verifySessionToken(".abc", SECRET)).toBeNull()
  })

  it("keeps non-ASCII names intact", async () => {
    const token = await createSessionToken("zoe@example.com", "Zoë", SECRET)
    expect((await verifySessionToken(token, SECRET))?.name).toBe("Zoë")
  })
})

describe("authSecret", () => {
  it("uses AUTH_SECRET when set", () => {
    expect(authSecret({ AUTH_SECRET: "s", VERCEL: "1" })).toBe("s")
  })

  it("fails closed on a Vercel deployment without AUTH_SECRET", () => {
    expect(() => authSecret({ VERCEL: "1" })).toThrow(/AUTH_SECRET/)
    expect(() => authSecret({ NODE_ENV: "production" })).toThrow(/AUTH_SECRET/)
  })

  it("falls back to a dev secret locally", () => {
    expect(authSecret({})).toBe("insecure-dev-secret")
  })
})

describe("allowlist", () => {
  const list = parseAllowlist(" Ada@Example.com, @team.dev ,,")

  it("parses addresses and domains, case-insensitively", () => {
    expect(list).toEqual(["ada@example.com", "@team.dev"])
  })

  it("admits listed addresses and domains", () => {
    expect(isAllowed("ADA@example.com", list)).toBe(true)
    expect(isAllowed("bob@team.dev", list)).toBe(true)
  })

  it("rejects everyone else, including look-alike domains", () => {
    expect(isAllowed("eve@example.com", list)).toBe(false)
    expect(isAllowed("eve@evilteam.dev", list)).toBe(false)
    expect(isAllowed("eve@team.dev.evil.com", list)).toBe(false)
  })

  it("admits nobody when the allowlist is empty", () => {
    expect(isAllowed("ada@example.com", parseAllowlist(undefined))).toBe(false)
  })
})

describe("safeNextPath", () => {
  it.each([
    ["/", "/"],
    ["/?tab=results", "/?tab=results"],
    [null, "/"],
    ["https://evil.com", "/"],
    ["//evil.com", "/"],
    ["/\\evil.com", "/"],
    ["/ok\r\nSet-Cookie: x", "/"],
    ["/\t/evil.com", "/"],
    ["/\n/evil.com", "/"],
    ["/%09/evil.com", "/%09/evil.com"],
    ["/a/../b?x=1#h", "/b?x=1#h"],
  ])("%s -> %s", (raw, expected) => {
    expect(safeNextPath(raw)).toBe(expected)
  })
})

describe("cookies", () => {
  it("reads one cookie from the header", () => {
    const request = new Request("https://x.test/", {
      headers: { cookie: "a=1; tc_session=abc.def; b=2" },
    })
    expect(readCookie(request, "tc_session")).toBe("abc.def")
    expect(readCookie(request, "missing")).toBeNull()
  })

  it("treats a malformed cookie value as absent", () => {
    const request = new Request("https://x.test/", {
      headers: { cookie: "tc_session=%E0%A4%A" },
    })
    expect(readCookie(request, "tc_session")).toBeNull()
  })

  it("serialises HttpOnly, SameSite=Lax cookies", () => {
    expect(serializeCookie("n", "v:/", { maxAge: 60, secure: true })).toBe(
      "n=v%3A%2F; Path=/; Max-Age=60; HttpOnly; SameSite=Lax; Secure"
    )
  })
})

describe("createApiToken", () => {
  it("signs {email, aud, exp} with the API secret", async () => {
    const { token, exp } = await createApiToken("ada@example.com", "api")
    const [payload, signature] = token.split(".")
    const { createHmac } = await import("node:crypto")
    expect(signature).toBe(
      createHmac("sha256", "api").update(payload).digest("hex")
    )
    expect(JSON.parse(Buffer.from(payload, "base64url").toString())).toEqual({
      email: "ada@example.com",
      aud: "trustcall-demo-api",
      exp,
    })
  })
})
