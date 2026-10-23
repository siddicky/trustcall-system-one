import { next } from "@vercel/functions"

import {
  authSecret,
  isAllowed,
  isHttps,
  parseAllowlist,
  readCookie,
  safeNextPath,
  serializeCookie,
  SESSION_COOKIE,
  verifySessionToken,
} from "./server/session.js"

// Vercel Routing Middleware: every page of the demo sits behind Google
// sign-in. Only the sign-in page, the auth functions and the hashed build
// assets (which the sign-in page itself loads) are public.
const PUBLIC_PREFIXES = ["/api/auth/", "/assets/"]
// Under `vercel dev` the sign-in page loads from Vite's dev server instead of
// /assets/. These paths do not exist in a build.
const DEV_PUBLIC_PREFIXES = [
  "/@vite/",
  "/@react-refresh",
  "/@fs/",
  "/@id/",
  "/src/",
  "/node_modules/",
]
const SIGN_IN_PATHS = ["/signin", "/signin.html"]

function publicPrefixes(): string[] {
  // Set by `vercel dev` only; deployments never carry it.
  return process.env.__VERCEL_DEV_RUNNING === "1"
    ? [...PUBLIC_PREFIXES, ...DEV_PUBLIC_PREFIXES]
    : PUBLIC_PREFIXES
}

export const config = {
  runtime: "nodejs",
}

function wantsHtml(request: Request): boolean {
  return (
    request.method === "GET" &&
    (request.headers.get("accept") ?? "").includes("text/html")
  )
}

function signInRedirect(request: Request, url: URL, error?: string): Response {
  const target = new URL("/signin", url)
  const nextPath = url.pathname + url.search
  if (nextPath !== "/") target.searchParams.set("next", nextPath)
  if (error) target.searchParams.set("error", error)
  const headers = new Headers({ Location: target.toString() })
  if (error) {
    headers.append(
      "Set-Cookie",
      serializeCookie(SESSION_COOKIE, "", {
        maxAge: 0,
        secure: isHttps(request),
      })
    )
  }
  return new Response(null, { status: 302, headers })
}

export default async function middleware(request: Request): Promise<Response> {
  const url = new URL(request.url)
  const { pathname } = url
  if (publicPrefixes().some((prefix) => pathname.startsWith(prefix))) {
    return next()
  }

  let secret: string
  try {
    secret = authSecret()
  } catch {
    return new Response(
      "Sign-in is not configured (AUTH_SECRET is missing). Set it in the Vercel project's environment variables.",
      { status: 503, headers: { "content-type": "text/plain" } }
    )
  }

  const session = await verifySessionToken(
    readCookie(request, SESSION_COOKIE),
    secret
  )
  const allowed =
    session !== null &&
    isAllowed(session.email, parseAllowlist(process.env.ALLOWED_EMAILS))

  if (SIGN_IN_PATHS.includes(pathname)) {
    // Already signed in: skip the sign-in page.
    if (allowed) {
      const dest = safeNextPath(url.searchParams.get("next"))
      return Response.redirect(new URL(dest, url), 302)
    }
    return next()
  }

  if (allowed) return next()

  // A valid session for an address since removed from ALLOWED_EMAILS.
  const error = session ? "not_allowed" : undefined
  if (wantsHtml(request)) return signInRedirect(request, url, error)
  return new Response("Sign in required.", {
    status: 401,
    headers: { "content-type": "text/plain" },
  })
}
