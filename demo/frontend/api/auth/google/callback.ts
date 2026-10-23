import {
  authSecret,
  createSessionToken,
  isAllowed,
  isHttps,
  parseAllowlist,
  readCookie,
  safeNextPath,
  serializeCookie,
  SESSION_COOKIE,
  SESSION_MAX_AGE,
  STATE_COOKIE,
} from "../../../server/session.js"

type IdTokenClaims = {
  iss?: string
  aud?: string
  email?: string
  email_verified?: boolean
  name?: string
}

const GOOGLE_ISSUERS = ["https://accounts.google.com", "accounts.google.com"]

function failure(request: Request, url: URL, error: string): Response {
  const headers = new Headers({
    Location: new URL(`/signin?error=${error}`, url).toString(),
  })
  headers.append(
    "Set-Cookie",
    serializeCookie(STATE_COOKIE, "", { maxAge: 0, secure: isHttps(request) })
  )
  return new Response(null, { status: 302, headers })
}

/** Exchange Google's code for an ID token, check it, and set the session. */
export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url)
  const clientId = process.env.GOOGLE_CLIENT_ID
  const clientSecret = process.env.GOOGLE_CLIENT_SECRET
  if (!clientId || !clientSecret) return failure(request, url, "config")

  let secret: string
  try {
    secret = authSecret()
  } catch {
    return failure(request, url, "config")
  }

  const code = url.searchParams.get("code")
  const state = url.searchParams.get("state")
  const stored = readCookie(request, STATE_COOKIE) ?? ""
  const sep = stored.indexOf(":")
  const expectedState = sep > 0 ? stored.slice(0, sep) : ""
  if (!code || !state || !expectedState || state !== expectedState) {
    return failure(request, url, "state")
  }

  const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: `${url.origin}/api/auth/google/callback`,
      grant_type: "authorization_code",
    }),
  })
  const tokens = (await tokenRes.json().catch(() => ({}))) as {
    id_token?: string
  }
  if (!tokenRes.ok || !tokens.id_token) return failure(request, url, "exchange")

  // The ID token came straight from Google's token endpoint over TLS, in
  // exchange for our client secret, so its signature needs no second check
  // (OpenID Connect Core 3.1.3.7). Issuer, audience and email are checked.
  let claims: IdTokenClaims
  try {
    const part = tokens.id_token.split(".")[1]
    claims = JSON.parse(
      Buffer.from(part, "base64url").toString("utf8")
    ) as IdTokenClaims
  } catch {
    return failure(request, url, "exchange")
  }
  if (!GOOGLE_ISSUERS.includes(claims.iss ?? "") || claims.aud !== clientId) {
    return failure(request, url, "exchange")
  }
  if (!claims.email || claims.email_verified !== true) {
    return failure(request, url, "unverified")
  }
  if (!isAllowed(claims.email, parseAllowlist(process.env.ALLOWED_EMAILS))) {
    return failure(request, url, "not_allowed")
  }

  const secure = isHttps(request)
  const nextPath = safeNextPath(stored.slice(sep + 1))
  const headers = new Headers({ Location: new URL(nextPath, url).toString() })
  headers.append(
    "Set-Cookie",
    serializeCookie(
      SESSION_COOKIE,
      await createSessionToken(claims.email, claims.name ?? "", secret),
      { maxAge: SESSION_MAX_AGE, secure }
    )
  )
  headers.append(
    "Set-Cookie",
    serializeCookie(STATE_COOKIE, "", { maxAge: 0, secure })
  )
  return new Response(null, { status: 302, headers })
}
