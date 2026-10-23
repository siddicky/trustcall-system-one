import {
  isHttps,
  randomState,
  safeNextPath,
  serializeCookie,
  STATE_COOKIE,
  STATE_MAX_AGE,
} from "../../../server/session.js"

/** Send the browser to Google's consent screen, remembering `state` and `next`. */
export function GET(request: Request): Response {
  const url = new URL(request.url)
  const clientId = process.env.GOOGLE_CLIENT_ID
  if (!clientId || !process.env.GOOGLE_CLIENT_SECRET) {
    return Response.redirect(new URL("/signin?error=config", url), 302)
  }

  const state = randomState()
  const nextPath = safeNextPath(url.searchParams.get("next"))
  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: `${url.origin}/api/auth/google/callback`,
    response_type: "code",
    scope: "openid email profile",
    state,
    prompt: "select_account",
  })
  return new Response(null, {
    status: 302,
    headers: {
      Location: `https://accounts.google.com/o/oauth2/v2/auth?${params}`,
      "Set-Cookie": serializeCookie(STATE_COOKIE, `${state}:${nextPath}`, {
        maxAge: STATE_MAX_AGE,
        secure: isHttps(request),
      }),
    },
  })
}
