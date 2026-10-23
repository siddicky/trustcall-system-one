import {
  authSecret,
  createApiToken,
  isAllowed,
  parseAllowlist,
  readCookie,
  SESSION_COOKIE,
  verifySessionToken,
} from "../../server/session.js"

/**
 * A bearer token for the backend, for an allowed signed-in user:
 * `{token, exp}`, 401 when signed out, 404 when the backend is not gated.
 */
export async function GET(request: Request): Promise<Response> {
  const apiSecret = process.env.DEMO_TOKEN_SECRET
  if (!apiSecret) {
    return Response.json({ detail: "API tokens are off." }, { status: 404 })
  }
  let session = null
  try {
    session = await verifySessionToken(
      readCookie(request, SESSION_COOKIE),
      authSecret()
    )
  } catch {
    // AUTH_SECRET missing: nobody is signed in.
  }
  if (
    !session ||
    !isAllowed(session.email, parseAllowlist(process.env.ALLOWED_EMAILS))
  ) {
    return Response.json({ detail: "Not signed in." }, { status: 401 })
  }
  return Response.json(await createApiToken(session.email, apiSecret), {
    headers: { "cache-control": "no-store" },
  })
}
