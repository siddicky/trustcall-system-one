import {
  authSecret,
  readCookie,
  SESSION_COOKIE,
  verifySessionToken,
} from "../../server/session.js"

/** The signed-in user, for the header: `{email, name}` or 401. */
export async function GET(request: Request): Promise<Response> {
  let session = null
  try {
    session = await verifySessionToken(
      readCookie(request, SESSION_COOKIE),
      authSecret()
    )
  } catch {
    // AUTH_SECRET missing: nobody is signed in.
  }
  if (!session) {
    return Response.json({ detail: "Not signed in." }, { status: 401 })
  }
  return Response.json(
    { email: session.email, name: session.name },
    { headers: { "cache-control": "no-store" } }
  )
}
