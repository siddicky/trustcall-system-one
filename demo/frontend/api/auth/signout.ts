import {
  isHttps,
  serializeCookie,
  SESSION_COOKIE,
} from "../../server/session.js"

/** Clear the session and return to the sign-in page. */
export function POST(request: Request): Response {
  const headers = new Headers({
    Location: new URL("/signin", request.url).toString(),
  })
  headers.append(
    "Set-Cookie",
    serializeCookie(SESSION_COOKIE, "", { maxAge: 0, secure: isHttps(request) })
  )
  return new Response(null, { status: 303, headers })
}
