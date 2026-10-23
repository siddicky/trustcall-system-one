# Trustcall System One demo: frontend

Vite, React, TypeScript and shadcn/ui. Deployed to Vercel (project
`trustcall-system-one-frontend`, root directory `demo/frontend`). It calls the
FastAPI backend in `demo/backend` at `VITE_API_URL`.

```bash
npm install
npm run dev        # http://localhost:5173, no sign-in
npm test
npm run build
```

## Google sign-in

On Vercel, every page sits behind Google sign-in, the same scheme as the
Siddicky client portal:

- `middleware.ts` (Vercel Routing Middleware) checks a signed `tc_session`
  cookie on every request. Signed-out page views go to `/signin`. Other
  requests get a 401. Public paths: `/signin`, `/api/auth/*` and the hashed
  build assets under `/assets/`.
- `api/auth/google/start.ts` and `api/auth/google/callback.ts` (Vercel
  Functions) run the OAuth code flow, check the ID token's issuer, audience
  and verified email against `ALLOWED_EMAILS`, and set a 7-day session.
- `api/auth/signout.ts` clears the session; `api/auth/me.ts` returns the
  signed-in account for the header.
- `server/session.ts` holds the HMAC session, allowlist and cookie helpers.

`npm run dev` (plain Vite) does not run the middleware or the functions, so
local development has no gate. Run `vercel dev` to try the gate locally.

### Environment variables (Vercel project settings)

| Name | Value |
| --- | --- |
| `GOOGLE_CLIENT_ID` | OAuth client ID (type "Web application") |
| `GOOGLE_CLIENT_SECRET` | Its secret |
| `AUTH_SECRET` | Random string that signs sessions: `openssl rand -hex 32`. Changing it signs everyone out. |
| `ALLOWED_EMAILS` | Comma-separated addresses (`me@example.com`) or domains (`@example.com`). Empty admits nobody. |
| `DEMO_TOKEN_SECRET` | Shared with the backend. `/api/auth/token` signs 10-minute bearer tokens with it, and `src/lib/api.ts` sends them on every backend call. |

A deployment without `AUTH_SECRET` answers 503 instead of opening up.
Removing an address from `ALLOWED_EMAILS` locks it out on its next request.

In Google Cloud Console, add each origin's callback to the client's
authorised redirect URIs, for example
`https://trustcall.siddicky.ca/api/auth/google/callback`. Preview deployments
use other hostnames, so Google sign-in only works on origins you register.

The backend at `VITE_API_URL` is on another origin and never sees the session
cookie, so it checks a bearer token instead: the frontend asks its own
`/api/auth/token` (signed-in, allowlisted users only) and attaches the token to
every backend request, retrying once with a fresh token after a 401.
