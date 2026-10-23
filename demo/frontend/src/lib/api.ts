export const API_URL: string = (
  import.meta.env.VITE_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "")

/** Error thrown by `apiFetch` for non-2xx responses and network failures. */
export class ApiError extends Error {
  status: number
  code: string
  /** Source line of a `/api/schema/parse` error, when the backend sent one. */
  line: number | null

  constructor(
    status: number,
    code: string,
    message: string,
    line: number | null = null
  ) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.code = code
    this.line = line
  }
}

type ErrorBody = {
  detail?:
    | { code?: string; message?: string; line?: number | null }
    | string
    | Array<{ msg?: string; loc?: unknown[] }>
}

async function toApiError(res: Response): Promise<ApiError> {
  let code = "http_error"
  let message = res.statusText || `HTTP ${res.status}`
  let line: number | null = null
  try {
    const body = (await res.json()) as ErrorBody
    const detail = body.detail
    if (Array.isArray(detail)) {
      code = "invalid_request"
      message = detail
        .map((d) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? ""}`)
        .join("; ")
    } else if (detail && typeof detail === "object") {
      code = detail.code ?? code
      message = detail.message ?? message
      line = detail.line ?? null
    } else if (typeof detail === "string") {
      message = detail
    }
  } catch {
    // Body was not JSON; keep the defaults.
  }
  return new ApiError(res.status, code, message, line)
}

type ApiToken = { token: string; exp: number }

let cachedToken: ApiToken | null = null
// When the site mints no tokens, ask again only after this time (seconds).
let noTokensUntil = 0

/**
 * A bearer token for the backend from the site's own `/api/auth/token`, cached
 * until a minute before it expires. Null where the site mints none (`vite dev`,
 * or a backend without DEMO_TOKEN_SECRET).
 */
async function apiToken(fresh = false): Promise<string | null> {
  const now = Date.now() / 1000
  if (!fresh && cachedToken && cachedToken.exp - 60 > now) {
    return cachedToken.token
  }
  if (!fresh && noTokensUntil > now) return null
  cachedToken = null
  try {
    const res = await fetch("/api/auth/token", { cache: "no-store" })
    if (res.status === 401) {
      throw new ApiError(
        401,
        "signed_out",
        "Your sign-in has expired. Reload the page to sign in again."
      )
    }
    const body = res.ok ? ((await res.json()) as Partial<ApiToken>) : null
    if (typeof body?.token !== "string" || typeof body.exp !== "number") {
      noTokensUntil = now + 300
      return null
    }
    cachedToken = { token: body.token, exp: body.exp }
    return cachedToken.token
  } catch (err) {
    if (err instanceof ApiError) throw err
    noTokensUntil = now + 300
    return null
  }
}

async function send(
  path: string,
  init: RequestInit,
  token: string | null
): Promise<Response> {
  const headers = new Headers(init.headers)
  if (token) headers.set("Authorization", `Bearer ${token}`)
  return fetch(`${API_URL}${path}`, { ...init, headers })
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await send(path, init, await apiToken())
    // A token can expire in flight or be refused after a secret rotation.
    if (res.status === 401) res = await send(path, init, await apiToken(true))
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err instanceof DOMException && err.name === "AbortError") throw err
    throw new ApiError(
      0,
      "unreachable",
      `Could not reach the API at ${API_URL}.`
    )
  }
  if (!res.ok) throw await toApiError(res)
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

/**
 * Fetch JSON from the backend. Non-2xx responses throw an `ApiError` parsed
 * from `{"detail": {"code", "message"}}` bodies; a network failure throws an
 * `ApiError` with status 0 and code `unreachable`.
 */
export async function apiFetch<T>(
  path: string,
  init: Omit<RequestInit, "body"> & { json?: unknown } = {}
): Promise<T> {
  const { json, headers, ...rest } = init
  return request<T>(path, {
    ...rest,
    headers: {
      ...(json !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: json !== undefined ? JSON.stringify(json) : undefined,
  })
}

/** POST a file as the multipart field `file`. */
export async function apiUpload<T>(
  path: string,
  file: File,
  init: Omit<RequestInit, "body" | "method"> = {}
): Promise<T> {
  const form = new FormData()
  form.append("file", file)
  return request<T>(path, { ...init, method: "POST", body: form })
}
