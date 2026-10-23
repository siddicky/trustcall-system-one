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

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, init)
  } catch (err) {
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
