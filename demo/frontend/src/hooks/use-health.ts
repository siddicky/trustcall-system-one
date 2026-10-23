import { useCallback, useEffect, useRef, useState } from "react"

import { apiFetch, ApiError } from "@/lib/api"
import type { Health, Provider } from "@/lib/types"

export type HealthState =
  | { status: "checking"; version: string | null }
  | { status: "connected"; provider: Provider; version: string }
  | { status: "no_provider"; version: string }
  | { status: "unreachable"; version: string | null; message: string }

const POLL_MS = 10_000

/** Poll `GET /api/health`; `recheck` forces a check now. */
export function useHealth(): { health: HealthState; recheck: () => void } {
  const [health, setHealth] = useState<HealthState>({
    status: "checking",
    version: null,
  })
  const versionRef = useRef<string | null>(null)

  const check = useCallback(async (signal?: AbortSignal) => {
    try {
      const body = await apiFetch<Health>("/api/health", { signal })
      versionRef.current = body.version
      setHealth(
        body.provider
          ? {
              status: "connected",
              provider: body.provider,
              version: body.version,
            }
          : { status: "no_provider", version: body.version }
      )
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return
      setHealth({
        status: "unreachable",
        version: versionRef.current,
        message:
          err instanceof ApiError ? err.message : "The API did not answer.",
      })
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    const first = window.setTimeout(() => void check(controller.signal), 0)
    const timer = window.setInterval(() => void check(), POLL_MS)
    const onFocus = () => void check()
    window.addEventListener("focus", onFocus)
    return () => {
      controller.abort()
      window.clearTimeout(first)
      window.clearInterval(timer)
      window.removeEventListener("focus", onFocus)
    }
  }, [check])

  const recheck = useCallback(() => {
    setHealth((h) => ({ status: "checking", version: h.version }))
    void check()
  }, [check])

  return { health, recheck }
}
