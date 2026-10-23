import { useEffect, useState } from "react"

export type SessionUser = { email: string; name: string }

/**
 * The Google account signed in to this deployment, from the same-origin
 * `GET /api/auth/me`. Null while loading and wherever the auth functions do
 * not run (`vite dev`).
 */
export function useSessionUser(): SessionUser | null {
  const [user, setUser] = useState<SessionUser | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    fetch("/api/auth/me", { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) return
        const body = (await res.json()) as Partial<SessionUser>
        if (typeof body.email === "string") {
          setUser({ email: body.email, name: body.name ?? body.email })
        }
      })
      .catch(() => {
        // Aborted or no auth functions: show nothing.
      })
    return () => controller.abort()
  }, [])

  return user
}
