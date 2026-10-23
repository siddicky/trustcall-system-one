import { Brand } from "@/components/app-header"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

/** Messages for the `error` codes set by the middleware and auth functions. */
const ERRORS: Record<string, string> = {
  not_allowed:
    "This Google account is not on the demo's access list. Ask the owner to add it.",
  unverified: "Google has not verified this account's email address.",
  state: "The sign-in attempt expired or was tampered with. Try again.",
  exchange: "Google did not complete the sign-in. Try again.",
  config: "Google sign-in is not configured on this deployment yet.",
}

export function SignInPage() {
  const params = new URLSearchParams(window.location.search)
  const error = params.get("error")
  const next = params.get("next")
  const start =
    "/api/auth/google/start" + (next ? `?${new URLSearchParams({ next })}` : "")

  return (
    <main className="grid min-h-svh place-items-center bg-background px-4 text-foreground">
      <div className="flex w-full max-w-sm flex-col gap-6">
        <Brand />
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sign in</CardTitle>
            <CardDescription>
              The Trustcall System One demo is private. Continue with an
              approved Google account.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {error ? (
              <Alert variant="destructive" data-testid="signin-error">
                <AlertDescription>
                  {ERRORS[error] ?? "Sign-in failed. Try again."}
                </AlertDescription>
              </Alert>
            ) : null}
            <Button asChild className="w-full">
              <a href={start} data-testid="signin-google">
                Continue with Google
              </a>
            </Button>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
