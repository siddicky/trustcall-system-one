import { HugeiconsIcon } from "@hugeicons/react"
import { AlertCircleIcon } from "@hugeicons/core-free-icons"

import { MonoLabel } from "@/components/common"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import type { HealthState } from "@/hooks/use-health"
import { API_URL } from "@/lib/api"

function Env({ children }: { children: string }) {
  return (
    <pre className="overflow-x-auto rounded-md border border-line bg-sunken p-3 font-code text-xs/[1.9] no-ligatures text-foreground/85">
      {children}
    </pre>
  )
}

/** Shown when the API is down or has no model credentials. */
export function ProviderProblem({
  health,
  onRecheck,
}: {
  health: HealthState
  onRecheck: () => void
}) {
  if (health.status === "connected" || health.status === "checking") return null
  const unreachable = health.status === "unreachable"
  return (
    <Card className="gap-4 p-4" data-testid="provider-problem">
      <Alert variant="destructive">
        <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} />
        <AlertTitle>
          {unreachable ? "The API is unreachable" : "No provider configured"}
        </AlertTitle>
        <AlertDescription>
          {unreachable ? (
            <>
              Nothing answered at <span className="font-code">{API_URL}</span>. Start the backend,
              or point <span className="font-code">VITE_API_URL</span> at it, then check again.
            </>
          ) : (
            <>
              Set <span className="font-code">CLOUDFLARE_ACCOUNT_ID</span> and{" "}
              <span className="font-code">CLOUDFLARE_API_TOKEN</span>, or{" "}
              <span className="font-code">TYPESAFE_API_KEY</span>. Restart the server after
              exporting them.
            </>
          )}
        </AlertDescription>
      </Alert>
      <div className="flex flex-wrap gap-4">
        {unreachable ? (
          <div className="min-w-0 flex-[1_1_300px]">
            <MonoLabel className="mb-2">Start the backend</MonoLabel>
            <Env>{`$ cd demo/backend\n$ uv run fastapi dev`}</Env>
          </div>
        ) : (
          <>
            <div className="min-w-0 flex-[1_1_300px]">
              <MonoLabel className="mb-2">Cloudflare Workers AI</MonoLabel>
              <Env>{`$ export CLOUDFLARE_ACCOUNT_ID=...\n$ export CLOUDFLARE_API_TOKEN=...`}</Env>
            </div>
            <div className="min-w-0 flex-[1_1_300px]">
              <MonoLabel className="mb-2">Or TypeSafe</MonoLabel>
              <Env>{`$ export TRUSTCALL_S1_PROVIDER=typesafe\n$ export TYPESAFE_API_KEY=...`}</Env>
            </div>
          </>
        )}
      </div>
      <div>
        <Button variant="outline" size="lg" onClick={onRecheck} data-testid="check-again">
          Check again
        </Button>
      </div>
    </Card>
  )
}
