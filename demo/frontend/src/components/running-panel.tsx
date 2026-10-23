import { HugeiconsIcon } from "@hugeicons/react"
import { Tick02Icon } from "@hugeicons/core-free-icons"
import { useEffect, useState } from "react"

import { CodeLabel } from "@/components/common"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { formatSeconds, plural } from "@/lib/format"

/** Milliseconds since `startedAt`, refreshed every 100 ms. */
export function useElapsed(startedAt: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(timer)
  }, [])
  return Math.max(0, now - startedAt)
}

function Step({
  state,
  label,
  time,
}: {
  state: "done" | "run" | "wait"
  label: string
  time: string
}) {
  return (
    <div
      className={
        "flex items-center gap-2 border-b border-line py-2 last:border-b-0 " +
        (state === "wait" ? "text-faint" : state === "run" ? "font-medium" : "")
      }
      data-state={state}
    >
      {state === "done" ? (
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 text-success" />
      ) : (
        <span
          aria-hidden
          className={
            "inline-block size-3.5 shrink-0 rounded-[4px] border " +
            (state === "run"
              ? "animate-pulse border-foreground bg-[linear-gradient(90deg,var(--foreground)_50%,transparent_50%)]"
              : "border-border")
          }
        />
      )}
      <span className={state === "done" ? "text-muted-foreground" : undefined}>{label}</span>
      <span className="ml-auto font-mono text-muted-foreground">{time}</span>
    </div>
  )
}

/**
 * The run in flight. The API answers once, when the run ends, so the bar
 * eases toward the end instead of counting requests it cannot see.
 */
export function RunningPanel({
  startedAt,
  schemaNames,
  leafPaths,
  source,
  chars,
  existingCount,
  onCancel,
}: {
  startedAt: number
  schemaNames: string[]
  leafPaths: number
  source: string
  chars: number
  existingCount: number
  onCancel: () => void
}) {
  const elapsed = useElapsed(startedAt)
  const progress = Math.min(92, (1 - Math.exp(-elapsed / 9000)) * 100)
  return (
    <Card className="gap-0 p-0" data-testid="running-panel">
      <div className="flex flex-wrap items-start justify-between gap-2 border-b border-line p-4">
        <div>
          <div className="text-sm font-medium">Running</div>
          <div className="text-muted-foreground">
            Extracting {schemaNames.join(", ")}
          </div>
        </div>
        <CodeLabel data-testid="elapsed">RUNNING · {formatSeconds(elapsed)}</CodeLabel>
      </div>
      <div className="flex flex-col gap-4 p-4">
        <div>
          <div className="flex items-baseline justify-between gap-2">
            <b className="font-medium">Extracting {schemaNames.join(", ")}</b>
            <span className="font-mono text-primary" data-testid="elapsed-big">
              {formatSeconds(elapsed)}
            </span>
          </div>
          <Progress
            value={progress}
            aria-label="Extraction progress"
            className="my-2.5 h-1.5"
          />
          <span className="text-muted-foreground">
            Waiting for the API · {source}
          </span>
        </div>
        <div>
          <Step state="done" label={`Read document · ${chars.toLocaleString("en-US")} chars`} time="done" />
          <Step
            state="done"
            label={`Compile schema · ${plural(leafPaths, "leaf path")}`}
            time="done"
          />
          <Step
            state="run"
            label={`Extract and verify · Clairic + Clef${existingCount ? ` · ${plural(existingCount, "existing record")}` : ""}`}
            time={formatSeconds(elapsed)}
          />
          <Step state="wait" label="Gate and apply patch" time="queued" />
        </div>
        <div className="rounded-md border border-line bg-sunken p-3 font-code text-xs/[2.1] no-ligatures">
          <div className="text-muted-foreground">$ POST /api/extract</div>
          <div>
            schemas {schemaNames.join(", ")}
          </div>
          <div className="text-muted-foreground">
            waiting for the response<span className="animate-pulse">_</span>
          </div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3 border-t border-line p-4">
        <Button variant="outline" size="lg" onClick={onCancel} data-testid="cancel-run">
          Cancel run
        </Button>
        <CodeLabel>THE API ANSWERS ONCE, WHEN THE RUN ENDS</CodeLabel>
      </div>
    </Card>
  )
}
