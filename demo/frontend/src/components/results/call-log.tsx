import { HugeiconsIcon } from "@hugeicons/react"
import { ComputerTerminal01Icon } from "@hugeicons/core-free-icons"

import { formatInt, formatMs, formatUsd } from "@/lib/format"
import type { CallRecord, CostReport } from "@/lib/types"
import { cn } from "@/lib/utils"

export function shortModel(model: string): string {
  return model.replace(/^@cf\/[^/]+\//, "")
}

function tokensOf(call: CallRecord): number {
  return (call.usage.input_tokens ?? 0) + (call.usage.output_tokens ?? 0)
}

export function costValue(cost: CostReport): number {
  return cost.usd > 0 ? cost.usd : cost.reported_usd
}

const COLS =
  "grid grid-cols-[64px_84px_84px_minmax(70px,1fr)_72px_64px_64px] gap-x-3"

/** A terminal-style panel with one line per model call. */
export function CallLog({
  calls,
  cost,
  runId,
  elapsedMs,
}: {
  calls: CallRecord[]
  cost: CostReport
  runId: string
  elapsedMs: number
}) {
  const retries = calls.reduce((n, c) => n + c.retries, 0)
  const fallbacks = calls.filter((c) => c.fallback_from).length
  const errors = calls.filter((c) => c.error).length
  return (
    <div
      className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-line bg-sunken"
      data-testid="call-log"
    >
      <div className="flex items-center gap-2 border-b border-line px-3 py-2">
        <HugeiconsIcon
          icon={ComputerTerminal01Icon}
          strokeWidth={2}
          className="size-3.5 text-muted-foreground"
        />
        <span className="code-label">run {runId}</span>
        <span className="code-label ml-auto">{formatMs(elapsedMs)}</span>
      </div>
      <div
        className="max-h-[420px] overflow-auto px-3 py-2 font-code text-xs/[2.1] no-ligatures"
        tabIndex={0}
        role="region"
        aria-label="Call log"
      >
        <div className="min-w-[700px]">
          <div className="text-muted-foreground">
            $ POST /api/extract
          </div>
          <div className={cn(COLS, "text-faint uppercase")}>
            <span>Model</span>
            <span>Kind</span>
            <span>Schema</span>
            <span>Field</span>
            <span>Latency</span>
            <span className="text-right">Tokens</span>
            <span>Status</span>
          </div>
          {calls.map((c, i) => (
            <div
              key={i}
              data-testid="call-line"
              className={cn(COLS, c.error && "text-destructive")}
              title={c.error ?? undefined}
            >
              <span>{shortModel(c.model)}</span>
              <span>{c.kind}</span>
              <span className="truncate text-muted-foreground">{c.schema || "-"}</span>
              <span className="truncate text-muted-foreground">{c.field ?? "-"}</span>
              <span>{formatMs(c.duration_ms)}</span>
              <span className="text-right">{formatInt(tokensOf(c))}</span>
              <span>
                {c.error
                  ? `err ${c.status_code ?? ""}`.trim()
                  : c.retries > 0
                    ? `ok r${c.retries}`
                    : c.fallback_from
                      ? "fallback"
                      : "ok"}
              </span>
            </div>
          ))}
          {calls.map((c, i) =>
            c.error ? (
              <div key={`e${i}`} className="text-destructive">
                {shortModel(c.model)} {c.kind}: {c.error}
              </div>
            ) : null
          )}
          <div className="text-success">
            ok {calls.length} {calls.length === 1 ? "call" : "calls"}
            {errors > 0 ? `, ${errors} failed` : ""}
          </div>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line px-3 py-2">
        <span className="code-label">
          {calls.length} REQUESTS · {retries} RETRIES · {fallbacks} FALLBACKS
        </span>
        <span className="code-label">
          {formatUsd(costValue(cost))}
          {cost.complete ? " EST." : "+ INCOMPLETE"}
        </span>
      </div>
    </div>
  )
}
