import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons"

import { StatusDot } from "@/components/common"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { HealthState } from "@/hooks/use-health"

const DOCS_URL = "https://github.com/siddicky/trustcall-system-one#readme"

function providerLabel(provider: string): string {
  return provider === "typesafe" ? "TYPESAFE" : "CLOUDFLARE"
}

export function ProviderStatus({ health }: { health: HealthState }) {
  let tone: "ok" | "bad" | "idle" = "idle"
  let label = "CHECKING"
  switch (health.status) {
    case "connected":
      tone = "ok"
      label = `${providerLabel(health.provider)} · CONNECTED`
      break
    case "no_provider":
      tone = "bad"
      label = "NOT CONFIGURED"
      break
    case "unreachable":
      tone = "bad"
      label = "UNREACHABLE"
      break
  }
  return (
    <Badge
      variant="outline"
      className="gap-1.5 pl-1.5 font-mono tracking-[0.06em] uppercase"
      data-testid="provider-status"
      data-status={health.status}
    >
      <StatusDot tone={tone} />
      {label}
    </Badge>
  )
}

export function AppHeader({ health }: { health: HealthState }) {
  const version = health.version
  return (
    <header className="flex min-h-[52px] flex-wrap items-center justify-between gap-3 border-b border-line bg-background/90 px-4 py-2 backdrop-blur sm:px-6">
      <div className="flex items-center gap-2.5">
        <span
          aria-hidden
          className="grid size-6 shrink-0 place-items-center rounded-md bg-foreground text-background"
        >
          <span className="font-mono text-[11px] leading-none font-bold">T</span>
        </span>
        <span className="font-mono text-[12px] font-medium tracking-[0.12em] text-foreground uppercase">
          Trustcall <span className="text-muted-foreground">/</span> System One
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <ProviderStatus health={health} />
        {version ? (
          <span className="code-label" data-testid="api-version">
            v{version}
          </span>
        ) : null}
        <Button variant="ghost" size="sm" asChild>
          <a href={DOCS_URL} target="_blank" rel="noreferrer">
            Docs
            <HugeiconsIcon icon={ArrowUpRight01Icon} strokeWidth={2} />
          </a>
        </Button>
      </div>
    </header>
  )
}
