import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon } from "@hugeicons/core-free-icons"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { PRESETS, type Preset } from "@/lib/presets"

export function PresetsMenu({ onLoad }: { onLoad: (preset: Preset) => void }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="lg" data-testid="presets-trigger">
          Load a preset
          <HugeiconsIcon icon={ArrowDown01Icon} strokeWidth={2} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel>Schema, document and existing records</DropdownMenuLabel>
        {PRESETS.map((p) => (
          <DropdownMenuItem
            key={p.id}
            data-testid={`preset-${p.id}`}
            className="flex-col items-start gap-0.5"
            onSelect={() => onLoad(p)}
          >
            <span className="font-medium">{p.label}</span>
            <span className="text-muted-foreground">{p.summary}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
