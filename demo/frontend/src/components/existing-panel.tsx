import type { ReactNode } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { Add01Icon, Delete02Icon } from "@hugeicons/core-free-icons"

import { CodeEditor } from "@/components/code-editor"
import { CodeLabel, Divider, Panel } from "@/components/common"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import type { ExistingCheck, ExistingDraft } from "@/lib/existing"
import { cn } from "@/lib/utils"

export interface RunOptionsState {
  enableInserts: boolean
  enableDeletes: boolean
  referenceDate: string
}

function Toggle({
  title,
  description,
  checked,
  onChange,
  testId,
}: {
  title: string
  description: string
  checked: boolean
  onChange: (v: boolean) => void
  testId: string
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 py-2">
      <span className="min-w-0 flex-1">
        <b className="block font-medium">{title}</b>
        <span className="mt-0.5 block text-muted-foreground">{description}</span>
      </span>
      <Switch
        data-testid={testId}
        checked={checked}
        onCheckedChange={onChange}
        aria-label={title}
      />
    </label>
  )
}

export function ExistingPanel({
  entries,
  checks,
  schemaNames,
  options,
  onAdd,
  onRemove,
  onChange,
  onOptions,
  footer,
}: {
  entries: ExistingDraft[]
  checks: ExistingCheck[]
  schemaNames: string[]
  options: RunOptionsState
  onAdd: () => void
  onRemove: (key: string) => void
  onChange: (key: string, patch: Partial<ExistingDraft>) => void
  onOptions: (patch: Partial<RunOptionsState>) => void
  footer: ReactNode
}) {
  return (
    <Panel
      title="Existing records"
      description="Optional. The text is read against them and the library returns patch ops, not a regenerated record."
      aside={<Badge variant="secondary" className="font-mono tracking-[0.06em] uppercase">Enables updates</Badge>}
    >
      <div className="flex flex-col gap-3" data-testid="existing-list">
        {entries.map((e, i) => {
          const check = checks[i]
          return (
            <div
              key={e.key}
              className="flex flex-col gap-2 rounded-md border border-line p-2"
              data-testid="existing-entry"
            >
              <div className="flex flex-wrap items-end gap-2">
                <div className="flex min-w-28 flex-[1_1_120px] flex-col gap-1.5">
                  <label className="text-xs font-medium" htmlFor={`ex-id-${e.key}`}>
                    Record id
                  </label>
                  <Input
                    id={`ex-id-${e.key}`}
                    value={e.id}
                    className="font-code"
                    aria-invalid={check?.error === "needs an id" || check?.error === "id is used twice"}
                    onChange={(ev) => onChange(e.key, { id: ev.target.value })}
                  />
                </div>
                <div className="flex min-w-28 flex-[1_1_140px] flex-col gap-1.5">
                  <span className="text-xs font-medium">Schema</span>
                  <Select value={e.schema} onValueChange={(v) => onChange(e.key, { schema: v })}>
                    <SelectTrigger aria-label="Schema of the record" className="w-full font-code">
                      <SelectValue placeholder="Pick a schema" />
                    </SelectTrigger>
                    <SelectContent>
                      {schemaNames.map((n) => (
                        <SelectItem key={n} value={n} className="font-code">
                          {n}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove record ${e.id || "(no id)"}`}
                  onClick={() => onRemove(e.key)}
                >
                  <HugeiconsIcon icon={Delete02Icon} strokeWidth={2} />
                </Button>
              </div>
              <CodeEditor
                language="json"
                value={e.text}
                lineNumbers={false}
                minHeight="96px"
                maxHeight="240px"
                ariaLabel={`JSON of record ${e.id || "(no id)"}`}
                testId="existing-json"
                onChange={(text) => onChange(e.key, { text })}
              />
              <div className="flex flex-wrap justify-between gap-2" data-testid="existing-status">
                <CodeLabel className={cn(check?.error && "text-destructive")}>
                  {check?.error
                    ? `INVALID · ${check.error}`
                    : `VALID JSON · MATCHES ${e.schema.toUpperCase()}`}
                </CodeLabel>
                {check?.warning ? (
                  <CodeLabel className="text-warning">{check.warning}</CodeLabel>
                ) : null}
              </div>
            </div>
          )
        })}
        {entries.length === 0 ? (
          <p className="text-muted-foreground">
            No existing records: the run extracts new ones.
          </p>
        ) : null}
        <div>
          <Button
            variant="outline"
            size="sm"
            onClick={onAdd}
            disabled={schemaNames.length === 0}
            data-testid="add-existing"
          >
            <HugeiconsIcon icon={Add01Icon} strokeWidth={2} />
            Add record
          </Button>
        </div>
      </div>

      <Divider />
      <div>
        <Toggle
          testId="switch-inserts"
          title="Allow inserts"
          description="Add records the text introduces that are not in the existing set."
          checked={options.enableInserts}
          onChange={(v) => onOptions({ enableInserts: v })}
        />
        <Toggle
          testId="switch-deletes"
          title="Allow deletes"
          description="Remove records or list items the text retracts. Off by default."
          checked={options.enableDeletes}
          onChange={(v) => onOptions({ enableDeletes: v })}
        />
        <div className="flex flex-wrap items-center justify-between gap-2 py-2">
          <span className="min-w-0 flex-1">
            <label htmlFor="reference-date" className="block font-medium">
              Reference date
            </label>
            <span className="mt-0.5 block text-muted-foreground">
              Optional. Relative dates such as "last month" resolve against it.
            </span>
          </span>
          <Input
            id="reference-date"
            type="date"
            value={options.referenceDate}
            className="w-40 font-code"
            onChange={(ev) => onOptions({ referenceDate: ev.target.value })}
          />
        </div>
      </div>
      <Divider />
      {footer}
    </Panel>
  )
}
