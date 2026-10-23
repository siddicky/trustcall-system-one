import { HugeiconsIcon } from "@hugeicons/react"
import { AlertCircleIcon } from "@hugeicons/core-free-icons"
import { useState, type KeyboardEvent, type ReactNode } from "react"

import { CodeEditor } from "@/components/code-editor"
import { CodeLabel, MonoLabel } from "@/components/common"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { compactJson, formatP, jsonEqual, prettyJson } from "@/lib/format"
import {
  quoteParts,
  recordProbability,
  recordTitle,
  type FieldRow,
} from "@/lib/results"
import type { JsonValue, PatchOp, RecordKind, ResultRecord } from "@/lib/types"
import { cn } from "@/lib/utils"

type View = "before" | "after" | "diff"

const KIND_BADGE: Record<RecordKind, "default" | "secondary" | "destructive"> = {
  update: "default",
  insert: "secondary",
  delete: "destructive",
}

function None() {
  return <span className="text-faint">none</span>
}

/** A value in the before or after column. Lists mark what was added or removed. */
function ValueView({
  value,
  other,
  side,
}: {
  value: JsonValue | undefined
  other: JsonValue | undefined
  side: "before" | "after"
}) {
  if (value === undefined || value === null) return <None />
  if (Array.isArray(value)) {
    const otherList = Array.isArray(other) ? other : []
    if (value.length === 0) return <span className="text-faint">empty list</span>
    return (
      <ul className="flex flex-col gap-0.5">
        {value.map((item, i) => {
          const differs = !otherList.some((o) => jsonEqual(o, item))
          const text = typeof item === "string" ? item : compactJson(item)
          return (
            <li
              key={i}
              className={cn(
                "flex gap-1.5",
                side === "after" && differs && "font-medium",
                side === "after" && !differs && "text-muted-foreground",
                side === "before" && differs && "line-through decoration-faint"
              )}
              data-diff={differs ? side : undefined}
            >
              {side === "after" && differs ? (
                <span aria-label="added" className="font-code text-success">
                  +
                </span>
              ) : null}
              <span className="min-w-0 break-words">{text}</span>
            </li>
          )
        })}
      </ul>
    )
  }
  if (typeof value === "object") {
    return (
      <span className="font-code no-ligatures break-all">{compactJson(value)}</span>
    )
  }
  return <span className="break-words">{String(value)}</span>
}

function ProbBar({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <span
      className="flex min-w-[120px] items-center gap-2"
      title={`${label}: ${formatP(value)}`}
    >
      <span className="relative block h-1.5 flex-1 overflow-hidden rounded-full bg-[#2a2e37]">
        <span
          className="absolute inset-y-0 left-0 rounded-full bg-[#c9cdd5]"
          style={{ width: `${pct}%` }}
        />
      </span>
      <em className="min-w-[30px] text-right font-mono text-xs not-italic">
        {formatP(value)}
      </em>
    </span>
  )
}

function Quote({ row, text }: { row: FieldRow; text: string }) {
  if (row.evidence.length > 0) {
    return (
      <div className="flex flex-col gap-1.5">
        {row.evidence.map((ev, i) => {
          const q = quoteParts(text, ev)
          return (
            <span key={i} className="leading-relaxed text-[#c9cdd5]">
              <span className="mr-1.5 inline-block rounded-sm border border-border px-1 py-0.5 font-mono text-[10px] tracking-[0.04em] text-foreground/90">
                {ev.n}
              </span>
              <q className="[quotes:none]">
                {q.pre}
                <b className="font-medium text-foreground">{q.hit}</b>
                {q.post}
              </q>
              {ev.located ? (
                <CodeLabel className="ml-1.5">LOCATED BY SEARCH</CodeLabel>
              ) : null}
            </span>
          )
        })}
      </div>
    )
  }
  if (row.answer.rejected_reason && !row.changed) {
    return (
      <span className="text-muted-foreground">
        Rejected: <span className="font-code">{row.answer.rejected_reason}</span>
      </span>
    )
  }
  if (!row.changed) {
    return <span className="text-faint">No mention in the text.</span>
  }
  return <span className="text-faint">No span for this value.</span>
}

function FieldTable({
  rows,
  text,
  activeKey,
  onSelect,
}: {
  rows: FieldRow[]
  text: string
  activeKey: string | null
  onSelect: (key: string) => void
}) {
  const press = (e: KeyboardEvent, key: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault()
      onSelect(key)
    }
  }
  const cols =
    "@3xl:grid @3xl:grid-cols-[130px_minmax(120px,1fr)_minmax(140px,1.1fr)_160px_minmax(220px,1.7fr)] @3xl:gap-x-3"
  return (
    <div className="@container" data-testid="field-table">
      <div
        className={cn(
          "hidden items-center border-y border-line px-3 py-0 text-xs font-medium",
          cols
        )}
      >
        {["Path", "Before", "After", "Probability", "Source quote"].map((h) => (
          <span key={h} className="flex h-10 items-center">
            {h}
          </span>
        ))}
      </div>
      <div className="border-t border-line @3xl:border-t-0">
        {rows.map((row) => {
          const active = activeKey === row.key
          return (
            <div
              key={row.key}
              role="button"
              tabIndex={0}
              aria-pressed={active}
              data-testid="field-row-result"
              data-path={row.pointer}
              data-changed={row.changed}
              onClick={() => onSelect(row.key)}
              onKeyDown={(e) => press(e, row.key)}
              className={cn(
                "flex cursor-pointer flex-col gap-1.5 border-b border-line px-3 py-2 text-xs/relaxed outline-none last:border-b-0 hover:bg-secondary/50 focus-visible:bg-secondary/60 focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:ring-inset",
                cols,
                "@3xl:items-start",
                active && "bg-secondary/70",
                !row.changed && "text-muted-foreground"
              )}
            >
              <span className="font-code no-ligatures break-all">
                {row.pointer}
              </span>
              <div className="min-w-0 @3xl:py-0">
                <span className="code-label mr-2 @3xl:hidden">Before</span>
                <span
                  className={cn(
                    row.changed && "text-[#c9cdd5]",
                    row.changed &&
                      !Array.isArray(row.before) &&
                      row.before !== undefined &&
                      row.before !== null &&
                      "line-through decoration-faint"
                  )}
                >
                  <ValueView value={row.before} other={row.after} side="before" />
                </span>
              </div>
              <div className="min-w-0">
                <span className="code-label mr-2 @3xl:hidden">After</span>
                {row.changed ? (
                  <div className={cn(!Array.isArray(row.after) && "font-medium")}>
                    <ValueView value={row.after} other={row.before} side="after" />
                  </div>
                ) : (
                  <CodeLabel className="text-faint">Unchanged</CodeLabel>
                )}
                {row.flags.length > 0 ? (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {row.flags.map((f) => (
                      <Badge
                        key={f}
                        variant="outline"
                        className="font-mono tracking-[0.06em] uppercase"
                      >
                        {f}
                      </Badge>
                    ))}
                  </div>
                ) : null}
              </div>
              <div>
                {row.p && (row.changed || row.evidence.length > 0) ? (
                  <ProbBar value={row.p.value} label={row.p.label} />
                ) : (
                  <CodeLabel>NO EVIDENCE</CodeLabel>
                )}
              </div>
              <div className="min-w-0">
                <Quote row={row} text={text} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function opTone(op: string | null): string {
  return op ?? "-"
}

function PatchList({
  patches,
  rows,
  onSelect,
}: {
  patches: PatchOp[]
  rows: FieldRow[]
  onSelect: (key: string) => void
}) {
  const rowFor = (path: string) =>
    rows.find((r) => r.pointer === path || path.startsWith(`${r.pointer}/`))
  return (
    <div data-testid="patch-list">
      {patches.map((op, i) => {
        const row = rowFor(op.path)
        return (
          <button
            key={i}
            type="button"
            disabled={!row}
            onClick={() => row && onSelect(row.key)}
            data-testid="patch-op"
            className="flex w-full flex-wrap items-center gap-2 border-b border-line px-4 py-2 text-left font-code text-xs no-ligatures last:border-b-0 enabled:hover:bg-secondary/50 focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:outline-none focus-visible:ring-inset"
          >
            <Badge variant="secondary" className="font-mono tracking-[0.06em] uppercase">
              {op.op}
            </Badge>
            <span className="text-muted-foreground">{op.path}</span>
            {op.op !== "remove" ? (
              <span className="min-w-0 break-words">{compactJson(op.value)}</span>
            ) : null}
            {op.repaired_from ? (
              <Badge variant="outline" className="font-mono uppercase">
                repaired
              </Badge>
            ) : null}
            {row?.p ? (
              <span className="ml-auto text-muted-foreground">
                P {formatP(row.p.value)}
              </span>
            ) : null}
          </button>
        )
      })}
    </div>
  )
}

function SubHeader({
  title,
  description,
  badge,
}: {
  title: string
  description: string
  badge: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-2 px-4 py-3">
      <div>
        <div className="text-sm font-medium">{title}</div>
        <div className="text-muted-foreground">{description}</div>
      </div>
      {badge}
    </div>
  )
}

export function RecordCard({
  record,
  rows,
  text,
  selected,
  activeKey,
  onSelectRecord,
  onSelectField,
}: {
  record: ResultRecord
  rows: FieldRow[]
  text: string
  selected: boolean
  activeKey: string | null
  onSelectRecord: () => void
  onSelectField: (key: string) => void
}) {
  const [view, setView] = useState<View>("diff")
  const prob = recordProbability(rows)
  const title = recordTitle(record)

  return (
    <section
      data-testid="record-card"
      data-kind={record.kind}
      data-selected={selected}
      onClick={onSelectRecord}
      className={cn(
        "flex min-w-0 flex-col gap-0 overflow-hidden rounded-lg bg-card text-xs/relaxed ring-1 ring-foreground/10",
        selected && "ring-foreground/30"
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Badge
            variant={KIND_BADGE[record.kind]}
            className="font-mono tracking-[0.06em] uppercase"
            data-testid="kind-badge"
          >
            {record.kind}
          </Badge>
          <span className="text-sm font-medium break-words">{title}</span>
          <CodeLabel>
            {[record.json_doc_id, record.schema_name].filter(Boolean).join(" · ")}
          </CodeLabel>
          <Badge
            variant={record.valid ? "outline" : "destructive"}
            className="font-mono tracking-[0.06em] uppercase"
            data-testid="valid-badge"
          >
            {record.valid ? "Valid" : `Invalid · ${record.errors.length}`}
          </Badge>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {record.delete_p !== null ? (
            <CodeLabel>DELETE P {formatP(record.delete_p)}</CodeLabel>
          ) : null}
          {record.entity?.qualify_p != null ? (
            <CodeLabel>QUALIFY P {formatP(record.entity.qualify_p)}</CodeLabel>
          ) : null}
          {prob ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="code-label rounded-sm underline decoration-dotted underline-offset-4 focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:outline-none"
                >
                  RECORD P {formatP(prob.mean)}
                </button>
              </TooltipTrigger>
              <TooltipContent>
                Mean of {prob.count} changed {prob.count === 1 ? "field" : "fields"}.
                {prob.weakest
                  ? ` Weakest: ${prob.weakest.pointer} ${formatP(prob.weakest.p)}.`
                  : ""}
              </TooltipContent>
            </Tooltip>
          ) : null}
          <div
            role="group"
            aria-label="View"
            className="inline-flex h-7 items-center rounded-lg bg-muted p-[3px]"
            onClick={(e) => e.stopPropagation()}
          >
            {(["before", "after", "diff"] as const).map((v) => (
              <button
                key={v}
                type="button"
                aria-pressed={view === v}
                data-testid={`view-${v}`}
                onClick={() => setView(v)}
                className={cn(
                  "h-full rounded-md border border-transparent px-2 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:outline-none",
                  view === v && "border-border bg-input/30 text-foreground"
                )}
              >
                {v === "before" ? "Before" : v === "after" ? "After" : "Diff"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {!record.valid && record.errors.length > 0 ? (
        <div className="px-4 pb-4">
          <Alert variant="destructive">
            <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} />
            <AlertTitle>Record does not validate</AlertTitle>
            <AlertDescription>
              <ul className="list-disc pl-4">
                {record.errors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        </div>
      ) : null}

      <div onClick={(e) => e.stopPropagation()}>
        {view === "diff" ? (
          rows.length > 0 ? (
            <FieldTable
              rows={rows}
              text={text}
              activeKey={selected ? activeKey : null}
              onSelect={(key) => {
                onSelectRecord()
                onSelectField(key)
              }}
            />
          ) : (
            <p className="border-t border-line px-4 py-4 text-muted-foreground">
              No field answers for this record.
            </p>
          )
        ) : (
          <div className="border-t border-line p-4">
            <MonoLabel className="mb-2">
              {view === "before" ? "Record before" : "Record after"}
            </MonoLabel>
            {(view === "before" ? record.before : record.after) ? (
              <CodeEditor
                language="json"
                readOnly
                lineNumbers={false}
                ariaLabel={`Record ${view}`}
                value={prettyJson(view === "before" ? record.before : record.after)}
                minHeight="60px"
                maxHeight="360px"
              />
            ) : (
              <p className="text-muted-foreground">
                {view === "before"
                  ? "This record is new: nothing existed before."
                  : "This record was deleted."}
              </p>
            )}
          </div>
        )}
      </div>

      <div
        className="grid grid-cols-1 border-t border-line @container lg:grid-cols-2"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="min-w-0 lg:border-r lg:border-line">
          <SubHeader
            title="Applied patch"
            description="Computed in code, checked one op at a time."
            badge={
              <Badge variant="outline" className="font-mono tracking-[0.06em] uppercase">
                RFC 6902 · {record.patches.length}{" "}
                {record.patches.length === 1 ? "op" : "ops"}
              </Badge>
            }
          />
          <div className="border-t border-line">
            {record.patches.length > 0 ? (
              <PatchList
                patches={record.patches}
                rows={rows}
                onSelect={(key) => {
                  onSelectRecord()
                  onSelectField(key)
                }}
              />
            ) : (
              <p className="px-4 py-3 text-muted-foreground">
                {record.kind === "insert"
                  ? "A new record: the value is the whole document, there is no patch."
                  : "No operation was applied."}
              </p>
            )}
          </div>
        </div>
        <div className="min-w-0 border-t border-line lg:border-t-0">
          <SubHeader
            title="Rejected ops"
            description="Computed, then refused. Nothing here touched the record."
            badge={
              <Badge
                variant={record.rejected.length > 0 ? "destructive" : "outline"}
                className="font-mono tracking-[0.06em] uppercase"
              >
                {record.rejected.length} refused
              </Badge>
            }
          />
          <div className="border-t border-line" data-testid="rejected-list">
            {record.rejected.length === 0 ? (
              <p className="px-4 py-3 text-muted-foreground">Nothing was refused.</p>
            ) : (
              record.rejected.map((r, i) => (
                <div
                  key={i}
                  className="flex flex-col gap-1.5 border-b border-line px-4 py-3 last:border-b-0"
                  data-testid="rejected-op"
                >
                  <div className="no-ligatures flex flex-wrap items-center gap-2 font-code text-xs">
                    <Badge variant="secondary" className="font-mono tracking-[0.06em] uppercase">
                      {opTone(r.op)}
                    </Badge>
                    <span className="text-muted-foreground break-all">{r.path}</span>
                    {r.p !== null ? (
                      <span className="text-muted-foreground">P {formatP(r.p)}</span>
                    ) : null}
                    <Badge variant="outline" className="ml-auto font-mono tracking-[0.06em]">
                      {r.reason.toUpperCase()}
                    </Badge>
                  </div>
                  {r.detail ? (
                    <p className="text-muted-foreground">{r.detail}</p>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
