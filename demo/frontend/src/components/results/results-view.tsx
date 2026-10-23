import { HugeiconsIcon } from "@hugeicons/react"
import {
  Alert02Icon,
  Copy01Icon,
  InformationCircleIcon,
  RefreshIcon,
} from "@hugeicons/core-free-icons"
import { toast } from "sonner"

import { CodeLabel, KeyValue, MonoLabel, Panel } from "@/components/common"
import { HighlightedText } from "@/components/highlighted-text"
import { CallLog, costValue, shortModel } from "@/components/results/call-log"
import { RecordCard } from "@/components/results/record-card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  formatInt,
  formatP,
  formatSeconds,
  formatUsd,
  plural,
  prettyJson,
} from "@/lib/format"
import { buildMarks, type FieldRow } from "@/lib/results"
import type { CostReport, ExtractResponse } from "@/lib/types"

export function headline(result: ExtractResponse): string {
  const count = (k: string) => result.records.filter((r) => r.kind === k).length
  const parts: string[] = []
  const updated = count("update")
  const inserted = count("insert")
  const deleted = count("delete")
  if (updated) parts.push(`${plural(updated, "record")} updated`)
  if (inserted) {
    parts.push(
      parts.length ? `${inserted} inserted` : `${plural(inserted, "record")} inserted`
    )
  }
  if (deleted) {
    parts.push(
      parts.length ? `${deleted} deleted` : `${plural(deleted, "record")} deleted`
    )
  }
  const ops = result.records.reduce((n, r) => n + r.patches.length, 0)
  const first = parts.length ? `${parts.join(", ")}.` : "No records changed."
  return ops > 0 ? `${first} ${plural(ops, "op")} applied.` : first
}

function lede(result: ExtractResponse): string {
  const rejected = result.records.reduce((n, r) => n + r.rejected.length, 0)
  const bits = [
    `${plural(rejected, "op")} refused`,
    `${plural(result.skipped.length, "entity", "entities")} skipped`,
  ]
  if (result.untouched.length) bits.push(`${result.untouched.length} untouched`)
  if (result.warnings.length) bits.push(plural(result.warnings.length, "warning"))
  return `${bits.join(", ")}. Every value below was checked before it was applied.`
}

function costSummary(cost: CostReport): { value: string; sub: string } {
  const value = formatUsd(costValue(cost)) + (cost.complete ? "" : "+")
  if (cost.complete) {
    return {
      value,
      sub: cost.reported_usd > 0 && cost.usd > 0 ? `REPORTED ${formatUsd(cost.reported_usd)}` : "ESTIMATED",
    }
  }
  const models = cost.unpriced_models.map(shortModel).join(", ")
  return { value, sub: `INCOMPLETE · ${models || "SOME MODELS"} UNPRICED` }
}

function Stat({
  label,
  value,
  sub,
  title,
  testId,
}: {
  label: string
  value: string
  sub: string
  title?: string
  testId: string
}) {
  return (
    <div className="min-w-[130px] flex-[1_1_130px]" title={title} data-testid={testId}>
      <MonoLabel>{label}</MonoLabel>
      <b className="mt-0.5 mb-1.5 block text-[28px] leading-none font-[420] tracking-[-0.03em]">
        {value}
      </b>
      <CodeLabel className="block">{sub}</CodeLabel>
    </div>
  )
}

export function ResultsView({
  result,
  runId,
  text,
  sourceLabel,
  rowsByRecord,
  activeRecord,
  activeKey,
  onSelectRecord,
  onSelectField,
  onRerun,
  canRerun,
}: {
  result: ExtractResponse
  runId: string
  text: string
  sourceLabel: string
  rowsByRecord: FieldRow[][]
  activeRecord: number
  activeKey: string | null
  onSelectRecord: (index: number) => void
  onSelectField: (key: string) => void
  onRerun: () => void
  canRerun: boolean
}) {
  const patches = result.records.reduce((n, r) => n + r.patches.length, 0)
  const rejected = result.records.reduce((n, r) => n + r.rejected.length, 0)
  const counts = (k: string) => result.records.filter((r) => r.kind === k).length
  const retries = result.calls.reduce((n, c) => n + c.retries, 0)
  const fallbacks = result.calls.filter((c) => c.fallback_from).length
  const cost = costSummary(result.cost)

  const record = result.records[activeRecord] ?? null
  const rows = rowsByRecord[activeRecord] ?? []
  const marks = record ? buildMarks(record, rows, text) : []
  const evidence = rows.flatMap((row) =>
    row.evidence.map((ev) => ({ row, ev }))
  )
  evidence.sort((a, b) => a.ev.n - b.ev.n)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(prettyJson(result))
      toast.success("Result JSON copied")
    } catch {
      toast.error("Could not copy the result")
    }
  }

  return (
    <div className="flex flex-col gap-4" data-testid="results">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mono-label flex flex-wrap items-center gap-2">
            <span>Workspace</span>
            <span>/</span>
            <span>Runs</span>
            <span>/</span>
            <b className="font-bold text-foreground">{runId.toUpperCase()}</b>
          </div>
          <h2
            className="mt-2 text-[28px] leading-[1.05] font-[420] tracking-[-0.045em] max-sm:text-2xl"
            data-testid="results-headline"
          >
            {headline(result)}
          </h2>
          <p className="mt-1.5 max-w-xl text-sm/relaxed text-muted-foreground">
            {lede(result)}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" size="lg" onClick={copy}>
            <HugeiconsIcon icon={Copy01Icon} strokeWidth={2} />
            Copy JSON
          </Button>
          <Button variant="ghost" size="lg" onClick={onRerun} disabled={!canRerun}>
            <HugeiconsIcon icon={RefreshIcon} strokeWidth={2} />
            Re-run
          </Button>
        </div>
      </div>

      <Card size="sm" className="px-3">
        <div className="flex flex-wrap gap-4" data-testid="stat-strip">
          <Stat
            testId="stat-records"
            label="Records"
            value={String(result.records.length)}
            sub={`${counts("update")} UPDATE · ${counts("insert")} INSERT · ${counts("delete")} DELETE`}
          />
          <Stat
            testId="stat-patches"
            label="Patches"
            value={String(patches)}
            sub="APPLIED OPS"
          />
          <Stat
            testId="stat-rejected"
            label="Rejected"
            value={String(rejected)}
            sub="COMPUTED, REFUSED"
          />
          <Stat
            testId="stat-calls"
            label="Calls"
            value={String(result.calls.length)}
            sub={`${retries} RETRIES · ${fallbacks} FALLBACKS`}
          />
          <Stat
            testId="stat-elapsed"
            label="Elapsed"
            value={formatSeconds(result.elapsed_ms)}
            sub={`${plural(result.attempts, "ATTEMPT")}`}
          />
          <Stat
            testId="stat-cost"
            label="Cost"
            value={cost.value}
            sub={cost.sub}
            title={
              result.cost.complete
                ? undefined
                : `Some models have no configured price: ${result.cost.unpriced_models.join(", ")}. The figure covers the priced models only.`
            }
          />
        </div>
      </Card>

      {result.records.length === 0 ? (
        <Panel
          title="No records"
          description="The models found nothing that matches the schemas in this text."
        />
      ) : null}
      {result.records.map((r, i) => (
        <RecordCard
          key={i}
          record={r}
          rows={rowsByRecord[i] ?? []}
          text={text}
          selected={i === activeRecord}
          activeKey={activeKey}
          onSelectRecord={() => onSelectRecord(i)}
          onSelectField={onSelectField}
        />
      ))}

      <div className="flex flex-wrap items-start gap-4">
        <div className="min-w-0 flex-[1.15_1_420px]">
          <Panel
            title="Source text"
            description="Highlighted spans back the fields of the selected record."
            aside={
              <CodeLabel>
                {sourceLabel} · {plural(evidence.length, "SPAN")}
              </CodeLabel>
            }
          >
            <HighlightedText
              text={text}
              marks={marks}
              activeKey={activeKey}
              onSelect={onSelectField}
              className="text-sm/[2]"
            />
            {evidence.length > 0 ? (
              <div className="flex flex-wrap gap-x-4 gap-y-1.5" data-testid="span-legend">
                {evidence.map(({ row, ev }) => (
                  <CodeLabel key={`${row.key}-${ev.n}`}>
                    {ev.n} · CHARS {ev.start}–{ev.end} · {row.pointer}
                    {ev.located ? " · LOCATED" : ""}
                  </CodeLabel>
                ))}
              </div>
            ) : (
              <CodeLabel>NO FIELD OF THIS RECORD HAS A SOURCE SPAN</CodeLabel>
            )}
          </Panel>
        </div>

        <div className="flex min-w-0 flex-[1_1_380px] flex-col gap-4">
          <Card className="min-w-0 gap-0 pb-0" data-testid="skipped-panel">
            <div className="flex flex-wrap items-start justify-between gap-2 px-4 pb-4">
              <div>
                <div className="text-sm font-medium">Skipped entities</div>
                <div className="text-muted-foreground">
                  Mentioned in the text, not turned into records.
                </div>
              </div>
              <Badge variant="outline" className="font-mono tracking-[0.06em] uppercase">
                {plural(result.skipped.length, "mention")}
              </Badge>
            </div>
            <div className="border-t border-line">
              {result.skipped.length === 0 ? (
                <p className="px-4 py-3 text-muted-foreground">
                  Every entity found became a record.
                </p>
              ) : (
                result.skipped.map((s, i) => (
                  <div
                    key={i}
                    className="flex flex-col gap-1.5 border-b border-line px-4 py-3 last:border-b-0"
                    data-testid="skipped-entity"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <b className="font-medium">{s.entity.label}</b>
                      <CodeLabel>
                        {s.schema_name.toUpperCase()} · {plural(s.entity.mentions.length, "MENTION")}
                      </CodeLabel>
                      <Badge
                        variant="outline"
                        className="ml-auto font-mono tracking-[0.06em]"
                      >
                        {s.reason.toUpperCase()}
                      </Badge>
                    </div>
                    {s.detail ? (
                      <p className="text-muted-foreground">{s.detail}</p>
                    ) : null}
                    {s.candidates.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {s.candidates.map(([id, p]) => (
                          <Badge
                            key={id}
                            variant="secondary"
                            className="font-code no-ligatures"
                          >
                            {id} · P {formatP(p)}
                          </Badge>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))
              )}
              {result.untouched.length > 0 ? (
                <div className="border-t border-line px-4 py-3">
                  <KeyValue
                    label="Untouched existing records"
                    value={result.untouched.join(", ")}
                  />
                </div>
              ) : null}
              {result.deleted.length > 0 ? (
                <div className="border-t border-line px-4 py-3">
                  <KeyValue label="Deleted records" value={result.deleted.join(", ")} />
                </div>
              ) : null}
            </div>
          </Card>

          <div className="flex flex-col gap-2" data-testid="warnings">
            <MonoLabel>Warnings</MonoLabel>
            {result.warnings.length === 0 ? (
              <Alert>
                <HugeiconsIcon icon={InformationCircleIcon} strokeWidth={2} className="text-info" />
                <AlertTitle>No warnings</AlertTitle>
                <AlertDescription>The run finished without library warnings.</AlertDescription>
              </Alert>
            ) : (
              result.warnings.map((w, i) => (
                <Alert key={i} data-testid="warning">
                  <HugeiconsIcon icon={Alert02Icon} strokeWidth={2} className="text-warning" />
                  <AlertTitle>Warning</AlertTitle>
                  <AlertDescription>{w}</AlertDescription>
                </Alert>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-start gap-4">
        <div className="min-w-0 flex-[1.7_1_520px]">
          <MonoLabel className="mb-3">Call log</MonoLabel>
          <CallLog
            calls={result.calls}
            cost={result.cost}
            runId={runId}
            elapsedMs={result.elapsed_ms}
          />
        </div>
        <Panel
          title="Usage"
          description={`Run ${runId}`}
          aside={
            <Badge variant="outline" className="font-mono tracking-[0.06em] uppercase">
              {cost.value} {result.cost.complete ? "est." : "incomplete"}
            </Badge>
          }
          className="min-w-0 flex-[1_1_300px]"
          bodyClassName="gap-0"
        >
          <KeyValue label="Requests" value={formatInt(result.usage.requests)} />
          <KeyValue label="Tokens in" value={formatInt(result.usage.input_tokens)} />
          <KeyValue label="Tokens out" value={formatInt(result.usage.output_tokens)} />
          <KeyValue label="Wall time" value={formatSeconds(result.elapsed_ms)} />
          <KeyValue
            label="Retries"
            value={String(result.calls.reduce((n, c) => n + c.retries, 0))}
          />
          <KeyValue label="Attempts" value={String(result.attempts)} />
          {Object.entries(result.usage.by_model).map(([model, u]) => (
            <KeyValue
              key={model}
              label={shortModel(model)}
              value={`${formatInt(u.requests)} req · ${formatInt(u.input_tokens)} in`}
            />
          ))}
        </Panel>
      </div>
    </div>
  )
}
