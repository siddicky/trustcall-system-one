import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { AlertCircleIcon, ArrowRight02Icon } from "@hugeicons/core-free-icons"
import { toast } from "sonner"

import { AppHeader } from "@/components/app-header"
import { CodeLabel } from "@/components/common"
import {
  DocumentPanel,
  docText,
  MAX_UPLOAD_BYTES,
  type DocMode,
  type DocState,
} from "@/components/document-panel"
import { ExistingPanel, type RunOptionsState } from "@/components/existing-panel"
import { PresetsMenu } from "@/components/presets-menu"
import { ProviderProblem } from "@/components/provider-problem"
import { ResultsView } from "@/components/results/results-view"
import { RunningPanel } from "@/components/running-panel"
import { SchemaPanel, type SyncError } from "@/components/schema/schema-panel"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { useHealth } from "@/hooks/use-health"
import { ApiError, apiFetch, apiUpload } from "@/lib/api"
import {
  createDraft,
  fromParsed,
  restoreSchemas,
  withFields,
  withJsonText,
  withMeta,
  withPydanticText,
  type SchemaDraft,
} from "@/lib/draft"
import {
  checkExisting,
  newExisting,
  toExistingIn,
  type ExistingDraft,
} from "@/lib/existing"
import { formatBytes } from "@/lib/format"
import { PRESETS, type Preset } from "@/lib/presets"
import { buildMarks, buildRows, type FieldRow } from "@/lib/results"
import {
  fieldsToPydantic,
  leafCount,
  makeField,
  topLevelModels,
  validateFields,
  type FormField,
} from "@/lib/schema"
import { loadJson, saveJson } from "@/lib/storage"
import type {
  Cardinality,
  DocumentInfo,
  ExtractRequest,
  ExtractResponse,
  SchemaParseResponse,
} from "@/lib/types"

const STORAGE_KEY = "trustcall-s1:schemas:v1"

type RunState =
  | { status: "idle" }
  | {
      status: "running"
      runId: string
      startedAt: number
      text: string
      schemaNames: string[]
      leafPaths: number
      source: string
      existingCount: number
    }
  | {
      status: "done"
      runId: string
      result: ExtractResponse
      text: string
      source: string
    }
  | { status: "error"; runId: string; code: string; message: string }

function defaultDrafts(): SchemaDraft[] {
  const s = PRESETS[0].schemas[0]
  return [createDraft(s)]
}

function newRunId(): string {
  return Math.random().toString(16).slice(2, 8)
}

function uniqueName(base: string, taken: string[]): string {
  if (!taken.includes(base)) return base
  for (let i = 2; ; i += 1) {
    if (!taken.includes(`${base}${i}`)) return `${base}${i}`
  }
}

function extension(name: string): string {
  const at = name.lastIndexOf(".")
  return at < 0 ? "" : name.slice(at).toLowerCase()
}

export function App() {
  const { health, recheck } = useHealth()

  const [initial] = useState(() => restoreSchemas(loadJson(STORAGE_KEY)))
  const [drafts, setDrafts] = useState<SchemaDraft[]>(
    () => initial?.drafts ?? defaultDrafts()
  )
  const [selectedId, setSelectedId] = useState<string | null>(
    () => initial?.selectedId ?? null
  )
  const [doc, setDoc] = useState<DocState>({
    mode: "file",
    file: null,
    pasted: "",
    upload: { status: "idle" },
    pastedLabel: null,
  })
  const [existing, setExisting] = useState<ExistingDraft[]>([])
  const [options, setOptions] = useState<RunOptionsState>({
    enableInserts: true,
    enableDeletes: false,
    referenceDate: "",
  })
  const [run, setRun] = useState<RunState>({ status: "idle" })
  const [activeRecord, setActiveRecord] = useState(0)
  const [activeKey, setActiveKey] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const uploadToken = useRef(0)
  const resultsRef = useRef<HTMLDivElement>(null)

  // Persist the schemas (the last schema survives a reload).
  useEffect(() => {
    saveJson(STORAGE_KEY, { v: 1, selectedId, drafts })
  }, [drafts, selectedId])

  const effectiveSelected =
    drafts.find((d) => d.id === selectedId)?.id ?? drafts[0]?.id ?? null
  const text = docText(doc)
  const schemaNames = drafts.map((d) => d.name)

  // ---------------------------------------------------------------- schemas
  const patchDraft = useCallback(
    (id: string, fn: (d: SchemaDraft) => SchemaDraft) =>
      setDrafts((all) => all.map((d) => (d.id === id ? fn(d) : d))),
    []
  )

  const renameInExisting = useCallback((from: string, to: string) => {
    setExisting((all) =>
      all.map((e) => (e.schema === from ? { ...e, schema: to } : e))
    )
  }, [])

  const addSchema = () => {
    const name = uniqueName("Schema", schemaNames)
    const draft = createDraft({
      name,
      cardinality: "one",
      fields: [makeField({ name: "name", description: "" })],
    })
    setDrafts((all) => [...all, draft])
    setSelectedId(draft.id)
  }

  const removeSchema = (id: string) => {
    setDrafts((all) => all.filter((d) => d.id !== id))
    if (effectiveSelected === id) {
      const next = drafts.find((d) => d.id !== id)
      setSelectedId(next?.id ?? null)
    }
  }

  const setMeta = (
    id: string,
    meta: { name?: string; description?: string }
  ) => {
    const old = drafts.find((d) => d.id === id)
    if (old && meta.name !== undefined && meta.name !== old.name) {
      renameInExisting(old.name, meta.name)
    }
    patchDraft(id, (d) => withMeta(d, meta))
  }

  const setCardinality = (id: string, cardinality: Cardinality) =>
    patchDraft(id, (d) => ({ ...d, cardinality }))

  const syncPydantic = async (id: string): Promise<SyncError | null> => {
    const draft = drafts.find((d) => d.id === id)
    if (!draft) return null
    const source = draft.pydantic
    try {
      const res = await apiFetch<SchemaParseResponse>("/api/schema/parse", {
        method: "POST",
        json: { source },
      })
      const models = topLevelModels(res.schemas)
      const [first, ...rest] = models
      setDrafts((all) => {
        const next = all.map((d) => (d.id === id ? fromParsed(d, first, source) : d))
        for (const m of rest) {
          const at = next.findIndex((d) => d.name === m.name && d.id !== id)
          if (at >= 0) {
            next[at] = fromParsed(next[at], m, source)
          } else {
            next.push(
              fromParsed(
                createDraft({ name: m.name, fields: [] }),
                m,
                source
              )
            )
          }
        }
        return next
      })
      if (first.name !== draft.name) renameInExisting(draft.name, first.name)
      toast.success(
        models.length === 1
          ? `Synced ${first.name} to the form`
          : `Synced ${models.length} schemas from the Pydantic source`
      )
      return null
    } catch (err) {
      if (err instanceof ApiError) {
        return { code: err.code, message: err.message, line: err.line }
      }
      return {
        code: "error",
        message: err instanceof Error ? err.message : "Sync failed",
        line: null,
      }
    }
  }

  const resetPydantic = (id: string) =>
    patchDraft(id, (d) => ({
      ...d,
      pydantic: fieldsToPydantic(d.name, d.fields, d.description),
      pydanticSynced: true,
    }))

  // --------------------------------------------------------------- document
  const uploadFile = async (file: File) => {
    const token = ++uploadToken.current
    const kind = extension(file.name) === ".pdf" ? "pdf" : "other"
    if (file.size > MAX_UPLOAD_BYTES) {
      setDoc((d) => ({
        ...d,
        mode: "file",
        file: null,
        upload: {
          status: "error",
          name: file.name,
          size: file.size,
          kind,
          code: "too_large",
          message: `The file is ${formatBytes(file.size)}; the limit is 10 MB.`,
        },
      }))
      return
    }
    setDoc((d) => ({
      ...d,
      mode: "file",
      upload: { status: "uploading", name: file.name },
    }))
    try {
      const info = await apiUpload<DocumentInfo>("/api/document", file)
      if (token !== uploadToken.current) return
      setDoc((d) => ({ ...d, mode: "file", file: info, upload: { status: "idle" } }))
    } catch (err) {
      if (token !== uploadToken.current) return
      const e =
        err instanceof ApiError
          ? err
          : new ApiError(0, "error", "The upload failed.")
      setDoc((d) => ({
        ...d,
        file: null,
        upload: {
          status: "error",
          name: file.name,
          size: file.size,
          kind,
          code: e.code,
          message: e.message,
        },
      }))
    }
  }

  // ----------------------------------------------------------------- presets
  const loadPreset = (p: Preset) => {
    const loaded = p.schemas.map((s) => createDraft(s))
    setDrafts(loaded)
    setSelectedId(loaded[0]?.id ?? null)
    setDoc({
      mode: "paste",
      file: null,
      pasted: p.text,
      upload: { status: "idle" },
      pastedLabel: p.documentLabel,
    })
    setExisting(
      p.existing.map((e) => ({
        ...newExisting(e.schema, e.id),
        text: JSON.stringify(e.record, null, 2),
      }))
    )
    setOptions((o) => ({
      ...o,
      enableInserts: p.enableInserts,
      enableDeletes: p.enableDeletes,
    }))
    setRun({ status: "idle" })
    setActiveRecord(0)
    setActiveKey(null)
    toast.success(`Loaded preset: ${p.label}`)
  }

  const reset = () => {
    abortRef.current?.abort()
    setDoc({
      mode: "file",
      file: null,
      pasted: "",
      upload: { status: "idle" },
      pastedLabel: null,
    })
    setExisting([])
    setRun({ status: "idle" })
    setActiveKey(null)
    setActiveRecord(0)
  }

  // --------------------------------------------------------------- existing
  const checks = useMemo(() => checkExisting(existing, drafts), [existing, drafts])

  // -------------------------------------------------------------------- run
  const nameIssues = useMemo(() => {
    const issues = new Map<string, string>()
    const counts = new Map<string, number>()
    for (const d of drafts) {
      const n = d.name.trim()
      counts.set(n, (counts.get(n) ?? 0) + 1)
    }
    for (const d of drafts) {
      const n = d.name.trim()
      const fieldIssues = validateFields(d.fields)
      if (!n) issues.set(d.id, "NEEDS A NAME")
      else if ((counts.get(n) ?? 0) > 1) issues.set(d.id, "NAME IS USED TWICE")
      else if (d.jsonError) issues.set(d.id, "INVALID JSON SCHEMA")
      else if (fieldIssues.length > 0) issues.set(d.id, fieldIssues[0].toUpperCase())
    }
    return issues
  }, [drafts])

  const blockReason = useMemo((): string | null => {
    if (run.status === "running") return "A run is in progress"
    if (health.status === "checking") return "Checking the API"
    if (health.status === "unreachable") return "The API is unreachable"
    if (health.status === "no_provider") return "No provider is configured"
    if (!text.trim()) return "Add a document or paste some text"
    if (drafts.length === 0) return "Add a schema"
    for (const d of drafts) {
      const issue = nameIssues.get(d.id)
      if (issue) return `${d.name || "A schema"}: ${issue.toLowerCase()}`
    }
    for (let i = 0; i < existing.length; i += 1) {
      const c = checks[i]
      if (c?.error) {
        return `Existing record ${existing[i].id || i + 1}: ${c.error}`
      }
    }
    return null
  }, [run.status, health.status, text, drafts, nameIssues, existing, checks])

  const startRun = async () => {
    if (blockReason) return
    const request: ExtractRequest = {
      text,
      schemas: drafts.map((d) => ({
        name: d.name.trim(),
        json_schema: d.jsonSchema,
        cardinality: d.cardinality,
      })),
      existing: toExistingIn(existing, checks),
      options: {
        enable_inserts: options.enableInserts,
        enable_deletes: options.enableDeletes,
        reference_date: options.referenceDate || null,
      },
    }
    const controller = new AbortController()
    abortRef.current = controller
    const runId = newRunId()
    const source =
      doc.mode === "file" && doc.file
        ? doc.file.filename
        : (doc.pastedLabel ?? "pasted text")
    setRun({
      status: "running",
      runId,
      startedAt: Date.now(),
      text,
      schemaNames: request.schemas.map((s) => s.name),
      leafPaths: drafts.reduce((n, d) => n + leafCount(d.fields), 0),
      source,
      existingCount: request.existing.length,
    })
    setActiveRecord(0)
    setActiveKey(null)
    window.requestAnimationFrame(() =>
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
    )
    try {
      const result = await apiFetch<ExtractResponse>("/api/extract", {
        method: "POST",
        json: request,
        signal: controller.signal,
      })
      setRun({ status: "done", runId, result, text, source })
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRun({ status: "idle" })
        toast("Run cancelled")
        return
      }
      const e =
        err instanceof ApiError ? err : new ApiError(0, "error", String(err))
      setRun({ status: "error", runId, code: e.code, message: e.message })
      if (e.code === "unreachable" || e.code === "no_provider") recheck()
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }

  const cancelRun = () => abortRef.current?.abort()

  // ---------------------------------------------------------------- results
  const rowsByRecord = useMemo<FieldRow[][]>(
    () =>
      run.status === "done"
        ? run.result.records.map((r) => buildRows(r, run.text))
        : [],
    [run]
  )
  const marks = useMemo(() => {
    if (run.status !== "done") return []
    const record = run.result.records[activeRecord]
    const rows = rowsByRecord[activeRecord]
    return record && rows ? buildMarks(record, rows, run.text) : []
  }, [run, rowsByRecord, activeRecord])
  const highlightsStale = run.status === "done" && run.text !== text

  const selectField = (key: string) =>
    setActiveKey((cur) => (cur === key ? null : key))

  const selectRecord = (i: number) => {
    if (i !== activeRecord) {
      setActiveRecord(i)
      setActiveKey(null)
    }
  }

  const schemaPaths = drafts.reduce((n, d) => n + leafCount(d.fields), 0)

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-svh bg-background text-foreground">
        <AppHeader health={health} />
        <main className="mx-auto flex w-full max-w-[1440px] flex-col gap-4 px-4 pt-6 pb-12 sm:px-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="mono-label flex flex-wrap items-center gap-2">
                <span>Workspace</span>
                <span>/</span>
                <b className="font-bold text-foreground">New run</b>
              </div>
              <h1 className="mt-2 text-[28px] leading-[1.05] font-[420] tracking-[-0.045em] max-sm:text-2xl">
                Patch, don&apos;t regenerate.
              </h1>
              <p className="mt-1.5 max-w-xl text-sm/relaxed text-muted-foreground">
                Drop a document, define the schema, run. Records come back with per-field
                probabilities, source spans and every op that was refused.
              </p>
            </div>
            <PresetsMenu onLoad={loadPreset} />
          </div>

          <ProviderProblem health={health} onRecheck={recheck} />

          <div className="flex flex-wrap items-start gap-4">
            <div className="flex min-w-0 flex-[1_1_440px] flex-col gap-4">
              <DocumentPanel
                doc={doc}
                marks={marks}
                activeKey={activeKey}
                highlightsStale={highlightsStale}
                hasRun={run.status === "done"}
                onSelectMark={selectField}
                onUpload={uploadFile}
                onClearFile={() => {
                  uploadToken.current += 1
                  setDoc((d) => ({ ...d, file: null, upload: { status: "idle" } }))
                }}
                onModeChange={(mode: DocMode) => setDoc((d) => ({ ...d, mode }))}
                onPasteChange={(pasted) =>
                  setDoc((d) => ({ ...d, pasted, pastedLabel: null }))
                }
              />
            </div>
            <div className="flex min-w-0 flex-[1.35_1_520px] flex-col gap-4">
              <SchemaPanel
                drafts={drafts}
                selectedId={effectiveSelected}
                nameIssues={nameIssues}
                onSelect={setSelectedId}
                onAdd={addSchema}
                onRemove={removeSchema}
                onCardinality={setCardinality}
                onMeta={setMeta}
                onFields={(id, update: (f: FormField[]) => FormField[]) =>
                  patchDraft(id, (d) => withFields(d, update(d.fields)))
                }
                onJsonText={(id, t) => patchDraft(id, (d) => withJsonText(d, t))}
                onPydanticText={(id, t) => patchDraft(id, (d) => withPydanticText(d, t))}
                onResetPydantic={resetPydantic}
                onSync={syncPydantic}
              />
              <ExistingPanel
                entries={existing}
                checks={checks}
                schemaNames={schemaNames}
                options={options}
                onAdd={() =>
                  setExisting((all) => [
                    ...all,
                    newExisting(
                      drafts.find((d) => d.id === effectiveSelected)?.name ??
                        schemaNames[0] ??
                        ""
                    ),
                  ])
                }
                onRemove={(key) => setExisting((all) => all.filter((e) => e.key !== key))}
                onChange={(key, patch) =>
                  setExisting((all) =>
                    all.map((e) => (e.key === key ? { ...e, ...patch } : e))
                  )
                }
                onOptions={(patch) => setOptions((o) => ({ ...o, ...patch }))}
                footer={
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      size="lg"
                      onClick={startRun}
                      disabled={blockReason !== null}
                      data-testid="run-button"
                      title={blockReason ?? undefined}
                    >
                      {run.status === "running" ? "Running" : "Run extraction"}
                      <HugeiconsIcon icon={ArrowRight02Icon} strokeWidth={2} />
                    </Button>
                    <Button variant="ghost" size="lg" onClick={reset}>
                      Reset
                    </Button>
                    <span className="ml-auto min-w-0" data-testid="run-hint">
                      {blockReason ? (
                        <CodeLabel className="text-warning" data-testid="run-reason">
                          {blockReason}
                        </CodeLabel>
                      ) : (
                        <CodeLabel>
                          CLAIRIC + CLEF · {schemaPaths} LEAF PATHS ·{" "}
                          {drafts.length} {drafts.length === 1 ? "SCHEMA" : "SCHEMAS"}
                        </CodeLabel>
                      )}
                    </span>
                  </div>
                }
              />
              {drafts.some((d) => !d.pydanticSynced) ? (
                <Alert data-testid="unsynced-note">
                  <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} className="text-warning" />
                  <AlertTitle>Pydantic edits are not synced</AlertTitle>
                  <AlertDescription>
                    The run uses the form&apos;s schema. Use Sync to form on the Pydantic tab to
                    apply the edits.
                  </AlertDescription>
                </Alert>
              ) : null}
            </div>
          </div>

          <div ref={resultsRef} className="scroll-mt-4">
            {run.status === "running" ? (
              <RunningPanel
                startedAt={run.startedAt}
                schemaNames={run.schemaNames}
                leafPaths={run.leafPaths}
                source={run.source}
                chars={run.text.length}
                existingCount={run.existingCount}
                onCancel={cancelRun}
              />
            ) : null}
            {run.status === "error" ? (
              <Alert variant="destructive" data-testid="run-error">
                <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} />
                <AlertTitle>
                  Run {run.runId} failed ·{" "}
                  <span className="font-code">{run.code}</span>
                </AlertTitle>
                <AlertDescription>{run.message}</AlertDescription>
              </Alert>
            ) : null}
            {run.status === "done" ? (
              <ResultsView
                result={run.result}
                runId={run.runId}
                text={run.text}
                sourceLabel={run.source}
                rowsByRecord={rowsByRecord}
                activeRecord={activeRecord}
                activeKey={activeKey}
                onSelectRecord={selectRecord}
                onSelectField={selectField}
                onRerun={startRun}
                canRerun={blockReason === null}
              />
            ) : null}
          </div>
        </main>
        <Toaster />
      </div>
    </TooltipProvider>
  )
}

export default App
