import { useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Add01Icon,
  AlertCircleIcon,
  Delete02Icon,
  PencilEdit02Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons"

import { CodeEditor } from "@/components/code-editor"
import { CodeLabel, Divider, Panel } from "@/components/common"
import { FormEditor, type FieldOps } from "@/components/schema/form-editor"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { fieldCount, type SchemaDraft } from "@/lib/draft"
import { plural } from "@/lib/format"
import {
  addField,
  leafCount,
  makeField,
  removeField,
  retype,
  updateField,
  type FormField,
} from "@/lib/schema"
import type { Cardinality } from "@/lib/types"
import { cn } from "@/lib/utils"

export interface SyncError {
  code: string
  message: string
  line: number | null
}

export interface SchemaPanelProps {
  drafts: SchemaDraft[]
  selectedId: string | null
  onSelect: (id: string) => void
  onAdd: () => void
  onRemove: (id: string) => void
  onCardinality: (id: string, cardinality: Cardinality) => void
  onMeta: (id: string, meta: { name?: string; description?: string }) => void
  onFields: (id: string, update: (fields: FormField[]) => FormField[]) => void
  onJsonText: (id: string, text: string) => void
  onPydanticText: (id: string, text: string) => void
  onResetPydantic: (id: string) => void
  /** Parse the Pydantic source with the API; resolves to an error or `null`. */
  onSync: (id: string) => Promise<SyncError | null>
  nameIssues: Map<string, string>
}

function CardinalityToggle({
  value,
  onChange,
  name,
}: {
  value: Cardinality
  onChange: (c: Cardinality) => void
  name: string
}) {
  return (
    <div
      role="group"
      aria-label={`Cardinality of ${name}`}
      className="inline-flex h-7 items-center rounded-lg bg-muted p-[3px]"
    >
      {(["one", "many"] as const).map((c) => (
        <button
          key={c}
          type="button"
          aria-pressed={value === c}
          data-testid={`cardinality-${c}`}
          onClick={() => onChange(c)}
          className={cn(
            "h-full rounded-md border border-transparent px-2 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:outline-none",
            value === c && "border-border bg-input/30 text-foreground"
          )}
        >
          {c === "one" ? "One" : "Many"}
        </button>
      ))}
    </div>
  )
}

function PydanticTab({
  draft,
  onChange,
  onReset,
  onSync,
}: {
  draft: SchemaDraft
  onChange: (text: string) => void
  onReset: () => void
  onSync: () => Promise<SyncError | null>
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<SyncError | null>(null)

  const sync = async () => {
    setBusy(true)
    setError(null)
    try {
      setError(await onSync())
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-2" data-testid="pydantic-tab">
      <CodeEditor
        language="python"
        value={draft.pydantic}
        ariaLabel={`Pydantic source of ${draft.name}`}
        testId="pydantic-editor"
        minHeight="200px"
        onChange={(text) => {
          setError(null)
          onChange(text)
        }}
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" onClick={sync} disabled={busy} data-testid="sync-to-form">
          <HugeiconsIcon icon={RefreshIcon} strokeWidth={2} className={cn(busy && "animate-spin")} />
          {busy ? "Syncing" : "Sync to form"}
        </Button>
        <Button variant="ghost" onClick={onReset} disabled={draft.pydanticSynced}>
          Reset to form
        </Button>
        <CodeLabel className="ml-auto" data-testid="pydantic-status">
          {draft.pydanticSynced ? "IN SYNC WITH FORM" : "EDITED · NOT SYNCED"}
        </CodeLabel>
      </div>
      {error ? (
        <Alert variant="destructive" data-testid="sync-error">
          <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} />
          <AlertTitle>
            {error.line ? `Line ${error.line}: ` : ""}
            <span className="font-code">{error.code}</span>
          </AlertTitle>
          <AlertDescription>{error.message}</AlertDescription>
        </Alert>
      ) : null}
      <p className="text-muted-foreground">
        The source is checked and run by the API (pydantic, typing, datetime and enum names
        only). Sync replaces this schema with the model it defines.
      </p>
    </div>
  )
}

function JsonTab({
  draft,
  onChange,
}: {
  draft: SchemaDraft
  onChange: (text: string) => void
}) {
  return (
    <div className="flex flex-col gap-2" data-testid="json-tab">
      <CodeEditor
        language="json"
        value={draft.jsonText}
        ariaLabel={`JSON Schema of ${draft.name}`}
        testId="json-editor"
        minHeight="200px"
        onChange={onChange}
      />
      {draft.jsonError ? (
        <Alert variant="destructive" data-testid="json-error">
          <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} />
          <AlertTitle>Invalid JSON Schema</AlertTitle>
          <AlertDescription>{draft.jsonError}</AlertDescription>
        </Alert>
      ) : (
        <CodeLabel>VALID · EDITS UPDATE THE FORM AS YOU TYPE</CodeLabel>
      )}
      <p className="text-muted-foreground">
        Optional fields use <span className="font-code">anyOf</span> with{" "}
        <span className="font-code">{'{"type": "null"}'}</span>. The API does not read{" "}
        <span className="font-code">{'"type": ["string", "null"]'}</span>.
      </p>
    </div>
  )
}

export function SchemaPanel(props: SchemaPanelProps) {
  const { drafts, selectedId, nameIssues } = props
  const selected = drafts.find((d) => d.id === selectedId) ?? null
  const [tab, setTab] = useState("form")

  const ops = (id: string): FieldOps => ({
    update: (fid, patch) =>
      props.onFields(id, (fields) => updateField(fields, fid, (f) => ({ ...f, ...patch }))),
    retype: (fid, type) =>
      props.onFields(id, (fields) =>
        updateField(fields, fid, (f) => retype(f, type))
      ),
    remove: (fid) => props.onFields(id, (fields) => removeField(fields, fid)),
    add: (parentId) =>
      props.onFields(id, (fields) => addField(fields, parentId, makeField({ name: "" }))),
  })

  return (
    <Panel
      title="Schema"
      description="Define the records to extract and how many of each."
      aside={
        <CodeLabel>
          {plural(drafts.length, "SCHEMA")}
          {selected ? ` · EDITING ${selected.name.toUpperCase() || "UNNAMED"}` : ""}
        </CodeLabel>
      }
    >
      <div className="flex flex-col gap-2" data-testid="schema-list">
        {drafts.map((d) => {
          const on = d.id === selectedId
          const issue = nameIssues.get(d.id)
          return (
            <div
              key={d.id}
              data-testid="schema-row"
              data-selected={on}
              className={cn(
                "flex flex-wrap items-center gap-2 rounded-md border border-line p-2",
                on && "border-border bg-secondary/50"
              )}
            >
              <button
                type="button"
                role="radio"
                aria-checked={on}
                aria-label={`Edit ${d.name || "unnamed schema"}`}
                onClick={() => props.onSelect(d.id)}
                className="flex min-w-36 flex-1 items-center gap-2 rounded-sm text-left focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:outline-none"
              >
                <span
                  aria-hidden
                  className={cn(
                    "relative size-4 shrink-0 rounded-full border border-border bg-input/20",
                    on && "border-foreground after:absolute after:inset-[3px] after:rounded-full after:bg-foreground"
                  )}
                />
                <span className="flex min-w-0 flex-col gap-px">
                  <span className="truncate font-code text-xs font-bold">
                    {d.name || "(unnamed)"}
                  </span>
                  <CodeLabel className={cn(issue && "text-destructive")}>
                    {issue ??
                      `${plural(fieldCount(d), "FIELD")} · ${plural(leafCount(d.fields), "LEAF PATH")}`}
                  </CodeLabel>
                </span>
              </button>
              <CardinalityToggle
                value={d.cardinality}
                name={d.name}
                onChange={(c) => props.onCardinality(d.id, c)}
              />
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Edit schema ${d.name}`}
                onClick={() => {
                  props.onSelect(d.id)
                  document.getElementById("schema-name")?.focus()
                }}
              >
                <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={2} />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Remove schema ${d.name}`}
                onClick={() => props.onRemove(d.id)}
              >
                <HugeiconsIcon icon={Delete02Icon} strokeWidth={2} />
              </Button>
            </div>
          )
        })}
        {drafts.length === 0 ? (
          <p className="rounded-md border border-dashed border-border px-3 py-4 text-muted-foreground">
            No schemas. Add one, or load a preset.
          </p>
        ) : null}
        <div>
          <Button variant="outline" size="sm" onClick={props.onAdd} data-testid="add-schema">
            <HugeiconsIcon icon={Add01Icon} strokeWidth={2} />
            Add schema
          </Button>
        </div>
      </div>

      {selected ? (
        <>
          <Divider />
          <div className="flex flex-wrap gap-3">
            <div className="flex min-w-[160px] flex-[1_1_180px] flex-col gap-2">
              <label htmlFor="schema-name" className="text-xs font-medium">
                Schema name
              </label>
              <Input
                id="schema-name"
                value={selected.name}
                aria-invalid={nameIssues.has(selected.id)}
                className="font-code"
                onChange={(e) => props.onMeta(selected.id, { name: e.target.value })}
              />
            </div>
            <div className="flex min-w-[160px] flex-[3_1_240px] flex-col gap-2">
              <label htmlFor="schema-desc" className="text-xs font-medium">
                Description
              </label>
              <Input
                id="schema-desc"
                value={selected.description}
                placeholder="What one record is"
                onChange={(e) => props.onMeta(selected.id, { description: e.target.value })}
              />
            </div>
          </div>

          <Tabs value={tab} onValueChange={setTab} className="gap-3">
            <TabsList>
              <TabsTrigger value="form" data-testid="tab-form">
                Form
              </TabsTrigger>
              <TabsTrigger value="pydantic" data-testid="tab-pydantic">
                Pydantic
                {!selected.pydanticSynced ? (
                  <span aria-label="unsynced edits" className="size-1.5 rounded-full bg-warning" />
                ) : null}
              </TabsTrigger>
              <TabsTrigger value="json" data-testid="tab-json">
                JSON Schema
                {selected.jsonError ? (
                  <span aria-label="invalid" className="size-1.5 rounded-full bg-destructive" />
                ) : null}
              </TabsTrigger>
            </TabsList>
            <TabsContent value="form" className="flex flex-col gap-2">
              <p className="text-muted-foreground">
                Each row becomes a typed question. Descriptions are the prompt, so write them as
                instructions. <span data-testid="leaf-count">{plural(leafCount(selected.fields), "leaf path")}</span>.
              </p>
              <FormEditor fields={selected.fields} ops={ops(selected.id)} />
            </TabsContent>
            <TabsContent value="pydantic">
              <PydanticTab
                key={selected.id}
                draft={selected}
                onChange={(text) => props.onPydanticText(selected.id, text)}
                onReset={() => props.onResetPydantic(selected.id)}
                onSync={() => props.onSync(selected.id)}
              />
            </TabsContent>
            <TabsContent value="json">
              <JsonTab
                draft={selected}
                onChange={(text) => props.onJsonText(selected.id, text)}
              />
            </TabsContent>
          </Tabs>
        </>
      ) : null}
    </Panel>
  )
}
