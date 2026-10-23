import { useRef, useState, type DragEvent } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  AlertCircleIcon,
  Cancel01Icon,
  Copy01Icon,
  File01Icon,
  Loading03Icon,
  Upload04Icon,
} from "@hugeicons/core-free-icons"
import { toast } from "sonner"

import { CodeLabel, MonoLabel, Panel, KeyValue } from "@/components/common"
import { HighlightedText } from "@/components/highlighted-text"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { formatBytes, formatInt } from "@/lib/format"
import type { Mark } from "@/lib/results"
import type { DocumentInfo } from "@/lib/types"
import { cn } from "@/lib/utils"

export type DocMode = "file" | "paste"

export type UploadState =
  | { status: "idle" }
  | { status: "uploading"; name: string }
  | {
      status: "error"
      name: string
      size: number
      kind: "pdf" | "other"
      code: string
      message: string
    }

export interface DocState {
  mode: DocMode
  file: DocumentInfo | null
  pasted: string
  upload: UploadState
  /** Label of a preset document pasted in. */
  pastedLabel: string | null
}

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024

const ERROR_TITLES: Record<string, string> = {
  pdf_no_text: "PDF has no text layer",
  pdf_parse_error: "PDF parse failed",
  unsupported_type: "Unsupported file type",
  too_large: "File too large",
  unreachable: "API unreachable",
}

export function docText(doc: DocState): string {
  return doc.mode === "file" ? (doc.file?.text ?? "") : doc.pasted
}

function FileRow({
  name,
  meta,
  badge,
  onRemove,
}: {
  name: string
  meta: string
  badge: { label: string; failed?: boolean }
  onRemove?: () => void
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border border-line p-2">
      <HugeiconsIcon
        icon={File01Icon}
        strokeWidth={2}
        className="size-4 shrink-0 text-muted-foreground"
      />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="truncate font-code text-xs" title={name} data-testid="file-name">
          {name}
        </span>
        <CodeLabel data-testid="file-meta">{meta}</CodeLabel>
      </div>
      <Badge variant={badge.failed ? "destructive" : "outline"} className="font-mono tracking-[0.06em] uppercase">
        {badge.label}
      </Badge>
      {onRemove ? (
        <Button variant="ghost" size="icon-sm" aria-label="Remove file" onClick={onRemove}>
          <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} />
        </Button>
      ) : null}
    </div>
  )
}

export function DocumentPanel({
  doc,
  marks,
  activeKey,
  highlightsStale,
  hasRun,
  onSelectMark,
  onUpload,
  onClearFile,
  onModeChange,
  onPasteChange,
}: {
  doc: DocState
  marks: Mark[]
  activeKey: string | null
  /** The text was edited after the last run, so the marks no longer apply. */
  highlightsStale: boolean
  hasRun: boolean
  onSelectMark: (key: string) => void
  onUpload: (file: File) => void
  onClearFile: () => void
  onModeChange: (mode: DocMode) => void
  onPasteChange: (text: string) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [editing, setEditing] = useState(false)
  const text = docText(doc)
  const showMarks = marks.length > 0 && !highlightsStale
  const { upload, file } = doc

  const pick = () => inputRef.current?.click()
  const take = (list: FileList | null) => {
    const f = list?.[0]
    if (f) onUpload(f)
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    take(e.dataTransfer.files)
  }
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success("Text copied")
    } catch {
      toast.error("Could not copy the text")
    }
  }

  const compact = file !== null || upload.status !== "idle"

  return (
    <Panel
      title="Document"
      description="The text is read by the API and sent to the models only when you run."
      aside={
        <div className="flex items-center gap-2">
          <CodeLabel className="hidden sm:inline">MD · PDF · TXT · 10 MB MAX</CodeLabel>
          <div
            role="group"
            aria-label="Document source"
            className="inline-flex h-7 items-center rounded-lg bg-muted p-[3px]"
          >
            {(["file", "paste"] as const).map((m) => (
              <button
                key={m}
                type="button"
                data-testid={`doc-mode-${m}`}
                aria-pressed={doc.mode === m}
                onClick={() => onModeChange(m)}
                className={cn(
                  "h-full rounded-md border border-transparent px-2 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:outline-none",
                  doc.mode === m && "border-border bg-input/30 text-foreground"
                )}
              >
                {m === "file" ? "Upload" : "Paste text"}
              </button>
            ))}
          </div>
        </div>
      }
      bodyClassName="gap-3"
    >
      {doc.mode === "file" ? (
        <>
          <div
            data-testid="dropzone"
            data-dragging={dragging}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={cn(
              "flex flex-col items-center gap-1.5 rounded-lg border border-dashed border-border bg-input/10 px-4 text-center transition-colors",
              compact ? "py-3" : "py-5",
              dragging && "border-foreground bg-input/30"
            )}
          >
            {!compact ? (
              <HugeiconsIcon
                icon={Upload04Icon}
                strokeWidth={2}
                className="size-6 text-muted-foreground"
              />
            ) : null}
            <b className="text-xs font-medium">
              {compact ? "Drop another file to replace it" : "Drop a .md or .pdf file"}
            </b>
            <span className="text-muted-foreground">
              {compact ? ".md, .markdown, .txt or .pdf" : "or choose a file from disk"}
            </span>
            <Button
              variant="outline"
              className="mt-1"
              onClick={pick}
              disabled={upload.status === "uploading"}
            >
              Browse files
            </Button>
            <input
              ref={inputRef}
              type="file"
              data-testid="file-input"
              accept=".md,.markdown,.txt,.pdf,text/markdown,text/plain,application/pdf"
              className="sr-only"
              tabIndex={-1}
              onChange={(e) => {
                take(e.target.files)
                e.target.value = ""
              }}
            />
          </div>

          {upload.status === "uploading" ? (
            <div
              className="flex items-center gap-2 rounded-md border border-line p-2"
              role="status"
            >
              <HugeiconsIcon
                icon={Loading03Icon}
                strokeWidth={2}
                className="size-4 animate-spin text-muted-foreground"
              />
              <span className="truncate font-code text-xs">{upload.name}</span>
              <CodeLabel className="ml-auto">READING</CodeLabel>
            </div>
          ) : null}

          {upload.status === "error" ? (
            <>
              <FileRow
                name={upload.name}
                meta={`${formatBytes(upload.size)} · 0 chars`}
                badge={{ label: "Failed", failed: true }}
              />
              <Alert variant="destructive" data-testid="upload-error">
                <HugeiconsIcon icon={AlertCircleIcon} strokeWidth={2} />
                <AlertTitle>{ERROR_TITLES[upload.code] ?? "Upload failed"}</AlertTitle>
                <AlertDescription>{upload.message}</AlertDescription>
              </Alert>
              <div>
                <MonoLabel className="mb-2">Extracted text</MonoLabel>
                <div className="flex flex-col items-center gap-1.5 rounded-lg border border-dashed border-border bg-input/10 px-4 py-6 text-center">
                  <CodeLabel>NO TEXT EXTRACTED</CodeLabel>
                  <span className="text-muted-foreground">
                    {upload.kind === "pdf"
                      ? "A scanned PDF has no text layer. Export a text PDF, or upload markdown."
                      : "Nothing was read from this file."}
                  </span>
                </div>
              </div>
              <div>
                <KeyValue label="Parser" value={upload.kind === "pdf" ? "pypdf" : "text"} />
                <KeyValue label="Error" value={upload.code.toUpperCase()} />
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" size="lg" onClick={pick}>
                  Replace file
                </Button>
                <Button variant="ghost" size="lg" onClick={() => onModeChange("paste")}>
                  Paste text instead
                </Button>
              </div>
            </>
          ) : null}

          {file && upload.status !== "error" ? (
            <FileRow
              name={file.filename}
              meta={`${file.pages !== null ? `${file.pages} ${file.pages === 1 ? "page" : "pages"} · ` : ""}${formatInt(file.chars)} chars`}
              badge={{ label: "Parsed" }}
              onRemove={onClearFile}
            />
          ) : null}

          {file && upload.status !== "error" ? (
            <>
              <div className="flex items-center justify-between gap-2">
                <MonoLabel>Extracted text</MonoLabel>
                <div className="flex items-center gap-2">
                  {highlightsStale ? <CodeLabel>HIGHLIGHTS STALE</CodeLabel> : null}
                  <Button variant="ghost" size="icon-sm" aria-label="Copy extracted text" onClick={copy}>
                    <HugeiconsIcon icon={Copy01Icon} strokeWidth={2} />
                  </Button>
                </div>
              </div>
              <HighlightedText
                text={text}
                marks={showMarks ? marks : []}
                activeKey={showMarks ? activeKey : null}
                onSelect={onSelectMark}
              />
            </>
          ) : null}
        </>
      ) : (
        <>
          <div className="flex items-center justify-between gap-2">
            <MonoLabel>
              {doc.pastedLabel ? `Pasted · ${doc.pastedLabel}` : "Pasted text"}
            </MonoLabel>
            <div className="flex items-center gap-2">
              <CodeLabel data-testid="paste-count">{formatInt(text.length)} CHARS</CodeLabel>
              {text ? (
                <Button variant="ghost" size="icon-sm" aria-label="Copy text" onClick={copy}>
                  <HugeiconsIcon icon={Copy01Icon} strokeWidth={2} />
                </Button>
              ) : null}
            </div>
          </div>
          {hasRun && !editing && showMarks ? (
            <>
              <HighlightedText
                text={text}
                marks={marks}
                activeKey={activeKey}
                onSelect={onSelectMark}
              />
              <div>
                <Button variant="outline" size="sm" onClick={() => setEditing(true)}>
                  Edit text
                </Button>
              </div>
            </>
          ) : (
            <>
            <Textarea
              data-testid="paste-input"
              aria-label="Text to extract from"
              value={doc.pasted}
              onChange={(e) => onPasteChange(e.target.value)}
              placeholder="Paste a note, an email or a transcript here."
              className="max-h-[420px] min-h-[220px] text-xs/[1.75]"
            />
            {hasRun && showMarks ? (
              <div>
                <Button variant="outline" size="sm" onClick={() => setEditing(false)}>
                  Show highlights
                </Button>
              </div>
            ) : null}
            </>
          )}
        </>
      )}
    </Panel>
  )
}
