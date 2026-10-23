import { useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Add01Icon,
  ArrowDown01Icon,
  Cancel01Icon,
  Delete02Icon,
} from "@hugeicons/core-free-icons"

import { CodeLabel, MonoLabel } from "@/components/common"
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
import {
  duplicateNames,
  FIELD_TYPES,
  hasChildren,
  hasEnum,
  ITEM_TYPES,
  type FieldType,
  type FormField,
  type ItemType,
} from "@/lib/schema"
import { cn } from "@/lib/utils"

export interface FieldOps {
  update: (id: string, patch: Partial<FormField>) => void
  retype: (id: string, type: FieldType) => void
  remove: (id: string) => void
  add: (parentId: string | null) => void
}

const GRID =
  "grid grid-cols-[24px_minmax(0,1fr)_112px] items-center gap-x-2 gap-y-1.5 @xl:grid-cols-[24px_minmax(110px,1.05fr)_132px_minmax(140px,2fr)_40px_28px]"

function TypeSelect({
  value,
  options,
  onChange,
  label,
  className,
  size,
}: {
  value: string
  options: readonly string[]
  onChange: (value: string) => void
  label: string
  className?: string
  size?: "sm" | "default"
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger
        size={size}
        aria-label={label}
        className={cn("w-full font-code", className)}
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o} value={o} className="font-code">
            {o}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function EnumValues({
  values,
  onChange,
  name,
}: {
  values: string[]
  onChange: (values: string[]) => void
  name: string
}) {
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState("")

  const commit = () => {
    const parts = draft
      .split(",")
      .map((p) => p.trim())
      .filter((p) => p && !values.includes(p))
    if (parts.length > 0) onChange([...values, ...parts])
    setDraft("")
    setAdding(false)
  }

  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
      <MonoLabel className="mr-1">Values</MonoLabel>
      {values.map((v) => (
        <Badge key={v} variant="secondary" className="gap-1 pr-1 font-code">
          {v}
          <button
            type="button"
            aria-label={`Remove value ${v} from ${name}`}
            className="rounded-full p-px text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:outline-none"
            onClick={() => onChange(values.filter((x) => x !== v))}
          >
            <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} className="size-2.5" />
          </button>
        </Badge>
      ))}
      {adding ? (
        <Input
          autoFocus
          aria-label={`New value for ${name}`}
          value={draft}
          placeholder="value, value"
          className="h-6 w-36 font-code"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault()
              commit()
            }
            if (e.key === "Escape") {
              setDraft("")
              setAdding(false)
            }
          }}
        />
      ) : (
        <Button variant="outline" size="xs" onClick={() => setAdding(true)}>
          + Add value
        </Button>
      )}
    </div>
  )
}

function FieldRow({
  field,
  depth,
  dup,
  ops,
  collapsed,
  toggle,
  isCollapsed,
}: {
  field: FormField
  depth: number
  dup: Set<string>
  ops: FieldOps
  collapsed: ReadonlySet<string>
  toggle: (id: string) => void
  isCollapsed: boolean
}) {
  const nested = hasChildren(field)
  const name = field.name.trim()
  const invalid = !name || dup.has(name)
  const childDup = nested ? duplicateNames(field.children) : null
  const label = name || "field"
  const parentLabel = field.name.trim() || "field"

  return (
    <>
      <div
        className={cn(
          "border-b border-line py-1.5 pr-3",
          depth > 0 && "bg-sunken/50"
        )}
        style={{ paddingLeft: 12 + depth * 20 }}
        data-testid="field-row"
        data-depth={depth}
      >
        <div className={GRID}>
          <span className="flex justify-center">
            {nested ? (
              <button
                type="button"
                aria-label={`${isCollapsed ? "Expand" : "Collapse"} ${label}`}
                aria-expanded={!isCollapsed}
                onClick={() => toggle(field.id)}
                className="grid size-6 place-items-center rounded-sm text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/30 focus-visible:outline-none"
              >
                <HugeiconsIcon
                  icon={ArrowDown01Icon}
                  strokeWidth={2}
                  className={cn("size-3.5 transition-transform", isCollapsed && "-rotate-90")}
                />
              </button>
            ) : null}
          </span>
          <Input
            aria-label="Field name"
            aria-invalid={invalid}
            value={field.name}
            placeholder="field_name"
            className="col-start-2 font-code @xl:col-start-auto"
            onChange={(e) => ops.update(field.id, { name: e.target.value })}
          />
          <TypeSelect
            label={`Type of ${label}`}
            value={field.type}
            options={FIELD_TYPES}
            onChange={(v) => ops.retype(field.id, v as FieldType)}
            className="col-start-3 @xl:col-start-auto"
          />
          <Input
            aria-label="Description"
            value={field.description}
            placeholder="Description (this is the prompt)"
            className="col-span-2 col-start-2 @xl:col-span-1 @xl:col-start-auto"
            onChange={(e) => ops.update(field.id, { description: e.target.value })}
          />
          <label className="col-start-2 flex items-center gap-2 text-xs text-muted-foreground @xl:col-start-auto @xl:justify-center">
            <Switch
              size="default"
              checked={field.required}
              aria-label={`Required: ${label}`}
              onCheckedChange={(v) => ops.update(field.id, { required: v })}
            />
            <span className="@xl:hidden">Required</span>
          </label>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={`Delete ${label}`}
            className="col-start-3 justify-self-end @xl:col-start-auto"
            onClick={() => ops.remove(field.id)}
          >
            <HugeiconsIcon icon={Delete02Icon} strokeWidth={2} />
          </Button>
        </div>

        {field.type === "array" ? (
          <div className="mt-1.5 flex flex-wrap items-center gap-2 pl-8">
            <MonoLabel className="mr-1">Item type</MonoLabel>
            <TypeSelect
              size="sm"
              label={`Item type of ${label}`}
              value={field.itemType}
              options={ITEM_TYPES}
              onChange={(v) => ops.update(field.id, { itemType: v as ItemType })}
              className="h-6 w-32"
            />
            <CodeLabel>
              LIST OF {field.itemType.toUpperCase()}
              {field.itemType === "object"
                ? ` · ${field.children.length} CHILD ${field.children.length === 1 ? "FIELD" : "FIELDS"}`
                : ""}
            </CodeLabel>
          </div>
        ) : null}

        {hasEnum(field) ? (
          <div className="pl-8">
            <EnumValues
              name={label}
              values={field.enumValues}
              onChange={(enumValues) => ops.update(field.id, { enumValues })}
            />
          </div>
        ) : null}
      </div>

      {nested && !isCollapsed ? (
        <>
          {field.children.map((child) => (
            <FieldRow
              key={child.id}
              field={child}
              depth={depth + 1}
              dup={childDup ?? new Set()}
              ops={ops}
              collapsed={collapsed}
              toggle={toggle}
              isCollapsed={collapsed.has(child.id)}
            />
          ))}
          <div
            className="border-b border-line bg-sunken/50 py-1.5 pr-3"
            style={{ paddingLeft: 12 + (depth + 1) * 20 }}
          >
            <Button variant="ghost" size="sm" onClick={() => ops.add(field.id)}>
              <HugeiconsIcon icon={Add01Icon} strokeWidth={2} />
              Add field to {parentLabel}
            </Button>
          </div>
        </>
      ) : null}
    </>
  )
}

/** The rows of the form editor and the Add field button. */
export function FormEditor({
  fields,
  ops,
}: {
  fields: FormField[]
  ops: FieldOps
}) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set())
  const toggle = (id: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  const dup = duplicateNames(fields)

  return (
    <div className="@container min-w-0" data-testid="form-editor">
      <div className="overflow-hidden rounded-md border border-line">
        <div
          className="hidden items-center border-b border-line px-3 py-2 @xl:block"
          aria-hidden
        >
          <div className={cn(GRID, "text-xs font-medium")}>
            <span />
            <span>Name</span>
            <span>Type</span>
            <span>Description</span>
            <span>Req.</span>
            <span />
          </div>
        </div>
        {fields.length === 0 ? (
          <p className="px-3 py-4 text-muted-foreground">
            No fields yet. Add one to describe what to extract.
          </p>
        ) : null}
        {fields.map((f) => (
          <FieldRow
            key={f.id}
            field={f}
            depth={0}
            dup={dup}
            ops={ops}
            collapsed={collapsed}
            toggle={toggle}
            isCollapsed={collapsed.has(f.id)}
          />
        ))}
        <div className="flex items-center px-3 py-2.5">
          <Button variant="outline" onClick={() => ops.add(null)} data-testid="add-field">
            <HugeiconsIcon icon={Add01Icon} strokeWidth={2} />
            Add field
          </Button>
        </div>
      </div>
    </div>
  )
}
