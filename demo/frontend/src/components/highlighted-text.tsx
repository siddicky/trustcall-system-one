import { Fragment, useEffect, useMemo, useRef } from "react"

import type { Mark } from "@/lib/results"
import { cn } from "@/lib/utils"

interface Piece {
  start: number
  end: number
  covering: Mark[]
}

/** Split `text` at every mark boundary. */
function split(text: string, marks: Mark[]): Piece[] {
  const cuts = new Set<number>([0, text.length])
  for (const m of marks) {
    cuts.add(Math.max(0, Math.min(text.length, m.start)))
    cuts.add(Math.max(0, Math.min(text.length, m.end)))
  }
  const sorted = [...cuts].sort((a, b) => a - b)
  const pieces: Piece[] = []
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const [start, end] = [sorted[i], sorted[i + 1]]
    if (end <= start) continue
    pieces.push({
      start,
      end,
      covering: marks.filter((m) => m.start <= start && m.end >= end),
    })
  }
  return pieces
}

/**
 * The text with its marks highlighted. A mark with a number shows it as a
 * superscript where it ends. The active row's marks are drawn stronger and
 * scrolled into view inside this element (never the page).
 */
export function HighlightedText({
  text,
  marks,
  activeKey,
  onSelect,
  className,
}: {
  text: string
  marks: Mark[]
  activeKey: string | null
  onSelect?: (key: string) => void
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const pieces = useMemo(() => split(text, marks), [text, marks])

  useEffect(() => {
    const box = ref.current
    if (!box || !activeKey) return
    const mark = box.querySelector<HTMLElement>(
      `mark[data-key="${CSS.escape(activeKey)}"]`
    )
    if (!mark) return
    const top =
      mark.getBoundingClientRect().top -
      box.getBoundingClientRect().top +
      box.scrollTop
    box.scrollTo({
      top: Math.max(0, top - box.clientHeight / 3),
      behavior: "smooth",
    })
  }, [activeKey, text])

  return (
    <div
      ref={ref}
      className={cn(
        "relative max-h-[340px] overflow-auto rounded-md border border-line bg-sunken p-3 text-xs/[1.9] break-words whitespace-pre-wrap text-foreground/85",
        className
      )}
      data-testid="highlighted-text"
    >
      {pieces.map((piece) => {
        const chunk = text.slice(piece.start, piece.end)
        if (piece.covering.length === 0) {
          return <Fragment key={piece.start}>{chunk}</Fragment>
        }
        const primary =
          piece.covering.find((m) => m.kind === "field") ?? piece.covering[0]
        const active = piece.covering.some(
          (m) => m.kind === "field" && m.key === activeKey
        )
        const ending = piece.covering.filter(
          (m) => m.end === piece.end && m.n !== undefined
        )
        return (
          <mark
            key={piece.start}
            data-hl={primary.kind}
            data-key={active ? activeKey : primary.key}
            data-active={active}
            onClick={
              onSelect && primary.kind === "field"
                ? () => onSelect(primary.key)
                : undefined
            }
            className={onSelect && primary.kind === "field" ? "cursor-pointer" : undefined}
          >
            {chunk}
            {ending.map((m) => (
              <sup key={`${m.key}-${m.n}`}>{m.n}</sup>
            ))}
          </mark>
        )
      })}
    </div>
  )
}
