import type { ComponentProps, ReactNode } from "react"

import { cn } from "@/lib/utils"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

/** Uppercase mono caption in the muted colour. */
export function CodeLabel({ className, ...props }: ComponentProps<"span">) {
  return <span className={cn("code-label", className)} {...props} />
}

/** The mockup's `.m-lab`: a section label. */
export function MonoLabel({ className, ...props }: ComponentProps<"p">) {
  return <p className={cn("mono-label", className)} {...props} />
}

export function StatusDot({
  tone,
  className,
}: {
  tone: "ok" | "bad" | "idle"
  className?: string
}) {
  return (
    <i
      aria-hidden
      className={cn(
        "inline-block size-1.5 shrink-0 rounded-full",
        tone === "ok" && "bg-success shadow-[0_0_9px_var(--success)]",
        tone === "bad" && "bg-destructive",
        tone === "idle" && "bg-muted-foreground/60",
        className
      )}
    />
  )
}

export function Panel({
  title,
  description,
  aside,
  children,
  className,
  bodyClassName,
  id,
}: {
  title: ReactNode
  description?: ReactNode
  aside?: ReactNode
  children?: ReactNode
  className?: string
  bodyClassName?: string
  id?: string
}) {
  return (
    <Card id={id} className={cn("min-w-0", className)}>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
        {aside ? <CardAction>{aside}</CardAction> : null}
      </CardHeader>
      {children !== undefined ? (
        <CardContent className={cn("flex min-w-0 flex-col gap-3", bodyClassName)}>
          {children}
        </CardContent>
      ) : null}
    </Card>
  )
}

export function Divider({ className }: { className?: string }) {
  return <div role="separator" className={cn("h-px w-full bg-line", className)} />
}

/** A key/value row, as in the Usage card. */
export function KeyValue({
  label,
  value,
  mono = true,
}: {
  label: ReactNode
  value: ReactNode
  mono?: boolean
}) {
  return (
    <div className="flex justify-between gap-3 border-b border-line py-2 last:border-b-0">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn("text-right", mono && "font-mono")}>{value}</span>
    </div>
  )
}
