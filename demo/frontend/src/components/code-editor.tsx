import CodeMirror from "@uiw/react-codemirror"
import { json } from "@codemirror/lang-json"
import { python } from "@codemirror/lang-python"
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language"
import { EditorView } from "@codemirror/view"
import { tags as t } from "@lezer/highlight"
import { useMemo } from "react"

import { cn } from "@/lib/utils"

/** Colours come from the theme tokens; orange is not used for syntax. */
const editorTheme = EditorView.theme(
  {
    "&": {
      color: "var(--foreground)",
      backgroundColor: "var(--sunken)",
      fontSize: "12px",
    },
    ".cm-scroller": {
      fontFamily: "var(--font-code)",
      lineHeight: "1.75",
    },
    ".cm-content": {
      caretColor: "var(--foreground)",
      fontVariantLigatures: "none",
      padding: "10px 0",
    },
    ".cm-line": { padding: "0 12px" },
    "&.cm-focused": { outline: "none" },
    ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--foreground)" },
    "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection":
      { backgroundColor: "#a3a9b640" },
    ".cm-gutters": {
      backgroundColor: "var(--sunken)",
      color: "var(--faint)",
      border: "none",
      fontVariantLigatures: "none",
    },
    ".cm-activeLine, .cm-activeLineGutter": { backgroundColor: "transparent" },
    ".cm-placeholder": { color: "var(--faint)" },
  },
  { dark: true }
)

const highlightStyle = HighlightStyle.define([
  { tag: [t.keyword, t.operatorKeyword, t.modifier], color: "var(--info)" },
  { tag: [t.string, t.special(t.string)], color: "#d0d3da" },
  { tag: [t.number, t.bool, t.null], color: "var(--warning)" },
  { tag: t.comment, color: "var(--faint)", fontStyle: "italic" },
  { tag: [t.className, t.definition(t.variableName)], color: "var(--foreground)", fontWeight: "600" },
  { tag: [t.propertyName], color: "var(--info)" },
  { tag: [t.typeName, t.function(t.variableName)], color: "#c9cdd5" },
  { tag: [t.punctuation, t.bracket, t.operator], color: "var(--muted-foreground)" },
])

export type CodeLanguage = "python" | "json"

export function CodeEditor({
  value,
  onChange,
  language,
  readOnly = false,
  minHeight = "140px",
  maxHeight = "420px",
  ariaLabel,
  testId,
  lineNumbers = true,
  className,
}: {
  value: string
  onChange?: (value: string) => void
  language: CodeLanguage
  readOnly?: boolean
  minHeight?: string
  maxHeight?: string
  ariaLabel: string
  testId?: string
  lineNumbers?: boolean
  className?: string
}) {
  const extensions = useMemo(
    () => [
      language === "python" ? python() : json(),
      editorTheme,
      syntaxHighlighting(highlightStyle),
      EditorView.contentAttributes.of({ "aria-label": ariaLabel }),
    ],
    [language, ariaLabel]
  )
  return (
    <div
      data-testid={testId}
      className={cn(
        "min-w-0 overflow-hidden rounded-md border border-line bg-sunken focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/30",
        className
      )}
    >
      <CodeMirror
        value={value}
        onChange={onChange}
        extensions={extensions}
        theme="none"
        readOnly={readOnly}
        editable={!readOnly}
        minHeight={minHeight}
        maxHeight={maxHeight}
        basicSetup={{
          lineNumbers,
          foldGutter: false,
          highlightActiveLine: false,
          highlightActiveLineGutter: false,
          autocompletion: false,
          indentOnInput: true,
          bracketMatching: true,
          closeBrackets: true,
          searchKeymap: false,
        }}
      />
    </div>
  )
}
