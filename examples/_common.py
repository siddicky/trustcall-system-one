"""Shared plumbing for the examples: mode selection, extractor, pretty printing.

Every example is ``main(offline: bool) -> result`` plus a CLI. The mode rule is the
same everywhere:

* **live** when ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN`` are both set
  and ``--offline`` is not passed: requests go to the real Clairic and Clef models;
* **offline** otherwise: requests are answered by a scripted transport
  (``examples/_offline.py``) and a banner says so. Offline output is a rehearsal of
  what the library does with plausible model answers, never a model's output.

Printing is plain text, with ANSI colour only when stdout is a terminal and
``NO_COLOR`` is not set. There are no dependencies beyond the package itself.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any

from _offline import OFFLINE_BANNER, OfflineScript

from trustcall_system_one import (
    ExtractionResult,
    Extractor,
    FieldAnswer,
    ResultRecord,
    create_extractor,
)

__all__ = [
    "annotate",
    "build_extractor",
    "cli",
    "fmt",
    "heading",
    "live_available",
    "paint",
    "patch_json",
    "print_banner",
    "resolve_offline",
    "show_before_after",
    "show_mentions",
    "show_overview",
    "show_patch_lines",
    "show_patches",
    "show_record",
    "show_rejected",
    "show_skipped",
    "show_source",
    "show_summary",
    "show_verification",
    "snippet",
]

LIVE_VARIABLES = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
Span = tuple[int, int]


# --------------------------------------------------------------------------- #
# Mode
# --------------------------------------------------------------------------- #


def live_available(environ: Mapping[str, str] | None = None) -> bool:
    """Whether both Cloudflare credentials are set (live mode is possible)."""
    env = os.environ if environ is None else environ
    return all(env.get(name) for name in LIVE_VARIABLES)


def resolve_offline(offline: bool, environ: Mapping[str, str] | None = None) -> bool:
    """Final mode: offline when asked to, or when there are no credentials."""
    return offline or not live_available(environ)


def print_banner(offline: bool) -> None:
    """Say loudly which mode the example runs in."""
    if offline:
        bar = "=" * len(OFFLINE_BANNER)
        print(paint(bar, "yellow"))
        print(paint(OFFLINE_BANNER, "yellow", "bold"))
        print(paint(bar, "yellow"))
        print(
            paint(
                "Spans, confidences and verification events below were scripted to be "
                "plausible;\nthe library's checks, patches and validation are real. "
                "Set CLOUDFLARE_ACCOUNT_ID and\nCLOUDFLARE_API_TOKEN and drop "
                "--offline to run on Clairic and Clef.",
                "dim",
            )
        )
    else:
        print(
            paint(
                "LIVE — requests go to Cloudflare (Clairic and Clef)", "green", "bold"
            )
        )
    print()


def build_extractor(
    schemas: Sequence[Any],
    *,
    offline: bool,
    script: OfflineScript | None = None,
    **options: Any,
) -> Extractor:
    """Create the extractor on the scripted transport (offline) or Cloudflare.

    Args:
        schemas: Schemas or ``One`` / ``Many`` wrappers.
        offline: Resolved mode (see ``resolve_offline``).
        script: The scripted models; required offline.
        **options: Passed to ``create_extractor``.

    Returns:
        An extractor. Use it as a context manager so a live transport is closed.
    """
    if offline:
        if script is None:
            raise ValueError("offline mode needs a script")
        return create_extractor(schemas, transport=script.transport, **options)
    # No transport given: the extractor creates a CloudflareTransport() from the
    # environment on first use and closes it with the extractor.
    return create_extractor(schemas, **options)


def cli(main: Callable[..., Any], doc: str | None = None) -> None:
    """Run an example's ``main`` from the command line (``--offline``)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    first = (doc or "").strip().splitlines()[0] if doc else None
    parser = argparse.ArgumentParser(description=first)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the scripted transport even if Cloudflare credentials are set",
    )
    args = parser.parse_args()
    main(offline=args.offline)


# --------------------------------------------------------------------------- #
# Colour and small formatting helpers
# --------------------------------------------------------------------------- #

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "reverse": "7",
}


def _color_on() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def paint(text: str, *styles: str) -> str:
    """Wrap ``text`` in ANSI styles when stdout is a colour terminal."""
    if not styles or not _color_on():
        return text
    codes = ";".join(_CODES[s] for s in styles)
    return f"\033[{codes}m{text}\033[0m"


def heading(title: str) -> None:
    """Print a section heading."""
    print(paint(title, "bold"))
    print(paint("-" * len(title), "dim"))


def fmt(value: Any) -> str:
    """A short display form of a field value."""
    if value is None:
        return "null"
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple, set, frozenset)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def snippet(text: str, span: Span, *, context: int = 28) -> str:
    """The span bracketed inside a little of its surrounding text, on one line."""
    start, end = span
    lead = text[max(0, start - context) : start]
    body = text[start:end]
    tail = text[end : end + context]
    flat = [" ".join(part.split()) for part in (lead, body, tail)]
    left = " " if lead[-1:].isspace() else ""
    right = " " if tail[:1].isspace() else ""
    pre = "…" if start - context > 0 else ""
    post = "…" if end + context < len(text) else ""
    mark = paint(f"[{flat[1]}]", "reverse", "bold") if _color_on() else f"[{flat[1]}]"
    return f"{pre}{flat[0].lstrip()}{left}{mark}{right}{flat[2].rstrip()}{post}"


def _spans_of(answer: FieldAnswer) -> list[tuple[Span, str]]:
    out: list[tuple[Span, str]] = []
    if answer.spans:
        for index, span in enumerate(answer.spans):
            out.append(((span[0], span[1]), f"{answer.pretty}[{index}]"))
    elif answer.span is not None and answer.value is not None:
        out.append(((answer.span[0], answer.span[1]), answer.pretty))
    return out


def annotate(text: str, marks: Sequence[tuple[Span, str]]) -> str:
    """The text with every ``(span, label)`` bracketed and tagged ``{label}``."""
    ordered = sorted(marks, key=lambda m: m[0])
    out: list[str] = []
    cursor = 0
    for (start, end), label in ordered:
        if start < cursor:  # overlapping marks: keep the first
            continue
        out.append(text[cursor:start])
        body = text[start:end]
        if _color_on():
            out.append(paint(body, "reverse", "bold") + paint(f"{{{label}}}", "dim"))
        else:
            out.append(f"[{body}]{{{label}}}")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Showing results
# --------------------------------------------------------------------------- #


def show_source(title: str, text: str, record: ResultRecord | None = None) -> None:
    """Print the input; with a record, every value's span is bracketed in it."""
    heading(title)
    marks: list[tuple[Span, str]] = []
    if record is not None:
        for answer in record.fields.values():
            marks.extend(_spans_of(answer))
    for line in annotate(text, marks).splitlines():
        print("  " + line)
    if marks:
        print()
        legend = "  [value]{field} = the span the model pointed at for that field"
        print(paint(legend, "dim"))
    print()


def _status(answer: FieldAnswer) -> str:
    parts = [answer.engine if answer.engine != "const" else "constant"]
    raw = answer.raw
    if isinstance(raw, Mapping) and isinstance(raw.get("noul"), float):
        parts.append(f"P(yes) {_num(raw['noul'])}")
    elif answer.value is None and answer.p_null is not None:
        parts.append(f"p_null {_num(answer.p_null)}")
    elif answer.confidence is not None:
        parts.append(f"conf {_num(answer.confidence)}")
    if isinstance(answer.attribution_p, float):
        parts.append(f"attributed {_num(answer.attribution_p)}")
    elif isinstance(answer.attribution_p, tuple) and answer.attribution_p:
        parts.append("attributed " + "/".join(_num(p) for p in answer.attribution_p))
    if answer.gate_p is not None:
        parts.append(f"gate {_num(answer.gate_p)}")
    if answer.reask:
        parts.append("re-asked")
    return " · ".join(parts)


def show_record(
    record: ResultRecord, text: str | None = None, *, snippets: bool = False
) -> None:
    """Print a record's fields with value, source engine and confidence.

    Args:
        record: The record.
        text: The original input; with ``snippets`` each span is shown in context.
        snippets: Show the bracketed span under each field.
    """
    state = "valid" if record.valid else "INVALID"
    who = record.entity.label if record.entity else record.schema_name
    ident = f" id={record.json_doc_id}" if record.json_doc_id is not None else ""
    heading(f"{record.schema_name}  ({record.kind}{ident}, {state}; {who})")
    for error in record.errors:
        print(paint(f"  validation error: {error}", "red"))
    answers = list(record.fields.values())
    name_w = max((len(a.pretty) for a in answers), default=8)
    value_w = min(max((len(fmt(a.value)) for a in answers), default=4), 44)
    pad = " " * (name_w + 4)
    for answer in answers:
        shown = fmt(answer.value)
        status = paint(_status(answer), "dim")
        print(f"  {answer.pretty:<{name_w}}  {shown:<{value_w}}  {status}")
        if answer.rejected_reason:
            was = f" {fmt(answer.raw)}" if isinstance(answer.raw, str) else ""
            reason = f"rejected{was}: {answer.rejected_reason}"
            print(pad + paint("- " + reason, "red"))
        if answer.value is None and answer.guess is not None:
            note = (
                f"guess {fmt(answer.guess)} ignored: a guess is never used as a value"
            )
            print(pad + paint("- " + note, "dim"))
        if snippets and text is not None:
            for span, _ in _spans_of(answer):
                print(pad + "- " + snippet(text, span))
        for note in answer.warnings:
            print(pad + paint(f"- note: {note}", "yellow"))
    print()


def show_rejected(record: ResultRecord, title: str = "Rejected") -> None:
    """Print answers and operations the library computed and then threw away."""
    if not record.rejected:
        return
    heading(title)
    for item in record.rejected:
        what = f"{item.op} " if item.op else ""
        p = f" (p={item.p:.2f})" if item.p is not None else ""
        print(f"  {paint(what + item.path, 'red')}  {item.reason}{p}")
        if item.detail:
            print(paint(f"      {item.detail}", "dim"))
    print()


def show_patch_lines(record: ResultRecord, title: str = "Applied operations") -> None:
    """Print the applied operations one per line (``op path value``)."""
    heading(title)
    if not record.patches:
        print("  (none)")
    for op in record.patches:
        value = (
            ""
            if op.op == "remove"
            else "  " + json.dumps(op.value, default=str, ensure_ascii=False)
        )
        print(f"  {paint(op.op, 'green'):<9} {op.path}{value}")
    print()


def show_mentions(record: ResultRecord, text: str) -> None:
    """Print where the text mentions the record's entity (spans in context)."""
    entity = record.entity
    if entity is None or not entity.mentions:
        return
    heading(f"Mentions of {entity.label}")
    for mention in entity.mentions:
        if mention.span is not None:
            print("  " + snippet(text, mention.span))
    print()


def show_overview(result: ExtractionResult) -> None:
    """One table for a multi-entity run: records, patches, skipped, untouched."""
    heading("Records")
    rows = []
    for record in result.records:
        who = record.entity.label if record.entity else record.schema_name
        ident = "-" if record.json_doc_id is None else record.json_doc_id
        changes = f"{len(record.patches)} ops" if record.kind == "update" else "new"
        rows.append((ident, record.kind, who, changes, str(len(record.rejected))))
    head = ("id", "kind", "entity", "changes", "rejected")
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(5)]
    for row in [head, *rows]:
        print(
            "  " + "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True))
        )
    for item in result.skipped:
        mention = item.entity.first_mention
        who = mention.text if mention else item.entity.label
        print(
            f"  skipped {who!r}: {paint(item.reason, 'yellow')} {item.detail}".rstrip()
        )
    untouched = ", ".join(result.untouched) or "none"
    print(f"  untouched existing records: {untouched}")
    print()


def patch_json(record: ResultRecord) -> list[dict[str, Any]]:
    """The applied operations of an update as an RFC 6902 patch document."""
    patch: list[dict[str, Any]] = []
    for op in record.patches:
        entry: dict[str, Any] = {"op": op.op, "path": op.path}
        if op.op != "remove":
            entry["value"] = op.value
        patch.append(entry)
    return patch


def show_patches(record: ResultRecord, title: str = "JSON Patch (RFC 6902)") -> None:
    """Print the operations that were applied, as JSON."""
    heading(title)
    print(json.dumps(patch_json(record), indent=2, default=str, ensure_ascii=False))
    print()


def _item(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _list_diff(old: list[Any], new: list[Any]) -> list[str]:
    """Lines showing a list's change: ``+`` for new items, ``~`` for edited ones."""
    lines: list[str] = []
    if all(isinstance(item, str) for item in [*old, *new]):
        for item in new:  # scalar lists compare as sets
            mark = "  " if item in old else paint("+ ", "green")
            lines.append(f"{mark}{fmt(item)}")
        return lines
    for index, item in enumerate(new):  # lists of objects compare by position
        was = old[index] if index < len(old) else None
        if was == item:
            lines.append(f"  {_item(item)}")
        elif was is None:
            lines.append(paint("+ ", "green") + _item(item))
        elif isinstance(was, dict) and isinstance(item, dict):
            edits = ", ".join(
                f"{k}: {paint(_item(was.get(k)), 'red')} -> "
                f"{paint(_item(item[k]), 'green')}"
                for k in item
                if was.get(k) != item[k]
            )
            lines.append(paint("~ ", "yellow") + _item(item) + f"   [{edits}]")
        else:
            lines.append(paint("~ ", "yellow") + _item(item))
    return lines


def show_before_after(record: ResultRecord) -> None:
    """Print the fields that differ between ``before`` and ``after``."""
    before, after = record.before or {}, record.after or {}
    heading("Before -> after")
    changed = [key for key in after if before.get(key) != after.get(key)]
    for key in changed:
        old, new = before.get(key), after.get(key)
        if isinstance(old, list) and isinstance(new, list):
            print(f"  {key}:")
            for line in _list_diff(old, new):
                print("      " + line)
        else:
            print(f"  {key}: {paint(fmt(old), 'red')} -> {paint(fmt(new), 'green')}")
    same = [key for key in after if key not in changed]
    if same:
        print(paint("  unchanged: " + ", ".join(same), "dim"))
    print()


_GLOSS = {
    "reask": "second, isolated request for one field",
    "adjudicate": "Clef picked between two disagreeing answers",
    "null_check": "Clef asked whether the input states it at all",
    "crosscheck": "second pass over the input alone, without the record",
    "list_dedup": "Clef asked whether a new element repeats an old one",
    "repair": "narrow re-ask with the validation error in the state",
}


def show_verification(
    result: ExtractionResult, kinds: Sequence[str] | None = None
) -> None:
    """Print the verification calls (re-asks, adjudications...) and every rejection."""
    wanted = tuple(kinds) if kinds is not None else tuple(_GLOSS)
    events = [c for c in result.calls if c.kind in wanted]
    rejected = [(r.schema_name, item) for r in result.records for item in r.rejected]
    heading("Verification")
    if not events and not rejected:
        print("  every answer passed on the first try")
    for call in events:
        where = f" {call.field}" if call.field else ""
        who = f" [{call.entity}]" if call.entity and call.entity != "e0" else ""
        print(f"  {call.kind:<8}{where}{who}  " + paint(_GLOSS[call.kind], "dim"))
    for _, item in rejected:
        p = f" p={item.p:.2f}" if item.p is not None else ""
        print(f"  {paint('rejected', 'red')}   {item.path}  {item.reason}{p}")
        if item.detail:
            print(paint(f"             {item.detail}", "dim"))
    print()


def show_skipped(result: ExtractionResult) -> None:
    """Print entities that were found and not turned into records."""
    if not result.skipped:
        return
    heading("Skipped")
    for item in result.skipped:
        mention = (
            item.entity.first_mention.text
            if item.entity.first_mention
            else item.entity.label
        )
        print(f"  {mention!r}: {paint(item.reason, 'yellow')}  {item.detail}")
    print()


def show_summary(result: ExtractionResult) -> None:
    """One line: requests by kind, input tokens, priced cost."""
    usage = result.usage
    kinds = " · ".join(
        f"{kind} {used.requests}" for kind, used in sorted(usage.by_kind.items())
    )
    models = " · ".join(
        f"{model.rsplit('/', 1)[-1]} {used.input_tokens:,}"
        for model, used in sorted(usage.by_model.items())
    )
    cost = f"${result.cost.usd:.6f}"
    if not result.cost.complete:
        unpriced = ", ".join(m.rsplit("/", 1)[-1] for m in result.cost.unpriced_models)
        cost += f" + {unpriced} unpriced"
    print(paint(f"{usage.requests} requests ({kinds})", "dim"))
    print(paint(f"{usage.input_tokens:,} input tokens ({models}) · {cost}", "dim"))
    for warning in result.warnings:
        print(paint(f"warning: {warning}", "yellow"))
