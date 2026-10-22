"""Scoring a result against a case."""

import json
from typing import Any

from evals.model import (
    ANY,
    Case,
    CaseResult,
    MustNot,
    Op,
    Val,
    folded,
    split_pointer,
    values_equal,
)
from trustcall_system_one import ExtractionResult, ResultRecord

__all__ = ["actual_ops", "final_documents", "score"]


def _who(record: ResultRecord) -> str:
    if record.kind == "insert":
        return record.entity.label if record.entity is not None else record.schema_name
    return record.json_doc_id or ""


def actual_ops(result: ExtractionResult) -> list[Op]:
    """Every operation a result made, in the vocabulary of the cases.

    An update contributes its applied patch operations, an insert one ``insert``
    (labelled with the entity) and a delete one ``delete`` (the record id).
    """
    ops: list[Op] = []
    for record in result.records:
        who = _who(record)
        if record.kind == "insert":
            ops.append(Op("insert", "", None, who))
        elif record.kind == "delete":
            ops.append(Op("delete", "", None, who))
        else:
            ops.extend(
                Op(p.op, p.path, p.value, who)  # type: ignore[arg-type]
                for p in record.patches
            )
    return ops


def final_documents(result: ExtractionResult) -> list[tuple[str, str, dict[str, Any]]]:
    """``(kind, who, document)`` of each record that has a document."""
    return [
        (r.kind, _who(r), r.after)
        for r in result.records
        if r.kind != "delete" and r.after is not None
    ]


def _walk(document: Any, pointer: str) -> tuple[bool, Any]:
    node = document
    for part in split_pointer(pointer):
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                return False, None
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False, None
    return True, node


def _record_matches(wanted: str | None, kind: str, who: str) -> bool:
    if wanted is None:
        return True
    if kind == "insert":
        return folded(wanted) in folded(who)
    return wanted == who


def _holds(found: bool, node: Any, value: Any, contains: str | None) -> bool:
    if contains is not None:
        return found and folded(contains) in folded(node)
    if value is None:
        return not found or node is None
    return found and values_equal(node, value)


def _violated(
    rule: MustNot, ops: list[Op], docs: list[tuple[str, str, dict[str, Any]]]
) -> bool:
    if rule.final:
        for kind, who, document in docs:
            if not _record_matches(rule.who, kind, who):
                continue
            found, node = _walk(document, rule.path or "")
            if not found or node is None:
                continue
            if rule.contains is not None:
                if folded(rule.contains) in folded(node):
                    return True
            elif rule.value is ANY or values_equal(node, rule.value):
                return True
        return False
    for op in ops:
        if rule.kind is not None and op.kind != rule.kind:
            continue
        if rule.path and not (
            op.path == rule.path or op.path.startswith(rule.path.rstrip("/") + "/")
        ):
            continue
        if rule.who is not None and not _record_matches(
            rule.who, op.kind, op.who or ""
        ):
            continue
        if rule.contains is not None and folded(rule.contains) not in folded(op.value):
            continue
        if rule.value is not ANY and not values_equal(rule.value, op.value):
            continue
        return True
    return False


def _shown(op: Op) -> str:
    value = (
        ""
        if op.value is ANY or op.value is None
        else f" {json.dumps(op.value, default=str)}"
    )
    return f"{op.who}: {op.kind} {op.path}{value}".strip()


def score(case: Case, result: ExtractionResult) -> CaseResult:
    """Score one result against its case.

    Operation precision and recall use a one-to-one matching of expected and
    produced operations (an expected operation is used by at most one produced
    one). Values are read from the final documents. A ``must_not`` is violated
    when any produced operation (or final value) matches it.

    Args:
        case: The case.
        result: What the extractor returned.

    Returns:
        The score (timing and errors are filled in by the runner).
    """
    out = CaseResult(case=case.id)
    ops = actual_ops(result)
    out.ops = [_shown(op) for op in ops]
    free = list(range(len(ops)))
    for expected in case.expect_ops:
        hit = next((i for i in free if expected.matches(ops[i])), None)
        if hit is None:
            out.missing.append(expected.label())
        else:
            free.remove(hit)
            out.matched_expected += 1
    # a legitimate extra (an allowed operation) is neither right nor wrong
    allowed = [i for i in free if any(a.matches(ops[i]) for a in case.allow_ops)]
    free = [i for i in free if i not in allowed]
    out.allowed = len(allowed)
    out.n_expected = len(case.expect_ops)
    out.n_actual = len(ops) - len(allowed)
    out.matched_actual = out.n_actual - len(free)
    out.extra = [_shown(ops[i]) for i in free]
    out.n_notes = sum(1 for e in case.expect_ops if _is_note(e))
    out.notes_found = sum(
        1 for e in case.expect_ops if _is_note(e) and not _missing(out, e.label())
    )
    docs = final_documents(result)
    for val in case.expect_values:
        out.n_values += 1
        holds = _value_holds(val, docs)
        if holds:
            out.values_correct += 1
        else:
            out.missing.append(
                f"{val.who}: {val.path} == {val.contains or val.value!r}"
            )
        if val.path.rsplit("/", 1)[-1] == "notes":
            out.n_notes += 1
            out.notes_found += holds
    rejections = [
        (r.path, r.reason, _who(rec)) for rec in result.records for r in rec.rejected
    ]
    for wanted in case.expect_rejected:
        out.n_rejections += 1
        if any(
            reason == wanted.reason
            and (wanted.path is None or path == wanted.path)
            and (wanted.who is None or _record_matches(wanted.who, "update", who))
            for path, reason, who in rejections
        ):
            out.rejections_found += 1
        else:
            out.missing.append(f"rejected {wanted.reason} {wanted.path or ''}".strip())
    for rule in case.must_not:
        if _violated(rule, ops, docs):
            out.violations.append((rule.category, rule.label()))
    out.n_records = sum(1 for r in result.records if r.kind != "delete")
    out.valid = sum(1 for r in result.records if r.kind != "delete" and r.valid)
    out.requests = result.usage.requests
    out.input_tokens = result.usage.input_tokens
    out.warnings = list(result.warnings)
    for record in result.records:
        who = _who(record)
        for answer in record.fields.values():
            if answer.attribution_p is not None:
                out.probabilities[f"{who} {answer.pretty} attribution"] = (
                    answer.attribution_p
                )
            if answer.gate_p is not None:
                out.probabilities[f"{who} {answer.pretty} gate"] = answer.gate_p
    return out


def _is_note(op: Op) -> bool:
    """An operation that adds to a ``notes`` list."""
    return op.kind == "add" and op.path.endswith("/notes/-")


def _missing(out: CaseResult, label: str) -> bool:
    return label in out.missing


def _value_holds(val: Val, docs: list[tuple[str, str, dict[str, Any]]]) -> bool:
    """Whether the value holds in EVERY record the value is about."""
    about = [
        document for kind, who, document in docs if _record_matches(val.who, kind, who)
    ]
    if not about:
        return False
    for document in about:
        found, node = _walk(document, val.path)
        if val.contains is not None and isinstance(node, list):
            ok = any(folded(val.contains) in folded(item) for item in node)
        else:
            ok = _holds(found, node, val.value, val.contains)
        if not ok:
            return False
    return True
