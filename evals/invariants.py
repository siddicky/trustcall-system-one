"""Invariants that hold for any result, whatever the models said.

They are what the live smoke tests and the cassette replays assert: no exact
values (those belong to the evals), only properties the library guarantees.
"""

import json
from collections.abc import Collection, Iterable, Sequence

import jsonpatch

from evals.model import folded, split_pointer
from trustcall_system_one import ExtractionResult, ResultRecord

__all__ = [
    "every_patch_applies",
    "ids_are_distinct_and_existing",
    "no_op_inside",
    "only_kinds",
    "spans_quote_the_input",
    "unmentioned_fields_are_untouched",
    "updates",
]


def updates(result: ExtractionResult) -> list[ResultRecord]:
    """The update records of a result."""
    return [r for r in result.records if r.kind == "update"]


def every_patch_applies(result: ExtractionResult) -> None:
    """Applying each record's patch to its ``before`` gives its ``after``."""
    for record in updates(result):
        assert record.before is not None and record.after is not None
        patch = [
            {
                "op": p.op,
                "path": p.path,
                **({} if p.op == "remove" else {"value": p.value}),
            }
            for p in record.patches
        ]
        applied = jsonpatch.apply_patch(record.before, patch)
        assert applied == record.after, (
            f"record {record.json_doc_id!r}: the patch does not give the document: "
            f"{json.dumps(patch, default=str)}"
        )


def _touches(path: str, other: str) -> bool:
    a, b = split_pointer(path), split_pointer(other)
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def unmentioned_fields_are_untouched(result: ExtractionResult) -> None:
    """A top-level field that no applied operation touches is exactly as it was."""
    for record in updates(result):
        assert record.before is not None and record.after is not None
        for key, was in record.before.items():
            if any(_touches(p.path, f"/{key}") for p in record.patches):
                continue
            assert record.after.get(key) == was, (
                f"record {record.json_doc_id!r}: {key!r} changed without an operation"
            )


def ids_are_distinct_and_existing(
    result: ExtractionResult, existing_ids: Collection[str]
) -> None:
    """Updates and deletes name existing records, each at most once."""
    named = [r.json_doc_id for r in result.records if r.kind in ("update", "delete")]
    assert all(n in existing_ids for n in named), f"unknown record in {named}"
    assert len(named) == len(set(named)), f"a record appears twice in {named}"


def only_kinds(result: ExtractionResult, allowed: Iterable[str]) -> None:
    """Every applied operation is one of ``allowed`` (for example no ``remove``)."""
    kinds = set(allowed)
    for record in updates(result):
        bad = [p for p in record.patches if p.op not in kinds]
        assert not bad, f"record {record.json_doc_id!r}: operations {bad}"


def no_op_inside(
    result: ExtractionResult, record_id: str, pointers: Sequence[str]
) -> None:
    """No operation of ``record_id`` touches anything at or below ``pointers``."""
    for record in updates(result):
        if record.json_doc_id != record_id:
            continue
        for p in record.patches:
            for pointer in pointers:
                assert not _touches(p.path, pointer), (
                    f"record {record_id!r}: {p.op} {p.path} touches {pointer}"
                )


def spans_quote_the_input(result: ExtractionResult, text: str) -> None:
    """Every span lies inside the input; an inserted text value is its span, verbatim.

    (An update's field holds the value the patched record has, which can differ from
    what the span quotes: a short form of the name was read and not applied.)
    """
    for record in result.records:
        for answer in record.fields.values():
            spans = list(answer.spans or ())
            if answer.span is not None:
                spans.append(answer.span)
            for start, end in spans:
                assert 0 <= start < end <= len(text), (
                    f"{answer.pretty}: span {(start, end)} is outside the input"
                )
            if (
                record.kind == "insert"
                and isinstance(answer.value, str)
                and answer.span is not None
                and answer.engine == "clairic"
            ):
                quoted = text[answer.span[0] : answer.span[1]]
                assert folded(answer.value) == folded(quoted), (
                    f"{answer.pretty}: {answer.value!r} is not its span {quoted!r}"
                )
