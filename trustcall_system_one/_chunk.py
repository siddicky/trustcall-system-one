"""Chunked Clairic asks: split a long input, then merge the per-chunk answers.

Clairic reports ``truncated: true`` when the state does not fit its context. The
engine then asks the same fields again on consecutive, overlapping chunks that
together cover the whole input, and merges the answers per field. Spans are
converted to offsets in the ORIGINAL input before merging, so a merged answer is
mapped with an identity state.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from trustcall_system_one._render import RenderedState
from trustcall_system_one._wire import ClairicAnswer

__all__ = [
    "DEFAULT_CHUNK_CHARS",
    "Part",
    "chunk_size",
    "identity_state",
    "merge_parts",
    "part_of",
    "split_ranges",
]

Span = tuple[int, int]

DEFAULT_CHUNK_CHARS = 8_000
"""Chunk size used when the truncated response does not say how much it read."""
_SAFETY = 0.8  # chunks are this fraction of what Clairic was seen to read


def split_ranges(ranges: Sequence[Span], size: int, overlap: int) -> list[Span]:
    """Cut ranges of the input into consecutive overlapping chunks.

    Args:
        ranges: Disjoint ``(start, end)`` ranges to cover (the whole input as
            ``[(0, len)]``, or the windows of a narrowed ask).
        size: Largest chunk, in characters.
        overlap: Characters shared by neighbouring chunks of one range, so that a
            mention shorter than this is whole in at least one chunk.

    Returns:
        The chunks in increasing order. A range no longer than ``size`` is one
        chunk.
    """
    step = max(1, size - overlap)
    chunks: list[Span] = []
    for first, last in ranges:
        start = first
        while True:
            end = min(last, start + size)
            chunks.append((start, end))
            if end >= last:
                break
            start += step
    return chunks


def chunk_size(read_chars: int | None, covered: int, overlap: int) -> int:
    """Pick a chunk size for a truncated ask.

    Args:
        read_chars: Characters of input the truncated response was seen to read,
            or ``None`` when that is unknown.
        covered: Characters the truncated request covered.
        overlap: The chunk overlap.

    Returns:
        A size below ``covered`` (so there are at least two chunks) whenever the
        range is long enough to split, and above ``2 * overlap``.
    """
    base = DEFAULT_CHUNK_CHARS if read_chars is None else int(read_chars * _SAFETY)
    return max(2 * overlap + 1, min(base, int(covered * 0.6)))


def identity_state(text: str) -> RenderedState:
    """A state whose offsets already are offsets in ``text`` (for merged answers)."""
    return RenderedState(
        text=text, input_span=(0, len(text)), segments=((0, len(text), 0),)
    )


@dataclass(frozen=True)
class Part:
    """One chunk's answer for a field, with spans in the original input.

    Attributes:
        answer: The raw answer (its own spans are in chunk coordinates).
        span: The span, mapped to the original input.
        spans: Per-element spans, mapped (``None`` where unmappable).
        start: Where the chunk starts in the original input.
    """

    answer: ClairicAnswer
    span: Span | None
    spans: tuple[Span | None, ...] | None
    start: int


def _map(anchored: RenderedState | None, pair: Sequence[int]) -> Span | None:
    if anchored is None or len(pair) != 2:
        return None
    return anchored.to_original((pair[0], pair[1]))


def part_of(answer: ClairicAnswer, anchored: RenderedState | None, start: int) -> Part:
    """Map an answer's spans from chunk coordinates to the original input."""
    spans = (
        tuple(_map(anchored, pair) for pair in answer.spans)
        if answer.spans is not None
        else None
    )
    return Part(
        answer, _map(anchored, answer.span) if answer.span else None, spans, start
    )


def _norm(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _merge_lists(parts: list[Part]) -> ClairicAnswer:
    elements: list[tuple[object, Span | None, int]] = []
    for part in parts:
        values = part.answer.value
        for index, value in enumerate(values):
            span = (
                part.spans[index]
                if part.spans is not None and index < len(part.spans)
                else None
            )
            duplicate = any(
                (
                    span is not None
                    and kept_span is not None
                    and _overlap(span, kept_span)
                )
                or ((span is None or kept_span is None) and _norm(value) == _norm(kept))
                for kept, kept_span, _ in elements
            )
            if not duplicate:
                elements.append((value, span, part.start))
    elements.sort(key=lambda item: item[1][0] if item[1] is not None else item[2])
    spans = [span for _, span, _ in elements]
    best = min(
        parts, key=lambda p: p.answer.p_null if p.answer.p_null is not None else 1.0
    )
    confidences = [
        p.answer.confidence for p in parts if p.answer.confidence is not None
    ]
    return best.answer.model_copy(
        update={
            "value": [value for value, _, _ in elements],
            "spans": (
                [[s[0], s[1]] for s in spans if s is not None]
                if all(s is not None for s in spans) and spans
                else None
            ),
            "span": None,
            "confidence": min(confidences) if confidences else None,
            "p_null": best.answer.p_null,
        }
    )


def merge_parts(parts: list[Part]) -> ClairicAnswer | None:
    """Merge one field's answers from several chunks into a single answer.

    Non-null beats null; among non-null scalars the higher confidence wins (the
    earlier chunk on a tie); lists are united in input order, dropping an element
    whose span overlaps one already kept (or, without spans, whose text matches).
    The result carries spans in original-input offsets, so map it with
    ``identity_state``.

    Args:
        parts: The field's per-chunk answers, in chunk order.

    Returns:
        The merged answer, or ``None`` when there were no answers.
    """
    if not parts:
        return None
    present = [p for p in parts if p.answer.value is not None]
    if not present:  # every chunk says null: report the surest one
        best = max(
            parts,
            key=lambda p: p.answer.p_null if p.answer.p_null is not None else 0.0,
        )
        return best.answer.model_copy(update={"span": None, "spans": None})
    if all(isinstance(p.answer.value, list) for p in present):
        return _merge_lists(present)
    winner = max(
        enumerate(present),
        key=lambda item: (
            item[1].answer.confidence if item[1].answer.confidence is not None else 0.0,
            -item[0],
        ),
    )[1]
    span = winner.span
    return winner.answer.model_copy(
        update={
            "span": [span[0], span[1]] if span is not None else None,
            "spans": None,
        }
    )
