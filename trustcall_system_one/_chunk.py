"""Chunked Clairic asks: split a long input, then merge the per-chunk answers.

Clairic reports ``truncated: true`` when the state does not fit its context. The
engine then asks the same fields again on consecutive, overlapping chunks that
together cover the whole input, and merges the answers per field. Spans are
converted to offsets in the ORIGINAL input before merging, so a merged answer is
mapped with an identity state.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from trustcall_system_one._render import RenderedState
from trustcall_system_one._wire import ClairicAnswer

__all__ = [
    "DEFAULT_CHUNK_CHARS",
    "Merged",
    "Part",
    "agree",
    "chunk_size",
    "identity_state",
    "merge_field",
    "merge_parts",
    "part_of",
    "split_ranges",
]

Span = tuple[int, int]

DEFAULT_CHUNK_CHARS = 8_000
"""Chunk size used when the truncated response does not say how much it read."""
_SAFETY = 0.8  # chunks are this fraction of what Clairic was seen to read
_AGREE_RATIO = 0.9
_DIGIT_SHARE = 0.3  # at this share of digits a string is compared exactly


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

    Raises:
        ValueError: If ``size`` is not positive or ``overlap`` is negative or
            not smaller than ``size`` (the chunks could not make progress).
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(
            f"chunk overlap must be at least 0 and smaller than the size "
            f"({size}), got {overlap}"
        )
    step = size - overlap
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _digit_heavy(text: str) -> bool:
    """Whether a string is mostly digits (a date, a code, an amount)."""
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(c.isdigit() for c in chars) / len(chars) >= _DIGIT_SHARE


def agree(first: Any, second: Any, *, fuzzy: bool = True) -> bool:
    """Whether two answers for one field say the same thing.

    Text agrees when it is equal after case and whitespace folding; with
    ``fuzzy`` it also agrees at a ``difflib.SequenceMatcher`` ratio of at least
    0.9, unless either string is mostly digits (``2020-01-01`` and
    ``2020-01-02``, or ISBNs that differ in one digit, are different values).
    Numbers compare by value; lists agree when they have the same size and every
    element has a distinct agreeing partner; anything else must be equal.

    Args:
        first: One answer.
        second: The other.
        fuzzy: Allow the text ratio. Pass ``False`` for dates, times, durations,
            numbers and codes, which compare exactly after normalisation.
    """
    if first is None or second is None:
        return first is None and second is None
    if isinstance(first, str) and isinstance(second, str):
        a, b = _norm(first), _norm(second)
        if a == b:
            return True
        if not fuzzy or _digit_heavy(a) or _digit_heavy(b):
            return False
        return SequenceMatcher(None, a, b).ratio() >= _AGREE_RATIO
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return False
        remaining = list(second)
        for item in first:
            for index, other in enumerate(remaining):
                if agree(item, other, fuzzy=fuzzy):
                    del remaining[index]
                    break
            else:
                return False
        return True
    if _is_number(first) and _is_number(second):
        return Decimal(str(first)) == Decimal(str(second))
    return bool(first == second)


def _overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _confidence(part: Part) -> float:
    confidence = part.answer.confidence
    return confidence if confidence is not None else 0.0


@dataclass
class _Element:
    value: object
    span: Span | None
    start: int
    confidence: float
    part: int = 0

    def replaces(self, other: "_Element") -> bool:
        """Whether this element replaces the one it overlaps.

        It must contain it, and be longer or (for the same span) more confident.
        An element without a span is replaced by any span with the same text.
        """
        assert self.span is not None
        if other.span is None:
            return True
        contains = self.span[0] <= other.span[0] and other.span[1] <= self.span[1]
        mine = self.span[1] - self.span[0]
        theirs = other.span[1] - other.span[0]
        return contains and (mine, self.confidence) > (theirs, other.confidence)


def _merge_lists(parts: list[Part]) -> ClairicAnswer:
    elements: list[_Element] = []
    for number, part in enumerate(parts):
        for index, value in enumerate(part.answer.value):
            span = (
                part.spans[index]
                if part.spans is not None and index < len(part.spans)
                else None
            )
            new = _Element(value, span, part.start, _confidence(part), number)

            def same(
                kept: _Element, value: object = value, number: int = number
            ) -> bool:
                # an element without a span can only be told from another by its
                # text, and only across chunks (the overlap of two windows reads
                # it twice); within one answer a repeated text is a repeated
                # element, not a duplicate
                return kept.part != number and _norm(kept.value) == _norm(value)

            if span is None:
                if not any(kept.span is None and same(kept) for kept in elements):
                    elements.append(new)
                continue
            clashes = [
                kept
                for kept in elements
                if (kept.span is not None and _overlap(span, kept.span))
                or (kept.span is None and same(kept))
            ]
            if not clashes:
                elements.append(new)
            elif len(clashes) == 1 and new.replaces(clashes[0]):
                # one element that the new span contains (or equals, with more
                # confidence): the fuller reading wins. A span that bridges two
                # kept elements replaces neither.
                elements[elements.index(clashes[0])] = new
    elements.sort(
        key=lambda item: item.span[0] if item.span is not None else item.start
    )
    spans = [item.span for item in elements]
    best = min(
        parts, key=lambda p: p.answer.p_null if p.answer.p_null is not None else 1.0
    )
    confidences = [
        p.answer.confidence for p in parts if p.answer.confidence is not None
    ]
    return best.answer.model_copy(
        update={
            "value": [item.value for item in elements],
            # an element whose span is unknown keeps its place with an empty pair,
            # so the others keep theirs
            "spans": (
                [[s[0], s[1]] if s is not None else [] for s in spans]
                if any(s is not None for s in spans)
                else None
            ),
            "span": None,
            "confidence": min(confidences) if confidences else None,
            "p_null": best.answer.p_null,
        }
    )


def settled(part: Part) -> ClairicAnswer:
    """Return one chunk's answer with its span in original-input offsets."""
    span = part.span
    return part.answer.model_copy(
        update={
            "span": [span[0], span[1]] if span is not None else None,
            "spans": None,
        }
    )


@dataclass(frozen=True)
class Merged:
    """The outcome of merging one field's chunk answers.

    Attributes:
        answer: The merged answer, with spans in original-input offsets
            (``None`` when there were no answers). When the chunks disagree it
            is the most confident candidate: what to use when nothing can
            adjudicate.
        candidates: The distinct non-null scalar candidates, most confident
            first (the earlier chunk on a tie). More than one means the chunks
            disagree and the caller should decide between them.
    """

    answer: ClairicAnswer | None
    candidates: tuple[Part, ...] = ()

    @property
    def conflict(self) -> bool:
        """Whether the chunks gave different non-null scalar values."""
        return len(self.candidates) > 1


def merge_field(parts: list[Part]) -> Merged:
    """Merge one field's answers from several chunks.

    Every non-null scalar is kept as a candidate. Candidates that agree (equal
    after case and whitespace folding, a text ratio of at least 0.9, or equal
    numbers) are one value, represented by the most confident of them. Lists
    are united in input order; when spans of different chunks overlap the longer
    span is kept (the more confident chunk on a tie). A null in one chunk never
    hides a value from another; when every chunk says null the answer is the one
    with the LOWEST ``p_null``, so a borderline null can still be checked.

    Args:
        parts: The field's per-chunk answers, in chunk order.

    Returns:
        The merged answer and, for disagreeing scalars, the candidates.
    """
    if not parts:
        return Merged(None)
    present = [p for p in parts if p.answer.value is not None]
    if not present:
        best = min(
            parts,
            key=lambda p: (
                p.answer.p_null is None,
                p.answer.p_null if p.answer.p_null is not None else 0.0,
            ),
        )
        return Merged(best.answer.model_copy(update={"span": None, "spans": None}))
    if all(isinstance(p.answer.value, list) for p in present):
        return Merged(_merge_lists(present))
    ranked = sorted(
        enumerate(present), key=lambda item: (-_confidence(item[1]), item[0])
    )
    fuzzy = present[0].answer.type == "text"  # dates, numbers, ... compare exactly
    distinct: list[Part] = []
    for _, candidate in ranked:
        if not any(
            agree(candidate.answer.value, kept.answer.value, fuzzy=fuzzy)
            for kept in distinct
        ):
            distinct.append(candidate)
    return Merged(settled(distinct[0]), tuple(distinct))


def merge_parts(parts: list[Part]) -> ClairicAnswer | None:
    """Merge one field's chunk answers without adjudicating disagreements.

    See ``merge_field``; when scalars disagree the most confident one is
    returned.

    Args:
        parts: The field's per-chunk answers, in chunk order.

    Returns:
        The merged answer, or ``None`` when there were no answers.
    """
    return merge_field(parts).answer
