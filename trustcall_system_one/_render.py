"""Model-facing state: Clairic text with sentinel blocks, Clef structured state.

Clairic reads one string. It is laid out as blocks with sentinel tags so that the
engine can tell where the user's text is::

    <focus entity="the Person">...</focus>
    <input>
    ...the user's text, verbatim...
    </input>
    <existing>...</existing>        (updates only)

Clairic answers text fields with spans (offsets into the state it was sent).
``RenderedState`` remembers where the input sits so a span can be mapped back to
an offset in the ORIGINAL input, and a span that lands outside the input block
(in ``<focus>`` or ``<existing>``) can be recognised.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ELLIPSIS",
    "RenderedState",
    "clef_state",
    "estimate_tokens",
    "fit_clef_state",
    "mention_windows",
    "narrow_text",
    "reanchor",
    "render_clairic",
]

ELLIPSIS = "…"
_INPUT_OPEN = "<input>\n"
_INPUT_CLOSE = "\n</input>"
_SEPARATOR = f"\n{ELLIPSIS}\n"
_CHARS_PER_TOKEN = 3.5

Span = tuple[int, int]
Segment = tuple[int, int, int]


@dataclass(frozen=True)
class RenderedState:
    """A Clairic state string and where the user's input sits inside it.

    Attributes:
        text: The state sent to Clairic.
        input_span: ``(start, end)`` of the input block's content in ``text``.
        existing_span: ``(start, end)`` of the ``<existing>`` content, if any.
        segments: Verbatim pieces of the input inside the input block, as
            ``(start in text, end in text, offset in the original input)``. A
            full render has one piece; a narrowed render has one per window.
    """

    text: str
    input_span: Span
    existing_span: Span | None = None
    segments: tuple[Segment, ...] = ()

    @property
    def input_offset(self) -> int:
        """Where the input block's content starts in ``text``.

        For a full render a span in the state minus this offset is a span in the
        original input.
        """
        return self.input_span[0]

    def to_original(self, span: Span) -> Span | None:
        """Map a span in ``text`` to a span in the original input.

        Args:
            span: ``(start, end)`` offsets into ``text``.

        Returns:
            The span in the original input, or ``None`` when it does not lie
            inside one verbatim piece of the input (it touches ``<focus>``,
            ``<existing>``, a separator or the sentinels).
        """
        start, end = span
        for seg_start, seg_end, original in self.segments:
            if seg_start <= start <= end <= seg_end:
                return (start - seg_start + original, end - seg_start + original)
        return None

    def in_existing(self, span: Span) -> bool:
        """Whether a span in ``text`` lies inside the ``<existing>`` block."""
        if self.existing_span is None:
            return False
        return self.existing_span[0] <= span[0] and span[1] <= self.existing_span[1]


def _attribute(value: str) -> str:
    return " ".join(value.replace('"', "'").split())


def mention_windows(
    length: int, spans: Sequence[Span], window_chars: int
) -> list[Span]:
    """Return merged, ordered windows of context around spans.

    Args:
        length: Length of the input text.
        spans: ``(start, end)`` spans in the input.
        window_chars: Characters kept on each side of a span.

    Returns:
        Disjoint ``(start, end)`` ranges in increasing order. Empty when there
        are no spans.
    """
    ranges = sorted(
        (max(0, start - window_chars), min(length, end + window_chars))
        for start, end in spans
    )
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def narrow_text(text: str, windows: Sequence[Span]) -> str:
    """Join the windows of ``text`` with ``…`` separators (for Clef states)."""
    if not windows:
        return text
    body = _SEPARATOR.join(text[start:end] for start, end in windows)
    lead = f"{ELLIPSIS}\n" if windows[0][0] > 0 else ""
    trail = f"\n{ELLIPSIS}" if windows[-1][1] < len(text) else ""
    return f"{lead}{body}{trail}"


def render_clairic(
    text: str,
    *,
    label: str,
    mentions: Sequence[str] = (),
    existing: str | None = None,
    windows: Sequence[Span] | None = None,
) -> RenderedState:
    """Lay out the Clairic state for one entity.

    Args:
        text: The user's input, verbatim.
        label: What to call the entity (``"the Person"``).
        mentions: How the input refers to the entity, when known.
        existing: The entity's existing record, rendered as text (updates).
        windows: Ranges of ``text`` to keep instead of all of it; they must be
            disjoint and in increasing order (see ``mention_windows``).

    Returns:
        The state and the map from state offsets back to ``text``.
    """
    focus = [f'<focus entity="{_attribute(label)}">', f"Answer only about {label}."]
    if mentions:
        quoted = ", ".join(f'"{_attribute(m)}"' for m in mentions)
        focus.append(f"The input mentions it as: {quoted}.")
    focus.append("</focus>\n")
    head = "\n".join(focus)

    pieces: list[tuple[str, int]]
    if windows:
        pieces = [(text[start:end], start) for start, end in windows]
    else:
        pieces = [(text, 0)]
    lead = ELLIPSIS + "\n" if windows and windows[0][0] > 0 else ""
    trail = "\n" + ELLIPSIS if windows and windows[-1][1] < len(text) else ""

    out = head + _INPUT_OPEN
    cursor = len(out) + len(lead)
    segments: list[Segment] = []
    block = lead
    for index, (piece, original) in enumerate(pieces):
        if index:
            block += _SEPARATOR
            cursor += len(_SEPARATOR)
        segments.append((cursor, cursor + len(piece), original))
        block += piece
        cursor += len(piece)
    block += trail
    out += block
    input_start = len(head) + len(_INPUT_OPEN)
    input_span = (input_start, len(out))
    out += _INPUT_CLOSE
    existing_span: Span | None = None
    if existing is not None:
        out += "\n<existing>\n"
        existing_span = (len(out), len(out) + len(existing))
        out += existing + "\n</existing>"
    return RenderedState(
        text=out,
        input_span=input_span,
        existing_span=existing_span,
        segments=tuple(segments),
    )


def reanchor(sent: RenderedState, state_text: str | None) -> RenderedState | None:
    """Re-locate the input in the ``state_text`` Clairic says it read.

    Clairic returns the state it actually used. When that is exactly what was
    sent nothing changes. Otherwise each verbatim piece of the input is searched
    for after the ``<input>`` sentinel.

    Args:
        sent: The state that was sent.
        state_text: ``state_text`` from the response, or ``None`` when absent
            (taken to mean the state was not changed).

    Returns:
        The state to map spans with, or ``None`` when the input cannot be found
        in ``state_text`` (spans are then unknown).
    """
    if state_text is None or state_text == sent.text:
        return sent
    cursor = max(0, state_text.find("<input>"))
    segments: list[Segment] = []
    for start, end, original in sent.segments:
        piece = sent.text[start:end]
        found = state_text.find(piece, cursor)
        if found < 0:
            return None
        segments.append((found, found + len(piece), original))
        cursor = found + len(piece)
    if not segments:
        return None
    existing_span: Span | None = None
    if sent.existing_span is not None:
        existing = sent.text[slice(*sent.existing_span)]
        found = state_text.find(existing, cursor)
        if found >= 0:
            existing_span = (found, found + len(existing))
    return RenderedState(
        text=state_text,
        input_span=(segments[0][0], segments[-1][1]),
        existing_span=existing_span,
        segments=tuple(segments),
    )


# --------------------------------------------------------------------------- #
# Clef
# --------------------------------------------------------------------------- #


def clef_state(
    text: str,
    *,
    entity: Mapping[str, Any] | None = None,
    record: Any = None,
    proposed_change: Any = None,
) -> dict[str, Any]:
    """Build the structured state of a Clef request.

    Instructions refer to the parts by backticked path (``input``,
    ``entity.label``, ``record``, ``proposed_change``).

    Args:
        text: The user's input.
        entity: What the question is about, for example ``{"label": ...,
            "mentions": [...]}``.
        record: An existing record (updates).
        proposed_change: A change under review (updates).

    Returns:
        ``{"input": text}`` plus the keys that were given.
    """
    state: dict[str, Any] = {"input": text}
    if entity is not None:
        state["entity"] = dict(entity)
    if record is not None:
        state["record"] = record
    if proposed_change is not None:
        state["proposed_change"] = proposed_change
    return state


def estimate_tokens(state: Any) -> float:
    """Estimate the token count of a Clef state as ``len(json) / 3.5``."""
    return len(json.dumps(state)) / _CHARS_PER_TOKEN


def fit_clef_state(
    state: dict[str, Any],
    budget: int,
    *,
    spans: Sequence[Span] | None = None,
    window_chars: int = 600,
) -> tuple[dict[str, Any], str | None]:
    """Shrink a Clef state that is too large for the model.

    Clef truncates its state silently, so a state estimated above ``budget``
    tokens is narrowed here: ``input`` is cut down to windows around ``spans``
    (the window halves until the state fits) or, with no spans to anchor on, to
    its beginning.

    Args:
        state: The state, with the text under ``"input"``.
        budget: Largest acceptable estimate, in tokens.
        spans: Spans in the input worth keeping.
        window_chars: Starting size of the windows.

    Returns:
        ``(state, warning)``. The state is returned unchanged with no warning
        when it already fits.
    """
    if estimate_tokens(state) <= budget:
        return state, None
    text = state["input"]
    if spans:
        size = window_chars
        while True:
            windows = mention_windows(len(text), spans, size)
            narrowed = {**state, "input": narrow_text(text, windows)}
            if estimate_tokens(narrowed) <= budget or size <= 50:
                break
            size //= 2
        return narrowed, (
            f"clef_state_narrowed: the input was cut to windows of {size} "
            f"characters around {len(spans)} mention(s) to fit the Clef budget"
        )
    limit = len(text)
    narrowed = state
    while estimate_tokens(narrowed) > budget and limit > 100:
        limit = int(limit * 0.8)
        narrowed = {**state, "input": text[:limit]}
    return narrowed, (
        f"clef_state_truncated: no mentions to anchor on, so the input was cut "
        f"to its first {limit} characters to fit the Clef budget"
    )
