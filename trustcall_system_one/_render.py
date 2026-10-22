"""Model-facing state: Clairic text with sentinel blocks, Clef structured state.

Clairic reads one string. It is laid out as blocks with sentinel tags so that the
engine can tell where the user's text is::

    <focus entity="the Person">...</focus>
    <input>
    ...the user's text, verbatim...
    </input>
    <existing id="..." schema="...">...</existing>        (updates only)

Clairic answers text fields with spans (offsets into the state it was sent).
``RenderedState`` remembers where the input sits so a span can be mapped back to
an offset in the ORIGINAL input, and a span that lands outside the input block
(in ``<focus>`` or ``<existing>``) can be recognised.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ELLIPSIS",
    "Focus",
    "Parent",
    "RenderedState",
    "clef_state",
    "discovery_focus",
    "entity_focus",
    "estimate_tokens",
    "fit_clef_state",
    "mention_windows",
    "narrow_text",
    "narrowed_spans",
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
Focus = tuple[str, tuple[str, ...]]
"""A focus header: the opening tag and the lines that follow it."""


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

    @property
    def input_chars(self) -> int:
        """How many characters of the original input this state contains."""
        return sum(end - start for start, end, _ in self.segments)

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


def narrowed_spans(
    length: int, windows: Sequence[Span], spans: Iterable[Span]
) -> list[Span]:
    """Where spans of a text fall in ``narrow_text(text, windows)``.

    Args:
        length: Length of the whole text.
        windows: The windows ``narrow_text`` is given (disjoint, increasing).
        spans: ``(start, end)`` spans in the whole text.

    Returns:
        The spans that lie inside one window, re-based into the narrowed text, in
        the order given; the others are left out.
    """
    if not windows:
        return list(spans)
    offset = len(f"{ELLIPSIS}\n") if windows[0][0] > 0 else 0
    starts: list[tuple[int, int, int]] = []
    for start, end in windows:
        starts.append((start, end, offset))
        offset += (end - start) + len(_SEPARATOR)
    found: list[Span] = []
    for first, last in spans:
        for start, end, base in starts:
            if start <= first and last <= end:
                found.append((base + first - start, base + last - start))
                break
    return found


Parent = tuple[str, Sequence[Span | None]]
"""The parent of a nested entity for a focus header: its label and mention spans."""


def _cited(spans: Sequence[Span | None]) -> str:
    return ", ".join(f"{span[0]}-{span[1]}" for span in spans if span)


def entity_focus(
    schema: str,
    kind: str,
    label: str,
    mentions: Sequence[tuple[str, Span | None]],
    parent: Parent | None = None,
) -> Focus:
    """The focus header of one entity of a collection.

    Args:
        schema: Name of the schema (``"Person"``).
        kind: What the entity is, for "any other ...": the schema title.
        label: The entity's label (``"Emma Thompson"``).
        mentions: Its mentions as ``(text, span)``, spans in the original input.
        parent: For an item of a nested list, the label of the entity it belongs to
            and where the input mentions that entity.

    Returns:
        The opening tag (``<focus schema="Person" entity="Emma Thompson">``, the
        one place the label is written; ``parent="..."`` names the owner of a
        nested item the same way) and the lines that say whom to answer for and,
        when the offsets are known, where the input mentions the entity (never
        the mention texts themselves).
    """
    header = f'<focus schema="{_attribute(schema)}" entity="{_attribute(label)}"'
    if parent is not None:
        header += f' parent="{_attribute(parent[0])}"'
    header += ">"
    lines = [
        "Answer each field for the entity named above only. "
        f"Ignore facts about any other {kind.lower()}."
    ]
    if parent is not None:
        lines.append(
            f"The entity is a {kind.lower()} of the parent named above: ignore any "
            f"{kind.lower()} that belongs to someone else."
        )
    # Offsets only: the header must not quote the mention texts. Clairic answers
    # with a span, and a quoted name in the header is a place a span can start at
    # (a live answer began in the quote and ended in the input, so its value held
    # the header's tail and the input tag).
    cited = _cited([span for _, span in mentions])
    if cited:
        lines.append(f"The entity is mentioned in the input at characters {cited}.")
    if parent is not None and (where := _cited(parent[1])):
        lines.append(f"The parent is mentioned in the input at characters {where}.")
    return header, tuple(lines)


def discovery_focus(schema: str, kind: str, parent: Parent | None = None) -> Focus:
    """The focus header of the discovery and count requests of a collection.

    Args:
        schema: Name of the schema.
        kind: What an entity is called.
        parent: For a nested list, the label of the entity the items belong to and
            where the input mentions it.
    """
    header = f'<focus schema="{_attribute(schema)}" task="discover"'
    if parent is not None:
        header += f' parent="{_attribute(parent[0])}"'
    header += ">"
    if parent is None:
        return header, (
            f"Look for every {kind.lower()} the input mentions, whoever they are. "
            "Answer from the input text only.",
        )
    lines = [
        f"Look for every {kind.lower()} of the parent named above that the input "
        f"mentions, and ignore any {kind.lower()} that belongs to someone else. "
        "Answer from the input text only."
    ]
    if where := _cited(parent[1]):
        lines.append(f"The parent is mentioned in the input at characters {where}.")
    return header, tuple(lines)


def render_clairic(
    text: str,
    *,
    label: str,
    mentions: Sequence[str] = (),
    existing: str | None = None,
    existing_id: str | None = None,
    existing_schema: str | None = None,
    windows: Sequence[Span] | None = None,
    trailer: str = "",
    focus: Focus | None = None,
) -> RenderedState:
    """Lay out the Clairic state for one entity.

    Args:
        text: The user's input, verbatim.
        label: What to call the entity (``"the Person"``).
        mentions: How the input refers to the entity, when known.
        existing: The entity's existing record, rendered as text (updates). It
            always follows the input block whole: only the input is ever cut.
        existing_id: The record's id, written as the ``id`` attribute of the
            ``<existing>`` tag.
        existing_schema: The record's schema name (the ``schema`` attribute).
        windows: Ranges of ``text`` to keep instead of all of it; they must be
            disjoint and in increasing order (see ``mention_windows``).
        focus: A header to use instead of the default one (``entity_focus``,
            ``discovery_focus``): the opening tag and its lines. ``label`` and
            ``mentions`` are then not used for the header.
        trailer: Further blocks written after everything else (the repair
            loop's ``<previous_attempt>``, ``<validation_errors>`` and
            ``<expected_schema>``). They are context only: a span that lands in
            them is outside the input, like one in ``<existing>``.

    Returns:
        The state and the map from state offsets back to ``text``.
    """
    if focus is not None:
        lines = [focus[0], *focus[1]]
    else:
        lines = [f'<focus entity="{_attribute(label)}">', f"Answer only about {label}."]
        if mentions:
            quoted = ", ".join(f'"{_attribute(m)}"' for m in mentions)
            lines.append(f"The input mentions it as: {quoted}.")
    if existing is not None:
        lines.append(
            "The existing record after the input is context only: answer from "
            "the input text, never from the existing record."
        )
    if trailer:
        lines.append(
            "The blocks after the input describe an earlier answer that was "
            "rejected. They are context only: answer from the input text, never "
            "from them."
        )
    lines.append("</focus>\n")
    head = "\n".join(lines)

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
        attrs = ""
        if existing_id is not None:
            attrs += f' id="{_attribute(existing_id)}"'
        if existing_schema is not None:
            attrs += f' schema="{_attribute(existing_schema)}"'
        out += f"\n<existing{attrs}>\n"
        existing_span = (len(out), len(out) + len(existing))
        out += existing + "\n</existing>"
    if trailer:
        out += "\n" + trailer
    return RenderedState(
        text=out,
        input_span=input_span,
        existing_span=existing_span,
        segments=tuple(segments),
    )


def _input_start(state_text: str, cursor: int) -> int:
    """Where the input content begins in a returned ``state_text``."""
    marker = state_text.find(_INPUT_OPEN, cursor)
    return cursor if marker < 0 else marker + len(_INPUT_OPEN)


def _common_prefix(piece: str, state_text: str, start: int) -> tuple[int, int] | None:
    """Return ``(start, length)`` of the longest prefix of ``piece`` at ``start``."""
    limit = min(len(piece), len(state_text) - start)
    length = 0
    while length < limit and piece[length] == state_text[start + length]:
        length += 1
    # A few characters in common can be chance; a real prefix has more.
    return (start, length) if length >= min(8, len(piece)) and length else None


def reanchor(sent: RenderedState, state_text: str | None) -> RenderedState | None:
    """Re-locate the input in the ``state_text`` Clairic says it read.

    Clairic returns the state it actually used. When that is exactly what was
    sent nothing changes. Otherwise each verbatim piece of the input is searched
    for after the ``<input>`` sentinel. If Clairic read only the beginning of the
    input (a truncated state), ``state_text`` holds a prefix of the input block:
    that prefix is anchored, so spans inside it still map, while spans beyond it
    do not (``RenderedState.to_original`` returns ``None`` for them).

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
    previous_end = sent.input_span[0]
    for start, end, original in sent.segments:
        piece = sent.text[start:end]
        found = state_text.find(piece, cursor)
        if found < 0:
            # Clairic may have read only the beginning of a long input: keep the
            # part of this piece that is there, and nothing after it. It starts
            # right after whatever stood between it and the previous piece (the
            # input block's opening, or a separator).
            between = sent.text[previous_end:start]
            where = cursor if segments else _input_start(state_text, cursor)
            if state_text.startswith(between, where):
                prefix = _common_prefix(piece, state_text, where + len(between))
                if prefix is not None:
                    segments.append((prefix[0], prefix[0] + prefix[1], original))
            break
        segments.append((found, found + len(piece), original))
        cursor = found + len(piece)
        previous_end = end
    if not segments:
        return None
    existing_span: Span | None = None
    if sent.existing_span is not None:
        existing = sent.text[slice(*sent.existing_span)]
        found = state_text.find(existing, segments[-1][1])
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
    extra: Mapping[str, Any] | None = None,
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
        extra: Further keys of the state (the mentions of a coreference
            question, the two sides of a merge check).

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
    if extra:
        state.update(extra)
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
