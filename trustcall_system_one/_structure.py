"""Block and sentence structure of an input, for the TypeSafe (Jev) stand-in.

It is recovered the way TypeSafe's structure recovery cookbook does it.

Free text (a ``text`` field, a list of notes) is read as a choice among *blocks*,
*sentences* and *clauses* that code found and the model refers to by id. The steps:

1. **Lines** (code): the input is split into non-blank lines, each with an id
   (``L014``), its exact offsets and whether a blank line precedes it.
2. **Stitching** (one request of yes/no questions): for each pair of adjacent lines
   not separated by a blank line, "does line ``L015`` pick up mid-sentence,
   continuing a sentence left unfinished at the end of line ``L014``?". A join
   probability of at least 0.2 merges the pair after a *dangling* line (no sentence
   ending punctuation) and of at least 0.5 after terminal punctuation, the
   cookbook's thresholds.
3. **Blocks and sentences** (code): the merged lines are blocks (``B003``); a block
   is split into sentences (``S007``) by the library's splitter, which works on a
   block now that its hard-wrapped lines are joined; the sentences are split into
   clauses and noun phrases (``S007.1``, ``S007.p1``).

Every id maps to an exact span of the original input, so a choice of an id is a
span, with no text for the model to copy.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from trustcall_system_one._candidates import (
    Candidate,
    sentence_spans,
    structure_pieces,
)

__all__ = [
    "Block",
    "Line",
    "Structure",
    "build_structure",
    "ends_terminal",
    "join_questions",
    "line_id",
    "to_lines",
]

JOIN_AFTER_DANGLING = 0.2
JOIN_AFTER_TERMINAL = 0.5
_DESCRIPTION_CHARS = 60
_MAX_PIECES_PER_SENTENCE = 12  # a row of a table is a sentence of its cells


def line_id(index: int) -> str:
    """The id of a line: ``L000``."""
    return f"L{index:03d}"


@dataclass(frozen=True)
class Line:
    """A non-blank line of the input.

    Attributes:
        id: ``L000``, ``L001``, ... in order.
        text: The line without surrounding whitespace.
        start: Offset of its first character in the input.
        end: Offset after its last character.
        gap: Whether a blank line comes between it and the line before.
    """

    id: str
    text: str
    start: int
    end: int
    gap: bool


def to_lines(text: str) -> list[Line]:
    """Split ``text`` into non-blank lines with offsets (a leading blank is no gap)."""
    lines: list[Line] = []
    gap = False
    position = 0
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped:
            start = position + len(raw) - len(raw.lstrip())
            lines.append(
                Line(line_id(len(lines)), stripped, start, start + len(stripped), gap)
            )
            gap = False
        elif lines:
            gap = True
        position += len(raw) + 1
    return lines


def ends_terminal(text: str) -> bool:
    """Whether a line ends with sentence or clause ending punctuation."""
    return re.search(r"[.!?:;…][\"')\]]*$", text) is not None


def join_questions(lines: Sequence[Line]) -> dict[str, dict[str, object]]:
    """The stitch questions: one yes/no per pair of adjacent lines with no gap.

    The ids are the line ids of the second line of the pair. The wording is the
    cookbook's: it asks for the narrowest fact ("picks up mid-sentence"), not whether
    the lines share a paragraph, which merges every unmarked list.
    """
    questions: dict[str, dict[str, object]] = {}
    for i in range(1, len(lines)):
        if lines[i].gap:
            continue
        questions[lines[i].id] = {
            "type": "noul",
            "instructions": (
                f"Does line {lines[i].id} pick up mid-sentence, continuing a "
                f"sentence left unfinished at the end of line {lines[i - 1].id}?"
            ),
            "criteria": {
                "true": (
                    "The line starts in the middle of a sentence that began on the "
                    "previous line - the line break tore the sentence apart"
                ),
                "false": (
                    "The line begins a new sentence, item, heading, or thought of "
                    "its own"
                ),
            },
        }
    return questions


def tag_lines(lines: Sequence[Line]) -> str:
    """The lines as the stitch state: ``L014| text``, blank where there is a gap."""
    return "\n".join(
        f"{chr(10) if line.gap else ''}{line.id}| {line.text}" for line in lines
    )


@dataclass(frozen=True)
class Block:
    """Lines merged into one block, with its sentences.

    Attributes:
        id: ``B000``, ``B001``, ...
        lines: The lines it holds.
        joined: The lines' text joined with one space.
        gap: Whether a blank line precedes the block.
        sentences: ``(id, start in joined, end in joined)`` of each sentence.
    """

    id: str
    lines: tuple[Line, ...]
    joined: str
    gap: bool
    sentences: tuple[tuple[str, int, int], ...] = ()

    @property
    def start(self) -> int:
        """Offset of the block's first character in the input."""
        return self.lines[0].start

    @property
    def end(self) -> int:
        """Offset after the block's last character in the input."""
        return self.lines[-1].end

    def original(self, start: int, end: int) -> tuple[int, int]:
        """Map ``joined[start:end]`` to offsets in the input."""
        cursor = 0
        first: int | None = None
        last = self.lines[0].start
        for line in self.lines:
            line_end = cursor + len(line.text)
            if first is None and start < line_end + 1:
                first = line.start + max(0, start - cursor)
            if end > cursor:
                last = line.start + min(len(line.text), end - cursor)
            cursor = line_end + 1
        return (first if first is not None else self.start, last)


@dataclass(frozen=True)
class Structure:
    """The blocks of an input and the candidates they offer.

    Attributes:
        text: The input.
        blocks: The blocks in order.
        candidates: Blocks, sentences, clauses and noun phrases as candidates whose
            ``id`` is the block, sentence or piece id and whose span is exact.
        by_id: The same candidates by id.
        stitched: How many line breaks were healed.
    """

    text: str
    blocks: tuple[Block, ...]
    candidates: tuple[Candidate, ...] = ()
    by_id: Mapping[str, Candidate] = field(default_factory=dict)
    stitched: int = 0

    def tagged(self) -> str:
        """The blocks as a state: ``B003| [S007] sentence [S008] sentence``."""
        out = []
        for block in self.blocks:
            body = " ".join(
                f"[{sid}] {block.joined[s:e]}" for sid, s, e in block.sentences
            )
            out.append(f"{chr(10) if block.gap else ''}{block.id}| {body}")
        return "\n".join(out)


def _short(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _DESCRIPTION_CHARS:
        return text
    return text[: _DESCRIPTION_CHARS - 1].rstrip() + "…"


def merge_lines(
    lines: Sequence[Line],
    joins: Mapping[str, float],
    *,
    after_dangling: float = JOIN_AFTER_DANGLING,
    after_terminal: float = JOIN_AFTER_TERMINAL,
) -> list[list[Line]]:
    """Merge lines into blocks with the cookbook's two thresholds.

    Args:
        lines: The lines.
        joins: P(the line continues the previous one) by line id.
        after_dangling: Cutoff when the previous line has no ending punctuation.
        after_terminal: Cutoff when it ends with ``.``, ``!``, ``?``, ``:`` or ``;``.

    Returns:
        The lines grouped into blocks.
    """
    groups: list[list[Line]] = []
    for i, line in enumerate(lines):
        bar = (
            after_terminal if i and ends_terminal(lines[i - 1].text) else after_dangling
        )
        if groups and not line.gap and joins.get(line.id, 0.0) >= bar:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _build_block(index: int, group: Sequence[Line], first_sentence: int) -> Block:
    joined = " ".join(line.text for line in group)
    sentences = tuple(
        (f"S{first_sentence + k:03d}", s, e)
        for k, (s, e) in enumerate(sentence_spans(joined))
    )
    if not sentences:
        sentences = ((f"S{first_sentence:03d}", 0, len(joined)),)
    return Block(f"B{index:03d}", tuple(group), joined, group[0].gap, sentences)


def build_structure(
    text: str,
    lines: Sequence[Line],
    joins: Mapping[str, float],
    *,
    after_dangling: float = JOIN_AFTER_DANGLING,
    after_terminal: float = JOIN_AFTER_TERMINAL,
) -> Structure:
    """Blocks, sentences and their candidates from the lines and the join answers.

    Args:
        text: The input.
        lines: Its lines (``to_lines``).
        joins: The stitch answers, P(join) by line id (a missing id counts as 0).
        after_dangling: See ``merge_lines``.
        after_terminal: See ``merge_lines``.

    Returns:
        The structure, with a candidate for each block, sentence, clause and phrase.
    """
    groups = merge_lines(
        lines, joins, after_dangling=after_dangling, after_terminal=after_terminal
    )
    blocks: list[Block] = []
    count = 0
    for index, group in enumerate(groups):
        block = _build_block(index, group, count)
        count += len(block.sentences)
        blocks.append(block)
    candidates: list[Candidate] = []
    for block in blocks:
        start, end = block.start, block.end
        candidates.append(
            Candidate(
                block.joined,
                start,
                end,
                block.joined,
                "block",
                None,
                ((start, end),),
                (),
                block.id,
                f"block of {len(block.sentences)} sentence(s): "
                f"'{_short(block.joined)}'",
            )
        )
        for sid, s, e in block.sentences:
            piece = block.joined[s:e]
            a, b = block.original(s, e)
            candidates.append(
                Candidate(
                    piece,
                    a,
                    b,
                    piece,
                    "sentence",
                    None,
                    ((a, b),),
                    (),
                    sid,
                    f"sentence: '{_short(piece)}'",
                )
            )
            candidates.extend(_pieces(block, sid, s, e))
    return Structure(
        text,
        tuple(blocks),
        tuple(candidates),
        {c.id: c for c in candidates},
        len(lines) - len(blocks),
    )


def _pieces(block: Block, sid: str, s: int, e: int) -> list[Candidate]:
    """Clauses and noun phrases inside one sentence, as ``S007.1`` and ``S007.p1``."""
    sentence = block.joined[s:e]
    found = structure_pieces(sentence)
    out: list[Candidate] = []
    seen = {sentence}
    for kind, tag in (("clause", ""), ("phrase", "p")):
        number = 0
        for piece in found[kind]:
            if piece.text in seen or number >= _MAX_PIECES_PER_SENTENCE:
                continue
            seen.add(piece.text)
            number += 1
            a, b = block.original(s + piece.start, s + piece.end)
            out.append(
                replace(
                    piece,
                    start=a,
                    end=b,
                    occurrences=((a, b),),
                    contexts=(),
                    id=f"{sid}.{tag}{number}",
                    description=f"part of {sid}: '{_short(piece.text)}'",
                    value=piece.text,
                )
            )
    return out


def structure_key(text: str) -> str:
    """A stable key of an input (for remembering its stitching)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
