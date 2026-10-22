"""Sentence-based reading of open-ended ``list[str]`` fields.

Clairic's ``list`` type is a span picker. It is confident for defined items
(names, contacts: 0.9995 live) and hopeless for open-ended ones ("things worth
remembering": fragments at confidence 1e-6 to 1e-23). For those fields the input
is split into sentences in code, Clef is asked one yes/no question per candidate
sentence ("is this sentence one of the <field> about <entity>?"), and every
sentence at or above ``verify.sentence_keep`` becomes a verbatim list element with
its span. The elements then go through the usual ladder (attribution), the
dedup-against-the-record check and the change gate.

This module holds the pure parts (splitting, scoping) and the Clef pass.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from trustcall_system_one._types import Span
from trustcall_system_one._wire import NoulAnswer, NoulCriteria, NoulQuestion

if TYPE_CHECKING:
    from trustcall_system_one._compile import FieldSpec
    from trustcall_system_one._verify import Ctx, Resolved

__all__ = [
    "ListStrategy",
    "Sentence",
    "candidate_sentences",
    "eligible",
    "read_sentences",
    "split_sentences",
]

ListStrategy = Literal["list", "sentences", "auto"]
"""How an open-ended ``list[str]`` field is read (see ``s1_field``)."""

STRATEGIES: tuple[str, ...] = ("list", "sentences", "auto")

# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def _words(*groups: str) -> frozenset[str]:
    return frozenset(word for group in groups for word in group.split())


# Words that end in a period without ending the sentence. Compared lower-cased and
# without internal dots ("e.g" for "e.g."). Words that are also ordinary words
# ("no", "am", "sat", "may") are deliberately absent.
_ABBREVIATIONS = _words(
    "mr mrs ms mx dr prof sr jr st mt ft rev hon gen col lt sgt capt cmdr",
    "cpl pvt gov sen rep pres supt insp fr br vs cf viz approx est dept univ",
    "assn assoc inc ltd co corp bros llc plc ave blvd rd hwy ste apt bldg",
    "eg ie ph phd md mba esq",
)
# Abbreviations that only count in front of a number: "No. 5", "Fig. 2", "Oct. 5".
_BEFORE_NUMBER = _words(
    "no nos fig figs vol vols pp ch sec art eq eqs",
    "jan feb mar apr jun jul aug sep sept oct nov dec",
)
_CLOSERS = "\"')]}\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
_CLOSERS += "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}"
_OPENERS = "\"'([{\N{LEFT DOUBLE QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}"
_APOSTROPHES = "'\N{RIGHT SINGLE QUOTATION MARK}"
_TERMINATOR = re.compile(rf"[.!?]+[{re.escape(_CLOSERS)}]*")
_WORD = rf"[A-Za-z{_APOSTROPHES}-]"
# "Me: ", "Friend: ", "Dr Wu: ", "Speaker 2: " (a short run of words, then a colon)
_LABEL = re.compile(
    rf"^[A-Z]{_WORD}*(?: {_WORD}+){{0,2}}:[ \t]+(?=\S)"
    rf"|^[A-Z]{_WORD}* \d{{1,2}}:[ \t]+(?=\S)"
)
_BULLET = re.compile(r"^(?:[-*]|\N{BULLET}|\(?\d{1,3}[.)])[ \t]+(?=\S)")
_WORD_BEFORE = re.compile(r"([A-Za-z](?:[A-Za-z.]*[A-Za-z])?)$")


@dataclass(frozen=True)
class Sentence:
    """A sentence (or clause) of the input.

    Attributes:
        text: The sentence exactly as written (``text[start:end]`` of the input).
        start: Offset of its first character in the input.
        end: Offset after its last character.
    """

    text: str
    start: int
    end: int

    @property
    def span(self) -> Span:
        """``(start, end)`` in the input."""
        return (self.start, self.end)


def _is_abbreviation(line: str, dot: int, number_follows: bool) -> bool:
    """Whether the period at ``dot`` ends an abbreviation or an initial."""
    found = _WORD_BEFORE.search(line[:dot])
    if found is None:
        return False
    word = found.group(1)
    before = found.start(1) - 1
    if before >= 0 and line[before].isdigit():
        return False  # the M of "$1.5M": a unit, not an initial
    bare = word.replace(".", "").lower()
    if len(bare) == 1 and word[0].isalpha():
        return True  # an initial ("J. Smith") or the "e" of "e.g."
    if bare in _ABBREVIATIONS or (number_follows and bare in _BEFORE_NUMBER):
        return True
    # "U.S", "a.m", "e.g": single letters separated by dots
    return "." in word and all(len(part) == 1 for part in word.split("."))


def _boundaries(line: str) -> list[int]:
    """Offsets in ``line`` after which a new sentence starts (end of the previous)."""
    ends: list[int] = []
    for match in _TERMINATOR.finditer(line):
        end = match.end()
        if end < len(line) and not line[end].isspace():
            continue  # "3.5", "example.com", "Dr.Smith": no space, no boundary
        rest = line[end:].lstrip()
        if not rest:
            continue  # the end of the line is handled by the caller
        terminator = match.group()
        starts_sentence = rest[0].isupper() or rest[0].isdigit() or rest[0] in _OPENERS
        if not starts_sentence:
            continue  # "approx. five", "e.g. lowercase"
        if terminator.rstrip(_CLOSERS) == "." and _is_abbreviation(
            line, match.start(), rest[0].isdigit()
        ):
            continue
        ends.append(end)
    return ends


def _clauses(line: str, offset: int) -> Iterable[tuple[str, int]]:
    """Split a line into sentences, then on ``; `` into clauses."""
    cuts = _boundaries(line)
    previous = 0
    pieces: list[tuple[int, int]] = []
    for cut in [*cuts, len(line)]:
        pieces.append((previous, cut))
        previous = cut
    for start, end in pieces:
        piece = line[start:end]
        base = 0
        for part in re.split(r"(?<=;)[ \t]+", piece):
            index = piece.find(part, base)
            base = index + len(part)
            yield part, offset + start + index


def split_sentences(text: str) -> list[Sentence]:
    """Split ``text`` into sentences and clauses, keeping offsets.

    A boundary is a ``.``, ``!`` or ``?`` (with any closing quote or bracket)
    followed by whitespace and a capital letter, digit, quote or bracket. It is
    not a boundary after an abbreviation or an initial ("Dr.", "Mrs.", "J.",
    "e.g.", "i.e.", "U.S.", "a.m.", "Inc."), inside a number ("3.5") or a name
    with dots ("example.com"), or before a lower-case word. A line break always
    ends a sentence, a leading bullet or list number and a leading speaker label
    ("Me: ", "Friend: ") are not part of the sentence, and ``;`` ends a clause.
    Pieces with no letter or digit are dropped.

    Args:
        text: The input.

    Returns:
        The sentences in order. ``text[s.start:s.end] == s.text`` for each.
    """
    out: list[Sentence] = []
    for line in re.finditer(r"[^\r\n]+", text):
        body, origin = line.group(), line.start()
        for pattern in (_BULLET, _LABEL):
            found = pattern.match(body)
            if found is not None:
                body, origin = body[found.end() :], origin + found.end()
        for piece, start in _clauses(body, origin):
            stripped = piece.strip()
            if not any(ch.isalnum() for ch in stripped):
                continue
            stripped = stripped.rstrip(";").rstrip()
            lead = len(piece) - len(piece.lstrip())
            at = start + lead
            out.append(Sentence(stripped, at, at + len(stripped)))
    return out


# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #


def _mentions(sentence: Sentence, spans: Sequence[Span], texts: Sequence[str]) -> bool:
    if any(a < sentence.end and sentence.start < b for a, b in spans):
        return True
    low = sentence.text.casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(t.casefold())}(?!\w)", low)
        for t in texts
        if t.strip()
    )


def candidate_sentences(
    sentences: Sequence[Sentence],
    *,
    spans: Sequence[Span] = (),
    texts: Sequence[str] = (),
    follow: int = 4,
) -> list[Sentence]:
    """The sentences worth asking about for one entity.

    Without mentions (a ``One``) every sentence is a candidate. For an entity
    that is mentioned, a candidate is a sentence that mentions it (a mention span
    that overlaps it, or one of the mention texts as a whole word) or one of the
    next ``follow`` sentences after such a sentence ("Sarah is doing well. She
    is teaching special education.").

    Args:
        sentences: The input's sentences.
        spans: Where the input mentions the entity.
        texts: How the input refers to the entity (matched when it has no span).
        follow: How many sentences after a mention still count.

    Returns:
        The candidates, in input order.
    """
    if not spans and not any(t.strip() for t in texts):
        return list(sentences)
    keep: set[int] = set()
    for index, sentence in enumerate(sentences):
        if _mentions(sentence, spans, texts):
            keep.update(range(index, min(len(sentences), index + follow + 1)))
    return [s for i, s in enumerate(sentences) if i in keep]


# --------------------------------------------------------------------------- #
# The Clef pass
# --------------------------------------------------------------------------- #


def eligible(spec: "FieldSpec") -> bool:
    """Whether a field can be read sentence by sentence: a list of free text."""
    clairic = spec.clairic
    return (
        spec.engine == "clairic"
        and clairic is not None
        and clairic.type == "list"
        and clairic.items == "text"
        and spec.item_kind == "str"
    )


def _described(spec: "FieldSpec") -> str:
    text = " ".join(spec.description.split()).strip(" .!?:;")
    text = re.sub(r"^(?:the|an|a)\s+", "", text, flags=re.IGNORECASE)
    return text or "the things the field asks for"


def _question(sentence: Sentence, described: str) -> NoulQuestion:
    return NoulQuestion(
        type="noul",
        instructions=(
            f"Is this sentence one of the {described}, about `entity.label`: "
            f'"{sentence.text}"?'
        ),
        criteria=NoulCriteria(
            true=f"The sentence is one of the {described} and it is about "
            "`entity.label`.",
            false=f"The sentence is not one of the {described}, or it is about "
            "someone or something else.",
        ),
    )


async def read_sentences(ctx: "Ctx", spec: "FieldSpec") -> "Resolved":
    """Read a ``list[str]`` field one sentence at a time.

    Splits the input, keeps the sentences that concern the entity, asks Clef one
    ``noul`` question per candidate (call kind ``sentences``; more than 64 are
    split over several requests by the run) and keeps the sentences at or above
    ``verify.sentence_keep``.

    Args:
        ctx: The entity context (the input, the entity and its mentions).
        spec: A list-of-text field (see ``eligible``).

    Returns:
        A resolved list answer: the kept sentences verbatim, their spans in the
        input, ``confidence`` the lowest probability among them. No kept sentence
        is a null answer. ``raw`` lists every candidate with its probability.
    """
    from trustcall_system_one._verify import Resolved

    limits, verify = ctx.run.limits, ctx.run.verify
    sentences = split_sentences(ctx.text)
    spans = ctx.mention_spans()
    texts = list(ctx.mentions)
    if ctx.info is not None:
        texts = [m.text for m in ctx.info.mentions] or texts
    candidates = candidate_sentences(
        sentences, spans=spans, texts=texts, follow=limits.sentence_follow
    )
    notes = ["list_strategy:sentences"]
    if len(candidates) > limits.max_sentences:
        ctx.warn(
            f"{spec.pretty}: the input has {len(candidates)} candidate sentences; "
            f"only the first {limits.max_sentences} were asked about "
            "(limits.max_sentences)"
        )
        notes.append(f"sentences_capped:{limits.max_sentences}")
        candidates = candidates[: limits.max_sentences]
    if not candidates:
        return Resolved(value=None, raw=[], notes=(*notes, "no_candidate_sentences"))
    described = _described(spec)
    questions = {
        f"sentence.{i}": _question(s, described) for i, s in enumerate(candidates)
    }
    answers = await ctx.run.call_clef(
        questions,
        ctx.entity_state(anchors=[s.span for s in candidates]),
        meta=ctx.meta("sentences", field=spec.wire_name),
    )
    scored: list[tuple[Sentence, float | None]] = []
    for index, sentence in enumerate(candidates):
        answer = answers.get(f"sentence.{index}")
        scored.append(
            (sentence, answer.noul if isinstance(answer, NoulAnswer) else None)
        )
    kept = [(s, p) for s, p in scored if p is not None and p >= verify.sentence_keep]
    raw = [{"sentence": s.text, "p": p} for s, p in scored]
    notes.append(f"sentences:{len(kept)}/{len(scored)}")
    if not kept:
        return Resolved(value=None, raw=raw, notes=tuple(notes))
    return Resolved(
        value=[s.text for s, _ in kept],
        spans=tuple(s.span for s, _ in kept),
        raw=raw,
        confidence=min(p for _, p in kept),
        notes=tuple(notes),
    )
