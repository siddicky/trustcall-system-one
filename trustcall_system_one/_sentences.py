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

import asyncio
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from trustcall_system_one._run import gather
from trustcall_system_one._types import Span
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
)

if TYPE_CHECKING:
    from trustcall_system_one._compile import FieldSpec
    from trustcall_system_one._verify import Ctx, Resolved

__all__ = [
    "ListStrategy",
    "Sentence",
    "candidate_sentences",
    "eligible",
    "list_kind",
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
# What can follow a time without starting a new sentence ("at 5 p.m. Monday").
_DAYS_MONTHS = _words(
    "monday tuesday wednesday thursday friday saturday sunday",
    "january february march april may june july august september october november",
    "december",
)
# A modal or a pronoun after a MONTH word starts a sentence ("5 p.m. May I suggest
# ...", "9 a.m. April will be ...") instead of continuing the time; a weekday goes on.
_MODALS = _words("will would can could shall should may might must")
_SUBJECTS = _words("i we you he she it they there")
_MONTHS = _words(
    "january february march april may june july august september october november",
    "december",
)
_CLOSERS = "\"')]}\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
_CLOSERS += "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}"
_OPENERS = "\"'([{\N{LEFT DOUBLE QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}"
_APOSTROPHES = "'\N{RIGHT SINGLE QUOTATION MARK}"
_TERMINATOR = re.compile(rf"[.!?]+[{re.escape(_CLOSERS)}]*")
_WORD = rf"[A-Za-z{_APOSTROPHES}-]"
# A speaker label at the start of a line: "Me: ", "Friend: ", "Dr Wu: ",
# "Speaker 2: " (a short run of words, then a colon). It is only a label when the
# same one starts at least two lines (see ``_speakers``).
_LABEL = re.compile(
    rf"^(?P<label>[A-Z]{_WORD}*(?: {_WORD}+){{0,2}}|[A-Z]{_WORD}* \d{{1,2}}):"
    r"[ \t]+(?=\S)"
)
_CJK_TERMINATOR = re.compile(
    "[\N{IDEOGRAPHIC FULL STOP}\uff01\uff1f]+[\u300d\u300f\uff09]*"
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
        speaker: In a transcript, the label of the line the sentence is on
            ("Sarah" for "Sarah: I got promoted."); the label is not part of
            ``text``.
        label_span: Where that label is in the input.
    """

    text: str
    start: int
    end: int
    speaker: str | None = None
    label_span: Span | None = None

    @property
    def span(self) -> Span:
        """``(start, end)`` in the input."""
        return (self.start, self.end)


def _is_initial(line: str, start: int, rest: str) -> bool:
    """Whether the single capital at ``start`` is an initial ("J. R. R. Tolkien").

    It is one when a capitalised word follows and the word before it is a
    capitalised word or another initial (or there is none: the start of the line).
    "So did I. We left." and "She got a B. Then" are sentence ends.
    """
    if not rest[:1].isupper():
        return False
    before = line[:start].rstrip()
    if not before:
        return True
    previous = before.split()[-1].lstrip("\"'([{" + _OPENERS)
    return previous[:1].isupper()


def _is_abbreviation(line: str, dot: int, rest: str) -> bool:
    """Whether the period at ``dot`` ends an abbreviation or an initial."""
    found = _WORD_BEFORE.search(line[:dot])
    if found is None:
        return False
    word = found.group(1)
    before = found.start(1) - 1
    if before >= 0 and line[before].isdigit():
        return False  # the M of "$1.5M": a unit, not an initial
    bare = word.replace(".", "").lower()
    if word.lower() in ("a.m", "p.m"):
        # "at 5 p.m. Then" ends the sentence; "at 5 p.m. Monday" does not: the
        # time goes on with a day or a month
        following = re.match(r"([A-Za-z]+)(?:\W+([A-Za-z0-9]+))?", rest)
        if following is None or following.group(1).lower() not in _DAYS_MONTHS:
            return False
        word = following.group(1).lower()
        after = (following.group(2) or "").lower()
        if word not in _MONTHS:
            return True  # a weekday goes on: "5 p.m. Friday this week"
        if word in _MODALS and after and not after[:1].isdigit():
            return False  # "May I suggest ...": the modal, not the month
        return after not in _MODALS and after not in _SUBJECTS
    if len(bare) == 1 and "." not in word:
        return word.isupper() and _is_initial(line, found.start(1), rest)
    if bare in _ABBREVIATIONS or (rest[:1].isdigit() and bare in _BEFORE_NUMBER):
        return True
    # "U.S", "e.g": single letters separated by dots
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
            line, match.start(), rest
        ):
            continue
        ends.append(end)
    # CJK terminators need no space after them
    for match in _CJK_TERMINATOR.finditer(line):
        if line[match.end() :].strip():
            ends.append(match.end())
    return sorted(set(ends))


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


def _speakers(text: str) -> set[str]:
    """The labels that start at least two lines: the speakers of a transcript."""
    seen: dict[str, int] = {}
    for line in re.finditer(r"[^\r\n]+", text):
        body = _BULLET.sub("", line.group(), count=1)
        found = _LABEL.match(body)
        if found is not None:
            label = found.group("label")
            seen[label] = seen.get(label, 0) + 1
    return {label for label, n in seen.items() if n >= 2}


def split_sentences(text: str) -> list[Sentence]:
    """Split ``text`` into sentences and clauses, keeping offsets.

    A boundary is a ``.``, ``!`` or ``?`` (with any closing quote or bracket)
    followed by whitespace and a capital letter, digit, quote or bracket, or a
    CJK full stop, exclamation or question mark. It is not a boundary after an
    abbreviation ("Dr.", "Mrs.", "Inc.", "St.", "e.g.", "i.e.", "U.S."), after an
    initial (a single capital in front of a capitalised word, after a capitalised
    word or another initial: "J. R. R. Tolkien"), inside a number ("3.5") or an
    address ("example.com"), or before a lower-case word. "a.m./p.m." end a
    sentence in front of a capital. A line break always ends a sentence, a leading
    bullet or list number is not part of the sentence, and ``;`` ends a clause.
    Pieces with no letter or digit are dropped.

    A transcript is text where at least two lines start with the same label
    ("Me: ...", "Sarah: ..."): there the label is not part of the sentence (it is
    kept in ``Sentence.speaker`` and ``label_span``). A label that starts a single
    line ("Her email: a@b.co", "Plan A: go home") is text like any other.

    Known limits: text written entirely in lower case has no capital to start a
    sentence and stays one sentence; and "Inc.", "St." and the like keep their
    sentence together even when a capital follows ("We met on Main St. Then ...").

    Args:
        text: The input.

    Returns:
        The sentences in order. ``text[s.start:s.end] == s.text`` for each.
    """
    out: list[Sentence] = []
    speakers = _speakers(text)
    for line in re.finditer(r"[^\r\n]+", text):
        body, origin = line.group(), line.start()
        found = _BULLET.match(body)
        if found is not None:
            body, origin = body[found.end() :], origin + found.end()
        speaker: str | None = None
        label_span: Span | None = None
        found = _LABEL.match(body)
        if found is not None and found.group("label") in speakers:
            speaker = found.group("label")
            label_span = (origin, origin + len(speaker))
            body, origin = body[found.end() :], origin + found.end()
        for piece, start in _clauses(body, origin):
            stripped = piece.strip()
            if not any(ch.isalnum() for ch in stripped):
                continue
            stripped = stripped.rstrip(";").rstrip()
            lead = len(piece) - len(piece.lstrip())
            at = start + lead
            out.append(Sentence(stripped, at, at + len(stripped), speaker, label_span))
    return out


# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #


def _overlaps(spans: Sequence[Span], start: int, end: int) -> bool:
    return any(a < end and start < b for a, b in spans)


def _word_in(text: str, needle: str) -> bool:
    return bool(
        re.search(rf"(?<!\w){re.escape(needle.casefold())}(?!\w)", text.casefold())
    )


def candidate_sentences(
    sentences: Sequence[Sentence],
    *,
    spans: Sequence[Span] = (),
    texts: Sequence[str] = (),
    follow: int = 4,
    others: Sequence[Span] = (),
) -> list[Sentence]:
    """The sentences worth asking about for one entity.

    Without mentions (a ``One``) every sentence is a candidate. For an entity
    that is mentioned, a candidate is a sentence that mentions it or one of the
    next ``follow`` sentences after such a sentence ("Sarah is doing well. She is
    teaching special education."). In a transcript the speaker's own lines count
    as mentioning her (the label is not part of the sentence).

    A mention is found by its span when it has one (``spans``); ``texts`` are the
    mentions that have none, matched as whole words. That keeps two people called
    "Emma" apart. The reach of a mention stops at a sentence that names another
    entity of the same collection (``others``: their mention spans), so Emma
    Thompson's sentences do not run on into Emma Watson's.

    Args:
        sentences: The input's sentences.
        spans: Where the input mentions the entity.
        texts: Mentions of the entity that have no span.
        follow: How many sentences after a mention still count.
        others: Where the input mentions the other entities of the collection
            (those of the entity itself may be included: they are ignored).

    Returns:
        The candidates, in input order.
    """
    if not spans and not any(t.strip() for t in texts):
        return list(sentences)
    foreign = [o for o in others if o not in set(spans)]

    def mentions(s: Sentence) -> bool:
        if _overlaps(spans, s.start, s.end):
            return True
        if s.label_span is not None and _overlaps(spans, *s.label_span):
            return True
        if s.speaker is not None and any(
            t.strip() and s.speaker.casefold() == t.strip().casefold() for t in texts
        ):
            return True
        return any(t.strip() and _word_in(s.text, t) for t in texts)

    def names_another(s: Sentence) -> bool:
        if _overlaps(foreign, s.start, s.end):
            return True
        return s.label_span is not None and _overlaps(foreign, *s.label_span)

    keep: set[int] = set()
    for index, sentence in enumerate(sentences):
        if not mentions(sentence):
            continue
        keep.add(index)
        for j in range(index + 1, min(len(sentences), index + follow + 1)):
            if mentions(sentences[j]):
                keep.add(j)
            elif names_another(sentences[j]):
                break
            else:
                keep.add(j)
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


def _tokens(value: Any) -> float:
    """Tokens of a Clef state or question set, with a floor for non-ASCII text.

    ``estimate_json_tokens`` is the calibrated estimate the budget is checked
    with; a language written in CJK or emoji still costs about a token per
    character, so that is the least the estimate may say.
    """
    from trustcall_system_one._render import estimate_json_tokens

    wide = sum(1 for ch in json.dumps(value, ensure_ascii=False) if ord(ch) > 0x2E7F)
    return max(estimate_json_tokens(value), float(wide))


async def read_sentences(ctx: "Ctx", spec: "FieldSpec") -> "Resolved":
    """Read a ``list[str]`` field one sentence at a time.

    Splits the input, keeps the sentences that concern the entity, asks Clef one
    ``noul`` question per candidate (call kind ``sentences``; more than a request
    takes (64, or 32 on Experiential Labs) are split over several requests by the run)
    and keeps the sentences at or above ``verify.sentence_keep`` (and the field's
    ``min_confidence``, per element).
    The tokens of the questions count against the Clef state budget: the input is
    narrowed to windows around the candidates by as much as they need.

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
    texts = [m.text for m in ctx.info.mentions if m.span is None] if ctx.info else []
    if ctx.info is None:
        texts = list(ctx.mentions)
    own = set(spans)
    candidates = candidate_sentences(
        sentences,
        spans=spans,
        texts=texts,
        follow=limits.sentence_follow,
        others=[o for o in ctx.others if o not in own],
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
    # The questions of a request share the model's window with the state, and the
    # sum must fit the budget. Each request is given its own state: the input
    # narrowed to windows around the sentences ITS questions quote (the question
    # holds the sentence, so the state only has to place it) into whatever the
    # questions leave. When even that does not fit, the chunk is halved.
    budget = ctx.run.clef_budget()
    ids = list(questions)
    span_of = {f"sentence.{i}": s.span for i, s in enumerate(candidates)}

    def cost(chunk: Sequence[str]) -> float:
        return _tokens({q: questions[q].model_dump() for q in chunk})

    def state_for(chunk: Sequence[str]) -> dict[str, Any]:
        room = max(500, int(budget - cost(chunk)))
        return ctx.fitted(spans=[span_of[q] for q in chunk], budget=room)

    per_question = max(cost([q]) for q in ids)
    size = max(1, min(ctx.run.clef_batch(), int(budget / 2 / per_question)))
    while True:
        chunks = [ids[i : i + size] for i in range(0, len(ids), size)]
        states = [state_for(c) for c in chunks]
        fits = all(
            _tokens(st) + cost(c) <= budget
            for st, c in zip(states, chunks, strict=True)
        )
        if fits or size == 1:
            break
        size = max(1, size // 2)
    if not fits:
        ctx.warn(
            f"{spec.pretty}: a sentence question and the narrowest state do not fit "
            f"the Clef budget of {budget} tokens together; the request may be "
            "truncated (limits.clef_state_token_budget)"
        )
    meta = ctx.meta("sentences", field=spec.wire_name)
    parts = await gather(
        *(
            ctx.run.call_clef({q: questions[q] for q in chunk}, st, meta=meta)
            for chunk, st in zip(chunks, states, strict=True)
        )
    )
    answers: dict[str, Any] = {}
    for part in parts:
        answers.update(part)
    floor = spec.min_confidence or 0.0
    scored: list[tuple[Sentence, float | None]] = []
    for index, sentence in enumerate(candidates):
        answer = answers.get(f"sentence.{index}")
        scored.append(
            (sentence, answer.noul if isinstance(answer, NoulAnswer) else None)
        )
    kept = [
        (s, p)
        for s, p in scored
        if p is not None and p >= verify.sentence_keep and p >= floor
    ]
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


ITEMS_VETO = 0.75
"""Probability of ``items`` at or above which a list is NOT read by sentences.

Only a clear ``items`` keeps a list of names out of the sentence path. The two
mistakes are not equal: a wrong ``items`` drops the field (its Clairic answer was
already distrusted), while a wrong ``statements`` only sends candidate sentences
through the per-sentence question, the attribution check, the novelty check and the
change gate, which refuse what is not stated. Clef answers this question sharply
(0.95 to 0.97 for ``statements`` on a notes field), but a reader that hedges does not:
live, Decisions gave ``items`` 0.65 to 0.67 for a notes field every time (15 of 15),
which a plain ``statements >= min_choice_prob`` rule turned into "no notes, ever". It
applies only to a hedging reader (the Decisions API or a hard pick, see
``_hedged_veto``): a graded answer of Cloudflare's Clef keeps the plain rule."""


async def _ask_kind(ctx: "Ctx", spec: "FieldSpec") -> str:
    description = " ".join(spec.description.split()) or spec.pretty
    question = ChoiceQuestion(
        type="choice",
        instructions=(
            f"Are the items of `field.name` ({description}) short named values "
            "(names, languages, tags, ids) or free-form statements and facts?"
        ),
        criteria={
            "items": "Short named values: names, languages, tags, ids, places.",
            "statements": "Free-form statements or facts, written as clauses or "
            "whole sentences.",
        },
    )
    state = {"field": {"name": spec.pretty, "description": description}}
    answers = await ctx.run.call_clef(
        {"kind": question}, state, meta=ctx.meta("list_kind", field=spec.wire_name)
    )
    answer = answers.get("kind")
    if not isinstance(answer, ChoiceAnswer):
        return "items"  # no answer: the field keeps its low-confidence rejection
    p_statements = answer.probabilities.get("statements", 0.0)
    if answer.choice == "statements" and p_statements >= ctx.run.gates.min_choice_prob:
        return "statements"
    veto = _hedged_veto(ctx, spec, answer)
    if veto is not None and answer.probabilities.get("items", 0.0) < veto:
        return "statements"  # a hedging reader's 0.65 for ``items`` is not clear
    return "items"


def _hedged_veto(ctx: "Ctx", spec: "FieldSpec", answer: ChoiceAnswer) -> float | None:
    """The ``items`` probability that vetoes the sentence path, for a hedging reader.

    Only an answer that is a hard pick (no probabilities, its confidence is the cap
    of the transport) or that the Decisions API gave hedges this way. A graded answer
    of any other provider keeps the plain rule (``statements`` chosen with at least
    ``gates.min_choice_prob``): there a list of phone numbers that Clef calls items
    at 0.70 must not be turned into sentences. ``ITEMS_VETO``, clamped to the cap of
    a hard pick (``DecisionsTransport(decisions_hard_cap=...)``), so that a hard
    ``items`` always vetoes.

    Returns:
        The veto probability, or ``None`` when the answer does not hedge.
    """
    from trustcall_system_one._run import _is_hard

    if _is_hard(answer):
        cap = answer.confidence
        return min(ITEMS_VETO, cap) if cap is not None and cap > 0 else ITEMS_VETO
    for call in reversed(ctx.run.calls):
        if call.kind == "list_kind" and call.field == spec.wire_name:
            return ITEMS_VETO if call.provider == "decisions" else None
    return None


async def list_kind(ctx: "Ctx", spec: "FieldSpec") -> str:
    """Whether a ``list[str]`` field holds short named values or free-form statements.

    One Clef ``choice`` (call kind ``list_kind``), asked once per schema and field
    in a run however many entities have the field (the schema is told apart by its
    class, not its name): the answer is kept on the run (concurrent askers share one
    request). Only statements are worth reading by
    sentences; a list of names or languages that Clairic is unsure of is not turned
    into sentences. ``"statements"`` needs ``statements`` chosen with at least
    ``gates.min_choice_prob``; for a hedging reader (the Decisions API, a hard pick)
    ``items`` below ``ITEMS_VETO`` (0.75, at most the hard cap) is enough: only a
    clear ``items`` keeps the list out of the sentence path (a hedging reader's 0.65
    for ``items`` is not clear). A missing answer is ``"items"``.

    Args:
        ctx: The entity context.
        spec: A list-of-text field.

    Returns:
        ``"items"`` or ``"statements"``.
    """
    key = (ctx.plan.schema, spec.wire_name)  # the class itself: names can collide
    pending = ctx.run.list_kinds.get(key)
    if pending is None:
        pending = ctx.run.list_kinds[key] = asyncio.ensure_future(_ask_kind(ctx, spec))
    return await asyncio.shield(pending)
