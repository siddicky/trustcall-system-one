"""Candidate generation for the TypeSafe (Jev) stand-in for Clairic.

Jev answers closed questions (a choice among options, a yes/no), so Clairic's "read
the value off the text" becomes "pick the value among the ones code found". This
module is the code half for pattern-shaped values, as in TypeSafe's pre-parsed value
extraction cookbook: a regex finds the candidate spans (tuned to over-find), the
model picks one, and code returns the exact text and normalises it.

Every candidate carries

* ``text``: exactly ``input[start:end]`` (so a span is an exact offset),
* ``value``: the normalised value the answer carries (text as it stands, or a number),
* ``reading``: how the value reads, when that is not the text itself (``$1,315.50``
  reads as ``1315.5 USD``),
* ``contexts``: the surroundings of the first occurrences, with the candidate marked.

Dates, times and durations are *not* resolved here: the model reads their parts
(``_dateparts``, the date extraction cookbook). Free text is handled by the block and
sentence structure of ``_structure``. The one derived candidate is a *sum*, offered
only for a number field whose instructions ask for a total.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from trustcall_system_one._wire import ClairicField

__all__ = [
    "MAX_OPTIONS",
    "Candidate",
    "context_snippet",
    "date_mentions",
    "item_type",
    "pattern_candidates",
    "sentence_spans",
    "structure_pieces",
]

MAX_OPTIONS = 254
"""Most candidates one choice can hold: a Jev choice takes 255 options and one is the
``none`` option."""

_MAX_PER_KIND = 4000
"""A guard against pathological input: no generator yields more than this."""


@dataclass(frozen=True)
class Candidate:
    """One value the input could state, with where it is written.

    Attributes:
        text: The words of the input, exactly ``input[start:end]``.
        start: Offset of the first character in the input.
        end: Offset after the last character.
        value: The normalised value the answer carries.
        kind: Which generator found it (``email``, ``name``, ``sentence``, ...).
        reading: How the value reads, when that is not the text itself.
        occurrences: Every place the same text and value occur (first one first;
            the span is the first), capped.
        contexts: A snippet around each of the first occurrences, with the
            candidate between guillemets.
        id: For a block, sentence or clause of the input structure, its id
            (``B003``, ``S007``, ``S007.1``); empty for a pattern-shaped candidate,
            whose own text is its option id.
        description: A short description of such a candidate for the model.
    """

    text: str
    start: int
    end: int
    value: Any
    kind: str
    reading: str | None = None
    occurrences: tuple[tuple[int, int], ...] = field(default=())
    contexts: tuple[str, ...] = field(default=())
    id: str = ""
    description: str | None = None

    @property
    def span(self) -> tuple[int, int]:
        """``(start, end)`` of the first occurrence."""
        return (self.start, self.end)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def context_snippet(text: str, start: int, end: int, width: int = 40) -> str:
    """The surroundings of ``text[start:end]`` with the candidate in guillemets.

    The window is ``width`` characters on each side, widened to the next word edge,
    and marked with an ellipsis where the text goes on. Line breaks stay (the
    structure of a header block says which address is which).

    Args:
        text: The input.
        start: Start of the candidate.
        end: End of the candidate.
        width: Characters of context on each side.

    Returns:
        ``…before «candidate» after…``.
    """
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    while 0 < lo < start and not text[lo - 1].isspace():
        lo += 1
    while end < hi < len(text) and not text[hi].isspace():
        hi -= 1
    head = ("…" if lo > 0 else "") + text[lo:start].lstrip(" \t")
    tail = text[end:hi].rstrip(" \t") + ("…" if hi < len(text) else "")
    return f"{head}«{text[start:end]}»{tail}"


def _words(group: str) -> frozenset[str]:
    return frozenset(group.split())


def _trim(
    text: str, start: int, end: int, trailing: str = ".,;:!?)]}\"'"
) -> tuple[int, int]:
    """Shrink ``[start, end)`` past surrounding spaces and trailing punctuation."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in trailing):
        end -= 1
    return start, end


def _overlaps(spans: Iterable[tuple[int, int]], start: int, end: int) -> bool:
    return any(s < end and start < e for s, e in spans)


def _make(
    text: str,
    start: int,
    end: int,
    value: Any,
    kind: str,
    reading: str | None = None,
) -> Candidate:
    return Candidate(text[start:end], start, end, value, kind, reading)


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #

EMAIL = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
)
URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.IGNORECASE)
PHONE = re.compile(r"(?<![\w@./])(?:\+[ \t]?)?\(?\d[\d \t().\-]{5,}\d(?![\w@])")
_LABELLED = re.compile(
    r"(?<![\w/])(?P<label>[A-Za-z][A-Za-z0-9 _/\-]{0,30}?):[ \t]+"
    r"(?P<value>(?:[^\n,;]|,(?=\d))*[^\s,;])"
)
_QUOTED = (
    re.compile(r"\"([^\"\n]{1,200})\""),
    re.compile(r"“([^”\n]{1,200})”"),
    re.compile(r"(?<!\w)'([^'\n]{2,200})'(?!\w)"),
)
_IDENTIFIER = re.compile(
    r"(?<![\w@./\-])[A-Za-z]{0,8}[-_#]?\d+(?:[-_./][A-Za-z0-9]+)*(?![\w@])"
)

_NAME_STOP = _words(
    "The A An This That These Those I We He She They It My Our Your His Her Their "
    "Its Hi Hello Hey Dear Thanks Thank Please And But Or So If When While Also "
    "Then Yes No Not Is Are Was Were Be Do Does Did Have Has Had Will Would Can "
    "Could Should May Might Must Let Here There What Which Who Whom Whose Why How "
    "Ok Okay Sure Well Now Today Tomorrow Yesterday Regards Best Cheers Sincerely "
    "Cc To From Subject Re Fwd Mr Mrs Ms Dr Mon Tue Wed Thu Fri Sat Sun "
    # the verb that opens an instruction ("Call Dana", "Please contact")
    "Call Contact Reply Send Sent See Note Notes Attn Forward Review Check "
    "Confirm Remember Ask Tell Get Give Take Make Meet Join Visit Email Phone "
    "Fax Address Total Date Due Kindly Greetings Attached Enclosed "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February "
    "March April June July August September October November December"
)
_PARTICLES = _words(
    "van von de del della der den di da dos das du la le bin ibn al el ap ben of"
)
_NAME_TOKEN = r"[A-Z][\w'\u2019\-]*\.?"
_PARTICLE_ALTERNATIVES = "|".join(sorted(_PARTICLES))
_NAME = re.compile(
    rf"{_NAME_TOKEN}(?:[ \t]+(?:(?:{_PARTICLE_ALTERNATIVES})[ \t]+)?{_NAME_TOKEN})*"
)
_DETERMINERS = _words(
    "a an the my our her his their your its this that these those some every each"
)
_NP_STOP = _words(
    "and or but nor so yet who whom whose that which where when while because "
    "with without in on at for to of from by is are was were be been being am has "
    "have had do does did will would can could should may might must not no as "
    "than then if also just only very there here about into onto over under "
    "after before during since until upon via per"
)
_WORD = re.compile(r"[A-Za-z][\w'\u2019\-]*")
_CLAUSE_SPLIT = re.compile(
    r"[,;:\u2014\u2013]|\s[-\u2013\u2014]\s|"
    r"\s(?:and|or|but|while|because|so|then|which|who|that)\s",
    re.IGNORECASE,
)


def _emails(text: str) -> list[Candidate]:
    out = []
    for m in EMAIL.finditer(text):
        start, end = _trim(text, m.start(), m.end(), ".,;:!?)]}\"'")
        out.append(_make(text, start, end, text[start:end], "email"))
    return out


def _urls(text: str) -> list[Candidate]:
    out = []
    for m in URL.finditer(text):
        start, end = _trim(text, m.start(), m.end())
        out.append(_make(text, start, end, text[start:end], "url"))
    return out


def phone_spans(text: str) -> list[tuple[int, int]]:
    """Spans that look like a telephone number (7 to 15 digits, not a date)."""
    spans = []
    for m in PHONE.finditer(text):
        digits = sum(ch.isdigit() for ch in m.group())
        if not 7 <= digits <= 15:
            continue
        if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", m.group().strip()):
            continue
        start, end = _trim(text, m.start(), m.end(), ".-)")
        spans.append((start, end))
    return spans


def _phones(text: str) -> list[Candidate]:
    return [_make(text, s, e, text[s:e], "phone") for s, e in phone_spans(text)]


def _labelled(text: str) -> list[Candidate]:
    out = []
    for m in _LABELLED.finditer(text):
        start, end = m.span("value")
        start, end = _trim(text, start, end, ".")
        if start < end and not text[start:end].startswith("//"):
            out.append(_make(text, start, end, text[start:end], "labelled"))
    return out


def _quoted(text: str) -> list[Candidate]:
    out = []
    for pattern in _QUOTED:
        for m in pattern.finditer(text):
            start, end = _trim(text, *m.span(1), trailing="")
            if start < end:
                out.append(_make(text, start, end, text[start:end], "quoted"))
    return out


def _identifiers(text: str) -> list[Candidate]:
    return [
        _make(text, m.start(), m.end(), m.group(), "identifier")
        for m in _IDENTIFIER.finditer(text)
        if re.search(r"[A-Za-z]", m.group())  # plain numbers are number candidates
    ]


def _names(text: str) -> list[Candidate]:
    out: list[Candidate] = []
    for m in _NAME.finditer(text):
        tokens = [
            (t.start() + m.start(), t.end() + m.start())
            for t in re.finditer(r"\S+", m.group())
        ]
        # drop leading and trailing stop words ("Hi Dana", "Dana Thanks")
        while tokens and text[tokens[0][0] : tokens[0][1]].rstrip(".") in _NAME_STOP:
            tokens.pop(0)
        while tokens and text[tokens[-1][0] : tokens[-1][1]].rstrip(".") in _NAME_STOP:
            tokens.pop()
        if not tokens:
            continue
        words: list[tuple[int, int]] = []
        for s, e in tokens:
            e = _strip_name_end(text, s, e)
            if s < e:
                words.append((s, e))
        if not words:
            continue
        start, end = words[0][0], words[-1][1]
        out.append(_make(text, start, end, text[start:end], "name"))
        if 2 <= len(words) <= 4:  # the parts alone: "Emma", "Thompson"
            for s, e in (words[0], words[-1]):
                if text[s:e] not in _NAME_STOP and len(text[s:e]) > 1:
                    out.append(_make(text, s, e, text[s:e], "name"))
    return out


def _strip_name_end(text: str, start: int, end: int) -> int:
    word = text[start:end]
    for suffix in ("'s", "\u2019s"):
        if word.endswith(suffix):
            return end - len(suffix)
    if word.endswith(".") and not re.fullmatch(r"[A-Z]\.", word):
        return end - 1
    return end


_PIVOTS = _words("as is are was were became named called at in from for with to of on")


def _run(text: str, end: int) -> int:
    """End of the run of up to four plain words that starts after ``end``."""
    count = 0
    last_end = end
    for w in re.finditer(r"[ \t]+(" + _WORD.pattern + r")", text[end:]):
        if w.group(1).lower() in _NP_STOP or count >= 4 or w.start() != last_end - end:
            break
        count += 1
        last_end = end + w.end()
    return last_end if count else end


def _noun_phrases(text: str) -> list[Candidate]:
    """Noun phrases after a determiner or after a linking word.

    "a senior engineer" (with and without the determiner), "as finance lead",
    "is Head of Ops".
    """
    out: list[Candidate] = []
    for m in _WORD.finditer(text):
        word = m.group().lower()
        if word in _DETERMINERS:
            last_end = _run(text, m.end())
            if last_end == m.end():
                continue
            out.append(
                _make(text, m.start(), last_end, text[m.start() : last_end], "phrase")
            )
            first = re.search(r"[ \t]+", text[m.end() :])
            if first is not None:
                s = m.end() + first.end()
                out.append(_make(text, s, last_end, text[s:last_end], "phrase"))
        elif word in _PIVOTS:
            start = re.match(r"[ \t]+", text[m.end() :])
            nxt = _WORD.match(text, m.end() + (start.end() if start else 0))
            if start is None or nxt is None or nxt.group().lower() in _DETERMINERS:
                continue
            if nxt.group().lower() in _NP_STOP:
                continue
            last_end = _run(text, nxt.end())
            out.append(
                _make(
                    text, nxt.start(), last_end, text[nxt.start() : last_end], "phrase"
                )
            )
    return out


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    # Imported here: ``_sentences`` imports the run context, which imports the
    # transports, which import this module.
    from trustcall_system_one._sentences import split_sentences

    spans = [(s.start, s.end) for s in split_sentences(text)]
    if not spans and text.strip():
        spans = [_trim(text, 0, len(text), "")]
    return spans


def _clauses(text: str) -> list[Candidate]:
    out: list[Candidate] = []
    for s_start, s_end in _sentence_spans(text):
        sentence = text[s_start:s_end]
        cursor = 0
        pieces: list[tuple[int, int]] = []
        for m in _CLAUSE_SPLIT.finditer(sentence):
            pieces.append((cursor, m.start()))
            cursor = m.end()
        pieces.append((cursor, len(sentence)))
        for a, b in pieces:
            start, end = _trim(text, s_start + a, s_start + b)
            piece = text[start:end]
            balanced = piece.count("(") == piece.count(")") and piece.count(
                "["
            ) == piece.count("]")
            if end - start >= 2 and any(ch.isalnum() for ch in piece) and balanced:
                out.append(_make(text, start, end, piece, "clause"))
    return out


def _sentences(text: str) -> list[Candidate]:
    out = []
    for start, end in _sentence_spans(text):
        start, end = _trim(text, start, end, "")
        if end > start:
            out.append(_make(text, start, end, text[start:end], "sentence"))
    return out


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

_CURRENCY = r"(?:[$€£¥₹]|(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|INR)(?![A-Za-z]))"
_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_EU_DIGITS = r"\d{1,3}(?:\.\d{3})+,\d{1,2}"
_NUMBER = re.compile(
    r"(?<![\w.,/])(?<!\w-)(?P<sign>[-\u2212])?(?P<cur>" + _CURRENCY + r"\s?)?"
    r"(?P<num>" + _EU_DIGITS + "|" + _DIGITS + r"|\.\d+)"
    r"(?P<suf>\s?(?:%|percent\b|per\s?cent\b|thousand\b|million\b|billion\b|k\b|K\b|bn\b)"
    r"|(?<=[$€£¥₹\d])\s?[mM]\b(?<![\d\s][mM]))?"
    r"(?P<cur2>\s(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|INR)(?![A-Za-z]))?"
    r"(?![\w@]|\.\d|,\d)"
)
_SYMBOL_CODE = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}
_MULTIPLIER = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
}
_UNIT_WORDS = {
    **{
        w: n
        for n, w in enumerate(
            [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]
        )
    }
}
_TENS_WORDS = {
    w: 10 * n
    for n, w in enumerate(
        ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"],
        start=2,
    )
}
_SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
_NUMBER_VOCAB = (
    set(_UNIT_WORDS)
    | set(_TENS_WORDS)
    | set(_SCALE_WORDS)
    | {"dozen", "and", "a", "half"}
)
_TOTAL = re.compile(
    r"\b(?:total|sum|altogether|overall|combined|in all|grand|add(?:ed)? up|"
    r"amounts? to|comes? to|amount due|balance|owed|including|incl\.?|all in)\b",
    re.IGNORECASE,
)
_PRODUCT = re.compile(
    r"(?<![\w.,])(?P<q>\d+(?:\.\d+)?)\s*(?:(?:[A-Za-z]+\s+){0,3}(?:at|@)|x|\u00d7|\*)\s*"
    r"(?P<p>" + _CURRENCY + r"?\s?(?:" + _DIGITS + r"))(?![\w@]|\.\d)",
    re.IGNORECASE,
)


def _decimal(num: str) -> Decimal:
    if re.fullmatch(_EU_DIGITS, num):
        num = num.replace(".", "").replace(",", ".")
    return Decimal(num.replace(",", ""))


def _json_number(value: Decimal, field_: ClairicField | None) -> int | float | None:
    """A Decimal as a JSON number; ``None`` when the field cannot hold it."""
    if field_ is not None and field_.decimals is not None:
        value = round(value, field_.decimals)
    whole = value == value.to_integral_value()
    if field_ is not None and field_.integer and not whole:
        return None
    return int(value) if whole else float(value)


def _in_bounds(value: float, field_: ClairicField | None) -> bool:
    if field_ is None or field_.type != "estimate":
        return True
    low, high = field_.min, field_.max
    if isinstance(low, (int, float)) and value < low:
        return False
    return not (isinstance(high, (int, float)) and value > high)


def _protected(text: str) -> list[tuple[int, int]]:
    """Spans whose digits are not quantities: addresses, dates, times, numbers."""
    spans = [m.span() for m in EMAIL.finditer(text)]
    spans += [m.span() for m in URL.finditer(text)]
    spans += phone_spans(text)
    spans += [m.span() for m in _ISO_DATE.finditer(text)]
    spans += [m.span() for m in _SLASH_DATE.finditer(text)]
    spans += [m.span() for m in _CLOCK.finditer(text)]
    spans += [m.span() for m in _DATE_LITERAL.finditer(text)]
    spans += [
        m.span()
        for m in _IDENTIFIER.finditer(text)
        if re.search(r"[A-Za-z]", m.group())
    ]
    return spans


def _numeric_literals(text: str, field_: ClairicField | None) -> list[Candidate]:
    protected = _protected(text)
    out: list[Candidate] = []
    for m in _NUMBER.finditer(text):
        if _overlaps(protected, m.start("num"), m.end("num")):
            continue
        value = _decimal(m.group("num"))
        suffix = (m.group("suf") or "").strip().lower().replace(" ", "")
        percent = suffix in ("%", "percent", "percent")
        if suffix.startswith("per") or percent:
            percent = True
        elif suffix in _MULTIPLIER:
            value *= _MULTIPLIER[suffix]
        if m.group("sign"):
            value = -value
        number = _json_number(value, field_)
        if number is None or not _in_bounds(number, field_):
            continue
        cur = (m.group("cur") or "").strip()
        code = _SYMBOL_CODE.get(cur, cur) or (m.group("cur2") or "").strip()
        reading = None
        shown = str(number)
        if code:
            reading = f"{shown} {code}"
        elif percent:
            reading = f"{shown} percent"
        elif shown != m.group().strip():
            reading = shown
        start, end = _trim(text, m.start(), m.end(), "")
        out.append(
            _make(text, start, end, number, "amount" if code else "number", reading)
        )
    return out


def _number_word_runs(text: str) -> list[tuple[int, int, int | float]]:
    """Runs of number words ("twenty-one", "two hundred and five", "a dozen")."""
    tokens = [
        (m.start(), m.end(), m.group().lower())
        for m in re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)
    ]
    runs: list[tuple[int, int, int | float]] = []
    i = 0
    while i < len(tokens):
        j = i
        words: list[str] = []
        while j < len(tokens):
            parts = tokens[j][2].split("-")
            if not all(p in _NUMBER_VOCAB for p in parts):
                break
            if j > i and text[tokens[j - 1][1] : tokens[j][0]].strip():
                break  # punctuation between the words ends the run
            words.extend(parts)
            j += 1
        if j == i:
            i += 1
            continue
        # a run is trimmed to the longest valid number inside it
        best = _longest_number(words)
        if best is not None:
            used, value = best
            # map the used words back to token positions
            count = 0
            end_token = i
            for k in range(i, j):
                count += len(tokens[k][2].split("-"))
                if count >= used:
                    end_token = k
                    break
            first = _first_valid_start(words)
            start_token = i
            if first:
                count = 0
                for k in range(i, j):
                    if count >= first:
                        start_token = k
                        break
                    count += len(tokens[k][2].split("-"))
            runs.append((tokens[start_token][0], tokens[end_token][1], value))
        i = j
    return runs


def _first_valid_start(words: list[str]) -> int:
    n = 0
    while (
        n < len(words)
        and words[n] in ("and", "a", "half")
        and not _a_number_follows(words, n)
    ):
        n += 1
    return n


def _a_number_follows(words: list[str], at: int) -> bool:
    rest = words[at + 1 : at + 2]
    if words[at] == "a":
        return bool(rest) and rest[0] in (
            "hundred",
            "thousand",
            "million",
            "billion",
            "dozen",
        )
    if words[at] == "half":
        return words[at : at + 3] == ["half", "a", "dozen"]
    return False


def _longest_number(words: list[str]) -> tuple[int, int | float] | None:
    start = _first_valid_start(words)
    best: tuple[int, int | float] | None = None
    for stop in range(start + 1, len(words) + 1):
        value = _parse_number_words(words[start:stop])
        if value is not None:
            best = (stop, value)
    return best


def _parse_number_words(words: list[str]) -> int | float | None:
    if not words:
        return None
    if words == ["half", "a", "dozen"]:
        return 6
    total = 0
    current = 0
    seen = False
    for index, word in enumerate(words):
        if word in _UNIT_WORDS:
            current += _UNIT_WORDS[word]
        elif word in _TENS_WORDS:
            current += _TENS_WORDS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in _SCALE_WORDS:
            total += (current or 1) * _SCALE_WORDS[word]
            current = 0
        elif word == "dozen":
            current = (current or 1) * 12
        elif (word == "a" and index == 0 and len(words) > 1) or (
            word == "and" and 0 < index < len(words) - 1
        ):
            continue
        else:
            return None
        seen = True
    return total + current if seen else None


def _number_words(text: str, field_: ClairicField | None) -> list[Candidate]:
    out = []
    for start, end, value in _number_word_runs(text):
        phrase = text[start:end]
        low = phrase.lower()
        if low in ("a", "and", "half"):
            continue
        before = (
            text[:start].rstrip().split(" ")[-1].lower() if text[:start].strip() else ""
        )
        after = text[end:].lstrip().split(" ")[0].lower().strip(".,;:!?")
        if low == "one" and (
            before in ("no", "any", "every", "some")
            or after in ("of", "another", "else", "thing", "day")
        ):
            continue
        number = _json_number(Decimal(str(value)), field_)
        if number is None or not _in_bounds(number, field_):
            continue
        out.append(_make(text, start, end, number, "number_word", str(number)))
    return out


def _is_total(field_: ClairicField | None) -> bool:
    return bool(field_ and field_.instructions and _TOTAL.search(field_.instructions))


def _sum_candidates(
    text: str, field_: ClairicField | None, literals: list[Candidate]
) -> list[Candidate]:
    """Sums of the amounts and products of the input, for a field that asks for a total.

    ``4 x 250 + 80`` is one sum (``1080``), written over the span from the first term
    to the last. A product is ``<quantity> x|*|@|at <amount>``; the other terms are
    amounts with a currency or numbers directly after a ``+``. Only the sum of the
    terms of one sentence and the sum of all the terms of the input are offered.
    """
    terms: list[tuple[int, int, Decimal]] = []
    taken: list[tuple[int, int]] = []
    for m in _PRODUCT.finditer(text):
        price = _NUMBER.search(m.group("p"))
        if price is None:
            continue
        value = Decimal(m.group("q")) * _decimal(price.group("num"))
        terms.append((m.start(), m.end(), value))
        taken.append((m.start(), m.end()))
    for lit in literals:
        if _overlaps(taken, lit.start, lit.end) or isinstance(lit.value, bool):
            continue
        plus = re.search(r"\+\s*$", text[max(0, lit.start - 4) : lit.start])
        if lit.kind == "amount" or plus:
            terms.append((lit.start, lit.end, Decimal(str(lit.value))))
    terms.sort()
    if len(terms) < 2:
        return []
    groups: list[list[tuple[int, int, Decimal]]] = []
    for s_start, s_end in _sentence_spans(text):
        inside = [t for t in terms if s_start <= t[0] and t[1] <= s_end]
        if len(inside) >= 2:
            groups.append(inside)
    chain = [terms[0]]
    for term in terms[1:]:  # terms joined by "+" ("4 x 250 + 80") are one sum
        if re.fullmatch(r"\s*(?:\+|plus)\s*", text[chain[-1][1] : term[0]]):
            chain.append(term)
        else:
            if len(chain) >= 2:
                groups.append(chain)
            chain = [term]
    if len(chain) >= 2:
        groups.append(chain)
    groups.append(terms)
    out = []
    for group in groups:
        number = _json_number(sum((t[2] for t in group), Decimal(0)), field_)
        if number is None or not _in_bounds(number, field_):
            continue
        start, end = group[0][0], group[-1][1]
        parts = " + ".join(str(_json_number(t[2], None)) for t in group)
        out.append(
            _make(text, start, end, number, "sum", f"{number} (the sum of {parts})")
        )
    return out


def _number_candidates(
    text: str, field_: ClairicField | None, composites: bool
) -> list[Candidate]:
    literals = _numeric_literals(text, field_)
    out = literals + _number_words(text, field_)
    if composites and _is_total(field_):
        out += _sum_candidates(text, field_, literals)
    return out


# --------------------------------------------------------------------------- #
# Date, time and duration mentions (found here, read by the model, resolved in
# ``_dateparts``: no calendar math happens in this module)
# --------------------------------------------------------------------------- #

MONTHS = {
    name: number
    for number, names in enumerate(
        (
            "january jan",
            "february feb",
            "march mar",
            "april apr",
            "may",
            "june jun",
            "july jul",
            "august aug",
            "september sep sept",
            "october oct",
            "november nov",
            "december dec",
        ),
        start=1,
    )
    for name in names.split()
}
WEEKDAYS = {
    name: number
    for number, names in enumerate(
        (
            "monday mon",
            "tuesday tue tues",
            "wednesday wed",
            "thursday thu thur thurs",
            "friday fri",
            "saturday sat",
            "sunday sun",
        )
    )
    for name in names.split()
}
_MONTH_NAMES = "|".join(sorted(MONTHS, key=len, reverse=True))
_WEEKDAY_NAMES = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_ISO_DATE = re.compile(r"(?<![\d\-/.])\d{4}-\d{2}-\d{2}(?!\d)")
_SLASH_DATE = re.compile(r"(?<![\d\-/.])\d{1,2}([/.\-])\d{1,2}\1(?:\d{4}|\d{2})(?!\d)")
_CLOCK = re.compile(
    r"(?<![\d:.\-/])\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:[ \t]?(?:a\.?m\.?|p\.?m\.?)(?![A-Za-z]))?",
    re.IGNORECASE,
)
# Every gap is a space or a tab, never a line break: a date does not run across lines.
_MONTH_ANY = f"(?:{_MONTH_NAMES.title()}|{_MONTH_NAMES.upper()})"
_DATE_LITERAL = re.compile(
    r"(?<![\d\-/.])\d{4}-\d{2}-\d{2}(?!\d)"
    r"|(?<![\d\-/.])\d{1,2}([/.\-])\d{1,2}\1(?:\d{4}|\d{2})(?!\d)"
    rf"|\b{_MONTH_ANY}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?(?!\d)"
    r"(?:(?:,[ \t]*|[ \t]+)\d{4}\b)?"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?[ \t]+(?:of[ \t]+)?{_MONTH_ANY}\.?"
    r"(?:(?:,[ \t]*|[ \t]+)\d{4}\b)?"
    r"|\b(?:the[ \t]+day[ \t]+after[ \t]+tomorrow|day[ \t]+after[ \t]+tomorrow"
    r"|today|tomorrow|yesterday)\b"
    rf"|\b(?:(?:next|last|this|coming|past)[ \t]+)?(?:{_WEEKDAY_NAMES})\b"
    r"|\b(?:in[ \t]+(?:\d+|an?|[a-z]+)[ \t]+(?:day|week|month|year)s?"
    r"|(?:\d+|an?|[a-z]+)[ \t]+(?:day|week|month|year)s?"
    r"[ \t]+(?:ago|from[ \t]+now))\b",
    re.IGNORECASE,
)


def _date_literals(text: str) -> list[Candidate]:
    """Where the input writes a date, found by pattern and left unread."""
    out = []
    protected = [m.span() for m in _CLOCK.finditer(text)]
    for m in _DATE_LITERAL.finditer(text):
        start, end = _trim(text, m.start(), m.end(), ".,;:")
        word = text[start:end].lower()
        if word in ("sat", "sun", "mon", "wed", "may", "mar") or _overlaps(
            protected, start, end
        ):
            continue
        if re.fullmatch(
            r"(?:in|an?|[a-z]+)\s+(?:day|week|month|year)s?\s+ago", word
        ) and (
            word.split()[0]
            not in ("a", "an", "one", "two", "three", "four", "five", "six")
        ):
            continue
        out.append(_make(text, start, end, text[start:end], "date"))
    for m in _CLOCK.finditer(text):
        start, end = _trim(text, m.start(), m.end(), ".")
        out.append(_make(text, start, end, text[start:end], "time"))
    return out


def date_mentions(text: str) -> list[Candidate]:
    """Spans of the input that write a date or time (for a list of dates)."""
    return _date_literals(text)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def item_type(field_: ClairicField) -> str:
    """What the candidates of a field are: ``text``, ``number``, ``date``, ...

    A list is read as its item type (``text``, ``number`` or ``date``); ``estimate``
    is a number.
    """
    if field_.type == "list":
        return field_.items or "text"
    if field_.type in ("estimate", "forecast"):
        return "number"
    return field_.type


def _finalize(found: list[Candidate], text: str, width: int) -> list[Candidate]:
    found.sort(key=lambda c: (c.start, c.end, c.kind))
    merged: dict[str, list[Candidate]] = {}
    for candidate in found:
        if candidate.text.strip():
            merged.setdefault(candidate.text, []).append(candidate)
    out = []
    for group in merged.values():
        first = group[0]
        places = tuple(dict.fromkeys(c.span for c in group))[:3]
        out.append(
            replace(
                first,
                occurrences=places,
                contexts=tuple(context_snippet(text, s, e, width) for s, e in places),
            )
        )
    out.sort(key=lambda c: (c.start, c.end))
    return out


MAX_ID_CHARS = 100
"""A verbatim span longer than this is not offered as an option of its own."""


def pattern_candidates(
    field_: ClairicField, text: str, *, width: int = 40
) -> list[Candidate]:
    """The pattern-shaped candidates of one field: spans found by regex, deduplicated.

    For a text field these are the spans of an email, URL, telephone number,
    labelled value, quoted string, identifier, name, amount, number and date; for a
    number field the numbers (amounts, percentages, number words and, for a field that
    asks for a total, sums); for a list the same as its item type. The model picks
    among them (the span itself is the option id, as in TypeSafe's pre-parsed value
    cookbook). Dates, times and durations are *not* candidates of a date field: the
    model reads their parts instead (``_dateparts``).

    Args:
        field_: The Clairic field spec.
        text: The input; the spans are offsets into it.
        width: Characters of context on each side of an occurrence.

    Returns:
        Candidates whose ``text`` is ``text[start:end]``, one per distinct text, in
        document order, each at most ``MAX_ID_CHARS`` long.
    """
    kind = item_type(field_)
    found: list[Candidate] = []
    if kind == "text":
        for generator in (
            _emails,
            _urls,
            _phones,
            _labelled,
            _quoted,
            _identifiers,
            _names,
        ):
            found.extend(generator(text)[:_MAX_PER_KIND])
        found.extend(_number_candidates(text, None, False)[:_MAX_PER_KIND])
        found.extend(_date_literals(text)[:_MAX_PER_KIND])
    elif kind == "number":
        found.extend(_number_candidates(text, field_, field_.type != "list"))
    elif kind == "date":
        found.extend(date_mentions(text))
    found = [c for c in found if len(c.text) <= MAX_ID_CHARS]
    return _finalize(found, text, width)


def structure_pieces(text: str) -> dict[str, list[Candidate]]:
    """Noun phrases and clauses of the input as raw candidates (see ``_structure``)."""
    return {"phrase": _noun_phrases(text), "clause": _clauses(text)}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Spans of the sentences of ``text`` (the library's splitter, offsets exact)."""
    return [_trim(text, s, e, "") for s, e in _sentence_spans(text)]
