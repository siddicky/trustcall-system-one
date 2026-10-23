"""Relative dates ("yesterday", "three days ago", "last Tuesday") read in code.

Clairic's ``reference_date`` is documented as the date that "anchors dates written
without a year": nothing in its contract says that it resolves a relative phrase, and
a date answer that is not an ISO string makes the field invalid (``_coerce``). Two
things follow from that, and both live here:

* a phrase that means ONE date whatever the convention (``yesterday``, ``3 days ago``,
  ``a month from now``, and ``in two weeks`` where the sentence says it is ahead) is
  resolved by ``resolve_phrase`` against the reference date, without any model, so a
  Clairic answer that hands the phrase back is read instead of rejected;
* ``find_phrases`` finds the phrases a text holds (the trigger of the escalation tier,
  ``_escalate``) and ``plausible`` checks what a model says such a phrase means: a
  weekday phrase ("last Tuesday") depends on a convention that the language does not
  settle, so the code does not pick a date for it, but it can refuse one that falls on
  another weekday or on the wrong side of the reference date.

What is NOT a relative date: ``N days later`` or ``earlier`` counts from another date
the text names ("arrived on 1 March and was inspected two days later"), not from
today, so it is neither found nor resolved; and ``in N weeks`` is a duration as often
as a distance from today ("the overhaul took three days, done in two weeks"), so it
counts only where its sentence is in the future tense ("will", "due", "scheduled",
"opens", ...) or says ``from now``.
"""

import re
from datetime import date, timedelta

from trustcall_system_one._dateparts import WEEKDAY_NAMES, _shift

__all__ = ["Phrase", "find_phrases", "plausible", "resolve_phrase", "sentence_around"]

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
}
_COUNT = r"(?:\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
_UNIT = r"(?:day|week|fortnight|month|year)s?"
_WEEKDAY = "|".join(name.lower() for name in WEEKDAY_NAMES)
_DAY_WORDS = (
    r"(?:the\s+day\s+after\s+tomorrow|day\s+after\s+tomorrow"
    r"|the\s+day\s+before\s+yesterday|day\s+before\s+yesterday"
    r"|today|tomorrow|yesterday)"
)
_PHRASE = re.compile(
    rf"\b(?:{_DAY_WORDS}"
    rf"|{_COUNT}[ \t]+{_UNIT}[ \t]+(?:ago|from[ \t]+now)"
    rf"|in[ \t]+{_COUNT}[ \t]+{_UNIT}"
    rf"|(?:next|last|this|coming|past)[ \t]+(?:{_WEEKDAY}))\b",
    re.IGNORECASE,
)
_COUNTED = re.compile(
    rf"(?:in[ \t]+(?P<ahead>{_COUNT})[ \t]+(?P<u1>{_UNIT})"
    rf"|(?P<n>{_COUNT})[ \t]+(?P<u2>{_UNIT})[ \t]+"
    r"(?P<way>ago|from[ \t]+now))",
    re.IGNORECASE,
)
_FUTURE = re.compile(
    r"\b(?:will|shall|would|going[ \t]+to|due|scheduled|expected|planned?|plans?"
    r"|set[ \t]+to|to[ \t]+be|should|must|upcoming|begins?|starts?|opens?|arrives?"
    r"|expires?|ends?|closes?|returns?|resumes?)\b|'ll\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?;\n]")
_DAYS = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
    "day after tomorrow": 2,
    "day before yesterday": -2,
}


class Phrase(tuple[int, int, str]):
    """A relative phrase of a text: ``(start, end, text)``."""

    __slots__ = ()

    @property
    def start(self) -> int:
        """Offset of its first character."""
        return self[0]

    @property
    def end(self) -> int:
        """Offset after its last character."""
        return self[1]

    @property
    def text(self) -> str:
        """The phrase as the text writes it."""
        return self[2]


def _fold(text: str) -> str:
    return " ".join(text.lower().split())


def sentence_around(text: str, start: int, end: int) -> str:
    """The sentence (or line) of ``text`` that holds ``text[start:end]``."""
    left = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    right = _SENTENCE_END.search(text, end)
    return text[left : right.start() if right else len(text)]


def _ahead(phrase: str, context: str | None) -> bool:
    """Whether an ``in N unit`` phrase is ahead of today: its sentence says so."""
    return context is not None and _FUTURE.search(context) is not None


def find_phrases(
    text: str, windows: list[tuple[int, int]] | None = None
) -> list[Phrase]:
    """The relative phrases a text holds, optionally only inside some windows.

    Args:
        text: The input.
        windows: ``(start, end)`` ranges to look in; the whole text when ``None``.

    Returns:
        The phrases in order of appearance, verbatim from ``text``.
    """
    found: list[Phrase] = []
    for start, end in windows or [(0, len(text))]:
        for match in _PHRASE.finditer(text, start, end):
            if _fold(match.group(0)).startswith("in ") and not _ahead(
                match.group(0), sentence_around(text, match.start(), match.end())
            ):
                continue  # "in two weeks" as a duration, not a date
            found.append(Phrase((match.start(), match.end(), match.group(0))))
    return found


def resolve_phrase(
    phrase: str, today: date, *, context: str | None = None
) -> date | None:
    """The date a phrase names, when it names exactly one whatever the convention.

    ``today``, ``tomorrow``, ``yesterday``, the day after tomorrow or before
    yesterday, ``N days|weeks|fortnights|months|years ago|from now`` and ``in N ...``
    when ``context`` (the sentence of the phrase) is in the future tense. ``N days
    later`` counts from another date and is never resolved, nor is ``in N ...`` without
    a future-tense context (a duration as often as a date). A weekday phrase ("last
    Tuesday", "next Friday") is ``None``: a reader who says "last Tuesday" on a Friday
    may mean either of two dates, and the code does not choose.

    Args:
        phrase: The phrase, as ``find_phrases`` returns it.
        today: The reference date.
        context: The sentence the phrase stands in (for ``in N ...``).

    Returns:
        The date, or ``None`` when the phrase is not one of those above.
    """
    folded = _fold(phrase)
    folded = re.sub(r"^the\s+", "", folded)
    if folded in _DAYS:
        return today + timedelta(days=_DAYS[folded])
    matched = _COUNTED.fullmatch(" ".join(phrase.split()))
    if matched is None:
        return None
    ahead = matched.group("ahead") is not None
    if ahead and not _ahead(phrase, context):
        return None
    count_word = (matched.group("ahead") or matched.group("n")).lower()
    unit = (matched.group("u1") or matched.group("u2")).lower().rstrip("s")
    count = int(count_word) if count_word.isdigit() else _NUMBER_WORDS[count_word]
    way = (matched.group("way") or "").lower()
    forward = ahead or way.startswith("from")
    if unit == "fortnight":
        count, unit = count * 2, "week"
    return _shift(today, count, f"{unit}s_{'ahead' if forward else 'ago'}")


def plausible(
    phrase: str, found: date, today: date, *, context: str | None = None
) -> bool:
    """Whether ``found`` can be what ``phrase`` means, relative to ``today``.

    A phrase that ``resolve_phrase`` settles must equal it. A weekday phrase must fall
    on that weekday, before ``today`` for ``last``/``past`` and after it for
    ``next``/``coming``, at most 14 days away (``this`` Tuesday: within the 7 days
    either side). Any other phrase is not judged here (``True``).

    Args:
        phrase: The relative phrase.
        found: The date a model gave for it.
        today: The reference date.
        context: The sentence the phrase stands in (see ``resolve_phrase``).

    Returns:
        ``False`` when the date cannot be what the phrase says.
    """
    exact = resolve_phrase(phrase, today, context=context)
    if exact is not None:
        return exact == found
    words = _fold(phrase).split()
    if len(words) == 2 and words[1] in _WEEKDAY.split("|"):
        if found.weekday() != [n.lower() for n in WEEKDAY_NAMES].index(words[1]):
            return False
        delta = (found - today).days
        if words[0] in ("last", "past"):
            return -14 <= delta < 0
        if words[0] in ("next", "coming"):
            return 0 < delta <= 14
        return -7 <= delta <= 7
    return True
