"""Dates, times and durations read as parts, then resolved in code.

This is TypeSafe's date extraction cookbook, generalised to the Clairic types
``date``, ``time``, ``datetime`` and ``duration``. The model never does calendar math
and never writes a date: it answers closed ``choice`` questions about the *parts*
the text names, all in one request, and code resolves and validates them.

* ``date``: the cookbook's seven choices: ``mode`` (``absolute``, ``relative`` or
  ``none``), ``month``, ``day``, ``year`` (1900 to 2050, plus ``none`` and
  ``out_of_range``), ``day_anchor`` (today, tomorrow, the day after, a weekday),
  ``weekday`` and ``week_offset``; and two of this library's for a date given as a
  distance from today ("three days ago", "in two weeks", "a month from now"):
  ``offset_count`` and ``offset_unit`` (the unit and the direction in one choice:
  ``days_ago``, ``weeks_ahead``, ...), read when ``day_anchor`` is ``offset``. Code
  reads only the parts ``mode`` calls for.
  A date without a year takes the reference date's year (the next if it is more than
  31 days past); "next Thursday" is the following calendar week, "this Thursday" this
  week, a bare "Thursday" the next one on or after the reference date.
* ``time``: ``mode`` (``stated`` or ``none``), ``hour`` on the 24-hour clock and
  ``minute``.
* ``datetime``: the date parts and the time parts together.
* ``duration``: ``mode`` and one count per unit (years to seconds).

The confidence of an answer is the weakest confidence among the parts that were
used (the cookbook's ``assemble``), and ``needs_review`` is its flag: no value, or a
confidence below ``REVIEW_BELOW`` (0.60 in the cookbook).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

__all__ = [
    "DATE_PARTS",
    "INFERRED_CONFIDENCE",
    "REVIEW_BELOW",
    "TIME_PARTS",
    "Assembled",
    "Part",
    "assemble_date",
    "assemble_datetime",
    "assemble_duration",
    "assemble_time",
    "date_questions",
    "duration_parts",
    "duration_questions",
    "time_questions",
]

REVIEW_BELOW = 0.60
"""Cookbook gate: a date below this confidence is flagged for a person."""

INFERRED_CONFIDENCE = 0.8
"""The most confidence a date gets when its year or its week was not stated but
inferred (a year-less date, a bare weekday). It sits under the library's
low-confidence bar (0.85), so the ladder re-asks and cross-checks it, and the answer
is flagged ``needs_review``."""

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
YEAR_WINDOW = list(range(1900, 2051))
DATE_PARTS = (
    "mode",
    "month",
    "day",
    "year",
    "day_anchor",
    "weekday",
    "week_offset",
    "offset_count",
    "offset_unit",
)
OFFSET_UNITS = tuple(
    f"{unit}_{way}"
    for unit in ("days", "weeks", "months", "years")
    for way in ("ago", "ahead")
)
"""The choices of ``offset_unit``: a unit and the direction from today."""
TIME_PARTS = ("mode", "hour", "minute")
DURATION_UNITS = ("years", "months", "weeks", "days", "hours", "minutes", "seconds")


def duration_parts() -> tuple[str, ...]:
    """The part names of a duration: ``mode`` and the units."""
    return ("mode", *DURATION_UNITS)


_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 7 * 86400.0,
    "months": 30.436875 * 86400.0,
    "years": 365.2425 * 86400.0,
}
_COUNTS = [
    "0.5",
    *(str(n) for n in range(1, 121)),  # every whole count to 120 (90 minutes, 100 ...)
    "150",
    "180",
    "240",
    "300",
    "360",
    "480",
    "600",
]  # beyond 120 only these; any other count is ``out_of_range`` (flagged, not guessed)
_ABSENT = "The document does not state this, or it is not this kind of date."


@dataclass(frozen=True)
class Part:
    """One choice answer: the option, its confidence and its probabilities."""

    choice: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class Assembled:
    """What the parts add up to.

    Attributes:
        value: The resolved value (ISO date, time or datetime text, or a number of
            the field's unit), or ``None``.
        confidence: The lowest confidence among the parts that were used.
        p_null: The probability that the input states no such value.
        note: Why there is no value, or empty.
        needs_review: No value, a confidence below ``REVIEW_BELOW``, or a year or
            week that was inferred and not stated.
        inferred: What was inferred: ``year`` (a date without a year) or ``week``
            (a bare weekday).
    """

    value: Any
    confidence: float | None
    p_null: float
    note: str = ""
    needs_review: bool = False
    inferred: tuple[str, ...] = ()


def _choice(instructions: str, criteria: dict[str, Any]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def date_questions(role: str) -> dict[str, dict[str, Any]]:
    """The seven date choices of the cookbook; ``role`` names the date wanted."""
    return {
        "mode": _choice(
            f"How is {role} written? 'absolute' = a calendar date naming a month "
            "(e.g. 'August 14', 'the 3rd of March'); 'relative' = given relative to "
            "today (today, tomorrow, the day after tomorrow, or a named weekday such "
            "as 'next Thursday'); 'none' = the document does not state this date.",
            {"absolute": None, "relative": None, "none": None},
        ),
        "month": _choice(
            f"If {role} is an absolute calendar date, which month is it in?",
            {m: None for m in MONTH_NAMES} | {"none": _ABSENT},
        ),
        "day": _choice(
            f"If {role} is an absolute calendar date, which day of the month (1-31)?",
            {str(d): None for d in range(1, 32)} | {"none": _ABSENT},
        ),
        "year": _choice(
            f"If {role} is an absolute calendar date, which year? Pick 'none' if the "
            "document states no year (code infers it), or 'out_of_range' if a year "
            "is stated but not in the list.",
            {str(y): None for y in YEAR_WINDOW}
            | {
                "out_of_range": "A year is stated for this date but is outside the "
                "listed range.",
                "none": "No year is stated for this date.",
            },
        ),
        "day_anchor": _choice(
            f"If {role} is relative to today, which day is it? 'today', 'tomorrow', "
            "'yesterday', 'day_after' (the day after tomorrow), 'weekday' (a named "
            "day of the week), or 'offset' (a number of days, weeks, months or years "
            "before or after today: 'three days ago', 'in two weeks').",
            {
                "today": None,
                "tomorrow": None,
                "yesterday": None,
                "day_after": None,
                "weekday": None,
                "offset": None,
                "none": _ABSENT,
            },
        ),
        "weekday": _choice(
            f"If {role} names a day of the week, which one?",
            {w: None for w in WEEKDAY_NAMES} | {"none": _ABSENT},
        ),
        "week_offset": _choice(
            f"If {role} names a weekday, which week is it in? 'next' for 'next "
            "Thursday' or 'Thursday next week'; 'current' for 'this Thursday'; "
            "'previous' for 'last Thursday' or 'Thursday last week'; 'none' for a "
            "bare weekday with no qualifier (just 'Thursday' / 'on Thursday').",
            {"current": None, "next": None, "previous": None, "none": _ABSENT},
        ),
        "offset_count": _choice(
            f"If {role} is a number of days, weeks, months or years from today ('three "
            "days ago', 'in two weeks', 'a month from now'), what is the number ('a' "
            "and 'an' are 1)? 'none' if no number is stated ('last week', 'next "
            "month'), 'out_of_range' if the number is above 60 or is not a whole "
            "number.",
            {str(n): None for n in range(1, 61)}
            | {
                "out_of_range": "A number is stated and is not one of 1 to 60.",
                "none": _ABSENT,
            },
        ),
        "offset_unit": _choice(
            f"If {role} is a number of days, weeks, months or years from today, which "
            "unit is it in, and is it before today ('ago', 'earlier') or after it "
            "('in two weeks', 'a month from now')? A phrase without a number ('last "
            "week', 'next month') is 'none'.",
            {u: None for u in OFFSET_UNITS} | {"none": _ABSENT},
        ),
    }


def time_questions(role: str) -> dict[str, dict[str, Any]]:
    """Three choices that read a time of day: ``mode``, ``hour`` and ``minute``."""
    return {
        "mode": _choice(
            f"Does the document state {role}, a time of day (such as '2:30 pm', "
            "'14:00', 'noon')? 'stated' if it does, 'none' if it does not.",
            {"stated": None, "none": None},
        ),
        "hour": _choice(
            f"If the document states {role}, which hour is it on the 24-hour clock "
            "(2 pm is 14, noon is 12, midnight is 0)?",
            {str(h): None for h in range(24)} | {"none": _ABSENT},
        ),
        "minute": _choice(
            f"If the document states {role}, which minute is it (0 for 'noon' or "
            "'2 pm')?",
            {str(m): None for m in range(60)} | {"none": _ABSENT},
        ),
    }


def duration_questions(role: str) -> dict[str, dict[str, Any]]:
    """``mode`` and one count per unit for a length of time."""
    out: dict[str, dict[str, Any]] = {
        "mode": _choice(
            f"Does the document state {role}, a length of time (such as '2 hours "
            "30 minutes', 'three weeks')? 'stated' if it does, 'none' if it does "
            "not.",
            {"stated": None, "none": None},
        )
    }
    for unit in DURATION_UNITS:
        out[unit] = _choice(
            f"If the document states {role}, how many {unit} does it say? Count only "
            f"the {unit} part (for '2 hours 30 minutes' hours is 2 and minutes is "
            f"30); 'none' if no {unit} part is stated; 'out_of_range' if the number "
            "is larger than the listed ones.",
            {c: None for c in _COUNTS}
            | {
                "none": f"No {unit} part is stated.",
                "out_of_range": "A larger number is stated.",
            },
        )
    return out


def _conf(parts: Mapping[str, Part], names: list[str]) -> float | None:
    values = [parts[n].confidence for n in names if n in parts]
    return min(values) if values else None


def _none_prob(parts: Mapping[str, Part]) -> float:
    """P(the input states no such value): the larger of the ``mode`` answers.

    A date or time or duration has a ``mode``; a datetime has two (``d.mode``,
    ``t.mode``), and it is null if either part is not stated.
    """
    modes = [parts[k] for k in ("mode", "d.mode", "t.mode") if k in parts]
    return max((float(m.probabilities.get("none", 0.0)) for m in modes), default=0.0)


def _result(
    value: Any,
    parts: Mapping[str, Part],
    used: list[str],
    note: str,
    review_below: float,
    inferred: tuple[str, ...] = (),
) -> Assembled:
    confidence = _conf(parts, used)
    if value is None:
        p_null = max(_none_prob(parts), 1.0 - (confidence or 0.0))
        inferred = ()
    else:
        p_null = _none_prob(parts)
    if inferred and confidence is not None:
        confidence = min(confidence, INFERRED_CONFIDENCE)
        note = note or "inferred " + " and ".join(inferred) + ", not stated"
    review = (
        value is None
        or confidence is None
        or confidence < review_below
        or bool(inferred)
    )
    return Assembled(
        value, confidence, min(1.0, max(0.0, p_null)), note, review, inferred
    )


def _shift(today: date, count: int, unit: str) -> date | None:
    """``today`` moved ``count`` units (``unit`` is one of ``OFFSET_UNITS``).

    ``None`` when there is no such date. A month or a year that lands past the end of
    a month stops on its last day (a month before 31 March is 28 February).
    """
    name, _, way = unit.partition("_")
    sign = -1 if way == "ago" else 1
    try:
        if name == "days":
            return today + timedelta(days=sign * count)
        if name == "weeks":
            return today + timedelta(weeks=sign * count)
        months = count * (12 if name == "years" else 1)
        year, month = divmod(today.year * 12 + today.month - 1 + sign * months, 12)
        month += 1
        for day in range(today.day, 28, -1):
            try:
                return date(year, month, day)
            except ValueError:
                continue
        return date(year, month, min(today.day, 28))
    except (OverflowError, ValueError):
        return None


def _resolve_weekday(today: date, weekday: str, week_offset: str) -> date:
    index = WEEKDAY_NAMES.index(weekday)
    monday = today - timedelta(days=today.weekday())
    if week_offset == "next":
        return monday + timedelta(days=7 + index)
    if week_offset == "current":
        return monday + timedelta(days=index)
    if week_offset == "previous":
        return monday + timedelta(days=index - 7)
    return today + timedelta(days=(index - today.weekday()) % 7)


def _date_of(
    parts: Mapping[str, Part], today: date
) -> tuple[date | None, list[str], str, tuple[str, ...]]:
    """The cookbook's ``assemble``: ``(date, parts used, note, inferred)``.

    ``inferred`` names what the text did not state and code filled in: the ``year``
    of a year-less date (the reference year, or the next if it is more than 31 days
    past: a bias to the future), the ``week`` of a bare weekday (the next one on or
    after the reference date).
    """
    mode = parts["mode"].choice if "mode" in parts else "none"
    used = ["mode"]
    if mode == "none":
        return None, used, "no such date stated", ()
    if mode == "absolute":
        used += ["month", "day", "year"]
        month = parts["month"].choice
        day = parts["day"].choice
        year = parts["year"].choice
        if "none" in (month, day) or not day.isdigit() or month not in MONTH_NAMES:
            return None, used, "absolute date incomplete", ()
        number = MONTH_NAMES.index(month) + 1
        if year == "out_of_range":
            return None, used, f"year outside {YEAR_WINDOW[0]}-{YEAR_WINDOW[-1]}", ()
        try:
            if year == "none":
                resolved = date(today.year, number, int(day))
                if resolved < today - timedelta(days=31):
                    resolved = date(today.year + 1, number, int(day))
                return resolved, used, "", ("year",)
            return date(int(year), number, int(day)), used, "", ()
        except ValueError:
            return None, used, f"impossible date: {year}-{month}-{day}", ()
    if mode == "relative":
        anchor = parts["day_anchor"].choice
        used.append("day_anchor")
        offsets = {"yesterday": -1, "today": 0, "tomorrow": 1, "day_after": 2}
        if anchor in offsets:
            return today + timedelta(days=offsets[anchor]), used, "", ()
        if anchor == "offset":
            used += ["offset_count", "offset_unit"]
            count = parts["offset_count"].choice if "offset_count" in parts else ""
            unit = parts["offset_unit"].choice if "offset_unit" in parts else ""
            if count == "out_of_range":
                return None, used, "relative offset out of range", ()
            if not count.isdigit() or unit not in OFFSET_UNITS:
                return None, used, "relative offset not read", ()
            shifted = _shift(today, int(count), unit)
            if shifted is None:
                return None, used, "relative offset out of range", ()
            return shifted, used, "", ()
        if anchor == "weekday":
            used += ["weekday", "week_offset"]
            weekday = parts["weekday"].choice
            if weekday not in WEEKDAY_NAMES:
                return None, used, "relative weekday not read", ()
            offset = parts["week_offset"].choice
            named = _resolve_weekday(today, weekday, offset)
            stated = offset in ("next", "current", "previous")
            return named, used, "", (() if stated else ("week",))
        return None, used, "relative day not read", ()
    return None, used, f"unrecognized mode: {mode}", ()


def assemble_date(
    parts: Mapping[str, Part], today: date, review_below: float = REVIEW_BELOW
) -> Assembled:
    """Resolve the date parts against ``today`` (the reference date)."""
    resolved, used, note, inferred = _date_of(parts, today)
    value = None if resolved is None else resolved.isoformat()
    return _result(value, parts, used, note, review_below, inferred)


def _time_of(parts: Mapping[str, Part]) -> tuple[str | None, list[str], str]:
    used = ["mode"]
    if "mode" not in parts or parts["mode"].choice != "stated":
        return None, used, "no such time stated"
    used += ["hour", "minute"]
    hour = parts["hour"].choice
    minute = parts["minute"].choice
    if not hour.isdigit():
        return None, used, "time incomplete"
    clock = int(minute) if minute.isdigit() else 0
    return f"{int(hour):02d}:{clock:02d}:00", used, ""


def assemble_time(
    parts: Mapping[str, Part], review_below: float = REVIEW_BELOW
) -> Assembled:
    """Resolve the time parts (a missing minute is 0)."""
    resolved, used, note = _time_of(parts)
    return _result(resolved, parts, used, note, review_below)


def assemble_datetime(
    date_parts: Mapping[str, Part],
    time_parts: Mapping[str, Part],
    today: date,
    review_below: float = REVIEW_BELOW,
) -> Assembled:
    """Resolve date and time parts together; a date without a time is no datetime."""
    day, used, note, inferred = _date_of(date_parts, today)
    clock, time_used, time_note = _time_of(time_parts)
    merged: dict[str, Part] = {f"d.{k}": v for k, v in date_parts.items()}
    merged.update({f"t.{k}": v for k, v in time_parts.items()})
    names = [f"d.{n}" for n in used] + [f"t.{n}" for n in time_used]
    if day is None or clock is None:
        return _result(None, merged, names, note or time_note, review_below)
    return _result(
        f"{day.isoformat()}T{clock}", merged, names, "", review_below, inferred
    )


def assemble_duration(
    parts: Mapping[str, Part],
    unit: str,
    *,
    integer: bool,
    decimals: int | None,
    review_below: float = REVIEW_BELOW,
) -> Assembled:
    """Add up the stated unit counts, in the field's ``unit``."""
    used = ["mode"]
    if "mode" not in parts or parts["mode"].choice != "stated":
        return _result(None, parts, used, "no such length stated", review_below)
    total = 0.0
    found = False
    for name in DURATION_UNITS:
        part = parts.get(name)
        if part is None or part.choice == "none":
            continue
        used.append(name)
        if part.choice == "out_of_range":
            return _result(None, parts, used, f"{name} out of range", review_below)
        total += float(part.choice) * _SECONDS[name]
        found = True
    if not found:
        return _result(None, parts, used, "no unit stated", review_below)
    amount = total / _SECONDS.get(unit, 1.0)
    amount = round(amount, 6 if decimals is None else decimals)
    if integer and amount != round(amount):
        return _result(
            None, parts, used, "not a whole number of the unit", review_below
        )
    value: Any = int(amount) if amount == int(amount) else amount
    return _result(value, parts, used, "", review_below)
