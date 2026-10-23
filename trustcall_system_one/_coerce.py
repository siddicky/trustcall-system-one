"""Convert model answers into Python values."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import TypeAdapter, ValidationError

from trustcall_system_one._compile import FieldSpec
from trustcall_system_one._config import GateConfig
from trustcall_system_one._relative import resolve_phrase
from trustcall_system_one._types import UNSTATED
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClairicAnswer,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

__all__ = ["UNSTATED", "Coerced", "coerce", "coerce_union"]

_EPS = 1e-9
_SECONDS_PER_UNIT: dict[str, float] = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 7 * 86400.0,
    "months": 30.436875 * 86400.0,
    "years": 365.2425 * 86400.0,
}
_UNIT_ALIASES: dict[str, str] = {
    "s": "seconds",
    "sec": "seconds",
    "secs": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "m": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days",
    "w": "weeks",
    "wk": "weeks",
    "wks": "weeks",
    "week": "weeks",
    "weeks": "weeks",
    "mo": "months",
    "month": "months",
    "months": "months",
    "y": "years",
    "yr": "years",
    "yrs": "years",
    "year": "years",
    "years": "years",
}
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)")
_CLOCK = re.compile(r"(\d+):([0-5]?\d)(?::([0-5]?\d(?:\.\d+)?))?")
_ISO_DURATION = re.compile(
    r"P(?:(?P<y>\d+(?:\.\d+)?)Y)?(?:(?P<mo>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<w>\d+(?:\.\d+)?)W)?(?:(?P<d>\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<mi>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)S)?)?",
    re.IGNORECASE,
)
_ISO_UNITS = {
    "y": "years",
    "mo": "months",
    "w": "weeks",
    "d": "days",
    "h": "hours",
    "mi": "minutes",
    "s": "seconds",
}


@dataclass(frozen=True)
class Coerced:
    """The outcome of converting one model answer.

    Attributes:
        value: The Python value (``None`` when the model said null or the
            conversion failed).
        ok: False when the answer could not be converted or the model
            abstained; ``reason`` says why.
        reason: Explanation for ``ok=False``, or a note that some list elements
            were dropped (``ok`` stays true then).
        span: ``(start, end)`` of the supporting text, when the answer had a
            well-formed ``span``.
        spans: Supporting spans of a list answer, when all were well-formed.
        p_null: The model's probability that the value is null.
        confidence: The model's confidence.
        guess: Clairic's ``guess``, kept as metadata only; it is never used as
            the value.
        score: P(yes) for a noul answer, the probability of the chosen option
            for a choice, the probability-weighted level for a score.
        kept: For a list answer, the position in the answer's own list of each
            element in ``value`` (``spans`` follow the same elements).
        failed: For a list answer, ``(position, message)`` of each element that
            could not be converted and was dropped.
    """

    value: Any
    ok: bool = True
    reason: str | None = None
    span: tuple[int, int] | None = None
    spans: tuple[tuple[int, int], ...] | None = None
    p_null: float | None = None
    confidence: float | None = None
    guess: Any = None
    score: float | None = None
    kept: tuple[int, ...] | None = None
    failed: tuple[tuple[int, str], ...] = ()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pair(value: Any) -> tuple[int, int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_int(part) for part in value)
        and 0 <= value[0] <= value[1]
    ):
        return (value[0], value[1])
    return None


def _spans(value: Any) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    pairs = [_pair(item) for item in value]
    if any(pair is None for pair in pairs):
        return None
    return tuple(pair for pair in pairs if pair is not None)


# --------------------------------------------------------------------------- #
# Scalar parsers (all raise ValueError with a short reason)
# --------------------------------------------------------------------------- #


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("a boolean is not a number")
    try:
        number = Decimal(value.strip() if isinstance(value, str) else str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{value!r} is not a number") from None
    if not number.is_finite():
        raise ValueError(f"{value!r} is not a finite number")
    return number


def _to_number(value: Any, post: str) -> Any:
    if post == "decimal":
        return _decimal(value)
    if post == "float":
        return float(_decimal(value))
    number = _decimal(value)
    if post == "int":
        if number != number.to_integral_value():
            raise ValueError(f"{value!r} is not a whole number")
        return int(number)
    if isinstance(value, str):
        return float(number) if number != number.to_integral_value() else int(number)
    return value


_DATE = TypeAdapter(date)
_DATETIME = TypeAdapter(datetime)
_TIME = TypeAdapter(time)
_NUMERIC = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _fixed_offset(offset: timedelta | None) -> timezone | None:
    return None if offset is None else timezone(offset)


def _checked_text(value: Any, what: str) -> str:
    """Return the stripped string, refusing non-strings and bare numbers.

    pydantic reads a bare number as a Unix timestamp (``"20261021"`` would
    become a date in 1970), which is never what a text answer means.
    """
    if not isinstance(value, str):
        raise ValueError(f"{value!r} is not a {what} string")
    text = value.strip()
    if _NUMERIC.fullmatch(text):
        raise ValueError(f"{value!r} is a bare number, not an ISO {what}")
    return text


# Dates and times go through pydantic rather than ``datetime.fromisoformat`` so
# that accepted formats do not depend on the Python version (3.11 widened them).


def _parse_datetime(value: Any) -> datetime:
    text = _checked_text(value, "datetime")
    try:
        result = _DATETIME.validate_python(text)
    except ValidationError:
        raise ValueError(f"{value!r} is not an ISO datetime") from None
    if result.tzinfo is None:
        return result
    return result.replace(tzinfo=_fixed_offset(result.utcoffset()))


def _parse_date(value: Any) -> date:
    text = _checked_text(value, "date")
    try:
        return _DATE.validate_python(text)
    except ValidationError:
        pass
    try:
        return _parse_datetime(text).date()
    except ValueError:
        raise ValueError(f"{value!r} is not an ISO date") from None


def _parse_time(value: Any) -> time:
    text = _checked_text(value, "time")
    try:
        result = _TIME.validate_python(text)
    except ValidationError:
        raise ValueError(f"{value!r} is not an ISO time") from None
    if result.tzinfo is None:
        return result
    return result.replace(tzinfo=_fixed_offset(result.tzinfo.utcoffset(None)))


def _timedelta(amount: float, unit: str) -> timedelta:
    try:
        return timedelta(seconds=amount * _SECONDS_PER_UNIT[unit])
    except (OverflowError, ValueError):
        raise ValueError(f"duration {amount} {unit} is out of range") from None


def _duration_text(text: str, default_unit: str) -> timedelta:
    """Parse ISO 8601, clock (``1:30:00``) or ``2 hours 30 min`` durations."""
    stripped = text.strip()
    iso = _ISO_DURATION.fullmatch(stripped)
    if iso is not None and any(iso.groupdict().values()):
        total = sum(
            float(amount) * _SECONDS_PER_UNIT[_ISO_UNITS[key]]
            for key, amount in iso.groupdict().items()
            if amount
        )
        return _timedelta(total, "seconds")
    clock = _CLOCK.fullmatch(stripped)
    if clock is not None:
        hours, minutes, seconds = clock.groups()
        total = int(hours) * 3600 + int(minutes) * 60 + float(seconds or 0)
        return _timedelta(total, "seconds")
    try:
        return _timedelta(float(_decimal(stripped)), default_unit)
    except ValueError:
        pass
    lowered = stripped.lower()
    parts = _DURATION_PART.findall(lowered)
    leftover = re.sub(r"\band\b|,", " ", _DURATION_PART.sub(" ", lowered))
    if not parts or leftover.strip():
        raise ValueError(f"{text!r} is not a duration")
    total = 0.0
    for amount, word in parts:
        unit = _UNIT_ALIASES.get(word)
        if unit is None:
            raise ValueError(f"{text!r} has an unknown duration unit {word!r}")
        total += float(amount) * _SECONDS_PER_UNIT[unit]
    return _timedelta(total, "seconds")


def _unit_of(tag: str, fallback: str) -> str:
    return tag.split(":", 1)[1] if ":" in tag else fallback


def _element(kind: str | None, value: Any) -> Any:
    """Convert one list element according to ``FieldSpec.item_kind``."""
    if value is None:
        raise ValueError("null element")
    if kind in (None, "str"):
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise ValueError(f"{value!r} is not text")
    if kind == "choice":
        if isinstance(value, str):
            return value
        raise ValueError(f"{value!r} is not text")
    if kind in ("int", "float", "decimal"):
        return _to_number(value, kind)
    if kind == "date":
        return _parse_date(value)
    if kind == "datetime":
        return _parse_datetime(value)
    if kind == "time":
        return _parse_time(value)
    if kind.startswith("timedelta"):
        unit = _unit_of(kind, "seconds")
        if isinstance(value, str):
            return _duration_text(value, unit)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _timedelta(float(value), unit)
        raise ValueError(f"{value!r} is not a duration")
    return value


# --------------------------------------------------------------------------- #
# Answer kinds
# --------------------------------------------------------------------------- #


def _list_value(
    spec: FieldSpec, value: Any
) -> tuple[list[Any], str | None, list[int], list[tuple[int, str]]]:
    """Parse list elements.

    Returns:
        The values, a note, the source index of each kept value and the
        ``(source index, message)`` of every element that did not parse.
    """
    if not isinstance(value, list):
        raise ValueError("a list answer must be an array")
    kept: list[Any] = []
    indexes: list[int] = []
    failed: list[tuple[int, str]] = []
    is_set = spec.post.startswith(("set", "frozenset"))
    for index, item in enumerate(value):
        try:
            parsed = _element(spec.item_kind, item)
        except ValueError as exc:
            failed.append((index, str(exc)))
            continue
        if is_set and parsed in kept:
            continue
        kept.append(parsed)
        indexes.append(index)
    reason = None
    if failed:
        reason = f"dropped {len(failed)} element(s): {failed[0][1]}"
    return kept, reason, indexes, failed


def _relative(value: Any, kind: str, today: date | None) -> tuple[Any, str] | None:
    """A date or datetime answer that is a relative phrase, resolved in code.

    Clairic documents ``reference_date`` as the anchor of dates written without a
    year, not of relative phrases, so an answer that hands "yesterday" or "three days
    ago" back is not an ISO date. It is read here when the phrase names one date
    whatever the convention (``_relative.resolve_phrase``) and the invoke has a
    reference date; a datetime is then that day at midnight, flagged in the note.
    """
    if today is None or not isinstance(value, str):
        return None
    found = resolve_phrase(" ".join(value.split()), today)
    if found is None:
        return None
    if kind == "datetime":
        return datetime.combine(
            found, time()
        ), "relative_date: resolved in code (midnight)"
    return found, "relative_date: resolved in code"


def _clairic_value(
    spec: FieldSpec, answer: ClairicAnswer, today: date | None = None
) -> tuple[Any, str | None]:
    assert spec.clairic is not None
    value = answer.value
    kind = spec.clairic.type
    if kind == "list":
        kept, note, _, _ = _list_value(spec, value)
        return kept, note
    if kind == "duration":
        unit = _unit_of(spec.post, spec.clairic.unit or "seconds")
        if isinstance(value, str):
            return _duration_text(value, unit), None
        return _timedelta(float(_decimal(value)), unit), None
    if kind in ("date", "datetime"):
        try:
            return (_parse_date if kind == "date" else _parse_datetime)(value), None
        except ValueError:
            relative = _relative(value, kind, today)
            if relative is None:
                raise
            return relative
    if kind == "time":
        return _parse_time(value), None
    if kind == "text":
        if isinstance(value, str):
            return value, None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), None
        raise ValueError(f"{value!r} is not text")
    if kind in ("number", "estimate"):
        if kind == "estimate" and spec.post == "int":
            return round(float(_decimal(value))), None
        return _to_number(value, spec.post), None
    return value, None


_PLACEHOLDER_EDGE = " \t\r\n*_`|."


def _folded(text: str) -> str:
    return " ".join(text.casefold().split()).strip(_PLACEHOLDER_EDGE)


def is_placeholder(value: str, placeholders: Sequence[str]) -> bool:
    """Whether a text says "no value yet" (``TBA``, ``N/A``, a dash) and is no value.

    Args:
        value: The text of a field.
        placeholders: The texts that are placeholders (``GateConfig.placeholders``).

    Returns:
        True when ``value`` is one of them, ignoring case, markdown emphasis, white
        space and a closing full stop (``"**N/A.**"`` is ``"n/a"``).
    """
    folded = _folded(value)
    return bool(folded) and any(folded == _folded(p) for p in placeholders)


def _clairic(
    spec: FieldSpec,
    answer: ClairicAnswer,
    gates: GateConfig = GateConfig(),  # noqa: B008 - frozen dataclass
    today: date | None = None,
) -> Coerced:
    spans = _spans(answer.spans)
    meta: dict[str, Any] = {
        "span": _pair(answer.span),
        "spans": spans,
        "p_null": answer.p_null,
        "confidence": answer.confidence,
        "guess": answer.guess,
    }
    if spec.clairic is None:
        return Coerced(None, False, "spec has no Clairic field", **meta)
    if answer.value is None:
        return Coerced(None, True, None, **meta)
    try:
        if spec.clairic.type == "list":
            value, note, indexes, failed = _list_value(spec, answer.value)
            meta["kept"] = tuple(indexes)
            meta["failed"] = tuple(failed)
            # Spans stay aligned with the values: they follow the elements that
            # survived, and are dropped when the model's counts already differ.
            if spans is not None and len(spans) == len(answer.value):
                meta["spans"] = tuple(spans[index] for index in indexes)
            else:
                meta["spans"] = None
        else:
            value, note = _clairic_value(spec, answer, today)
            if (
                spec.clairic.type == "text"
                and isinstance(value, str)
                and is_placeholder(value, gates.placeholders)
            ):
                # "TBA" in a serial number cell says there is none yet
                return Coerced(None, True, "placeholder", **meta)
    except ValueError as exc:
        return Coerced(None, False, str(exc), **meta)
    return Coerced(value, True, note, **meta)


def _noul(spec: FieldSpec, answer: NoulAnswer, gates: GateConfig) -> Coerced:
    if not isinstance(spec.clef, NoulQuestion):
        return Coerced(None, False, "spec does not hold a noul question")
    probability = answer.noul
    margin = gates.noul_margin
    if probability > 0.5 and probability >= 0.5 + margin - _EPS:
        return Coerced(True, score=probability)
    if probability < 0.5 and probability <= 0.5 - margin + _EPS:
        return Coerced(False, score=probability)
    return Coerced(None, False, "abstain", score=probability)


def _choice(spec: FieldSpec, answer: ChoiceAnswer) -> Coerced:
    if not isinstance(spec.clef, ChoiceQuestion) or spec.key_map is None:
        return Coerced(None, False, "spec does not hold a choice question")
    probability = answer.probabilities.get(answer.choice)
    if answer.choice not in spec.key_map:
        return Coerced(
            None,
            False,
            f"unknown option {answer.choice!r}",
            confidence=answer.confidence,
            score=probability,
        )
    value = spec.key_map[answer.choice]
    if value is UNSTATED:
        # The model says the input does not state the value. That is a null, as
        # a Clairic null is; ``prune_nulls`` then applies a field's default.
        return Coerced(
            None,
            True,
            "unstated",
            confidence=answer.confidence,
            score=probability,
        )
    return Coerced(value, confidence=answer.confidence, score=probability)


def _score(spec: FieldSpec, answer: ScoreAnswer) -> Coerced:
    if not isinstance(spec.clef, ScoreQuestion) or spec.key_map is None:
        return Coerced(None, False, "spec does not hold a score question")
    if answer.probabilities:
        best = min(
            answer.probabilities,
            key=lambda level: (-answer.probabilities[level], _level_order(level)),
        )
    else:
        best = str(round(answer.score))
    if best not in spec.key_map:
        return Coerced(
            None,
            False,
            f"unknown level {best!r}",
            confidence=answer.confidence,
            score=answer.score,
        )
    return Coerced(spec.key_map[best], confidence=answer.confidence, score=answer.score)


def _level_order(level: str) -> float:
    try:
        return float(level)
    except ValueError:
        return math.inf


def _truth(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "y", "1", "on"):
        return True
    if lowered in ("false", "no", "n", "0", "off"):
        return False
    raise ValueError(f"{text!r} is not a boolean")


def _union_value(type_name: str, text: str) -> Any:
    """Parse the Clairic text as the member type that Clef chose."""
    if type_name == "str":
        return text
    if type_name in ("int", "float", "decimal"):
        return _to_number(text, type_name)
    if type_name == "bool":
        return _truth(text)
    if type_name == "date":
        return _parse_date(text)
    if type_name == "datetime":
        return _parse_datetime(text)
    if type_name == "time":
        return _parse_time(text)
    if type_name == "duration":
        return _duration_text(text, "seconds")
    raise ValueError(f"unknown union member type {type_name!r}")


def coerce_union(
    spec: FieldSpec,
    text_answer: ClairicAnswer,
    type_answer: ChoiceAnswer,
    gates: GateConfig = GateConfig(),  # noqa: B008 - frozen dataclass
) -> Coerced:
    """Convert the two answers for a union of primitives (``int | str`` ...).

    The Clairic answer holds the value as text and the Clef answer (the spec's
    ``<field>.__type`` question) says which member type it is; the text is then
    parsed with that member's parser (``int``, ``float``, ``decimal``, ``bool``,
    ``date``, ``datetime``, ``time``, ``duration`` or ``str``). Span, ``p_null``,
    ``confidence`` and ``guess`` come from the Clairic answer and ``score`` is
    the probability of the chosen type.

    Never raises: a null text is a valid ``None``; an unknown or low-confidence
    type (below ``gates.min_choice_prob`` or ``gates.min_choice_confidence``) or
    text that does not parse as the chosen type is ``ok=False`` with a reason.

    Args:
        spec: A spec compiled from a union of primitives.
        text_answer: The Clairic answer for ``spec.wire_name``.
        type_answer: The Clef answer for ``spec.clef_name``.
        gates: Supplies the choice thresholds.

    Returns:
        The converted value plus answer metadata.
    """
    if spec.post != "union" or spec.clairic is None or spec.clef is None:
        return Coerced(None, False, "spec is not a union of primitives")
    base = _clairic(spec, text_answer, gates)
    if not base.ok or base.value is None:
        return base
    chosen = _choice(spec, type_answer)
    if not chosen.ok:
        return replace(
            base, value=None, ok=False, reason=chosen.reason, score=chosen.score
        )
    below_prob = chosen.score is not None and chosen.score < gates.min_choice_prob
    below_conf = (
        chosen.confidence is not None
        and chosen.confidence < gates.min_choice_confidence
    )
    if below_prob or below_conf:
        return replace(
            base, value=None, ok=False, reason="type_uncertain", score=chosen.score
        )
    try:
        value = _union_value(chosen.value, base.value)
    except ValueError as exc:
        return replace(base, value=None, ok=False, reason=str(exc), score=chosen.score)
    return replace(base, value=value, score=chosen.score)


def coerce(
    spec: FieldSpec,
    answer: ClairicAnswer | NoulAnswer | ChoiceAnswer | ScoreAnswer,
    gates: GateConfig = GateConfig(),  # noqa: B008 - frozen dataclass
    today: date | None = None,
) -> Coerced:
    """Convert a model answer into the Python value for a field.

    Never raises for bad model output: a value that cannot be converted, an
    answer of the wrong kind for the spec, or a noul answer inside the abstain
    band gives ``ok=False`` with a ``reason``.

    * Clairic: a null value is a valid ``None`` (``guess`` is only reported).
      Text stays ``str``; numbers become ``int`` (only when integral), ``float``
      or ``Decimal`` according to the spec; dates, times and datetimes are
      parsed from ISO 8601 (a trailing ``Z`` is accepted); durations become a
      ``timedelta`` in the spec's unit (a month is 30.436875 days and a year
      365.2425 days, so those units are lossy). Lists parse each element per
      ``item_kind``, drop elements that do not parse (noted in ``reason``) and,
      for sets, drop duplicates keeping the first occurrence (the value is still
      a list; pydantic builds the set). A ``span`` or ``spans`` is kept only if
      every pair is ``[start, end]`` with ``0 <= start <= end``.
    * A date or datetime answer that is a relative phrase naming one date ("yesterday",
      "three days ago", "in two weeks") is resolved against ``today`` (the invoke's
      reference date) when there is one, instead of failing as "not an ISO date".
    * Clef noul: yes at ``noul >= 0.5 + margin``, no at ``noul <= 0.5 - margin``,
      otherwise ``"abstain"``.
    * Clef choice: ``spec.key_map[choice]``; an unknown option is not ok.
    * Clef score: the most probable level mapped through ``spec.key_map``; the
      weighted score is kept in ``Coerced.score``.
    * A choice whose option maps to ``UNSTATED`` gives ``ok=True``,
      ``value=None`` and ``reason="unstated"``, like a Clairic null, for
      nullable and defaulted fields alike (``prune_nulls`` applies the default).
    * A union of primitives is refused here (``ok=False``); it needs both
      answers, see ``coerce_union``.

    Args:
        spec: The field the answer belongs to.
        answer: The parsed model answer.
        gates: Supplies ``noul_margin``.
        today: The reference date, for a relative-phrase answer of a date field.

    Returns:
        The converted value plus answer metadata.
    """
    if spec.post == "union":
        return Coerced(
            None, False, "a union of primitives needs two answers: use coerce_union"
        )
    if isinstance(answer, ClairicAnswer):
        return _clairic(spec, answer, gates, today)
    if isinstance(answer, NoulAnswer):
        return _noul(spec, answer, gates)
    if isinstance(answer, ChoiceAnswer):
        return _choice(spec, answer)
    if isinstance(answer, ScoreAnswer):
        return _score(spec, answer)
    return Coerced(None, False, f"unsupported answer type {type(answer).__name__}")
