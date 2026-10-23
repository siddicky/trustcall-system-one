"""Logit shifts of the probabilities in a Clef response body.

``CalibratedTransport`` uses these to move one provider's probabilities towards the
numbers the engine's thresholds were tuned on, without changing the thresholds. The
shift is monotone: a higher probability never ends up below a lower one.

* ``noul``: ``p' = sigmoid(logit(p) + shift)``. A positive shift makes yes likelier.
* ``choice`` and ``score``: the shift moves the logit of the *leading* option (the
  answer's ``choice``; for a score the most probable level), the other options keep
  their proportions and share what is left, so the probabilities still sum to 1. A
  positive shift makes the model surer of what it picked, a negative one less sure.
  ``choice`` is then the most probable option again (ties keep the original pick),
  ``confidence`` follows (it stays equal to the top probability when it was, else
  it is shifted itself) and a score is recomputed as the probability-weighted level.

Left alone: a probability of exactly 0 or 1, an answer flagged ``hard`` (a bare pick
of the Decisions API, capped on purpose so that it can never authorise a removal on
its own) and anything that is not a well-formed answer.
"""

import contextlib
import math
from collections.abc import Mapping
from typing import Any

__all__ = ["calibrate_body", "shift_probability"]

_SAME = 1e-6
"""How close ``confidence`` must be to the top probability to count as equal to it."""


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def shift_probability(p: float, shift: float) -> float:
    """``sigmoid(logit(p) + shift)``; 0, 1 and a zero shift are returned unchanged.

    Args:
        p: A probability.
        shift: The logit shift (positive raises ``p``).

    Returns:
        The shifted probability, in [0, 1]. Monotone in ``p`` for a fixed shift.
    """
    if shift == 0 or not 0.0 < p < 1.0:
        return p
    z = math.log(p) - math.log1p(-p) + shift
    return min(1.0, max(0.0, _sigmoid(z)))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _argmax(probs: Mapping[str, float], prefer: str | None) -> str:
    best = max(probs.values())
    if prefer is not None and probs.get(prefer) == best:
        return prefer
    return next(k for k, v in probs.items() if v == best)


def _shift_distribution(
    probs: Mapping[str, Any], top: str | None, shift: float
) -> tuple[dict[str, float], str] | None:
    """The shifted, renormalised odds and their leader, or ``None`` (unchanged)."""
    numbers = {k: _number(v) for k, v in probs.items()}
    if not numbers or any(v is None or v < 0 for v in numbers.values()):
        return None
    clean = {k: float(v) for k, v in numbers.items() if v is not None}
    total = sum(clean.values())
    if total <= 0:
        return None
    given = {k: v / total for k, v in clean.items()}
    lead = top if top in given else _argmax(given, None)
    p_lead = given[lead]
    if not 0.0 < p_lead < 1.0:
        return None
    new_lead = shift_probability(p_lead, shift)
    scale = (1.0 - new_lead) / (1.0 - p_lead)
    moved = {k: (new_lead if k == lead else v * scale) for k, v in given.items()}
    norm = sum(moved.values())
    moved = {k: v / norm for k, v in moved.items()}
    return moved, lead


def _confidence(
    answer: Mapping[str, Any], before: float, after: float, s: float
) -> Any:
    """The new ``confidence``: the top probability if it was one, else shifted."""
    given = _number(answer.get("confidence"))
    if given is None:
        return answer.get("confidence")
    if abs(given - before) <= _SAME:
        return after
    return shift_probability(given, s)


def _noul(answer: Mapping[str, Any], shift: float) -> dict[str, Any] | None:
    p = _number(answer.get("noul"))
    if p is None or not 0.0 < p < 1.0:
        return None
    return {**answer, "noul": shift_probability(p, shift)}


def _choice(answer: Mapping[str, Any], shift: float) -> dict[str, Any] | None:
    probs = answer.get("probabilities")
    if not isinstance(probs, Mapping):
        return None
    pick = answer.get("choice")
    done = _shift_distribution(probs, pick if isinstance(pick, str) else None, shift)
    if done is None:
        return None
    moved, lead = done
    raw = [v for v in (_number(x) for x in probs.values()) if v is not None]
    chosen = _argmax(moved, lead)
    return {
        **answer,
        "choice": chosen,
        "probabilities": moved,
        "confidence": _confidence(answer, max(raw), moved[chosen], shift),
    }


def _score(answer: Mapping[str, Any], shift: float) -> dict[str, Any] | None:
    probs = answer.get("probabilities")
    if not isinstance(probs, Mapping):
        return None
    done = _shift_distribution(probs, None, shift)
    if done is None:
        return None
    moved, _ = done
    raw = [v for v in (_number(x) for x in probs.values()) if v is not None]
    out = {
        **answer,
        "probabilities": moved,
        "confidence": _confidence(answer, max(raw), max(moved.values()), shift),
    }
    with contextlib.suppress(ValueError):  # levels that are not indexes: unchanged
        out["score"] = sum(int(level) * p for level, p in moved.items())
    return out


def calibrate_body(
    body: Mapping[str, Any], shift: float
) -> tuple[Mapping[str, Any], int]:
    """Shift the probabilities of every answer of a Clef response body.

    Args:
        body: A Clef response body (``{model, answers, usage}``).
        shift: The logit shift; 0 returns ``body`` itself.

    Returns:
        ``(new body, answers changed)``. The input is never modified; the body is
        returned as it came when nothing changed.
    """
    answers = body.get("answers")
    if shift == 0 or not isinstance(answers, Mapping):
        return body, 0
    changed: dict[str, Any] = {}
    for qid, answer in answers.items():
        if not isinstance(answer, Mapping) or answer.get("hard"):
            continue
        kind = answer.get("type")
        shifted = (
            _noul(answer, shift)
            if kind == "noul"
            else _choice(answer, shift)
            if kind == "choice"
            else _score(answer, shift)
            if kind == "score"
            else None
        )
        if shifted is not None:
            changed[qid] = shifted
    if not changed:
        return body, 0
    return {**body, "answers": {**answers, **changed}}, len(changed)
