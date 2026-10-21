"""Test helpers: a schema-validating fake transport and response builders."""

import copy
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from trustcall_system_one._wire import (
    ClairicRequest,
    ClefRequest,
    body_model,
    parse_clairic,
    parse_clef,
)
from trustcall_system_one.transport import CallMeta

__all__ = [
    "FakeTransport",
    "choice",
    "clairic_answer",
    "clairic_response",
    "clef_response",
    "noul",
    "score",
]

Handler = Callable[
    [str, dict[str, Any], CallMeta],
    "dict[str, Any] | Awaitable[dict[str, Any] | None] | None",
]


class FakeTransport:
    """A ``Transport`` that answers from a handler and validates both sides.

    The request is validated against the published request schema before the
    handler runs and the handler's response is validated after, so tests fail
    on malformed traffic. An invalid request raises pydantic's
    ``ValidationError``; an invalid response raises ``ModelResponseError``.

    Attributes:
        calls: Every ``(model, payload, meta)`` seen, in order. Payloads are
            deep copies taken before the handler ran.
    """

    def __init__(self, handler: Handler) -> None:
        """Create a fake transport.

        Args:
            handler: ``handler(model, payload, meta)`` returning a bare response
                body, or an awaitable of one. Returning ``None`` or raising
                ``KeyError`` marks the call as unhandled.
        """
        self._handler = handler
        self.calls: list[tuple[str, dict[str, Any], CallMeta]] = []

    @property
    def last_retries(self) -> int:
        """Always 0; the fake never retries."""
        return 0

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Validate the request, call the handler, validate and return the body.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing context handed to the handler.
            timeout: Ignored.

        Returns:
            The handler's response body.

        Raises:
            AssertionError: If the handler returns ``None`` or raises ``KeyError``.
        """
        body = copy.deepcopy(dict(payload))
        self.calls.append((model, body, meta))
        is_clairic = body_model(model) == "clairic"
        if is_clairic:
            ClairicRequest.model_validate(body)
        else:
            ClefRequest.model_validate(body)

        unhandled = AssertionError(
            f"FakeTransport: unhandled call\n  model: {model}\n  meta: {meta}\n"
            f"  payload: {body}"
        )
        try:
            result = self._handler(model, copy.deepcopy(body), meta)
            if inspect.isawaitable(result):
                result = await result
        except KeyError as exc:
            raise unhandled from exc
        if result is None:
            raise unhandled
        if is_clairic:
            parse_clairic(result)
        else:
            parse_clef(result)
        return result


def clairic_answer(
    type: str,
    value: Any,
    *,
    p_null: float = 1e-6,
    confidence: float = 0.99,
    span: Sequence[int] | None = None,
    spans: Sequence[Sequence[int]] | None = None,
    guess: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a schema-valid Clairic answer.

    Args:
        type: The field type echoed by the model.
        value: The answer value (``None`` for null).
        p_null: Probability the value is null.
        confidence: Model confidence.
        span: ``[start, end]`` of a text answer in the state.
        spans: Spans of a list answer.
        guess: The best guess when ``value`` is null.
        **extra: Further keys, such as ``mean`` or ``p10``.

    Returns:
        The answer object.
    """
    answer: dict[str, Any] = {
        "type": type,
        "value": value,
        "p_null": p_null,
        "confidence": confidence,
    }
    if span is not None:
        answer["span"] = list(span)
    if spans is not None:
        answer["spans"] = [list(s) for s in spans]
    if guess is not None:
        answer["guess"] = guess
    answer.update(extra)
    return answer


def clairic_response(
    answers: Mapping[str, dict[str, Any]],
    *,
    truncated: bool = False,
    state_text: str | None = None,
    input_tokens: int = 100,
) -> dict[str, Any]:
    """Build a schema-valid Clairic response body.

    Args:
        answers: Answers keyed by field name.
        truncated: Whether the state was truncated.
        state_text: Optional text that the spans index into.
        input_tokens: Reported input tokens.

    Returns:
        The response body.
    """
    body: dict[str, Any] = {
        "model": "clairic",
        "answers": dict(answers),
        "truncated": truncated,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
    if state_text is not None:
        body["state_text"] = state_text
    return body


def noul(p: float) -> dict[str, Any]:
    """Build a noul answer with probability ``p`` of yes."""
    return {"type": "noul", "noul": p}


def choice(
    choice: str,
    probabilities: Mapping[str, float],
    confidence: float | None = None,
) -> dict[str, Any]:
    """Build a choice answer.

    Args:
        choice: The chosen option.
        probabilities: Probability per option.
        confidence: Defaults to the highest probability.

    Returns:
        The answer object.
    """
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": dict(probabilities),
        "confidence": max(probabilities.values()) if confidence is None else confidence,
    }


def score(
    score: float,
    probabilities: Mapping[str, float],
    legend: Mapping[str, Any] | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Build a score answer.

    Args:
        score: The probability-weighted level.
        probabilities: Probability per level index.
        legend: Level index to description; defaults to empty.
        confidence: Defaults to the highest probability.

    Returns:
        The answer object.
    """
    return {
        "type": "score",
        "score": score,
        "legend": dict(legend or {}),
        "probabilities": dict(probabilities),
        "confidence": max(probabilities.values()) if confidence is None else confidence,
    }


def clef_response(
    answers: Mapping[str, dict[str, Any]],
    *,
    input_tokens: int = 100,
    model: str = "clef",
) -> dict[str, Any]:
    """Build a schema-valid Clef response body.

    Args:
        answers: Answers keyed by question id.
        input_tokens: Reported input tokens.
        model: The model name echoed in the body.

    Returns:
        The response body.
    """
    return {
        "model": model,
        "answers": dict(answers),
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
