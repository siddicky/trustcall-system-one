"""Test helpers: a schema-validating fake transport and response builders."""

import copy
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from trustcall_system_one._wire import (
    CLAIRIC,
    CLEF,
    CLEF_FLASH,
    ChoiceAnswer,
    ChoiceQuestion,
    ClairicRequest,
    ClairicResponse,
    ClefRequest,
    ClefResponse,
    ScoreAnswer,
    ScoreQuestion,
    body_model,
    parse_clairic,
    parse_clef,
)
from trustcall_system_one.transport import CallMeta, TransportResponse

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


_FULL_IDS = (CLAIRIC, CLEF, CLEF_FLASH)


def _unhandled(model: str, meta: CallMeta, payload: Any, why: str) -> AssertionError:
    return AssertionError(
        f"FakeTransport: {why}\n  model: {model}\n  meta: {meta}\n  payload: {payload}"
    )


def _check_clairic_response(
    request: ClairicRequest, response: ClairicResponse, model: str
) -> str | None:
    if response.model.strip() not in (body_model(model), model):
        return f"response model {response.model!r} does not match {model}"
    if set(response.answers) != set(request.fields):
        return (
            f"answer keys {sorted(response.answers)} differ from the requested "
            f"fields {sorted(request.fields)}"
        )
    return None


def _check_clef_response(
    request: ClefRequest, response: ClefResponse, model: str
) -> str | None:
    if response.model != body_model(model):
        return f"response model {response.model!r} must be {body_model(model)!r}"
    if set(response.answers) != set(request.questions):
        return (
            f"answer keys {sorted(response.answers)} differ from the asked "
            f"questions {sorted(request.questions)}"
        )
    for qid, question in request.questions.items():
        answer = response.answers[qid]
        if answer.type != question.type:
            return f"{qid}: answered as {answer.type} but asked as {question.type}"
        if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
            options = set(question.criteria)
            if answer.choice not in options:
                return (
                    f"{qid}: choice {answer.choice!r} is not one of {sorted(options)}"
                )
            if not set(answer.probabilities) <= options:
                return f"{qid}: probabilities name options that were not offered"
        if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
            levels = {str(i) for i in range(len(question.criteria))}
            if set(answer.probabilities) != levels:
                return f"{qid}: probabilities must be keyed {sorted(levels)}"
    return None


class FakeTransport:
    """A ``Transport`` that answers from a handler and validates both sides.

    It is deliberately at least as strict as the real API and
    ``CloudflareTransport``:

    * ``model`` must be a full ``@cf/...`` id, and must agree with the body's
      ``model`` selector.
    * The payload is JSON round-tripped (``allow_nan=False``, as httpx does)
      and validated in strict mode, so lax coercions and values that cannot be
      encoded fail here instead of in production.
    * The response must parse, carry exactly the requested field or question
      ids, and, for Clef, answer each question with its own type, with
      options and levels that were actually offered.

    An invalid request raises pydantic's ``ValidationError`` (or ``TypeError``
    / ``ValueError`` when it cannot be JSON-encoded), an invalid response
    raises ``ModelResponseError`` and a response inconsistent with its request
    raises ``AssertionError``.

    Attributes:
        calls: Every ``(model, payload, meta)`` seen, in order. Payloads are
            the JSON round-tripped copies taken before the handler ran.
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

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Validate the request, call the handler, validate and return the body.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing context handed to the handler.
            timeout: Ignored.

        Returns:
            The handler's response body, with ``retries=0``.

        Raises:
            ValueError: If ``model`` is not a full id of a System One model.
            AssertionError: If the handler returns ``None`` or raises
                ``KeyError``, or its response does not fit the request.
        """
        if model not in _FULL_IDS:
            raise ValueError(
                f"unknown System One model: {model!r}; FakeTransport needs a "
                f"full id, one of {_FULL_IDS}"
            )
        try:
            text = json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            self.calls.append((model, copy.deepcopy(dict(payload)), meta))
            raise
        body: dict[str, Any] = json.loads(text)
        self.calls.append((model, body, meta))

        clairic_request: ClairicRequest | None = None
        clef_request: ClefRequest | None = None
        if body_model(model) == "clairic":
            clairic_request = ClairicRequest.model_validate_json(text, strict=True)
            selector = clairic_request.model
        else:
            clef_request = ClefRequest.model_validate_json(text, strict=True)
            selector = clef_request.model
        if body_model(selector.strip()) != body_model(model):
            raise _unhandled(
                model, meta, body, f"URL model does not match body model {selector!r}"
            )

        started = time.perf_counter()
        try:
            result = self._handler(model, copy.deepcopy(body), meta)
            if inspect.isawaitable(result):
                result = await result
        except KeyError as exc:
            raise _unhandled(model, meta, body, "unhandled call") from exc
        if result is None:
            raise _unhandled(model, meta, body, "unhandled call")

        problem: str | None
        if clairic_request is not None:
            problem = _check_clairic_response(
                clairic_request, parse_clairic(result), model
            )
        else:
            assert clef_request is not None
            problem = _check_clef_response(clef_request, parse_clef(result), model)
        if problem is not None:
            raise _unhandled(
                model, meta, body, f"response does not fit the request: {problem}"
            )
        return TransportResponse(
            body=result,
            retries=0,
            duration_ms=(time.perf_counter() - started) * 1000,
            status_code=200,
        )


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
    model: Literal["clef", "clef-flash"] = "clef",
) -> dict[str, Any]:
    """Build a schema-valid Clef response body.

    Args:
        answers: Answers keyed by question id.
        input_tokens: Reported input tokens.
        model: The model echoed in the body. It must match the request's body
            model, which ``FakeTransport`` checks, so a handler serving
            ``clef-flash`` passes ``model="clef-flash"``.

    Returns:
        The response body.

    Raises:
        ValueError: If ``model`` is not ``"clef"`` or ``"clef-flash"``.
    """
    if model not in ("clef", "clef-flash"):
        raise ValueError(f"model must be 'clef' or 'clef-flash', got {model!r}")
    return {
        "model": model,
        "answers": dict(answers),
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
