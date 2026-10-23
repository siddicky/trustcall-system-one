"""Test helpers: a fake transport, ``FakeWorld`` and response builders."""

import copy
import hashlib
import inspect
import json
import os
import re
import threading
import time
import typing
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from datetime import time as dtime
from typing import Any, Literal, cast

from pydantic import BaseModel

from trustcall_system_one._coref import identity_compatible
from trustcall_system_one._errors import (
    AuthError,
    BadRequestError,
    ModelResponseError,
    RateLimitError,
    ServerError,
    TransportError,
)
from trustcall_system_one._types import CallKind
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
from trustcall_system_one.transport import (
    DEFAULT_MAX_QUESTIONS,
    CallMeta,
    Transport,
    TransportResponse,
    _can_serve,
    classify_failure,
    max_questions,
    prefers_keep_uncertain,
    state_token_budget,
)

_TYPESAFE = "typesafe"
"""``TypeSafeTransport.provider``: the label a cassette of Jev's answers carries."""

__all__ = [
    "CassetteMissError",
    "CassetteTransport",
    "EscalationHandler",
    "FakeEscalation",
    "FakeTransport",
    "FakeWorld",
    "RecordingTransport",
    "WorldEntity",
    "choice",
    "clairic_answer",
    "clairic_response",
    "clef_response",
    "noul",
    "request_key",
    "route",
    "score",
    "scrub",
    "states_relative_change",
    "trace_to_cassette",
]

Handler = Callable[
    [str, dict[str, Any], CallMeta],
    "dict[str, Any] | Awaitable[dict[str, Any] | None] | None",
]


_FULL_IDS = (CLAIRIC, CLEF, CLEF_FLASH)

EscalationHandler = Callable[[dict[str, Any], CallMeta], "dict[str, Any] | None"]
"""``handler(payload, meta)`` of a ``FakeEscalation``: the JSON the model writes."""


def route(*, default: Handler | None = None, **by_kind: Handler) -> Handler:
    """Build a ``FakeTransport`` handler that dispatches on ``CallMeta.kind``.

    Args:
        default: Handler for kinds without their own; without one, such a call is
            unhandled (``FakeTransport`` raises ``AssertionError``).
        **by_kind: Handlers keyed by call kind, for example
            ``route(presence=..., extract=..., attribution=...)``. Each has the
            ``FakeTransport`` handler signature ``(model, payload, meta)``.

    Returns:
        A handler to pass to ``FakeTransport``.

    Raises:
        ValueError: If a key is not a known call kind.
    """
    known = set(typing.get_args(CallKind))
    unknown = set(by_kind) - known
    if unknown:
        raise ValueError(
            f"unknown call kind(s) {sorted(unknown)}; known: {sorted(known)}"
        )

    def handler(
        model: str, payload: dict[str, Any], meta: CallMeta
    ) -> "dict[str, Any] | Awaitable[dict[str, Any] | None] | None":
        chosen = by_kind.get(meta.kind, default)
        if chosen is None:
            return None  # unhandled
        return chosen(model, payload, meta)

    return handler


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


class FakeEscalation:
    """A scripted stand-in for the escalation model (``OpenAIResponsesTransport``).

    It answers the escalation tier's calls from ``handler(payload, meta)``, which
    returns the JSON object the model would write (``None``: the model gave no
    answer). Like the real transport it takes a payload with ``input`` and ``schema``
    and checks that an answer has exactly the keys of the schema (strict mode). The
    default handler never answers, so an escalation configured with it changes
    nothing.

    Attributes:
        calls: Every ``(model, payload, meta)`` seen, in order.
        model: The model it stands for (``gpt-6-luna``).
        provider: ``openai``.
    """

    provider = "openai"
    max_questions = 8
    state_token_budget = 100_000

    def __init__(
        self,
        handler: EscalationHandler | None = None,
        *,
        model: str = "gpt-6-luna",
        reasoning_tokens: int | Mapping[str, int] = 0,
    ) -> None:
        """Create the stand-in.

        Args:
            handler: ``handler(payload, meta)``: the JSON the model writes, or
                ``None`` for no answer (the default: it never answers).
            model: The model it stands for.
            reasoning_tokens: The reasoning tokens each call reports (part of its
                output tokens): a number, or a mapping from the call's
                ``reasoning_effort`` to one (0 for an effort not listed).
        """
        self._handler = handler
        self.model = model
        self._reasoning = reasoning_tokens
        self.calls: list[tuple[str, dict[str, Any], CallMeta]] = []

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        """Whether this stands in for ``model``: not Clairic and not Clef."""
        return model not in ("clairic", "clef", "clef-flash", *_FULL_IDS)

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Answer one escalation call (see the class docstring)."""
        body = json.loads(json.dumps(payload, allow_nan=False))
        self.calls.append((model, body, meta))
        if meta.kind != "escalate":
            raise AssertionError(f"FakeEscalation got a {meta.kind!r} call")
        schema = body.get("schema")
        if not isinstance(schema, dict) or not isinstance(body.get("input"), str):
            raise AssertionError("an escalation payload needs 'input' and 'schema'")
        answer = self._handler(body, meta) if self._handler is not None else None
        if answer is not None and set(answer) != set(schema.get("properties", {})):
            raise AssertionError(
                f"the answer's keys {sorted(answer)} are not the schema's "
                f"{sorted(schema.get('properties', {}))} (strict mode)"
            )
        reasoning = (
            self._reasoning
            if isinstance(self._reasoning, int)
            else self._reasoning.get(str(body.get("reasoning_effort") or "none"), 0)
        )
        reply = {
            "model": model,
            "status": "completed",
            "parsed": answer,
            "text": None if answer is None else json.dumps(answer),
            "refusal": None,
            "no_answer": None if answer is not None else "scripted: no answer",
            "logprobs": None,
            "usage": {
                "input_tokens": 10 + len(body["input"]) // 4,
                "output_tokens": 5 + reasoning,
                "reasoning_tokens": reasoning,
            },
        }
        return TransportResponse(
            body=reply,
            status_code=200,
            duration_ms=1.0,
            provider="openai",
            served_model=model,
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


# --------------------------------------------------------------------------- #
# FakeWorld
# --------------------------------------------------------------------------- #

Span = tuple[int, int]
_INPUT_OPEN = "<input>\n"
_INPUT_CLOSE = "\n</input>"
_ELLIPSIS = "…"
_YES, _NO = 0.96, 0.03
_ATTR_YES, _ATTR_NO = 0.95, 0.04


_FILLER = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "onto",
        "that",
        "this",
        "these",
        "those",
        "she",
        "her",
        "hers",
        "his",
        "him",
        "they",
        "them",
        "their",
        "you",
        "your",
        "our",
        "was",
        "were",
        "are",
        "has",
        "have",
        "had",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "which",
        "will",
        "would",
        "can",
        "could",
        "not",
        "but",
        "now",
        "new",
        "also",
        "just",
        "today",
        "then",
        "than",
        "there",
        "here",
        "its",
        "said",
        "says",
        "called",
        "calls",
        "told",
        "asked",
        "wants",
        "wanted",
        "moved",
    ]
)
"""Words that carry no news by themselves, for ``FakeWorld``'s novelty check."""


def _content(text: str) -> set[str]:
    """The words and numbers of a text that can tell something new."""
    return {
        word
        for word in re.findall(r"\w+", _fold(text))
        if len(word) >= 3 and word not in _FILLER
    }


_NUMBER_WORDS = {
    word: number
    for number, word in enumerate(
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
        ]
    )
}
_NUMBER = re.compile(
    r"(?<![\w.])(?P<n>\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")(?![\w.])",
    re.IGNORECASE,
)
_RELATIVE_AFTER = re.compile(
    r"\s+(?:more|fewer|less|extra|additional)\b", re.IGNORECASE
)
_RELATIVE_WORD = re.compile(r"\b(?:double|triple|twice|halve|half|another)\b", re.I)


def states_relative_change(text: str, old: Any, new: Any) -> bool:
    """Whether a text describes a change from ``old`` to ``new``, not ``new`` itself.

    For ``FakeWorld`` and scripted examples, which have no model: the answer depends
    on the CHANGE, not on the words of the input anywhere. The new value is stated if
    it occurs as a number that is not followed by "more", "fewer", "less", "extra"
    or "additional" ("the tea towels at 5"); it is relative if it occurs only as
    such an amount ("two more") or as the difference from the old value, or if
    "double", "triple", "half" or "another" is said and ``new`` is what that gives
    from ``old``.

    Args:
        text: The input.
        old: What the record holds.
        new: What the operation would write.
    """
    if isinstance(new, bool) or not isinstance(new, (int, float)):
        return False
    if isinstance(old, bool) or not isinstance(old, (int, float)):
        return False
    relative_amounts: list[float] = []
    for found in _NUMBER.finditer(text):
        raw = found.group("n").lower()
        value = float(_NUMBER_WORDS[raw]) if raw in _NUMBER_WORDS else float(raw)
        if _RELATIVE_AFTER.match(text, found.end()):
            relative_amounts.append(value)
        elif value == new:
            return False  # stated outright
    if any(v == new or v == abs(new - old) for v in relative_amounts):
        return True
    if _RELATIVE_WORD.search(text):
        return new in (old * 2, old * 3, old / 2, old + 1) if old else False
    return False


def _fold(value: Any) -> str:
    """Compare-ready text: case and spacing folded; dates, bools and numbers plain."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date, dtime)):
        return value.isoformat()
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return " ".join(str(value).casefold().split())


def _slug(value: Any) -> str:
    return re.sub(r"\W+", "_", _fold(value)).strip("_")


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Dotted-path leaves of a nested value (the whole value is kept as well)."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if prefix:
        out[prefix] = value
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)


def _clairic_text(value: Any) -> Any:
    if isinstance(value, (datetime, date, dtime)):
        return value.isoformat()
    return value


@dataclass
class WorldEntity:
    """An entity of a ``FakeWorld``, or something that only looks like one.

    Attributes:
        label: What it is called.
        schema: Name of the schema it belongs to.
        mentions: The mention specs it was declared with.
        fields: The field values it was declared with.
        real: ``True`` for an entity, ``False`` for a non-entity.
        matches: The existing record it is (see ``FakeWorld.entity``), or ``None``
            for an entity that is new.
        odds: Declared match probabilities (see ``FakeWorld.entity``).
        odds_confidence: The confidence of the declared match answer.
        confirm: What the pairwise ``match_confirm`` answers about this entity, when
            declared (a Clef that says yes to a new person: 0.75).
        descriptive: Mentions that only a broadened discovery finds (see
            ``FakeWorld.entity``).
        flat: The field values by dotted path.
        spans: Where the text mentions it (known once the world is used).
        hidden: The part of ``spans`` that are descriptive mentions.
        parent: For an item of a nested list, the label of the entity it belongs
            to (``None`` for an entity of a top-level schema).
    """

    label: str
    schema: str
    mentions: tuple[tuple[str, int | None], ...]
    fields: dict[str, Any]
    real: bool
    matches: Any = None
    odds: dict[str, float] | None = None
    odds_confidence: float | None = None
    confirm: float | None = None
    descriptive: tuple[tuple[str, int | None], ...] = ()
    flat: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    hidden: list[Span] = field(default_factory=list)
    parent: str | None = None


def _pfold(label: str | None) -> str | None:
    """A parent's label as compared: folded, without a leading article."""
    if label is None:
        return None
    return re.sub(r"^the\s+", "", _fold(label))


class _View:
    """Where the input sits in a rendered Clairic state, and back.

    Mirrors what the library renders: a full input, or windows of it joined by
    ``…`` lines. Pieces are located in the world's text with ``str.find``.
    """

    def __init__(self, state: str, source: str) -> None:
        start = state.find(_INPUT_OPEN)
        end = state.rfind(_INPUT_CLOSE)
        if start < 0 or end < 0:
            raise AssertionError("FakeWorld: the request's state has no <input> block")
        start += len(_INPUT_OPEN)
        block = state[start:end]
        self.segments: list[tuple[int, int, int]] = []
        if block == source:
            self.segments.append((start, end, 0))
        elif (at := source.find(block)) >= 0 and block:
            self.segments.append((start, end, at))
        else:
            body, offset, cursor = block, start, 0
            if body.startswith(_ELLIPSIS + "\n"):
                body, offset = body[len(_ELLIPSIS) + 1 :], offset + len(_ELLIPSIS) + 1
            if body.endswith("\n" + _ELLIPSIS):
                body = body[: -len(_ELLIPSIS) - 1]
            for piece in body.split(f"\n{_ELLIPSIS}\n"):
                at = source.find(piece, cursor)
                if at < 0:
                    raise AssertionError(
                        "FakeWorld: the text of the world is not the input of the "
                        "request"
                    )
                self.segments.append((offset, offset + len(piece), at))
                cursor = at + len(piece)
                offset += len(piece) + len(_ELLIPSIS) + 2
        self.state = state

    def to_state(self, span: Span) -> Span | None:
        for s0, s1, original in self.segments:
            length = s1 - s0
            if original <= span[0] and span[1] <= original + length:
                return (s0 + span[0] - original, s0 + span[1] - original)
        return None


class FakeWorld:
    """A declarative ground truth that answers every request the library makes.

    Declare the text, the entities in it (with their mentions and field values)
    and the things that merely look like entities; ``transport()`` then gives a
    ``FakeTransport`` whose handler answers EVERY call kind consistently from
    that truth, so a test states what the text contains and asserts what the
    library made of it. Faults (a value that bleeds onto the wrong entity, an
    answer the model is unsure of) are injected on top.

    Faults are injected with ``bleed`` (a fact offered for the wrong entity),
    ``low_confidence`` (an answer the model is unsure of), ``coref`` (doubt that
    two mention texts are one entity), ``count`` (a miscount) and ``noise``
    (discovery that lists pronouns, articles, possessives, an item without a span or
    items out of order); ``handler`` is the handler behind ``transport()`` for a
    test that overrides one kind of answer.

    Declaring::

        world = FakeWorld("Emma Thompson is my friend. Michael Chen is my boss.")
        world.entity(
            "Emma Thompson",
            schema=Person,
            mentions=["Emma Thompson"],
            fields={"name": "Emma Thompson", "relationship": "my friend"},
        )
        world.non_entity("Sunny", schema=Person, mentions=["Sunny"])  # a dog
        world.bleed("Michael Chen", "notes", "taken up rock climbing")
        extractor = create_extractor([Many(Person)], transport=world.transport())

    A mention given as a plain string stands for every whole-word occurrence of
    it in the text, except occurrences inside a longer mention of any entity
    ("Emma" inside "Emma Thompson"); ``(text, n)`` picks the ``n``-th occurrence
    only, which is how two people who are both called "Emma" are told apart.
    Text values of ``fields`` must be verbatim pieces of the text (Clairic only
    quotes); a value that is not raises an ``AssertionError`` when asked for.
    A ``fields`` key is a dotted path (``"address.city"``) or a nested dict; a
    value may be a model instance (for a union choice).

    How the focused entity is found: for Clairic the ``<focus ... entity="...">``
    header (``schema`` and ``entity`` attributes and the ``a-b`` offsets of its
    ``is mentioned in the input at characters`` line), for Clef ``state.entity``
    (``label`` and the ``mentions`` with a ``span`` or, where the input was
    narrowed, a ``context``).
    The label comes first (the entity's label or one of its mention texts); the
    mention spans settle which of several entities share it, and a label that
    names one entity while the spans belong to another is an ``AssertionError``, as
    is a shared label with nothing to tell the entities apart. Without a label the
    answer is "the only entity of ``CallMeta.schema``" (the ``One`` path has no
    focus). With ``check_state`` (the default) a Clef request about an entity whose
    input holds none of the entity's mentions is an ``AssertionError`` too.

    Nested lists and mappings: an item of a ``list[Model]`` / ``dict[str, X]``
    field is an entity declared with ``parent="<label of the entity that holds
    it>"``::

        world.entity(
            "Rex",
            schema=Pet,
            parent="Emma Thompson",
            mentions=["Rex"],
            fields={"name": "Rex"},
        )

    The library scopes every request about the items to their parent (the focus
    header's ``parent="..."`` attribute; ``state.entity.parent`` or ``state.parent``
    of a Clef state), and the world answers it for the items declared with that
    parent only. The parent is named as the library names it: the label of an
    entity is the canonical label of its mentions, a ``One`` schema's parent is
    ``"the <Title>"`` (a leading "the" does not matter here). A mapping entry is an
    item whose fields are ``key`` and ``value`` (a model's fields below it:
    ``{"key": "mugs", "value": {"quantity": 3}}``) of the schema
    ``<Parent><Field>Entry`` (the value model's own name for a model value).
    ``bleed_item(parent, item)`` offers another parent's item in discovery;
    ``delete(id, p, parent=...)`` declares a removal (an item's id is its index in
    the list, or the key of a mapping entry).

    The request shapes every call kind is answered from (the shapes the library
    sends; ``CallMeta.kind`` selects the answer):

    * ``presence`` (Clef): question id ``presence``: 0.97 if the world has an
      entity of ``meta.schema``, else 0.03; ids ``<group>.__presence`` (optional
      sub-models): 0.97 if the focused entity has fields below ``<group>``.
    * ``discover`` (Clairic ``list`` field ``mentions``): every mention occurrence
      of the schema's entities AND non-entities, in order, with spans in the
      rendered state found with ``str.find`` over the input block (windows of a
      narrowed state included) and mapped from the world's text. A broadened
      discovery (its instruction allows "descriptive references") also lists the
      ``descriptive=`` mentions that the plain discovery leaves out.
    * ``count`` (Clairic ``number`` field ``count``): the number of entities of
      the schema, or the ``count()`` override.
    * ``coref`` (Clef): ``pair.m{i}.m{j}`` over ``state.mentions`` (each with a
      ``span``) is 0.96 if both mentions belong to one world entity, else 0.03;
      ``assign.m{k}`` (choice over ``c{n}`` clusters, each described once in
      ``state.clusters.c{n}.mentions`` with spans, plus ``__new__``; the mention is
      ``state.mentions.m{k}``, a request holds a batch of them) picks the cluster of
      the same entity or ``__new__``; ``merge`` over ``state.a`` and ``state.b`` is
      0.96 only if EVERY mention on both sides belongs to one entity, else 0.03.
    * ``qualify`` (Clef): ``qualify.e{n}`` over ``state.entities.e{n}`` (``label``
      and ``mentions`` with spans): 0.97 for an entity, 0.03 for a non-entity.
    * ``extract`` / ``crosscheck`` / ``reask`` / ``repair`` (Clairic, one field
      per wire name): the focused entity's value for that field; text and list
      answers carry spans (the occurrence of the value nearest to the entity's
      mentions); a missing field is a confident null; ``bleed`` values are added
      to ``extract`` and ``crosscheck`` answers only; ``low_confidence`` lowers the
      ``extract`` and ``crosscheck`` confidence of one field.
    * ``clef_fields`` and ``repair`` (Clef): ``noul`` is 0.95 for a truthy value,
      ``choice`` picks the option that names the entity's value (an ``unstated``
      option when it has none; ``<field>.__type`` picks by Python type;
      ``<field>.el{i}`` picks the option an element of a list is), ``score`` puts
      its mass on the value as a level.
    * ``attribution`` (Clef): ``attr.<wire>[.<i>]`` quotes the value in its
      instruction; 0.95 if the value belongs to the focused entity in the world
      (any of its field values or list elements), else 0.04.
    * ``null_check`` (``nullcheck.<wire>``) and ``stated`` (``stated.<wire>``):
      0.95 if the focused entity has the field, else 0.05.
    * ``adjudicate`` (Clef choice ``a``/``b``/``neither``): the option whose text
      is the entity's value, else ``neither``.
    * ``list_dedup`` (``dedup.<i>``): 0.95 if the two quoted texts are the same
      after folding, else 0.05.
    * ``sentences`` (``sentence.<i>``, one per candidate sentence of a ``list[str]``
      field read by sentences; ``CallMeta.field`` is the field's wire name): 0.95
      if the quoted sentence contains (after folding) one of the focused entity's
      values for that field, else 0.04. A ``bleed`` for the field counts as a value
      here (a model that says yes to a sentence about someone else; attribution
      then refuses it).
    * ``list_kind`` (Clef choice ``kind``: ``items`` or ``statements``): the world's
      values of that field (any entity of the schema) are ``statements`` at 0.9 if
      one has four words or more, else ``items``.
    * ``record_presence`` (``record.<i>``, one per existing record whose identity has
      a word in the input and that no qualified entity fits): 0.97 if a real entity of
      the schema has a name or mention that fits (``identity_compatible``) one of the
      record's text values, else 0.03.
    * ``novelty`` (``novelty.<i>``, one per sentence appended to a note field of an
      update; the state holds the record after the update's other operations): 0.95
      unless the record already holds the very sentence (0.05), or the sentence
      contains a text declared with ``redundant(text, p)``, which answers ``p``.
    * ``gate`` (Clef ``gate``): 0.9 if ``proposed_change.to`` is new compared to
      ``record`` (not equal to what the record holds at the path, nor an element
      of the list it is appended to), else 0.05.
    * ``relative_edit`` (``relative``, one per replacement of an ordered quantity in
      an update): 0.95 if the input describes the change from ``proposed_change.from``
      to ``proposed_change.to`` instead of giving the new value (``states_relative_
      change``: "two more" for 3 to 2, not "the tea towels at 5" for 4 to 5), else
      0.04.
    * ``match_confirm`` (``confirm``, asked only when the choice left an entity
      undecided and exactly one record is a strong name fit): the entity's
      ``confirm=`` if declared (a gullible Clef), else 0.95 if its ``matches=``
      describes the record in the state and no ``match_odds=`` was declared, else
      0.05.
    * ``match`` (Clef choice, question id ``match``): the state is ``{input, entity:
      {label, mentions}}`` (the focused entity is found as for every Clef request)
      and the options are ``rec_<i>``, whose criteria is the existing record as a
      JSON object, plus ``__new__``. The entity's ``matches=`` picks the option: the
      record it describes gets 0.92 and the other options share the rest; an entity
      without ``matches=`` gets ``__new__`` at 0.92. A ``matches=`` that fits no
      offered record, or several, is an ``AssertionError`` (see ``entity``).
      ``match_odds=`` declares the probabilities instead (and ``match_confidence=``
      the confidence), for the ambiguous paths.
    * ``delete`` (Clef ``noul``): up to 64 questions ``delete.r<k>`` per request, over
      the state ``{input, records: {r<k>: <record>, ...}}``; ``CallMeta.field`` holds
      the ids of the records of the request, comma-joined, in question order. The
      probability is the one declared with ``delete(record_id, p)``, else 0.05.
    """

    def __init__(self, text: str, *, check_state: bool = True) -> None:
        """Start a world.

        Args:
            text: The input text every request will carry.
            check_state: Make the world fail, with an ``AssertionError``, when a
                Clef request about a focused entity carries an input that holds
                none of the entity's mentions (a state the engine narrowed away
                from the very entity it asks about). On by default.
        """
        self.text = text
        self.check_state = check_state
        self._noise: dict[str, bool] = {}
        self._items: list[WorldEntity] = []
        self._counts: dict[str, int] = {}
        self._doubts: list[tuple[str, str, float]] = []
        self._bleeds: list[tuple[str, str, Any]] = []
        self._low: list[tuple[str, str, float, bool]] = []
        self._deletes: dict[tuple[str | None, str], float] = {}
        self._item_bleeds: list[tuple[str | None, str]] = []
        self._redundant: list[tuple[str, float]] = []
        self._qualifies: list[tuple[str, float]] = []
        self._presence_of: dict[str, float] = {}
        self._ready = False
        self._transport: FakeTransport | None = None

    # -- declaring -------------------------------------------------------- #

    @staticmethod
    def _name(schema: Any) -> str:
        return schema if isinstance(schema, str) else str(schema.__name__)

    def _add(
        self,
        label: str,
        schema: Any,
        mentions: Sequence[str | tuple[str, int]],
        fields: Mapping[str, Any] | None,
        *,
        real: bool,
        matches: Any = None,
        match_odds: Mapping[str, float] | None = None,
        match_confidence: float | None = None,
        descriptive: Sequence[str | tuple[str, int]] = (),
        parent: "str | WorldEntity | None" = None,
    ) -> WorldEntity:
        def specs(
            items: Sequence[str | tuple[str, int]],
        ) -> tuple[tuple[str, int | None], ...]:
            found: list[tuple[str, int | None]] = []
            for item in items:
                text, nth = (item, None) if isinstance(item, str) else item
                if text not in self.text:
                    raise ValueError(
                        f"the mention {text!r} of {label!r} is not in the text"
                    )
                found.append((text, nth))
            return tuple(found)

        element = WorldEntity(
            label=label,
            schema=self._name(schema),
            mentions=specs(mentions),
            fields=dict(fields or {}),
            real=real,
            matches=matches,
            descriptive=specs(descriptive),
            parent=None if parent is None else self._label(parent),
        )
        element.odds = None if match_odds is None else dict(match_odds)
        element.odds_confidence = match_confidence
        for key, value in element.fields.items():
            _flatten(key, value, element.flat)
        self._items.append(element)
        self._ready = False
        return element

    def entity(
        self,
        label: str,
        *,
        schema: Any,
        mentions: Sequence[str | tuple[str, int]],
        fields: Mapping[str, Any] | None = None,
        matches: Any = None,
        match_odds: Mapping[str, float] | None = None,
        match_confidence: float | None = None,
        descriptive: Sequence[str | tuple[str, int]] = (),
        parent: "str | WorldEntity | None" = None,
        confirm: float | None = None,
    ) -> "WorldEntity":
        """Declare an entity of a schema that the text really contains.

        Args:
            label: What the entity is called (``"Emma Thompson"``); also how
                ``bleed``, ``low_confidence`` and ``count`` refer to it.
            schema: The schema (class or its name) the entity belongs to.
            mentions: The surface forms the text uses (see the class docstring).
            fields: The true value of each field, by dotted path.
            matches: The existing record this entity is, for the ``match`` calls
                (which show a record's content, never its id): a pydantic model
                or a dict (every key it names must equal the record's), or a
                string that occurs in the record's JSON text ("Emma"). Left out
                (``None``), the entity is new: ``__new__`` is its match.
            match_odds: The probabilities a ``match`` call gives instead of the
                0.92 of ``matches``, to exercise ambiguous and low-confidence
                matching without overriding the handler: each key is a string that
                occurs in exactly one record's JSON text ("Emma Thompson"), or
                ``"__new__"``, and each value its probability; the options left
                out share the rest equally. The chosen option is the likeliest.
            match_confidence: The answer's confidence (default the likeliest
                option's probability).
            descriptive: Mentions of the entity that only a broadened discovery
                (one that allows descriptive references such as "my cousin") finds;
                the first, plain discovery does not list them.
            parent: For an item of a nested list (``list[Pet]`` of a ``Person``),
                the entity it belongs to (the ``WorldEntity`` that ``entity``
                returned, or its label; ``"Emma Thompson"``; the
                label of a ``One`` parent is ``"the Order"``, and a leading "the"
                does not matter). Requests about the items of one parent see only
                the items declared with it. For a mapping, declare the entry's
                ``key`` and ``value`` as its fields.
            confirm: The probability the pairwise ``match_confirm`` question gives
                for this entity and ANY record, true or not: a gullible Clef that
                says 0.75 to a new "Emma" next to the record "Emma Thompson". Left
                out, it answers from what the world declares: 0.95 for the record
                ``matches`` describes, else 0.05.

        Returns:
            The declared entity (pass it, or its label, to ``bleed`` and
            ``low_confidence``).

        Raises:
            ValueError: If a mention is not in the text.
        """
        element = self._add(
            label,
            schema,
            mentions,
            fields,
            real=True,
            matches=matches,
            match_odds=match_odds,
            match_confidence=match_confidence,
            descriptive=descriptive,
            parent=parent,
        )
        element.confirm = confirm
        return element

    def non_entity(
        self,
        label: str,
        *,
        schema: Any,
        mentions: Sequence[str | tuple[str, int]],
        parent: "str | WorldEntity | None" = None,
    ) -> "WorldEntity":
        """Declare something that looks like an entity of a schema and is not.

        Discovery finds its mentions (a model is easily fooled) and qualification
        says no: a dog called Sunny is not a ``Person``.
        """
        return self._add(label, schema, mentions, None, real=False, parent=parent)

    def bleed_item(
        self, parent: "str | WorldEntity", item: "WorldEntity | str"
    ) -> "FakeWorld":
        """Inject a fault: discovery for ``parent`` also offers another parent's item.

        A model asked for Michael's pets that lists Emma's dog too. Qualification
        ("is this a Pet belonging to Michael?") says no, so the item is skipped
        as ``not_qualified`` and never reaches Michael's record.
        """
        self._item_bleeds.append((_pfold(self._label(parent)), self._label(item)))
        return self

    def bleed(
        self, entity: "WorldEntity | str", field_: str, value: Any
    ) -> "FakeWorld":
        """Inject a fault: offer ``value`` for an entity's field.

        The value is offered in every ``extract`` and ``crosscheck`` answer for that
        entity's field (replacing a scalar, appended to a list), as if the model
        had picked up a fact stated about someone else. If it does not belong to
        the entity in the world, attribution refuses it.
        """
        self._bleeds.append((self._label(entity), field_, value))
        return self

    def low_confidence(
        self,
        entity: "WorldEntity | str",
        field_: str,
        confidence: float,
        *,
        persistent: bool = False,
    ) -> "FakeWorld":
        """Inject a fault: the model is unsure of one field's first answer.

        ``extract`` and ``crosscheck`` answers for the field carry ``confidence``;
        a re-ask (``reask``) is confident and gives the true value, unless
        ``persistent``: then the model is as unsure of the re-ask (Clairic's list
        type on an open-ended field: confidence 1e-6 to 1e-23 however it is asked).
        """
        self._low.append((self._label(entity), field_, confidence, persistent))
        return self

    def delete(
        self,
        record_id: str,
        p: float = 0.97,
        *,
        parent: "str | WorldEntity | None" = None,
    ) -> "FakeWorld":
        """Declare that the text says an existing record should be removed.

        ``delete`` calls about the record with this id answer ``p`` instead of the
        default 0.05 (a ``delete`` request names the ids of its records in
        ``CallMeta.field``, comma-joined, in question order). For an item of a
        nested list the id is its index in the list (a mapping entry's key) and
        ``parent`` is the label of the entity whose list it is.
        """
        self._deletes[
            (None if parent is None else _pfold(self._label(parent)), record_id)
        ] = p
        return self

    def coref(self, first: str, second: str, p: float) -> "FakeWorld":
        """Inject a fault: pairwise coreference answers ``p`` for two mention texts.

        Every pair of one mention with the text ``first`` and one with ``second``
        (in either order) gets ``p`` instead of 0.96 or 0.03: a model that is
        unsure whether "Emma" is "Emma Thompson".
        """
        self._doubts.append((_fold(first), _fold(second), p))
        return self

    def noise(
        self,
        *,
        pronouns: bool = False,
        determiners: bool = False,
        possessives: bool = False,
        missing_span: bool = False,
        shuffle: bool = False,
    ) -> "FakeWorld":
        """Make discovery answers messy, the way a real model's can be.

        Args:
            pronouns: Also list the pronouns of the text ("she", "them").
            determiners: Start a mention with the article before it ("the Smiths").
            possessives: End a mention with the possessive after it ("Sarah's").
            missing_span: Give the second item no span.
            shuffle: List the items out of order.
        """
        self._noise = {
            "pronouns": pronouns,
            "determiners": determiners,
            "possessives": possessives,
            "missing_span": missing_span,
            "shuffle": shuffle,
        }
        return self

    def redundant(self, text: str, p: float = 0.05) -> "FakeWorld":
        """Inject a fault: a note containing ``text`` adds nothing to its record.

        The ``novelty`` check answers ``p`` (default 0.05) for a sentence that
        contains ``text`` (case and accents ignored): "Her mobile is now ..." when the
        phone is being written to its own field. Without this the check judges the
        sentence by its words: it is news (0.95) when it has a word or number (of
        three letters or more, not a filler such as "the" or "she") that the record
        it is shown does not hold, else it repeats the record (0.05).
        """
        self._redundant.append((_fold(text), p))
        return self

    def count(self, schema: Any, n: int) -> "FakeWorld":
        """Override the count Clairic gives for a schema (a miscounting model)."""
        self._counts[self._name(schema)] = n
        return self

    @staticmethod
    def _label(entity: "WorldEntity | str") -> str:
        return entity if isinstance(entity, str) else entity.label

    # -- the text --------------------------------------------------------- #

    def _finalise(self) -> None:
        """Locate every mention occurrence; a mention inside a longer one is dropped."""
        if self._ready:
            return
        found: list[tuple[Span, WorldEntity]] = []
        hidden: set[Span] = set()
        for element in self._items:
            for plain, specs in (
                (True, element.mentions),
                (False, element.descriptive),
            ):
                for text, nth in specs:
                    hits = [
                        m.span()
                        for m in re.finditer(
                            rf"(?<!\w){re.escape(text)}(?!\w)", self.text
                        )
                    ]
                    if nth is not None:
                        hits = hits[nth : nth + 1]
                    found.extend((span, element) for span in hits)
                    if not plain:
                        hidden.update(hits)
        keep: list[tuple[Span, WorldEntity]] = []
        for span, element in found:
            inside = any(
                other != span
                and other[0] <= span[0]
                and span[1] <= other[1]
                and (other[1] - other[0]) > (span[1] - span[0])
                for other, _ in found
            )
            if not inside and all(span != k[0] for k in keep):
                keep.append((span, element))
        for element in self._items:
            element.spans = sorted(s for s, e in keep if e is element)
            element.hidden = [s for s in element.spans if s in hidden]
        self._ready = True

    def _scoped(
        self, items: Sequence[WorldEntity], parent: str | None
    ) -> list[WorldEntity]:
        """The elements that belong to a parent (to no parent: the top-level ones)."""
        if parent is None:
            top = [e for e in items if e.parent is None]
            return top or list(items)
        want = _pfold(parent)
        return [e for e in items if e.parent is not None and _pfold(e.parent) == want]

    def _occurrences(
        self, schema: str, *, broadened: bool = False, parent: str | None = None
    ) -> list[tuple[Span, WorldEntity]]:
        self._finalise()
        pool = self._scoped([e for e in self._items if e.schema == schema], parent)
        extra = [
            e
            for who, label in self._item_bleeds
            if parent is not None and who == _pfold(parent)
            for e in self._items
            if e.schema == schema and _fold(e.label) == _fold(label) and e not in pool
        ]
        found = [
            (span, e)
            for e in [*pool, *extra]
            for span in e.spans
            if broadened or span not in e.hidden
        ]
        return sorted(found, key=lambda item: item[0])

    def _owner(self, span: Span, text: str = "") -> WorldEntity | None:
        """The world element a mention span belongs to (by overlap, else by text)."""
        self._finalise()
        best: tuple[int, WorldEntity] | None = None
        for element in self._items:
            for other in element.spans:
                overlap = min(span[1], other[1]) - max(span[0], other[0])
                if overlap > 0 and (best is None or overlap > best[0]):
                    best = (overlap, element)
        if best is not None:
            return best[1]
        if text:
            for element in self._items:
                if any(_fold(text) == _fold(m) for m, _ in element.mentions):
                    return element
        return None

    def _from_context(self, entry: Mapping[str, Any]) -> Span | None:
        """Where a mention without a span is, found from the context around it."""
        context, text = str(entry.get("context", "")), str(entry.get("text", ""))
        at = self.text.find(context) if context else -1
        if at < 0 or text not in context:
            return None
        middle = (len(context) - len(text)) / 2
        found = [m.start() for m in re.finditer(re.escape(text), context)]
        start = at + min(found, key=lambda i: abs(i - middle))
        return (start, start + len(text))

    def _resolve(
        self,
        schema: str,
        label: str | None,
        spans: Sequence[Span],
        parent: str | None = None,
    ) -> WorldEntity | None:
        """The focused element: by label, settled by spans, else the only entity.

        A label that names an element and spans that belong to another one are a
        bug in whoever built the request: that fails loudly. The label and the
        "only entity" are looked for among the items of the request's parent.
        """
        self._finalise()
        pool = self._scoped([e for e in self._items if e.schema == schema], parent)
        named: list[WorldEntity] = []
        if label:
            named = [e for e in pool if _fold(e.label) == _fold(label)] or [
                e
                for e in pool
                if any(
                    _fold(label) == _fold(m) for m, _ in (*e.mentions, *e.descriptive)
                )
            ]
        owner: WorldEntity | None = None
        for span in spans:
            found = self._owner(span)
            if found is not None and found.schema == schema:
                owner = found
                break
        if owner is not None:
            if named and owner not in named:
                raise AssertionError(
                    f"FakeWorld: the request is about {label!r} but its mention "
                    f"spans {list(spans)} belong to {owner.label!r}"
                )
            return owner
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            raise AssertionError(
                f"FakeWorld: {len(named)} entities are called {label!r} and the "
                "request carries no mention spans to tell them apart"
            )
        real = [e for e in pool if e.real]
        if len(real) == 1 and not (label and len(pool) > 1):
            return real[0]
        return None

    # -- the focused entity of a request ----------------------------------- #

    @staticmethod
    def _parent(payload: dict[str, Any]) -> str | None:
        """The parent a request is about: the focus header's, or the state's."""
        state = payload["state"]
        if isinstance(state, str):
            found = re.search(r'<focus[^>]*\bparent="([^"]*)"', state)
            return found.group(1) if found else None
        if not isinstance(state, dict):
            return None
        entity = state.get("entity")
        holder = entity.get("parent") if isinstance(entity, dict) else None
        if holder is None:
            holder = state.get("parent")
        return str(holder["label"]) if isinstance(holder, dict) else None

    def _focus(self, payload: dict[str, Any], meta: CallMeta) -> WorldEntity | None:
        state = payload["state"]
        parent = self._parent(payload)
        if isinstance(state, str):
            found = re.search(r'<focus[^>]*\bentity="([^"]*)"', state)
            label = found.group(1) if found else None
            line = re.search(
                r"entity is mentioned in the input at characters ([^\n]*)", state
            )
            cited = [
                (int(a), int(b))
                for a, b in re.findall(r"(\d+)-(\d+)", line.group(1) if line else "")
            ]
            return self._resolve(meta.schema, label, cited, parent)
        entity = state.get("entity") if isinstance(state, dict) else None
        if not isinstance(entity, dict):
            return self._resolve(meta.schema, None, [], parent)
        spans: list[Span] = []
        for m in entity.get("mentions", []):
            if not isinstance(m, dict):
                continue
            if "span" in m:
                spans.append((m["span"][0], m["span"][1]))
            elif (placed := self._from_context(m)) is not None:
                spans.append(placed)
        element = self._resolve(meta.schema, entity.get("label"), spans, parent)
        if entity.get("mentions"):
            self._check_mentions(element, str(state.get("input", "")))
        return element

    def _check_mentions(self, element: WorldEntity | None, shown: str) -> None:
        """Fail if a Clef input holds none of the entity's mentions."""
        if not self.check_state or element is None:
            return
        self._finalise()
        texts = {self.text[a:b] for a, b in element.spans}
        if texts and not any(text in shown for text in texts):
            raise AssertionError(
                f"FakeWorld: the Clef state about {element.label!r} does not "
                f"contain any of its mentions ({sorted(texts)}): the input was "
                "narrowed away from the entity the question is about"
            )

    @staticmethod
    def _keys(wire: str) -> list[str]:
        """The field keys a wire name may stand for (a union member may sit inside)."""
        parts = wire.split(".")
        return [wire] + [
            ".".join(parts[:i] + parts[i + 1 :]) for i in range(1, len(parts) - 1)
        ]

    @staticmethod
    def _lookup(element: WorldEntity, wire: str) -> tuple[bool, Any]:
        """The entity's value for a wire name."""
        for key in FakeWorld._keys(wire):
            if key in element.flat:
                return True, element.flat[key]
        return False, None

    def _value(
        self, element: WorldEntity, wire: str, *, bleed: bool
    ) -> tuple[bool, Any]:
        """The value to answer for a field, with the injected bleed when asked."""
        present, value = self._lookup(element, wire)
        if bleed:
            keys = self._keys(wire)
            for label, path, bled in self._bleeds:
                if _fold(label) != _fold(element.label) or path not in keys:
                    continue
                grow = present and isinstance(value, list)
                value = [*value, bled] if grow else bled
                present = True
        return present, value

    # -- answering --------------------------------------------------------- #

    @property
    def handler(self) -> Handler:
        """The ``FakeTransport`` handler behind ``transport()``.

        Wrap it to override one kind of answer, for example
        ``FakeTransport(route(default=world.handler, presence=my_presence))``.
        """
        return self._handle

    def transport(self) -> FakeTransport:
        """The fake transport that answers every request from this world."""
        if self._transport is None:
            self._transport = FakeTransport(self._handle)
        return self._transport

    def _handle(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        method = {
            "presence": self._presence,
            "discover": self._discover,
            "count": self._count,
            "coref": self._coref,
            "qualify": self._qualify,
            "extract": self._extract,
            "crosscheck": self._extract,
            "reask": self._extract,
            "repair": self._repair,
            "clef_fields": self._clef_fields,
            "attribution": self._attribution,
            "null_check": self._has_field,
            "stated": self._has_field,
            "adjudicate": self._adjudicate,
            "list_dedup": self._dedup,
            "sentences": self._sentences,
            "list_kind": self._list_kind,
            "novelty": self._novelty,
            "record_presence": self._record_presence,
            "gate": self._gate,
            "relative_edit": self._relative_edit,
            "match": self._match,
            "match_confirm": self._match_confirm,
            "delete": self._delete,
        }.get(meta.kind)
        if method is None:
            raise AssertionError(f"FakeWorld has no answer for {meta.kind!r} calls")
        return method(model, payload, meta)

    # presence ------------------------------------------------------------- #

    def _presence(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        answers = {}
        for qid in payload["questions"]:
            if qid.endswith(".__presence"):
                element = self._focus(payload, meta)
                group = qid[: -len(".__presence")]
                here = element is not None and any(
                    key == group or key.startswith(group + ".") for key in element.flat
                )
                answers[qid] = noul(0.97 if here else 0.03)
            else:
                pool = self._scoped(
                    [e for e in self._items if e.schema == meta.schema],
                    self._parent(payload),
                )
                real = any(e.real for e in pool)
                answers[qid] = noul(0.97 if real else 0.03)
        return clef_response(answers, model=payload["model"])

    # discovery ------------------------------------------------------------ #

    def _view(self, payload: dict[str, Any]) -> _View:
        return _View(payload["state"], self.text)

    def _discover(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        view = self._view(payload)
        values: list[str] = []
        spans: list[list[int]] = []
        wide = any(
            "descriptive references" in str(spec.get("instructions", ""))
            for spec in payload["fields"].values()
        )
        found = [
            span
            for span, _ in self._occurrences(
                meta.schema, broadened=wide, parent=self._parent(payload)
            )
        ]
        found = self._noisy(found)
        for index, span in enumerate(found):
            mapped = view.to_state(span)
            if mapped is not None:
                values.append(self.text[span[0] : span[1]])
                gap = self._noise.get("missing_span") and index == 1
                spans.append([] if gap else list(mapped))
        answer = clairic_answer(
            "list", values, spans=spans, p_null=0.01, confidence=0.93
        )
        return clairic_response(
            {name: answer for name in payload["fields"]}, state_text=payload["state"]
        )

    def _noisy(self, spans: list[Span]) -> list[Span]:
        """The listed spans after the noise of ``noise()``."""
        noise = self._noise
        out: list[Span] = []
        for start, end in spans:
            if noise.get("determiners"):
                before = re.search(r"\b(the|a|an|this|that) $", self.text[:start], re.I)
                if before:
                    start -= len(before.group(0))
            if noise.get("possessives") and re.match(
                "['\u2019]s", self.text[end : end + 2]
            ):
                end += 2
            out.append((start, end))
        if noise.get("pronouns"):
            for m in re.finditer(
                r"\b(she|he|they|her|him|them|his|hers|their)\b", self.text, re.I
            ):
                if not any(a < m.end() and m.start() < b for a, b in out):
                    out.append(m.span())
        out.sort()
        if noise.get("shuffle"):
            out.reverse()
        return out

    def _count(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        pool = self._scoped(
            [e for e in self._items if e.schema == meta.schema], self._parent(payload)
        )
        n = self._counts.get(meta.schema, sum(e.real for e in pool))
        answer = clairic_answer("number", n, p_null=0.01, confidence=0.9)
        return clairic_response(
            {name: answer for name in payload["fields"]}, state_text=payload["state"]
        )

    # coreference and qualification ------------------------------------------ #

    def _owner_of_entry(self, entry: Mapping[str, Any]) -> Any:
        span = entry.get("span")
        owner = (
            self._owner((span[0], span[1]), str(entry.get("text", "")))
            if span
            else self._owner((-1, -1), str(entry.get("text", "")))
        )
        return owner if owner is not None else ("unknown", entry.get("text"))

    def _coref(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        state = payload["state"]
        answers: dict[str, Any] = {}
        for qid, question in payload["questions"].items():
            if qid.startswith("pair."):
                first, second = qid.split(".")[1:3]
                one, other = state["mentions"][first], state["mentions"][second]
                same = self._owner_of_entry(one) is self._owner_of_entry(other)
                answers[qid] = noul(_YES if same else _NO)
                pair = {_fold(one["text"]), _fold(other["text"])}
                for a, b, p in self._doubts:
                    if pair == {a, b}:
                        answers[qid] = noul(p)
            elif qid.startswith("assign."):
                owner = self._owner_of_entry(
                    state["mentions"][qid.removeprefix("assign.")]
                )
                options = dict(question["criteria"])
                chosen = "__new__"
                for option in options:
                    if option == "__new__":
                        continue
                    held = state["clusters"][option]["mentions"]
                    if any(self._owner_of_entry(m) is owner for m in held):
                        chosen = option
                        break
                answers[qid] = _pick(chosen, list(options), 0.95)
            elif qid == "merge":
                owners = [
                    self._owner_of_entry(m)
                    for side in ("a", "b")
                    for m in state[side]["mentions"]
                ]
                same = all(owner is owners[0] for owner in owners)
                answers[qid] = noul(_YES if same else _NO)
            else:
                raise AssertionError(f"FakeWorld: unknown coref question {qid!r}")
        return clef_response(answers, model=payload["model"])

    def _qualify(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        state = payload["state"]
        parent = self._parent(payload)
        answers = {}
        for qid in payload["questions"]:
            entry = state["entities"][qid.removeprefix("qualify.")]
            spans = [
                (m["span"][0], m["span"][1]) for m in entry["mentions"] if "span" in m
            ]
            element = self._resolve(meta.schema, entry["label"], spans, parent)
            self._check_mentions(element, str(state.get("input", "")))
            real = element is not None and element.real
            if real and element is not None and parent is not None:
                # an item of somebody else's list is not one of this parent's
                real = element.parent is not None and _pfold(element.parent) == _pfold(
                    parent
                )
            answers[qid] = noul(0.97 if real else 0.03)
        return clef_response(answers, model=payload["model"])

    # extraction ----------------------------------------------------------- #

    def _place(self, view: _View, element: WorldEntity, text: str) -> Span | None:
        """The span, in the state, of the occurrence of ``text`` nearest the entity."""
        at = [m.span() for m in re.finditer(re.escape(text), self.text)]
        if not at:
            raise AssertionError(
                f"FakeWorld: {text!r} (a value of {element.label!r}) is not in the "
                "text; Clairic only quotes the input"
            )
        self._finalise()
        inside = [(span, view.to_state(span)) for span in at]
        inside = [(span, mapped) for span, mapped in inside if mapped is not None]
        if not inside:
            return None

        def distance(item: tuple[Span, Span | None]) -> int:
            if not element.spans:
                return item[0][0]
            return min(abs(item[0][0] - s[0]) for s in element.spans)

        return min(inside, key=distance)[1]

    def _clairic_answer(
        self,
        spec: Mapping[str, Any],
        element: WorldEntity | None,
        wire: str,
        view: _View,
        *,
        primary: bool,
    ) -> dict[str, Any]:
        kind = spec["type"]
        null = clairic_answer(kind, None, p_null=0.99, confidence=0.95)
        if element is None:
            return null
        present, value = self._value(element, wire, bleed=primary)
        if not present or value is None:
            return null
        confidence = 0.99
        for label, path, c, persistent in self._low:
            if (primary or persistent) and (
                _fold(label) == _fold(element.label) and path in self._keys(wire)
            ):
                confidence = c
        if kind == "list":
            values = value if isinstance(value, list) else [value]
            texts = [_clairic_text(v) for v in values]
            if spec.get("items") == "text":
                placed = [(t, self._place(view, element, str(t))) for t in texts]
                kept = [(t, p) for t, p in placed if p is not None]
                if not kept:
                    return null
                return clairic_answer(
                    "list",
                    [t for t, _ in kept],
                    spans=[list(p) for _, p in kept if p is not None],
                    confidence=confidence,
                )
            return clairic_answer("list", texts, confidence=confidence)
        if kind == "text":
            span = self._place(view, element, str(value))
            if span is None:
                return null
            return clairic_answer(
                "text", str(value), span=list(span), confidence=confidence
            )
        if kind in ("date", "time", "datetime"):
            return clairic_answer(kind, _clairic_text(value), confidence=confidence)
        if kind == "number":
            try:
                number: Any = float(value) if not isinstance(value, bool) else value
            except (TypeError, ValueError):
                return null
            if isinstance(number, float) and number == int(number):
                number = int(number)
            return clairic_answer("number", number, confidence=confidence)
        return null

    def _extract(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        element = self._focus(payload, meta)
        view = self._view(payload)
        primary = meta.kind in ("extract", "crosscheck")
        answers = {
            wire: self._clairic_answer(spec, element, wire, view, primary=primary)
            for wire, spec in payload["fields"].items()
        }
        return clairic_response(answers, state_text=payload["state"])

    def _repair(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        if "fields" in payload:
            return self._extract(model, payload, meta)
        return self._clef_fields(model, payload, meta)

    # Clef-typed fields ----------------------------------------------------- #

    @staticmethod
    def _candidates(value: Any) -> list[str]:
        if isinstance(value, bool):
            return ["yes", "true"] if value else ["no", "false"]
        if isinstance(value, BaseModel):
            return [type(value).__name__, *FakeWorld._candidates(value.model_dump())]
        if isinstance(value, Mapping):
            return [str(v) for v in value.values() if isinstance(v, (str, bool))]
        return [str(value)]

    def _answer_choice(
        self, question: Mapping[str, Any], present: bool, value: Any
    ) -> dict[str, Any]:
        options = dict(question["criteria"])
        chosen: str | None = None
        if present and value is not None:
            wanted = [_slug(c) for c in self._candidates(value)]
            for option in options:
                if _slug(option) in wanted:
                    chosen = option
                    break
            if chosen is None:
                for option, text in options.items():
                    if any(
                        _slug(text) == w or _slug(str(text)[:40]) == w for w in wanted
                    ):
                        chosen = option
                        break
            if chosen is None:
                for option, text in options.items():
                    if any(w and w in _slug(text) for w in wanted):
                        chosen = option
                        break
        if chosen is None:
            chosen = next(
                (o for o in options if o.startswith("unstated")),
                next((o for o in options if o.startswith("none_of_these")), None)
                or next(iter(options)),
            )
        return _pick(chosen, list(options), 0.95)

    def _clef_fields(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        element = self._focus(payload, meta)
        answers: dict[str, Any] = {}
        for qid, question in payload["questions"].items():
            kind = question["type"]
            if qid.endswith(".__type"):
                wire = qid[: -len(".__type")]
                present, value = (
                    self._lookup(element, wire) if element else (False, None)
                )
                names = {
                    int: "int",
                    float: "float",
                    str: "str",
                    bool: "bool",
                }
                shown = names.get(type(value), type(value).__name__)
                answers[qid] = self._answer_choice(
                    question, present, shown if present else None
                )
                continue
            element_choice = re.match(r"(.*)\.el(\d+)$", qid)
            if element_choice and kind == "choice":
                asked = re.search(
                    r'Which option is "(.*)" in the input', question["instructions"]
                )
                text = asked.group(1) if asked else ""
                options = dict(question["criteria"])
                chosen = next(
                    (o for o, t in options.items() if _fold(t) == _fold(text)), None
                )
                if chosen is None:
                    chosen = next(o for o in options if o.startswith("none_of_these"))
                answers[qid] = _pick(chosen, list(options), 0.95)
                continue
            present, value = self._lookup(element, qid) if element else (False, None)
            if kind == "noul":
                answers[qid] = noul(0.95 if present and value else 0.05)
            elif kind == "choice":
                answers[qid] = self._answer_choice(question, present, value)
            else:
                levels = len(question["criteria"])
                level = 0
                if present and isinstance(value, (int, float)):
                    level = max(0, min(levels - 1, int(value)))
                probs = {
                    str(i): (0.8 if i == level else 0.2 / (levels - 1))
                    for i in range(levels)
                }
                legend = {str(i): str(t) for i, t in enumerate(question["criteria"])}
                answers[qid] = score(float(level), probs, legend, confidence=0.8)
        return clef_response(answers, model=payload["model"])

    # verification ---------------------------------------------------------- #

    def _belongs(self, element: WorldEntity | None, text: str) -> bool:
        if element is None:
            return False
        wanted = _fold(text)
        for value in element.flat.values():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, (Mapping, BaseModel)):
                    continue
                if _fold(item) == wanted:
                    return True
        return False

    def _attribution(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        element = self._focus(payload, meta)
        answers = {}
        for qid, question in payload["questions"].items():
            found = re.search(
                r'(?:\bis|\bbe|:) "(.*)"'
                r"(?:\?\s*$|, or does it follow\b| (?:one of the|an?)\b)",
                question["instructions"],
                re.S,
            )
            if found is None:  # "Does `input` ask for 2 of `entity.label` (as ...)?"
                found = re.search(
                    r"ask for (.*?) of `entity\.label`", question["instructions"]
                )
            text = found.group(1) if found else ""
            answers[qid] = noul(_ATTR_YES if self._belongs(element, text) else _ATTR_NO)
        return clef_response(answers, model=payload["model"])

    def _has_field(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        element = self._focus(payload, meta)
        answers = {}
        for qid in payload["questions"]:
            wire = qid.split(".", 1)[1] if "." in qid else qid
            present = element is not None and self._lookup(element, wire)[0]
            answers[qid] = noul(0.95 if present else 0.05)
        return clef_response(answers, model=payload["model"])

    def _adjudicate(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        element = self._focus(payload, meta)
        answers = {}
        for qid, question in payload["questions"].items():
            options = dict(question["criteria"])
            wire = meta.field or qid.removeprefix("adjudicate.")
            present, value = self._lookup(element, wire) if element else (False, None)
            chosen = "neither" if "neither" in options else next(iter(options))
            if present:
                held = {
                    _fold(v) for v in (value if isinstance(value, list) else [value])
                }
                for option, text in options.items():
                    if _fold(text) in held:
                        chosen = option
                        break
            answers[qid] = _pick(chosen, list(options), 0.9)
        return clef_response(answers, model=payload["model"])

    def _list_kind(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        wire = meta.field or ""
        values: list[Any] = []
        for item in self._items:
            if item.schema != meta.schema:
                continue
            present, value = self._lookup(item, wire)
            if present:
                values.extend(value if isinstance(value, list) else [value])
        statements = any(len(str(v).split()) >= 4 for v in values)
        options = {"items": "items", "statements": "statements"}
        answers = {
            qid: _pick("statements" if statements else "items", list(options), 0.9)
            for qid in payload["questions"]
        }
        return clef_response(answers, model=payload["model"])

    def _sentences(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        element = self._focus(payload, meta)
        present, value = (
            self._value(element, meta.field or "", bleed=True)
            if element is not None
            else (False, None)
        )
        held = [
            _fold(v)
            for v in (value if isinstance(value, list) else [value])
            if present and v is not None
        ]
        answers = {}
        for qid, question in payload["questions"].items():
            found = re.search(r': "(.*)"\?\s*$', question["instructions"], re.S)
            text = _fold(found.group(1)) if found else ""
            hit = any(v and v in text for v in held)
            answers[qid] = noul(0.95 if hit else 0.04)
        return clef_response(answers, model=payload["model"])

    def _record_presence(
        self, model: str, payload: dict[str, Any], meta: CallMeta
    ) -> Any:
        record = payload["state"].get("record") or {}
        values = [v for v in record.values() if isinstance(v, str)]
        hit = any(
            e.real
            and e.schema == meta.schema
            and any(
                identity_compatible(name, value)
                for name in [e.label, *(text for text, _ in e.mentions)]
                for value in values
            )
            for e in self._items
        )
        return clef_response(
            {qid: noul(0.97 if hit else 0.03) for qid in payload["questions"]},
            model=payload["model"],
        )

    def _novelty(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        answers = {}
        for qid, question in payload["questions"].items():
            found = re.search(
                r'^Does the sentence "(.*)" say anything',
                question["instructions"],
                re.S,
            )
            text = _fold(found.group(1)) if found else ""
            record = payload["state"].get("record")
            held = _fold(json.dumps(record, ensure_ascii=False))
            known = _content(held)
            p = 0.95
            if text and f'"{text}"' in held:
                p = 0.05  # the record holds this very sentence already
            elif text and not (_content(text) - known):
                p = 0.05  # every word of it is in the record: nothing new
            for part, odds in self._redundant:
                if part in text:
                    p = odds
            answers[qid] = noul(p)
        return clef_response(answers, model=payload["model"])

    def _dedup(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        answers = {}
        for qid, question in payload["questions"].items():
            found = re.search(
                r'Is "(.*)" the same fact as "(.*)", as far as',
                question["instructions"],
                re.S,
            )
            same = found is not None and _fold(found.group(1)) == _fold(found.group(2))
            answers[qid] = noul(0.95 if same else 0.05)
        return clef_response(answers, model=payload["model"])

    @staticmethod
    def _held(record: Any, change: Mapping[str, Any]) -> bool:
        """Whether the record already holds what a proposed change would write."""
        target = change.get("to")
        parts = [p for p in str(change.get("path", "")).split("/") if p]
        node: Any = record
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            if last and part == "-":
                return isinstance(node, list) and any(
                    _fold(item) == _fold(target) for item in node
                )
            if isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            elif isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return False
        if isinstance(node, list) and not isinstance(target, list):
            return any(_fold(item) == _fold(target) for item in node)
        return _fold(node) == _fold(target)

    def _gate(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        state = payload["state"]
        held = self._held(state.get("record"), state.get("proposed_change", {}))
        return clef_response(
            {qid: noul(0.05 if held else 0.9) for qid in payload["questions"]},
            model=payload["model"],
        )

    def _relative_edit(
        self, model: str, payload: dict[str, Any], meta: CallMeta
    ) -> Any:
        state = payload["state"]
        change = state.get("proposed_change") or {}
        relative = states_relative_change(
            str(state.get("input", "")), change.get("from"), change.get("to")
        )
        return clef_response(
            {qid: noul(0.95 if relative else 0.04) for qid in payload["questions"]},
            model=payload["model"],
        )

    # matching and deletes --------------------------------------------------- #

    @staticmethod
    def _is_record(spec: Any, record: Any) -> bool:
        """Whether a ``matches=`` declaration describes a record."""
        if isinstance(spec, BaseModel):
            spec = spec.model_dump(mode="json")
        if isinstance(spec, Mapping):
            return isinstance(record, Mapping) and all(
                key in record and _fold(record[key]) == _fold(value)
                for key, value in spec.items()
            )
        return _fold(spec) in _fold(json.dumps(record, ensure_ascii=False))

    def _match_confirm(
        self, model: str, payload: dict[str, Any], meta: CallMeta
    ) -> Any:
        element = self._focus(payload, meta)
        record = payload["state"].get("record")
        if element is not None and element.confirm is not None:
            p = element.confirm
        else:
            same = (
                element is not None
                and element.matches is not None
                and element.odds is None  # a declared doubt is a doubt here too
                and self._is_record(element.matches, record)
            )
            p = 0.95 if same else 0.05
        return clef_response(
            {qid: noul(p) for qid in payload["questions"]}, model=payload["model"]
        )

    def _match(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        element = self._focus(payload, meta)
        answers: dict[str, Any] = {}
        for qid, question in payload["questions"].items():
            options = dict(question["criteria"])
            chosen = "__new__"
            if element is not None and element.odds is not None:
                answers[qid] = self._odds_answer(element, options)
                continue
            if element is not None and element.matches is not None:
                found = [
                    option
                    for option, record in options.items()
                    if option != "__new__" and self._is_record(element.matches, record)
                ]
                if len(found) != 1:
                    raise AssertionError(
                        f"FakeWorld: matches={element.matches!r} of "
                        f"{element.label!r} fits {len(found)} of the "
                        f"{len(options) - 1} records offered (it must fit exactly "
                        "one; was it cut by the prefilter?)"
                    )
                chosen = found[0]
            answers[qid] = _pick(chosen, list(options), 0.92)
        return clef_response(answers, model=payload["model"])

    def _odds_answer(
        self, element: WorldEntity, options: Mapping[str, Any]
    ) -> dict[str, Any]:
        """A match answer with the probabilities the entity was declared with."""
        assert element.odds is not None
        probs: dict[str, float] = {}
        for spec, p in element.odds.items():
            if spec == "__new__":
                probs["__new__"] = p
                continue
            found = [
                option
                for option, record in options.items()
                if option != "__new__" and self._is_record(spec, record)
            ]
            if len(found) != 1:
                raise AssertionError(
                    f"FakeWorld: match_odds key {spec!r} of {element.label!r} fits "
                    f"{len(found)} of the {len(options) - 1} records offered (it "
                    "must fit exactly one)"
                )
            probs[found[0]] = p
        rest = [o for o in options if o not in probs]
        left = max(0.0, 1.0 - sum(probs.values()))
        for option in rest:
            probs[option] = left / len(rest)
        best = max(probs, key=lambda option: probs[option])
        return choice(best, {o: probs[o] for o in options}, element.odds_confidence)

    def _delete(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        ids = (meta.field or "").split(",")
        questions = list(payload["questions"])
        state = payload["state"]
        if len(ids) != len(questions) or not all(
            qid.removeprefix("delete.") in state.get("records", {}) for qid in questions
        ):
            raise AssertionError(
                f"FakeWorld: a delete request asks {questions} about "
                f"{sorted(state.get('records', {}))} but CallMeta.field "
                f"{meta.field!r} holds {len(ids)} record id(s)"
            )
        parent = _pfold(self._parent(payload))
        answers = {
            qid: noul(self._deletes.get((parent, rid), 0.05))
            for qid, rid in zip(questions, ids, strict=True)
        }
        return clef_response(answers, model=payload["model"])


def _pick(option: str, options: Sequence[str], p: float) -> dict[str, Any]:
    """A choice answer: ``option`` gets ``p``, the others share the rest."""
    rest = (1 - p) / max(1, len(options) - 1)
    return choice(option, {o: (p if o == option else rest) for o in options})


# --------------------------------------------------------------------------- #
# Recording and replaying real traffic
# --------------------------------------------------------------------------- #

_SECRET_ENV = (
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_AUTH_TOKEN",
    "EXPERIENTIAL_API_KEY",
    "TYPESAFE_API_KEY",
    "OPENAI_API_KEY",
)
_REDACTED = "<redacted>"
_MIN_SECRET = 16
"""Shortest exact value that counts as a credential (an account id has 32
characters, a token 40 or more); a shorter "secret" is a word, and rewriting a word
would alter content that has to replay as it was."""


def request_key(model: str, payload: Mapping[str, Any]) -> str:
    """The stable key a recorded call is stored and found under.

    A hash of the model id and the canonical JSON of the payload (sorted keys,
    no insignificant whitespace), so the same request always has the same key
    whatever order its dictionaries were built in, and any change to the request
    (a different state, field, instruction or threshold-driven option) is a
    different key.

    Args:
        model: Full ``@cf/...`` model id.
        payload: The request body.

    Returns:
        A 24-character hexadecimal string.
    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(f"{model}\n{canonical}".encode()).hexdigest()[:24]


def _secrets(extra: Iterable[str] = ()) -> list[str]:
    """Exact credential values: the Cloudflare environment variables and ``extra``."""
    values = [os.environ.get(name, "") for name in _SECRET_ENV]
    return [v for v in (*values, *extra) if len(v) >= _MIN_SECRET]


_API_ACCOUNT = re.compile(
    r"(?<=api\.cloudflare\.com/client/v4/accounts/)[^/\s\"'?#]+", re.IGNORECASE
)
_GATEWAY_PATH = re.compile(
    r"(?<=gateway\.ai\.cloudflare\.com/v1/)[^/\s\"'?#]+", re.IGNORECASE
)
_EXPERIENTIAL_KEY = re.compile(r"\bxpl_[A-Za-z0-9_-]{16,}")
_TYPESAFE_KEY = re.compile(r"\bapikey_[A-Za-z0-9_-]{16,}")
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")
_ACCOUNT_ID = re.compile(r"(?<=accounts/)[0-9a-fA-F]{32}(?![0-9A-Za-z])")
_BEARER_HEADER = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)bearer\s+[^\s\"',}]+"
)
_BEARER_TOKEN = re.compile(
    r"(?i)\bBearer\s+(?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{20,}"
)


def scrub(text: str, extra: Iterable[str] = ()) -> str:
    """Remove credentials and account ids from text that is about to be written.

    For error messages, reports and other text that is not replayed as data: the
    exact values of the Cloudflare environment variables, of
    ``EXPERIENTIAL_API_KEY``, ``TYPESAFE_API_KEY``, ``OPENAI_API_KEY`` and of
    ``extra`` (for example
    ``CloudflareTransport.sensitive_values()`` or
    ``ExperientialTransport.sensitive_values()`` or ``TypeSafeTransport``), any
    ``xpl_`` Experiential Labs key, any ``apikey_`` TypeSafe key, any ``sk-`` OpenAI key
    (``sk-proj-...`` too), the account id in a
    Cloudflare API URL (``api.cloudflare.com/client/v4/accounts/<id>``), an AI
    Gateway URL (``gateway.ai.cloudflare.com/v1/<id>/``) or an ``accounts/<32 hex
    characters>`` path, and the bearer token of an ``Authorization`` header (or a
    token-shaped ``Bearer`` value of 20 characters with a digit in it) are replaced
    by ``<redacted>``.

    Everything else is left alone: ``accounts/receivable`` is a department, "bearer
    instruments" are paper, and a short value is a word. A request or a response
    that is about to be recorded is never passed through this (a cassette must
    replay what the model said): ``RecordingTransport`` skips a call that holds an
    exact credential instead.

    Args:
        text: The text.
        extra: Further secret values to replace (those of 16 characters or more).

    Returns:
        The text without them.
    """
    for secret in _secrets(extra):
        text = text.replace(secret, _REDACTED)
    text = _API_ACCOUNT.sub(_REDACTED, text)
    text = _GATEWAY_PATH.sub(_REDACTED, text)
    text = _ACCOUNT_ID.sub(_REDACTED, text)
    text = _EXPERIENTIAL_KEY.sub(_REDACTED, text)
    text = _TYPESAFE_KEY.sub(_REDACTED, text)
    text = _OPENAI_KEY.sub(_REDACTED, text)
    text = _BEARER_HEADER.sub(rf"\1Bearer {_REDACTED}", text)
    return _BEARER_TOKEN.sub(f"Bearer {_REDACTED}", text)


class CassetteMissError(AssertionError):
    """A call has no recording in the cassette.

    Attributes:
        kind: The call kind (``CallMeta.kind``).
        field: The field the call was about (``CallMeta.field``), if any.
        schema: The schema (``CallMeta.schema``).
        model: The full model id.
        key: The key the request hashes to (``request_key``).
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        field: str | None,
        schema: str,
        model: str,
        key: str,
    ) -> None:
        """Create the error.

        Args:
            message: What went wrong.
            kind: The call kind.
            field: The field the call was about, if any.
            schema: The schema.
            model: The full model id.
            key: The request key.
        """
        super().__init__(message)
        self.kind = kind
        self.field = field
        self.schema = schema
        self.model = model
        self.key = key


class RecordingTransport:
    """A ``Transport`` that passes calls to another one and writes them down.

    One JSON line is appended per call to ``path``::

        {
            "key",
            "model",
            "kind",
            "scope",
            "schema",
            "field",
            "request",
            "response",
            "status",
            "duration_ms",
            "provider",
            "served_model",
            "keep_uncertain",
        }

    ``provider`` and ``served_model`` say who answered (``typesafe`` and the Jev
    version, say) when the transport knows, so a replay accounts the call under the
    model that answered and a report says what a cassette was recorded on; a
    cassette without them replays as it always did. ``keep_uncertain`` is what the
    wrapped transport says of the call (``prefers_keep_uncertain(model, kind)``): a
    replay settles ``RepairConfig.keep_uncertain`` from it, so a cassette of Jev's
    answers replays with the setting on and one of Cloudflare's with it off.

    ``key`` is ``request_key(model, request)``. ``request`` is the payload that was
    sent and ``response`` the bare response body. A Clef line also carries
    ``max_questions``, the question limit of the wrapped transport, so that a replay
    chunks its questions the same way (``CassetteTransport.max_questions``). A call
    that fails is re-raised and, with ``record_errors``, also written with
    ``response: null`` and an ``error`` object. Failures are not recorded by default:
    a transient 429 or a network error replayed later would poison every run that
    uses the cassette.

    Only payloads and bodies are written: never headers, tokens or account ids
    (the wrapped transport owns those). As a second line of defence a call whose
    request or response holds the exact value of ``CLOUDFLARE_ACCOUNT_ID``,
    ``CLOUDFLARE_API_TOKEN``, ``CLOUDFLARE_AUTH_TOKEN``, ``EXPERIENTIAL_API_KEY``,
    ``TYPESAFE_API_KEY`` or of the credentials of the wrapped ``CloudflareTransport``,
    ``ExperientialTransport`` or ``RoutingTransport`` (``sensitive_values()``; 16
    characters or more) is NOT written, with a ``RuntimeWarning``: a cassette
    replays exactly what was asked and answered, so it is never edited. The message
    of a recorded error is scrubbed (see ``scrub``).

    Attributes:
        path: The cassette file (its directory is created).
        written: Lines written so far by this object.
    """

    def __init__(
        self,
        inner: Transport,
        path: str | os.PathLike[str],
        *,
        append: bool = True,
        record_errors: bool = False,
    ) -> None:
        """Wrap a transport.

        Args:
            inner: The transport that answers.
            path: The JSONL file to write.
            append: Append to an existing file; ``False`` starts it afresh.
            record_errors: Also write the calls that failed.
        """
        self._inner = inner
        self._record_errors = record_errors
        self.path = os.fspath(path)
        self.written = 0
        self._lock = threading.Lock()
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not append:
            with open(self.path, "w", encoding="utf-8"):
                pass

    def max_questions(self, model: str) -> int:
        """Questions one Clef request may carry: the wrapped transport's limit.

        Recording against Experiential (32) must chunk by 32, not by the default
        64 that a transport without a limit gets.
        """
        return max_questions(self._inner, model)

    def state_token_budget(self, model: str) -> int | None:
        """The state budget of the wrapped transport (``None``: it says none)."""
        return state_token_budget(self._inner, model)

    def prefers_keep_uncertain(self, model: str, kind: str | None = None) -> bool:
        """The wrapped transport's preference (the ``keep_uncertain`` default)."""
        return prefers_keep_uncertain(self._inner, model, kind)

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        """Whether the wrapped transport can serve the request (a router asks)."""
        return _can_serve(self._inner, model, payload)

    def classify_failure(self, exc: Exception) -> str | None:
        """The wrapped transport's category of a failure."""
        classify = getattr(self._inner, "classify_failure", None)
        if callable(classify):
            return cast("str | None", classify(exc))
        return classify_failure(exc)

    def sensitive_values(self) -> tuple[str, ...]:
        """The credentials passed explicitly to the wrapped transport."""
        sensitive = getattr(self._inner, "sensitive_values", None)
        return tuple(sensitive()) if callable(sensitive) else ()

    def _write(self, line: Mapping[str, Any]) -> None:
        sensitive = getattr(self._inner, "sensitive_values", None)
        extra = tuple(sensitive()) if callable(sensitive) else ()
        bodies = json.dumps(
            [line.get("request"), line.get("response")],
            ensure_ascii=False,
            allow_nan=False,
        )
        if any(secret in bodies for secret in _secrets(extra)):
            # A cassette replays what the model said and what was asked, so it is
            # never edited: a call that holds a credential is not recorded at all.
            warnings.warn(
                f"RecordingTransport: a {line.get('kind')} call holds a credential "
                "in its request or response and was not recorded",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        written = dict(line)
        error = written.get("error")
        if isinstance(error, Mapping):  # error text is not replayed as data
            written["error"] = {
                **error,
                "message": scrub(str(error.get("message", "")), extra),
            }
        text = json.dumps(written, ensure_ascii=False, allow_nan=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            self.written += 1

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run the call on the wrapped transport and record it.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing context.
            timeout: Passed on.

        Returns:
            What the wrapped transport returned.
        """
        request = json.loads(json.dumps(payload, allow_nan=False))
        head = {
            "key": request_key(model, request),
            "model": model,
            "kind": meta.kind,
            "scope": meta.scope,
            "schema": meta.schema,
            "field": meta.field,
            "request": request,
        }
        # what the wrapped transport says of this call: a replay settles
        # ``RepairConfig.keep_uncertain`` from it (``CassetteTransport``)
        head["keep_uncertain"] = prefers_keep_uncertain(self._inner, model, meta.kind)
        if "questions" in request:  # a Clef request: the limit it was chunked by
            head["max_questions"] = self.max_questions(model)
            budget = self.state_token_budget(model)
            if budget is not None:  # and the size its state was narrowed to
                head["state_token_budget"] = budget
        started = time.perf_counter()
        try:
            result = await self._inner.run(model, payload, meta=meta, timeout=timeout)
        except Exception as exc:
            if self._record_errors:
                self._write(
                    {
                        **head,
                        "response": None,
                        "status": getattr(exc, "status_code", None),
                        "duration_ms": (time.perf_counter() - started) * 1000,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
            raise
        line = {
            **head,
            "response": result.body,
            "status": result.status_code,
            "duration_ms": result.duration_ms or (time.perf_counter() - started) * 1000,
        }
        if result.provider:
            line["provider"] = result.provider
        if result.served_model:
            line["served_model"] = result.served_model
        if result.warnings:  # what the transport changed in the request
            line["warnings"] = list(result.warnings)
        self._write(line)
        return result


_ERRORS: dict[str, type[Exception]] = {
    cls.__name__: cls
    for cls in (
        AuthError,
        BadRequestError,
        ModelResponseError,
        RateLimitError,
        ServerError,
        TransportError,
    )
}


def _no_usage() -> dict[str, float]:
    return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}


class CassetteTransport:
    """A ``Transport`` that replays recorded calls by request key.

    The cassette is one JSONL file as ``RecordingTransport`` writes it, a
    directory of ``*.jsonl`` files, or a list of either. A call is looked up under
    ``request_key(model, payload)``. Several recordings of one key (the same
    request made more than once) replay in the order they were recorded.

    A call with no recording (or one more call than recordings) raises
    ``CassetteMissError`` naming the kind and field, unless ``strict`` is false
    and an ``inner`` transport is given: the call then goes to it, and only that
    call. Wrap the inner transport in a ``RecordingTransport`` to keep what it
    answers. A recorded failure is replayed as that failure only when there is no
    inner transport to ask; with one it is skipped and the request is forwarded
    (a 429 recorded once must not fail every later run).

    Attributes:
        hits: Calls answered from the cassette.
        misses: ``(model, payload, meta)`` of each call that was not found
            (whether it then fell through or raised).
        forwarded: Calls the inner transport answered.
        replayed: What the calls answered from the cassette used (``requests``,
            ``input_tokens``, ``output_tokens`` and the ``cost`` they reported): a
            replay spends none of it, which a cost estimate must not count.
    """

    def __init__(
        self,
        path: "str | os.PathLike[str] | Iterable[str | os.PathLike[str]]",
        *,
        strict: bool = True,
        inner: Transport | None = None,
        provider: str | None = None,
    ) -> None:
        """Load a cassette.

        Args:
            path: A JSONL file, a directory of them, or an iterable of those.
            strict: Raise on a miss. When false, a miss goes to ``inner``.
            inner: Where a miss goes when ``strict`` is false. Without one, a
                miss raises even when not strict.
            provider: Replay only what this provider recorded (the ``provider`` of
                the recording): a request is a miss when only another provider (or
                a provider-less, older recording) answered it. For a cassette that
                stands in for ONE provider of a routed stack, so that a spec that
                routes a call elsewhere never gets another provider's answer.

        Raises:
            FileNotFoundError: If a path does not exist.
            ValueError: If a line is not a recorded call.
        """
        self.strict = strict
        self._inner = inner
        self._provider = provider
        self._entries: dict[str, list[dict[str, Any]]] = {}
        self._cursor: dict[str, int] = {}
        self._lock = threading.Lock()
        self._recorded_limit: int | None = None
        self._recorded_budgets: dict[str, int] = {}
        self.hits = 0
        self.forwarded = 0
        self.replayed: dict[str, float] = _no_usage()
        self.misses: list[tuple[str, dict[str, Any], CallMeta]] = []
        paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
        for entry in paths:
            self._load(os.fspath(entry))

    def _load(self, path: str) -> None:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith(".jsonl"):
                    self._load(os.path.join(path, name))
            return
        with open(path, encoding="utf-8") as handle:
            for number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    entry = json.loads(raw)
                    key = entry["key"]
                    entry["model"], entry["request"]
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: not a recorded call") from exc
                self._entries.setdefault(key, []).append(entry)
                limit = entry.get("max_questions")
                if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
                    smaller = (
                        self._recorded_limit is None or limit < self._recorded_limit
                    )
                    self._recorded_limit = limit if smaller else self._recorded_limit
                budget = entry.get("state_token_budget")
                if (
                    isinstance(budget, int)
                    and not isinstance(budget, bool)
                    and budget > 0
                ):
                    try:
                        name = body_model(entry["model"])
                    except ValueError:
                        continue
                    known = self._recorded_budgets.get(name)
                    if known is None or budget < known:  # the smallest of each model
                        self._recorded_budgets[name] = budget

    def max_questions(self, model: str) -> int:
        """Questions one Clef request may carry when it is replayed.

        The limit the cassette was recorded with (the smallest, if several), so the
        questions are chunked as they were and the request keys match; and, when a
        miss can go to an ``inner`` transport, no more than that one takes. Without
        either, the default (64).
        """
        limits = []
        if self._recorded_limit is not None:
            limits.append(self._recorded_limit)
        if self._inner is not None:
            limits.append(max_questions(self._inner, model))
        return min(limits) if limits else DEFAULT_MAX_QUESTIONS

    def state_token_budget(self, model: str) -> int | None:
        """The state budget the requests of ``model`` were recorded with (smallest).

        Per model: a cassette recorded through a routing transport holds ``clef`` and
        ``clef-flash`` requests narrowed to different budgets, and each must be
        replayed with its own for the request keys to match. When a miss can go to an
        ``inner`` transport, no larger than that one takes for the model. ``None`` if
        neither says one.
        """
        found: list[int] = []
        try:
            recorded = self._recorded_budgets.get(body_model(model))
        except ValueError:
            recorded = None
        if recorded is not None:
            found.append(recorded)
        if self._inner is not None:
            inner = state_token_budget(self._inner, model)
            if inner is not None:
                found.append(inner)
        return min(found) if found else None

    def prefers_keep_uncertain(self, model: str, kind: str | None = None) -> bool:
        """What the recording said of a call of ``kind`` to ``model``, replayed.

        ``RecordingTransport`` writes the decision itself (``keep_uncertain``) on
        every call, so a replay settles ``RepairConfig.keep_uncertain`` as the live
        run did: every recorded call of that model family and kind (of the provider
        this cassette stands for, if it has one) must have said yes. A call kind
        that was not recorded, or a recording that predates the field, falls back to
        the provider this cassette stands for, then to the ``inner`` transport that
        serves a miss; a cassette that knows none of them says no.
        """
        try:
            family = "clairic" if body_model(model) == "clairic" else "clef"
        except ValueError:
            family = None
        said: list[bool] = []
        for entries in self._entries.values():
            for entry in entries:
                value = entry.get("keep_uncertain")
                if not isinstance(value, bool):
                    continue
                if (
                    self._provider is not None
                    and entry.get("provider") != self._provider
                ):
                    continue
                if kind is not None and entry.get("kind") != kind:
                    continue
                try:
                    of = (
                        "clairic" if body_model(entry["model"]) == "clairic" else "clef"
                    )
                except ValueError:
                    continue
                if family is None or of == family:
                    said.append(value)
        if said:
            return all(said)
        if self._provider is not None:
            return self._provider == _TYPESAFE
        if self._inner is not None:
            return prefers_keep_uncertain(self._inner, model, kind)
        return False

    @property
    def provider(self) -> str | None:
        """The provider this cassette stands for (``None``: it replays any)."""
        return self._provider

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        """Whether a request can be served: what the ``inner`` transport says.

        A cassette with no ``inner`` answers whatever it recorded and says nothing
        (so, true); a router above it that must know what the provider behind a
        recording can serve wraps it in an adapter that does (the evals do).
        """
        return _can_serve(self._inner, model, payload) if self._inner else True

    def classify_failure(self, exc: Exception) -> str | None:
        """The ``inner`` transport's category of a failure (else by type)."""
        classify = getattr(self._inner, "classify_failure", None)
        if callable(classify):
            return cast("str | None", classify(exc))
        return classify_failure(exc)

    def sensitive_values(self) -> tuple[str, ...]:
        """The credentials passed explicitly to the ``inner`` transport."""
        sensitive = getattr(self._inner, "sensitive_values", None)
        return tuple(sensitive()) if callable(sensitive) else ()

    def providers(self) -> dict[str, tuple[str, ...]]:
        """Who answered the recorded calls: ``{"clairic": (...), "clef": (...)}``.

        The ``provider`` of each recording by the model it was for; a recording
        made before the field existed has none and is not listed.
        """
        found: dict[str, dict[str, None]] = {"clairic": {}, "clef": {}}
        for entries in self._entries.values():
            for entry in entries:
                provider = entry.get("provider")
                try:
                    family = (
                        "clairic" if body_model(entry["model"]) == "clairic" else "clef"
                    )
                except ValueError:
                    continue
                if isinstance(provider, str) and provider:
                    found[family][provider] = None
        return {k: tuple(sorted(v)) for k, v in found.items()}

    def fork(
        self,
        *,
        strict: bool | None = None,
        inner: Transport | None = None,
        provider: str | None = None,
    ) -> "CassetteTransport":
        """A new cassette over the same recordings, replaying from the start.

        The recordings are shared, not read again, so many runs can start from one
        snapshot of a directory that is still being written to.

        Args:
            strict: Raise on a miss; defaults to this cassette's setting.
            inner: Where a miss goes when not strict; defaults to this cassette's.
            provider: Replay only what this provider recorded (see the constructor);
                defaults to this cassette's.
        """
        other = CassetteTransport.__new__(CassetteTransport)
        other.strict = self.strict if strict is None else strict
        other._inner = self._inner if inner is None else inner
        other._provider = self._provider if provider is None else provider
        other._entries = self._entries
        other._recorded_limit = self._recorded_limit
        other._recorded_budgets = dict(self._recorded_budgets)
        other._cursor = {}
        other._lock = threading.Lock()
        other.hits = other.forwarded = 0
        other.replayed = _no_usage()
        other.misses = []
        return other

    def __len__(self) -> int:
        """The number of recorded calls."""
        return sum(len(v) for v in self._entries.values())

    def reset(self) -> None:
        """Start replaying every key from its first recording again."""
        with self._lock:
            self._cursor.clear()
            self.hits = self.forwarded = 0
            self.replayed = _no_usage()
            self.misses.clear()

    def unused(self) -> list[dict[str, Any]]:
        """The recordings that were not replayed (``key``, ``kind``, ``field``)."""
        out = []
        for key, entries in self._entries.items():
            for entry in entries[self._cursor.get(key, 0) :]:
                out.append(
                    {
                        k: entry.get(k)
                        for k in ("key", "model", "kind", "schema", "field")
                    }
                )
        return out

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Replay the recording of this request.

        Args:
            model: Full ``@cf/...`` model id.
            payload: Request body.
            meta: Routing context.
            timeout: Ignored (or passed to ``inner`` on a fall-through).

        Returns:
            The recorded response, with the recorded status and duration.

        Raises:
            CassetteMissError: If nothing is recorded under the request's key (or
                every recording of it was replayed) and there is no inner
                transport to ask.
            TransportError: The recorded error, if the recorded call failed.
        """
        body = json.loads(json.dumps(payload, allow_nan=False))
        key = request_key(model, body)
        with self._lock:
            everything = self._entries.get(key, [])
            entries = (
                everything
                if self._provider is None
                else [e for e in everything if e.get("provider") == self._provider]
            )
            index = self._cursor.get(key, 0)
            # with a live fallback a recorded failure is a transient accident, not
            # an answer: it is skipped and the request goes to the inner transport
            skip_errors = not self.strict and self._inner is not None
            while skip_errors and index < len(entries) and entries[index].get("error"):
                index += 1
            entry = entries[index] if index < len(entries) else None
            if entry is not None:
                self._cursor[key] = index + 1
                self.hits += 1
            else:
                if skip_errors:
                    self._cursor[key] = index
                self.misses.append((model, body, meta))
        if entry is None:
            if not self.strict and self._inner is not None:
                self.forwarded += 1
                return await self._inner.run(model, payload, meta=meta, timeout=timeout)
            raise self._miss(
                model, key, meta, len(entries), other=len(everything) - len(entries)
            )
        error = entry.get("error")
        if error:
            cls = _ERRORS.get(error.get("type", ""), TransportError)
            failure = cls(error.get("message", "recorded failure"))
            if isinstance(failure, TransportError):
                failure.status_code = entry.get("status")
            raise failure
        self._count_replayed(entry["response"])
        return TransportResponse(
            body=copy.deepcopy(entry["response"]),
            retries=0,
            duration_ms=float(entry.get("duration_ms") or 0.0),
            status_code=entry.get("status"),
            provider=entry.get("provider"),
            served_model=entry.get("served_model"),
            warnings=tuple(str(w) for w in entry.get("warnings") or ()),
        )

    def _count_replayed(self, body: Any) -> None:
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            return
        with self._lock:
            self.replayed["requests"] += 1
            for key in ("input_tokens", "output_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    self.replayed[key] += value
            cost = usage.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                self.replayed["cost"] += float(cost)

    @staticmethod
    def _miss(
        model: str, key: str, meta: CallMeta, recorded: int, other: int = 0
    ) -> CassetteMissError:
        what = f"a {meta.kind!r} call" + (
            f" about field {meta.field!r}" if meta.field else ""
        )
        why = (
            f"all {recorded} recording(s) of it were already replayed"
            if recorded
            else f"{other} recording(s) of it were made by another provider, not "
            "the one this call is routed to"
            if other
            else "no call with this request was recorded"
        )
        return CassetteMissError(
            f"cassette miss: {what} (schema {meta.schema!r}, scope {meta.scope!r}, "
            f"model {model}, key {key}): {why}. The request changed since the "
            "cassette was recorded; record it again, or replay with strict=False "
            "and an inner transport.",
            kind=meta.kind,
            field=meta.field,
            schema=meta.schema,
            model=model,
            key=key,
        )


def trace_to_cassette(
    trace: str | os.PathLike[str], cassette: str | os.PathLike[str]
) -> int:
    """Turn a wire trace into a cassette.

    A trace line is ``{"url", "http", "request", "response"}``: the request body
    that was sent, the full REST response and the URL it went to (the model is the
    part after ``/ai/run/``). Such a line has everything a cassette key needs,
    except the call kind, which is left empty (it is informational; a lookup uses
    the key only). The REST envelope is removed from the response. The lines are
    written as they are: a line whose request or response holds an exact credential
    (the Cloudflare environment variables) is left out with a ``RuntimeWarning``,
    never edited, so a recorded answer replays as the model gave it.

    Args:
        trace: The JSONL trace file.
        cassette: The JSONL file to write.

    Returns:
        The number of calls written.

    Raises:
        ValueError: If a line has no request, response or model url.
    """
    lines = []
    with open(trace, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            url = str(row.get("url", ""))
            marker = "/ai/run/"
            if marker not in url or not isinstance(row.get("request"), dict):
                raise ValueError(f"{trace}:{number}: not a wire trace line")
            model = url.split(marker, 1)[1].split("?", 1)[0]
            response = row.get("response")
            if (
                isinstance(response, dict)
                and "success" in response
                and "result" in response
            ):
                response = response["result"]
            bodies = json.dumps([row["request"], response], ensure_ascii=False)
            if any(secret in bodies for secret in _secrets()):
                # a cassette is never edited (it replays what was said): a call that
                # holds a credential is left out, as when recording
                warnings.warn(
                    f"trace_to_cassette: line {number} holds a credential in its "
                    "request or response and was not converted",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            lines.append(
                {
                    "key": request_key(model, row["request"]),
                    "model": model,
                    "kind": "",
                    "scope": "",
                    "schema": "",
                    "field": None,
                    "request": row["request"],
                    "response": response,
                    "status": row.get("http"),
                    "duration_ms": 0.0,
                }
            )
    with open(cassette, "w", encoding="utf-8") as out:
        for line in lines:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
    return len(lines)
