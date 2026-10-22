"""Test helpers: a fake transport, ``FakeWorld`` and response builders."""

import copy
import inspect
import json
import re
import time
import typing
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from datetime import time as dtime
from typing import Any, Literal

from pydantic import BaseModel

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
from trustcall_system_one.transport import CallMeta, TransportResponse

__all__ = [
    "FakeTransport",
    "FakeWorld",
    "WorldEntity",
    "choice",
    "clairic_answer",
    "clairic_response",
    "clef_response",
    "noul",
    "route",
    "score",
]

Handler = Callable[
    [str, dict[str, Any], CallMeta],
    "dict[str, Any] | Awaitable[dict[str, Any] | None] | None",
]


_FULL_IDS = (CLAIRIC, CLEF, CLEF_FLASH)


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
        matches: The existing record it matches (stored for M8).
        descriptive: Mentions that only a broadened discovery finds (see
            ``FakeWorld.entity``).
        flat: The field values by dotted path.
        spans: Where the text mentions it (known once the world is used).
        hidden: The part of ``spans`` that are descriptive mentions.
    """

    label: str
    schema: str
    mentions: tuple[tuple[str, int | None], ...]
    fields: dict[str, Any]
    real: bool
    matches: Any = None
    descriptive: tuple[tuple[str, int | None], ...] = ()
    flat: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    hidden: list[Span] = field(default_factory=list)


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
    header (``schema`` and ``entity`` attributes and the ``at characters a-b``
    spans of its mention line), for Clef ``state.entity`` (``label`` and the
    ``mentions`` with a ``span`` or, where the input was narrowed, a ``context``).
    The label comes first (the entity's label or one of its mention texts); the
    mention spans settle which of several entities share it, and a label that
    names one entity while the spans belong to another is an ``AssertionError``, as
    is a shared label with nothing to tell the entities apart. Without a label the
    answer is "the only entity of ``CallMeta.schema``" (the ``One`` path has no
    focus). With ``check_state`` (the default) a Clef request about an entity whose
    input holds none of the entity's mentions is an ``AssertionError`` too.

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
    * ``gate`` (Clef ``gate``): 0.9 if ``proposed_change.to`` is new compared to
      ``record`` (not equal to what the record holds at the path, nor an element
      of the list it is appended to), else 0.05.
    * ``match`` (choice): ``__new__`` (matching arrives with M8; the ``matches=``
      given to ``entity()`` is only stored) and ``delete`` (0.03).
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
        self._low: list[tuple[str, str, float]] = []
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
        descriptive: Sequence[str | tuple[str, int]] = (),
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
        )
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
        descriptive: Sequence[str | tuple[str, int]] = (),
    ) -> "WorldEntity":
        """Declare an entity of a schema that the text really contains.

        Args:
            label: What the entity is called (``"Emma Thompson"``); also how
                ``bleed``, ``low_confidence`` and ``count`` refer to it.
            schema: The schema (class or its name) the entity belongs to.
            mentions: The surface forms the text uses (see the class docstring).
            fields: The true value of each field, by dotted path.
            matches: The existing record this entity is (used by M8's matching;
                stored only).
            descriptive: Mentions of the entity that only a broadened discovery
                (one that allows descriptive references such as "my cousin") finds;
                the first, plain discovery does not list them.

        Returns:
            The declared entity (pass it, or its label, to ``bleed`` and
            ``low_confidence``).

        Raises:
            ValueError: If a mention is not in the text.
        """
        return self._add(
            label,
            schema,
            mentions,
            fields,
            real=True,
            matches=matches,
            descriptive=descriptive,
        )

    def non_entity(
        self, label: str, *, schema: Any, mentions: Sequence[str | tuple[str, int]]
    ) -> "WorldEntity":
        """Declare something that looks like an entity of a schema and is not.

        Discovery finds its mentions (a model is easily fooled) and qualification
        says no: a dog called Sunny is not a ``Person``.
        """
        return self._add(label, schema, mentions, None, real=False)

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
        self, entity: "WorldEntity | str", field_: str, confidence: float
    ) -> "FakeWorld":
        """Inject a fault: the model is unsure of one field's first answer.

        ``extract`` and ``crosscheck`` answers for the field carry ``confidence``;
        a re-ask (``reask``) is confident and gives the true value.
        """
        self._low.append((self._label(entity), field_, confidence))
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

    def _occurrences(
        self, schema: str, *, broadened: bool = False
    ) -> list[tuple[Span, WorldEntity]]:
        self._finalise()
        found = [
            (span, e)
            for e in self._items
            if e.schema == schema
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
    ) -> WorldEntity | None:
        """The focused element: by label, settled by spans, else the only entity.

        A label that names an element and spans that belong to another one are a
        bug in whoever built the request: that fails loudly.
        """
        self._finalise()
        pool = [e for e in self._items if e.schema == schema]
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

    def _focus(self, payload: dict[str, Any], meta: CallMeta) -> WorldEntity | None:
        state = payload["state"]
        if isinstance(state, str):
            found = re.search(r'<focus[^>]*\bentity="([^"]*)"', state)
            label = found.group(1) if found else None
            line = re.search(r"The input mentions [^\n]*? as: ([^\n]*)", state)
            cited = [
                (int(a), int(b))
                for a, b in re.findall(
                    r"at characters (\d+)-(\d+)", line.group(1) if line else ""
                )
            ]
            return self._resolve(meta.schema, label, cited)
        entity = state.get("entity") if isinstance(state, dict) else None
        if not isinstance(entity, dict):
            return self._resolve(meta.schema, None, [])
        spans: list[Span] = []
        for m in entity.get("mentions", []):
            if not isinstance(m, dict):
                continue
            if "span" in m:
                spans.append((m["span"][0], m["span"][1]))
            elif (placed := self._from_context(m)) is not None:
                spans.append(placed)
        element = self._resolve(meta.schema, entity.get("label"), spans)
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
            "gate": self._gate,
            "match": self._match,
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
                real = any(e.real for e in self._items if e.schema == meta.schema)
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
        found = [span for span, _ in self._occurrences(meta.schema, broadened=wide)]
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
        n = self._counts.get(
            meta.schema, sum(e.real for e in self._items if e.schema == meta.schema)
        )
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
        answers = {}
        for qid in payload["questions"]:
            entry = state["entities"][qid.removeprefix("qualify.")]
            spans = [
                (m["span"][0], m["span"][1]) for m in entry["mentions"] if "span" in m
            ]
            element = self._resolve(meta.schema, entry["label"], spans)
            self._check_mentions(element, str(state.get("input", "")))
            real = element is not None and element.real
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
        if primary:
            for label, path, c in self._low:
                if _fold(label) == _fold(element.label) and path in self._keys(wire):
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
            found = re.search(r'(?:is|:) "(.*)"\?\s*$', question["instructions"], re.S)
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

    # matching (M8) --------------------------------------------------------- #

    def _match(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {
                qid: _pick("__new__", list(question["criteria"]), 0.95)
                for qid, question in payload["questions"].items()
            },
            model=payload["model"],
        )

    def _delete(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {qid: noul(0.03) for qid in payload["questions"]}, model=payload["model"]
        )


def _pick(option: str, options: Sequence[str], p: float) -> dict[str, Any]:
    """A choice answer: ``option`` gets ``p``, the others share the rest."""
    rest = (1 - p) / max(1, len(options) - 1)
    return choice(option, {o: (p if o == option else rest) for o in options})
