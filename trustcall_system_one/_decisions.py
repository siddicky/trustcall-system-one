"""Translation between Clef bodies and the Decisions API (``/v1/decisions``).

The Decisions API takes ``{model, input, questions: [...]}`` and answers with
``{model, answers: [...], usage, provider}`` (see
``docs/schemas/experiential-decisions-response.example.json``). Only its
``choice`` question type has been observed, so every Clef question type is sent as
one:

* ``choice``: the options are the choices, under the same ids.
* ``noul``: two choices, ``yes`` and ``no``; ``noul`` is the probability of ``yes``.
* ``score``: one choice per level, named ``"0"``, ``"1"``, ...; ``score`` is the
  probability-weighted level and ``legend`` maps the levels to their descriptions.

The Decisions API returns a probability per choice. If an answer ever comes
without them it is *hard*: all of the probability goes to the chosen option, the
answer is tagged ``"hard": true`` and the confidence is 1.0, which is what the
probabilities say.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from trustcall_system_one._errors import BadRequestError, ModelResponseError
from trustcall_system_one._wire import (
    ChoiceQuestion,
    ClefRequest,
    NoulQuestion,
    ScoreQuestion,
    parse_clef,
)

__all__ = [
    "DECISIONS_MODEL",
    "DecisionsRequest",
    "DecisionsResponse",
    "Plan",
    "from_decisions_response",
    "render_state",
    "to_decisions_request",
]

DECISIONS_MODEL = "gpt-6-luna-decisions"
"""The model the Decisions API serves by default."""

YES = "yes"
NO = "no"
_MAX_QUESTIONS = 64


# --------------------------------------------------------------------------- #
# Wire models
# --------------------------------------------------------------------------- #


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionsChoice(_Request):
    """One option of a question."""

    value: str = Field(min_length=1)
    description: str | None = None


class DecisionsQuestion(_Request):
    """A question; ``choice`` is the one type used."""

    type: Literal["choice"]
    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    choices: list[DecisionsChoice] = Field(min_length=2, max_length=255)


class DecisionsRequest(_Request):
    """A Decisions request body."""

    model: str = Field(min_length=1)
    input: str
    questions: list[DecisionsQuestion] = Field(min_length=1, max_length=_MAX_QUESTIONS)

    def to_payload(self) -> dict[str, Any]:
        """The JSON-ready body, without unset optionals."""
        return self.model_dump(exclude_none=True, mode="json")


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow")


class DecisionsProbability(_Response):
    """The probability of one choice."""

    value: str
    probability: float


class DecisionsAnswer(_Response):
    """The answer to one question."""

    type: str
    name: str
    choice: str
    probabilities: list[DecisionsProbability] | None = None
    confidence: float | None = None


def _cost_or_none(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


class DecisionsUsage(_Response):
    """Token accounting and cost of one call."""

    input_tokens: int
    output_tokens: int
    cost: Annotated[float | None, BeforeValidator(_cost_or_none)] = None


class DecisionsResponse(_Response):
    """A Decisions response body."""

    model: str
    answers: list[DecisionsAnswer]
    usage: DecisionsUsage


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def render_value(value: Any) -> str:
    """A value as text: a string as it is, anything else as indented JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def render_state(state: Any) -> str:
    """Render a Clef ``state`` (or structured ``instructions``) as prompt text.

    A string passes through. An object becomes one labelled section per key,
    ``## key`` followed by the value; a string value is written out as it is and
    anything else as indented JSON. The labels keep the key names the questions
    refer to (``input``, ``entity``, ``record``, ``proposed_change``); the user's text
    stays unescaped, which reads better than a JSON string full of escaped
    newlines and quotes; nested data keeps its structure.
    An array or scalar is written as indented JSON.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping) and state:
        return "\n\n".join(
            f"## {key}\n{render_value(val)}" for key, val in state.items()
        )
    return render_value(state)


@dataclass(frozen=True)
class Plan:
    """What a question was turned into, to read its answer back.

    Attributes:
        qid: The Clef question id (the Decisions ``name``).
        kind: ``"noul"``, ``"choice"`` or ``"score"``.
        values: The Decisions choice values, in order.
        legend: For a score, the level descriptions in order.
    """

    qid: str
    kind: Literal["noul", "choice", "score"]
    values: tuple[str, ...]
    legend: tuple[Any, ...] = ()


def _describe(value: Any) -> str | None:
    return None if value is None else render_value(value)


def _question(qid: str, question: Any) -> tuple[DecisionsQuestion, Plan]:
    instructions = render_state(question.instructions)
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        yes = _describe(criteria.true) if criteria else None
        no = _describe(criteria.false) if criteria else None
        choices = [
            DecisionsChoice(value=YES, description=yes or "The answer is yes."),
            DecisionsChoice(value=NO, description=no or "The answer is no."),
        ]
        return (
            DecisionsQuestion(
                type="choice", name=qid, instructions=instructions, choices=choices
            ),
            Plan(qid, "noul", (YES, NO)),
        )
    if isinstance(question, ChoiceQuestion):
        choices = [
            DecisionsChoice(value=option, description=_describe(detail))
            for option, detail in question.criteria.items()
        ]
        return (
            DecisionsQuestion(
                type="choice", name=qid, instructions=instructions, choices=choices
            ),
            Plan(qid, "choice", tuple(question.criteria)),
        )
    if isinstance(question, ScoreQuestion):
        levels = [str(i) for i in range(len(question.criteria))]
        instructions += (
            f"\n\nChoose the level that fits best: the levels are numbered 0 (lowest) "
            f"to {len(levels) - 1} (highest)."
        )
        choices = [
            DecisionsChoice(value=level, description=_describe(detail))
            for level, detail in zip(levels, question.criteria, strict=True)
        ]
        return (
            DecisionsQuestion(
                type="choice", name=qid, instructions=instructions, choices=choices
            ),
            Plan(qid, "score", tuple(levels), tuple(question.criteria)),
        )
    raise BadRequestError(f"unsupported question type for {qid!r}")  # pragma: no cover


def to_decisions_request(
    payload: Mapping[str, Any], *, model: str = DECISIONS_MODEL
) -> tuple[dict[str, Any], list[Plan]]:
    """Translate a Clef request body into a Decisions request body.

    Args:
        payload: A Clef request (``model``, ``state``, ``questions``).
        model: The Decisions model to ask.

    Returns:
        The Decisions body, and one ``Plan`` per question, in order, to read the
        answers back with ``from_decisions_response``.

    Raises:
        BadRequestError: If the payload is not a valid Clef request, or uses
            something the Decisions API has no equivalent for (``images``).
    """
    try:
        request = ClefRequest.model_validate(payload)
    except ValidationError as exc:
        raise BadRequestError(f"not a valid Clef request: {exc}") from exc
    if request.images:
        raise BadRequestError(
            "the Decisions API fallback cannot take images; send them to Clef only"
        )
    questions: list[DecisionsQuestion] = []
    plans: list[Plan] = []
    for qid, question in request.questions.items():
        translated, plan = _question(qid, question)
        questions.append(translated)
        plans.append(plan)
    try:
        body = DecisionsRequest(
            model=model, input=render_state(request.state), questions=questions
        )
    except ValidationError as exc:  # pragma: no cover - the Clef limits are tighter
        raise BadRequestError(f"not a valid Decisions request: {exc}") from exc
    return body.to_payload(), plans


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


def _bad(message: str, body: Any) -> ModelResponseError:
    return ModelResponseError(f"Decisions response: {message}", raw=body)


def _answer(plan: Plan, given: DecisionsAnswer, body: Any) -> dict[str, Any]:
    if given.choice not in plan.values:
        raise _bad(
            f"{plan.qid!r}: {given.choice!r} is not an option {plan.values}", body
        )
    hard = given.probabilities is None
    if given.probabilities is None:
        probabilities = {v: 1.0 if v == given.choice else 0.0 for v in plan.values}
    else:
        reported = {p.value: p.probability for p in given.probabilities}
        unknown = sorted(set(reported) - set(plan.values))
        if unknown:
            raise _bad(
                f"{plan.qid!r}: probabilities for unknown options {unknown}", body
            )
        probabilities = {v: reported.get(v, 0.0) for v in plan.values}
    confidence = (
        given.confidence
        if given.confidence is not None
        else max(probabilities.values())
    )
    answer: dict[str, Any]
    if plan.kind == "noul":
        total = probabilities[YES] + probabilities[NO]
        if total > 0:
            yes = probabilities[YES] / total
        else:
            yes = 1.0 if given.choice == YES else 0.0
        answer = {"type": "noul", "noul": yes}
    elif plan.kind == "choice":
        answer = {
            "type": "choice",
            "choice": given.choice,
            "probabilities": probabilities,
            "confidence": confidence,
        }
    else:
        total = sum(probabilities.values())
        score = (
            sum(int(level) * p for level, p in probabilities.items()) / total
            if total > 0
            else float(given.choice)
        )
        answer = {
            "type": "score",
            "score": score,
            "legend": {str(i): detail for i, detail in enumerate(plan.legend)},
            "probabilities": probabilities,
            "confidence": confidence,
        }
    if hard:
        answer["hard"] = True
    return answer


def from_decisions_response(
    body: Any, plans: Sequence[Plan], *, clef_model: str
) -> dict[str, Any]:
    """Translate a Decisions response into a Clef response body.

    Args:
        body: The decoded Decisions response.
        plans: The plans ``to_decisions_request`` returned.
        clef_model: The ``model`` the Clef request named (``clef``/``clef-flash``),
            echoed in the response as the Clef models do.

    Returns:
        A body that ``parse_clef`` accepts: the answers in the order of the
        questions, ``usage`` (with ``cost`` when reported) and the extra keys
        ``provider`` (``"decisions"``) and ``decisions_model``. An answer without
        probabilities carries ``"hard": true``.

    Raises:
        ModelResponseError: If the body is not a Decisions response, does not
            answer exactly the questions asked, names an option that was not offered
            or does not become a valid Clef response.
    """
    try:
        parsed = DecisionsResponse.model_validate(body)
    except ValidationError as exc:
        raise ModelResponseError(
            f"Decisions response does not match the schema: {exc}", raw=body
        ) from exc
    given: dict[str, DecisionsAnswer] = {}
    for answer in parsed.answers:
        if answer.name in given:
            raise _bad(f"{answer.name!r} is answered twice", body)
        given[answer.name] = answer
    expected = [plan.qid for plan in plans]
    if set(given) != set(expected):
        missing = sorted(set(expected) - set(given))
        extra = sorted(set(given) - set(expected))
        raise _bad(
            f"answers do not match the questions (missing {missing}, extra {extra})",
            body,
        )
    answers: dict[str, Any] = {}
    for plan in plans:
        answers[plan.qid] = _answer(plan, given[plan.qid], body)
    usage: dict[str, Any] = {
        "input_tokens": parsed.usage.input_tokens,
        "output_tokens": parsed.usage.output_tokens,
    }
    if parsed.usage.cost is not None:
        usage["cost"] = parsed.usage.cost
    clef = {
        "model": clef_model,
        "answers": answers,
        "usage": usage,
        "provider": "decisions",
        "decisions_model": parsed.model,
    }
    parse_clef(clef)  # the result is a valid Clef response, or this raises
    return clef
