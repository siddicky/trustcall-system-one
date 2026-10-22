"""TypeSafe (Jev) as a stand-in for Clef and for Clairic.

Everything here is pure translation and orchestration; the HTTP is the transport's
(``transport.TypeSafeTransport``), which hands this module an ``ask`` callable.

**Clef** needs almost no translation: Jev's ``noul``, ``choice`` and ``score``
questions and answers are Clef's (same ids, same state, ``score`` levels are the
position in ``criteria``, the ``legend`` comes back as ``{"0": ..., "1": ...}``).
``to_jev_clef`` swaps the ``model`` and refuses ``images``; ``from_jev`` checks that
every question was answered with its own type and returns the Clef-shaped response.

**Clairic** is a model that reads values off text; Jev only answers closed questions,
so Clairic's fields are served by *selection*, following TypeSafe's extraction
cookbooks (see ``docs/DESIGN.md``, "TypeSafe-only provider"):

* pattern-shaped values (emails, phones, amounts, ids, names, numbers): code finds
  the candidate spans, one ``choice`` per field picks one (the span itself is the
  option id) or ``none`` (pre-parsed value extraction cookbook);
* free text: the input is split into lines, one request of yes/no questions stitches
  hard-wrapped lines into blocks, and the options are block, sentence and clause ids
  of the tagged blocks in the state (structure recovery cookbook);
* dates, times, datetimes and durations: the model picks the parts the document names
  and code resolves them against ``reference_date`` (date extraction cookbook);
* lists: one yes/no per candidate;
* every selected scalar is then verified by yes/no questions about whether it is
  hallucinated or lifted from incidental text, and a field that fires (P(wrong) above
  0.7) is escalated: to the caller's hook if there is one, else returned as a low
  confidence or null answer for the library's ladder to reject or re-ask (SDE cascade
  cookbook).

All independent questions over the same state go out in one request (the fan-out
pattern); the transport splits a request only for the question limit or the context.
"""

import asyncio
import inspect
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from pydantic import ValidationError

from trustcall_system_one._candidates import (
    MAX_OPTIONS,
    Candidate,
    item_type,
    pattern_candidates,
)
from trustcall_system_one._dateparts import (
    DATE_PARTS,
    REVIEW_BELOW,
    TIME_PARTS,
    Assembled,
    Part,
    assemble_date,
    assemble_datetime,
    assemble_duration,
    assemble_time,
    date_questions,
    duration_parts,
    duration_questions,
    time_questions,
)
from trustcall_system_one._errors import BadRequestError, ModelResponseError
from trustcall_system_one._structure import (
    JOIN_AFTER_DANGLING,
    JOIN_AFTER_TERMINAL,
    Structure,
    build_structure,
    join_questions,
    structure_key,
    tag_lines,
    to_lines,
)
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ClairicField,
    ClairicRequest,
    ClefRequest,
    ClefResponse,
    NoulAnswer,
    ScoreAnswer,
    parse_clef,
)

__all__ = [
    "NONE",
    "Ask",
    "EscalationHook",
    "EscalationRequest",
    "RunStats",
    "TypeSafeConfig",
    "from_jev",
    "run_clairic",
    "split_state",
    "to_jev_clef",
]

NONE = "none"
"""The option of every selection that says "none of these is the value"."""
FIRE_QUESTIONS = ("hallucinated", "off_target")
_NONE_TEXT = "None of these is the requested value."


@dataclass(frozen=True)
class TypeSafeConfig:
    """The knobs of the Clairic stand-in; the defaults are the cookbooks' numbers.

    The thresholds were tuned by TypeSafe on its own documents, not on this
    library's, so treat them as defaults to evaluate (``docs/DESIGN.md``).

    Attributes:
        join_after_dangling: Structure recovery: a line is joined to the previous
            one at this join probability when the previous line has no sentence
            ending punctuation (cookbook: 0.2).
        join_after_terminal: ... and when it ends with ``.``, ``!``, ``?``, ``:`` or
            ``;`` (cookbook: 0.5).
        stitch: Whether to ask the stitch questions (one request) at all. Off, every
            line is its own block.
        review_below: Date extraction: confidence below which an answer is flagged
            ``needs_review`` (cookbook: 0.60).
        verify_selection: Whether to verify each selected scalar (SDE cascade stage
            2, one more request).
        fire_threshold: Verification: P(wrong) above this escalates the field
            (cookbook: 0.7).
        context_chars: Characters of context quoted around a candidate in a yes/no.
        max_list_candidates: Most candidates one list field asks about.
        max_input_chars: Longest input read in one pass; a longer one is cut and the
            response says ``truncated`` so the library re-asks on windows.
        request_token_budget: Estimated tokens one Jev request may hold (Jev takes
            64k per request, 32k per question). A request over it is split.
    """

    join_after_dangling: float = JOIN_AFTER_DANGLING
    join_after_terminal: float = JOIN_AFTER_TERMINAL
    stitch: bool = True
    review_below: float = REVIEW_BELOW
    verify_selection: bool = True
    fire_threshold: float = 0.7
    context_chars: int = 40
    max_list_candidates: int = 400
    max_input_chars: int = 60_000
    request_token_budget: int = 40_000

    def __post_init__(self) -> None:
        """Validate the thresholds."""
        for name in (
            "join_after_dangling",
            "join_after_terminal",
            "review_below",
            "fire_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
        for name in (
            "context_chars",
            "max_list_candidates",
            "max_input_chars",
            "request_token_budget",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True)
class EscalationRequest:
    """What the escalation hook gets for a field the verification doubts.

    Attributes:
        field: The Clairic field name.
        spec: The field spec as sent (``type``, ``instructions``, ...).
        input: The input text.
        focus: The focus header (who the answer is about).
        selected: The value that was selected and is doubted.
        p_wrong: The highest P(wrong) of the verification questions.
        signals: P(wrong) by verification question.
    """

    field: str
    spec: Mapping[str, Any]
    input: str
    focus: str
    selected: Any
    p_wrong: float
    signals: Mapping[str, float]


EscalationHook = Callable[
    [EscalationRequest],
    "Awaitable[Mapping[str, Any] | None] | Mapping[str, Any] | None",
]
"""A callable that answers a doubted field with a stronger model (or ``None``).

It may be sync or async and returns a mapping with ``value`` (``None`` for "not
stated") and optionally ``span`` (``[start, end]`` in the input), ``confidence`` and
``p_null``; or ``None`` to leave the field to the library's ladder.
"""

Ask = Callable[
    [Any, Mapping[str, Mapping[str, Any]], str], Awaitable[Mapping[str, Any]]
]
"""``await ask(state, questions, stage)`` -> the answers by question id."""


@dataclass
class RunStats:
    """Requests, tokens and time of one Clairic call (all its Jev requests)."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    model: str | None = None
    stages: Counter[str] = field(default_factory=Counter)
    request_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Clef
# --------------------------------------------------------------------------- #


def to_jev_clef(payload: Mapping[str, Any], jev_model: str) -> dict[str, Any]:
    """The Jev request for a Clef request: the same state and questions.

    Args:
        payload: The Clef request body.
        jev_model: The Jev model (or alias) to ask.

    Returns:
        ``{model, state, questions}``; the questions are passed on as they are.

    Raises:
        BadRequestError: If the payload is not a Clef request or holds images
            (Jev reads text).
    """
    try:
        request = ClefRequest.model_validate(payload)
    except ValidationError as exc:
        raise BadRequestError(f"not a valid Clef request: {exc}") from exc
    if request.images:
        raise BadRequestError(
            "the TypeSafe transport cannot send images: Jev reads text (put the "
            "text of the image in the state)"
        )
    return {
        "model": jev_model,
        "state": payload["state"],
        "questions": dict(payload["questions"]),
    }


def from_jev(
    body: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
) -> ClefResponse:
    """Check a Jev response against the questions asked and parse it as Clef's.

    Args:
        body: The response body.
        questions: The questions that were sent, by id.

    Returns:
        The parsed response (Jev's shape is Clef's).

    Raises:
        ModelResponseError: If it is not a valid response, an answer is missing,
            or one does not match its question (type, option or level).
    """
    response = parse_clef(body)
    for qid, question in questions.items():
        answer = response.answers.get(qid)
        if answer is None:
            raise ModelResponseError(f"Jev did not answer question {qid!r}", raw=body)
        kind = question.get("type")
        if answer.type != kind:
            raise ModelResponseError(
                f"Jev answered question {qid!r} with a {answer.type}, not a {kind}",
                raw=body,
            )
        criteria = question.get("criteria")
        if (
            isinstance(answer, ChoiceAnswer)
            and isinstance(criteria, Mapping)
            and answer.choice not in criteria
        ):
            raise ModelResponseError(
                f"Jev chose {answer.choice!r} for {qid!r}, which is not an option",
                raw=body,
            )
        if isinstance(answer, ScoreAnswer) and isinstance(criteria, list):
            levels = {str(i) for i in range(len(criteria))}
            if any(level not in levels for level in answer.probabilities):
                raise ModelResponseError(
                    f"Jev gave {qid!r} a level that is not in the rubric", raw=body
                )
    return response


# --------------------------------------------------------------------------- #
# The Clairic state
# --------------------------------------------------------------------------- #

_OPEN = "<input>\n"
_CLOSE = "\n</input>"
_CHARACTERS = re.compile(
    r"The (entity|parent) is mentioned in the input at characters ([0-9\-, ]+)\."
)


@dataclass(frozen=True)
class StateParts:
    """A Clairic state split into its frame and its input.

    Attributes:
        text: The whole state.
        head: What precedes the input content (the focus block).
        input: The input content as sent.
        input_start: Where the input content starts in ``text`` (span offsets are
            relative to the state).
        input_end: Where it ends.
    """

    text: str
    head: str
    input: str
    input_start: int
    input_end: int


def split_state(text: str) -> StateParts:
    """Find the input block of a Clairic state (``<input>`` ... ``</input>``).

    A state without the sentinel (a plain string) is all input.
    """
    at = text.find(_OPEN)
    if at < 0:
        return StateParts(text, "", text, 0, len(text))
    start = at + len(_OPEN)
    end = text.rfind(_CLOSE, start)
    while end >= 0:
        rest = text[end + len(_CLOSE) :]
        if not rest or rest.startswith("\n<"):
            break
        end = text.rfind(_CLOSE, start, end)
    if end < 0:
        end = len(text)
    return StateParts(text, text[:at], text[start:end], start, end)


def _focus(head: str, content: str) -> str:
    """The focus block as text for the model.

    The offsets the library quotes ("mentioned in the input at characters 10-23")
    mean nothing to a model that cannot count characters, so they become the quoted
    words when the input is whole, and are left out when it was narrowed.
    """
    text = head.strip()
    whole = not (content.startswith("…\n") or "\n…\n" in content)

    def replace(m: re.Match[str]) -> str:
        if not whole:
            return ""
        words: list[str] = []
        for pair in m.group(2).split(","):
            ends = pair.strip().split("-")
            if len(ends) != 2 or not all(e.strip().isdigit() for e in ends):
                continue
            a, b = int(ends[0]), int(ends[1])
            if 0 <= a < b <= len(content) and b - a <= 80:
                piece = content[a:b]
                if piece not in words:
                    words.append(piece)
        if not words:
            return ""
        quoted = ", ".join(json.dumps(w, ensure_ascii=False) for w in words)
        return f"The {m.group(1)} is mentioned in the input as: {quoted}."

    return _CHARACTERS.sub(replace, text).strip()


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #


def _desc(spec: ClairicField, name: str) -> str:
    return (spec.instructions or f"the {name}").strip()


def _pick_instructions(desc: str, structured: bool) -> str:
    options = (
        "Each option is an exact quoted span of `input`, or the id of a block (B..), "
        "sentence (S..) or part of a sentence (S...) listed in `input`."
        if structured
        else "Each option is an exact quoted span of `input`."
    )
    desc = desc if desc.rstrip().endswith((".", "?", "!", ":")) else desc.rstrip() + "."
    return (
        f"For the entity named in `focus`, from what `input` states: {desc} "
        f"Which option is it? {options} Choose {NONE} if `input` does not state it."
    )


def _criteria(options: list[Candidate]) -> dict[str, Any]:
    criteria: dict[str, Any] = {}
    for c in options:
        criteria[c.id or c.text] = c.description if c.id else c.reading
    criteria[NONE] = _NONE_TEXT
    return criteria


def _noul(
    instructions: Any, criteria: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = dict(criteria)
    return question


def _candidate_ref(c: Candidate) -> dict[str, Any]:
    """How a yes/no refers to a candidate.

    By its id for a block or sentence (the state holds the text), by its words and a
    little context otherwise.
    """
    if c.id and c.kind in ("block", "sentence"):
        return {"id": c.id}
    ref: dict[str, Any] = {"id": c.id} if c.id else {}
    ref["text"] = c.text
    if c.contexts:
        ref["context"] = c.contexts[0]
    return ref


def _member_question(desc: str, c: Candidate) -> dict[str, Any]:
    return _noul(
        {
            "question": (
                "For the entity named in `focus`, from what `input` states: is the "
                "`candidate` one of the items described by `field`?"
            ),
            "field": desc,
            "candidate": _candidate_ref(c),
        }
    )


_VERIFY = {
    "hallucinated": (
        "Is the `extracted_field` unsupported by, or absent from, the source text?",
        "the `extracted_field` is a hallucination -- not supported by, or absent "
        "from, the source text",
        "the `extracted_field` is supported by the source text",
    ),
    "off_target": (
        "Does the source text fail to genuinely report the thing the `field_spec` "
        "describes, so the value was pulled from incidental text?",
        "the source does not genuinely provide this field -- the value was pulled "
        "from incidental text",
        "the source genuinely reports this field",
    ),
}


def _verify_questions(name: str, spec: ClairicField, value: Any) -> dict[str, Any]:
    field_spec = {
        "path": name,
        "type": spec.type,
        "description": spec.instructions or "",
    }
    return {
        f"{name}::{metric}": _noul(
            {
                "field_spec": field_spec,
                "extracted_field": value,
                "main_question": question,
            },
            {"true": true, "false": false},
        )
        for metric, (question, true, false) in _VERIFY.items()
    }


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


@dataclass
class _Job:
    name: str
    spec: ClairicField
    kind: str  # "pick", "parts" or "list"
    options: list[Candidate] = field(default_factory=list)
    chunks: list[list[Candidate]] = field(default_factory=list)
    finalists: list[Candidate] = field(default_factory=list)
    answer: ChoiceAnswer | None = None
    members: dict[int, float] = field(default_factory=dict)
    parts: dict[str, Part] = field(default_factory=dict)
    time_parts: dict[str, Part] = field(default_factory=dict)
    elements: dict[int, dict[str, Part]] = field(default_factory=dict)
    out: dict[str, Any] = field(default_factory=dict)


def _lookup(c: Candidate) -> str:
    return c.id or c.text


def _top(answer: ChoiceAnswer, by_option: Mapping[str, Candidate]) -> Candidate | None:
    ranked = sorted(
        (o for o in answer.probabilities if o != NONE and o in by_option),
        key=lambda o: -answer.probabilities[o],
    )
    return by_option[ranked[0]] if ranked else None


def _parts_of(answers: Mapping[str, Any], prefix: str, names: Any) -> dict[str, Part]:
    parts: dict[str, Part] = {}
    for part in names:
        answer = answers.get(f"{prefix}{part}")
        if isinstance(answer, ChoiceAnswer):
            parts[part] = Part(answer.choice, answer.confidence, answer.probabilities)
    return parts


def _today(reference: str | None) -> date:
    if reference:
        try:
            return date.fromisoformat(reference[:10])
        except ValueError:
            pass
    return datetime.now(timezone.utc).date()


def _role(spec: ClairicField, name: str) -> str:
    text = _desc(spec, name).rstrip(".")
    return text[:1].lower() + text[1:] if text[:2].upper() != text[:2] else text


def _structured(spec: ClairicField) -> bool:
    return item_type(spec) == "text"


async def run_clairic(
    payload: Mapping[str, Any],
    *,
    ask: Ask,
    stats: RunStats,
    config: TypeSafeConfig,
    memo: dict[str, dict[str, float]],
    escalate: EscalationHook | None = None,
    model_name: str = "jev",
) -> dict[str, Any]:
    """Serve a Clairic request with Jev.

    Args:
        payload: The Clairic request body (``ClairicRequest``).
        ask: Sends questions over a state and returns the answers; counts the
            requests in ``stats``.
        stats: Where requests and tokens are counted.
        config: Thresholds and limits.
        memo: Stitch answers by input (kept by the transport across calls).
        escalate: Answers a field the verification doubts, or ``None``.
        model_name: The model name for the response before Jev has answered.

    Returns:
        A Clairic response body.

    Raises:
        BadRequestError: If the payload is not a Clairic request or asks for
            something Jev cannot serve (images, ``forecast``).
    """
    try:
        request = ClairicRequest.model_validate(payload)
    except ValidationError as exc:
        raise BadRequestError(f"not a valid Clairic request: {exc}") from exc
    if request.images:
        raise BadRequestError(
            "the TypeSafe transport cannot send images: Jev reads text"
        )
    forecasts = [n for n, f in request.fields.items() if f.type == "forecast"]
    if forecasts:
        raise BadRequestError(
            f"Jev cannot forecast: field(s) {forecasts} have type 'forecast'"
        )
    parts = split_state(request.state)
    truncated = len(parts.input) > config.max_input_chars
    content = parts.input
    if truncated:
        cut = content.rfind(" ", 0, config.max_input_chars)
        content = content[: cut if cut > 0 else config.max_input_chars]
    focus = _focus(parts.head, parts.input)
    today = _today(request.reference_date)

    structure: Structure | None = None
    if any(_structured(f) for f in request.fields.values()):
        structure = await _structure(content, ask, stats, config, memo)
    view = structure.tagged() if structure is not None else content
    state: dict[str, Any] = {"input": view}
    if focus:
        state = {"focus": focus, **state}

    jobs = _plan(request, content, structure, config, today)
    questions: dict[str, dict[str, Any]] = {}
    for job in jobs:
        questions.update(_stage_one(job))
    answers: Mapping[str, Any] = (
        await ask(state, questions, "select") if questions else {}
    )

    followups: dict[str, dict[str, Any]] = {}
    for job in jobs:
        followups.update(_absorb(job, answers))
    if followups:
        answers = await ask(state, followups, "refine")
        for job in jobs:
            _absorb_followups(job, answers)

    base = parts.input_start
    for job in jobs:
        _finish(job, base, request, today, config)

    await _verify(jobs, request, content, focus, ask, config, escalate, base)

    usage = {
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "passes": stats.requests,
    }
    body: dict[str, Any] = {
        "model": stats.model or model_name,
        "answers": {job.name: job.out for job in jobs},
        "truncated": truncated,
        "usage": usage,
        "jev": {
            "requests": stats.requests,
            "stages": dict(stats.stages),
            "retries": stats.retries,
        },
    }
    options = request.options
    if options is not None and options.return_state_text:
        body["state_text"] = (
            parts.text[: parts.input_start + len(content)] if truncated else parts.text
        )
    return body


async def _structure(
    content: str,
    ask: Ask,
    stats: RunStats,
    config: TypeSafeConfig,
    memo: dict[str, dict[str, float]],
) -> Structure:
    """Lines, the stitch request and the blocks (cookbook: structure recovery)."""
    lines = to_lines(content)
    joins: dict[str, float] = {}
    questions = join_questions(lines) if config.stitch else {}
    if questions:
        key = structure_key(content)
        known = memo.get(key)
        if known is None:
            state = {"input": tag_lines(lines)}
            answered = await ask(state, questions, "stitch")
            known = {
                qid: a.noul for qid, a in answered.items() if isinstance(a, NoulAnswer)
            }
            if len(memo) >= 256:
                memo.pop(next(iter(memo)))
            memo[key] = known
        joins = known
    return build_structure(
        content,
        lines,
        joins,
        after_dangling=config.join_after_dangling,
        after_terminal=config.join_after_terminal,
    )


def _plan(
    request: ClairicRequest,
    content: str,
    structure: Structure | None,
    config: TypeSafeConfig,
    today: date,
) -> list[_Job]:
    jobs: list[_Job] = []
    for name, spec in request.fields.items():
        kind = item_type(spec)
        if spec.type == "list":
            job = _Job(name, spec, "list")
            if kind in ("text", "number", "date"):
                job.options = _candidates(spec, content, structure, config)
        elif kind in ("date", "time", "datetime", "duration"):
            job = _Job(name, spec, "parts")
        else:
            job = _Job(name, spec, "pick")
            job.options = _candidates(spec, content, structure, config)
            job.chunks = [
                job.options[i : i + MAX_OPTIONS]
                for i in range(0, len(job.options), MAX_OPTIONS)
            ]
        jobs.append(job)
    return jobs


_PRIORITY = {
    "email": 0, "url": 0, "phone": 0, "identifier": 0, "labelled": 0, "quoted": 0,
    "name": 0, "amount": 0, "sum": 0, "number": 1, "number_word": 1, "date": 1,
    "time": 1, "sentence": 2, "clause": 3, "block": 4, "phrase": 5,
}  # fmt: skip


def _candidates(
    spec: ClairicField,
    content: str,
    structure: Structure | None,
    config: TypeSafeConfig,
) -> list[Candidate]:
    found = pattern_candidates(spec, content, width=config.context_chars)
    if _structured(spec) and structure is not None:
        taken = {c.id for c in structure.candidates}
        found = [c for c in found if c.text != NONE and c.text not in taken]
        found = [*structure.candidates, *found]
    else:
        found = [c for c in found if c.text != NONE]
    if spec.type == "list" and len(found) > config.max_list_candidates:
        found = sorted(found, key=lambda c: _PRIORITY.get(c.kind, 6))[
            : config.max_list_candidates
        ]
    return sorted(found, key=lambda c: (c.start, c.end, _PRIORITY.get(c.kind, 6)))


def _stage_one(job: _Job) -> dict[str, Any]:
    name, spec = job.name, job.spec
    desc = _desc(spec, name)
    out: dict[str, Any] = {}
    if job.kind == "pick":
        for k, chunk in enumerate(job.chunks):
            qid = f"{name}::pick" if len(job.chunks) == 1 else f"{name}::pick{k}"
            out[qid] = {
                "type": "choice",
                "instructions": _pick_instructions(desc, _structured(spec)),
                "criteria": _criteria(chunk),
            }
    elif job.kind == "list":
        for i, candidate in enumerate(job.options):
            out[f"{name}::in{i}"] = _member_question(desc, candidate)
    else:
        role = _role(spec, name)
        if spec.type in ("date", "datetime"):
            for part, q in date_questions(role).items():
                out[f"{name}::{part}"] = q
        if spec.type in ("time", "datetime"):
            for part, q in time_questions(role).items():
                out[
                    f"{name}::t.{part}"
                    if spec.type == "datetime"
                    else f"{name}::{part}"
                ] = q
        if spec.type == "duration":
            for part, q in duration_questions(role).items():
                out[f"{name}::{part}"] = q
    return out


def _absorb(job: _Job, answers: Mapping[str, Any]) -> dict[str, Any]:
    """Read the first round of answers; return the questions of a second round."""
    name, spec = job.name, job.spec
    if job.kind == "pick":
        if len(job.chunks) == 1:
            answer = answers.get(f"{name}::pick")
            job.answer = answer if isinstance(answer, ChoiceAnswer) else None
            return {}
        finalists: list[tuple[float, Candidate]] = []
        for k, chunk in enumerate(job.chunks):
            answer = answers.get(f"{name}::pick{k}")
            if not isinstance(answer, ChoiceAnswer):
                continue
            top = _top(answer, {_lookup(c): c for c in chunk})
            if top is not None:
                finalists.append((answer.probabilities[_lookup(top)], top))
        finalists.sort(key=lambda f: -f[0])
        job.finalists = [c for _, c in finalists[:MAX_OPTIONS]]
        job.finalists.sort(key=lambda c: (c.start, c.end))
        if not job.finalists:
            return {}
        return {
            f"{name}::final": {
                "type": "choice",
                "instructions": _pick_instructions(
                    _desc(spec, name), _structured(spec)
                ),
                "criteria": _criteria(job.finalists),
            }
        }
    if job.kind == "list":
        for i in range(len(job.options)):
            answer = answers.get(f"{name}::in{i}")
            if isinstance(answer, NoulAnswer):
                job.members[i] = answer.noul
        if spec.items == "date":
            follow: dict[str, Any] = {}
            role_of = _role(spec, name)
            for i, p in job.members.items():
                if p >= 0.5:
                    candidate = job.options[i]
                    context = (
                        candidate.contexts[0] if candidate.contexts else candidate.text
                    )
                    role = (
                        f'the date written "{candidate.text}" (as in {context}), '
                        f"one of {role_of}"
                    )
                    for part, q in date_questions(role).items():
                        follow[f"{name}::e{i}::{part}"] = q
            return follow
        return {}
    if spec.type in ("date", "datetime"):
        job.parts = _parts_of(answers, f"{name}::", DATE_PARTS)
    if spec.type == "time":
        job.parts = _parts_of(answers, f"{name}::", TIME_PARTS)
    elif spec.type == "datetime":
        job.time_parts = _parts_of(answers, f"{name}::t.", TIME_PARTS)
    elif spec.type == "duration":
        job.parts = _parts_of(answers, f"{name}::", duration_parts())
    return {}


def _absorb_followups(job: _Job, answers: Mapping[str, Any]) -> None:
    name = job.name
    if job.kind == "pick" and job.finalists:
        answer = answers.get(f"{name}::final")
        job.answer = answer if isinstance(answer, ChoiceAnswer) else None
    elif job.kind == "list" and job.spec.items == "date":
        for i, p in job.members.items():
            if p >= 0.5:
                job.elements[i] = _parts_of(answers, f"{name}::e{i}::", DATE_PARTS)


def _span(base: int, candidate: Candidate) -> list[int]:
    return [base + candidate.start, base + candidate.end]


def _null_threshold(spec: ClairicField) -> float:
    return spec.null_threshold if spec.null_threshold is not None else 0.5


def _finish(
    job: _Job, base: int, request: ClairicRequest, today: date, config: TypeSafeConfig
) -> None:
    spec = job.spec
    out: dict[str, Any] = {"type": spec.type, "value": None}
    job.out = out
    if job.kind == "pick":
        options = job.finalists if len(job.chunks) > 1 else job.options
        if not options:
            out.update(p_null=1.0, confidence=1.0)  # nothing in the input to pick
            return
        answer = job.answer
        if answer is None:
            out.update(p_null=1.0, confidence=0.0)
            return
        top = _top(answer, {_lookup(c): c for c in options})
        p_null = float(answer.probabilities.get(NONE, 0.0))
        out["p_null"] = p_null
        out["confidence"] = answer.confidence
        if top is not None:
            out["guess"] = top.value
        if top is not None and p_null <= _null_threshold(spec):
            out["value"] = top.value
            out["span"] = _span(base, top)
        return
    if job.kind == "parts":
        _finish_parts(job, request, today, config)
        return
    _finish_list(job, base, today, config)


def _finish_parts(
    job: _Job, request: ClairicRequest, today: date, config: TypeSafeConfig
) -> None:
    spec, out = job.spec, job.out
    result: Assembled
    if spec.type == "date":
        result = assemble_date(job.parts, today, config.review_below)
    elif spec.type == "time":
        result = assemble_time(job.parts, config.review_below)
    elif spec.type == "datetime":
        result = assemble_datetime(
            job.parts, job.time_parts, today, config.review_below
        )
    else:
        result = assemble_duration(
            job.parts,
            spec.unit or "seconds",
            integer=bool(spec.integer),
            decimals=spec.decimals,
            review_below=config.review_below,
        )
    out["p_null"] = result.p_null
    out["confidence"] = 0.0 if result.confidence is None else result.confidence
    out["needs_review"] = result.needs_review
    if result.note:
        out["note"] = result.note
    if result.value is not None and result.p_null <= _null_threshold(spec):
        out["value"] = result.value
    elif result.value is not None:
        out["guess"] = result.value


def _finish_list(job: _Job, base: int, today: date, config: TypeSafeConfig) -> None:
    out = job.out
    accepted = [i for i, p in job.members.items() if p >= 0.5]
    structural = [
        i
        for i in accepted
        if job.options[i].kind in ("block", "sentence", "clause", "phrase")
    ]
    drop = {
        i
        for i in structural
        for j in structural
        if i != j
        and job.options[j].start <= job.options[i].start
        and job.options[i].end <= job.options[j].end
        and (job.options[j].end - job.options[j].start)
        > (job.options[i].end - job.options[i].start)
    }
    elements: list[Any] = []
    spans: list[list[int]] = []
    probs: list[float] = []
    for i in accepted:
        if i in drop:
            continue
        candidate = job.options[i]
        value: Any = candidate.value
        if job.spec.items == "date":
            result = (
                assemble_date(job.elements.get(i, {}), today, config.review_below)
                if job.elements.get(i)
                else None
            )
            if result is None or result.value is None:
                continue
            value = result.value
        if value in elements:
            continue
        elements.append(value)
        spans.append(_span(base, candidate))
        probs.append(job.members[i])
    if elements:
        out["value"] = elements
        out["spans"] = spans
        out["confidence"] = min(probs)
    else:
        rest = [p for p in job.members.values()]
        out["confidence"] = 1.0 - max(rest) if rest else 1.0


# --------------------------------------------------------------------------- #
# Verification and escalation (SDE cascade)
# --------------------------------------------------------------------------- #


async def _verify(
    jobs: list[_Job],
    request: ClairicRequest,
    content: str,
    focus: str,
    ask: Ask,
    config: TypeSafeConfig,
    escalate: EscalationHook | None,
    base: int,
) -> None:
    """Stage 2 and 3 of the cascade for the selected scalars."""
    if not config.verify_selection:
        return
    chosen = [j for j in jobs if j.kind == "pick" and j.out.get("value") is not None]
    if not chosen:
        return
    questions: dict[str, dict[str, Any]] = {}
    for j in chosen:
        questions.update(_verify_questions(j.name, j.spec, j.out["value"]))
    state: dict[str, Any] = {
        "source_text": content,
        "extraction": {j.name: j.out["value"] for j in chosen},
    }
    if focus:
        state["focus"] = focus
    answers = await ask(state, questions, "verify")
    for j in chosen:
        signals = {
            metric: float(a.noul)
            for metric in FIRE_QUESTIONS
            if isinstance(a := answers.get(f"{j.name}::{metric}"), NoulAnswer)
        }
        if not signals:
            continue
        wrong = max(signals.values())
        j.out["p_wrong"] = wrong
        if wrong <= config.fire_threshold:
            continue
        await _escalate(j, request, content, focus, signals, wrong, escalate, base)


async def _escalate(
    job: _Job,
    request: ClairicRequest,
    content: str,
    focus: str,
    signals: Mapping[str, float],
    wrong: float,
    escalate: EscalationHook | None,
    base: int,
) -> None:
    out = job.out
    selected = out["value"]
    if escalate is not None:
        given = escalate(
            EscalationRequest(
                job.name,
                request.fields[job.name].model_dump(exclude_none=True),
                content,
                focus,
                selected,
                wrong,
                dict(signals),
            )
        )
        if inspect.isawaitable(given):
            given = await given
        if given is not None:
            value = given.get("value")
            out["value"] = value
            out["p_null"] = float(
                given.get("p_null", 0.0 if value is not None else 1.0)
            )
            out["confidence"] = float(given.get("confidence", 0.75))
            if given.get("span") is not None:
                out["span"] = [
                    base + int(given["span"][0]),
                    base + int(given["span"][1]),
                ]
            else:
                out.pop("span", None)
            out["escalated"] = "hook"
            return
    out["guess"] = selected
    out["p_null"] = max(float(out.get("p_null", 0.0)), wrong)
    out["confidence"] = min(float(out.get("confidence", 1.0)), 1.0 - wrong)
    out["escalated"] = "unavailable"
    if out["p_null"] > _null_threshold(request.fields[job.name]):
        out["value"] = None
        out.pop("span", None)


async def gather_all(*aws: Awaitable[Any]) -> list[Any]:
    """Run awaitables concurrently; if one raises, cancel the rest and re-raise."""
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
