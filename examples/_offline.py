"""Scripted model replies for the examples' offline mode.

Nothing in here talks to a model. ``OfflineScript`` plays the part of Clairic and
Clef behind a ``FakeTransport``: you declare, per field or question, the answer a
model would plausibly give, and the script builds a schema-valid response for
whatever request the library actually sends.

The ideas are the ones the unit tests use (``tests/unit/extract_helpers.py``), but
this module is self-contained so an example never imports from ``tests``:

* Clairic text answers are *located in the request's state*: you say what the
  value is and which text supports it, and the span is computed with ``str.find``
  over the state the library really rendered (``<focus>``, ``<input>``, ...), so
  the offsets are the ones a real model would return.
* Answers are looked up by wire name (Clairic field / Clef question id), and, for
  multi-entity runs, optionally by ``"<entity label>::<wire name>"`` first.
* A list is consumed one request at a time (the last item repeats), which is how
  a re-ask returns something different from the first answer.

The collection engine's requests (milestone M7) have fixed shapes, and the helpers
at the bottom of this file answer exactly those:

* ``presence`` (Clef): one question, id ``presence``.
* ``discover`` (Clairic): field ``mentions``, a ``list`` of ``text``; the state is
  ``<focus schema="..." task="discover">`` plus ``<input>``; spans are offsets in
  that state.
* ``count`` (Clairic): field ``count``, an integer ``number``, same state.
* ``coref`` (Clef): ONE structured state ``{"input", "mentions": {"m0": {"text",
  "span", "context"}, ...}}`` and one question per pair, id ``pair.m0.m1``.
* ``qualify`` (Clef): ONE request for all entities, state ``{"input", "entities":
  {"e0": {"label", "mentions"}, ...}}`` and one question per entity, id
  ``qualify.e0``.
* ``extract`` and the verification calls about one entity carry
  ``<focus schema="..." entity="Emma Thompson">`` (Clairic) or ``state.entity.label``
  (Clef), which is what ``"<label>::<wire name>"`` keys are matched against.

* ``match`` (Clef, M8): one request per entity, state ``{"input", "entity":
  {"label", "mentions"}}`` and ONE ``choice`` question, id ``match``, whose options
  are ``rec_<i>`` (the description is the existing record as a JSON object) and
  ``__new__``.
* ``delete`` (Clef, M8, ``enable_deletes``): one ``noul`` per existing record, ids
  ``delete.<k>``, state ``{"input", "record"}``; ``CallMeta.entity`` holds the
  record ids of the request, comma-joined.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from trustcall_system_one import CLAIRIC, CLEF_FLASH, CallMeta
from trustcall_system_one.testing import (
    FakeTransport,
    choice,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    score,
)

__all__ = [
    "OFFLINE_BANNER",
    "Call",
    "OfflineScript",
    "ScriptError",
    "all_mentions",
    "change_gate",
    "choose",
    "entity_count",
    "mention_discovery",
    "nothing",
    "pair_coreference",
    "pick",
    "qualifier",
    "record_matcher",
    "says",
    "scale",
]

OFFLINE_BANNER = "OFFLINE DEMO — answers come from a scripted transport, not the models"

_INPUT_OPEN = "<input>\n"
_INPUT_CLOSE = "\n</input>"


class ScriptError(AssertionError):
    """The library asked something the example's script has no answer for."""


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


def says(
    type_: str,
    value: Any,
    *,
    find: str | None = None,
    occurrence: int = 0,
    find_all: Sequence[str | tuple[str, int]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Describe a Clairic answer; the span is computed when a request arrives.

    Args:
        type_: Clairic type echoed back (``"text"``, ``"number"``, ``"date"``...).
        value: The answer value.
        find: Text that supports the value. Its span in the request's input block
            is the answer's ``span`` (``occurrence`` picks the n-th match).
        occurrence: Which occurrence of ``find`` supports the value.
        find_all: For a list answer, the supporting text of every element; the
            elements and ``spans`` are the ones found in the state (an element
            missing from a narrowed state is dropped, like a real model's). An
            item may be ``(text, occurrence)`` to pick a later match.
        **extra: Further answer keys: ``confidence``, ``p_null``, ``guess``...

    Returns:
        A lazy answer description for ``OfflineScript``.
    """
    spec: dict[str, Any] = {"type": type_, "value": value, **extra}
    if find is not None:
        spec["_find"] = (find, occurrence)
    if find_all is not None:
        spec["_find_all"] = list(find_all)
    return spec


def nothing(
    type_: str = "text", *, p_null: float = 0.97, guess: Any = None
) -> dict[str, Any]:
    """Describe a confident null: the input does not say."""
    extra: dict[str, Any] = {"p_null": p_null, "confidence": 0.93}
    if guess is not None:
        extra["guess"] = guess
    return says(type_, None, **extra)


def pick(
    option: str, options: Sequence[str], p: float = 0.95, **rest: float
) -> dict[str, Any]:
    """A Clef ``choice`` answer: ``option`` gets ``p``, the others share the rest.

    Args:
        option: The chosen option id.
        options: Every option id of the question.
        p: Probability of ``option``.
        **rest: Explicit probabilities for other options (they must sum with ``p``
            to at most 1; the remaining options share what is left).
    """
    others = [o for o in options if o != option]
    left = max(0.0, 1.0 - p - sum(rest.values()))
    free = [o for o in others if o not in rest]
    probabilities = {option: p, **rest}
    for other in free:
        probabilities[other] = left / len(free) if free else 0.0
    return choice(option, {o: round(probabilities[o], 4) for o in options})


def input_bounds(state: str) -> tuple[int, int]:
    """Where the user's text sits in a rendered Clairic state."""
    start = state.find(_INPUT_OPEN)
    start = 0 if start < 0 else start + len(_INPUT_OPEN)
    end = state.rfind(_INPUT_CLOSE)
    return start, len(state) if end < 0 else end


def locate(
    state: str, needle: str, occurrence: int = 0, source: str | None = None
) -> list[int] | None:
    """``[start, end]`` of ``needle`` inside the input block of ``state``.

    With ``source`` (the original text) ``occurrence`` counts matches in the
    ORIGINAL text and the match is mapped into the state by its surrounding text,
    so it still means the same place when the state was narrowed to windows. A
    needle that is not in ``source`` at all is a script error.
    """
    start, end = input_bounds(state)
    if source is not None:
        at = -1
        for _ in range(occurrence + 1):
            at = source.find(needle, at + 1)
            if at < 0:
                raise ScriptError(
                    f"{needle!r} (occurrence {occurrence}) is not in the text"
                )
        for margin in (24, 12, 6, 0):
            if margin == 0 and source.count(needle) != 1:
                # Without surrounding text a match in the state could be another
                # occurrence than the one meant: only a needle that occurs once
                # in the text may be found without context.
                if state.find(needle, start, end) >= 0:
                    raise ScriptError(
                        f"{needle!r} (occurrence {occurrence}) is not in the "
                        "narrowed state, which holds another occurrence of it"
                    )
                return None
            lo, hi = max(0, at - margin), min(len(source), at + len(needle) + margin)
            found = state.find(source[lo:hi], start, end)
            if found >= 0:
                begin = found + (at - lo)
                return [begin, begin + len(needle)]
        return None  # the state was narrowed to windows that miss it
    at = start - 1
    for _ in range(occurrence + 1):
        at = state.find(needle, at + 1, end)
        if at < 0:
            return None
    return [at, at + len(needle)]


def all_mentions(state: str, names: Sequence[str]) -> list[tuple[str, list[int]]]:
    """Every non-overlapping occurrence of any of ``names``, longest name first.

    Returns:
        ``(text, [start, end])`` pairs in order of appearance, spans in ``state``.
    """
    start, end = input_bounds(state)
    ordered = sorted(set(names), key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, ordered)) + r")(?!\w)")
    return [
        (m.group(1), [m.start(1), m.end(1)])
        for m in pattern.finditer(state, start, end)
    ]


def build_clairic(
    spec: Mapping[str, Any], state: str, source: str | None = None
) -> dict[str, Any]:
    """Turn a ``says(...)`` description into a schema-valid Clairic answer."""
    data = dict(spec)
    kind = data.pop("type")
    find = data.pop("_find", None)
    find_all = data.pop("_find_all", None)
    if find is not None:
        span = locate(state, find[0], find[1], source)
        if span is None:  # the text is not in this (narrowed) state: no answer
            return clairic_answer(kind, None, p_null=0.99, confidence=0.95)
        data["span"] = span
    if find_all is not None:
        found = []
        for item in find_all:
            text, nth = item if isinstance(item, tuple) else (item, 0)
            found.append((text, locate(state, text, nth, source)))
        present = [(text, span) for text, span in found if span is not None]
        if not present:
            return clairic_answer(kind, None, p_null=0.99, confidence=0.95)
        data["value"] = [text for text, _ in present]
        data["spans"] = [span for _, span in present]
    value = data.pop("value")
    return clairic_answer(kind, value, **data)


# --------------------------------------------------------------------------- #
# One request, as a handler sees it
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Call:
    """A request the library sent, with helpers for answering it.

    Attributes:
        model: Full ``@cf/...`` model id.
        payload: The JSON body (``state`` plus ``fields`` or ``questions``).
        meta: What the library says the call is for (kind, schema, entity, field).
    """

    model: str
    payload: dict[str, Any]
    meta: CallMeta

    @property
    def is_clairic(self) -> bool:
        """Whether this is a Clairic (extraction) request."""
        return self.model == CLAIRIC

    @property
    def state(self) -> Any:
        """The Clairic state text, or the Clef state (string or object)."""
        return self.payload["state"]

    @property
    def asked(self) -> dict[str, Any]:
        """The requested fields (Clairic) or questions (Clef) by id."""
        return dict(self.payload["fields" if self.is_clairic else "questions"])

    @property
    def ids(self) -> list[str]:
        """Ids of the fields or questions, in request order."""
        return list(self.asked)

    @property
    def label(self) -> str | None:
        """The entity asked about (the ``<focus>`` ``entity`` or ``entity.label``)."""
        state = self.state
        if isinstance(state, str):
            found = re.search(r'<focus\b[^>]*\bentity="([^"]*)"', state)
            return found.group(1) if found else None
        entity = state.get("entity") if isinstance(state, dict) else None
        if isinstance(entity, dict):
            label = entity.get("label")
            return label if isinstance(label, str) else None
        return None

    def instructions(self, ident: str) -> str:
        """The instructions the library wrote for one field or question."""
        return str(self.asked[ident].get("instructions", ""))

    @property
    def input_text(self) -> str:
        """The user's text as this request carries it."""
        state = self.state
        if isinstance(state, str):
            start, end = input_bounds(state)
            return state[start:end]
        return str(state.get("input", "")) if isinstance(state, dict) else ""


Handler = Callable[[Call], Mapping[str, Any]]


class _Prebuilt(dict[str, Any]):
    """A Clairic answer that is already complete (it carries its own spans)."""


_MISSING: Any = object()


def _matches(ident: str, key: str) -> bool:
    """Whether a table key stands for a wire name or question id.

    A key is the id itself, its last path segment(s) (``plan`` for
    ``customer.plan``) or a prefix of it (``issue`` for ``issue.__type``), so a
    table keeps working whatever nested names the compiler gives a field.
    """
    return ident == key or ident.endswith("." + key) or ident.startswith(key + ".")


def _find(table: Mapping[str, Any], label: str | None, ident: str) -> Any:
    """Look an id up: ``who::key`` for the entity first, then the plain key."""
    if label is not None:
        low = label.lower()
        for key, value in table.items():
            who, sep, name = key.partition("::")
            if sep and who.lower() in low and _matches(ident, name):
                return value
    if ident in table:
        return table[ident]
    for key, value in table.items():
        if "::" not in key and _matches(ident, key):
            return value
    return _MISSING


def _lookup(table: Mapping[str, Any], label: str | None, key: str, what: str) -> Any:
    found = _find(table, label, key)
    if found is _MISSING:
        raise ScriptError(f"the script has no {what} for {key!r} (entity {label!r})")
    return found


def _consume(value: Any) -> Any:
    """One item of a list per call; the last one repeats."""
    if isinstance(value, list):
        item = value[0]
        if len(value) > 1:
            value.pop(0)
        return item
    return value


def _clef_answer(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        raise TypeError("script a Clef answer as a float or an answer object")
    return noul(float(value)) if isinstance(value, (int, float)) else dict(value)


class OfflineScript:
    """The scripted stand-in for the two models, behind a ``FakeTransport``.

    Attributes:
        transport: Pass this to ``create_extractor(transport=...)``.
        handlers: ``{call kind: handler}``. The built-in handlers answer the
            single-entity kinds from the tables below; set your own for other
            kinds (``script.handlers["gate"] = change_gate(...)``).
        calls: Every ``(model, payload, meta)`` the library sent, in order.
    """

    def __init__(
        self,
        *,
        presence: float | Mapping[str, float] = 0.96,
        clairic: Mapping[str, Mapping[str, Any]] | None = None,
        clef: Mapping[str, Any] | None = None,
        attribution: Mapping[str, float | list[float]] | None = None,
        attribution_default: float = 0.96,
        bleed: Sequence[tuple[str, str, float]] = (),
        reask: Mapping[str, Any] | None = None,
        null_check: Mapping[str, float] | None = None,
        adjudicate: Mapping[str, Mapping[str, Any]] | None = None,
        absent: Sequence[str] = (),
        repair: Mapping[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        """Declare the models' answers.

        Args:
            presence: P(yes) of the presence gate (one number or per schema name).
            clairic: Clairic answers (``says(...)``) by wire name, for ``extract``
                calls. A ``"<label>::<name>"`` key wins for that entity.
            clef: Clef answers by question id for ``clef_fields`` calls (a float is
                a ``noul`` answer, a dict is used as is, see ``pick``).
            attribution: P(yes) by question id (``attr.<wire>``, ``attr.<wire>.<i>``),
                a list being consumed one call at a time.
            attribution_default: P(yes) for every other attribution question.
            bleed: ``(entity label part, value part, p)``: an attribution question
                about a matching entity and value gets ``p`` (the value belongs to
                somebody else).
            reask: Clairic answers by wire name for ``reask`` calls (a list is
                consumed one call at a time).
            null_check: P(yes) by ``nullcheck.<wire>`` question id.
            adjudicate: Choice answers by field wire name for ``adjudicate`` calls.
            absent: Question-id parts (``("shipping_address.__presence",)``) that
                mark a presence question as about something the text never
                mentions; those questions get 0.03. Matched on the id, never on
                the instruction wording, and every question of a request is
                answered (sibling gates may share one request).
            repair: Clairic answers by wire name for ``repair`` calls (the narrow
                re-asks that follow a validation error); a list is consumed one
                call at a time.
            source: The original text. With it, ``says(find=..., occurrence=n)``
                means the n-th match in that text, wherever the library cut the
                state down to windows.
        """
        self.source = source
        self.presence = presence
        self.absent = tuple(absent)
        self.clairic = dict(clairic or {})
        self.clef = dict(clef or {})
        self.attribution = {
            k: list(v) if isinstance(v, list) else v
            for k, v in (attribution or {}).items()
        }
        self.attribution_default = attribution_default
        self.bleed = list(bleed)
        self.reask = {
            k: list(v) if isinstance(v, list) else v for k, v in (reask or {}).items()
        }
        self.repair = {
            k: list(v) if isinstance(v, list) else v for k, v in (repair or {}).items()
        }
        self.null_check = dict(null_check or {})
        self.adjudicate = dict(adjudicate or {})
        self.handlers: dict[str, Handler] = {
            "presence": self._presence,
            "extract": self._extract,
            "crosscheck": self._extract,  # the same reader, minus the record
            "clef_fields": self._clef_fields,
            "attribution": self._attribution,
            "reask": self._reask,
            "repair": self._repair,
            "null_check": self._null_check,
            "stated": self._stated,
            "adjudicate": self._adjudicate,
            "relative_edit": self._relative_edit,
        }
        self.transport = FakeTransport(self._dispatch)

    @property
    def calls(self) -> list[tuple[str, dict[str, Any], CallMeta]]:
        """Every request seen so far."""
        return self.transport.calls

    # -- dispatch ----------------------------------------------------------- #

    def _dispatch(
        self, model: str, payload: dict[str, Any], meta: CallMeta
    ) -> dict[str, Any] | None:
        handler = self.handlers.get(meta.kind)
        if handler is None:
            raise ScriptError(
                f"the script has no handler for {meta.kind!r} calls "
                f"(schema={meta.schema!r}, entity={meta.entity!r}, "
                f"field={meta.field!r})"
            )
        call = Call(model, payload, meta)
        answers = handler(call)
        tokens = max(12, len(json.dumps(payload)) // 4)
        if call.is_clairic:
            state = call.state
            built = {
                ident: spec
                if isinstance(spec, _Prebuilt)
                else build_clairic(spec, state, self.source)
                for ident, spec in answers.items()
            }
            return clairic_response(built, state_text=state, input_tokens=tokens)
        flavour: Literal["clef", "clef-flash"] = (
            "clef-flash" if model == CLEF_FLASH else "clef"
        )
        return clef_response(
            {ident: _clef_answer(a) for ident, a in answers.items()},
            input_tokens=tokens,
            model=flavour,
        )

    # -- built-in handlers (single-entity kinds) ----------------------------- #

    def _presence(self, call: Call) -> Mapping[str, Any]:
        p = self.presence
        if isinstance(p, Mapping):
            p = p[call.meta.schema]
        return {
            qid: 0.03 if any(word.lower() in qid.lower() for word in self.absent) else p
            for qid in call.ids
        }

    def _extract(self, call: Call) -> Mapping[str, Any]:
        return {
            name: _lookup(self.clairic, call.label, name, "Clairic answer")
            for name in call.ids
        }

    def _clef_fields(self, call: Call) -> Mapping[str, Any]:
        answers: dict[str, Any] = {}
        for qid in call.ids:
            answer = _lookup(self.clef, call.label, qid, "Clef answer")
            answers[qid] = answer(call, qid) if callable(answer) else answer
        return answers

    def _attribution(self, call: Call) -> Mapping[str, Any]:
        answers: dict[str, Any] = {}
        label = call.label or ""
        for qid in call.ids:
            text = call.instructions(qid)
            bled = [
                p
                for who, value, p in self.bleed
                if who.lower() in label.lower() and value in text
            ]
            if bled:
                answers[qid] = bled[0]
                continue
            known = _find(self.attribution, call.label, qid)
            if known is _MISSING:
                known = self.attribution_default
            answers[qid] = _consume(known)
        return answers

    def _reask(self, call: Call) -> Mapping[str, Any]:
        return {
            name: _consume(_lookup(self.reask, call.label, name, "re-ask answer"))
            for name in call.ids
        }

    def _repair(self, call: Call) -> Mapping[str, Any]:
        """A narrow re-ask: one field, with the error and the input only."""
        state = call.state
        marker = (
            "<validation_errors>" if isinstance(state, str) else "validation_errors"
        )
        if marker not in state:
            raise ScriptError(
                f"a repair request should carry the validation errors in its state "
                f"(field {call.meta.field!r})"
            )
        if len(call.ids) != 1 or call.meta.field != call.ids[0]:
            raise ScriptError("a repair request asks exactly one field, named in meta")
        return {
            name: _consume(_lookup(self.repair, call.label, name, "repair answer"))
            for name in call.ids
        }

    def _null_check(self, call: Call) -> Mapping[str, Any]:
        return {
            qid: _lookup(self.null_check, call.label, qid, "null check")
            for qid in call.ids
        }

    def _stated(self, call: Call) -> Mapping[str, Any]:
        """ "Does the input state it?" for a Clef answer an update would apply: yes."""
        return {qid: 0.96 for qid in call.ids}

    def _relative_edit(self, call: Call) -> Mapping[str, Any]:
        """A change relative to the old value? The examples state new values: no."""
        return {qid: 0.04 for qid in call.ids}

    def _adjudicate(self, call: Call) -> Mapping[str, Any]:
        assert call.meta.field is not None
        answer = _lookup(self.adjudicate, call.label, call.meta.field, "adjudication")
        return {"adjudicate": answer}


# --------------------------------------------------------------------------- #
# Handlers for the collection engine and the update path (M5, M7-M9)
#
# These follow docs/DESIGN.md ("Collection engine", "Verification ladder"). They
# answer from a small description of the world instead of from wire names, so
# they keep working if the exact ids change; the shapes they read (the
# ``mentions`` object of a coref state with its ``m0``, ``m1`` ... ids, the
# ``criteria`` of a match question, the ``proposed_change`` of a gate state) are
# the ones the engine sends: discovery, count, coreference and qualification since
# M7, matching since M8.
# --------------------------------------------------------------------------- #


def mention_discovery(names: Sequence[str]) -> Handler:
    """Clairic ``list`` of every mention of a known set of names.

    Args:
        names: Every surface form the text uses for the entities, short forms
            included ("Emma Thompson", "Emma"). Pronouns are not mentions.

    Returns:
        A handler for ``discover`` calls: one answer per requested field, a list
        answer with the verbatim mentions and their spans, in order of appearance.
    """

    def handler(call: Call) -> Mapping[str, Any]:
        state = call.state
        found = all_mentions(state, names)
        answer = clairic_answer(
            "list",
            [text for text, _ in found],
            spans=[span for _, span in found],
            p_null=0.01,
            confidence=0.93,
        )
        return {name: _Prebuilt(answer) for name in call.ids}

    return handler


def pair_coreference(entity_of: Mapping[str, str]) -> Handler:
    """Clef ``noul`` for "do these two mentions name the same entity?".

    Args:
        entity_of: Mention text to the entity it names ("Emma" -> "emma").

    Returns:
        A handler for ``coref`` calls. Each question names two mentions of the
        structured state's ``mentions`` object (``m0``, ``m1`` ...) by id.
    """

    def handler(call: Call) -> Mapping[str, Any]:
        state = call.state
        mentions = state.get("mentions", {}) if isinstance(state, dict) else {}
        answers: dict[str, Any] = {}
        for qid in call.ids:
            ids = re.findall(r"m\d+", qid) or re.findall(
                r"m\d+", call.instructions(qid)
            )
            texts = [mentions[i]["text"] for i in ids if i in mentions]
            same = len(texts) == 2 and (
                entity_of.get(texts[0], texts[0]) == entity_of.get(texts[1], texts[1])
            )
            answers[qid] = 0.97 if same else 0.03
        return answers

    return handler


def qualifier(table: Mapping[str, float], default: float = 0.97) -> Handler:
    """Clef ``noul`` for "is this entity really a <Schema>?".

    The engine asks about every entity in one request: ``state["entities"]`` maps
    ``e0``, ``e1`` ... to ``{"label", "mentions"}`` and each question is
    ``qualify.e<n>``.

    Args:
        table: Part of an entity label (case-insensitive) to P(yes).
        default: P(yes) for every other entity.
    """

    def handler(call: Call) -> Mapping[str, Any]:
        state = call.state
        entities = state.get("entities", {}) if isinstance(state, dict) else {}
        answers: dict[str, Any] = {}
        for qid in call.ids:
            label = str(entities.get(qid.removeprefix("qualify."), {}).get("label", ""))
            answers[qid] = next(
                (p for part, p in table.items() if part.lower() in label.lower()),
                default,
            )
        return answers

    return handler


def record_matcher(records: Mapping[str, str | None]) -> Handler:
    """Clef ``choice`` over the existing records plus ``__new__``.

    Args:
        records: Part of an entity label (case-insensitive) to the text that
            identifies its record in the options' descriptions (the record
            rendered as JSON), or ``None`` for an entity with no record.

    Returns:
        A handler for ``match`` calls: picks the option whose description
        contains the identifying text, or ``__new__``.
    """

    def handler(call: Call) -> Mapping[str, Any]:
        label = (call.label or "").lower()
        wanted = next((w for part, w in records.items() if part.lower() in label), None)
        answers: dict[str, Any] = {}
        for qid in call.ids:
            options = dict(call.asked[qid].get("criteria", {}))
            chosen = "__new__"
            if wanted is not None:
                chosen = next(
                    (o for o, text in options.items() if wanted in str(text)),
                    "__new__",
                )
            answers[qid] = pick(chosen, list(options), 0.96)
        return answers

    return handler


def change_gate(
    default: float = 0.93, deny: Mapping[str, float] | None = None
) -> Handler:
    """Clef ``noul`` for "does the input support this proposed change?".

    Args:
        default: P(yes) for every proposed change.
        deny: Part of a proposed change (its JSON text) to the P(yes) it gets
            instead, for changes the input does not support.
    """
    blocked = dict(deny or {})

    def handler(call: Call) -> Mapping[str, Any]:
        state = call.state
        change = state.get("proposed_change") if isinstance(state, dict) else None
        text = json.dumps(change, default=str)
        p = next((p for part, p in blocked.items() if part in text), default)
        return {qid: p for qid in call.ids}

    return handler


def entity_count(n: int) -> Handler:
    """Clairic ``number`` for "how many entities does the text mention?"."""

    def handler(call: Call) -> Mapping[str, Any]:
        return {
            name: says("number", n, p_null=0.01, confidence=0.9) for name in call.ids
        }

    return handler


def choose(keyword: str, p: float = 0.95) -> Callable[[Call, str], dict[str, Any]]:
    """A Clef ``choice`` answer picking the option that mentions ``keyword``.

    Use it as a ``clef=`` entry when the exact option ids are the compiler's
    business (a union of models, say): the answer is whichever option's id or
    description contains the keyword, case-insensitively.
    """

    def answer(call: Call, qid: str) -> dict[str, Any]:
        options = dict(call.asked[qid].get("criteria", {}))
        for option, description in options.items():
            if keyword.lower() in f"{option} {description}".lower():
                return pick(option, list(options), p)
        raise ScriptError(f"no option of {qid!r} mentions {keyword!r}: {options}")

    return answer


def scale(level: int, p: float = 0.7) -> Callable[[Call, str], dict[str, Any]]:
    """A Clef ``score`` answer on an ordered scale; ``level`` (0-based) is likeliest."""

    def answer(call: Call, qid: str) -> dict[str, Any]:
        criteria = call.asked[qid].get("criteria", [])
        n = len(criteria)
        near = [i for i in (level - 1, level + 1) if 0 <= i < n]
        far = [i for i in range(n) if i != level and i not in near]
        left = 1.0 - p
        weights = {i: (0.8 if far else 1.0) * left / len(near) for i in near}
        weights.update({i: 0.2 * left / len(far) for i in far})
        weights[level] = p
        total = sum(weights.values())
        probs = {str(i): round(weights.get(i, 0.0) / total, 4) for i in range(n)}
        expected = sum(i * probs[str(i)] for i in range(n))
        legend = {str(i): str(text) for i, text in enumerate(criteria)}
        return score(round(expected, 3), probs, legend, confidence=p)

    return answer
