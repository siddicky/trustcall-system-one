"""Scripted model replies for the extractor tests."""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from trustcall_system_one import CallMeta, Extractor, create_extractor
from trustcall_system_one._wire import CLAIRIC
from trustcall_system_one.testing import (
    FakeTransport,
    choice,
    clairic_answer,
    clairic_response,
    clef_response,
    noul,
    route,
)

__all__ = ["Script", "ca", "kind_counts", "kinds", "make_extractor", "pick"]


def ca(
    type_: str,
    value: Any,
    *,
    find: str | None = None,
    find_all: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A scripted Clairic answer; ``find`` is located in the state for the span."""
    spec: dict[str, Any] = {"type": type_, "value": value, **extra}
    if find is not None:
        spec["_find"] = find
    if find_all is not None:
        spec["_find_all"] = find_all
    return spec


def _locate(state: str, needle: str) -> list[int]:
    start = state.index(needle, state.index("<input>"))
    return [start, start + len(needle)]


def _build(spec: Mapping[str, Any], state: str) -> dict[str, Any]:
    data = dict(spec)
    find = data.pop("_find", None)
    find_all = data.pop("_find_all", None)
    if find is not None:
        data["span"] = _locate(state, find)
    if find_all is not None:
        data["spans"] = [_locate(state, needle) for needle in find_all]
    return clairic_answer(data.pop("type"), data.pop("value"), **data)


def pick(option: str, options: list[str], p: float = 0.95) -> dict[str, Any]:
    """A choice answer for ``option`` with the rest sharing what is left."""
    rest = (1 - p) / max(1, len(options) - 1)
    return choice(option, {o: (p if o == option else rest) for o in options})


@dataclass
class Script:
    """A scripted world that answers every call kind of an insert.

    Attributes:
        presence: Probability returned by the presence gate.
        clairic: Clairic answers by wire name (see ``ca``) for ``extract`` calls.
        clef: Clef answers by question id for ``clef_fields`` calls.
        attribution: Attribution probability by question id (a list is consumed
            one call at a time, the last value repeating); ``attribution_default``
            answers the rest.
        reask: Clairic answers by wire name for ``reask`` calls (a list is
            consumed one call at a time).
        null_check: Probability by question id for ``null_check`` calls.
        adjudicate: Choice answers by field wire name for ``adjudicate`` calls.
        truncated: Number of ``extract`` calls to flag as truncated.
    """

    presence: float | None = 0.95
    clairic: dict[str, dict[str, Any]] = field(default_factory=dict)
    clef: dict[str, dict[str, Any]] = field(default_factory=dict)
    attribution: dict[str, float | list[float]] = field(default_factory=dict)
    attribution_default: float = 0.95
    reask: dict[str, dict[str, Any] | list[dict[str, Any]]] = field(
        default_factory=dict
    )
    null_check: dict[str, float] = field(default_factory=dict)
    adjudicate: dict[str, dict[str, Any]] = field(default_factory=dict)
    truncated: int = 0
    state_text: Callable[[str], str] | None = None
    clef_input_tokens: int = 100
    transport: FakeTransport = field(init=False)

    def __post_init__(self) -> None:
        self.transport = FakeTransport(
            route(
                presence=self._presence,
                extract=self._extract,
                clef_fields=self._clef_fields,
                attribution=self._attribution,
                reask=self._reask,
                null_check=self._null_check,
                adjudicate=self._adjudicate,
            )
        )

    # ---- handlers ------------------------------------------------------ #

    def _presence(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        assert self.presence is not None
        return clef_response({"presence": noul(self.presence)})

    def _extract(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        assert model == CLAIRIC
        state = payload["state"]
        if self.state_text is not None:
            state = self.state_text(state)  # what Clairic says it read
        answers = {
            name: _build(self.clairic[name], state) for name in payload["fields"]
        }
        truncated = self.truncated > 0
        self.truncated -= 1
        return clairic_response(answers, truncated=truncated, state_text=state)

    def _clef_fields(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        questions = payload["questions"]
        return clef_response(
            {qid: self.clef[qid] for qid in questions},
            input_tokens=self.clef_input_tokens,
        )

    def _attribution(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        answers = {}
        for qid in payload["questions"]:
            value = self.attribution.get(qid, self.attribution_default)
            if isinstance(value, list):
                p = value[0]
                if len(value) > 1:
                    value.pop(0)
            else:
                p = value
            answers[qid] = noul(p)
        return clef_response(answers)

    def _reask(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        state = payload["state"]
        answers = {}
        for name in payload["fields"]:
            spec = self.reask[name]
            if isinstance(spec, list):
                chosen = spec[0]
                if len(spec) > 1:
                    spec.pop(0)
            else:
                chosen = spec
            answers[name] = _build(chosen, state)
        return clairic_response(answers, state_text=state)

    def _null_check(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        return clef_response(
            {qid: noul(self.null_check[qid]) for qid in payload["questions"]}
        )

    def _adjudicate(self, model: str, payload: dict[str, Any], meta: CallMeta) -> Any:
        assert meta.field is not None
        return clef_response({"adjudicate": self.adjudicate[meta.field]})

    # ---- inspection ----------------------------------------------------- #

    def calls(self, kind: str) -> list[tuple[str, dict[str, Any], CallMeta]]:
        """The recorded calls of one kind."""
        return [c for c in self.transport.calls if c[2].kind == kind]


def kinds(transport: FakeTransport) -> list[str]:
    """The call kinds a transport saw, in order."""
    return [meta.kind for _, _, meta in transport.calls]


def kind_counts(transport: FakeTransport) -> Counter[str]:
    """How many calls of each kind a transport saw."""
    return Counter(kinds(transport))


def make_extractor(schemas: list[Any], script: Script, **kwargs: Any) -> Extractor:
    """Create an extractor on the script's fake transport."""
    return create_extractor(schemas, transport=script.transport, **kwargs)
