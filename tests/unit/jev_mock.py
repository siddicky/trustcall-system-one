"""A scripted Jev for tests: an ``httpx`` handler that answers from a ``FakeWorld``.

``JevBackend`` plays the TypeSafe API. It never sees the engine's Clairic or Clef
request (the transport turns them into Jev requests), so ``Bridge`` wraps the
transport under test and leaves the request the engine made in a context variable;
the backend then asks the world what the *ideal* Clairic/Clef answer is and answers
the Jev questions the transport really sent so that they say the same:

* a Clef request: the questions must be the engine's, unchanged, and the answer is
  the world's;
* the stitch questions: no line joins the next (0.04);
* a ``choice`` of a candidate: the option whose id or description is the value the
  world has, else ``none``;
* the yes/no of a list candidate: yes for the elements the world lists;
* date parts: the parts of the ISO date the world has;
* the verification: nothing is wrong (0.05).

The world is the ground truth, the transport is under test, and Jev is the only
thing replaced.
"""

import contextvars
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import jsonschema

from evals.model import WorldLike
from trustcall_system_one import CLAIRIC, CLEF, CallMeta, Transport, TransportResponse
from trustcall_system_one._candidates import pattern_candidates
from trustcall_system_one._dateparts import (
    DATE_PARTS,
    MONTH_NAMES,
)
from trustcall_system_one._wire import ClairicField

MODEL = "jev-1.14.0"
_OPENAPI = json.loads(
    (
        Path(__file__).resolve().parents[2] / "docs/schemas/typesafe-openapi.json"
    ).read_text()
)


def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        {"$ref": f"#/components/schemas/{name}", "components": _OPENAPI["components"]}
    )


REQUEST_SCHEMA = _validator("SystemOneRequest")
RESPONSE_SCHEMA = _validator("SystemOneResponse")
_SENTENCE = re.compile(r"\[(S\d+)\] ((?:(?!\[S\d+\]).)*)")


def _choice(options: list[str], pick: str, p: float = 0.95) -> dict[str, Any]:
    rest = (1 - p) / max(1, len(options) - 1)
    return {
        "type": "choice",
        "choice": pick,
        "confidence": p,
        "probabilities": {o: (p if o == pick else rest) for o in options},
    }


def _number(text: str) -> float | None:
    cleaned = re.sub(r"[^0-9.\-]", "", text.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


class JevBackend:
    """The handler of a ``httpx.MockTransport``; ``requests`` logs what it got."""

    def __init__(self, world: WorldLike) -> None:
        self.world = world
        self._fake = world.transport()
        self.current: contextvars.ContextVar[
            tuple[str, dict[str, Any], CallMeta, dict[str, Any]]
        ] = contextvars.ContextVar("jev_current")
        self.requests: list[dict[str, Any]] = []
        self.overrides: dict[
            str, Callable[[dict[str, Any]], dict[str, Any] | None]
        ] = {}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        errors = [e.message[:200] for e in REQUEST_SCHEMA.iter_errors(body)]
        assert not errors, f"not a valid TypeSafe request: {errors}"
        assert request.headers["authorization"].startswith("Bearer ")
        model, payload, meta, cache = self.current.get()
        if model != CLAIRIC:
            assert body["questions"] == payload["questions"], (
                "Clef went through changed"
            )
            assert body["state"] == payload["state"]
            sent = await self._fake.run(model, payload, meta=meta)
            reply = {
                "model": MODEL,
                "answers": sent.body["answers"],
                "usage": sent.body["usage"],
            }
            assert not list(RESPONSE_SCHEMA.iter_errors(reply))
            return httpx.Response(200, json=reply)
        answers = await self._clairic(body, payload, meta, cache)
        usage = {"input_tokens": 100 + len(request.content) // 4, "output_tokens": 10}
        return httpx.Response(
            200, json={"model": MODEL, "answers": answers, "usage": usage}
        )

    async def _true_answers(
        self, payload: dict[str, Any], meta: CallMeta, cache: dict[str, Any]
    ) -> dict[str, Any]:
        if "truth" not in cache:
            sent = await self._fake.run(CLAIRIC, payload, meta=meta)
            cache["truth"] = sent.body["answers"]
        return cache["truth"]  # type: ignore[no-any-return]

    async def _clairic(
        self,
        body: dict[str, Any],
        payload: dict[str, Any],
        meta: CallMeta,
        cache: dict[str, Any],
    ) -> dict[str, Any]:
        truth = await self._true_answers(payload, meta, cache)
        state = body["state"]
        text = state["input"] if isinstance(state, dict) and "input" in state else ""
        sentences = {sid: t.strip() for sid, t in _SENTENCE.findall(text)}
        out: dict[str, Any] = {}
        for qid, question in body["questions"].items():
            if qid in self.overrides or any(qid.endswith(s) for s in self.overrides):
                key = next(k for k in self.overrides if qid == k or qid.endswith(k))
                given = self.overrides[key](question)
                if given is not None:
                    out[qid] = given
                    continue
            out[qid] = self._answer(qid, question, truth, sentences)
        return out

    def _answer(
        self,
        qid: str,
        question: dict[str, Any],
        truth: dict[str, Any],
        sentences: dict[str, str],
    ) -> dict[str, Any]:
        if re.fullmatch(r"L\d+", qid):  # stitch
            return {"type": "noul", "noul": 0.04}
        name, _, rest = qid.partition("::")
        if rest in ("hallucinated", "off_target"):
            return {"type": "noul", "noul": 0.05}
        answer = truth.get(name, {})
        value = answer.get("value")
        if question["type"] == "noul":
            wanted = value if isinstance(value, list) else []
            cand = question["instructions"]["candidate"]
            piece = cand.get("text") or sentences.get(cand.get("id", ""), "")
            hit = any(str(w).casefold() == piece.casefold() for w in wanted)
            return {"type": "noul", "noul": 0.95 if hit else 0.03}
        options = list(question["criteria"])
        if rest.endswith(("pick", "final")) or re.fullmatch(r"pick\d+", rest):
            return _choice(options, self._pick(question, value, sentences))
        return self._part(rest, options, value)

    @staticmethod
    def _pick(question: dict[str, Any], value: Any, sentences: dict[str, str]) -> str:
        options: dict[str, Any] = question["criteria"]
        if value is None:
            return "none"
        text = str(value)
        if text in options:
            return text
        for oid in options:
            if sentences.get(oid, "").casefold() == text.casefold():
                return str(oid)
        for oid, description in options.items():
            if isinstance(description, str) and f"'{text}'" in description:
                return oid
        if isinstance(value, (int, float)):
            for oid, description in options.items():
                if isinstance(description, str) and description.startswith(
                    f"{value} ("
                ):
                    return str(oid)  # a sum: "1080 (the sum of ...)"
            for oid in options:
                found = pattern_candidates(ClairicField(type="number"), oid)
                if any(c.text == oid and c.value == value for c in found):
                    return str(oid)
        return "none"

    @staticmethod
    def _part(part: str, options: list[str], value: Any) -> dict[str, Any]:
        part = part.removeprefix("t.")
        if part == "mode" and "absolute" in options:
            return _choice(options, "none" if value is None else "absolute")
        if value is None or not isinstance(value, str) or part not in DATE_PARTS:
            return _choice(options, "none")
        year, month, day = value[:10].split("-")
        picks = {
            "month": MONTH_NAMES[int(month) - 1],
            "day": str(int(day)),
            "year": year,
        }
        return _choice(options, picks.get(part, "none"))


class Bridge:
    """Wraps the transport under test: leaves the engine's request for the backend."""

    def __init__(self, inner: Transport, backend: JevBackend) -> None:
        self.inner = inner
        self.backend = backend

    def max_questions(self, model: str) -> int:
        return int(getattr(self.inner, "max_questions", 64))

    async def run(
        self,
        model: str,
        payload: Any,
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        token = self.backend.current.set((model, dict(payload), meta, {}))
        try:
            return await self.inner.run(model, payload, meta=meta, timeout=timeout)
        finally:
            self.backend.current.reset(token)


Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]
__all__ = ["CLEF", "MODEL", "Bridge", "JevBackend"]
