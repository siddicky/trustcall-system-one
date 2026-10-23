"""``OpenAIResponsesTransport``: the escalation tier's client (OpenAI Responses API).

The default setup (Cloudflare Clairic for extraction, Clef for decisions) is the
whole library; this transport is only used by the opt-in escalation tier
(``EscalationConfig``), which asks ``gpt-6-luna`` about the few cases the default
setup fails. **The text of those cases goes to OpenAI.** A key in the environment
never routes anything here: the tier runs only when it is configured explicitly.

A call is one ``POST {base_url}/v1/responses`` with a strict JSON schema; the reply is
the parsed JSON object. The retry policy is the library's (see
``ExperientialTransport``): a request that may have reached the server is never sent
again, because a timeout can leave a call that was run and charged with its result
lost.
"""

import asyncio
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

import httpx

from trustcall_system_one._config import REASONING_EFFORTS, ConcurrencyConfig
from trustcall_system_one._errors import (
    AuthError,
    BadRequestError,
    ModelResponseError,
    RateLimitError,
    TransportError,
    UncertainOutcomeError,
)
from trustcall_system_one.transport import (
    _DEFAULT_TIMEOUT,
    _UNCERTAIN_NOTE,
    ExperientialTransport,
    _correlation_id,
    _json_or_text,
    _parse_retry_after,
    classify_failure,
)

__all__ = [
    "EFFORT_TIMEOUTS",
    "OPENAI_BASE_URL",
    "OPENAI_DEFAULT_MODEL",
    "REASONING_EFFORTS",
    "OpenAIResponsesTransport",
]

OPENAI_BASE_URL = "https://api.openai.com"
"""Where requests go unless ``base_url`` or ``OPENAI_BASE_URL`` says otherwise."""
OPENAI_DEFAULT_MODEL = "gpt-6-luna"
"""The model the escalation tier asks."""

EFFORT_TIMEOUTS: Mapping[str, float] = {
    "none": 60.0,
    "low": 120.0,
    "medium": 180.0,
    "high": 300.0,
    "xhigh": 600.0,
    "max": 900.0,
}
"""Seconds an attempt may take to answer (the read timeout), by ``reasoning.effort``:
a higher effort thinks longer before the first byte. Overridden by ``timeout`` and by
the ``timeout`` of a call. A timeout after the request was sent is still an uncertain
outcome that is never retried."""

_SYSTEM_ONE = frozenset(
    {"clairic", "clef", "clef-flash", "@cf/cloudflare/clairic", "@cf/cloudflare/clef"}
)
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s|d)")
_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_SAFE_PARAM = re.compile(r"[A-Za-z0-9_.\[\]-]{1,80}")
_SAFE_WORD = re.compile(r"[A-Za-z0-9_.-]{1,80}")


def _duration(value: str | None) -> float | None:
    """Seconds in an ``x-ratelimit-reset-*`` value (``6m0s``, ``20ms``, ``1.5s``)."""
    if not value:
        return None
    text = value.strip()
    parts = _DURATION.findall(text)
    if not parts or _DURATION.sub("", text):
        try:
            seconds = float(text)
        except ValueError:
            return None
        return max(0.0, seconds) if math.isfinite(seconds) else None
    return sum(float(number) * _UNITS[unit] for number, unit in parts)


def _safe(value: object, pattern: re.Pattern[str]) -> str | None:
    """A short identifier from an error object, or ``None`` if it is anything else."""
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _clean_error(body: Any) -> dict[str, str | None]:
    """The ``type``, ``code`` and ``param`` of an OpenAI error, never its ``message``.

    The message of a 4xx can quote the request, which holds the user's text, and a
    401 quotes a part of the key; the three identifiers are short names.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return {"type": None, "code": None, "param": None}
    return {
        "type": _safe(error.get("type"), _SAFE_WORD),
        "code": _safe(error.get("code"), _SAFE_WORD),
        "param": _safe(error.get("param"), _SAFE_PARAM),
    }


class OpenAIResponsesTransport(ExperientialTransport):
    """Asks an OpenAI model for a strict-JSON answer through the Responses API.

    The request is ``POST {base_url}/v1/responses`` with ``Authorization: Bearer
    <key>`` and the body ``{model, reasoning: {effort}, instructions, input,
    text: {format: {type: "json_schema", strict: true, ...}}, store: false}``;
    ``store`` is false so that OpenAI does not keep the text. The payload ``run``
    takes is ``{"input": str, "schema": <JSON schema>, "schema_name": str,
    "instructions": str | None, "top_logprobs": int | None}``; the response body it
    returns is ``{"model", "status", "parsed", "text", "refusal", "no_answer",
    "usage", "logprobs"}``:

    * ``parsed`` is the JSON object the model wrote (``None`` for no answer);
    * a ``refusal`` content item, and a response whose ``status`` is not
      ``completed`` (``incomplete``, ``failed``), are **no answer**: ``parsed`` is
      ``None`` and ``no_answer`` says why. The response arrived and was charged, so
      it is not an uncertain outcome and nothing is retried;
    * ``usage`` is ``{input_tokens, output_tokens, reasoning_tokens}``.

    ``top_logprobs`` asks for ``include: ["message.output_text.logprobs"]`` and returns
    the entries untouched in ``logprobs``. They are off by default and the library
    does not use them: under a strict schema the constrained decoding masks the
    alternatives (``top_logprobs`` holds only the chosen token), so P(yes) against
    P(no) cannot be computed from them (live probe of ``gpt-6-luna``).

    **Retries** (the library's rules): a connect error (nothing was sent) and a 429
    are retried, a 429 honouring ``Retry-After`` and the ``x-ratelimit-reset-*`` hints
    (a 429 whose error code is ``insufficient_quota`` is not: waiting will not clear
    it). A timeout, a dropped connection, a 408 or any 5xx raises
    ``UncertainOutcomeError`` after exactly ONE request. 401 and 403 are
    ``AuthError``, other 4xx ``BadRequestError``.

    **No user text in an error.** The message and the kept body hold the HTTP status,
    OpenAI's ``error.type``, ``error.code`` and ``error.param`` and the request id,
    never ``error.message`` (it can quote the request).

    **The key** is read from ``api_key`` or ``OPENAI_API_KEY`` on each call; it is in
    the httpx request that an ``UncertainOutcomeError`` chains, so scrub
    ``Authorization`` headers and ``sk-`` keys in an error tracker.
    """

    provider = "openai"
    _api_name = "OpenAI API"
    _retry_needs_hint = False
    max_questions = 8
    """Questions per request when this transport is asked about Clef (it serves
    none; the number only keeps the limits it advertises small)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = OPENAI_DEFAULT_MODEL,
        reasoning_effort: str = "none",
        state_token_budget: int = 100_000,
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | None = None,
        max_retries: int = 6,
        backoff_base: float = 0.25,
        backoff_max: float = 8.0,
        max_retry_after: float = 60.0,
        call_deadline: float | str | None = "auto",
        queue_timeout: float | None = None,
        min_attempt_budget: float | None = None,
        concurrency: ConcurrencyConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Create a transport.

        The key is resolved on each ``run``: a missing one raises ``AuthError`` there.

        Args:
            api_key: OpenAI API key; else ``OPENAI_API_KEY``.
            base_url: The API host (``https://api.openai.com``); else
                ``OPENAI_BASE_URL``; ``/v1/responses`` is added. A value that ends
                in ``/v1`` or ``/v1/responses`` is taken as it is.
            model: The model (``gpt-6-luna``).
            reasoning_effort: ``reasoning.effort``: ``none`` (the default), ``low``,
                ``medium``, ``high``, ``xhigh`` or ``max``. A call can carry its own
                (``payload["reasoning_effort"]``, which is what
                ``EscalationConfig.reasoning_effort`` sets).
            state_token_budget: The most tokens of input the escalation tier sends
                in one request.
            client: An ``httpx.AsyncClient`` to use (never closed here).
            timeout: Timeouts of the clients this transport creates. Without one the
                read timeout follows the effort (``EFFORT_TIMEOUTS``: 60 s for
                ``none`` up to 900 s for ``max``) and the connect timeout is 10 s.
            max_retries: Retries for what is safe to retry (connect errors, 429).
            backoff_base: Smallest delay before a retry without a hint, in seconds.
            backoff_max: Largest jittered delay.
            max_retry_after: Longest wait asked for that is honoured; a longer one
                fails at once with ``RateLimitError``.
            call_deadline: Total seconds across attempts and waits, ``None`` for no
                deadline, or ``"auto"`` (the default): 300 s, or 2.5 times the
                effort's read timeout when that is more.
            queue_timeout: Longest wait for a limiter slot, or ``None``.
            min_attempt_budget: See ``ExperientialTransport``.
            concurrency: Concurrency limits of the adaptive limiter.
            sleep: Awaitable sleep, injectable for tests.
            clock: Monotonic clock, injectable for tests.
            extra_headers: Headers added to the defaults.

        Raises:
            ValueError: If a retry, backoff or budget setting is out of range.
        """
        host = (
            base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
        ).strip()
        host = host.rstrip("/")
        if host.endswith("/responses"):
            url = host
        elif host.endswith("/v1"):
            url = f"{host}/responses"
        else:
            url = f"{host}/v1/responses"
        if state_token_budget < 1:
            raise ValueError("state_token_budget must be at least 1")
        _check_effort(reasoning_effort)
        read = EFFORT_TIMEOUTS[reasoning_effort]
        self._follows_effort = timeout is None
        self._auto_deadline = call_deadline == "auto"
        if timeout is None:
            timeout = httpx.Timeout(read, connect=_DEFAULT_TIMEOUT.connect)
        elif timeout.read is not None:
            read = float(timeout.read)
        if call_deadline == "auto":
            deadline: float | None = max(300.0, 2.5 * read)
        elif call_deadline is None or isinstance(call_deadline, (int, float)):
            deadline = None if call_deadline is None else float(call_deadline)
        else:
            raise ValueError("call_deadline must be a number, None or 'auto'")
        super().__init__(
            api_key=api_key,
            base_url=url,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            max_retry_after=max_retry_after,
            call_deadline=deadline,
            queue_timeout=queue_timeout,
            min_attempt_budget=min_attempt_budget,
            concurrency=concurrency,
            sleep=sleep or asyncio.sleep,
            clock=clock or time.monotonic,
            extra_headers=extra_headers,
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.state_token_budget = state_token_budget

    # -- what it serves ---------------------------------------------------- #

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        """Whether this transport answers ``model``: anything but Clairic and Clef."""
        return model not in _SYSTEM_ONE and bool(model)

    @staticmethod
    def _check_model(model: str) -> None:
        if model in _SYSTEM_ONE:
            raise BadRequestError(
                "OpenAIResponsesTransport serves OpenAI models, not "
                f"{model!r}; Clairic and Clef go through the main transport"
            )

    def classify_failure(self, exc: Exception) -> str | None:
        """Say whether a failure means nothing was run (the library's categories)."""
        return classify_failure(exc)

    def _token(self) -> str:
        token = (self._api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not token:
            raise AuthError("No OpenAI API key: pass api_key or set OPENAI_API_KEY")
        return token

    def _served_model(self, body: Mapping[str, Any]) -> str | None:
        return str(body.get("model") or self.model)

    def _error_model(self) -> str | None:
        return self.model

    # -- request ----------------------------------------------------------- #

    def _prepare(self, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        schema = payload.get("schema")
        text = payload.get("input")
        if not isinstance(schema, Mapping) or not isinstance(text, str):
            raise BadRequestError(
                "an OpenAI request needs an 'input' string and a 'schema' object"
            )
        name = str(payload.get("schema_name") or "answer")
        name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "answer"
        body: dict[str, Any] = {
            "model": payload.get("model") or self.model,
            "input": text,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": dict(schema),
                }
            },
            "store": False,
        }
        effort = payload.get("reasoning_effort") or self.reasoning_effort
        _check_effort(effort)
        body["reasoning"] = {"effort": effort}
        if payload.get("instructions"):
            body["instructions"] = str(payload["instructions"])
        if payload.get("max_output_tokens"):
            body["max_output_tokens"] = int(payload["max_output_tokens"])
        top = payload.get("top_logprobs")
        if isinstance(top, int) and not isinstance(top, bool) and top > 0:
            body["include"] = ["message.output_text.logprobs"]
            body["top_logprobs"] = min(top, 20)
        return body, None

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: Any,
        timeout: float | None = None,
    ) -> Any:
        """Run one call (see the class docstring); ``payload["model"]`` is ``model``."""
        self._check_model(model)
        outgoing, context = self._prepare({**payload, "model": model})
        deadline: float | Literal["default"] | None = "default"
        if timeout is None and self._injected_client is None and self._follows_effort:
            # the read timeout, and the "auto" deadline, follow the effort the call
            # runs at (its own, else the transport's); a ``timeout`` the caller gave
            # the transport, or this call, is never raised or lowered
            effort = payload.get("reasoning_effort") or self.reasoning_effort
            _check_effort(effort)
            timeout = EFFORT_TIMEOUTS[effort]
            if self._auto_deadline:
                deadline = max(300.0, 2.5 * timeout)
        del meta  # nothing about the call leaves the process
        return await self._post(
            model, outgoing, context, timeout=timeout, call_deadline=deadline
        )

    # -- response ---------------------------------------------------------- #

    def _read_success(self, response: httpx.Response, context: Any) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelResponseError(
                "OpenAI API returned a non-JSON success response"
            ) from exc
        if not isinstance(body, dict):
            raise ModelResponseError("OpenAI API response is not a JSON object")
        raw_usage = body.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        details = usage.get("output_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else 0
        out: dict[str, Any] = {
            "model": str(body.get("model") or self.model),
            "status": body.get("status"),
            "parsed": None,
            "text": None,
            "refusal": None,
            "no_answer": None,
            "logprobs": None,
            "usage": {
                "input_tokens": _count(usage.get("input_tokens")),
                "output_tokens": _count(usage.get("output_tokens")),
                "reasoning_tokens": _count(reasoning),
            },
        }
        status = body.get("status")
        if status != "completed":  # incomplete, failed, cancelled or not stated
            incomplete = body.get("incomplete_details")
            why = incomplete.get("reason") if isinstance(incomplete, dict) else None
            shown = _safe(status, _SAFE_WORD) or "missing"
            out["no_answer"] = f"status {shown}" + (
                f" ({_safe(why, _SAFE_WORD)})" if _safe(why, _SAFE_WORD) else ""
            )
            return out
        text: str | None = None
        for item in body.get("output") or []:
            if not (isinstance(item, dict) and item.get("type") == "message"):
                continue
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and text is None:
                    if isinstance(part.get("text"), str):
                        text = part["text"]
                        entries = part.get("logprobs")
                        out["logprobs"] = entries if entries else None
                elif part.get("type") == "refusal":
                    out["refusal"] = str(part.get("refusal") or "")[:200]
        if out["refusal"] is not None and text is None:
            out["no_answer"] = "refusal"
            return out
        if text is None:
            out["no_answer"] = "no output_text in the response"
            return out
        out["text"] = text
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise ModelResponseError(
                "OpenAI API answered text that is not JSON under a strict schema"
            ) from exc
        if not isinstance(parsed, dict):
            raise ModelResponseError("OpenAI API answered JSON that is not an object")
        out["parsed"] = parsed
        return out

    # -- errors and retries ------------------------------------------------ #

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """The wait a 429 asks for: ``Retry-After``, else the ``x-ratelimit-reset-*``.

        When a ``remaining`` header says a resource is exhausted, the longest of the
        exhausted resources' resets; otherwise the shortest reset that is given.
        """
        explicit = _parse_retry_after(response.headers.get("Retry-After"))
        if explicit is not None:
            return explicit
        resets: dict[str, float] = {}
        for resource in ("requests", "tokens"):
            seconds = _duration(response.headers.get(f"x-ratelimit-reset-{resource}"))
            if seconds is not None:
                resets[resource] = seconds
        if not resets:
            return None
        spent = [
            r
            for r in resets
            if (response.headers.get(f"x-ratelimit-remaining-{r}") or "").strip() == "0"
        ]
        if spent:
            return max(resets[r] for r in spent)
        return min(resets.values())

    def _refused_unprocessed(self, response: httpx.Response) -> bool:
        if response.status_code != 429:
            return False
        # an empty quota does not refill by waiting
        return _clean_error(_json_or_text(response)).get("code") != "insufficient_quota"

    @classmethod
    def _error_for_status(
        cls, response: httpx.Response, *, retry_after: float | None
    ) -> TransportError:
        status = response.status_code
        body = _json_or_text(response)
        clean = _clean_error(body)
        shown = (
            ", ".join(f"{k}={v}" for k, v in clean.items() if v) or "no error object"
        )
        message = f"{cls._api_name} returned HTTP {status}: {shown}"
        correlation = _correlation_id(response)
        if correlation:
            message += f" (request id {correlation})"
        kept = {"error": clean}
        error: TransportError
        if status in (401, 403):
            error = AuthError(message, status_code=status, body=kept)
        elif status == 429:
            error = RateLimitError(
                message, status_code=status, body=kept, retry_after=retry_after
            )
        elif status == 408 or status >= 500:
            error = UncertainOutcomeError(
                f"{message}. {_UNCERTAIN_NOTE}", status_code=status, body=kept
            )
        elif 300 <= status < 500:
            error = BadRequestError(
                message,
                status_code=status,
                body=kept,
                cf_errors=[{k: v for k, v in clean.items() if v}],
            )
        else:
            error = TransportError(message, status_code=status, body=kept)
        error.retry_after = retry_after
        error.request_id = correlation
        return error


def _check_effort(effort: object) -> None:
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {list(REASONING_EFFORTS)}, got {effort!r}"
        )


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
