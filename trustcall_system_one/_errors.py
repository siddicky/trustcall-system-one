"""Exception hierarchy for trustcall-system-one."""

import re
from typing import Any

__all__ = [
    "AmbiguousMatchError",
    "AuthError",
    "BadRequestError",
    "ExtractionValidationError",
    "ModelResponseError",
    "RateLimitError",
    "RequestTooLargeError",
    "ServerError",
    "StateTruncatedError",
    "TransportError",
    "TrustcallSystemOneError",
    "UncertainOutcomeError",
    "is_request_rejection",
    "is_size_rejection",
]


class TrustcallSystemOneError(Exception):
    """Base class for all trustcall-system-one errors.

    An error that escapes ``Extractor.ainvoke`` or ``invoke`` carries what the
    failed invoke had spent before it failed, so the cost of a failure is not
    lost.

    Attributes:
        calls: The ``CallRecord`` log up to the failure, the failed call included.
        usage: The ``Usage`` totals up to the failure (``None`` outside an invoke).
        cost: The ``CostReport`` for ``usage`` (``None`` outside an invoke).
    """

    calls: tuple[Any, ...] = ()
    usage: Any = None
    cost: Any = None

    # What a transport knows about the failed request (set by the transports and read
    # by the engine's call log; ``None`` when nothing is known).
    provider: str | None = None
    """Who raised it (``cloudflare``, ``experiential`` or ``decisions``)."""
    primary_error: BaseException | None = None
    """When a fallback also failed, the error of the primary that came first."""
    reported_cost: float | None = None
    """The cost in USD that a response reported although it could not be used."""
    served_model: str | None = None
    """The model that was asked, when it is not the one the request named."""
    maybe_sent: bool = False
    """Whether any attempt failed after the request may have reached the server
    (set by ``CloudflareTransport``, which retries: the last failure alone does not
    say whether an earlier attempt may have been run)."""


class TransportError(TrustcallSystemOneError):
    """Base class for HTTP and transport failures.

    Attributes:
        status_code: HTTP status, if a response was received.
        body: The decoded response body, if any.
        retries: Retries performed before giving up (set by the transport).
        retry_after: Seconds the server asked the client to wait, if it did.
        request_id: The id the server gave the request (a correlation or request
            id header), when the transport saw one.
    """

    retries: int = 0
    retry_after: float | None = None
    request_id: str | None = None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: object | None = None,
    ) -> None:
        """Create a transport error."""
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthError(TransportError):
    """Request was rejected as unauthenticated or forbidden (401/403)."""


class RateLimitError(TransportError):
    """Request was rate limited (429) after retries were exhausted."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: object | None = None,
        retry_after: float | None = None,
    ) -> None:
        """Create a rate limit error."""
        super().__init__(message, status_code=status_code, body=body)
        self.retry_after = retry_after


class BadRequestError(TransportError):
    """Request was malformed (400)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: object | None = None,
        cf_errors: list[dict[str, object]] | None = None,
    ) -> None:
        """Create a bad request error."""
        super().__init__(message, status_code=status_code, body=body)
        self.cf_errors: list[dict[str, object]] = [] if cf_errors is None else cf_errors


class RequestTooLargeError(BadRequestError):
    """A Clef request the engine would not send: it is over the size budget.

    Raised by the engine itself, before anything is put on the wire, when a request
    stays over ``limits.clef_state_token_budget`` after its state was narrowed and
    its questions split. The engine treats it like the provider's own "request too
    large" rejection: that one request is dropped (its questions go unanswered)
    with a warning, and the invoke goes on.
    """


class ServerError(TransportError):
    """Server failed (5xx) after retries were exhausted."""


class UncertainOutcomeError(TransportError):
    """The request may have reached the server, so its outcome is unknown.

    Raised by transports that never retry after a request could have been
    received (``ExperientialTransport``): a timeout, a connection that broke
    after the request was sent, or a 5xx. The server may have run the model and
    may have charged for it. Retrying is the caller's decision, because it can
    double the charge.
    """


class ModelResponseError(TrustcallSystemOneError):
    """Model response did not match the published schema.

    Attributes:
        raw: The body that did not match.
        request_id: The id the server gave the request, when the transport saw one.
    """

    request_id: str | None = None

    def __init__(
        self,
        message: str,
        *,
        raw: object | None = None,
        request_id: str | None = None,
    ) -> None:
        """Create a model response error."""
        super().__init__(message)
        self.raw = raw
        self.request_id = request_id


class ExtractionValidationError(TrustcallSystemOneError):
    """Terminal validation failure in strict mode.

    Attributes:
        errors: The pydantic error strings.
        record: The invalid ``ResultRecord``, when the engine attached it.
        calls: The ``CallRecord`` log of the invoke up to the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        errors: list[str] | None = None,
        record: Any = None,
        calls: tuple[Any, ...] = (),
    ) -> None:
        """Create an extraction validation error."""
        super().__init__(message)
        self.errors: list[str] = [] if errors is None else errors
        self.record = record
        self.calls = calls


class StateTruncatedError(TrustcallSystemOneError):
    """Model reported truncated state and narrowing did not fix it."""


class AmbiguousMatchError(TrustcallSystemOneError):
    """An entity could not be matched to an existing record (``on_ambiguous="raise"``).

    Raised after matching and before any extraction request of the entities, so
    nothing was written.

    Attributes:
        schema_name: The schema whose entity was ambiguous.
        label: The entity's label.
        candidates: ``(record id or "__new__", probability)`` of the options, the
            most likely first.
    """

    def __init__(
        self,
        message: str,
        *,
        schema_name: str = "",
        label: str = "",
        candidates: tuple[tuple[str, float], ...] = (),
    ) -> None:
        """Create an ambiguous match error."""
        super().__init__(message)
        self.schema_name = schema_name
        self.label = label
        self.candidates = candidates


_SIZE_WORDS = re.compile(
    r"too (?:large|long|big)|requires \d+ tokens|maximum (?:is|context|number)|"
    r"max(?:imum)? (?:input|context|token)|exceeds?|context (?:length|window)|"
    r"payload too|token limit|too many tokens",
    re.IGNORECASE,
)


def is_size_rejection(exc: BaseException) -> bool:
    """Whether a provider refused one request because of its size.

    HTTP 413, a local ``RequestTooLargeError``, or a 400/422 whose message talks
    about tokens, size or a maximum (Cloudflare: ``Request requires 69639 tokens;
    the maximum is 65529``).
    """
    if isinstance(exc, RequestTooLargeError):
        return True
    if not isinstance(exc, BadRequestError):
        return False
    if exc.status_code == 413:
        return True
    return exc.status_code in (400, 422) and bool(_SIZE_WORDS.search(str(exc)))


def is_request_rejection(exc: BaseException) -> bool:
    """Whether an error says that THIS request was refused, not that the service is.

    A 400, 413 or 422 ``BadRequestError`` (or the engine's own
    ``RequestTooLargeError``) is about one request's content or size: the same
    request would be refused again, but every other request is unaffected, so the
    engine drops that request and goes on. Everything else keeps raising: a rejected
    key (``AuthError``), a rate limit, a server failure, a wrong URL or model (404),
    and above all ``UncertainOutcomeError`` (the request may have run and been
    charged; retrying or guessing is the caller's decision).
    """
    if isinstance(exc, RequestTooLargeError):
        return True
    return isinstance(exc, BadRequestError) and exc.status_code in (400, 413, 422)
