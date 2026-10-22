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
    "exception_chain",
    "is_request_rejection",
    "is_size_rejection",
    "may_have_run",
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
    """A Clef request that was refused locally, before anything was put on the wire.

    Raised by the engine when a request stays over its size budget
    (``Run.clef_budget``) after its state was narrowed and its questions split, and
    by ``TypeSafeTransport`` when a request cannot be made to fit Jev's context (a
    question as large as the context, a state that cannot be narrowed). The engine
    treats it like the provider's own "request too large" rejection
    (``is_size_rejection``): that one request is dropped (its questions go
    unanswered) with a ``request_rejected`` warning, and the invoke goes on. Nothing
    was sent, so it is safe to contain; an error that says ``maybe_sent`` or carries
    a ``partial_usage`` is never treated as one, whatever its class.
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


_SIZE_MESSAGE = re.compile(
    r"requires\s+\d[\d,]*\s+tokens|max_tokens_exceeded|maximum context length",
    re.IGNORECASE,
)
"""The size refusals that are recognised by their text: Cloudflare's ``Request
requires 69639 tokens; the maximum is 65529``, the Decisions API's
``max_tokens_exceeded`` and the common "maximum context length"."""


def exception_chain(exc: BaseException) -> list[BaseException]:
    """An error and its chain: ``primary_error``, ``__cause__``, ``__context__``."""
    seen: list[BaseException] = []
    pending: list[BaseException | None] = [exc]
    while pending:
        link = pending.pop()
        if link is None or any(link is other for other in seen):
            continue
        seen.append(link)
        pending.extend(
            (getattr(link, "primary_error", None), link.__cause__, link.__context__)
        )
    return seen


def may_have_run(exc: BaseException) -> bool:
    """Whether the request behind an error may have been run, and charged.

    True when the error, or anything in its chain (a fallback's error carries the
    primary's as ``primary_error``), is an ``UncertainOutcomeError`` or a
    ``ServerError``, says ``maybe_sent``, or reports a ``partial_usage``. Such a
    request is never contained, resent or split: the caller decides.
    """
    for link in exception_chain(exc):
        if isinstance(link, (UncertainOutcomeError, ServerError)):
            return True
        if getattr(link, "maybe_sent", False):
            return True
        if getattr(link, "partial_usage", None) is not None:
            return True
    return False


def is_size_rejection(exc: BaseException) -> bool:
    """Whether a provider refused one request because of its size.

    HTTP 413, a ``RequestTooLargeError`` (a local refusal: the engine's own, or
    ``TypeSafeTransport``'s when Jev's context cannot hold the request), or a
    400/422 whose message or body is one of the recognised size refusals
    (``_SIZE_MESSAGE``). Nothing else counts: a field error is not a size error
    however it is worded. An error that may have been run and charged
    (``may_have_run``: ``maybe_sent``, ``partial_usage``, an uncertain outcome) is
    never one, whatever its class.
    """
    if not isinstance(exc, BadRequestError) or may_have_run(exc):
        return False
    if isinstance(exc, RequestTooLargeError):
        return True
    if exc.status_code == 413:
        return True
    if exc.status_code not in (400, 422):
        return False
    return bool(_SIZE_MESSAGE.search(f"{exc} {exc.body!s}"))


def is_request_rejection(exc: BaseException) -> bool:
    """Whether an error is a refusal of THIS request that the engine contains.

    Contained by default are only the size refusals (``is_size_rejection``): the
    same request would be refused again, every other request is unaffected, and
    the engine drops that request and goes on. Everything else keeps raising: any
    other 400 or 422 (a malformed field is a bug that every request shares, not a
    one-off), a rejected key (``AuthError``), a rate limit, a server failure, a wrong
    URL or model (404), and above all anything that may have been run and charged
    (``UncertainOutcomeError``, ``ServerError``, ``maybe_sent``, ``partial_usage``,
    also behind a fallback's ``primary_error``): that is the caller's decision.
    """
    return is_size_rejection(exc)
