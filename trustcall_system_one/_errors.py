"""Exception hierarchy for trustcall-system-one."""

from typing import Any

__all__ = [
    "AmbiguousMatchError",
    "AuthError",
    "BadRequestError",
    "ExtractionValidationError",
    "ModelResponseError",
    "RateLimitError",
    "ServerError",
    "StateTruncatedError",
    "TransportError",
    "TrustcallSystemOneError",
    "UncertainOutcomeError",
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


class TransportError(TrustcallSystemOneError):
    """Base class for HTTP and transport failures.

    Attributes:
        status_code: HTTP status, if a response was received.
        body: The decoded response body, if any.
        retries: Retries performed before giving up (set by the transport).
        retry_after: Seconds the server asked the client to wait, if it did.
    """

    retries: int = 0
    retry_after: float | None = None

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
    """Model response did not match the published schema."""

    def __init__(self, message: str, *, raw: object | None = None) -> None:
        """Create a model response error."""
        super().__init__(message)
        self.raw = raw


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
