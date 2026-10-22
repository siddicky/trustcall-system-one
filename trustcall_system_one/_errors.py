"""Exception hierarchy for trustcall-system-one."""

from typing import Any

__all__ = [
    "AuthError",
    "BadRequestError",
    "ExtractionValidationError",
    "ModelResponseError",
    "RateLimitError",
    "ServerError",
    "StateTruncatedError",
    "TransportError",
    "TrustcallSystemOneError",
]


class TrustcallSystemOneError(Exception):
    """Base class for all trustcall-system-one errors."""


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
