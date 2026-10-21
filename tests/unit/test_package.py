"""Tests for the package skeleton and the exception hierarchy."""

import importlib

from trustcall_system_one import (
    AuthError,
    BadRequestError,
    ExtractionValidationError,
    ModelResponseError,
    RateLimitError,
    ServerError,
    StateTruncatedError,
    TransportError,
    TrustcallSystemOneError,
)


def test_version() -> None:
    package = importlib.import_module("trustcall_system_one")
    assert package.__version__ == "0.1.0a0"


def test_all_names_are_importable() -> None:
    package = importlib.import_module("trustcall_system_one")
    assert package.__all__
    for name in package.__all__:
        assert hasattr(package, name), f"{name} is listed in __all__ but missing"


def test_transport_errors_subclass_transport_error() -> None:
    for cls in (AuthError, RateLimitError, BadRequestError, ServerError):
        assert issubclass(cls, TransportError)


def test_top_level_errors_subclass_base() -> None:
    for cls in (
        TransportError,
        ModelResponseError,
        ExtractionValidationError,
        StateTruncatedError,
    ):
        assert issubclass(cls, TrustcallSystemOneError)


def test_rate_limit_error_exposes_retry_after() -> None:
    err = RateLimitError("x", status_code=429, retry_after=2.0)
    assert err.retry_after == 2.0
    assert err.status_code == 429


def test_bad_request_error_defaults_cf_errors_to_empty_list() -> None:
    assert BadRequestError("x").cf_errors == []


def test_model_response_error_exposes_raw() -> None:
    assert ModelResponseError("x", raw={"a": 1}).raw == {"a": 1}
