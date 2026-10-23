"""The extraction endpoint: request models, provider detection and the run."""

import os
import time
from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from trustcall_system_one import (
    AuthError,
    BadRequestError,
    ExtractionValidationError,
    Many,
    One,
    RateLimitError,
    RequestTooLargeError,
    Transport,
    TransportError,
    TrustcallSystemOneError,
    create_extractor,
    provider_from_env,
)

from app.documents import MAX_CHARS
from app.serialize import serialize_result

__all__ = [
    "ExtractRequest",
    "detect_provider",
    "get_transport",
    "run_extraction",
]

Provider = Literal["cloudflare", "typesafe"]


class SchemaIn(BaseModel):
    """One schema to extract.

    Attributes:
        name: Name of the schema; it becomes the JSON Schema ``title``, which is
            the name the library knows the schema by (``existing[].schema``).
        json_schema: A JSON Schema of type object.
        cardinality: ``one`` extracts a single record, ``many`` any number.
    """

    name: str = Field(min_length=1, max_length=128)
    json_schema: dict[str, Any]
    cardinality: Literal["one", "many"] = "one"


class ExistingIn(BaseModel):
    """An existing record to update.

    Attributes:
        id: The id the caller knows the record by (it comes back as
            ``json_doc_id``).
        schema_name: The ``name`` of the schema it belongs to (JSON key
            ``schema``).
        record: The record.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1)
    schema_name: str = Field(alias="schema")
    record: dict[str, Any]


class OptionsIn(BaseModel):
    """Options of an extraction.

    Attributes:
        enable_inserts: Allow inserting new records next to existing ones.
        enable_deletes: Allow deleting existing records of a ``many`` schema.
        reference_date: Date that relative dates ("last month") resolve against.
    """

    enable_inserts: bool = False
    enable_deletes: bool = False
    reference_date: date | None = None


class ExtractRequest(BaseModel):
    """Body of ``POST /api/extract``.

    Attributes:
        text: The text to extract from (at most 200,000 characters).
        schemas: The schemas to extract.
        existing: Existing records to update.
        options: Extraction options.
    """

    text: str
    schemas: list[SchemaIn] = Field(min_length=1, max_length=16)
    existing: list[ExistingIn] = Field(default_factory=list, max_length=500)
    options: OptionsIn = Field(default_factory=OptionsIn)


def detect_provider(environ: Mapping[str, str] | None = None) -> Provider | None:
    """Find which provider the environment configures, the way the library does.

    Args:
        environ: The environment to read; defaults to ``os.environ``.

    Returns:
        ``"cloudflare"`` when ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN``
        are set, ``"typesafe"`` when ``TRUSTCALL_S1_PROVIDER=typesafe`` and
        ``TYPESAFE_API_KEY`` are set, else ``None``.
    """
    env = os.environ if environ is None else environ
    try:
        chosen = provider_from_env(env)
    except ValueError:
        return None
    if chosen == "typesafe":
        return "typesafe" if (env.get("TYPESAFE_API_KEY") or "").strip() else None
    has_account = (env.get("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    has_token = (env.get("CLOUDFLARE_API_TOKEN") or "").strip()
    return "cloudflare" if has_account and has_token else None


def get_transport() -> Transport | None:
    """Dependency that supplies the transport of an extraction.

    Returns:
        ``None``, so that the library builds its default transport from the
        environment. Tests override this dependency with a ``FakeTransport``.
    """
    return None


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message})


def _library_error(exc: TrustcallSystemOneError) -> HTTPException:
    """Map an error of the library to the API's error response."""
    message = str(exc) or type(exc).__name__
    if isinstance(exc, AuthError):
        return _error(502, "auth_error", message)
    if isinstance(exc, RateLimitError):
        return _error(429, "rate_limited", message)
    if isinstance(exc, RequestTooLargeError):
        return _error(400, "request_too_large", message)
    if isinstance(exc, BadRequestError):
        return _error(400, "bad_request", message)
    if isinstance(exc, ExtractionValidationError):
        return _error(422, "validation_failed", message)
    if isinstance(exc, TransportError):
        return _error(502, "upstream_error", message)
    return _error(502, "extraction_error", message)


async def run_extraction(
    request: ExtractRequest, transport: Transport | None
) -> dict[str, Any]:
    """Run one extraction and serialise its result.

    Args:
        request: The validated request.
        transport: The transport to use; ``None`` for the library's default,
            which needs credentials in the environment.

    Returns:
        ``serialize_result`` of the result, including ``elapsed_ms``.

    Raises:
        HTTPException: 413 for a text over 200,000 characters, 503
            ``no_provider`` without a transport and credentials, 400 for a
            schema or request the library refuses, and the status of
            ``_library_error`` for an error of the library.
    """
    if len(request.text) > MAX_CHARS:
        raise _error(413, "too_large", f"The text is over {MAX_CHARS:,} characters.")
    if transport is None and detect_provider() is None:
        raise _error(
            503,
            "no_provider",
            "No model provider is configured: set CLOUDFLARE_ACCOUNT_ID and "
            "CLOUDFLARE_API_TOKEN, or TRUSTCALL_S1_PROVIDER=typesafe and "
            "TYPESAFE_API_KEY.",
        )
    schemas: list[One | Many] = []
    for item in request.schemas:
        json_schema = {**item.json_schema, "title": item.name}
        schemas.append(
            Many(json_schema) if item.cardinality == "many" else One(json_schema)
        )
    existing = [(e.id, e.schema_name, e.record) for e in request.existing]
    try:
        extractor = create_extractor(
            schemas,
            transport=transport,
            enable_inserts=request.options.enable_inserts,
            enable_deletes=request.options.enable_deletes,
            reference_date=request.options.reference_date,
        )
    except (ValueError, TypeError) as exc:
        raise _error(400, "invalid_schema", str(exc)) from exc
    async with extractor:
        started = time.perf_counter()
        try:
            result = await extractor.ainvoke(request.text, existing=existing or None)
        except TrustcallSystemOneError as exc:
            raise _library_error(exc) from exc
        except ValueError as exc:
            raise _error(400, "invalid_request", str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
    return serialize_result(result, elapsed_ms)
