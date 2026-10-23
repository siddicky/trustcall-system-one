"""Turn the source of pydantic models into JSON Schemas, safely.

``POST /api/schema/parse`` runs Python that strangers send over the internet. Three
layers stand between that code and the server:

1. ``check_source`` (``app._schema_guard``), an ``ast`` allowlist, rejects anything
   but a few imports, class definitions, simple assignments and expressions built
   from an explicit list of ``module.member`` names. Nothing runs before it passes.
2. The code runs in a separate worker process (``app._schema_worker``): ``python -I``,
   an environment without credentials, a temporary working directory, resource limits
   (CPU, memory, file size, processes) and a wall-clock timeout. The worker checks
   the allowlist again before it executes anything.
3. In the worker, the code runs with a curated ``__builtins__`` and an
   ``__import__`` that knows only the allowlisted modules.

The server process never executes submitted code.
"""

import json
from typing import Any

from fastapi import HTTPException

from app._sandbox import (
    Limits,
    SandboxCrash,
    SandboxTimeout,
    SandboxUnavailable,
    run_worker,
)
from app._schema_guard import MAX_SOURCE_CHARS, SourceError, check_source

__all__ = [
    "MAX_SOURCE_CHARS",
    "SCHEMA_LIMITS",
    "SourceError",
    "check_source",
    "parse_models",
    "run_schema_worker",
    "to_http",
]

SCHEMA_LIMITS = Limits(timeout=5.0, cpu_seconds=3, memory_bytes=512 * 1024 * 1024)
"""Limits of the schema worker: 5 s wall clock, 3 s CPU, 512 MB of address space."""

_WORKER_CODES = frozenset(
    {"unsafe_source", "syntax_error", "no_models", "invalid_model"}
)


def to_http(exc: SourceError) -> HTTPException:
    """Return the ``HTTPException`` that renders as the API's error body."""
    return HTTPException(
        exc.status,
        detail={"code": exc.code, "message": exc.message, "line": exc.line},
    )


def run_schema_worker(
    source: str,
    *,
    limits: Limits = SCHEMA_LIMITS,
    skip_check: bool = False,
) -> list[dict[str, Any]]:
    """Build the schemas of a source in the sandboxed worker process.

    This does not check the source in the server first; ``parse_models`` does.

    Args:
        source: Python source defining pydantic models.
        limits: Resource limits and timeout of the worker.
        skip_check: Turn the worker's own allowlist check off. For tests of the
            process boundary only; the server never sets it.

    Returns:
        ``{"name", "json_schema"}`` for each top-level model, in source order.

    Raises:
        SourceError: The worker refused the source or the model failed
            (``unsafe_source``, ``syntax_error``, ``no_models``, ``invalid_model``);
            ``timeout`` if the worker ran out of time; ``invalid_model`` if it
            crashed; ``sandbox_unavailable`` (503) if it could not be started.
    """
    extra = ("--skip-check-for-tests",) if skip_check else ()
    try:
        answer = run_worker(
            "app._schema_worker",
            json.dumps({"source": source}).encode(),
            limits,
            extra,
        )
    except SandboxTimeout as exc:
        raise SourceError(
            "timeout",
            f"Building the models took too long ({exc}); the limit is "
            f"{limits.timeout:g} s.",
        ) from exc
    except SandboxCrash as exc:
        raise SourceError(
            "invalid_model",
            f"The models could not be built within the sandbox limits ({exc}).",
        ) from exc
    except SandboxUnavailable as exc:
        raise SourceError("sandbox_unavailable", str(exc), status=503) from exc
    if not isinstance(answer, dict):
        raise SourceError("invalid_model", "The worker answered with something odd.")
    if answer.get("ok") is True and isinstance(answer.get("schemas"), list):
        schemas: list[dict[str, Any]] = answer["schemas"]
        return schemas
    code = answer.get("code")
    line = answer.get("line")
    raise SourceError(
        code if code in _WORKER_CODES else "invalid_model",
        str(answer.get("message", "The model could not be built.")),
        line if isinstance(line, int) else None,
    )


def parse_models(source: str) -> list[dict[str, Any]]:
    """Check a source, then build the JSON Schema of each model in a sandbox.

    Args:
        source: Python source defining pydantic models.

    Returns:
        ``{"name", "json_schema"}`` for each top-level ``BaseModel`` subclass, in
        source order (``model_json_schema()``).

    Raises:
        SourceError: If the source is unsafe, does not parse, defines no model, fails
            when it runs or does not finish in time. See ``run_schema_worker``.
    """
    check_source(source)
    return run_schema_worker(source)
