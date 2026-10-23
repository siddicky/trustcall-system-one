"""Worker process of ``POST /api/schema/parse``: check, execute, build the schemas.

Started by ``app.schemas`` through ``app._sandbox.run_worker`` as ``python -I`` with a
stripped environment, resource limits and a timeout. It reads ``{"source": str}`` from
stdin and writes one JSON object to stdout:

* ``{"ok": true, "schemas": [{"name", "json_schema"}, ...]}``
* ``{"ok": false, "code", "message", "line"}``

It checks the source with the same allowlist as the server before it executes
anything (the server's check is only an early rejection), then runs it with a
curated ``__builtins__`` and an ``__import__`` that knows the allowlisted modules.

``--skip-check-for-tests`` turns the allowlist off so that tests can prove that the
process boundary holds on its own. The server never passes it.
"""

import ast
import builtins
import json
import sys
import traceback
import types
import uuid
from typing import Any

from app._sandbox import apply_limits
from app._schema_guard import (
    ALLOWED_MODULES,
    EXPOSED_BUILTINS,
    SourceError,
    check_source,
)

__all__ = ["main"]

SKIP_CHECK_FLAG = "--skip-check-for-tests"


def _safe_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> types.ModuleType:
    """``__import__`` for the executed source: allowlisted modules only."""
    if level != 0 or name not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    safe: dict[str, Any] = {
        "__build_class__": builtins.__build_class__,
        "__name__": "builtins",
        "__import__": _safe_import,
    }
    for name in EXPOSED_BUILTINS:
        safe[name] = getattr(builtins, name)
    return safe


def _failing_line(exc: BaseException) -> int | None:
    """The line of the submitted source where ``exc`` was raised, if any."""
    line: int | None = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<schema>":
            line = frame.lineno
    return line


def build_schemas(tree: ast.Module) -> list[dict[str, Any]]:
    """Execute a syntax tree and return the JSON Schema of each top-level model.

    Args:
        tree: The parsed source.

    Returns:
        ``{"name", "json_schema"}`` for each top-level ``BaseModel`` subclass, in
        source order.

    Raises:
        SourceError: ``invalid_model`` if the source fails when it runs and
            ``no_models`` if it defines no model.
    """
    from pydantic import BaseModel

    class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    module_name = f"_demo_schema_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    namespace = module.__dict__
    namespace["__builtins__"] = _safe_builtins()
    sys.modules[module_name] = module  # pydantic resolves annotations in here
    try:
        exec(compile(tree, "<schema>", "exec"), namespace)
        models = [
            namespace[name]
            for name in class_names
            if isinstance(namespace.get(name), type)
            and issubclass(namespace[name], BaseModel)
            and namespace[name] is not BaseModel
        ]
        schemas = [
            {"name": model.__name__, "json_schema": model.model_json_schema()}
            for model in models
        ]
    except Exception as exc:
        raise SourceError(
            "invalid_model", f"{type(exc).__name__}: {exc}", _failing_line(exc)
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    if not schemas:
        raise SourceError(
            "no_models", "The source defines no pydantic BaseModel subclass."
        )
    return schemas


def _run(source: str, *, skip_check: bool) -> dict[str, Any]:
    try:
        if skip_check:
            tree = ast.parse(source, filename="<schema>", mode="exec")
        else:
            tree = check_source(source)
        return {"ok": True, "schemas": build_schemas(tree)}
    except SourceError as exc:
        return {
            "ok": False,
            "code": exc.code,
            "message": exc.message,
            "line": exc.line,
        }


def main(argv: list[str]) -> int:
    """Entry point: apply the limits, read the request, write the answer.

    Args:
        argv: Limit arguments (see ``app._sandbox.apply_limits``) and, for tests,
            ``--skip-check-for-tests``.

    Returns:
        The exit status.
    """
    rest = apply_limits(argv)
    skip_check = SKIP_CHECK_FLAG in rest
    request = json.loads(sys.stdin.read())
    answer = _run(str(request["source"]), skip_check=skip_check)
    sys.stdout.write(json.dumps(answer, default=str))
    sys.stdout.flush()
    return 0
