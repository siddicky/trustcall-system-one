"""Turn the source of pydantic models into JSON Schemas, safely.

``POST /api/schema/parse`` runs Python that strangers send over the internet, so
nothing is executed before the whole syntax tree has passed ``check_source``, an
allowlist: imports from a few modules, class definitions, annotated and plain
assignments, and expressions made of names, attributes, subscripts, constants,
tuples, lists, dicts, ``|`` and calls of ``Field`` and the other imported
callables. The code then runs with a curated ``__builtins__`` and an ``__import__``
that only knows the same modules (defence in depth).

String annotations (forward references) are refused outside ``Literal[...]``:
pydantic ``eval``s them, which would reach the real builtins.
"""

import ast
import builtins
import sys
import traceback
import types
import uuid
from typing import Any, NoReturn

from fastapi import HTTPException
from pydantic import BaseModel

__all__ = ["MAX_SOURCE_CHARS", "SourceError", "check_source", "parse_models"]

MAX_SOURCE_CHARS = 20_000
"""Longest accepted model source, in characters."""

ALLOWED_MODULES = frozenset(
    {
        "pydantic",
        "typing",
        "typing_extensions",
        "datetime",
        "enum",
        "decimal",
        "uuid",
        "annotated_types",
    }
)
"""Modules the source may import from."""

ALLOWED_NAMES = frozenset(
    {
        # pydantic
        "BaseModel",
        "Field",
        "ConfigDict",
        "AnyUrl",
        "HttpUrl",
        "SecretStr",
        "AwareDatetime",
        "NaiveDatetime",
        "FutureDate",
        "PastDate",
        "PositiveInt",
        "NegativeInt",
        "NonNegativeInt",
        "NonPositiveInt",
        "PositiveFloat",
        "NegativeFloat",
        "NonNegativeFloat",
        "StrictStr",
        "StrictInt",
        "StrictFloat",
        "StrictBool",
        "conint",
        "confloat",
        "constr",
        "conlist",
        # typing and typing_extensions
        "Annotated",
        "Any",
        "ClassVar",
        "Dict",
        "Final",
        "FrozenSet",
        "Iterable",
        "List",
        "Literal",
        "Mapping",
        "Optional",
        "Self",
        "Sequence",
        "Set",
        "Tuple",
        "Union",
        # datetime, decimal, uuid
        "date",
        "datetime",
        "time",
        "timedelta",
        "Decimal",
        "UUID",
        # enum
        "Enum",
        "IntEnum",
        "StrEnum",
        "auto",
        # annotated_types
        "Ge",
        "Gt",
        "Le",
        "Lt",
        "Len",
        "MaxLen",
        "MinLen",
        "MultipleOf",
    }
)
"""Names that may be imported (and, being imported, called). Anything that takes a
callable (validators, serializers, ``TypeAdapter``) or imports by string
(``import_string``) is left out on purpose."""

_BUILTIN_NAMES = (
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "bytes",
)
_FORBIDDEN_NODES = {
    ast.Lambda: "lambda",
    ast.FunctionDef: "def",
    ast.AsyncFunctionDef: "async def",
    ast.ListComp: "comprehension",
    ast.SetComp: "comprehension",
    ast.DictComp: "comprehension",
    ast.GeneratorExp: "generator expression",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
    ast.With: "with",
    ast.AsyncWith: "async with",
    ast.Try: "try",
    ast.While: "while",
    ast.For: "for",
    ast.Starred: "starred expression",
    ast.JoinedStr: "f-string",
    ast.Await: "await",
    ast.Yield: "yield",
    ast.YieldFrom: "yield",
    ast.NamedExpr: "walrus assignment",
}


class SourceError(Exception):
    """A rejection of the submitted source.

    Attributes:
        status: HTTP status of the response.
        code: ``unsafe_source``, ``syntax_error``, ``no_models`` or
            ``invalid_model``.
        message: What is wrong.
        line: 1-based line of the problem, when known.
    """

    def __init__(
        self, code: str, message: str, line: int | None = None, status: int = 400
    ) -> None:
        """Create the error.

        Args:
            code: The error code of the API.
            message: A human-readable explanation.
            line: The line of the problem, if known.
            status: The HTTP status to answer with.
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.line = line
        self.status = status

    def to_http(self) -> HTTPException:
        """Return the ``HTTPException`` that renders as the API's error body."""
        return HTTPException(
            self.status,
            detail={"code": self.code, "message": self.message, "line": self.line},
        )


def _reject(node: ast.AST | None, message: str) -> NoReturn:
    raise SourceError("unsafe_source", message, getattr(node, "lineno", None))


class _Guard:
    """Walks a module and raises ``SourceError`` at the first thing not allowed."""

    def __init__(self) -> None:
        self.callable_names: set[str] = {"Field"}

    # -- statements --------------------------------------------------------- #

    def module(self, tree: ast.Module) -> None:
        for stmt in tree.body:
            self.stmt(stmt)

    def stmt(self, node: ast.stmt) -> None:
        if type(node) in _FORBIDDEN_NODES:
            _reject(node, f"'{_FORBIDDEN_NODES[type(node)]}' is not allowed")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    _reject(node, f"import of '{alias.name}' is not allowed")
                self.name(alias.asname, node)
        elif isinstance(node, ast.ImportFrom):
            self.import_from(node)
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.Expr):
            value = node.value
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                _reject(node, "only docstrings may stand alone as expressions")
        elif isinstance(node, ast.ClassDef):
            self.class_def(node)
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                _reject(node, "only plain names may be annotated")
            self.name(node.target.id, node)
            self.expr(node.annotation, str_ok=False)
            if node.value is not None:
                self.expr(node.value, str_ok=True)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    _reject(node, "only plain names may be assigned")
                self.name(target.id, node)
            self.expr(node.value, str_ok=True)
        else:
            _reject(node, f"'{type(node).__name__}' statements are not allowed")

    def import_from(self, node: ast.ImportFrom) -> None:
        if node.level or node.module not in ALLOWED_MODULES:
            _reject(node, f"import from '{node.module}' is not allowed")
        for alias in node.names:
            if alias.name not in ALLOWED_NAMES:
                _reject(
                    node,
                    f"'{alias.name}' cannot be imported from '{node.module}' here",
                )
            self.name(alias.asname, node)
            self.callable_names.add(alias.asname or alias.name)

    def class_def(self, node: ast.ClassDef) -> None:
        if node.decorator_list:
            _reject(node, "decorators are not allowed")
        if node.keywords:
            _reject(node, "class keywords (such as metaclass=) are not allowed")
        self.name(node.name, node)
        for base in node.bases:
            if not isinstance(base, ast.Name):
                _reject(base, "base classes must be plain names")
            self.name(base.id, base)
        for stmt in node.body:
            self.stmt(stmt)

    # -- expressions -------------------------------------------------------- #

    @staticmethod
    def name(name: str | None, node: ast.AST) -> None:
        if name is not None and name.startswith("__"):
            _reject(node, f"the name '{name}' is not allowed")

    def expr(self, node: ast.expr, *, str_ok: bool) -> None:
        """Check an expression.

        Args:
            node: The expression.
            str_ok: Whether a string constant is plain data here. It is not
                where pydantic would read it as a forward reference and
                ``eval`` it (annotations, subscripts, ``|``).
        """
        if type(node) in _FORBIDDEN_NODES:
            _reject(node, f"'{_FORBIDDEN_NODES[type(node)]}' is not allowed")
        if isinstance(node, ast.Name):
            self.name(node.id, node)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and not str_ok:
                _reject(
                    node,
                    "string annotations (forward references) are not allowed; "
                    "define a model before the models that use it",
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                _reject(node, f"the attribute '{node.attr}' is not allowed")
            self.expr(node.value, str_ok=False)
        elif isinstance(node, ast.Subscript):
            self.expr(node.value, str_ok=False)
            base = node.value
            literal = (isinstance(base, ast.Name) and base.id == "Literal") or (
                isinstance(base, ast.Attribute) and base.attr == "Literal"
            )
            if isinstance(node.slice, ast.Slice):
                _reject(node, "slices are not allowed")
            self.expr(node.slice, str_ok=literal)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self.expr(element, str_ok=str_ok)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    _reject(node, "'**' unpacking is not allowed")
                self.expr(key, str_ok=str_ok)
                self.expr(value, str_ok=str_ok)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, ast.BitOr):
                _reject(node, "only the '|' operator is allowed")
            self.expr(node.left, str_ok=False)
            self.expr(node.right, str_ok=False)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)) or not isinstance(
                node.operand, ast.Constant
            ):
                _reject(node, "only signed number constants are allowed")
        elif isinstance(node, ast.Call):
            self.call(node)
        else:
            _reject(node, f"the expression '{type(node).__name__}' is not allowed")

    def call(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in self.callable_names:
            _reject(
                node,
                "calls are only allowed for Field and the names imported from "
                "pydantic, typing, datetime, enum, decimal, uuid and annotated_types",
            )
        for arg in node.args:
            self.expr(arg, str_ok=True)
        for keyword in node.keywords:
            if keyword.arg is None:
                _reject(node, "'**' unpacking is not allowed")
            self.name(keyword.arg, node)
            self.expr(keyword.value, str_ok=True)


def check_source(source: str) -> ast.Module:
    """Parse the source and verify it against the allowlist.

    Args:
        source: Python source defining pydantic models.

    Returns:
        The verified syntax tree.

    Raises:
        SourceError: ``unsafe_source`` for anything outside the allowlist or
            for a source over ``MAX_SOURCE_CHARS``, and ``syntax_error`` for
            code that does not parse.
    """
    if len(source) > MAX_SOURCE_CHARS:
        raise SourceError(
            "unsafe_source", f"The source is over {MAX_SOURCE_CHARS:,} characters."
        )
    try:
        tree = ast.parse(source, filename="<schema>", mode="exec")
    except SyntaxError as exc:
        raise SourceError("syntax_error", exc.msg, exc.lineno) from exc
    except (RecursionError, MemoryError, ValueError) as exc:
        raise SourceError("unsafe_source", "The source is too deeply nested.") from exc
    try:
        _Guard().module(tree)
    except RecursionError as exc:
        raise SourceError("unsafe_source", "The source is too deeply nested.") from exc
    return tree


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
    for name in _BUILTIN_NAMES:
        safe[name] = getattr(builtins, name)
    return safe


def _failing_line(exc: BaseException) -> int | None:
    """The line of the submitted source where ``exc`` was raised, if any."""
    line: int | None = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<schema>":
            line = frame.lineno
    return line


def parse_models(source: str) -> list[dict[str, Any]]:
    """Run a verified source and return the JSON Schema of each model.

    Args:
        source: Python source defining pydantic models.

    Returns:
        ``{"name", "json_schema"}`` for each top-level ``BaseModel`` subclass, in
        source order (``model_json_schema()``).

    Raises:
        SourceError: If the source is unsafe, does not parse, defines no model or
            fails when it runs.
    """
    tree = check_source(source)
    class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    module_name = f"_demo_schema_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    namespace = module.__dict__
    namespace["__builtins__"] = _safe_builtins()
    sys.modules[module_name] = module  # pydantic resolves annotations in here
    try:
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
        except SourceError:
            raise
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
