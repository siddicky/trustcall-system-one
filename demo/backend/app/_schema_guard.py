"""The syntax-tree allowlist for submitted pydantic model source.

``POST /api/schema/parse`` runs Python that strangers send over the internet. This
module is the first of its three layers: nothing is executed before the whole tree
has passed ``check_source``. The check runs in the server (to reject early) and
again in the worker process, right before ``exec`` (``app._schema_worker``).

It imports nothing outside the standard library, so the worker can load it without
FastAPI.

What the allowlist is made of:

* Imports. ``import m`` and ``from m import n`` for the modules and members in
  ``MODULE_MEMBERS`` only.
* Attribute access, only as ``module.member`` on a module name the source imported
  (the member must be listed for that module) or ``EnumClass.MEMBER`` on an enum
  class defined earlier in the source. No chains, no other bases. This is what
  keeps ``uuid.os`` or ``datetime.sys`` out: allowed modules re-export other
  modules.
* Names are bound once. No assignment, annotation with a value, class, import
  alias or ``def`` may rebind an imported name, an allowlisted name, a module name
  or a builtin the executed code sees.
* Top-level assignments bind fresh names to constants or simple literals.
* Calls only of names imported from an allowed module and of ``module.member``
  attributes. ``default_factory`` accepts ``list``, ``dict`` or ``set`` only.
* Expressions: names, constants, tuples, lists, dicts, subscripts, ``|``, signed
  numbers.

Everything else is rejected: dunder names and attributes, ``lambda``, ``def``,
comprehensions, decorators, ``global``/``nonlocal``, ``with``/``try``/``while``/
``for``, starred expressions, f-strings, other operators and string annotations
(pydantic ``eval``s them).
"""

import ast
from typing import NoReturn

__all__ = [
    "ALLOWED_MODULES",
    "EXPOSED_BUILTINS",
    "MAX_SOURCE_CHARS",
    "MODULE_MEMBERS",
    "SourceError",
    "check_source",
]

MAX_SOURCE_CHARS = 20_000
"""Longest accepted model source, in characters."""

_PYDANTIC = frozenset(
    {
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
    }
)
_TYPING = frozenset(
    {
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
    }
)

MODULE_MEMBERS: dict[str, frozenset[str]] = {
    "pydantic": _PYDANTIC,
    "typing": _TYPING,
    "typing_extensions": _TYPING,
    "datetime": frozenset({"date", "datetime", "time", "timedelta"}),
    "decimal": frozenset({"Decimal"}),
    "uuid": frozenset({"UUID"}),
    "enum": frozenset({"Enum", "IntEnum", "StrEnum", "auto"}),
    "annotated_types": frozenset(
        {"Ge", "Gt", "Le", "Lt", "Len", "MaxLen", "MinLen", "MultipleOf"}
    ),
}
"""Per module, the members that may be imported or accessed as ``module.member``.
Anything that takes a callable (validators, serializers, ``TypeAdapter``) or imports
by string (``import_string``) is left out on purpose."""

ALLOWED_MODULES = frozenset(MODULE_MEMBERS)
"""Modules the source may import."""

EXPOSED_BUILTINS = (
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
"""The builtins the executed code can see (the worker exposes exactly these, plus
``__build_class__`` and a restricted ``__import__``)."""

_ALLOWLISTED_NAMES = frozenset(
    name for members in MODULE_MEMBERS.values() for name in members
)
# Allowlisted names that are also plausible field names; a class body may bind them
# when the source did not import them (nothing is rebound then). Everywhere else
# every allowlisted name is reserved.
_FIELD_NAME_OK = frozenset({"date", "time", "datetime", "timedelta", "auto"})
_RESERVED = _ALLOWLISTED_NAMES | ALLOWED_MODULES | frozenset(EXPOSED_BUILTINS)
_RESERVED_IN_CLASS = (_ALLOWLISTED_NAMES - _FIELD_NAME_OK) | frozenset(EXPOSED_BUILTINS)

_ENUM_BASES = frozenset({"enum.Enum", "enum.IntEnum", "enum.StrEnum"})
_LITERALS = frozenset({"typing.Literal", "typing_extensions.Literal"})
_FACTORY_NAMES = frozenset({"list", "dict", "set"})
_CALLABLE_KEYWORDS = frozenset(
    {
        "alias_generator",
        "field_title_generator",
        "model_title_generator",
        "json_schema_extra",
    }
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
        code: ``unsafe_source``, ``syntax_error``, ``no_models``, ``invalid_model``,
            ``timeout`` or ``sandbox_unavailable``.
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


def _reject(node: ast.AST | None, message: str) -> NoReturn:
    raise SourceError("unsafe_source", message, getattr(node, "lineno", None))


class _Guard:
    """Walks a module and raises ``SourceError`` at the first thing not allowed."""

    def __init__(self) -> None:
        # Names bound at module level -> what they are bound to. An import is
        # "module:<m>" or "import:<m>.<n>", a class "class", a constant "const".
        self.bound: dict[str, str] = {}
        self.modules: dict[str, str] = {}  # module alias -> module
        self.callables: set[str] = set()  # imported (callable) names
        self.origin: dict[str, str] = {}  # imported name -> "<m>.<n>"
        self.enums: dict[str, set[str]] = {}  # enum class -> member names
        self.in_class = False

    # -- bindings ----------------------------------------------------------- #

    def bind(
        self, name: str, origin: str, node: ast.AST, *, genuine: bool = False
    ) -> None:
        """Record a module-level binding of ``name``, or reject a rebinding.

        Args:
            name: The name being bound.
            origin: What it is bound to (see ``self.bound``).
            node: The statement, for the error line.
            genuine: True for an unaliased import of the allowlisted name itself,
                the only binding of a reserved name that is allowed.
        """
        if name.startswith("__"):
            _reject(node, f"the name '{name}' is not allowed")
        if origin.startswith(("module:", "import:")) and self.bound.get(name) == origin:
            return  # the same import again
        if name in self.bound:
            _reject(node, f"'{name}' is already defined and cannot be rebound")
        if name in _RESERVED and not genuine:
            _reject(
                node,
                f"'{name}' is a name of the allowlist or a builtin and cannot be "
                "rebound",
            )
        self.bound[name] = origin

    def bind_in_class(self, name: str, node: ast.AST) -> None:
        """Check that a class body may bind ``name`` (an attribute or field)."""
        if name.startswith("__"):
            _reject(node, f"the name '{name}' is not allowed")
        if name in self.bound or name in _RESERVED_IN_CLASS:
            _reject(
                node,
                f"'{name}' is an imported name, an allowlisted name or a builtin "
                "and cannot be rebound",
            )

    # -- statements --------------------------------------------------------- #

    def module(self, tree: ast.Module) -> None:
        for stmt in tree.body:
            self.stmt(stmt)

    def stmt(self, node: ast.stmt) -> None:
        if type(node) in _FORBIDDEN_NODES:
            _reject(node, f"'{_FORBIDDEN_NODES[type(node)]}' is not allowed")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if self.in_class:
                _reject(node, "imports are only allowed at the top of the source")
            if isinstance(node, ast.Import):
                self.import_(node)
            else:
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
            self.ann_assign(node)
        elif isinstance(node, ast.Assign):
            self.assign(node)
        else:
            _reject(node, f"'{type(node).__name__}' statements are not allowed")

    def import_(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES:
                _reject(node, f"import of '{alias.name}' is not allowed")
            name = alias.asname or alias.name
            self.bind(
                name,
                f"module:{alias.name}",
                node,
                genuine=name == alias.name,
            )
            self.modules[name] = alias.name

    def import_from(self, node: ast.ImportFrom) -> None:
        if node.level or node.module not in MODULE_MEMBERS:
            _reject(node, f"import from '{node.module}' is not allowed")
        members = MODULE_MEMBERS[node.module]
        for alias in node.names:
            if alias.name not in members:
                _reject(
                    node,
                    f"'{alias.name}' cannot be imported from '{node.module}' here",
                )
            name = alias.asname or alias.name
            self.bind(
                name,
                f"import:{node.module}.{alias.name}",
                node,
                genuine=name == alias.name,
            )
            self.callables.add(name)
            self.origin[name] = f"{node.module}.{alias.name}"

    def class_def(self, node: ast.ClassDef) -> None:
        if node.decorator_list:
            _reject(node, "decorators are not allowed")
        if node.keywords:
            _reject(node, "class keywords (such as metaclass=) are not allowed")
        is_enum = False
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.name(base.id, base)
                is_enum = is_enum or base.id in self.enums
                is_enum = is_enum or self.origin.get(base.id) in _ENUM_BASES
            elif isinstance(base, ast.Attribute):
                qualified = self.attribute(base)
                is_enum = is_enum or qualified in _ENUM_BASES
            else:
                _reject(base, "base classes must be plain or qualified names")
        nested = self.in_class
        if nested:
            self.bind_in_class(node.name, node)
        else:
            self.bind(node.name, "class", node)
        self.in_class = True
        try:
            for stmt in node.body:
                self.stmt(stmt)
        finally:
            self.in_class = nested
        if is_enum and not nested:
            self.enums[node.name] = self.members_of(node)

    @staticmethod
    def members_of(node: ast.ClassDef) -> set[str]:
        """The names a class body assigns (the members of an enum)."""
        members: set[str] = set()
        for stmt in node.body:
            targets: list[ast.expr] = []
            if isinstance(stmt, ast.Assign):
                targets = list(stmt.targets)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets = [stmt.target]
            members.update(
                t.id
                for t in targets
                if isinstance(t, ast.Name) and not t.id.startswith("_")
            )
        return members

    def assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                _reject(node, "only plain names may be assigned")
            self.target(target.id, node)
        self.value(node.value)

    def ann_assign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            _reject(node, "only plain names may be annotated")
        self.expr(node.annotation, str_ok=False)
        if node.value is None:
            self.name(node.target.id, node)  # an annotation alone binds nothing
            return
        self.target(node.target.id, node)
        self.value(node.value)

    def target(self, name: str, node: ast.AST) -> None:
        """Bind an assignment target in the current scope."""
        if self.in_class:
            self.bind_in_class(name, node)
        else:
            self.bind(name, "const", node)

    def value(self, node: ast.expr) -> None:
        """Check the right-hand side of an assignment.

        At module level it is a constant or a simple literal; in a class body any
        allowed expression (``Field(...)``, an enum member, a type).
        """
        if self.in_class:
            self.expr(node, str_ok=True)
        elif not _is_literal(node):
            _reject(
                node,
                "a top-level assignment may only bind a constant or a simple literal",
            )

    # -- expressions -------------------------------------------------------- #

    @staticmethod
    def name(name: str | None, node: ast.AST) -> None:
        if name is not None and name.startswith("__"):
            _reject(node, f"the name '{name}' is not allowed")

    def attribute(self, node: ast.Attribute) -> str:
        """Check ``module.member`` or ``EnumClass.MEMBER``; return the full name."""
        base = node.value
        if not isinstance(base, ast.Name):
            _reject(
                node,
                "attribute access is only allowed on an imported module or an enum "
                "class, one level deep",
            )
        if base.id in self.modules:
            module = self.modules[base.id]
            if node.attr not in MODULE_MEMBERS[module]:
                _reject(node, f"'{module}.{node.attr}' is not allowed")
            return f"{module}.{node.attr}"
        if base.id in self.enums:
            if node.attr not in self.enums[base.id]:
                _reject(node, f"'{base.id}' has no member '{node.attr}'")
            return f"{base.id}.{node.attr}"
        _reject(
            node,
            f"attribute access on '{base.id}' is not allowed (only on imported "
            "modules and enum classes)",
        )

    def is_literal(self, node: ast.expr) -> bool:
        """Whether ``node`` names ``typing.Literal``."""
        if isinstance(node, ast.Name):
            return self.origin.get(node.id) in _LITERALS
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and self.modules.get(node.value.id) in {"typing", "typing_extensions"}
            and node.attr == "Literal"
        )

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
            self.attribute(node)
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript)):
                _reject(node, "only names can be subscripted")
            self.expr(node.value, str_ok=False)
            if isinstance(node.slice, ast.Slice):
                _reject(node, "slices are not allowed")
            self.expr(node.slice, str_ok=self.is_literal(node.value))
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
        if isinstance(func, ast.Attribute):
            self.attribute(func)  # rejects anything but module.member
        elif not (isinstance(func, ast.Name) and func.id in self.callables):
            _reject(
                node,
                "calls are only allowed for names imported from pydantic, typing, "
                "datetime, enum, decimal, uuid and annotated_types (or module.name "
                "of those)",
            )
        for arg in node.args:
            self.expr(arg, str_ok=True)
        for keyword in node.keywords:
            if keyword.arg is None:
                _reject(node, "'**' unpacking is not allowed")
            self.name(keyword.arg, node)
            if keyword.arg == "default_factory":
                value = keyword.value
                if not (isinstance(value, ast.Name) and value.id in _FACTORY_NAMES):
                    _reject(
                        node,
                        "default_factory only accepts list, dict or set",
                    )
                continue
            if keyword.arg in _CALLABLE_KEYWORDS and not _is_literal(keyword.value):
                _reject(node, f"'{keyword.arg}' only accepts constants")
            self.expr(keyword.value, str_ok=True)


def _is_literal(node: ast.expr) -> bool:
    """Whether ``node`` is a constant or a tuple, list or dict of them."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(
            node.operand, ast.Constant
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is not None and _is_literal(k) and _is_literal(v)
            for k, v in zip(node.keys, node.values, strict=True)
        )
    return False


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
