"""Turn the schema forms users may pass into a pydantic model.

Supported inputs are a ``BaseModel`` subclass (returned unchanged), a
``TypedDict`` class, a plain annotated function, and a JSON Schema mapping
(which needs the optional ``dydantic`` extra).
"""

import copy
import inspect
import json
import re
import threading
import types
import typing
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from typing import Annotated, Any, Union

import typing_extensions
from pydantic import BaseModel, Field, create_model

__all__ = ["SchemaLike", "coerce_schema", "unwrap_required"]

SchemaLike = Union[  # noqa: UP007 - runtime alias mixing typing constructs
    type[BaseModel], type[Any], Callable[..., Any], Mapping[str, Any]
]
"""Anything ``coerce_schema`` accepts."""

_JSONSCHEMA_HINT = (
    "JSON Schema input needs the optional 'dydantic' package; install it with "
    "`pip install trustcall-system-one[jsonschema]`"
)
_CACHE_SIZE = 256
_cache: OrderedDict[Hashable, type[BaseModel]] = OrderedDict()
_cache_lock = threading.Lock()

_WRAPPERS = tuple(
    getattr(typing_extensions, name)
    for name in ("Required", "NotRequired", "ReadOnly")
    if hasattr(typing_extensions, name)
)


def unwrap_required(annotation: Any) -> Any:
    """Strip ``Required``/``NotRequired``/``ReadOnly`` wrappers from a TypedDict value.

    The wrappers may sit outside or inside an ``Annotated`` layer.

    Args:
        annotation: A value annotation as returned by ``get_type_hints`` with
            ``include_extras=True``.

    Returns:
        The annotation without the qualifiers.
    """
    origin = typing_extensions.get_origin(annotation)
    if origin in _WRAPPERS:
        return unwrap_required(typing_extensions.get_args(annotation)[0])
    if origin is Annotated:
        inner, *metadata = typing_extensions.get_args(annotation)
        stripped = unwrap_required(inner)
        if stripped is not inner:
            rebuilt: Any = stripped
            for item in metadata:
                rebuilt = Annotated[rebuilt, item]
            return rebuilt
    return annotation


def _parse_arg_docs(doc: str) -> dict[str, str]:
    """Read per-parameter descriptions from a Google-style ``Args:`` section."""
    descriptions: dict[str, list[str]] = {}
    in_section = False
    section_indent = 0
    entry_indent: int | None = None
    current: str | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not in_section:
            if stripped in ("Args:", "Arguments:", "Parameters:", "Params:"):
                in_section = True
                section_indent = indent
            continue
        if not stripped:
            continue
        if indent <= section_indent:
            break
        if entry_indent is None:
            entry_indent = indent
        if indent <= entry_indent:
            match = re.match(r"\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if match is None:
                current = None
                continue
            current = match.group(1)
            descriptions[current] = [match.group(2)]
        elif current is not None:
            descriptions[current].append(stripped)
    return {
        name: " ".join(part for part in parts if part).strip()
        for name, parts in descriptions.items()
    }


_UNION_ORIGINS: frozenset[Any] = frozenset({Union, types.UnionType})


def _convert_annotation(annotation: Any, active: tuple[type[Any], ...] = ()) -> Any:
    """Replace TypedDict classes anywhere inside an annotation with models.

    Handles ``Optional``/unions, ``Annotated`` and subscripted containers such as
    ``list[...]`` and ``dict[...]``.

    Raises:
        TypeError: If a TypedDict refers to itself.
    """
    if isinstance(annotation, (str, typing.ForwardRef)):
        # Python 3.10 leaves forward references nested in generics unresolved;
        # a reference to a TypedDict that is being converted is a cycle.
        name = annotation if isinstance(annotation, str) else annotation.__forward_arg__
        if any(name == td.__name__ for td in active):
            raise TypeError(
                f"recursive TypedDict {name!r} cannot be used as a schema; use a "
                "pydantic model"
            )
        return annotation
    if isinstance(annotation, type) and typing_extensions.is_typeddict(annotation):
        if annotation in active:
            raise TypeError(
                f"recursive TypedDict {annotation.__name__!r} cannot be used as a "
                "schema; use a pydantic model"
            )
        return _typed_dict_model(annotation, active)
    origin = typing_extensions.get_origin(annotation)
    if origin is None or origin is typing_extensions.Literal:
        return annotation
    args = typing_extensions.get_args(annotation)
    if origin is Annotated:
        inner = _convert_annotation(args[0], active)
        if inner is args[0]:
            return annotation
        rebuilt: Any = inner
        for item in args[1:]:
            rebuilt = Annotated[rebuilt, item]
        return rebuilt
    if any(isinstance(arg, (list, dict)) for arg in args):
        return annotation
    converted = tuple(_convert_annotation(arg, active) for arg in args)
    if all(new is old for new, old in zip(converted, args, strict=True)):
        return annotation
    if origin in _UNION_ORIGINS:
        return Union[converted]  # noqa: UP007
    try:
        return origin[converted]
    except TypeError:
        return annotation


def _from_typed_dict(schema: type[Any]) -> type[BaseModel]:
    return _typed_dict_model(schema, ())


def _typed_dict_model(
    schema: type[Any], active: tuple[type[Any], ...]
) -> type[BaseModel]:
    hints = typing_extensions.get_type_hints(schema, include_extras=True)
    required = getattr(schema, "__required_keys__", frozenset(hints))
    fields: dict[str, Any] = {}
    for key, annotation in hints.items():
        stripped = _convert_annotation(unwrap_required(annotation), (*active, schema))
        # A key that may be absent becomes a field with a ``None`` default so the
        # generated model accepts the same documents as the TypedDict.
        fields[key] = (stripped, ...) if key in required else (stripped, None)
    model = create_model(schema.__name__, **fields)
    model.__doc__ = schema.__doc__
    return model


def _from_callable(function: Callable[..., Any]) -> type[BaseModel]:
    name = getattr(function, "__name__", None) or type(function).__name__
    signature = inspect.signature(function)
    try:
        hints = typing_extensions.get_type_hints(function, include_extras=True)
    except Exception:  # unresolvable forward refs, partial objects, ...
        hints = {}
    doc = inspect.getdoc(function) or ""
    descriptions = _parse_arg_docs(doc)
    fields: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{name}: *args and **kwargs cannot be turned into a schema "
                f"(parameter {parameter.name!r})"
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(
                f"{name}: parameter {parameter.name!r} needs a type annotation"
            )
        annotation = _convert_annotation(annotation)
        description = descriptions.get(parameter.name)
        if description:
            annotation = Annotated[annotation, Field(description=description)]
        default = (
            ... if parameter.default is inspect.Parameter.empty else parameter.default
        )
        fields[parameter.name] = (annotation, default)
    model = create_model(name, **fields)
    model.__doc__ = doc or None
    return model


def _from_json_schema(schema: Mapping[str, Any]) -> type[BaseModel]:
    try:
        from dydantic import create_model_from_schema
    except ImportError as exc:
        raise ImportError(_JSONSCHEMA_HINT) from exc
    if schema.get("type", "object") != "object":
        raise ValueError("a JSON Schema schema must describe an object")
    body = copy.deepcopy(dict(schema))
    model: type[BaseModel] = create_model_from_schema(body)
    description = schema.get("description")
    if model.__doc__ is None and isinstance(description, str):
        model.__doc__ = description
    return model


def _build(schema: Any) -> type[BaseModel]:
    if isinstance(schema, type):
        if typing_extensions.is_typeddict(schema):
            return _from_typed_dict(schema)
        raise TypeError(
            f"cannot use class {schema.__name__!r} as a schema: pass a pydantic "
            "BaseModel subclass or a TypedDict"
        )
    if isinstance(schema, Mapping):
        return _from_json_schema(schema)
    if callable(schema):
        return _from_callable(schema)
    raise TypeError(
        "schema must be a BaseModel subclass, TypedDict, annotated function or "
        f"JSON Schema mapping, got {type(schema).__name__}"
    )


def _cache_key(schema: Any) -> Hashable | None:
    if isinstance(schema, Mapping):
        try:
            return ("json", json.dumps(schema, sort_keys=True))
        except (TypeError, ValueError):
            return None
    try:
        hash(schema)
    except TypeError:
        return None
    return ("object", schema)


def coerce_schema(schema: SchemaLike) -> type[BaseModel]:
    """Return a pydantic model for any supported schema form.

    * A ``BaseModel`` subclass is returned unchanged.
    * A ``TypedDict`` becomes a model with the same name, fields, required keys
      and docstring. Keys that may be absent get a ``None`` default.
    * A plain function becomes a model named after it with one field per
      parameter. Every parameter needs an annotation; defaults are preserved;
      the docstring becomes the model docstring and a Google-style ``Args:``
      section supplies field descriptions.
    * A JSON Schema mapping is built with ``dydantic`` (the ``jsonschema``
      extra); its ``title`` names the model.

    Equal inputs (the same class or function, or JSON Schemas that serialise
    identically) return the same model object.

    Args:
        schema: The schema in any of the forms above.

    Returns:
        A pydantic model class.

    Raises:
        TypeError: If the input is none of the supported forms, or a function
            parameter is unannotated or variadic.
        ImportError: If a JSON Schema is passed and ``dydantic`` is missing.
        ValueError: If a JSON Schema does not describe an object.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    key = _cache_key(schema)
    if key is None:
        return _build(schema)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
    model = _build(schema)
    with _cache_lock:
        existing = _cache.setdefault(key, model)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return existing
