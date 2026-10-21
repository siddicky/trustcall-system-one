"""Schema compiler: turn a pydantic model into a plan of Clairic and Clef asks.

``compile_schema`` is pure and memoised. For every top-level field it decides
which model answers it (a Clairic field, a Clef question, nothing, or one of the
kinds handled by later engines), what the question says, how the answer maps
back to a Python value, and which wire names are used.
"""

import collections.abc
import datetime as dt
import enum
import functools
import inspect
import ipaddress
import pathlib
import re
import types
import typing
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any, Literal, Union

import annotated_types
import pydantic_core
import typing_extensions
from pydantic import AnyUrl, BaseModel
from pydantic.fields import FieldInfo

from trustcall_system_one._config import GateConfig, Limits
from trustcall_system_one._names import (
    FieldNameMap,
    FieldNameMapBuilder,
    Path,
    pretty,
)
from trustcall_system_one._schemas import SchemaLike, coerce_schema
from trustcall_system_one._wire import (
    ChoiceQuestion,
    ClairicField,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
)

__all__ = [
    "S1_KEY",
    "Engine",
    "FieldSpec",
    "FrozenMap",
    "Question",
    "SchemaPlan",
    "compile_schema",
    "s1_field",
]

S1_KEY = "trustcall_system_one"
"""Key under which ``s1_field`` options live in ``json_schema_extra``."""

Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]  # noqa: UP007
"""Any Clef question model."""

Engine = Literal[
    "clairic",
    "clef",
    "skip",
    "submodel",
    "collection",
    "mapping",
    "union_models",
]
"""Which part of the system answers a field."""

_DEFERRED_ENGINES = frozenset({"submodel", "collection", "mapping", "union_models"})
_CLAIRIC_TYPES = (
    "text",
    "number",
    "date",
    "time",
    "datetime",
    "duration",
    "list",
    "estimate",
    "forecast",
)
_DURATION_UNITS = ("seconds", "minutes", "hours", "days", "weeks", "months", "years")
_CLEF_KINDS = ("noul", "choice", "score")
_S1_KEYS = frozenset(
    {
        "instructions",
        "null_threshold",
        "min_confidence",
        "clairic_type",
        "unit",
        "decimals",
        "integer",
        "identity",
        "skip",
        "clef",
        "criteria",
    }
)
_IDENTITY_NAMES = (
    "name",
    "full_name",
    "title",
    "label",
    "id",
    "key",
    "slug",
    "email",
    "handle",
    "sku",
)
_MAX_CHOICE_OPTIONS = 255
_MIN_SCORE_LEVELS, _MAX_SCORE_LEVELS = 2, 10
_AGGREGATION_RE = re.compile(
    r"\b(?:total|sum|average|mean|count) of\b|\baggregate\b", re.IGNORECASE
)
_OPTION_BAD_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CONSTRAINT_KEYS = (
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "min_length",
    "max_length",
    "pattern",
    "decimal_places",
    "max_digits",
)
_NONE_TYPE = type(None)
_UNION_ORIGINS: frozenset[Any] = frozenset({Union, types.UnionType})
_LITERAL_ORIGINS: frozenset[Any] = frozenset(
    {typing.Literal, typing_extensions.Literal}
)
_LIST_ORIGINS: frozenset[Any] = frozenset(
    {
        list,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
    }
)
_SET_ORIGINS: frozenset[Any] = frozenset({set, collections.abc.Set})
_MAPPING_ORIGINS: frozenset[Any] = frozenset(
    {dict, collections.abc.Mapping, collections.abc.MutableMapping}
)
_TEXT_LIKE: tuple[type[Any], ...] = (
    str,
    uuid.UUID,
    pathlib.PurePath,
    ipaddress.IPv4Address,
    ipaddress.IPv6Address,
    ipaddress.IPv4Network,
    ipaddress.IPv6Network,
    pydantic_core.Url,
    AnyUrl,
)
_UNION_TYPE_NAMES: dict[Any, str] = {
    str: "str",
    bool: "bool",
    int: "int",
    float: "float",
    Decimal: "decimal",
    dt.datetime: "datetime",
    dt.date: "date",
    dt.time: "time",
    dt.timedelta: "duration",
}
_UNION_TYPE_HELP: dict[str, str] = {
    "str": "Plain text.",
    "bool": "True or false.",
    "int": "A whole number.",
    "float": "A number that may have a fractional part.",
    "decimal": "An exact decimal number.",
    "datetime": "A date together with a time of day.",
    "date": "A calendar date.",
    "time": "A time of day.",
    "duration": "A length of time.",
}


# --------------------------------------------------------------------------- #
# s1_field
# --------------------------------------------------------------------------- #


def s1_field(
    *,
    instructions: str | None = None,
    null_threshold: float | None = None,
    min_confidence: float | None = None,
    clairic_type: str | None = None,
    unit: str | None = None,
    decimals: int | None = None,
    integer: bool | None = None,
    identity: bool = False,
    skip: bool = False,
    clef: str | None = None,
    criteria: Mapping[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    """Build ``json_schema_extra`` carrying per-field extraction options.

    Use it as ``Field(json_schema_extra=s1_field(...))``. Only options that were
    set appear in the result.

    Args:
        instructions: Replaces the synthesised field description.
        null_threshold: Clairic ``null_threshold`` in ``(0, 1]``.
        min_confidence: Minimum answer confidence in ``(0, 1]``.
        clairic_type: Force the Clairic field type.
        unit: Clairic unit label; for durations one of seconds, minutes, hours,
            days, weeks, months or years.
        decimals: Digits after the decimal point, 0 to 6.
        integer: Force or forbid fractional numbers.
        identity: Mark the field as part of the record's identity.
        skip: Never ask for this field; pydantic supplies the default.
        clef: Route the field through Clef as ``"noul"``, ``"choice"`` or
            ``"score"`` (``"score"`` is for ordered enums and literals).
        criteria: Clef criteria: ``{"true": ..., "false": ...}`` for noul, an
            option-to-description mapping for choice, or an ordered list of
            level descriptions for score.

    Returns:
        ``{"trustcall_system_one": {...}}``.

    Raises:
        ValueError: If an option is out of range or unknown.
    """
    options: dict[str, Any] = {}
    if instructions is not None:
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a non-empty string")
        options["instructions"] = instructions
    if null_threshold is not None:
        _check_unit_interval("null_threshold", null_threshold)
        options["null_threshold"] = null_threshold
    if min_confidence is not None:
        _check_unit_interval("min_confidence", min_confidence)
        options["min_confidence"] = min_confidence
    if clairic_type is not None:
        if clairic_type not in _CLAIRIC_TYPES:
            raise ValueError(
                f"clairic_type must be one of {list(_CLAIRIC_TYPES)}, "
                f"got {clairic_type!r}"
            )
        options["clairic_type"] = clairic_type
    if unit is not None:
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("unit must be a non-empty string")
        options["unit"] = unit
    if decimals is not None:
        if isinstance(decimals, bool) or not isinstance(decimals, int):
            raise ValueError("decimals must be an int")
        if not 0 <= decimals <= 6:
            raise ValueError("decimals must be between 0 and 6")
        options["decimals"] = decimals
    if integer is not None:
        if not isinstance(integer, bool):
            raise ValueError("integer must be a bool")
        options["integer"] = integer
    if identity:
        options["identity"] = True
    if skip:
        options["skip"] = True
    if clef is not None:
        if clef not in _CLEF_KINDS:
            raise ValueError(f"clef must be one of {list(_CLEF_KINDS)}, got {clef!r}")
        options["clef"] = clef
    if criteria is not None:
        if isinstance(criteria, (str, bytes)) or not isinstance(
            criteria, (Mapping, list, tuple)
        ):
            raise ValueError("criteria must be a mapping or a list")
        options["criteria"] = criteria
    return {S1_KEY: options}


def _check_unit_interval(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


class FrozenMap(Mapping[str, Any]):
    """An immutable, hashable string-keyed mapping (values are frozen deeply)."""

    __slots__ = ("_data", "_hash")

    def __init__(
        self, items: Mapping[str, Any] | Iterable[tuple[str, Any]] = ()
    ) -> None:
        """Copy ``items``; nested mappings and sequences are frozen too."""
        source = items.items() if isinstance(items, Mapping) else items
        self._data: dict[str, Any] = {key: _freeze(val) for key, val in source}
        try:
            self._hash = hash(frozenset(self._data.items()))
        except TypeError:  # a value that cannot be hashed
            self._hash = hash(frozenset(self._data))

    def __getitem__(self, key: str) -> Any:
        """Return the value for ``key``."""
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate keys in insertion order."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._data)

    def __hash__(self) -> int:
        """Return a hash of the contents."""
        return self._hash

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"FrozenMap({self._data!r})"


@dataclass(frozen=True)
class FieldSpec:
    """How one field is asked for and how its answer maps back.

    Attributes:
        path: Position of the field in the document.
        pretty: ``path`` joined with dots.
        wire_name: Name of the Clairic field (``engine == "clairic"``) or Clef
            question (``engine == "clef"``). Empty when nothing is sent.
        annotation: The field's type annotation.
        required: False for ``Optional`` fields and fields with a default.
        has_default: Whether pydantic supplies a default when the key is absent.
        engine: Which part of the system answers the field. ``skip`` fields are
            never asked; ``submodel``, ``collection``, ``mapping`` and
            ``union_models`` are recognised but expanded by later engines.
        clairic: The Clairic field to send, if any.
        clef: The Clef question to send, if any.
        key_map: For Clef choice and score, option id to Python value. For a
            primitive union, option id to member type name.
        null_threshold: Resolved null threshold.
        min_confidence: Resolved minimum answer confidence.
        description: The description the instructions were built from.
        instructions: The (clipped) instruction string that was sent.
        item_kind: Element parser for list fields (``"str"``, ``"int"``,
            ``"float"``, ``"decimal"``, ``"date"``, ``"datetime"``, ``"time"``,
            ``"timedelta:<unit>"`` or ``"choice"``).
        element_choices: For ``list[Literal|Enum]``, option id to value.
        post: Tag naming the coercion applied to the answer, such as ``"int"``,
            ``"decimal"``, ``"timedelta:hours"``, ``"set"`` or ``"union"``.
        identity: Whether the field was marked ``s1_field(identity=True)``.
        extra: The ``s1_field`` options.
        warnings: Short machine-readable warning tags.
        clef_name: Name of the Clef question, when there is one. It equals
            ``wire_name`` for Clef-engine fields and ``<field>.__type`` for a
            union of primitives (whose value is asked from Clairic).
    """

    path: Path
    pretty: str
    wire_name: str
    annotation: Any = field(hash=False)
    required: bool
    has_default: bool
    engine: Engine
    clairic: ClairicField | None = field(default=None, hash=False)
    clef: Question | None = field(default=None, hash=False)
    key_map: Mapping[str, Any] | None = None
    null_threshold: float | None = None
    min_confidence: float | None = None
    description: str = ""
    instructions: str = ""
    item_kind: str | None = None
    element_choices: Mapping[str, Any] | None = None
    post: str = ""
    identity: bool = False
    extra: Mapping[str, Any] = field(default_factory=FrozenMap)
    warnings: tuple[str, ...] = ()
    clef_name: str = ""


@dataclass(frozen=True, eq=False)
class SchemaPlan:
    """Everything the engines need to know about one schema.

    Attributes:
        schema: The pydantic model.
        name: Model class name.
        title: Model title (``model_config["title"]`` or the class name).
        description: Model docstring, or an empty string.
        specs: One spec per top-level field, in declaration order.
        clairic_names: Wire names of the Clairic fields.
        clef_names: Wire names of the Clef questions.
        identity: Names of the fields that identify a record.
        warnings: Human-readable compile warnings.
    """

    schema: type[BaseModel]
    name: str
    title: str
    description: str
    specs: tuple[FieldSpec, ...]
    clairic_names: FieldNameMap
    clef_names: FieldNameMap
    identity: tuple[str, ...]
    warnings: tuple[str, ...]

    def spec(self, path: Path) -> FieldSpec:
        """Return the spec for a path.

        Raises:
            KeyError: If no field has that path.
        """
        for spec in self.specs:
            if spec.path == path:
                return spec
        raise KeyError(path)

    def _select(self, paths: Iterable[Path] | None) -> list[FieldSpec]:
        if paths is None:
            return list(self.specs)
        wanted = {tuple(path) for path in paths}
        known = {spec.path for spec in self.specs}
        missing = wanted - known
        if missing:
            raise KeyError(sorted(pretty(path) for path in missing))
        return [spec for spec in self.specs if spec.path in wanted]

    def clairic_fields(
        self, paths: Iterable[Path] | None = None
    ) -> dict[str, ClairicField]:
        """Return the Clairic fields to send, keyed by wire name.

        Args:
            paths: Restrict to these paths; ``None`` means every field.

        Raises:
            KeyError: If a path is not a field of the schema.
        """
        return {
            spec.wire_name: spec.clairic
            for spec in self._select(paths)
            if spec.clairic is not None
        }

    def clef_questions(
        self, paths: Iterable[Path] | None = None
    ) -> dict[str, Question]:
        """Return the Clef questions to send, keyed by wire name.

        Args:
            paths: Restrict to these paths; ``None`` means every field.

        Raises:
            KeyError: If a path is not a field of the schema.
        """
        return {
            spec.clef_name: spec.clef
            for spec in self._select(paths)
            if spec.clef is not None
        }

    def deferred(self) -> tuple[FieldSpec, ...]:
        """Return the specs that later engines expand (sub-models, collections)."""
        return tuple(spec for spec in self.specs if spec.engine in _DEFERRED_ENGINES)


# --------------------------------------------------------------------------- #
# Annotation analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Shape:
    """An annotation with ``Annotated`` layers and ``Optional`` removed."""

    base: Any
    nullable: bool = False
    metadata: tuple[Any, ...] = ()
    description: str | None = None
    members: tuple["_Shape", ...] | None = None


def _flatten_metadata(items: Iterable[Any]) -> Iterator[Any]:
    for item in items:
        if isinstance(item, FieldInfo):
            yield from _flatten_metadata(item.metadata)
            continue
        yield item
        if isinstance(item, annotated_types.GroupedMetadata):
            yield from _flatten_metadata(iter(item))


def _strip(annotation: Any) -> _Shape:
    """Remove ``Annotated``, ``NewType``, aliases and ``Optional`` layers."""
    metadata: list[Any] = []
    description: str | None = None
    nullable = False
    current = annotation
    for _ in range(64):
        if isinstance(current, typing_extensions.TypeAliasType):
            current = current.__value__
            continue
        if hasattr(current, "__supertype__"):
            current = current.__supertype__
            continue
        origin = typing_extensions.get_origin(current)
        if origin is Annotated:
            inner, *extras = typing_extensions.get_args(current)
            metadata.extend(_flatten_metadata(extras))
            for extra in extras:
                if isinstance(extra, FieldInfo) and description is None:
                    description = extra.description
            current = inner
            continue
        if origin in _UNION_ORIGINS:
            args = typing_extensions.get_args(current)
            others = [arg for arg in args if arg is not _NONE_TYPE]
            nullable = nullable or len(others) != len(args)
            if len(others) == 1:
                current = others[0]
                continue
            members = tuple(_strip(arg) for arg in others)
            nullable = nullable or any(member.nullable for member in members)
            return _Shape(
                current, nullable, tuple(metadata), description, members=members
            )
        if origin in _LITERAL_ORIGINS and None in typing_extensions.get_args(current):
            nullable = True
        break
    if current is None:
        current = _NONE_TYPE
    return _Shape(current, nullable, tuple(metadata), description)


def _constraints(metadata: Iterable[Any]) -> dict[str, Any]:
    """Collect ``ge``/``le``/``min_length``/... from constraint objects."""
    found: dict[str, Any] = {}
    for item in metadata:
        for key in _CONSTRAINT_KEYS:
            value = getattr(item, key, None)
            if isinstance(value, re.Pattern):
                value = value.pattern
            if value is not None and not callable(value):
                found[key] = value
    return found


def _is_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _is_enum(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, enum.Enum)


def _literal_values(tp: Any) -> tuple[list[Any], bool]:
    """Return a Literal's non-None values and whether ``None`` was among them."""
    values = list(typing_extensions.get_args(tp))
    return [v for v in values if v is not None], any(v is None for v in values)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def _clean(text: str) -> str:
    return " ".join(text.split())


def _humanise(name: str) -> str:
    spaced = _CAMEL_BOUNDARY.sub("_", name)
    return _clean(spaced.replace("_", " ").replace("-", " ")).lower()


def _end_sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rstrip()
    space = cut.rfind(" ")
    if space > (limit - 3) * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "..."


def _fmt(value: Any) -> str:
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def _numeric_hints(c: Mapping[str, Any], *, temporal: bool) -> list[str]:
    parts: list[str] = []
    words = (
        {"ge": "on or after", "gt": "after", "le": "on or before", "lt": "before"}
        if temporal
        else {
            "ge": "at least",
            "gt": "greater than",
            "le": "at most",
            "lt": "less than",
        }
    )
    for key in ("ge", "gt", "le", "lt"):
        if key in c:
            parts.append(f"{words[key]} {_fmt(c[key])}")
    hints = [f"Must be {' and '.join(parts)}."] if parts else []
    if "multiple_of" in c:
        hints.append(f"Must be a multiple of {_fmt(c['multiple_of'])}.")
    if "max_digits" in c:
        hints.append(f"At most {c['max_digits']} digits in total.")
    return hints


def _length_hints(c: Mapping[str, Any], unit: str) -> list[str]:
    low, high = c.get("min_length"), c.get("max_length")
    if low is not None and high is not None:
        return [f"Between {low} and {high} {unit}."]
    if low is not None:
        return [f"At least {low} {unit}."]
    if high is not None:
        return [f"At most {high} {unit}."]
    return []


def _context(title: str, path: Path) -> str:
    """Return the " → "-joined model title and ancestor field names."""
    ancestors = [_humanise(str(part)) for part in path[:-1] if isinstance(part, str)]
    return " → ".join([title, *ancestors])


def _member_description(member: enum.Enum) -> str | None:
    attrs = vars(member)
    for key in ("description", "__doc__"):
        text = attrs.get(key)
        if isinstance(text, str) and text.strip():
            return _clean(text)
    return None


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@dataclass
class _Option:
    """One choice (or score level): id, Python value and description."""

    id: str
    value: Any
    description: str | None


@dataclass
class _Kind:
    """The outcome of classifying a field, before names and text exist."""

    engine: Engine
    clairic: dict[str, Any] | None = None
    clef: Literal["noul", "choice", "score"] | None = None
    hints: list[str] = field(default_factory=list)
    options: list[_Option] = field(default_factory=list)
    key_map: dict[str, Any] | None = None
    item_kind: str | None = None
    element_choices: dict[str, Any] | None = None
    post: str = ""
    warnings: list[str] = field(default_factory=list)
    union_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Ctx:
    """Everything classification needs to know about one field."""

    path: Path
    pretty: str
    options: Mapping[str, Any]
    constraints: Mapping[str, Any]
    nullable: bool
    required: bool
    gates: GateConfig
    limits: Limits
    stack: tuple[type[BaseModel], ...]


def _option_id(text: str, used: set[str]) -> str:
    base = _OPTION_BAD_CHARS.sub("_", text)[:100] or "_"
    candidate, number = base, 1
    while candidate in used:
        number += 1
        suffix = f"_{number}"
        candidate = f"{base[: 100 - len(suffix)]}{suffix}"
    used.add(candidate)
    return candidate


def _options_for(values: list[Any], descriptions: list[str | None]) -> list[_Option]:
    used: set[str] = set()
    return [
        _Option(_option_id(str(value), used), value, description)
        for value, description in zip(values, descriptions, strict=True)
    ]


def _choice_kind(ctx: _Ctx, values: list[Any], descriptions: list[str | None]) -> _Kind:
    """Build the Clef choice or score kind for a set of option values."""
    options = _options_for(values, descriptions)
    wanted = ctx.options.get("clef")
    if wanted not in (None, "choice", "score"):
        raise ValueError(
            f"field {ctx.pretty!r}: s1_field(clef={wanted!r}) cannot be used on "
            "a Literal or Enum"
        )
    if wanted == "score":
        if not _MIN_SCORE_LEVELS <= len(options) <= _MAX_SCORE_LEVELS:
            raise ValueError(
                f"field {ctx.pretty!r}: a score needs {_MIN_SCORE_LEVELS} to "
                f"{_MAX_SCORE_LEVELS} levels, got {len(options)}"
            )
        return _Kind(
            "clef",
            clef="score",
            options=options,
            key_map={str(i): option.value for i, option in enumerate(options)},
        )
    if len(options) > _MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"field {ctx.pretty!r} has {len(options)} options; a Clef choice "
            f"supports at most {_MAX_CHOICE_OPTIONS}"
        )
    if len(options) < 2:
        return _Kind("skip", warnings=["single_option"])
    return _Kind(
        "clef",
        clef="choice",
        options=options,
        key_map={option.id: option.value for option in options},
    )


def _bool_kind(ctx: _Ctx) -> _Kind:
    wanted = ctx.options.get("clef")
    if wanted == "score":
        raise ValueError(f"field {ctx.pretty!r}: a bool cannot be scored")
    if wanted == "choice" or (ctx.nullable and wanted != "noul"):
        options = [
            _Option("yes", True, "The input says this is true."),
            _Option("no", False, "The input says this is false."),
            _Option("unstated", None, "The input does not say."),
        ]
        return _Kind(
            "clef",
            clef="choice",
            options=options,
            key_map={option.id: option.value for option in options},
        )
    return _Kind("clef", clef="noul")


def _number_kind(ctx: _Ctx, base: Any) -> _Kind:
    c = ctx.constraints
    hints = _numeric_hints(c, temporal=False)
    if base is int:
        clairic: dict[str, Any] = {"type": "number", "integer": True}
        post = "int"
    elif base is Decimal:
        decimals = ctx.options.get("decimals", c.get("decimal_places", 2))
        clairic = {"type": "number", "decimals": decimals}
        post = "decimal"
    else:
        clairic = {"type": "number"}
        decimals = ctx.options.get("decimals", c.get("decimal_places"))
        if decimals is not None:
            clairic["decimals"] = decimals
        post = "float"
    kind = _Kind("clairic", clairic=clairic, hints=hints, post=post)
    if "decimals" in clairic and not 0 <= clairic["decimals"] <= 6:
        clairic["decimals"] = min(6, max(0, clairic["decimals"]))
        kind.warnings.append("decimals_clamped")
    low = c.get("ge", c.get("gt"))
    high = c.get("le", c.get("lt"))
    if ctx.limits.estimate_for_bounded_numbers and low is not None and high is not None:
        bounds = {"min": _bound(low, base), "max": _bound(high, base)}
        clairic.clear()
        clairic.update(type="estimate", **bounds)
    return kind


def _bound(value: Any, base: Any) -> Any:
    return int(value) if base is int else float(value)


def _item_kind(item: _Shape, unit: str) -> tuple[str, str] | None:
    """Return ``(item_kind, clairic items)`` for a supported list element."""
    base = item.base
    table: dict[Any, tuple[str, str]] = {
        str: ("str", "text"),
        int: ("int", "number"),
        float: ("float", "number"),
        Decimal: ("decimal", "number"),
        dt.datetime: ("datetime", "text"),
        dt.date: ("date", "date"),
        dt.time: ("time", "text"),
        dt.timedelta: (f"timedelta:{unit}", "text"),
    }
    if base is bool:
        return None
    found = table.get(base)
    if found is not None:
        return found
    if isinstance(base, type) and issubclass(base, str) and not _is_enum(base):
        return ("str", "text")
    return None


def _list_kind(ctx: _Ctx, container: str, item: _Shape) -> _Kind:
    unit = ctx.options.get("unit", "seconds")
    c = ctx.constraints
    hints = _length_hints(c, "items")
    post = "" if container == "list" else container
    base = item.base
    if _is_model(base) or (
        item.members is not None and all(_is_model(m.base) for m in item.members)
    ):
        return _model_kind(
            ctx,
            "collection",
            [base] if _is_model(base) else [m.base for m in item.members or ()],
        )
    if base is typing.Any or base is object or base is dict:
        return _Kind("skip")
    if typing_extensions.get_origin(base) in _LITERAL_ORIGINS or _is_enum(base):
        values, descriptions = _choice_values(base)
        options = _options_for(values, descriptions)
        kind = _Kind(
            "clairic",
            clairic={"type": "list", "items": "text"},
            hints=hints,
            item_kind="choice",
            element_choices={option.id: option.value for option in options},
            post=post,
        )
        kind.hints.append(
            "Each item must be one of: " + ", ".join(o.id for o in options[:30]) + "."
        )
        return kind
    parsed = _item_kind(item, unit)
    if parsed is None:
        return _Kind("skip", warnings=["unsupported_list_item"])
    item_kind, items = parsed
    kind = _Kind(
        "clairic",
        clairic={"type": "list", "items": items},
        hints=hints,
        item_kind=item_kind,
        post=post,
    )
    if base in (dt.datetime, dt.time, dt.timedelta):
        kind.warnings.append("lossy_list")
        if container == "list":
            kind.post = f"list[{item_kind.split(':')[0]}]"
    if base is dt.timedelta and unit in ("months", "years"):
        kind.warnings.append("lossy_duration")
    return kind


def _choice_values(base: Any) -> tuple[list[Any], list[str | None]]:
    if _is_enum(base):
        members = list(base)
        return (
            [member.value for member in members],
            [_member_description(member) for member in members],
        )
    values, _ = _literal_values(base)
    return values, [None] * len(values)


def _model_kind(ctx: _Ctx, engine: Engine, models: list[Any]) -> _Kind:
    """Record a nested-model field, applying the depth and recursion guards."""
    if len(ctx.path) + 1 > ctx.limits.max_depth or any(
        model in ctx.stack for model in models
    ):
        return _Kind("skip", warnings=["max_depth"])
    return _Kind(engine)


def _union_kind(ctx: _Ctx, shape: _Shape) -> _Kind:
    members = shape.members or ()
    if all(_is_model(m.base) for m in members):
        return _model_kind(ctx, "union_models", [m.base for m in members])
    if all(typing_extensions.get_origin(m.base) in _LITERAL_ORIGINS for m in members):
        values: list[Any] = []
        for member in members:
            values.extend(_literal_values(member.base)[0])
        return _choice_kind(ctx, values, [None] * len(values))
    names: list[str] = []
    for member in members:
        name = _UNION_TYPE_NAMES.get(member.base)
        if name is None:
            return _Kind("skip", warnings=["unsupported_union"])
        if name not in names:
            names.append(name)
    if len(names) == 1:
        single = next(m for m in members if _UNION_TYPE_NAMES.get(m.base) == names[0])
        return _classify(ctx, single)
    return _Kind(
        "clairic",
        clairic={"type": "text"},
        clef="choice",
        options=[_Option(name, name, _UNION_TYPE_HELP[name]) for name in names],
        key_map={name: name for name in names},
        post="union",
        union_names=names,
    )


def _classify(ctx: _Ctx, shape: _Shape) -> _Kind:
    base = shape.base
    c = ctx.constraints
    if shape.members is not None:
        return _union_kind(ctx, shape)
    origin = typing_extensions.get_origin(base)
    if base is typing.Any or base is object:
        return _any_kind(ctx)
    if base in (dict, collections.abc.Mapping) or origin in _MAPPING_ORIGINS:
        args = typing_extensions.get_args(base)
        if not args or args[1] is typing.Any or args[1] is object:
            return _any_kind(ctx)
        if args[0] is not str:
            return _Kind("skip", warnings=["unsupported_mapping_key"])
        return _Kind("mapping")
    if origin in _LITERAL_ORIGINS:
        values, _ = _literal_values(base)
        return _choice_kind(ctx, values, [None] * len(values))
    if _is_enum(base):
        values, descriptions = _choice_values(base)
        return _choice_kind(ctx, values, descriptions)
    if _is_model(base):
        return _model_kind(ctx, "submodel", [base])
    if origin in _LIST_ORIGINS or base is list:
        args = typing_extensions.get_args(base)
        return _list_kind(ctx, "list", _strip(args[0]) if args else _Shape(typing.Any))
    if origin in _SET_ORIGINS or base is set:
        args = typing_extensions.get_args(base)
        return _list_kind(ctx, "set", _strip(args[0]) if args else _Shape(typing.Any))
    if origin is frozenset or base is frozenset:
        args = typing_extensions.get_args(base)
        return _list_kind(
            ctx, "frozenset", _strip(args[0]) if args else _Shape(typing.Any)
        )
    if base is bool:
        return _bool_kind(ctx)
    if base is dt.datetime:
        return _Kind(
            "clairic",
            clairic={"type": "datetime"},
            hints=_numeric_hints(c, temporal=True),
        )
    if base is dt.date:
        return _Kind(
            "clairic", clairic={"type": "date"}, hints=_numeric_hints(c, temporal=True)
        )
    if base is dt.time:
        return _Kind(
            "clairic", clairic={"type": "time"}, hints=_numeric_hints(c, temporal=True)
        )
    if base is dt.timedelta:
        unit = ctx.options.get("unit", "seconds")
        kind = _Kind(
            "clairic",
            clairic={"type": "duration", "unit": unit},
            hints=_numeric_hints(c, temporal=False),
            post=f"timedelta:{unit}",
        )
        if unit in ("months", "years"):
            kind.warnings.append("lossy_duration")
        return kind
    if base in (int, float, Decimal):
        return _number_kind(ctx, base)
    if isinstance(base, type) and issubclass(base, int):
        return _number_kind(ctx, int)
    if isinstance(base, type) and issubclass(base, float):
        return _number_kind(ctx, float)
    if isinstance(base, type) and issubclass(base, _TEXT_LIKE):
        hints = _length_hints(c, "characters")
        if "pattern" in c:
            hints.append(f"Must match the pattern {c['pattern']}.")
        return _Kind("clairic", clairic={"type": "text"}, hints=hints)
    return _Kind("skip", warnings=["unsupported_type"])


def _any_kind(ctx: _Ctx) -> _Kind:
    if ctx.limits.any_policy == "text":
        return _Kind("clairic", clairic={"type": "text"})
    return _Kind("skip")


# --------------------------------------------------------------------------- #
# Building specs
# --------------------------------------------------------------------------- #


def _s1_options(info: FieldInfo, pretty_name: str) -> dict[str, Any]:
    extra = info.json_schema_extra
    options = extra.get(S1_KEY) if isinstance(extra, dict) else None
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ValueError(f"field {pretty_name!r}: {S1_KEY} options must be a mapping")
    unknown = set(options) - _S1_KEYS
    if unknown:
        raise ValueError(
            f"field {pretty_name!r}: unknown s1_field options {sorted(unknown)}"
        )
    return dict(options)


def _description(info: FieldInfo, shape: _Shape, name: str) -> str:
    extra = info.json_schema_extra
    schema_description = extra.get("description") if isinstance(extra, dict) else None
    for candidate in (
        info.description,
        schema_description,
        info.title,
        shape.description,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return _clean(candidate)
    return _humanise(name)


def _synthesise(
    kind: _Kind,
    *,
    description: str,
    prefix: str,
    limit: int,
    role: Literal["clairic", "noul", "choice", "score", "union_type"],
) -> str:
    """Build the instruction string for one question."""
    if role == "clairic":
        parts = [description]
        if kind.hints:
            parts = [_end_sentence(description), *kind.hints]
        body = " ".join(parts)
    elif role == "noul":
        body = f"Does the input say this is true: {description}?"
    elif role == "choice":
        body = f"Pick the option that best fits: {description}"
    elif role == "score":
        body = f"Rate on the scale, lowest level first: {description}"
    else:
        body = f"What kind of value is this: {description}?"
    return _clip(f"{prefix}: {body}" if prefix else body, limit)


def _choice_criteria(kind: _Kind, override: Any) -> Any:
    if kind.clef == "score":
        levels = [option.description or str(option.value) for option in kind.options]
        if isinstance(override, (list, tuple)):
            if len(override) != len(levels):
                raise ValueError(
                    f"score criteria has {len(override)} levels, expected {len(levels)}"
                )
            return list(override)
        return levels
    mapping = override if isinstance(override, Mapping) else {}
    out: dict[str, Any] = {}
    for option in kind.options:
        text = mapping.get(option.id, mapping.get(str(option.value)))
        out[option.id] = text if text is not None else option.description
    return out


def _compile_field(
    name: str,
    info: FieldInfo,
    *,
    title: str,
    gates: GateConfig,
    limits: Limits,
    stack: tuple[type[BaseModel], ...],
    clairic_names: FieldNameMapBuilder,
    clef_names: FieldNameMapBuilder,
) -> tuple[FieldSpec, list[str]]:
    path: Path = (name,)
    pretty_name = pretty(path)
    options = _s1_options(info, pretty_name)
    shape = _strip(info.annotation)
    metadata = list(_flatten_metadata(info.metadata)) + list(shape.metadata)
    nullable = shape.nullable
    has_default = not info.is_required()
    required = info.is_required() and not nullable
    ctx = _Ctx(
        path=path,
        pretty=pretty_name,
        options=options,
        constraints=_constraints(metadata),
        nullable=nullable,
        required=required,
        gates=gates,
        limits=limits,
        stack=stack,
    )
    kind = _Kind("skip") if options.get("skip") else _classify(ctx, shape)
    wanted_clef = options.get("clef")
    if wanted_clef is not None and kind.clef is None and kind.engine != "skip":
        raise ValueError(
            f"field {pretty_name!r}: s1_field(clef={wanted_clef!r}) only applies "
            "to bool, Literal and Enum fields"
        )
    description = _description(info, shape, name)
    if "instructions" in options:
        description = _clean(options["instructions"])
    messages: list[str] = []
    warnings = list(kind.warnings)
    if _AGGREGATION_RE.search(description):
        warnings.append("aggregation")
        messages.append(
            f"{pretty_name}: description {description!r} looks like an aggregate; "
            "derive it with a pydantic computed_field instead of extracting it"
        )
    for tag in kind.warnings:
        messages.append(f"{pretty_name}: {tag}")

    null_threshold = options.get(
        "null_threshold",
        gates.required_null_threshold if required else gates.optional_null_threshold,
    )
    min_confidence = options.get("min_confidence")
    if min_confidence is None:
        min_confidence = (
            gates.min_confidence
            if kind.clairic is not None
            else gates.min_choice_confidence
        )

    wire = ""
    clef_name = ""
    clairic_field: ClairicField | None = None
    clef_question: Question | None = None
    instructions = description
    shortened_clairic = shortened_clef = False

    if kind.clairic is not None:
        wire = clairic_names.add(path)
        shortened_clairic = wire in clairic_names.shortened
    if kind.clef is not None:
        if kind.clairic is not None:
            clef_name = clef_names.add(path, suffix="__type")
        else:
            clef_name = clef_names.add(path)
            wire = clef_name
        shortened_clef = clef_name in clef_names.shortened

    if kind.clairic is not None:
        prefix = _context(title, path) if shortened_clairic else ""
        instructions = _synthesise(
            kind,
            description=description,
            prefix=prefix,
            limit=limits.max_instruction_chars,
            role="clairic",
        )
        fields = dict(kind.clairic)
        override = options.get("clairic_type")
        if override is not None:
            fields["type"] = override
            if override == "list":
                fields.setdefault("items", "text")
        if "unit" in options and fields["type"] != "duration":
            fields["unit"] = options["unit"]
        if "integer" in options:
            fields["integer"] = options["integer"]
        if "decimals" in options and "decimals" not in fields:
            fields["decimals"] = options["decimals"]
        fields["null_threshold"] = null_threshold
        try:
            clairic_field = ClairicField(instructions=instructions, **fields)
        except ValueError as exc:
            raise ValueError(f"field {pretty_name!r}: {exc}") from exc

    if kind.clef is not None:
        prefix = _context(title, path) if shortened_clef else ""
        role: Literal["noul", "choice", "score", "union_type"] = (
            "union_type" if kind.clairic is not None else kind.clef
        )
        question_text = _synthesise(
            kind,
            description=description,
            prefix=prefix,
            limit=limits.max_instruction_chars,
            role=role,
        )
        if kind.clairic is None:
            instructions = question_text
        override_criteria = options.get("criteria")
        if kind.clef == "noul":
            mapping = (
                override_criteria if isinstance(override_criteria, Mapping) else {}
            )
            true_text = mapping.get(
                "true", f'The input states or clearly implies: "{description}".'
            )
            false_text = mapping.get(
                "false",
                "The input states or clearly implies the opposite of: "
                f'"{description}".',
            )
            clef_question = NoulQuestion(
                type="noul",
                instructions=question_text,
                criteria=NoulCriteria(true=true_text, false=false_text),
            )
        elif kind.clef == "choice":
            clef_question = ChoiceQuestion(
                type="choice",
                instructions=question_text,
                criteria=_choice_criteria(kind, override_criteria),
            )
        else:
            clef_question = ScoreQuestion(
                type="score",
                instructions=question_text,
                criteria=_choice_criteria(kind, override_criteria),
            )

    spec = FieldSpec(
        path=path,
        pretty=pretty_name,
        wire_name=wire,
        annotation=info.annotation,
        required=required,
        has_default=has_default,
        engine=kind.engine,
        clairic=clairic_field,
        clef=clef_question,
        key_map=FrozenMap(kind.key_map) if kind.key_map is not None else None,
        null_threshold=null_threshold,
        min_confidence=min_confidence,
        description=description,
        instructions=instructions,
        item_kind=kind.item_kind,
        element_choices=(
            FrozenMap(kind.element_choices)
            if kind.element_choices is not None
            else None
        ),
        post=kind.post,
        identity=bool(options.get("identity", False)),
        extra=FrozenMap(options),
        warnings=tuple(warnings),
        clef_name=clef_name,
    )
    return spec, messages


def _identity(specs: tuple[FieldSpec, ...]) -> tuple[str, ...]:
    marked = tuple(str(spec.path[0]) for spec in specs if spec.identity)
    if marked:
        return marked

    def is_text(spec: FieldSpec) -> bool:
        return spec.engine == "clairic" and _strip(spec.annotation).base is str

    by_name = {str(spec.path[0]): spec for spec in specs}
    for candidate in _IDENTITY_NAMES:
        spec = by_name.get(candidate)
        if spec is not None and is_text(spec):
            return (candidate,)
    for spec in specs:
        if spec.required and is_text(spec):
            return (str(spec.path[0]),)
    return ()


def _compile_model(
    model: type[BaseModel], gates: GateConfig, limits: Limits
) -> SchemaPlan:
    title = model.model_config.get("title") or model.__name__
    doc = inspect.cleandoc(model.__doc__) if model.__doc__ else ""
    clairic_names = FieldNameMapBuilder("clairic")
    clef_names = FieldNameMapBuilder("clef")
    specs: list[FieldSpec] = []
    messages: list[str] = []
    for name, info in model.model_fields.items():
        spec, field_messages = _compile_field(
            name,
            info,
            title=title,
            gates=gates,
            limits=limits,
            stack=(model,),
            clairic_names=clairic_names,
            clef_names=clef_names,
        )
        specs.append(spec)
        messages.extend(field_messages)
    frozen_specs = tuple(specs)
    return SchemaPlan(
        schema=model,
        name=model.__name__,
        title=title,
        description=doc,
        specs=frozen_specs,
        clairic_names=clairic_names.build(),
        clef_names=clef_names.build(),
        identity=_identity(frozen_specs),
        warnings=tuple(messages),
    )


_compile_cached = functools.lru_cache(maxsize=256)(_compile_model)


def compile_schema(
    schema: SchemaLike,
    *,
    gates: GateConfig = GateConfig(),  # noqa: B008 - frozen dataclass
    limits: Limits = Limits(),  # noqa: B008 - frozen dataclass
) -> SchemaPlan:
    """Compile a schema into a plan of Clairic fields and Clef questions.

    The result is memoised on ``(model, gates, limits)``, so compiling the same
    schema twice returns the same object. Plain functions, TypedDicts and JSON
    Schemas are converted with ``coerce_schema`` first.

    Args:
        schema: A pydantic model, TypedDict, annotated function or JSON Schema.
        gates: Thresholds that set the null thresholds and confidence floors.
        limits: Structural limits such as ``max_depth`` and ``any_policy``.

    Returns:
        The compiled plan.

    Raises:
        ValueError: If a field cannot be expressed (for example a Literal with
            more than 255 options) or an ``s1_field`` option is invalid.
        TypeError: If ``schema`` is not a supported form.
    """
    return _compile_cached(coerce_schema(schema), gates, limits)
