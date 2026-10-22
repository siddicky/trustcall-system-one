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
import sys
import types
import typing
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any, Literal, Union

import annotated_types
import pydantic
import pydantic.networks
import pydantic_core
import typing_extensions
from pydantic import AliasChoices, AliasPath, BaseModel
from pydantic.fields import FieldInfo

from trustcall_system_one._config import GateConfig, Limits
from trustcall_system_one._names import (
    FieldNameMap,
    FieldNameMapBuilder,
    Path,
    pretty,
)
from trustcall_system_one._schemas import SchemaLike, coerce_schema
from trustcall_system_one._types import UNSTATED
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
    "FieldGroup",
    "FieldSpec",
    "FrozenMap",
    "Question",
    "Role",
    "SchemaPlan",
    "UnionGroup",
    "UnionMember",
    "compile_schema",
    "render_instructions",
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
    "const",
]
"""Which part of the system answers a field.

``const`` is a single-option ``Literal`` or ``Enum`` on a required field: its
value is filled in by code and nothing is asked. ``submodel`` and
``union_models`` are container fields: the compiler expands them into field
groups (their ``clef`` is the presence question of an optional sub-model, or the
member choice of a union). ``collection`` and ``mapping`` are recorded for later
engines and not extracted yet.
"""

_DEFERRED_ENGINES = frozenset({"collection", "mapping"})
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
_SHORT_DOC = 40
_MIN_CHOICE_OPTIONS = 2
_UNSTATED_TEXT = "The input does not state this."
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


def _pydantic_types(*names: str) -> tuple[type[Any], ...]:
    """Return the named pydantic classes that exist in the installed version."""
    found: list[type[Any]] = []
    for name in names:
        candidate = getattr(pydantic, name, None) or getattr(
            pydantic.networks, name, None
        )
        if isinstance(candidate, type):
            found.append(candidate)
    return tuple(found)


_TEXT_LIKE: tuple[type[Any], ...] = (
    str,
    uuid.UUID,
    pathlib.PurePath,
    ipaddress.IPv4Address,
    ipaddress.IPv6Address,
    ipaddress.IPv4Network,
    ipaddress.IPv6Network,
    pydantic_core.Url,
    pydantic_core.MultiHostUrl,
    *_pydantic_types(
        "AnyUrl",
        "_BaseUrl",
        "_BaseMultiHostUrl",
        "EmailStr",
        "NameEmail",
        "IPvAnyAddress",
        "IPvAnyInterface",
        "IPvAnyNetwork",
    ),
)
_TEMPORAL_ALIASES: dict[Any, Any] = {
    **dict.fromkeys(_pydantic_types("PastDate", "FutureDate"), dt.date),
    **dict.fromkeys(
        _pydantic_types(
            "AwareDatetime", "NaiveDatetime", "PastDatetime", "FutureDatetime"
        ),
        dt.datetime,
    ),
}
_DOC_TYPE: Any = getattr(typing_extensions, "Doc", None)
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
            never asked. ``submodel`` and ``union_models`` are containers that
            the compiler expands into ``FieldGroup`` and ``UnionGroup`` objects;
            ``collection`` and ``mapping`` are recognised but expanded by later
            engines.
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
        nullable: Whether ``None`` is a valid value (``Optional`` fields).
        alias: The key ``model_validate`` expects for this field when it is not
            the field name (from ``alias`` or a plain-string
            ``validation_alias``); ``None`` when the name itself is the key.
        context: Parent context put in front of the Clairic instructions of a
            nested field or one whose name was shortened (empty otherwise).
        clef_context: The same for the Clef question.
        hints: Type and constraint sentences that follow the description in the
            Clairic instructions.
        models: For ``submodel``, ``collection`` and ``union_models`` fields, the
            resolved model classes (a single one except for a union of models).
            Forward references are resolved once, here, so expanders never
            resolve names again. Empty for every other engine.
        wire_path: The path the wire names were allocated for. It is ``path``
            except below a union of models, where it carries the member's label
            after the union field (members may share field names, while every
            wire name is unique in the plan). Empty means ``path``.
        scope: For a nested field, its ancestors' names (``"author → address"``,
            ``"pet (Dog)"`` below a union member); empty at the top level.
        member_labels: Labels of the union members this field is below, outermost
            first (empty outside unions of models).
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
    nullable: bool = False
    alias: str | None = None
    models: tuple[type[BaseModel], ...] = field(default=(), hash=False)
    context: str = ""
    clef_context: str = ""
    hints: tuple[str, ...] = ()
    wire_path: Path = ()
    scope: str = ""
    member_labels: tuple[str, ...] = ()

    @property
    def owners(self) -> tuple[str, ...]:
        """Humanised names of the fields this one is nested in, outermost first."""
        return tuple(
            _humanise(part) for part in self.path[:-1] if isinstance(part, str)
        )

    @property
    def scoped(self) -> str:
        """The description with the field's ancestors, for questions that stand alone.

        ``"city (author → address)"`` for a nested field, the plain description
        at the top level. Verification questions use it, since they do not carry
        the parent-context prefix that the extraction instructions do.
        """
        return f"{self.description} ({self.scope})" if self.scope else self.description


@dataclass(frozen=True, eq=False)
class FieldGroup:
    """The fields of one model in the plan: the root or a nested sub-model.

    One group is one Clairic pass under ``passes.split="submodel"``.

    Attributes:
        path: Where the group sits in the document (``()`` for the root; below a
            union of models, the union field's path).
        model: The pydantic model the group's fields belong to.
        optional: Whether the sub-model may be absent (its field is nullable or
            has a default). An optional group is gated by a presence question;
            a required group, the root and a union member are always asked.
        nullable: Whether ``None`` is a valid value of the sub-model field.
        has_default: Whether the sub-model field has a default.
        description: What the group is (the sub-model field's description; the
            model's docstring for the root).
        alias: The key ``model_validate`` expects for the sub-model field when
            it is not the field name.
        spec: The sub-model field's own spec (``engine="submodel"``; its
            ``clef`` is the presence question of an optional group). ``None``
            for the root and for a union member.
        specs: The fields directly in this group, not those of child groups:
            leaves (Clairic, Clef and const fields) and the skipped and deferred
            ones.
        children: The nested sub-model groups, in declaration order.
        unions: The unions of models that are fields of this group.
        identity: Names of the group's fields that identify a record.
        member: For the root group of a union member, the member's label.
        typed_dict: Whether the group was declared as a ``TypedDict`` (its value
            is a plain dict, not an instance of ``model``).
    """

    path: Path
    model: type[BaseModel]
    optional: bool
    nullable: bool
    has_default: bool
    description: str = ""
    alias: str | None = None
    spec: "FieldSpec | None" = None
    specs: tuple["FieldSpec", ...] = ()
    children: tuple["FieldGroup", ...] = ()
    unions: tuple["UnionGroup", ...] = ()
    identity: tuple[str, ...] = ()
    member: str = ""
    typed_dict: bool = False

    @property
    def pretty(self) -> str:
        """``path`` joined with dots (empty for the root)."""
        return pretty(self.path)

    def walk(self) -> Iterator["FieldGroup"]:
        """Yield this group and every nested group below it, depth first.

        Union members are not entered (see ``UnionGroup.members``).
        """
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True, eq=False)
class UnionMember:
    """One member model of a union, compiled as its own subtree.

    Attributes:
        label: The member's unique name (the model name, suffixed when two
            members share it). It is the value of the union spec's ``key_map``.
        model: The member model.
        group: The member's fields; its ``path`` is the union field's path.
        options: The Clef option ids that select this member (one per
            discriminator value, or the single option named after the model).
    """

    label: str
    model: type[BaseModel]
    group: FieldGroup
    options: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class UnionGroup:
    """A field that is a union of models.

    The choice is one Clef question (``spec.clef``); the member subtrees are only
    asked after the choice resolves.

    Attributes:
        path: Where the union field sits in the document.
        spec: The union field's spec (``engine="union_models"``). Its ``clef`` is
            the choice, ``key_map`` maps an option id to a member label (or to
            ``UNSTATED`` for the ``unstated`` option of an optional union).
        members: The compiled members.
        discriminator: Name of the tag field when the members share one (from
            ``Field(discriminator=...)`` or a common ``Literal`` field), else
            ``None``. Options are then the tag values, else the model names.
        option_tags: For a discriminated union, option id to tag value.
        optional: Whether the union may be absent (nullable or defaulted).
        nullable: Whether ``None`` is valid.
        has_default: Whether the field has a default.
        alias: The key ``model_validate`` expects, when not the field name.
    """

    path: Path
    spec: "FieldSpec"
    members: tuple[UnionMember, ...]
    discriminator: str | None = None
    option_tags: Mapping[str, Any] = field(default_factory=FrozenMap)
    optional: bool = False
    nullable: bool = False
    has_default: bool = False
    alias: str | None = None

    def member_for(self, option: str) -> UnionMember | None:
        """Return the member an option selects (``None`` for ``unstated``)."""
        assert self.spec.key_map is not None
        label = self.spec.key_map.get(option)
        return next((m for m in self.members if m.label == label), None)


@dataclass(frozen=True, eq=False)
class SchemaPlan:
    """Everything the engines need to know about one schema.

    Attributes:
        schema: The pydantic model.
        name: Model class name.
        title: Model title (``model_config["title"]`` or the class name).
        description: Model docstring, or an empty string.
        specs: One spec per field, depth first in declaration order: a sub-model
            or union field comes first (the container), then the fields of its
            group. Fields below a union of models are not here, because they are
            only asked once a member is chosen (see ``all_specs``).
        clairic_names: Wire names of the Clairic fields.
        clef_names: Wire names of the Clef questions.
        identity: Names of the root fields that identify a record.
        warnings: Human-readable compile warnings.
        groups: The root group followed by one group per nested sub-model, depth
            first (union members are reached through ``unions``).
        unions: Every union of models outside the members of other unions.
        member_groups: The groups inside the members of unions (member roots and
            their nested sub-models), depth first.
        member_specs: The specs of those groups, including nested union fields.
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
    groups: tuple[FieldGroup, ...] = ()
    unions: tuple[UnionGroup, ...] = ()
    member_groups: tuple[FieldGroup, ...] = ()
    member_specs: tuple[FieldSpec, ...] = ()

    @property
    def root(self) -> FieldGroup:
        """The group of the schema's own fields."""
        return self.groups[0]

    def all_groups(self) -> Iterator[FieldGroup]:
        """Yield every group, including those inside union members."""
        yield from self.groups
        yield from self.member_groups

    def all_specs(self) -> Iterator[FieldSpec]:
        """Yield every spec: ``specs`` and then those below unions of models."""
        yield from self.specs
        yield from self.member_specs

    def spec(self, path: Path, member: str | None = None) -> FieldSpec:
        """Return the spec for a path.

        Several members of a union of models may declare the same field, so a path
        below a union can be ambiguous; then ``member`` (the member's label, see
        ``UnionMember.label``) says which one, or ``spec_by_wire_path`` names the
        exact field.

        Args:
            path: The field's document path.
            member: Restrict to fields below this union member.

        Raises:
            KeyError: If no field has that path (in that member).
            ValueError: If the path belongs to several members and ``member``
                does not settle which.
        """
        found = [s for s in self.all_specs() if s.path == path]
        if member is not None:
            found = [s for s in found if member in s.member_labels]
        if not found:
            raise KeyError(path)
        if len(found) > 1:
            labels = sorted({label for s in found for label in s.member_labels})
            raise ValueError(
                f"the path {pretty(path)!r} is declared by several union members "
                f"({', '.join(labels)}); pass member=... or use spec_by_wire_path"
            )
        return found[0]

    def spec_by_wire_path(self, wire_path: Path) -> FieldSpec:
        """Return the spec whose wire names were allocated for ``wire_path``.

        Below a union of models the member's label follows the union field
        (``("pet", "Dog", "name")``), which tells members apart.

        Raises:
            KeyError: If no field has that wire path.
        """
        for spec in self.all_specs():
            if (spec.wire_path or spec.path) == wire_path:
                return spec
        raise KeyError(wire_path)

    def _select(self, paths: Iterable[Path] | None) -> list[FieldSpec]:
        every = list(self.all_specs())
        if paths is None:
            return every
        wanted = {tuple(path) for path in paths}
        known = {spec.path for spec in every}
        missing = wanted - known
        if missing:
            raise KeyError(sorted(pretty(path) for path in missing))
        return [spec for spec in every if spec.path in wanted]

    def clairic_fields(
        self, paths: Iterable[Path] | None = None
    ) -> dict[str, ClairicField]:
        """Return the Clairic fields to send, keyed by wire name.

        The fields are deep copies, so callers may change them without
        affecting the cached plan. Fields below a union of models are included
        (their wire names are unique in the plan), though the engine asks them
        only after the union's choice.

        Args:
            paths: Restrict to these paths; ``None`` means every field.

        Raises:
            KeyError: If a path is not a field of the schema.
        """
        return {
            spec.wire_name: spec.clairic.model_copy(deep=True)
            for spec in self._select(paths)
            if spec.clairic is not None
        }

    def clef_questions(
        self, paths: Iterable[Path] | None = None
    ) -> dict[str, Question]:
        """Return the Clef questions to send, keyed by wire name.

        The questions are deep copies, so callers may change them without
        affecting the cached plan. They include the presence questions of
        optional sub-models and the choices of unions of models.

        Args:
            paths: Restrict to these paths; ``None`` means every field.

        Raises:
            KeyError: If a path is not a field of the schema.
        """
        return {
            spec.clef_name: spec.clef.model_copy(deep=True)
            for spec in self._select(paths)
            if spec.clef is not None
        }

    def deferred(self) -> tuple[FieldSpec, ...]:
        """Return the specs that later engines expand (collections, mappings)."""
        return tuple(
            spec for spec in self.all_specs() if spec.engine in _DEFERRED_ENGINES
        )


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
    """Whether ``tp`` is a model-like class: a pydantic model or a TypedDict.

    A nested TypedDict is recorded like a nested model; the expander converts it
    with ``coerce_schema``.
    """
    return isinstance(tp, type) and (
        issubclass(tp, BaseModel) or typing_extensions.is_typeddict(tp)
    )


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


def _ancestors(path: Path, members: Mapping[int, str] | None = None) -> list[str]:
    """Humanised names of a path's ancestors, each union field followed by its member.

    ``members`` maps the index of a union field in ``path`` to the member's tag
    (``"Dog"`` or ``"Dog: A dog"``), giving ``"pet (Dog)"``.
    """
    names: list[str] = []
    for index, part in enumerate(path[:-1]):
        if not isinstance(part, str):
            continue
        name = _humanise(part)
        if members and index in members:
            name = f"{name} ({members[index]})"
        names.append(name)
    return names


def _context(title: str, path: Path, members: Mapping[int, str] | None = None) -> str:
    """Return the " → "-joined model title and ancestor field names.

    Below a union of models the member follows the union field:
    ``"Owner → pet (Dog)"``.
    """
    return " → ".join([title, *_ancestors(path, members)])


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
    """One choice (or score level): id, Python value and descriptions.

    ``description`` is the raw description (an enum member's own text, if any)
    and ``criterion`` the text sent as the option's criteria entry.
    """

    id: str
    value: Any
    description: str | None
    criterion: str | None


@dataclass(frozen=True)
class _Candidate:
    """A Literal value or Enum member before it gets an option id."""

    value: Any
    name: str | None = None
    description: str | None = None


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
    bounds: dict[str, Any] = field(default_factory=dict)
    models: tuple[type[BaseModel], ...] = ()
    union: "_UnionInfo | None" = None
    typed_dict: bool = False


@dataclass
class _UnionInfo:
    """What the expander needs to know about a union of models."""

    labels: list[str]
    discriminator: str | None
    option_members: dict[str, int]
    option_tags: dict[str, Any]
    criteria: dict[str, str]
    originals: list[Any] = field(default_factory=list)


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
    model: type[BaseModel]
    discriminator: Any = None
    chain: tuple[type[BaseModel], ...] = ()


def _resolve(ctx: _Ctx, tp: Any) -> Any:
    """Resolve a leftover forward reference to a class.

    pydantic leaves a reference nested in a generic (``list["Pet"]``) as a
    string on some Python versions. Names are looked up, in order, as the model
    itself, in the namespace the model was defined in
    (``__pydantic_parent_namespace__``, the locals of the defining function) and
    in the globals of the model's module. The defining namespace comes before
    the module so that a function-local class wins over an unrelated module-level
    class of the same name.
    """
    if isinstance(tp, typing.ForwardRef):
        tp = tp.__forward_arg__
    if not isinstance(tp, str):
        return tp
    if tp == ctx.model.__name__:
        return ctx.model
    local = getattr(ctx.model, "__pydantic_parent_namespace__", None)
    if isinstance(local, Mapping):
        found: Any = local.get(tp)
        if type(found).__name__ == "_PydanticWeakRef":  # newer pydantic wraps values
            found = found()
        if isinstance(found, type):
            return found
    found = getattr(sys.modules.get(ctx.model.__module__), tp, None)
    return found if isinstance(found, type) else tp


def _as_model(tp: Any) -> type[BaseModel]:
    """Return a pydantic model for a model-like class (TypedDicts are converted)."""
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return tp
    return coerce_schema(tp)


def _option_id(text: str, used: set[str]) -> str:
    base = _OPTION_BAD_CHARS.sub("_", text)[:100] or "_"
    candidate, number = base, 1
    while candidate in used:
        number += 1
        suffix = f"_{number}"
        candidate = f"{base[: 100 - len(suffix)]}{suffix}"
    used.add(candidate)
    return candidate


def _criterion(option_id: str, candidate: _Candidate) -> str:
    """Return the criteria text for an option.

    The text carries what the id alone would lose: an enum member's name and
    value, or a literal's original spelling when sanitising changed it, followed
    by the member's own description. With nothing to add it is the original
    value.
    """
    raw = str(candidate.value)
    if candidate.name is not None:
        label: str | None = (
            candidate.name if raw == candidate.name else f"{candidate.name} ({raw})"
        )
    else:
        label = raw if option_id != raw else None
    if label == option_id:
        label = None
    parts = [part for part in (label, candidate.description) if part]
    return ": ".join(parts) if parts else raw


def _options_for(candidates: list[_Candidate]) -> list[_Option]:
    used: set[str] = set()
    options: list[_Option] = []
    for candidate in candidates:
        text = candidate.name if candidate.name is not None else str(candidate.value)
        option_id = _option_id(text, used)
        options.append(
            _Option(
                option_id,
                candidate.value,
                candidate.description,
                _criterion(option_id, candidate),
            )
        )
    return options


def _unstated_option(options: list[_Option]) -> _Option:
    used = {option.id for option in options}
    return _Option(_option_id("unstated", used), UNSTATED, None, _UNSTATED_TEXT)


def _candidates(base: Any) -> list[_Candidate]:
    if _is_enum(base):
        return [
            _Candidate(member.value, member.name, _member_description(member))
            for member in base
        ]
    values, _ = _literal_values(base)
    return [_Candidate(value) for value in values]


def _choice_kind(ctx: _Ctx, candidates: list[_Candidate]) -> _Kind:
    """Build the Clef choice or score kind for a set of option values.

    A choice on a field that is nullable or has a default gets an extra
    ``unstated`` option, so an unmentioned field is not forced to a real value.
    """
    options = _options_for(candidates)
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
    if len(options) > _MAX_CHOICE_OPTIONS - (0 if ctx.required else 1):
        raise ValueError(
            f"field {ctx.pretty!r} has {len(options) + (0 if ctx.required else 1)} "
            f"options; a Clef choice supports at most {_MAX_CHOICE_OPTIONS}"
        )
    if len(options) == 1 and ctx.required:
        only = options[0]
        return _Kind("const", key_map={only.id: only.value}, post="const")
    if len(options) < 2:
        return _Kind("skip", warnings=["single_option"])
    if not ctx.required:
        options.append(_unstated_option(options))
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
            _Option("yes", True, None, "The input says this is true."),
            _Option("no", False, None, "The input says this is false."),
        ]
        if not ctx.required:
            options.append(_unstated_option(options))
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
    else:
        clairic = {"type": "number"}
        decimals = ctx.options.get("decimals", c.get("decimal_places"))
        if decimals is not None:
            clairic["decimals"] = decimals
        post = "decimal" if base is Decimal else "float"
    kind = _Kind("clairic", clairic=clairic, hints=hints, post=post)
    if "decimals" in clairic and not 0 <= clairic["decimals"] <= 6:
        clairic["decimals"] = min(6, max(0, clairic["decimals"]))
        kind.warnings.append("decimals_clamped")
    low = c.get("ge", c.get("gt"))
    high = c.get("le", c.get("lt"))
    if low is not None and high is not None:
        kind.bounds = {"min": _bound(low, base), "max": _bound(high, base)}
        if ctx.limits.estimate_for_bounded_numbers:
            clairic.clear()
            clairic.update(type="estimate", **kind.bounds)
    return kind


def _bound(value: Any, base: Any) -> Any:
    return int(value) if base is int else float(value)


def _item_kind(base: Any, unit: str) -> tuple[str, str] | None:
    """Return ``(item_kind, clairic items)`` for a supported list element."""
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
    if isinstance(base, type) and issubclass(base, _TEXT_LIKE) and not _is_enum(base):
        return ("str", "text")
    return None


def _list_kind(ctx: _Ctx, container: str, item: _Shape) -> _Kind:
    unit = ctx.options.get("unit", "seconds")
    c = ctx.constraints
    hints = _length_hints(c, "items")
    post = "" if container == "list" else container
    resolved = _resolve(ctx, item.base)
    base = _TEMPORAL_ALIASES.get(resolved, resolved)
    if _is_model(base):
        return _model_kind(ctx, "collection", (base,))
    if item.members is not None:
        members = [_resolve(ctx, m.base) for m in item.members]
        if all(_is_model(m) for m in members):
            return _model_kind(ctx, "collection", tuple(members))
    if base is typing.Any or base is object or base is dict:
        return _Kind("skip")
    if typing_extensions.get_origin(base) in _LITERAL_ORIGINS or _is_enum(base):
        candidates = _candidates(base)
        options = _options_for(candidates)
        kind = _Kind(
            "clairic",
            clairic={"type": "list", "items": "text"},
            hints=hints,
            item_kind="choice",
            element_choices={option.id: option.value for option in options},
            post=post,
        )
        kind.hints.append(
            "Each item must be one of: "
            + ", ".join(str(c.value) for c in candidates[:30])
            + "."
        )
        return kind
    parsed = _item_kind(base, unit)
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


def _too_deep(ctx: _Ctx) -> bool:
    """Whether the leaves of a sub-model at this field would be past ``max_depth``."""
    return len(ctx.path) + 1 > ctx.limits.max_depth


def _too_recursive(ctx: _Ctx, model: type[BaseModel]) -> bool:
    """Whether ``model`` would reappear on its ancestor chain too many times."""
    return ctx.chain.count(model) > ctx.limits.max_recursion


def _model_kind(ctx: _Ctx, engine: Engine, models: tuple[Any, ...]) -> _Kind:
    """Record a nested-model field, applying the depth and recursion guards.

    A sub-model is expanded by the compiler down to ``limits.max_depth`` (the
    longest leaf path), and a model may reappear on its own ancestor chain at
    most ``limits.max_recursion`` times; past either limit the field is skipped
    with a ``max_depth`` or ``max_recursion`` warning. An optional sub-model gets
    a presence question. Collections are recorded for a later engine.
    """
    if _too_deep(ctx):
        return _Kind("skip", warnings=["max_depth"])
    resolved = tuple(_as_model(model) for model in models)
    if engine == "submodel" and _too_recursive(ctx, resolved[0]):
        return _Kind("skip", warnings=["max_recursion"])
    typed = typing_extensions.is_typeddict(models[0])
    if engine == "submodel" and not ctx.required:
        return _Kind(engine, clef="noul", models=resolved, typed_dict=typed)
    return _Kind(engine, models=resolved, typed_dict=typed)


def _member_doc(model: type[BaseModel]) -> str:
    """The model's docstring on one line, or an empty string."""
    return _clean(inspect.cleandoc(model.__doc__)) if model.__doc__ else ""


def _member_tag(model: type[BaseModel], label: str) -> str:
    """The label of a union member, with its docstring when that is short."""
    doc = _member_doc(model).rstrip(".")
    return f"{label}: {doc}" if doc and len(doc) <= _SHORT_DOC else label


def _member_text(model: type[BaseModel], label: str) -> str:
    """Describe a union member for the choice: its docstring, else its fields."""
    doc = _member_doc(model)
    if doc:
        return doc
    names = ", ".join(model.model_fields) or "no fields"
    return f"{label}: has the fields {names}."


def _tag_candidates(model: type[BaseModel], name: str) -> list[_Candidate] | None:
    """The values of a model's ``Literal`` or ``Enum`` field, or ``None``."""
    info = model.model_fields.get(name)
    if info is None:
        return None
    shape = _strip(info.annotation)
    if shape.members is not None or shape.nullable:
        return None
    base = shape.base
    if typing_extensions.get_origin(base) in _LITERAL_ORIGINS or _is_enum(base):
        return _candidates(base) or None
    return None


def _disjoint(per_member: list[list[_Candidate]]) -> bool:
    flat = [repr(c.value) for candidates in per_member for c in candidates]
    return len(set(flat)) == len(flat)


def _find_discriminator(
    ctx: _Ctx, models: list[type[BaseModel]]
) -> tuple[str, list[list[_Candidate]]] | None:
    """The tag field of a union of models and each member's tag values.

    An explicit ``Field(discriminator="name")`` wins; otherwise the first field
    of the first member that every member declares as a required ``Literal`` or
    ``Enum`` with values no other member shares.
    """
    named = ctx.discriminator
    if isinstance(named, str):
        per = [_tag_candidates(m, named) for m in models]
        found = [c for c in per if c is not None]
        if len(found) == len(models) and _disjoint(found):
            return named, found
    for name in models[0].model_fields:
        per = [_tag_candidates(m, name) for m in models]
        found = [c for c in per if c is not None]
        if (
            len(found) == len(models)
            and all(m.model_fields[name].is_required() for m in models)
            and _disjoint(found)
        ):
            return name, found
    return None


def _union_models_kind(
    ctx: _Ctx, members: tuple[Any, ...], labels: list[str] | None = None
) -> _Kind:
    """Classify a union of models: one Clef choice over its members.

    The options are the discriminator's values when the members share a tag
    field (from ``Field(discriminator=...)`` or a common ``Literal``), else the
    member names; each option's criteria is the member's docstring, falling back
    to its field list. An optional union gets an ``unstated`` option. A member
    that would reappear on the ancestor chain too often is left out (the field is
    skipped when none is left); with a single option left, nothing is asked.

    Args:
        ctx: The field.
        members: The member models.
        labels: The members' labels, when already fixed (the compiler pins them
            when it drops members after expanding them).
    """
    if _too_deep(ctx):
        return _Kind("skip", warnings=["max_depth"])
    originals = list(members)
    models = [_as_model(m) for m in members]
    notes: list[str] = []
    if labels is None:
        labels = []
        for model in models:
            base, number = model.__name__, 1
            label = base
            while label in labels:
                number += 1
                label = f"{base}_{number}"
            labels.append(label)
        keep = [i for i, m in enumerate(models) if not _too_recursive(ctx, m)]
        if len(keep) != len(models):
            notes.append("max_recursion")
            models = [models[i] for i in keep]
            originals = [originals[i] for i in keep]
            labels = [labels[i] for i in keep]
        if not models:
            return _Kind("skip", warnings=notes)
    tag = _find_discriminator(ctx, models)
    candidates: list[_Candidate] = []
    owners: list[int] = []
    if tag is not None:
        discriminator: str | None = tag[0]
        for index, found in enumerate(tag[1]):
            for candidate in found:
                candidates.append(candidate)
                owners.append(index)
    else:
        discriminator = None
        for index, label in enumerate(labels):
            candidates.append(_Candidate(label))
            owners.append(index)
    options = _options_for(candidates)
    texts = [_member_text(m, label) for m, label in zip(models, labels, strict=True)]
    info = _UnionInfo(
        labels=labels,
        discriminator=discriminator,
        option_members={},
        option_tags={},
        criteria={},
        originals=originals,
    )
    key_map: dict[str, Any] = {}
    for option, candidate, owner in zip(options, candidates, owners, strict=True):
        option.value = labels[owner]
        option.criterion = texts[owner]
        key_map[option.id] = labels[owner]
        info.option_members[option.id] = owner
        if discriminator is not None:
            info.option_tags[option.id] = candidate.value
        info.criteria[option.id] = texts[owner]
    if not ctx.required:
        unstated = _unstated_option(options)
        options.append(unstated)
        key_map[unstated.id] = UNSTATED
    if len(options) > _MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"field {ctx.pretty!r}: the union offers {len(options)} options (one "
            "per discriminator value, plus `unstated` when optional) but a Clef "
            f"choice supports at most {_MAX_CHOICE_OPTIONS}"
        )
    return _Kind(
        "union_models",
        # a single option leaves nothing to ask: the member is simply used
        clef="choice" if len(options) >= _MIN_CHOICE_OPTIONS else None,
        options=options,
        key_map=key_map,
        models=tuple(models),
        union=info,
        warnings=notes,
    )


def _union_kind(ctx: _Ctx, shape: _Shape) -> _Kind:
    members = shape.members or ()
    resolved = [_resolve(ctx, m.base) for m in members]
    if all(_is_model(m) for m in resolved):
        return _union_models_kind(ctx, tuple(resolved))
    if all(typing_extensions.get_origin(m.base) in _LITERAL_ORIGINS for m in members):
        candidates: list[_Candidate] = []
        for member in members:
            candidates.extend(_candidates(member.base))
        return _choice_kind(ctx, candidates)
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
        options=[
            _Option(name, name, _UNION_TYPE_HELP[name], _UNION_TYPE_HELP[name])
            for name in names
        ],
        key_map={name: name for name in names},
        post="union",
        union_names=names,
    )


def _classify(ctx: _Ctx, shape: _Shape) -> _Kind:
    base = _resolve(ctx, shape.base)
    c = ctx.constraints
    if shape.members is not None:
        return _union_kind(ctx, shape)
    base = _TEMPORAL_ALIASES.get(base, base)
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
    if origin in _LITERAL_ORIGINS or _is_enum(base):
        return _choice_kind(ctx, _candidates(base))
    if _is_model(base):
        return _model_kind(ctx, "submodel", (base,))
    if origin is tuple:
        args = typing_extensions.get_args(base)
        if len(args) == 2 and args[1] is Ellipsis:
            return _list_kind(ctx, "list", _strip(args[0]))
        return _Kind("skip", warnings=["unsupported_type"])
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


def _doc_text(metadata: list[Any]) -> str | None:
    """Return the text of a ``typing_extensions.Doc`` annotation, if present."""
    if _DOC_TYPE is None:
        return None
    for item in metadata:
        if isinstance(item, _DOC_TYPE):
            text = getattr(item, "documentation", None)
            if isinstance(text, str):
                return text
    return None


def _description(info: FieldInfo, shape: _Shape, name: str, metadata: list[Any]) -> str:
    """Pick the field description: Field, ``Doc``, title, then the name."""
    extra = info.json_schema_extra
    schema_description = extra.get("description") if isinstance(extra, dict) else None
    for candidate in (
        info.description,
        schema_description,
        _doc_text(metadata),
        info.title,
        shape.description,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return _clean(candidate)
    return _humanise(name)


def _validation_key(name: str, info: FieldInfo) -> tuple[str | None, str | None]:
    """Return ``(alias, warning)``: the key ``model_validate`` expects.

    ``alias`` is ``None`` when the field name itself is accepted. A
    ``validation_alias`` wins over ``alias``; an ``AliasChoices`` contributes
    its first string choice. An ``AliasPath`` (or choices without a plain string)
    cannot be written as a flat key, so the field name is used with a warning.
    """
    candidate: Any = info.validation_alias
    if candidate is None:
        candidate = info.alias
    if isinstance(candidate, AliasChoices):
        strings = [c for c in candidate.choices if isinstance(c, str)]
        candidate = strings[0] if strings else candidate
    if isinstance(candidate, AliasPath):
        path = candidate.path
        candidate = (
            path[0] if len(path) == 1 and isinstance(path[0], str) else candidate
        )
    if isinstance(candidate, str):
        return (None if candidate == name else candidate), None
    if candidate is None:
        return None, None
    return None, "alias_path"


Role = Literal["clairic", "noul", "choice", "score", "union_type", "presence"]


def render_instructions(
    role: Role,
    description: str,
    *,
    prefix: str = "",
    hints: Iterable[str] = (),
    limit: int = 600,
) -> str:
    """Build the instruction string of one Clairic field or Clef question.

    The compiler uses this for the plan; the engine calls it again with a
    templated description, so a question is rebuilt from its parts rather than
    patched (the compiled text may have been clipped).

    Args:
        role: ``"clairic"`` for a Clairic field, ``"noul"``, ``"choice"`` or
            ``"score"`` for a Clef question, ``"union_type"`` for the type
            question of a union of primitives, ``"presence"`` for the question
            whether the input states an optional sub-model.
        description: What is asked for (it is the part that gets templated).
        prefix: Parent context (``FieldSpec.context``), kept as is in front.
        hints: Type and constraint sentences, for a Clairic field.
        limit: Longest result in characters; longer text is clipped.

    Returns:
        The instruction text.
    """
    if role == "clairic":
        parts = [description]
        extra = list(hints)
        if extra:
            parts = [_end_sentence(description), *extra]
        body = " ".join(parts)
    elif role == "noul":
        body = f"Does the input say this is true: {description}?"
    elif role == "choice":
        body = f"Pick the option that best fits: {description}"
    elif role == "score":
        body = f"Rate on the scale, lowest level first: {description}"
    elif role == "presence":
        body = f"Does `input` state {description}?"
    else:
        body = f"What kind of value is this: {description}?"
    return _clip(f"{prefix}: {body}" if prefix else body, limit)


def _synthesise(
    kind: _Kind,
    *,
    description: str,
    prefix: str,
    limit: int,
    role: Role,
) -> str:
    """Build the instruction string for one question."""
    return render_instructions(
        role, description, prefix=prefix, hints=kind.hints, limit=limit
    )


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
        out[option.id] = text if text is not None else option.criterion
    return out


@dataclass
class _Draft:
    """A classified field that still needs wire names and question text."""

    name: str
    path: Path
    wire_path: Path
    info: FieldInfo
    kind: _Kind
    options: Mapping[str, Any]
    description: str
    required: bool
    nullable: bool
    has_default: bool
    null_threshold: float
    min_confidence: float
    alias: str | None
    warnings: list[str]
    messages: list[tuple[str, str]]
    model: type[BaseModel]
    ctx: _Ctx
    members: dict[int, str] = field(default_factory=dict)
    member_labels: tuple[str, ...] = ()


_NUMBER_TYPES = frozenset({"number", "estimate", "forecast"})


def _warning_message(pretty_name: str, tag: str, ctx: _Ctx) -> str:
    """Turn a short warning tag into a plan warning."""
    if tag == "max_recursion":
        return (
            f"{pretty_name}: max_recursion: {ctx.model.__name__} reappears on its "
            f"own ancestor chain more than limits.max_recursion="
            f"{ctx.limits.max_recursion} times; the field is skipped"
        )
    if tag == "max_depth":
        return (
            f"{pretty_name}: max_depth: the field is nested deeper than "
            f"limits.max_depth={ctx.limits.max_depth}; it is skipped"
        )
    return f"{pretty_name}: {tag}"


def _draft_field(
    model: type[BaseModel],
    name: str,
    info: FieldInfo,
    *,
    prefix: Path = (),
    wire_prefix: Path | None = None,
    chain: tuple[type[BaseModel], ...] = (),
    members: Mapping[int, str] | None = None,
    member_labels: tuple[str, ...] = (),
    gates: GateConfig,
    limits: Limits,
) -> _Draft:
    """Classify one field and resolve everything that does not need a name.

    Args:
        model: The model that declares the field.
        name: The field's name.
        info: The field.
        prefix: Document path of the model (empty for the root model).
        wire_prefix: What wire names are allocated under (``prefix`` unless the
            model is a member of a union).
        chain: The models of the enclosing groups, root first (for the
            recursion guard).
        members: For a field below union members, the tag of each member by the
            index of its union field in ``prefix``.
        member_labels: The labels of those members, outermost first.
        gates: Thresholds.
        limits: Structural limits.
    """
    path: Path = (*prefix, name)
    wire_path: Path = (*(prefix if wire_prefix is None else wire_prefix), name)
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
        model=model,
        discriminator=info.discriminator,
        chain=chain,
    )
    kind = _Kind("skip") if options.get("skip") else _classify(ctx, shape)
    wanted_clef = options.get("clef")
    if (
        wanted_clef is not None
        and kind.clef is None
        and kind.engine not in ("skip", "const")
    ):
        raise ValueError(
            f"field {pretty_name!r}: s1_field(clef={wanted_clef!r}) only "
            "applies to bool, Literal and Enum fields"
        )
    description = _description(info, shape, name, metadata)
    if "instructions" in options:
        description = _clean(options["instructions"])
    warnings = list(kind.warnings)
    messages: list[tuple[str, str]] = []
    if _AGGREGATION_RE.search(description):
        warnings.append("aggregation")
        messages.append(
            (
                "aggregation",
                f"{pretty_name}: description {description!r} looks like an "
                "aggregate; derive it with a pydantic computed_field instead of "
                "extracting it",
            )
        )
    override = options.get("clairic_type")
    if override is not None and kind.clairic is not None:
        native = kind.clairic["type"]
        if override != native and not (
            override in _NUMBER_TYPES and native in _NUMBER_TYPES
        ):
            warnings.append("clairic_type_mismatch")
            messages.append(
                (
                    "clairic_type_mismatch",
                    f"{pretty_name}: clairic_type={override!r} does not match the "
                    f"{native!r} field the annotation implies",
                )
            )
    alias, alias_warning = _validation_key(name, info)
    if alias_warning is not None:
        warnings.append(alias_warning)
        messages.append(
            (
                alias_warning,
                f"{pretty_name}: the validation alias is an AliasPath or has no "
                "plain string choice; documents are keyed by the field name",
            )
        )
    for tag in kind.warnings:
        messages.append((tag, _warning_message(pretty_name, tag, ctx)))
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
    return _Draft(
        name=name,
        path=path,
        wire_path=wire_path,
        info=info,
        kind=kind,
        options=options,
        description=description,
        required=required,
        nullable=nullable,
        has_default=has_default,
        null_threshold=null_threshold,
        min_confidence=min_confidence,
        alias=alias,
        warnings=warnings,
        messages=messages,
        model=model,
        ctx=ctx,
        members=dict(members or {}),
        member_labels=member_labels,
    )


def _build_spec(
    draft: _Draft,
    *,
    title: str,
    limits: Limits,
    wire: str,
    clef_name: str,
    shortened_clairic: bool,
    shortened_clef: bool,
) -> FieldSpec:
    """Write the question text and assemble the spec once names are known."""
    kind, options, description = draft.kind, draft.options, draft.description
    path, pretty_name = draft.path, pretty(draft.path)
    clairic_field: ClairicField | None = None
    clef_question: Question | None = None
    instructions = description
    nested = len(path) > 1  # nested fields always carry their parent context
    context = _context(title, path, draft.members)

    if kind.clairic is not None:
        prefix = context if shortened_clairic or nested else ""
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
            if override in ("estimate", "forecast"):
                for key, bound in kind.bounds.items():
                    fields.setdefault(key, bound)
        if "unit" in options and fields["type"] != "duration":
            fields["unit"] = options["unit"]
        if "integer" in options:
            fields["integer"] = options["integer"]
        if "decimals" in options and "decimals" not in fields:
            fields["decimals"] = options["decimals"]
        fields["null_threshold"] = draft.null_threshold
        try:
            clairic_field = ClairicField(instructions=instructions, **fields)
        except ValueError as exc:
            raise ValueError(f"field {pretty_name!r}: {exc}") from exc

    if kind.clef is not None:
        prefix = context if shortened_clef or nested else ""
        role: Role = (
            "union_type"
            if kind.clairic is not None
            else "presence"
            if kind.engine == "submodel"
            else kind.clef
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
            if kind.engine == "submodel":
                default_false = f'The input does not mention: "{description}".'
            else:
                default_false = (
                    "The input states or clearly implies the opposite of: "
                    f'"{description}".'
                )
            false_text = mapping.get("false", default_false)
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

    return FieldSpec(
        path=path,
        pretty=pretty_name,
        wire_name=wire,
        annotation=draft.info.annotation,
        required=draft.required,
        has_default=draft.has_default,
        engine=kind.engine,
        clairic=clairic_field,
        clef=clef_question,
        key_map=FrozenMap(kind.key_map) if kind.key_map is not None else None,
        null_threshold=draft.null_threshold,
        min_confidence=draft.min_confidence,
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
        warnings=tuple(draft.warnings),
        clef_name=clef_name,
        nullable=draft.nullable,
        alias=draft.alias,
        models=kind.models,
        context=context if (shortened_clairic or nested) and kind.clairic else "",
        clef_context=context if (shortened_clef or nested) and kind.clef else "",
        hints=tuple(kind.hints) if kind.clairic is not None else (),
        wire_path=draft.wire_path,
        scope=" → ".join(_ancestors(path, draft.members)),
        member_labels=draft.member_labels,
    )


def _identity(specs: tuple[FieldSpec, ...]) -> tuple[str, ...]:
    """Names of the fields (of one group) that identify a record."""
    marked = tuple(str(spec.path[-1]) for spec in specs if spec.identity)
    if marked:
        return marked

    def is_text(spec: FieldSpec) -> bool:
        return spec.engine == "clairic" and _strip(spec.annotation).base is str

    by_name = {str(spec.path[-1]): spec for spec in specs}
    for candidate in _IDENTITY_NAMES:
        spec = by_name.get(candidate)
        if spec is not None and is_text(spec):
            return (candidate,)
    for spec in specs:
        if spec.required and is_text(spec):
            return (str(spec.path[-1]),)
    return ()


@dataclass
class _GroupDraft:
    """The classified fields of one model, before names and text exist.

    ``dead`` marks a group that can never validate because a required field of
    it was skipped by a limit.
    """

    model: type[BaseModel]
    path: Path
    wire_path: Path
    container: _Draft | None
    items: "list[_Draft | _GroupDraft | _UnionDraft]"
    member: str = ""
    dead: bool = False
    typed_dict: bool = False


@dataclass
class _UnionDraft:
    """A classified union of models and the drafts of its members."""

    draft: _Draft
    members: list[_GroupDraft]


@dataclass
class _Budget:
    """Counts the groups a schema compiles to against ``limits.max_groups``."""

    root: str
    limit: int
    groups: int = 0

    def take(self, model: type[BaseModel]) -> None:
        self.groups += 1
        if self.groups > self.limit:
            raise ValueError(
                f"{self.root} compiles to more than {self.limit} field groups "
                f"(limits.max_groups) while expanding {model.__name__}; reduce it "
                "with limits.max_recursion or limits.max_depth, mark the recursive "
                "fields s1_field(skip=True), or raise limits.max_groups"
            )


_LIMIT_TAGS = frozenset({"max_depth", "max_recursion", "no_viable_member"})


def _skipped_by_limit(draft: _Draft) -> bool:
    """Whether a required field was dropped by a limit (so it can never validate)."""
    return (
        draft.required
        and draft.kind.engine == "skip"
        and any(tag in _LIMIT_TAGS for tag in draft.kind.warnings)
    )


def _expand(
    model: type[BaseModel],
    *,
    path: Path,
    wire_path: Path,
    chain: tuple[type[BaseModel], ...],
    members: Mapping[int, str],
    member_labels: tuple[str, ...],
    container: _Draft | None,
    member: str,
    gates: GateConfig,
    limits: Limits,
    budget: _Budget,
) -> _GroupDraft:
    """Classify the fields of ``model`` and, recursively, of its sub-models.

    Expansion stops where a field is skipped for ``max_depth`` or
    ``max_recursion``, so a self-referencing model unrolls to the limits. A union
    member that can never validate because a required field of it was skipped is
    left out of the union (and the union field is skipped when no member is left).

    Raises:
        ValueError: If the schema compiles to more than ``limits.max_groups``
            groups.
    """
    budget.take(model)
    items: list[_Draft | _GroupDraft | _UnionDraft] = []
    dead = False

    def child(
        sub: type[BaseModel], draft: _Draft, *, label: str = "", typed: bool = False
    ) -> _GroupDraft:
        deeper = dict(members)
        labels = member_labels
        if label:
            deeper[len(draft.path) - 1] = _member_tag(sub, label)
            labels = (*member_labels, label)
        group = _expand(
            sub,
            path=draft.path,
            wire_path=(*draft.wire_path, label) if label else draft.wire_path,
            chain=(*chain, sub),
            members=deeper,
            member_labels=labels,
            container=None if label else draft,
            member=label,
            gates=gates,
            limits=limits,
            budget=budget,
        )
        group.typed_dict = typed
        return group

    for name, info in model.model_fields.items():
        draft = _draft_field(
            model,
            name,
            info,
            prefix=path,
            wire_prefix=wire_path,
            chain=chain,
            members=members,
            member_labels=member_labels,
            gates=gates,
            limits=limits,
        )
        kind = draft.kind
        if _skipped_by_limit(draft):
            dead = True
        if kind.engine == "submodel":
            sub = child(kind.models[0], draft, typed=kind.typed_dict)
            items.append(sub)
            if sub.dead and draft.required:
                dead = True
        elif kind.engine == "union_models":
            assert kind.union is not None
            labels = list(kind.union.labels)
            originals = list(kind.union.originals)
            groups = [
                child(
                    sub,
                    draft,
                    label=label,
                    typed=typing_extensions.is_typeddict(original),
                )
                for sub, label, original in zip(
                    kind.models, labels, originals, strict=True
                )
            ]
            alive = [i for i, group in enumerate(groups) if not group.dead]
            if len(alive) != len(groups):
                dropped = [labels[i] for i in range(len(groups)) if i not in alive]
                draft.warnings.append("member_excluded")
                draft.messages.append(
                    (
                        "member_excluded",
                        f"{draft.ctx.pretty}: member(s) {', '.join(dropped)} left "
                        "out of the union: a required field of them was skipped by "
                        "a limit, so they could never validate",
                    )
                )
                if alive:
                    draft.kind = _union_models_kind(
                        draft.ctx,
                        tuple(originals[i] for i in alive),
                        [labels[i] for i in alive],
                    )
                    groups = [groups[i] for i in alive]
                else:
                    draft.kind = _Kind("skip", warnings=["no_viable_member"])
                    draft.warnings.append("no_viable_member")
                    draft.messages.append(
                        (
                            "no_viable_member",
                            f"{draft.ctx.pretty}: no_viable_member: every member of "
                            "the union was left out; the field is skipped",
                        )
                    )
                    groups = []
                if _skipped_by_limit(draft):
                    dead = True
            if groups:
                items.append(_UnionDraft(draft, groups))
            else:
                items.append(draft)
        else:
            items.append(draft)
    return _GroupDraft(model, path, wire_path, container, items, member, dead)


def _name_requests(
    node: _GroupDraft,
    clairic: list[tuple[Path, str | None]],
    clef: list[tuple[Path, str | None]],
) -> None:
    """List the wire names the tree needs, in a fixed (depth-first) order."""
    if node.container is not None and node.container.kind.clef is not None:
        clef.append((node.container.wire_path, "__presence"))
    for item in node.items:
        if isinstance(item, _GroupDraft):
            _name_requests(item, clairic, clef)
        elif isinstance(item, _UnionDraft):
            if item.draft.kind.clef is not None:
                clef.append((item.draft.wire_path, None))
            for member in item.members:
                _name_requests(member, clairic, clef)
        else:
            if item.kind.clairic is not None:
                clairic.append((item.wire_path, None))
            if item.kind.clef is not None:
                clef.append(
                    (
                        item.wire_path,
                        "__type" if item.kind.clairic is not None else None,
                    )
                )


class _Assembler:
    """Turns a tree of drafts into specs, groups and union groups."""

    def __init__(
        self,
        title: str,
        limits: Limits,
        clairic_names: Mapping[Path, str],
        clef_names: Mapping[Path, str],
        clairic_short: frozenset[str],
        clef_short: frozenset[str],
    ) -> None:
        self.title = title
        self.limits = limits
        self.clairic_names = clairic_names
        self.clef_names = clef_names
        self.clairic_short = clairic_short
        self.clef_short = clef_short
        self.flat: list[FieldSpec] = []
        self.member_specs: list[FieldSpec] = []
        self.groups: list[FieldGroup | None] = []
        self.member_groups: list[FieldGroup | None] = []
        self.unions: list[UnionGroup] = []
        self.messages: list[str] = []
        self.seen: set[tuple[type[BaseModel], str, str]] = set()

    def spec(self, draft: _Draft, *, in_member: bool) -> FieldSpec:
        wire = self.clairic_names.get(draft.wire_path, "")
        clef_name = self.clef_names.get(draft.wire_path, "")
        if draft.kind.clairic is None:
            wire = clef_name
        built = _build_spec(
            draft,
            title=self.title,
            limits=self.limits,
            wire=wire,
            clef_name=clef_name,
            shortened_clairic=wire in self.clairic_short
            and draft.kind.clairic is not None,
            shortened_clef=clef_name in self.clef_short,
        )
        # a recursive model declares the same field at every level: say it once
        for tag, text in draft.messages:
            key = (draft.model, draft.name, tag)
            if key not in self.seen:
                self.seen.add(key)
                self.messages.append(text)
        (self.member_specs if in_member else self.flat).append(built)
        return built

    def group(
        self, node: _GroupDraft, *, in_member: bool, description: str = ""
    ) -> FieldGroup:
        sink = self.member_groups if in_member else self.groups
        slot = len(sink)
        sink.append(None)
        container = (
            self.spec(node.container, in_member=in_member)
            if node.container is not None
            else None
        )
        specs: list[FieldSpec] = []
        children: list[FieldGroup] = []
        unions: list[UnionGroup] = []
        for item in node.items:
            if isinstance(item, _GroupDraft):
                children.append(self.group(item, in_member=in_member))
            elif isinstance(item, _UnionDraft):
                unions.append(self.union(item, in_member=in_member))
            else:
                specs.append(self.spec(item, in_member=in_member))
        draft = node.container
        built = FieldGroup(
            path=node.path,
            model=node.model,
            optional=draft is not None and not draft.required,
            nullable=draft is not None and draft.nullable,
            has_default=draft is not None and draft.has_default,
            description=draft.description if draft is not None else description,
            alias=draft.alias if draft is not None else None,
            spec=container,
            specs=tuple(specs),
            children=tuple(children),
            unions=tuple(unions),
            identity=_identity(tuple(specs)),
            member=node.member,
            typed_dict=node.typed_dict,
        )
        sink[slot] = built
        return built

    def union(self, node: _UnionDraft, *, in_member: bool) -> UnionGroup:
        draft = node.draft
        spec = self.spec(draft, in_member=in_member)
        info = draft.kind.union
        assert info is not None
        members: list[UnionMember] = []
        for index, member in enumerate(node.members):
            group = self.group(
                member,
                in_member=True,
                description=_member_text(member.model, info.labels[index]),
            )
            members.append(
                UnionMember(
                    label=info.labels[index],
                    model=member.model,
                    group=group,
                    options=tuple(
                        option
                        for option, owner in info.option_members.items()
                        if owner == index
                    ),
                )
            )
        built = UnionGroup(
            path=draft.path,
            spec=spec,
            members=tuple(members),
            discriminator=info.discriminator,
            option_tags=FrozenMap(info.option_tags),
            optional=not draft.required,
            nullable=draft.nullable,
            has_default=draft.has_default,
            alias=draft.alias,
        )
        if not in_member:
            self.unions.append(built)
        return built


def _compile_model(
    model: type[BaseModel], gates: GateConfig, limits: Limits
) -> SchemaPlan:
    title = model.model_config.get("title") or model.__name__
    doc = inspect.cleandoc(model.__doc__) if model.__doc__ else ""
    tree = _expand(
        model,
        path=(),
        wire_path=(),
        chain=(model,),
        members={},
        member_labels=(),
        container=None,
        member="",
        gates=gates,
        limits=limits,
        budget=_Budget(model.__name__, limits.max_groups),
    )
    # Names are allocated for the whole schema at once so that a collision gives
    # every colliding path the digest form, whatever the declaration order.
    clairic_requests: list[tuple[Path, str | None]] = []
    clef_requests: list[tuple[Path, str | None]] = []
    _name_requests(tree, clairic_requests, clef_requests)
    clairic_builder = FieldNameMapBuilder("clairic")
    clef_builder = FieldNameMapBuilder("clef")
    clairic_names = dict(
        zip(
            (r[0] for r in clairic_requests),
            clairic_builder.add_all(clairic_requests),
            strict=True,
        )
    )
    clef_assigned = dict(
        zip(
            (r[0] for r in clef_requests),
            clef_builder.add_all(clef_requests),
            strict=True,
        )
    )
    assembler = _Assembler(
        title,
        limits,
        clairic_names,
        clef_assigned,
        clairic_builder.shortened,
        clef_builder.shortened,
    )
    root = assembler.group(tree, in_member=False, description=doc)
    groups = tuple(g for g in assembler.groups if g is not None)
    return SchemaPlan(
        schema=model,
        name=model.__name__,
        title=title,
        description=doc,
        specs=tuple(assembler.flat),
        clairic_names=clairic_builder.build(),
        clef_names=clef_builder.build(),
        identity=root.identity,
        warnings=tuple(assembler.messages),
        groups=groups,
        unions=tuple(assembler.unions),
        member_groups=tuple(g for g in assembler.member_groups if g is not None),
        member_specs=tuple(assembler.member_specs),
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
