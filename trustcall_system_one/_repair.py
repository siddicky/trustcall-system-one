"""The validation repair loop: one narrow re-ask per failing field.

When the assembled record (an insert) or the patched record (an update) fails
pydantic validation, the loop asks again about *only the fields that failed*, with
the error in hand, instead of starting over:

1. each error's ``loc`` is mapped to a document path (``_diff.document_path``:
   the union member segment is dropped, errors of members that were not chosen
   are ignored) and then to the field of the plan it implicates (the longest
   known prefix). An error about a whole model (``loc == ()``, a
   ``model_validator``) implicates every field of that (sub-)model;
2. every implicated field gets ONE single-field request, all of them in
   parallel (call kind ``repair``, ``CallMeta.field`` the wire name): a Clairic
   request whose state is the focus header, the ``<input>`` (never the existing
   record, so a repair cannot echo it) and the blocks ``<previous_attempt>``,
   ``<validation_errors>`` and ``<expected_schema>``, or a Clef question when the
   field is adapted to Clef. The field is adapted to the error class (see
   ``classify``): a parsing error asks for a ``number`` or a date, a bad
   ``Literal`` or ``Enum`` becomes a Clef choice, a bound or a length is stated
   in prose, and so on;
3. what comes back goes through the verification ladder like any answer (the
   caller then applies the usual diff and change gate for an update);
4. the round is ONE attempt however many fields it re-asked. The loop stops when
   the record is valid, when ``repair.max_attempts`` rounds are used, when the
   error signature repeats (``repair.stop_on_repeat``) or when every field that
   is left is out of reach.

A field that failed a ``pattern`` (or another format check) is a special case.
Clairic only quotes the input, so it cannot reformat: such a field is re-asked
once, with the pattern in prose, and only when the input holds text that could
satisfy it; the answer is accepted only if it points at the input and matches.
Otherwise the field is not asked again.

This module has the pieces (``Problem``, ``locate``, ``run_round``) and the
loop (``repair_loop``); the callers in ``_entity`` supply how a field's value is
found and replaced, for inserts and for updates.
"""

import dataclasses
import functools
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from trustcall_system_one._compile import (
    FieldGroup,
    FieldSpec,
    FrozenMap,
    SchemaPlan,
    choice_for,
    render_instructions,
)
from trustcall_system_one._diff import document_path, pointer
from trustcall_system_one._run import Question, gather
from trustcall_system_one._verify import (
    Ctx,
    FieldState,
    Resolved,
    accept,
    ask_clairic,
    resolve,
    resolve_clef,
    run_ladder,
)
from trustcall_system_one._wire import (
    ChoiceAnswer,
    ClairicField,
    NoulCriteria,
    NoulQuestion,
)

__all__ = [
    "Entry",
    "Hit",
    "Outcome",
    "Pending",
    "Problem",
    "RepairState",
    "Target",
    "classify",
    "excerpt",
    "locate",
    "problems_of",
    "repair_loop",
    "run_round",
    "signature",
]

Path = tuple[str | int, ...]
Signature = frozenset[tuple[str, str]]

_MISSING_THRESHOLD = 0.95
"""``null_threshold`` of the re-ask of a required field that came back null."""
_SHOWN_CHARS = 200
_EXCERPT_CHARS = 4000
_TOKENS = 2000

_INTEGER = frozenset(
    {"int_parsing", "int_from_float", "int_parsing_size", "int_type"},
)
_FLOAT = frozenset({"float_parsing", "float_type"})
_DECIMAL = frozenset({"decimal_parsing", "decimal_type"})
_TEMPORAL = {
    "date_parsing": "date",
    "date_type": "date",
    "date_from_datetime_parsing": "date",
    "date_from_datetime_inexact": "date",
    "datetime_parsing": "datetime",
    "datetime_type": "datetime",
    "datetime_from_date_parsing": "datetime",
    "time_parsing": "time",
    "time_type": "time",
}
_CHOICE = frozenset({"enum", "literal_error"})
_BOOL = frozenset({"bool_parsing", "bool_type"})
_BOUNDS = frozenset(
    {"greater_than", "greater_than_equal", "less_than", "less_than_equal"}
    | {"multiple_of", "finite_number"}
)
_LENGTH = frozenset({"string_too_short", "string_too_long", "too_short", "too_long"})
_FORMAT_FAMILIES = ("url_", "uuid_", "ip_")
_ORDER = (
    "missing",
    "number",
    "temporal",
    "choice",
    "bool",
    "format",
    "bounds",
    "length",
    "other",
)


# --------------------------------------------------------------------------- #
# Errors as data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Problem:
    """One validation error, in the shape the loop needs.

    Attributes:
        loc: Where pydantic puts it: validation keys from the root of the
            document (the union member segment, if any, is still in it).
        type: The error type (``int_parsing``, ``missing``, ``value_error`` ...).
        msg: The message.
        input: The input value the validator saw.
        ctx: The error's context (``{"ge": 0}``, ``{"pattern": ...}``).
    """

    loc: Path
    type: str
    msg: str
    input: Any = None
    ctx: Mapping[str, Any] = field(default_factory=dict, hash=False)


def problems_of(
    exc: ValidationError, prefix: Sequence[str | int] = ()
) -> list[Problem]:
    """The errors of a ``ValidationError`` as ``Problem`` objects.

    Args:
        exc: The error.
        prefix: Where the validated document sits in the whole document (the
            keys down to a sub-document that was validated on its own).
    """
    found: list[Problem] = []
    for error in exc.errors(include_url=False):
        found.append(
            Problem(
                loc=(*prefix, *error["loc"]),
                type=error["type"],
                msg=error["msg"],
                input=error.get("input"),
                ctx=dict(error.get("ctx") or {}),
            )
        )
    return found


def signature(
    plan: SchemaPlan, problems: Sequence[Problem], chosen: Mapping[str, str]
) -> Signature:
    """The normalised set of ``(document pointer, error type)`` of some errors.

    List indexes are normalised (``/notes/2`` and ``/notes/3`` are both
    ``/notes/*``): an error that moves along a list is the same error.
    """
    found: set[tuple[str, str]] = set()
    for problem in problems:
        path = document_path(plan, problem.loc, chosen)
        if path is not None:
            normal = tuple("*" if isinstance(p, int) else p for p in path)
            found.add((pointer(normal), problem.type))
    return frozenset(found)


# --------------------------------------------------------------------------- #
# From an error location to the field
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class Entry:
    """A field that can be re-asked, with its place in the document.

    Attributes:
        path: Its document keys (aliases where there are some).
        spec: Its compiled spec.
        owner: The model that declares it.
        members: For a field below a union of models, ``(union pointer, member
            label)`` of every member it belongs to.
    """

    path: tuple[str, ...]
    spec: FieldSpec
    owner: type[BaseModel]
    members: tuple[tuple[str, str], ...] = ()


def _key(item: Any) -> str:
    return item.alias if item.alias is not None else str(item.path[-1])


def _walk(
    group: FieldGroup,
    base: tuple[str, ...],
    members: tuple[tuple[str, str], ...],
    out: list[Entry],
) -> None:
    for spec in group.specs:
        if spec.engine in ("clairic", "clef"):
            out.append(Entry((*base, _key(spec)), spec, group.model, members))
    for child in group.children:
        _walk(child, (*base, _key(child)), members, out)
    for union in group.unions:
        at = (*base, _key(union))
        for member in union.members:
            _walk(member.group, at, (*members, (pointer(at), member.label)), out)


@functools.lru_cache(maxsize=256)
def _entries(plan: SchemaPlan) -> tuple[Entry, ...]:
    found: list[Entry] = []
    _walk(plan.root, (), (), found)
    return tuple(found)


@dataclass
class Hit:
    """A field that some validation errors implicate.

    Attributes:
        entry: The field.
        problems: The errors that implicate it.
        direct: Whether an error points at the field itself, rather than at a
            model that contains it (a model-level check implicates its fields,
            and says nothing about which one is wrong).
        direct_ids: ``id`` of the errors that point at the field itself.
        named: Whether the field is implicated because the message of a
            model-level error names it (as opposed to every field of the model).
    """

    entry: Entry
    problems: list[Problem] = field(default_factory=list)
    direct: bool = False
    direct_ids: set[int] = field(default_factory=set)
    named: bool = False


def _fits(entry: Entry, chosen: Mapping[str, str]) -> bool:
    return all(chosen.get(at, label) == label for at, label in entry.members)


def _names_of(entry: Entry) -> set[str]:
    """What a message may call a field: its name, key, spaced name and title."""
    name = str(entry.spec.path[-1])
    names = {name, entry.path[-1], name.replace("_", " ")}
    info = entry.owner.model_fields.get(name)
    if info is not None and info.title:
        names.add(info.title)
    return {n.lower() for n in names if len(n) >= 3}


_PREFIX = re.compile(r"^(?:Value error|Assertion failed), ")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def message_text(message: str) -> str:
    """The part of a model-level message that can name a field.

    pydantic prefixes the message of a validator's error with ``Value error, ``
    or ``Assertion failed, ``, and a message may quote a value; neither names a
    field, so both are left out.
    """
    return _QUOTED.sub(" ", _PREFIX.sub("", message))


def _named(entry: Entry, message: str) -> bool:
    """Whether a message mentions the field (as a whole word)."""
    message = message_text(message)
    return any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", message, re.I)
        for n in _names_of(entry)
    )


def locate(
    plan: SchemaPlan, problems: Sequence[Problem], chosen: Mapping[str, str]
) -> list[Hit]:
    """Map validation errors to the fields they implicate.

    Each error location becomes a document path (``document_path``: union member
    segments dropped, errors of members that were not chosen ignored). The field
    is the longest known prefix of that path (a list index below a list field
    stays with the field). A path that stops at a model (a ``model_validator``,
    ``loc == ()`` for the root) implicates every field below it, or, when the
    message of the error names fields (by name, key, spaced name or title), only
    those.

    Args:
        plan: The compiled schema.
        problems: The errors.
        chosen: The union members that were chosen, by union pointer.

    Returns:
        The implicated fields in plan order, each with its errors.
    """
    entries = [e for e in _entries(plan) if _fits(e, chosen)]
    hits: dict[int, Hit] = {}
    for problem in problems:
        path = document_path(plan, problem.loc, chosen)
        if path is None:
            continue
        along = [e for e in entries if path[: len(e.path)] == e.path]
        if along:
            deepest = max(len(e.path) for e in along)
            picked = [(e, True) for e in along if len(e.path) == deepest]
        elif problem.type == "missing":
            continue  # a whole sub-model that is absent: nothing inside to re-ask
        else:
            below = [e for e in entries if e.path[: len(path)] == path]
            # a check that names fields in its message is about those fields
            named = [e for e in below if _named(e, problem.msg)]
            picked = [(e, False) for e in (named or below)]
        for entry, direct in picked:
            hit = hits.setdefault(id(entry), Hit(entry))
            hit.problems.append(problem)
            hit.direct = hit.direct or direct
            hit.named = hit.named or (not direct and bool(named))
            if direct:
                hit.direct_ids.add(id(problem))
    return [hits[id(e)] for e in entries if id(e) in hits]


# --------------------------------------------------------------------------- #
# What to ask
# --------------------------------------------------------------------------- #


def _category(problem: Problem, spec: FieldSpec) -> str:
    kind = problem.type
    if kind == "missing" or problem.input is None:
        # a required field that has no value: pydantic says ``missing`` for an
        # absent key and a type error (``string_type``...) for a null
        return "missing"
    clairic = spec.clairic
    scalar = spec.engine == "clairic" and clairic is not None and clairic.type != "list"
    if scalar and (kind in _INTEGER or kind in _FLOAT or kind in _DECIMAL):
        return "number"
    if scalar and kind in _TEMPORAL:
        return "temporal"
    if kind in _CHOICE:
        return "choice"
    if kind in _BOOL:
        return "bool"
    if kind == "string_pattern_mismatch" or kind.startswith(_FORMAT_FAMILIES):
        return "format"
    if kind in _BOUNDS:
        return "bounds"
    if kind in _LENGTH:
        return "length"
    return "other"


def classify(hit: Hit, spec: FieldSpec) -> tuple[str, Problem]:
    """The error class that decides how a field is re-asked.

    ``missing`` (a required field with no value: ``missing``, or the type error
    that a null gets), ``number`` (``int_parsing``, ``float_parsing``,
    ``decimal_parsing`` of a field that was asked as something else),
    ``temporal`` (``date_parsing``, ``datetime_parsing``, ``time_parsing``),
    ``choice`` (``enum``, ``literal_error``), ``bool`` (``bool_parsing``), ``format``
    (``string_pattern_mismatch`` and the other format checks), ``bounds``
    (``greater_than*``, ``less_than*``, ``multiple_of``), ``length``
    (``string_too_short/long``, ``too_short/long``) and ``other``.
    When several errors implicate the field the most specific class wins. A field
    that is only implicated through a model-level error is always ``other``.

    Args:
        hit: The field and its errors.
        spec: The field's current spec.

    Returns:
        The class and the error that decided it.

    Raises:
        ValueError: If the hit has no error.
    """
    if not hit.problems:
        raise ValueError("a hit needs at least one problem")
    if not hit.direct:
        return "other", hit.problems[0]
    ranked = sorted(
        ((_category(p, spec), p) for p in hit.problems),
        key=lambda pair: _ORDER.index(pair[0]),
    )
    return ranked[0]


def _fmt(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _constraint(problem: Problem, allowed: str | None = None) -> str:
    """A sentence fragment for what the error says the value must satisfy."""
    kind, ctx = problem.type, problem.ctx
    if kind == "greater_than":
        return f"greater than {_fmt(ctx.get('gt'))}"
    if kind == "greater_than_equal":
        return f"at least {_fmt(ctx.get('ge'))}"
    if kind == "less_than":
        return f"less than {_fmt(ctx.get('lt'))}"
    if kind == "less_than_equal":
        return f"at most {_fmt(ctx.get('le'))}"
    if kind == "multiple_of":
        return f"a multiple of {_fmt(ctx.get('multiple_of'))}"
    if kind == "finite_number":
        return "a finite number"
    if kind == "string_too_short":
        return f"at least {ctx.get('min_length')} characters"
    if kind == "string_too_long":
        return f"at most {ctx.get('max_length')} characters"
    if kind == "too_short":
        return f"at least {ctx.get('min_length')} items"
    if kind == "too_long":
        return f"at most {ctx.get('max_length')} items"
    if kind == "string_pattern_mismatch":
        return (
            f"text that matches the pattern {ctx.get('pattern')}, quoted verbatim "
            "from the input"
        )
    if kind in _CHOICE:
        return f"one of: {allowed or ctx.get('expected', 'the allowed values')}"
    if kind in _BOOL:
        return "true or false"
    if kind in _INTEGER:
        return "a whole number"
    if kind in _FLOAT or kind in _DECIMAL:
        return "a number"
    if kind in _TEMPORAL:
        return {
            "date": "an ISO 8601 date (YYYY-MM-DD)",
            "datetime": "an ISO 8601 datetime",
            "time": "an ISO 8601 time",
        }[_TEMPORAL[kind]]
    if kind == "missing":
        return "a value that the input states"
    return problem.msg


def _jsonable(value: Any) -> Any:
    try:
        return to_jsonable_python(value)
    except Exception:
        return repr(value)


def _shown(value: Any, limit: int = _SHOWN_CHARS) -> str:
    text = repr(_jsonable(value))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _clip(value: Any, limit: int = 300) -> Any:
    """A JSON value for a block; long text and structures are cut."""
    value = _jsonable(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _dotted(path: Sequence[str | int]) -> str:
    return ".".join(str(part) for part in path)


@dataclass
class Target:
    """One field to re-ask.

    Attributes:
        hit: The field and the errors that implicate it.
        spec: The field's spec as it stands now (an earlier repair may have
            adapted it).
        previous: The rejected value (``None`` when there was none).
        type_answer: For a union of primitives, the Clef choice of the member
            type, which a re-ask keeps.
        show_input: Whether ``<validation_errors>`` may show the input a
            validator saw. An update never does: that input may be a value of
            the record (the field's own answer is in ``<previous_attempt>``).
    """

    hit: Hit
    spec: FieldSpec
    previous: Any = None
    type_answer: ChoiceAnswer | None = None
    show_input: bool = True

    @property
    def wire(self) -> str:
        """The wire name the request uses."""
        return self.spec.wire_name


@dataclass
class _Ask:
    """A field adapted to its error class, ready to be sent."""

    spec: FieldSpec
    category: str
    constraint: str
    message: str
    problem: Problem
    threshold: float | None = None


def _adapt_number(spec: FieldSpec, kind: str) -> FieldSpec:
    clairic = spec.clairic
    assert clairic is not None
    if clairic.type == "number":
        return spec
    integer = kind in _INTEGER
    post = "int" if integer else "decimal" if kind in _DECIMAL else "float"
    changed = ClairicField(
        type="number",
        instructions=clairic.instructions,
        integer=True if integer else None,
        null_threshold=clairic.null_threshold,
    )
    return dataclasses.replace(spec, clairic=changed, post=post)


def _adapt_temporal(spec: FieldSpec, kind: str) -> FieldSpec:
    clairic = spec.clairic
    assert clairic is not None
    wanted = _TEMPORAL[kind]
    if clairic.type == wanted:
        return spec
    changed = ClairicField(
        type=wanted,
        instructions=clairic.instructions,
        null_threshold=clairic.null_threshold,
    )
    return dataclasses.replace(spec, clairic=changed, post="")


def _to_clef(
    spec: FieldSpec, question: Question, key_map: Mapping[str, Any] | None
) -> FieldSpec:
    """The same field, asked through Clef instead of Clairic."""
    return dataclasses.replace(
        spec,
        engine="clef",
        clairic=None,
        clef=question,
        key_map=FrozenMap(key_map) if key_map is not None else None,
        clef_name=spec.wire_name,
        post="",
        item_kind=None,
        element_choices=None,
        hints=(),
    )


def _adapt(ctx: Ctx, target: Target) -> _Ask:
    """Adapt a field to the class of its error (see ``classify``)."""
    spec = target.spec
    problems = target.hit.problems
    category, problem = classify(target.hit, spec)
    allowed = None
    if category == "choice":
        found = choice_for(spec.annotation, spec.description)
        if found is not None:
            allowed = ", ".join(repr(value) for value in found[1].values())
    constraint = (
        _constraint(Problem((), "missing", ""))
        if category == "missing"
        else "; ".join(dict.fromkeys(_constraint(p, allowed) for p in problems))
    )
    ask = _Ask(
        spec,
        category,
        constraint,
        "; ".join(dict.fromkeys(p.msg for p in problems)),
        problem,
    )
    limit = ctx.run.limits.max_instruction_chars
    prefix = spec.context or spec.clef_context
    if category == "number":
        ask.spec = _adapt_number(spec, problem.type)
    elif category == "temporal":
        ask.spec = _adapt_temporal(spec, problem.type)
    elif category == "choice":
        found = choice_for(
            spec.annotation,
            ctx.desc(spec),
            prefix=prefix,
            required=not spec.nullable and not spec.has_default,
            limit=limit,
        )
        if found is not None:
            ask.spec = _to_clef(spec, found[0], found[1])
    elif category == "bool":
        question = NoulQuestion(
            type="noul",
            instructions=render_instructions(
                "noul", ctx.desc(spec), prefix=prefix, limit=limit
            ),
            criteria=NoulCriteria(
                true="The input says this is true.",
                false="The input says this is false.",
            ),
        )
        ask.spec = _to_clef(spec, question, None)
    elif category == "missing" and spec.engine == "clairic":
        ask.threshold = _MISSING_THRESHOLD
    return ask


def _suffix(ask: _Ask, target: Target) -> str:
    """The prose that follows a field's own instructions."""
    parts = [
        f"Must satisfy: {ask.constraint}",
        f"The previous answer {_shown(target.previous)} was rejected because: "
        f"{ask.message}",
    ]
    if ask.category == "missing":
        parts.append("The input states this; quote it")
    elif ask.category == "format":
        parts.append("Quote the exact text from the input; do not reformat it")
    return " ".join(part if part.endswith(".") else part + "." for part in parts)


# --------------------------------------------------------------------------- #
# The state of a request
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=64)
def _schema_of(model: type[BaseModel]) -> Mapping[str, Any] | None:
    try:
        return model.model_json_schema(by_alias=True, mode="validation")
    except Exception:
        return None


def _resolve_ref(node: Any, defs: Mapping[str, Any]) -> Any:
    for _ in range(16):
        if not (isinstance(node, Mapping) and "$ref" in node):
            break
        target = defs.get(str(node["$ref"]).rsplit("/", 1)[-1])
        if target is None:
            break
        node = target
    return node


def _step(node: Any, part: str | int, defs: Mapping[str, Any], depth: int = 0) -> Any:
    node = _resolve_ref(node, defs)
    if not isinstance(node, Mapping) or depth > 8:
        return None
    if isinstance(part, int):
        if "items" in node:
            return node["items"]
    else:
        properties = node.get("properties")
        if isinstance(properties, Mapping) and part in properties:
            return properties[part]
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in node.get(key, ()):
            found = _step(branch, part, defs, depth + 1)
            if found is not None:
                return found
    return None


def _inline(node: Any, defs: Mapping[str, Any], depth: int = 0) -> Any:
    """Replace every ``$ref`` by its definition (as deep as ``depth`` allows)."""
    if isinstance(node, Mapping):
        if "$ref" in node:
            if depth >= 6:
                return {"type": "object"}
            resolved = _resolve_ref(node, defs)
            if resolved is node:
                return {k: v for k, v in node.items() if k != "$ref"}
            return _inline(resolved, defs, depth + 1)
        return {k: _inline(v, defs, depth) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline(item, defs, depth) for item in node]
    return node


def excerpt(model: type[BaseModel], path: Sequence[str | int]) -> str:
    """The JSON-schema excerpt for a document path, with every ``$ref`` resolved.

    Args:
        model: The root model.
        path: Document keys (and list indexes) down to the field.

    Returns:
        Indented JSON text for the deepest part of the path the schema knows;
        empty braces when the model has no JSON schema.
    """
    schema = _schema_of(model)
    if schema is None:
        return "{}"
    defs = schema.get("$defs", {})
    node: Any = schema
    for part in path:
        found = _step(node, part, defs)
        if found is None:
            break
        node = found
    text = json.dumps(_inline(node, defs), indent=2, ensure_ascii=False, default=str)
    if len(text) > _EXCERPT_CHARS:
        text = text[:_EXCERPT_CHARS] + "\n... (cut)"
    return text


def _error_objects(target: Target) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for problem in target.hit.problems:
        item: dict[str, Any] = {
            "field": _dotted(problem.loc) or "(record)",
            "message": problem.msg,
            "type": problem.type,
        }
        # The input is shown only where it is the field's own value: a scalar at
        # a leaf. For a model-level or container error pydantic's input is the
        # whole document (on an update: the record).
        if (
            target.show_input
            and problem.type != "missing"
            and id(problem) in target.hit.direct_ids
            and (problem.input is None or isinstance(problem.input, (str, int, float)))
        ):
            item["input"] = _clip(problem.input)
        objects.append(item)
    return objects


@dataclass(frozen=True)
class _Blocks:
    """The three descriptions of a rejected answer."""

    previous: Any
    errors: list[dict[str, Any]]
    schema: str

    def trailer(self) -> str:
        """They as the blocks that follow the input of a Clairic state."""
        previous = json.dumps(self.previous, ensure_ascii=False, default=str)
        errors = json.dumps(self.errors, indent=2, ensure_ascii=False, default=str)
        return (
            f"<previous_attempt>\n{previous}\n</previous_attempt>\n"
            f"<validation_errors>\n{errors}\n</validation_errors>\n"
            f"<expected_schema>\n{self.schema}\n</expected_schema>"
        )

    def clef_keys(self) -> dict[str, Any]:
        """They as the extra keys of a Clef state."""
        try:
            schema: Any = json.loads(self.schema)
        except ValueError:
            schema = self.schema
        return {
            "previous_attempt": self.previous,
            "validation_errors": self.errors,
            "expected_schema": schema,
        }


def _blocks(ctx: Ctx, target: Target) -> _Blocks:
    return _Blocks(
        _jsonable(target.previous),
        _error_objects(target),
        excerpt(ctx.plan.schema, target.hit.entry.path),
    )


# --------------------------------------------------------------------------- #
# Patterns and other formats
# --------------------------------------------------------------------------- #


def _without_anchors(pattern: str) -> str:
    core = pattern
    if core.startswith("^"):
        core = core[1:]
    elif core.startswith("\\A"):
        core = core[2:]
    if core.endswith("\\$"):
        return core
    if core.endswith("$"):
        core = core[:-1]
    elif core.endswith("\\Z"):
        core = core[:-2]
    return core


def _pattern_in(pattern: str, text: str) -> bool:
    """Whether the pattern matches anywhere in the text, anchors aside."""
    for candidate in dict.fromkeys((pattern, _without_anchors(pattern))):
        try:
            if re.search(candidate, text, re.MULTILINE):
                return True
        except re.error:
            continue
    return False


def _adapter(entry: Entry) -> "TypeAdapter[Any] | None":
    """A validator for the field alone: its type and its constraints."""
    info = entry.owner.model_fields.get(str(entry.spec.path[-1]))
    if info is None or info.annotation is None:
        return None
    try:
        if not info.metadata:
            return TypeAdapter(info.annotation)
        params = (info.annotation, *info.metadata)
        return TypeAdapter(cast(Any, Annotated)[params])
    except Exception:
        return None


def _tokens(text: str) -> list[str]:
    found = [t.strip(".,;:!?()[]<>\"'") for t in re.findall(r"\S+", text)]
    return [t for t in dict.fromkeys(found) if t][:_TOKENS]


def _candidate_exists(problem: Problem, entry: Entry, text: str) -> bool:
    """Whether the input holds a verbatim piece that could satisfy the format.

    For a regex ``pattern``: a match of the pattern (without its ``^``/``$``
    anchors) anywhere in the input. For another format check: a piece of the
    input (a whitespace-separated token) that the field's own validation accepts.
    """
    pattern = problem.ctx.get("pattern")
    if problem.type == "string_pattern_mismatch" and isinstance(pattern, str):
        return _pattern_in(pattern, text)
    adapter = _adapter(entry)
    if adapter is None:
        return True  # cannot tell: let the one re-ask decide
    for token in _tokens(text):
        try:
            adapter.validate_python(token)
        except ValidationError:
            continue
        return True
    return False


def _matches(problem: Problem, entry: Entry, value: Any) -> bool:
    """Whether a re-asked value meets the format the error complained about."""
    adapter = _adapter(entry)
    if adapter is not None:
        try:
            adapter.validate_python(value)
        except ValidationError:
            return False
        return True
    pattern = problem.ctx.get("pattern")
    if isinstance(pattern, str) and isinstance(value, str):
        try:
            return re.search(pattern, value) is not None
        except re.error:
            return False
    return True


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #


@dataclass
class RepairState:
    """What the loop of one record remembers.

    Attributes:
        rounds: Repair rounds used (one round is one attempt).
        seen: The error signatures of the rounds so far.
        exhausted: Wire names that are not asked again: a format error without a
            candidate in the input, one that was re-asked once already, or one
            whose re-ask was not accepted.
        formatted: Wire names that had their one format re-ask.
        tentative: Answers to a model-level error that named no field, installed
            but not yet known to be needed (see ``repair_loop``).
        bad: ``(wire name, value)`` pairs that an error implicated by name and
            that a repair then replaced: never put back.
    """

    rounds: int = 0
    seen: set[Signature] = field(default_factory=set)
    exhausted: set[str] = field(default_factory=set)
    formatted: set[str] = field(default_factory=set)
    tentative: list["Outcome"] = field(default_factory=list)
    bad: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class Outcome:
    """What became of one field's repair.

    Attributes:
        target: The field.
        spec: The spec the answer was asked with (the field's spec from now on if
            the repair stands).
        note: One line of history (``repair[1] (number): ... -> ...``).
        category: The error class the field was asked as (see ``classify``).
        state: A Clairic answer that stands (it passed the ladder), else ``None``.
        resolved: A Clef answer that stands, else ``None``.
        failfast: The field is not asked again.
        undo: Put the field's earlier answer back (set by the caller that
            installed this outcome, ``None`` before).
        redo: Install this outcome again after an ``undo``.
        restores: The value the field held before this outcome was installed
            (what ``undo`` puts back).
    """

    target: Target
    spec: FieldSpec
    note: str
    category: str = "other"
    state: FieldState | None = None
    resolved: Resolved | None = None
    failfast: bool = False
    undo: Callable[[], None] | None = None
    redo: Callable[[], None] | None = None
    restores: Any = None

    @property
    def repaired(self) -> bool:
        """Whether a new value stands."""
        return self.state is not None or self.resolved is not None


def _same(first: Any, second: Any) -> bool:
    return bool(_jsonable(first) == _jsonable(second))


def _stamp(resolved: Resolved, note: str) -> Resolved:
    """Mark an answer as repaired and put the note in its history."""
    return dataclasses.replace(resolved, repaired=True, notes=(*resolved.notes, note))


def _note(label: str, ask: _Ask, target: Target, resolved: Resolved) -> str:
    """The history line of a repair that went through."""
    return (
        f"{label} ({ask.category}): {_shown(target.previous)} -> "
        f"{_shown(resolved.value)}"
    )


async def _ask_clef(
    ctx: Ctx, ask: _Ask, target: Target, blocks: _Blocks, label: str
) -> Outcome:
    spec = ask.spec
    wire = spec.wire_name
    failed = Outcome(target, spec, "", ask.category)
    assert spec.clef is not None
    base = spec.clef.instructions
    text = base if isinstance(base, str) else json.dumps(base)
    question = spec.clef.model_copy(
        update={
            "instructions": f"{text.rstrip()} {_suffix(ask, target)} "
            "See `validation_errors` and `expected_schema`."
        }
    )
    state = ctx.fitted()
    answers = await ctx.run.call_clef(
        {wire: question},
        {**state, **blocks.clef_keys()},
        meta=ctx.meta("repair", field=wire),
    )
    resolved = resolve_clef(spec, answers.get(wire), ctx)
    why = None
    if not resolved.ok:
        why = f"{resolved.reason}: {resolved.detail}"
    elif resolved.value is None:
        why = "the input does not state a value"
    elif _same(resolved.value, target.previous):
        why = "the answer is the value that was rejected"
    if why is not None:
        failed.note = f"{label} ({ask.category}) failed: {why}"
        return failed
    note = _note(label, ask, target, resolved)
    return Outcome(target, spec, note, ask.category, resolved=_stamp(resolved, note))


async def _ask_clairic(
    ctx: Ctx,
    ask: _Ask,
    target: Target,
    blocks: _Blocks,
    label: str,
) -> Outcome:
    spec = ask.spec
    wire = spec.wire_name
    is_format = ask.category == "format"
    failed = Outcome(target, spec, "", ask.category, failfast=is_format)

    def fail(why: str) -> Outcome:
        failed.note = f"{label} ({ask.category}) failed: {why}"
        return failed

    base = ctx.clairic_field(spec, null_threshold=ask.threshold)
    text = base.instructions or ""
    joiner = " " if text.endswith((".", "!", "?")) else ". "
    sent = base.model_copy(
        update={"instructions": text + joiner + _suffix(ask, target)}
    )
    asking = dataclasses.replace(ctx, trailer=blocks.trailer())
    asked = await ask_clairic(
        asking,
        {wire: sent},
        meta=ctx.meta("repair", field=wire),
        descriptions={wire: spec.scoped},
    )
    answer, anchored = asked[wire]
    resolved = await resolve(ctx, spec, answer, anchored, target.type_answer)
    if "span_outside_input" in resolved.notes:
        return fail("the answer was read from the blocks after the input, not from it")
    rejection = accept(spec, resolved, null_threshold=ask.threshold)
    if rejection is not None:
        return fail(f"{rejection[0]}: {rejection[1]}")
    if resolved.value is None:
        return fail("the input does not state a value")
    if _same(resolved.value, target.previous):
        return fail("the answer is the value that was rejected")
    if is_format:
        if resolved.span is None and not resolved.spans:
            return fail("the answer does not point at text in the input")
        if not _matches(ask.problem, target.hit.entry, resolved.value):
            return fail(f"{_shown(resolved.value)} still does not meet the format")
    note = _note(label, ask, target, resolved)
    return Outcome(
        target,
        spec,
        note,
        ask.category,
        state=FieldState(spec, resolved, target.type_answer),
    )


async def _ask_one(ctx: Ctx, state: RepairState, target: Target) -> Outcome:
    """Re-ask one field and turn the answer into an outcome (before the ladder)."""
    ask = _adapt(ctx, target)
    label = f"repair[{state.rounds + 1}]"
    blocks = _blocks(ctx, target)
    if ask.category == "format":
        state.formatted.add(target.wire)
    if ask.spec.engine == "clef":
        return await _ask_clef(ctx, ask, target, blocks, label)
    return await _ask_clairic(ctx, ask, target, blocks, label)


async def run_round(
    ctx: Ctx, state: RepairState, targets: Sequence[Target]
) -> list[Outcome]:
    """Re-ask every target in parallel, then verify what came back.

    The answers of the Clairic fields go through the verification ladder
    together (``run_ladder`` on ``ctx``, so one attribution request covers them
    all); one that the ladder throws away is not used. A Clef answer was already
    judged by its thresholds.

    Args:
        ctx: The entity context without the existing record (input only).
        state: The loop's memory (the round number and the format re-asks).
        targets: The fields to re-ask.

    Returns:
        One outcome per target: the new value if one stands, else why not.
    """
    outcomes = list(await gather(*(_ask_one(ctx, state, t) for t in targets)))
    asked = [o for o in outcomes if o.state is not None]
    await run_ladder(ctx, [o.state for o in asked if o.state is not None])
    for outcome in asked:
        value = outcome.state
        assert value is not None
        if value.has_value():
            value.resolved = _stamp(value.resolved, outcome.note)
            continue
        label = f"repair[{state.rounds + 1}] ({outcome.category})"
        outcome.note = (
            f"{label} failed: the answer did not pass verification ({value.reject})"
        )
        outcome.failfast = outcome.category == "format"
        outcome.state = None
    return outcomes


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


@dataclass
class Pending:
    """The errors a record still has.

    Attributes:
        problems: The validation errors (empty when the record is valid).
        chosen: The union members that were chosen, by union pointer.
    """

    problems: list[Problem]
    chosen: dict[str, str] = field(default_factory=dict)


def _blocker(ctx: Ctx, state: RepairState, target: Target) -> str | None:
    """Why a field with a format error is not re-asked (``None``: it may be)."""
    category, problem = classify(target.hit, target.spec)
    if category != "format":
        return None
    if not ctx.run.repair.allow_pattern_reask:
        return "re-asking for a format is switched off (allow_pattern_reask)"
    if target.wire in state.formatted:
        return "it was re-asked once for its format already"
    if not _candidate_exists(problem, target.hit.entry, ctx.text):
        return "the input holds no text that could satisfy the format"
    return None


async def repair_loop(
    ctx: Ctx,
    state: RepairState,
    pending: Pending,
    *,
    refresh: Callable[[], Awaitable[Pending]],
    target_for: Callable[[Hit], Target | None],
    apply: Callable[[list[Outcome]], Awaitable[None]],
) -> Pending:
    """Repair a record round by round until it validates or nothing more can be done.

    Args:
        ctx: The entity context without the existing record (input only).
        state: The loop's memory.
        pending: The errors the record has now.
        refresh: Rebuilds the record from the changed answers and returns its
            errors (empty when it is valid).
        target_for: The field a hit stands for, with its rejected value; ``None``
            when it cannot be repaired (it is not one of this record's answers,
            or an update did not touch it).
        apply: Puts the outcomes into the record's answers; a field that an
            outcome marks ``failfast`` is not asked again.

    Returns:
        The errors that are left (empty when the record validates).
    """
    config = ctx.run.repair
    plan = ctx.plan
    while pending.problems and state.rounds < config.max_attempts:
        sig = signature(plan, pending.problems, pending.chosen)
        if config.stop_on_repeat and sig in state.seen:
            break
        state.seen.add(sig)
        targets: list[Target] = []
        skipped: list[Outcome] = []
        for hit in locate(plan, pending.problems, pending.chosen):
            target = target_for(hit)
            if target is None or target.wire in state.exhausted:
                continue
            blocker = _blocker(ctx, state, target)
            if blocker is None:
                targets.append(target)
                continue
            skipped.append(
                Outcome(
                    target,
                    target.spec,
                    f"repair[{state.rounds + 1}] skipped: {blocker}",
                    classify(hit, target.spec)[0],
                    failfast=True,
                )
            )
        asked = await run_round(ctx, state, targets) if targets else []
        outcomes = [*skipped, *asked]
        state.exhausted.update(o.target.wire for o in outcomes if o.failfast)
        if outcomes:
            await apply(outcomes)
        if not targets:
            break
        state.rounds += 1
        chosen = pending.chosen
        pending = await refresh()
        fresh = [
            o
            for o in outcomes
            if o.repaired and o.undo is not None and not o.target.hit.direct
        ]
        groups = (
            [o for o in fresh if o.target.hit.named],  # judged as a set
            [o for o in fresh if not o.target.hit.named],  # judged one by one later
        )
        kept: list[list[Outcome]] = []
        for group in groups:
            if group and pending.problems:
                # answers to a model-level error stay only if that error went away
                cause = _errors(plan, _model_level(group), chosen)
                after = _errors(plan, pending.problems, pending.chosen)
                if not cause - after:
                    pending = await _revert(group, refresh)
                    group = []
            kept.append(group)
        for outcome in kept[0]:  # what a message named is never reinstated
            state.bad.add(_pair(outcome))
        state.tentative.extend(kept[1])
        if not pending.problems and state.tentative:
            pending = await _minimise(state, refresh)
    if pending.problems and state.tentative:  # still invalid: none of it helped
        pending = await _revert(_free(state, state.tentative), refresh)
        state.tentative.clear()
    return pending


def _errors(
    plan: SchemaPlan, problems: Sequence[Problem], chosen: Mapping[str, str]
) -> set[tuple[str, str, str]]:
    """Errors with their messages: a new validator's error is not the old one."""
    found: set[tuple[str, str, str]] = set()
    for problem in problems:
        path = document_path(plan, problem.loc, chosen)
        if path is not None:
            normal = tuple("*" if isinstance(p, int) else p for p in path)
            found.add((pointer(normal), problem.type, problem.msg))
    return found


def _pair(outcome: Outcome) -> tuple[str, str]:
    """The field and the value this outcome replaced."""
    return outcome.target.wire, json.dumps(
        _jsonable(outcome.restores), sort_keys=True, default=str
    )


def _free(state: RepairState, outcomes: list[Outcome]) -> list[Outcome]:
    """The outcomes whose undoing would not bring back a value known to be wrong."""
    return [o for o in outcomes if _pair(o) not in state.bad]


def _model_level(outcomes: list[Outcome]) -> list[Problem]:
    """The model-level errors that made the repair of these fields necessary."""
    return [
        p
        for o in outcomes
        for p in o.target.hit.problems
        if id(p) not in o.target.hit.direct_ids
    ]


async def _revert(
    outcomes: list[Outcome], refresh: Callable[[], Awaitable[Pending]]
) -> Pending:
    """Take answers back (a repair that did not help)."""
    for outcome in reversed(outcomes):
        assert outcome.undo is not None
        outcome.undo()
    return await refresh()


async def _minimise(
    state: RepairState, refresh: Callable[[], Awaitable[Pending]]
) -> Pending:
    """Once the record validates, keep only the unnamed tentative answers it needs.

    These are the answers to a model-level error that named no field (answers
    to the fields a message names are a set and are never taken back one by
    one). An answer that would bring back a value that a message named and a
    repair replaced is not a candidate. Each answer is taken back in turn and
    stays back as long as the record still validates without it; this repeats
    until no single answer can go, so answers that are only needed together are
    kept.
    """
    kept = list(state.tentative)
    changed = True
    while changed:
        changed = False
        for outcome in _free(state, kept):
            assert outcome.undo is not None and outcome.redo is not None
            outcome.undo()
            if (await refresh()).problems:
                outcome.redo()
            else:
                kept.remove(outcome)
                changed = True
    state.tentative.clear()
    state.tentative.extend(kept)
    return await refresh()
