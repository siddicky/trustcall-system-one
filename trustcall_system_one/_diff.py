"""Patch operations computed in code: diff the answers against a record, apply.

The diff walks the FIELD PLAN, never the raw document: a tree of ``GroupNode`` /
``LeafNode`` / ``UnionNode`` objects (built by ``_entity`` from what was asked
and answered) is compared with the existing record, and the differences become
``Change`` objects, each one operation the change gate will judge:

* an answer the record lacks (absent, or null) -> ``add``;
* an answer that differs from the record (after normalisation: ISO dates,
  ``math.isclose`` numbers, stripped text) -> ``replace``;
* an answer equal to the record -> nothing;
* a null answer, or a sub-model the input does not state -> nothing, recorded as
  ``RejectedOp(op=None, reason="not_mentioned")``: silence is not deletion;
* an identity field is never replaced by a token subset of its value (``Dana``
  must not overwrite ``Dana Whitfield``): ``identity_subset``;
* a scalar list has set semantics: only elements the record does not hold are
  added (``add /notes/-``); elements of the record that the answer does not
  repeat are left alone; a near-duplicate of a record element is a
  ``duplicate`` (or goes to a Clef check, see ``Change.near``);
* an optional sub-model the record lacks and the input states is ONE ``add`` of
  the whole sub-document; a union whose chosen member is not the record's is ONE
  ``replace`` of the whole union value;
* removals and set-to-null exist only with ``allow_null_removals``.

``sort_changes`` orders operations (replace, then remove highest array index
first, then add) and ``apply_ops`` applies them with ``jsonpatch``, repairing a
string append and missing parents on the way.
"""

import copy
import math
import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Literal, get_args, get_origin

import jsonpatch
import jsonpointer
from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from trustcall_system_one._compile import (
    FieldGroup,
    FieldSpec,
    SchemaPlan,
    UnionGroup,
    UnionMember,
)
from trustcall_system_one._types import PatchOp, RejectedOp

__all__ = [
    "MISSING",
    "ApplyResult",
    "Change",
    "DiffResult",
    "GroupNode",
    "LeafNode",
    "UnionNode",
    "apply_ops",
    "build_doc",
    "diff_group",
    "document_path",
    "is_change",
    "jsonable",
    "leaves_with_old",
    "list_paths",
    "lookup",
    "new_positions",
    "ops_for_errors",
    "pointer",
    "same",
    "sort_changes",
    "token_subset",
]

Span = tuple[int, int]
Op = Literal["add", "replace", "remove"]


class _Missing:
    """The type of ``MISSING``: no value at all (as opposed to ``None``)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()
"""A key that is not in the record."""

NEAR_LOW = 0.6
"""Text ratio from which a new list element is a *possible* repeat (Clef decides)."""


# --------------------------------------------------------------------------- #
# The tree the diff walks
# --------------------------------------------------------------------------- #


@dataclass
class LeafNode:
    """One field of a group, with the answer that stands for it.

    Attributes:
        key: The key in the document (the alias when the field has one).
        spec: The field.
        value: The candidate value (a Python value); ``None`` when the input
            does not state it or the answer was thrown away.
        handled: Whether a ``None`` value has been explained already (the
            answer was rejected by an earlier step, which reported why).
        identity: Whether the field identifies the record.
        spans: Spans in the input that support the value (they narrow the input
            of the Clef checks).
        default: The field's default as a JSON value (``MISSING`` if it has
            none). An answer equal to it is never *added* to a record: the
            default already says it.
        ref: An opaque back-reference for the builder of the tree.
    """

    key: str
    spec: FieldSpec
    value: Any = None
    handled: bool = False
    identity: bool = False
    spans: tuple[Span, ...] = ()
    default: Any = MISSING
    ref: Any = field(default=None, repr=False, compare=False)

    def is_default(self, value: Any) -> bool:
        """Whether ``value`` is the field's default (after normalisation)."""
        return self.default is not MISSING and same(
            _kind(self.spec), jsonable(value), self.default
        )


@dataclass
class GroupNode:
    """A group of fields: the root, a sub-model or a chosen union member.

    Attributes:
        key: The key in the document (``""`` for the root).
        group: The compiled group.
        present: Whether the group stands (it was stated by the input). A group
            that does not stand produces no operation.
        leaves: The group's own fields.
        children: Its sub-model groups.
        unions: Its unions of models.
    """

    key: str
    group: FieldGroup
    present: bool = True
    leaves: list[LeafNode] = field(default_factory=list)
    children: list["GroupNode"] = field(default_factory=list)
    unions: list["UnionNode"] = field(default_factory=list)

    def walk_leaves(self) -> Iterator[LeafNode]:
        """Every leaf below this group that stands, union members included."""
        if not self.present:
            return
        yield from self.leaves
        for child in self.children:
            yield from child.walk_leaves()
        for union in self.unions:
            if union.member is not None:
                yield from union.member.walk_leaves()


@dataclass
class UnionNode:
    """A union of models: the member the input chose, and its fields.

    Attributes:
        key: The key in the document.
        union: The compiled union.
        chosen: The member the choice picked (``None``: not stated).
        member: The chosen member's fields.
        handled: Whether a missing choice has been explained already.
    """

    key: str
    union: UnionGroup
    chosen: UnionMember | None = None
    member: GroupNode | None = None
    handled: bool = False


@dataclass
class Change:
    """One computed operation, before the change gate.

    Attributes:
        op: ``add``, ``replace`` or ``remove``.
        path: JSON Pointer of the target (``/notes/-`` for an appended element).
        value: The value for ``add`` and ``replace`` (JSON-compatible).
        old: What the record holds there (``None`` when nothing).
        field_path: Dotted field path, for the gate's ``proposed_change.field``.
        fields: The dotted paths of every answer behind the operation (one for
            a leaf, all leaves of a sub-document or union member).
        spans: Spans in the input that support it.
        near: For an appended list element that resembles elements of the
            record, ``(old element, text ratio)`` pairs, most similar first.
        blocked: Set when the change was thrown away before the gate.
    """

    op: Op
    path: str
    value: Any = None
    old: Any = None
    field_path: str = ""
    fields: tuple[str, ...] = ()
    spans: tuple[Span, ...] = ()
    near: list[tuple[str, float]] = field(default_factory=list)

    @property
    def removes(self) -> bool:
        """Whether this takes a value away (a removal, or a set-to-null)."""
        return self.op == "remove" or (self.op == "replace" and self.value is None)

    def as_op(self, *, repaired_from: str | None = None) -> PatchOp:
        """The operation as it is reported."""
        return PatchOp(self.op, self.path, self.value, repaired_from)


@dataclass
class DiffResult:
    """What ``diff_group`` found.

    Attributes:
        changes: The operations, in plan order (see ``sort_changes``).
        rejected: Answers that produced no operation, with the reason.
    """

    changes: list[Change] = field(default_factory=list)
    rejected: list[RejectedOp] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def pointer(parts: Sequence[str | int]) -> str:
    """Render document keys as an RFC 6901 JSON Pointer."""
    return "".join(
        "/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts
    )


def jsonable(value: Any) -> Any:
    """A Python answer as a JSON-compatible value (dates ISO, decimals text)."""
    return to_jsonable_python(value)


def lookup(record: Any, parts: Sequence[str | int]) -> Any:
    """Follow keys through a record; ``MISSING`` when any step is absent."""
    value = record
    for part in parts:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return MISSING
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def _moment(kind: str | None, text: str) -> datetime | date | time | None:
    """Parse an ISO date, time or datetime string as the kind says (else ``None``)."""
    cleaned = text.strip()
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"
    parsers: dict[str | None, Callable[[str], datetime | date | time]] = {
        "date": date.fromisoformat,
        "datetime": datetime.fromisoformat,
        "time": time.fromisoformat,
    }
    try:
        return parsers[kind](cleaned)
    except (KeyError, ValueError):
        return None


def same(kind: str | None, first: Any, second: Any) -> bool:
    """Whether two JSON values are the same after normalisation.

    Strings compare stripped; numbers by value (``math.isclose``, and a numeric
    string against a number when the field is numeric); ISO dates, times and
    datetimes as the moments they name (``Z`` is ``+00:00``); everything else by
    equality.

    Args:
        kind: The Clairic type of the field (or of a list's items): ``number``,
            ``date``, ``datetime``, ``time``, ``text`` ... or ``None``.
        first: One value.
        second: The other.
    """
    if first is None or second is None:
        return first is None and second is None
    if isinstance(first, bool) or isinstance(second, bool):
        return isinstance(first, bool) and isinstance(second, bool) and first == second
    numeric = kind in ("number", "estimate", "forecast")
    if (_is_number(first) or (numeric and isinstance(first, str))) and (
        _is_number(second) or (numeric and isinstance(second, str))
    ):
        a, b = _decimal(first), _decimal(second)
        if a is not None and b is not None:
            return a == b or math.isclose(
                float(a), float(b), rel_tol=1e-9, abs_tol=1e-9
            )
        return False
    if isinstance(first, str) and isinstance(second, str):
        if first.strip() == second.strip():
            return True
        if kind in ("date", "datetime", "time"):
            a_moment, b_moment = _moment(kind, first), _moment(kind, second)
            if a_moment is not None and type(a_moment) is type(b_moment):
                try:
                    return bool(a_moment == b_moment)
                except TypeError:  # naive against aware
                    return False
        return False
    return bool(first == second)


def _kind(spec: FieldSpec) -> str | None:
    clairic = spec.clairic
    if clairic is None:
        return None
    return (clairic.items or None) if clairic.type == "list" else clairic.type


def _is_list_spec(spec: FieldSpec) -> bool:
    return spec.clairic is not None and spec.clairic.type == "list"


_TOKEN = re.compile(r"\w+")


def token_subset(new: str, old: str) -> bool:
    """Whether ``new`` is a proper token subset of ``old`` (case-folded words).

    ``"Dana"`` is one of ``"Dana Whitfield"``, ``"Sarah"`` of ``"Sarah
    Johnson"``; ``"Dana Whitfield"`` is not a proper subset of itself and
    ``"Dana Smith"`` is not one of ``"Dana Whitfield"``.
    """
    new_tokens = set(_TOKEN.findall(new.casefold()))
    old_tokens = set(_TOKEN.findall(old.casefold()))
    return bool(new_tokens) and new_tokens < old_tokens


_NEGATION = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"\d+")


def _fold(text: str) -> str:
    """Case, punctuation and spacing folded away: ``"A, b."`` is ``"a b"``."""
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


def _differs(new: str, old: str) -> bool:
    """Whether two texts differ in a number or in having a negation.

    Such a difference makes a different fact however alike the words are
    ("Meeting on May 5" and "May 3"; "Allergic to peanuts" and "Not allergic
    to peanuts").
    """
    if sorted(_DIGITS.findall(new)) != sorted(_DIGITS.findall(old)):
        return True
    return bool(_NEGATION.search(new)) != bool(_NEGATION.search(old))


def _ratio(first: str, second: str) -> float:
    return SequenceMatcher(None, first.casefold(), second.casefold()).ratio()


# --------------------------------------------------------------------------- #
# Which answers would change the record
# --------------------------------------------------------------------------- #


def new_positions(spec: FieldSpec, value: Any, old: Any) -> list[int]:
    """The positions of the elements of a list answer that the record lacks.

    Args:
        spec: A list field.
        value: The answered list (Python values).
        old: What the record holds (``MISSING`` or ``None`` for nothing).

    Returns:
        Positions in ``value``; every position when the record has no list. An
        element repeated inside the answer is counted once.
    """
    elements = [jsonable(item) for item in value]
    kind = _kind(spec)
    held = old if isinstance(old, list) else []
    fresh: list[int] = []
    for index, element in enumerate(elements):
        if any(same(kind, element, other) for other in held):
            continue
        if any(same(kind, element, elements[j]) for j in fresh):
            continue
        fresh.append(index)
    return fresh


def is_change(spec: FieldSpec, value: Any, old: Any, default: Any = MISSING) -> bool:
    """Whether a non-null answer would make an operation for this field.

    Args:
        spec: The field.
        value: The answer (a Python value).
        old: What the record holds (``MISSING`` or ``None`` for nothing).
        default: The field's default as a JSON value; an answer equal to it is
            not added to a record that lacks the field.
    """
    if value is None:
        return False
    if _is_list_spec(spec):
        return bool(new_positions(spec, value, old))
    if old is MISSING or old is None:
        return default is MISSING or not same(_kind(spec), jsonable(value), default)
    return not same(_kind(spec), jsonable(value), old)


# --------------------------------------------------------------------------- #
# The diff
# --------------------------------------------------------------------------- #


def _member_tags(member: UnionMember) -> dict[str, Any]:
    """The single-value ``Literal`` tags a member declares with a default.

    A tag the member only defaults (``kind: Literal["dog"] = "dog"``) is not a
    field that is asked, but a union value written without it could be taken for
    another member whose keys fit, so it is written out.
    """
    tags: dict[str, Any] = {}
    for spec in member.group.specs:
        info = member.model.model_fields.get(str(spec.path[-1]))
        if info is None or get_origin(info.annotation) is not Literal:
            continue
        values = get_args(info.annotation)
        if len(values) == 1 and not info.is_required() and info.default == values[0]:
            tags[spec.alias if spec.alias is not None else str(spec.path[-1])] = (
                jsonable(values[0])
            )
    return tags


def build_doc(node: GroupNode) -> dict[str, Any] | None:
    """The sub-document a group's answers make (non-null values only).

    Returns:
        A dict keyed by document keys, or ``None`` when nothing stands.
    """
    if not node.present:
        return None
    doc: dict[str, Any] = {}
    for leaf in node.leaves:
        if leaf.value is not None and not leaf.is_default(leaf.value):
            doc[leaf.key] = jsonable(leaf.value)
    for child in node.children:
        sub = build_doc(child)
        if sub:
            doc[child.key] = sub
    for union in node.unions:
        if union.member is not None and union.chosen is not None:
            sub = build_doc(union.member)
            if sub is not None:
                doc[union.key] = {**_member_tags(union.chosen), **sub}
    return doc or None


def _leaf_paths(node: GroupNode) -> tuple[str, ...]:
    return tuple(leaf.spec.pretty for leaf in node.walk_leaves())


def _spans(node: GroupNode) -> tuple[Span, ...]:
    return tuple(span for leaf in node.walk_leaves() for span in leaf.spans)


def _member_of(union: UnionGroup, value: Mapping[str, Any]) -> UnionMember | None:
    """The member a record's union value is (by its tag, else by validating)."""
    if union.discriminator is not None:
        for member in union.members:
            tag = next(
                (
                    spec
                    for spec in member.group.specs
                    if spec.path[-1] == union.discriminator
                ),
                None,
            )
            if tag is None:
                continue
            key = tag.alias if tag.alias is not None else str(tag.path[-1])
            tags = {
                union.option_tags[o] for o in member.options if o in union.option_tags
            }
            if key in value and value[key] in tags:
                return member
    for member in union.members:
        try:
            member.model.model_validate(value)
        except ValidationError:
            continue
        return member
    return None


def leaves_with_old(
    node: GroupNode, record: Mapping[str, Any] | None
) -> Iterator[tuple[LeafNode, Any]]:
    """Every standing leaf with what the record holds for it.

    The record is followed the way the diff follows it: a sub-model the record
    lacks (or holds as something that is not an object) and a union member other
    than the record's have no record to compare with, so their leaves are
    ``MISSING``.
    """
    if not node.present:
        return
    for leaf in node.leaves:
        yield leaf, (lookup(record, [leaf.key]) if record is not None else MISSING)
    for child in node.children:
        sub = lookup(record, [child.key]) if record is not None else MISSING
        yield from leaves_with_old(child, sub if isinstance(sub, Mapping) else None)
    for union in node.unions:
        if union.member is None or union.chosen is None:
            continue
        sub = lookup(record, [union.key]) if record is not None else MISSING
        current = _member_of(union.union, sub) if isinstance(sub, Mapping) else None
        same_member = current is not None and current.label == union.chosen.label
        yield from leaves_with_old(
            union.member, sub if same_member and isinstance(sub, Mapping) else None
        )


class _Differ:
    """Collects the changes and rejections of one record."""

    def __init__(self, *, allow_null_removals: bool) -> None:
        self.removals = allow_null_removals
        self.result = DiffResult()

    def reject(
        self,
        op: str | None,
        path: str,
        reason: str,
        detail: str = "",
        p: float | None = None,
    ) -> None:
        self.result.rejected.append(RejectedOp(op, path, reason, detail, p))

    # -- groups ------------------------------------------------------------ #

    def group(
        self, node: GroupNode, record: Mapping[str, Any] | None, base: tuple[str, ...]
    ) -> None:
        for leaf in node.leaves:
            self.leaf(node, leaf, record, base)
        for child in node.children:
            self.child(child, record, base)
        for union in node.unions:
            self.union(union, record, base)

    def child(
        self, node: GroupNode, record: Mapping[str, Any] | None, base: tuple[str, ...]
    ) -> None:
        parts = (*base, node.key)
        path = pointer(parts)
        old = lookup(record, [node.key]) if record is not None else MISSING
        if not node.present:
            self.reject(None, path, "not_mentioned", "the input does not state it")
            return
        if isinstance(old, Mapping):
            self.group(node, old, parts)
            return
        doc = build_doc(node)
        if not doc:
            self.reject(None, path, "not_mentioned", "the input states none of it")
            return
        self.result.changes.append(
            Change(
                "add",
                path,
                doc,
                None,
                node.group.pretty,
                _leaf_paths(node),
                _spans(node),
            )
        )

    def union(
        self, node: UnionNode, record: Mapping[str, Any] | None, base: tuple[str, ...]
    ) -> None:
        parts = (*base, node.key)
        path = pointer(parts)
        old = lookup(record, [node.key]) if record is not None else MISSING
        if node.chosen is None or node.member is None:
            if not node.handled:
                self.reject(None, path, "not_mentioned", "the input does not state it")
            return
        if isinstance(old, Mapping):
            current = _member_of(node.union, old)
            if current is not None and current.label == node.chosen.label:
                self.group(node.member, old, parts)
                return
        doc = build_doc(node.member)
        if doc is None:
            self.reject(None, path, "not_mentioned", "the input states none of it")
            return
        doc = {**_member_tags(node.chosen), **doc}
        has_old = isinstance(old, Mapping)
        self.result.changes.append(
            Change(
                "replace" if has_old else "add",
                path,
                doc,
                dict(old) if isinstance(old, Mapping) else None,
                node.union.spec.pretty,
                _leaf_paths(node.member),
                _spans(node.member),
            )
        )

    # -- leaves ------------------------------------------------------------ #

    def leaf(
        self,
        owner: GroupNode,
        leaf: LeafNode,
        record: Mapping[str, Any] | None,
        base: tuple[str, ...],
    ) -> None:
        spec = leaf.spec
        path = pointer((*base, leaf.key))
        old = lookup(record, [leaf.key]) if record is not None else MISSING
        if leaf.value is None:
            self.silent(leaf, path, old)
            return
        if _is_list_spec(spec):
            self.list(leaf, path, old)
            return
        new = jsonable(leaf.value)
        if old is MISSING or old is None:
            if leaf.is_default(leaf.value):
                return  # the default already says it
            self.result.changes.append(
                Change("add", path, new, None, spec.pretty, (spec.pretty,), leaf.spans)
            )
            return
        if same(_kind(spec), new, old):
            return
        if (
            leaf.identity
            and isinstance(new, str)
            and isinstance(old, str)
            and token_subset(new, old)
        ):
            self.reject(
                "replace",
                path,
                "identity_subset",
                f"{new!r} is only part of the record's {old!r}",
            )
            return
        self.result.changes.append(
            Change("replace", path, new, old, spec.pretty, (spec.pretty,), leaf.spans)
        )

    def silent(self, leaf: LeafNode, path: str, old: Any) -> None:
        """A field with no answer: nothing to do, or a gated removal."""
        if leaf.handled:
            return  # an earlier step threw the answer away and said why
        if not self.removals or old is MISSING or old is None:
            self.reject(None, path, "not_mentioned", "the input does not state it")
            return
        spec = leaf.spec
        if _is_list_spec(spec) and not old:
            return  # an empty list: nothing to take away
        if spec.nullable:
            op: Op = "replace"
        elif spec.has_default:
            op = "remove"
        else:
            self.reject("remove", path, "required_field", f"{spec.pretty} is required")
            return
        self.result.changes.append(
            Change(op, path, None, old, spec.pretty, (spec.pretty,), ())
        )

    def list(self, leaf: LeafNode, path: str, old: Any) -> None:
        spec = leaf.spec
        elements = [jsonable(item) for item in leaf.value]
        fresh = new_positions(spec, leaf.value, old)
        if old is MISSING or old is None:
            if fresh:
                self.result.changes.append(
                    Change(
                        "add",
                        path,
                        [elements[i] for i in fresh],
                        None,
                        spec.pretty,
                        (spec.pretty,),
                        leaf.spans,
                    )
                )
            return
        held = (
            [item for item in old if isinstance(item, str)]
            if isinstance(old, list)
            else []
        )
        for index in fresh:
            element = elements[index]
            near: list[tuple[str, float]] = []
            target = f"{path}/-"
            if spec.item_kind == "str" and isinstance(element, str):
                folded = _fold(element)
                twin = next((other for other in held if _fold(other) == folded), None)
                if twin is not None:
                    # the same words in other case, punctuation or spacing
                    self.reject(
                        "add", target, "duplicate", f"{element!r} repeats {twin!r}", 1.0
                    )
                    continue
                ratios = sorted(
                    ((other, _ratio(element, other)) for other in held),
                    key=lambda pair: -pair[1],
                )
                # a changed number or a toggled negation is a different fact: it
                # is never dropped and not worth asking about
                near = [
                    (other, ratio)
                    for other, ratio in ratios
                    if ratio >= NEAR_LOW and not _differs(element, other)
                ]
            span = (leaf.spans[index],) if index < len(leaf.spans) else leaf.spans
            self.result.changes.append(
                Change(
                    "add",
                    target,
                    element,
                    None,
                    spec.pretty,
                    (spec.pretty,),
                    span,
                    near,
                )
            )


def diff_group(
    root: GroupNode,
    record: Mapping[str, Any],
    *,
    allow_null_removals: bool = False,
) -> DiffResult:
    """Compare the answers of a record's schema with the record.

    Args:
        root: The tree of answers (see ``GroupNode``).
        record: The existing record, keyed like the schema validates.
        allow_null_removals: Whether a null answer for a field the record holds
            proposes a removal (``remove``, or ``replace`` with null for a
            nullable field); the change gate decides.

    Returns:
        The changes (not yet sorted) and the answers that made none.
    """
    differ = _Differ(allow_null_removals=allow_null_removals)
    differ.group(root, record, ())
    return differ.result


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #

_RANK: dict[str, int] = {"replace": 0, "remove": 1, "add": 2}


def _removal_key(path: str) -> tuple[tuple[int, int, str], ...]:
    """Sort key putting the highest array index first (other parts in order)."""
    parts = path.split("/")[1:]
    return tuple((0, -int(p), "") if p.isdigit() else (1, 0, p) for p in parts)


def sort_changes(changes: Sequence[Change]) -> list[Change]:
    """Order operations deterministically: replace, remove, add.

    Replacements come first (they cannot move anything), removals next with the
    highest array index first (so an earlier removal never shifts a later
    index), then additions (appends with ``/-``) in plan order. Ties keep the
    order of ``changes``.
    """
    indexed = list(enumerate(changes))
    indexed.sort(
        key=lambda pair: (
            _RANK[pair[1].op] if not pair[1].removes else _RANK["remove"],
            _removal_key(pair[1].path) if pair[1].removes else (),
            pair[0],
        )
    )
    return [change for _, change in indexed]


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #

_PATCH_ERRORS = (jsonpatch.JsonPatchException, jsonpointer.JsonPointerException)


@dataclass
class ApplyResult:
    """What ``apply_ops`` did.

    Attributes:
        doc: The patched document (a new object; the input is not changed).
        applied: The operations that were applied, in order, as applied (a
            repaired operation is the rewritten one, with ``repaired_from``).
        failed: Operations that could not be applied, with the error.
    """

    doc: Any
    applied: list[PatchOp] = field(default_factory=list)
    failed: list[tuple[PatchOp, str]] = field(default_factory=list)


def _as_patch(op: PatchOp) -> dict[str, Any]:
    entry: dict[str, Any] = {"op": op.op, "path": op.path}
    if op.op != "remove":
        entry["value"] = op.value
    return entry


def _fix_string_concat(doc: Any, op: PatchOp) -> PatchOp | None:
    """Turn an append to a string into the replacement that concatenates.

    An ``add /field/-`` whose ``/field`` is a string (not a list) cannot apply;
    the intent was to extend the text, so it becomes ``replace /field`` with the
    old text followed by the new.
    """
    if op.op != "add" or not op.path.endswith("/-") or not isinstance(op.value, str):
        return None
    parent = op.path[:-2]
    try:
        existing = jsonpointer.resolve_pointer(doc, parent)
    except jsonpointer.JsonPointerException:
        return None
    if not isinstance(existing, str):
        return None
    return PatchOp("replace", parent, existing + op.value, repaired_from=op.path)


def _fix_missing_parents(
    doc: Any, op: PatchOp, lists: Collection[str] | None = None
) -> list[PatchOp] | None:
    """Add the objects (or the list) an ``add`` needs above its target.

    The missing ancestors are added outermost first, each as ``{}`` (``[]`` when
    the next step is ``-``); the original operation follows unchanged. Which
    parent is a list comes from the plan (``lists``, the pointers of the list
    fields): an all-digit key of an object is not an index. Without a plan only
    ``-`` says "list". Returns ``None`` if the first existing ancestor cannot
    hold a child, or if nothing is missing.
    """
    if op.op != "add":
        return None
    parts = jsonpointer.JsonPointer(op.path).parts
    if len(parts) < 2:
        return None
    # the deepest ancestor that exists
    depth = 0
    node: Any = doc
    for part in parts[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
            depth += 1
        else:
            break
    if depth == len(parts) - 1:
        return None  # the parent exists: something else is wrong
    if not isinstance(node, dict):
        return None
    fixes: list[PatchOp] = []
    for index in range(depth, len(parts) - 1):
        following = parts[index + 1]
        here = pointer(parts[: index + 1])
        is_list = following == "-" or (lists is not None and here in lists)
        container: Any = [] if is_list else {}
        fixes.append(
            PatchOp(
                "add",
                pointer(parts[: index + 1]),
                container,
                repaired_from=op.path,
            )
        )
    return [*fixes, op]


def _try(doc: Any, ops: Sequence[PatchOp]) -> Any:
    return jsonpatch.apply_patch(doc, [_as_patch(op) for op in ops], in_place=False)


def apply_ops(
    doc: Mapping[str, Any],
    ops: Sequence[PatchOp],
    *,
    lists: Collection[str] | None = None,
) -> ApplyResult:
    """Apply operations to a copy of ``doc``, one by one, repairing what it can.

    When an operation conflicts with the document, two repairs are tried in
    turn: ``_fix_string_concat`` (an append to a string becomes a replacement
    that concatenates) and ``_fix_missing_parents`` (the missing parents are
    added first). A rewritten operation, and each parent that was added,
    carries ``repaired_from``: the path of the operation as first computed. An
    operation that cannot be applied even so is reported in ``failed`` and the
    document goes on without it.

    Args:
        doc: The document.
        ops: The operations, in the order to apply them.
        lists: Pointers of the fields the schema says are lists (``list_paths``),
            to tell a missing list parent from a missing object parent.

    Returns:
        The patched copy, the applied operations and the failures.
    """
    result = ApplyResult(copy.deepcopy(dict(doc)))
    for op in ops:
        try:
            result.doc = _try(result.doc, [op])
            result.applied.append(op)
            continue
        except _PATCH_ERRORS as exc:
            error = str(exc)
        concat = _fix_string_concat(result.doc, op)
        if concat is not None:
            try:
                result.doc = _try(result.doc, [concat])
                result.applied.append(concat)
                continue
            except _PATCH_ERRORS as exc:
                error = str(exc)
        parents = _fix_missing_parents(result.doc, op, lists)
        if parents is not None:
            try:
                result.doc = _try(result.doc, parents)
                result.applied.extend(parents)
                continue
            except _PATCH_ERRORS as exc:
                error = str(exc)
        result.failed.append((op, error))
    return result


def _names_of(item: FieldSpec | FieldGroup | UnionGroup) -> set[str]:
    """The keys a field answers to in a location: its name and its alias."""
    names = {str(item.path[-1])}
    if item.alias is not None:
        names.add(item.alias)
    return names


def _member_for(union: UnionGroup, tag: str | int) -> UnionMember | None:
    """The member a location's tag segment names (its label or a tag value)."""
    text = str(tag)
    for member in union.members:
        values = {
            str(union.option_tags[o]) for o in member.options if o in union.option_tags
        }
        if text == member.label or text in values or text in member.options:
            return member
    return None


def document_path(
    plan: SchemaPlan,
    loc: Sequence[str | int],
    chosen: Mapping[str, str] | None = None,
) -> tuple[str | int, ...] | None:
    """Turn a pydantic error location into a path in the document.

    Pydantic puts the union member between a union field and the field below it
    (``("pet", "Dog", "tricks")``: the model name, or the tag value of a
    discriminated union), which is not a key of the document (``/pet/tricks``).
    This walks the plan and drops that segment. Keys may be field names or
    aliases; what the plan does not know (list indexes, mapping keys, unknown
    keys) is kept as it is.

    Args:
        plan: The compiled schema.
        loc: The ``loc`` of a validation error.
        chosen: For the unions the update decided, the document pointer of the
            union (``"/pet"``) to the label of the member that was chosen. An
            error that is about another member (it was not chosen, so it is not
            the document's) is ignored.

    Returns:
        The document path (validation keys, then indexes), or ``None`` for an
        error about a member that was not chosen.
    """
    group: FieldGroup | None = plan.root
    path: list[str | int] = []
    parts = list(loc)
    i = 0
    while i < len(parts) and group is not None:
        part = parts[i]
        key = str(part)
        union = next((u for u in group.unions if key in _names_of(u)), None)
        if union is not None:
            path.append(union.alias if union.alias is not None else str(union.path[-1]))
            i += 1
            group = None
            if i < len(parts):
                member = _member_for(union, parts[i])
                if member is not None:
                    wanted = (chosen or {}).get(pointer(path))
                    if wanted is not None and member.label != wanted:
                        return None
                    group = member.group
                    i += 1
            continue
        child = next((c for c in group.children if key in _names_of(c)), None)
        if child is not None:
            path.append(child.alias if child.alias is not None else str(child.path[-1]))
            group = child
            i += 1
            continue
        spec = next((x for x in group.specs if key in _names_of(x)), None)
        path.append(spec.alias if spec is not None and spec.alias is not None else part)
        i += 1
        group = None
    path.extend(parts[i:])
    return tuple(path)


def list_paths(root: GroupNode) -> frozenset[str]:
    """Document pointers of the list fields below a tree (see ``apply_ops``)."""
    found: set[str] = set()

    def walk(node: GroupNode, base: tuple[str, ...]) -> None:
        if not node.present:
            return
        for leaf in node.leaves:
            if _is_list_spec(leaf.spec):
                found.add(pointer((*base, leaf.key)))
        for child in node.children:
            walk(child, (*base, child.key))
        for union in node.unions:
            if union.member is not None:
                walk(union.member, (*base, union.key))

    walk(root, ())
    return frozenset(found)


def ops_for_errors(
    ops: Sequence[PatchOp], locations: Sequence[Sequence[str | int]]
) -> list[PatchOp]:
    """The operations that a validation error is about.

    An operation is implicated when the error's location and the operation's
    path lie on one line: one is a prefix of the other (``/address`` of
    ``/address/city``; ``/notes/-`` stands for ``/notes``). An error without a
    location (a model-wide check) implicates nothing.

    Args:
        ops: The applied operations.
        locations: ``loc`` of each validation error.

    Returns:
        The implicated operations, in their order.
    """
    pointers = [pointer(loc) for loc in locations if loc]
    chosen: list[PatchOp] = []
    for op in ops:
        target = op.path[:-2] if op.path.endswith("/-") else op.path
        for place in pointers:
            if (
                place == target
                or place.startswith(target + "/")
                or target.startswith(place + "/")
            ):
                chosen.append(op)
                break
    return chosen
