"""Flatten nested documents to leaf paths and back, and prune absent values."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from trustcall_system_one._names import Path, pretty

if TYPE_CHECKING:
    from trustcall_system_one._compile import FieldGroup, SchemaPlan, UnionGroup

__all__ = ["flatten", "prune_nulls", "unflatten"]


def flatten(doc: Mapping[str, Any]) -> dict[Path, Any]:
    """Flatten a JSON-like document into ``{path: leaf}``.

    Dict keys become ``str`` path segments and list positions become ``int``
    segments. Scalars, ``None``, empty dicts and empty lists are leaves; an
    empty container is kept as a leaf so that ``unflatten`` restores it. Tuples
    are leaves too (they are not JSON arrays).

    Args:
        doc: The document. Every dict key must be a ``str``.

    Returns:
        Leaf values keyed by path, in document order.

    Raises:
        TypeError: If a dict key is not a string.
    """
    leaves: dict[Path, Any] = {}

    def walk(value: Any, path: Path) -> None:
        if isinstance(value, Mapping) and value:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"document keys must be strings, got {key!r} at "
                        f"{pretty(path) or '<root>'}"
                    )
                walk(child, (*path, key))
        elif isinstance(value, list) and value:
            for index, child in enumerate(value):
                walk(child, (*path, index))
        else:
            leaves[path] = value

    walk(doc, ())
    # An empty document flattens to no leaves; a non-empty one never has path ().
    leaves.pop((), None)
    return leaves


def _segment_kind(segment: object) -> str:
    if isinstance(segment, bool):
        raise TypeError(f"path segments must be str or int, got {segment!r}")
    if isinstance(segment, int):
        return "index"
    if isinstance(segment, str):
        return "key"
    raise TypeError(f"path segments must be str or int, got {segment!r}")


def _build(entries: list[tuple[Path, Any]], depth: int, warnings: list[str]) -> Any:
    first_path = entries[0][0]
    if any(len(path) == depth for path, _ in entries):
        if len(entries) > 1:
            raise ValueError(
                f"conflicting paths: {pretty(first_path[:depth]) or '<root>'} is "
                "both a value and a container"
            )
        if depth == 0:
            raise ValueError("a path must not be empty")
        return entries[0][1]
    kinds = {_segment_kind(path[depth]) for path, _ in entries}
    if len(kinds) > 1:
        raise ValueError(
            f"{pretty(first_path[:depth]) or '<root>'} mixes dict keys and list indexes"
        )
    groups: dict[str | int, list[tuple[Path, Any]]] = {}
    for path, value in entries:
        groups.setdefault(path[depth], []).append((path, value))
    if kinds == {"key"}:
        return {
            key: _build(group, depth + 1, warnings) for key, group in groups.items()
        }
    indexes = sorted(index for index in groups if isinstance(index, int))
    if indexes[0] < 0:
        raise ValueError(f"negative list index at {pretty(first_path[:depth])}")
    missing = sorted(set(range(indexes[-1] + 1)) - set(indexes))
    if missing:
        warnings.append(
            f"list at {pretty(first_path[:depth]) or '<root>'} is missing "
            f"indexes {missing}; the remaining items were compacted"
        )
    return [_build(groups[index], depth + 1, warnings) for index in indexes]


def _level_keys(
    group: "FieldGroup", members: Mapping[Path, str]
) -> dict[str, tuple[str | None, bool, "FieldGroup | None"]]:
    """Map each field name of a group to ``(alias, has_default, nested group)``.

    The nested group is the one that describes the field's value when that is a
    document of its own: a sub-model's group, or the chosen member of a union.
    """
    keys: dict[str, tuple[str | None, bool, FieldGroup | None]] = {}
    for spec in group.specs:
        keys[str(spec.path[-1])] = (spec.alias, spec.has_default, None)
    for child in group.children:
        keys[str(child.path[-1])] = (child.alias, child.has_default, child)
    for union in group.unions:
        label = members.get(union.path)
        chosen = next((m for m in union.members if m.label == label), None)
        keys[str(union.path[-1])] = (
            union.alias,
            union.has_default,
            chosen.group if chosen is not None else _guess_member(union),
        )
    return keys


def _guess_member(union: "UnionGroup") -> "FieldGroup | None":
    """Without a recorded choice, a union's value cannot be told apart by key."""
    return union.members[0].group if len(union.members) == 1 else None


def _walk_level(
    doc: Mapping[str, Any],
    group: "FieldGroup",
    members: Mapping[Path, str],
    *,
    prune: bool,
) -> dict[str, Any]:
    """Rename one level's keys to validation keys (and prune defaultable nulls)."""
    keys = _level_keys(group, members)
    by_alias = {
        alias: name for name, (alias, _, _) in keys.items() if alias is not None
    }
    out: dict[str, Any] = {}
    for key, value in doc.items():
        name = key if key in keys else by_alias.get(key)
        if name is None:
            out[key] = value  # not a field of the plan: kept unchanged
            continue
        alias, has_default, nested = keys[name]
        if prune and value is None and has_default:
            continue
        if nested is not None and isinstance(value, Mapping):
            value = _walk_level(value, nested, members, prune=prune)
        out[alias if alias is not None else name] = value
    return out


def unflatten(
    items: Mapping[Path, Any],
    warnings: list[str] | None = None,
    *,
    plan: "SchemaPlan | None" = None,
    members: Mapping[Path, str] | None = None,
) -> dict[str, Any]:
    """Rebuild a nested document from ``{path: leaf}``.

    ``str`` segments build dicts and ``int`` segments build lists. A list whose
    indexes have holes (for example ``0`` and ``2`` but no ``1``) is compacted:
    items keep their relative order and take consecutive positions.

    Args:
        items: Leaf values keyed by path.
        warnings: Optional list that receives one message per compacted list.
        plan: When given, the keys of fields that have an ``alias`` (or a
            string ``validation_alias``) are renamed to it at every level of the
            document (sub-models and the chosen member of a union included), so
            the result is accepted by ``model_validate`` even without
            ``populate_by_name``.
        members: For a plan with unions of models, which member (by label) was
            chosen at each union path, so its fields are renamed with their own
            aliases.

    Returns:
        The document. An empty mapping gives ``{}``.

    Raises:
        ValueError: If paths conflict (a value and a container at the same
            place, dict keys mixed with list indexes, a negative index, an empty
            path) or the top level is a list.
        TypeError: If a path segment is neither ``str`` nor ``int``.
    """
    if not items:
        return {}
    collected: list[str] = [] if warnings is None else warnings
    entries = [(tuple(path), value) for path, value in items.items()]
    document = _build(entries, 0, collected)
    if not isinstance(document, dict):
        raise ValueError("the top level of a document must be an object")
    if plan is None:
        return document
    return _walk_level(document, plan.root, members or {}, prune=False)


def prune_nulls(
    doc: Mapping[str, Any],
    plan: "SchemaPlan",
    *,
    members: Mapping[Path, str] | None = None,
) -> dict[str, Any]:
    """Drop ``None`` values that pydantic can default, at every level.

    A ``None`` is removed when the field has a default (including the ``None``
    default of a ``NotRequired`` key), so validation supplies that default. A
    ``None`` for a field without a default is kept, which is what an
    ``Optional`` field without a default needs to validate; for a required
    non-``Optional`` field it stays so validation reports the gap. The same
    applies inside sub-models and the chosen member of a union of models. Keys
    the plan does not know are kept unchanged. Keys may be field names or their
    aliases; the result is keyed by the validation key (the alias when the field
    has one), so it can go straight to ``model_validate``.

    Args:
        doc: A document.
        plan: The compiled schema.
        members: For unions of models, the chosen member label at each union
            path (see ``unflatten``).

    Returns:
        A new dict without the prunable ``None`` values.
    """
    return _walk_level(doc, plan.root, members or {}, prune=True)
