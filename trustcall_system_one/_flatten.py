"""Flatten nested documents to leaf paths and back, and prune absent values."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from trustcall_system_one._names import Path, pretty

if TYPE_CHECKING:
    from trustcall_system_one._compile import SchemaPlan

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


def _validation_keys(doc: Mapping[str, Any], plan: "SchemaPlan") -> dict[str, Any]:
    """Rename top-level keys to the keys ``model_validate`` accepts."""
    keys = {
        spec.path[0]: spec.alias
        for spec in plan.specs
        if len(spec.path) == 1 and spec.alias is not None
    }
    return {keys.get(key, key): value for key, value in doc.items()}


def unflatten(
    items: Mapping[Path, Any],
    warnings: list[str] | None = None,
    *,
    plan: "SchemaPlan | None" = None,
) -> dict[str, Any]:
    """Rebuild a nested document from ``{path: leaf}``.

    ``str`` segments build dicts and ``int`` segments build lists. A list whose
    indexes have holes (for example ``0`` and ``2`` but no ``1``) is compacted:
    items keep their relative order and take consecutive positions.

    Args:
        items: Leaf values keyed by path.
        warnings: Optional list that receives one message per compacted list.
        plan: When given, top-level keys of fields that have an ``alias`` (or a
            string ``validation_alias``) are renamed to it, so the result is
            accepted by ``model_validate`` even without ``populate_by_name``.

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
    return _validation_keys(document, plan) if plan is not None else document


def prune_nulls(doc: Mapping[str, Any], plan: "SchemaPlan") -> dict[str, Any]:
    """Drop top-level ``None`` values that pydantic can default.

    A ``None`` is removed when the field has a default (including the ``None``
    default of a ``NotRequired`` key), so validation supplies that default. A
    ``None`` for a field without a default is kept, which is what an
    ``Optional`` field without a default needs to validate; for a required
    non-``Optional`` field it stays so validation reports the gap. Keys the
    plan does not know are kept unchanged. Keys may be field names or their
    aliases; the result is keyed by the validation key (the alias when the field
    has one), so it can go straight to ``model_validate``.

    Args:
        doc: A top-level document.
        plan: The compiled schema.

    Returns:
        A new dict without the prunable ``None`` values.
    """
    defaulted: set[str] = set()
    for spec in plan.specs:
        if len(spec.path) == 1 and spec.has_default:
            defaulted.add(str(spec.path[0]))
            if spec.alias is not None:
                defaulted.add(spec.alias)
    pruned = {
        key: value
        for key, value in doc.items()
        if not (value is None and key in defaulted)
    }
    return _validation_keys(pruned, plan)
