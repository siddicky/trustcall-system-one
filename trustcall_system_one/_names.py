"""Stable wire names for field paths.

Clairic field names and Clef question ids have different character sets and
length limits. This module turns a leaf path such as ``("author", "name")``
into a name both models accept, deterministically: the same path always gets
the same name, and names that are too long are shortened with a keyed digest
rather than Python's per-process ``hash()``.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

__all__ = [
    "CLAIRIC_FIELD_RE",
    "CLEF_ID_RE",
    "FieldNameMap",
    "FieldNameMapBuilder",
    "Path",
    "pretty",
    "wire_name",
]

CLAIRIC_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
"""Valid Clairic field names."""

CLEF_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")
"""Valid Clef question ids."""

Path = tuple[str | int, ...]
"""A leaf's position in a document: dict keys are ``str``, list indices ``int``."""

Target = Literal["clairic", "clef"]

_LIMITS: dict[str, int] = {"clairic": 64, "clef": 100}
_BAD_CHARS = re.compile(r"[^A-Za-z0-9_\-]")
_UNDERSCORE_RUNS = re.compile(r"_+")
_CLAIRIC_FIRST = re.compile(r"[A-Za-z_]")
_DIGEST_SIZES = (3, 4, 6, 8)
_HEAD_CHARS = 12
_TAIL_CHARS = 24


def pretty(path: Path) -> str:
    """Render a path for people: segments joined with dots.

    Args:
        path: A leaf path.

    Returns:
        For example ``"author.name"`` or ``"tags.0"``.
    """
    return ".".join(str(segment) for segment in path)


def _sanitise(segment: str | int) -> str:
    """Make one path segment safe for both namespaces (never empty)."""
    text = _BAD_CHARS.sub("_", str(segment))
    text = _UNDERSCORE_RUNS.sub("_", text).strip("_")
    return text or "_"


def _digest(path: Path, size: int) -> str:
    key = "\x1f".join(str(segment) for segment in path).encode()
    return hashlib.blake2s(key, digest_size=size).hexdigest()


def _plain_name(segments: list[str], target: Target) -> str:
    name = ".".join(segments)
    if target == "clairic" and not _CLAIRIC_FIRST.match(name):
        name = f"f_{name}"
    return name


def _unshortened(path: Path, target: Target, suffix: str | None) -> str:
    segments = [_sanitise(segment) for segment in path]
    return _plain_name(segments + ([suffix] if suffix else []), target)


def _shorten(
    segments: list[str],
    digest_path: Path,
    target: Target,
    taken: set[str],
) -> str:
    """Build ``head_tail_digest``, widening the digest until it is free."""
    limit = _LIMITS[target]
    stem = f"{segments[0][:_HEAD_CHARS]}_{segments[-1][:_TAIL_CHARS]}"
    if target == "clairic" and not _CLAIRIC_FIRST.match(stem):
        stem = f"f_{stem}"
    for size in _DIGEST_SIZES:
        digest = _digest(digest_path, size)
        name = f"{stem[: limit - len(digest) - 1]}_{digest}"
        if name not in taken:
            return name
    raise ValueError(
        f"cannot find a free {target} wire name for {pretty(digest_path)!r}"
    )


def wire_name(
    path: Path,
    *,
    target: Target,
    taken: set[str],
    suffix: str | None = None,
) -> str:
    """Return the wire name for a path.

    Each segment is sanitised (characters outside ``[A-Za-z0-9_-]`` become
    ``_``, runs of ``_`` collapse, leading and trailing ``_`` are stripped, an
    empty result becomes ``_``) and the segments are joined with ``.``. Clairic
    names that would not start with a letter or ``_`` get an ``f_`` prefix.

    If the result fits the target's length limit and is not in ``taken`` it is
    used as is. Otherwise the name is shortened to
    ``<head>_<tail>_<digest>``: ``head`` is the first sanitised segment cut to
    12 characters, ``tail`` the last cut to 24, and ``digest`` a BLAKE2s hash of
    the unsanitised path (6 hex characters, widened to 8, 12 and 16 when the
    short form is taken too). The function does not modify ``taken``.

    Args:
        path: The leaf path. It must not be empty.
        target: ``"clairic"`` (64 characters) or ``"clef"`` (100 characters).
        taken: Names already in use in this namespace.
        suffix: An optional final component such as ``"__type"``, appended to
            the name after a dot and mixed into the digest. It is used verbatim
            and must already be valid in the target namespace.

    Returns:
        A name that matches ``CLAIRIC_FIELD_RE`` or ``CLEF_ID_RE``.

    Raises:
        ValueError: If the path is empty, the target is unknown, or no free name
            exists even with the widest digest.
    """
    if not path:
        raise ValueError("a wire name needs a non-empty path")
    if target not in _LIMITS:
        raise ValueError(f"unknown wire target {target!r}")
    plain = _unshortened(path, target, suffix)
    if len(plain) <= _LIMITS[target] and plain not in taken:
        return plain
    segments = [_sanitise(segment) for segment in path]
    digest_path = path if suffix is None else (*path, suffix)
    return _shorten(segments, digest_path, target, taken)


class FieldNameMap:
    """An immutable two-way map between paths and wire names for one namespace.

    Build one with ``FieldNameMapBuilder``.
    """

    __slots__ = ("_name_to_path", "_path_to_name", "_shortened", "_target")

    def __init__(
        self,
        target: Target,
        entries: Iterable[tuple[Path, str, bool]] = (),
    ) -> None:
        """Create a map from ``(path, name, shortened)`` triples."""
        path_to_name: dict[Path, str] = {}
        name_to_path: dict[str, Path] = {}
        shortened: set[str] = set()
        for path, name, was_shortened in entries:
            if path in path_to_name or name in name_to_path:
                raise ValueError(f"duplicate wire name entry for {pretty(path)!r}")
            path_to_name[path] = name
            name_to_path[name] = path
            if was_shortened:
                shortened.add(name)
        self._target = target
        self._path_to_name = MappingProxyType(path_to_name)
        self._name_to_path = MappingProxyType(name_to_path)
        self._shortened = frozenset(shortened)

    @property
    def target(self) -> Target:
        """The namespace these names belong to."""
        return self._target

    @property
    def path_to_name(self) -> Mapping[Path, str]:
        """Read-only map from path to wire name."""
        return self._path_to_name

    @property
    def name_to_path(self) -> Mapping[str, Path]:
        """Read-only map from wire name back to path."""
        return self._name_to_path

    @property
    def shortened(self) -> frozenset[str]:
        """Names that were shortened or de-duplicated with a digest."""
        return self._shortened

    def pretty(self, name: str) -> str:
        """Return the dotted path a wire name stands for.

        Raises:
            KeyError: If the name is unknown.
        """
        return pretty(self._name_to_path[name])

    def __len__(self) -> int:
        """Return the number of names."""
        return len(self._path_to_name)

    def __contains__(self, item: object) -> bool:
        """Return whether ``item`` is a known path or wire name."""
        return item in self._path_to_name or item in self._name_to_path

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"FieldNameMap({self._target!r}, {dict(self._path_to_name)!r})"


class FieldNameMapBuilder:
    """Allocates wire names for one namespace, in the order paths are added."""

    def __init__(self, target: Target) -> None:
        """Start an empty namespace.

        Args:
            target: ``"clairic"`` or ``"clef"``.
        """
        if target not in _LIMITS:
            raise ValueError(f"unknown wire target {target!r}")
        self._target: Target = target
        self._taken: set[str] = set()
        self._entries: list[tuple[Path, str, bool]] = []
        self._paths: set[Path] = set()

    @property
    def taken(self) -> frozenset[str]:
        """Names allocated so far."""
        return frozenset(self._taken)

    @property
    def shortened(self) -> frozenset[str]:
        """Allocated names that were shortened or de-duplicated with a digest."""
        return frozenset(name for _, name, short in self._entries if short)

    def add(self, path: Path, *, suffix: str | None = None) -> str:
        """Allocate a name for one path.

        Prefer ``add_all`` when the whole set of paths is known: ``add`` can only
        see the names allocated before it, so the outcome depends on call order.

        Args:
            path: The leaf path.
            suffix: Optional extra component; see ``wire_name``. The entry is
                recorded under ``(*path, suffix)``.

        Returns:
            The allocated name.

        Raises:
            ValueError: If the path was already added or no name is free.
        """
        return self.add_all([(path, suffix)])[0]

    def add_all(self, entries: Sequence[tuple[Path, str | None]]) -> list[str]:
        """Allocate names for a batch of paths, independent of their order.

        Paths whose plain names coincide (for example ``"a b"`` and ``"a_b"``)
        or that are too long all get the digest form; none of them keeps the
        plain name. Digest names are allocated after the plain ones, so a digest
        form can never take a name a plain path needs.

        Args:
            entries: ``(path, suffix)`` pairs; see ``wire_name`` for ``suffix``.

        Returns:
            The names, in the order of ``entries``.

        Raises:
            ValueError: If a path repeats (in the batch or from an earlier call)
                or no free name exists.
        """
        limit = _LIMITS[self._target]
        keys: list[Path] = []
        plains: list[str] = []
        for path, suffix in entries:
            if not path:
                raise ValueError("a wire name needs a non-empty path")
            key: Path = path if suffix is None else (*path, suffix)
            if key in self._paths or key in keys:
                raise ValueError(f"path {pretty(key)!r} already has a wire name")
            keys.append(key)
            plains.append(_unshortened(path, self._target, suffix))
        counts = Counter(plains)
        names: list[str | None] = [None] * len(entries)
        for index, plain in enumerate(plains):
            if counts[plain] == 1 and len(plain) <= limit and plain not in self._taken:
                names[index] = plain
                self._taken.add(plain)
        for index, (path, suffix) in enumerate(entries):
            if names[index] is not None:
                continue
            segments = [_sanitise(segment) for segment in path]
            digest_path = path if suffix is None else (*path, suffix)
            name = _shorten(segments, digest_path, self._target, self._taken)
            names[index] = name
            self._taken.add(name)
        result: list[str] = []
        for key, plain, assigned in zip(keys, plains, names, strict=True):
            assert assigned is not None
            self._paths.add(key)
            self._entries.append((key, assigned, assigned != plain))
            result.append(assigned)
        return result

    def build(self) -> FieldNameMap:
        """Freeze the allocated names into an immutable map."""
        return FieldNameMap(self._target, self._entries)
