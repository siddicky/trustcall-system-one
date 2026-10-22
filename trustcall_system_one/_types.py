"""Result and accounting types shared across the package."""

import dataclasses
from dataclasses import dataclass
from typing import Literal

from trustcall_system_one._wire import ClairicUsage, ClefUsage, normalize_model_id

__all__ = ["UNSTATED", "CallKind", "CallRecord", "Usage"]


class _Unstated:
    """The type of ``UNSTATED``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSTATED"

    def __copy__(self) -> "_Unstated":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "_Unstated":
        return self

    def __reduce__(self) -> str:
        return "UNSTATED"


UNSTATED = _Unstated()
"""Sentinel for a Clef choice that says the input does not state the value.

It is never a field value: for a nullable field it stands for ``None`` and for a
field with a default it means "leave the default".
"""

CallKind = Literal[
    "presence",
    "discover",
    "count",
    "coref",
    "qualify",
    "match",
    "delete",
    "extract",
    "crosscheck",
    "clef_fields",
    "attribution",
    "reask",
    "null_check",
    "adjudicate",
    "list_dedup",
    "gate",
    "repair",
    "raw",
]

_TOKEN_FIELDS = (
    "state_tokens",
    "image_tokens",
    "schema_tokens",
    "input_tokens",
    "output_tokens",
    "passes",
    "requests",
)


@dataclass
class Usage:
    """Token and request totals, with breakdowns by model and by call kind.

    The nested ``Usage`` values in ``by_model`` and ``by_kind`` are leaf totals
    and have empty breakdowns of their own.
    """

    state_tokens: int = 0
    image_tokens: int = 0
    schema_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    passes: int = 0
    requests: int = 0
    by_model: dict[str, "Usage"] = dataclasses.field(default_factory=dict)
    by_kind: dict[str, "Usage"] = dataclasses.field(default_factory=dict)

    def _add_totals(self, other: "Usage") -> None:
        for name in _TOKEN_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def add_call(self, model: str, kind: str, usage: ClairicUsage | ClefUsage) -> None:
        """Record one successful call.

        Args:
            model: Model id the call went to; bare names are stored as the full
                ``@cf/...`` id.
            kind: The ``CallMeta.kind`` of the call.
            usage: The ``usage`` object parsed from the response.
        """
        call = Usage(
            state_tokens=getattr(usage, "state_tokens", None) or 0,
            image_tokens=getattr(usage, "image_tokens", None) or 0,
            schema_tokens=getattr(usage, "schema_tokens", None) or 0,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            passes=getattr(usage, "passes", None) or 0,
            requests=1,
        )
        self._add_totals(call)
        self.by_model.setdefault(normalize_model_id(model), Usage())._add_totals(call)
        self.by_kind.setdefault(kind, Usage())._add_totals(call)

    def merge(self, other: "Usage") -> None:
        """Add ``other`` (totals and breakdowns) into this instance in place."""
        self._add_totals(other)
        for mine, theirs in (
            (self.by_model, other.by_model),
            (self.by_kind, other.by_kind),
        ):
            for key, value in theirs.items():
                mine.setdefault(key, Usage())._add_totals(value)

    def __add__(self, other: "Usage") -> "Usage":
        """Return the sum of two usages without mutating either."""
        if not isinstance(other, Usage):
            return NotImplemented
        total = Usage()
        total.merge(self)
        total.merge(other)
        return total


@dataclass(frozen=True)
class CallRecord:
    """One transport call, as logged on a result.

    ``error`` is the failure message for a call that raised (``None`` on
    success) and ``status_code`` the HTTP status when one was received.
    """

    model: str
    kind: CallKind
    schema: str = ""
    entity: str | None = None
    field: str | None = None
    scope: str = ""
    n_items: int = 0
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    duration_ms: float = 0.0
    retries: int = 0
    error: str | None = None
    status_code: int | None = None
