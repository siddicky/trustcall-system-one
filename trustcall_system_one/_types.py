"""Result and accounting types shared across the package."""

import dataclasses
from dataclasses import dataclass

from trustcall_system_one._wire import ClairicUsage, ClefUsage

__all__ = ["CallRecord", "Usage"]

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
            model: Full model id the call went to.
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
        self.by_model.setdefault(model, Usage())._add_totals(call)
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
    """One transport call, as logged on a result."""

    model: str
    kind: str
    schema: str = ""
    entity: str | None = None
    field: str | None = None
    scope: str = ""
    n_items: int = 0
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    duration_ms: float = 0.0
    retries: int = 0
