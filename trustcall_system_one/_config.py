"""Configuration dataclasses."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from trustcall_system_one._types import Usage
from trustcall_system_one._wire import CLAIRIC, CLEF, CLEF_FLASH, normalize_model_id

__all__ = ["ConcurrencyConfig", "CostReport", "GateConfig", "Limits", "Prices"]

V = TypeVar("V")


class _FrozenMap(Mapping[str, V]):
    """An immutable, hashable mapping with model ids normalised to full ids."""

    def __init__(self, items: Mapping[str, V]) -> None:
        self._data: dict[str, V] = {}
        for key, value in items.items():
            full = normalize_model_id(key)
            if full in self._data:
                raise ValueError(f"model {full} is listed more than once")
            self._data[full] = value
        self._hash = hash(frozenset(self._data.items()))

    def __getitem__(self, key: str) -> V:
        return self._data[normalize_model_id(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"{dict(self._data)!r}"


def _default_per_model() -> Mapping[str, int]:
    return {CLAIRIC: 32, CLEF: 32, CLEF_FLASH: 32}


def _default_prices() -> Mapping[str, float]:
    # Only Clairic's price is published.
    return {CLAIRIC: 0.10}


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Limits on in-flight requests, applied per event loop.

    Model keys may be bare names (``"clef"``) or full ids; they are stored as
    full ids. The instance is immutable and hashable.

    Attributes:
        max_in_flight: Cap on concurrent requests across all models.
        per_model: Starting (and maximum) concurrency per model id.
        default_per_model: Limit for models missing from ``per_model``.
        adaptive: Halve a model's limit on 429 and restore it on success.
        recover_after: Consecutive successes needed to raise a limit by one.
    """

    max_in_flight: int = 64
    per_model: Mapping[str, int] = field(default_factory=_default_per_model)
    default_per_model: int = 32
    adaptive: bool = True
    recover_after: int = 20

    def __post_init__(self) -> None:
        """Validate the limits and freeze the per-model mapping."""
        if self.max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if self.default_per_model < 1:
            raise ValueError("default_per_model must be at least 1")
        if self.recover_after < 1:
            raise ValueError("recover_after must be at least 1")
        if any(limit < 1 for limit in self.per_model.values()):
            raise ValueError("per-model limits must be at least 1")
        object.__setattr__(self, "per_model", _FrozenMap(self.per_model))

    def limit_for(self, model: str) -> int:
        """Return the configured concurrency cap for a model id."""
        return self.per_model.get(model, self.default_per_model)


@dataclass(frozen=True)
class CostReport:
    """The priced part of a usage total.

    Attributes:
        usd: Cost of the models that have a configured price.
        unpriced_models: Models that used input tokens but have no price.
        complete: True when every model with input tokens was priced.
    """

    usd: float
    unpriced_models: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class Prices:
    """Input-token prices in USD per million tokens, keyed by model id.

    Keys may be bare names or full ids; they are stored as full ids. The
    instance is immutable and hashable.
    """

    per_million_input: Mapping[str, float] = field(default_factory=_default_prices)

    def __post_init__(self) -> None:
        """Validate the prices and freeze the mapping."""
        if any(
            not math.isfinite(price) or price < 0
            for price in self.per_million_input.values()
        ):
            raise ValueError("prices must be finite and non-negative")
        object.__setattr__(
            self, "per_million_input", _FrozenMap(self.per_million_input)
        )

    def cost(self, usage: Usage) -> CostReport:
        """Price a usage total.

        Args:
            usage: Accumulated usage with a per-model breakdown.

        Returns:
            A report whose ``usd`` sums the priced models only. Models that
            consumed input tokens without a price are listed in
            ``unpriced_models`` and make ``complete`` false. Input tokens with
            no per-model breakdown are reported under ``"(unattributed)"``.
        """
        total = 0.0
        unpriced: set[str] = set()
        for model, used in usage.by_model.items():
            if used.input_tokens == 0:
                continue
            price = self.per_million_input.get(model)
            if price is None:
                unpriced.add(normalize_model_id(model))
            else:
                total += used.input_tokens * price / 1_000_000
        if not usage.by_model and usage.input_tokens:
            unpriced.add("(unattributed)")
        return CostReport(
            usd=total, unpriced_models=tuple(sorted(unpriced)), complete=not unpriced
        )


def _check_probability(name: str, value: float) -> None:
    """Require ``value`` to be a real number in ``(0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


@dataclass(frozen=True)
class GateConfig:
    """Thresholds that decide whether a model answer is accepted.

    The instance is immutable and hashable, so it can be part of a compile
    cache key. Every probability-like threshold must lie in ``(0, 1]``.

    Attributes:
        required_null_threshold: Clairic ``null_threshold`` for required fields.
        optional_null_threshold: Clairic ``null_threshold`` for optional fields
            and fields with defaults.
        min_confidence: Minimum Clairic answer confidence to accept a value.
        min_choice_prob: Minimum probability of the chosen Clef option.
        min_choice_confidence: Minimum Clef choice or score confidence.
        noul_margin: A noul answer must be at least this far from 0.5 to count
            as yes or no; closer answers abstain. Must be in ``[0, 0.5)``.
        add_threshold: Change-gate probability needed to add a value.
        replace_threshold: Change-gate probability needed to replace a value.
        remove_threshold: Change-gate probability needed to remove a value.
        allow_null_removals: Whether a null answer may remove an existing value.
    """

    required_null_threshold: float = 0.7
    optional_null_threshold: float = 0.3
    min_confidence: float = 0.5
    min_choice_prob: float = 0.55
    min_choice_confidence: float = 0.5
    noul_margin: float = 0.15
    add_threshold: float = 0.55
    replace_threshold: float = 0.6
    remove_threshold: float = 0.85
    allow_null_removals: bool = False

    def __post_init__(self) -> None:
        """Validate every threshold."""
        for name in (
            "required_null_threshold",
            "optional_null_threshold",
            "min_confidence",
            "min_choice_prob",
            "min_choice_confidence",
            "add_threshold",
            "replace_threshold",
            "remove_threshold",
        ):
            _check_probability(name, getattr(self, name))
        margin = self.noul_margin
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise ValueError(f"noul_margin must be a number, got {margin!r}")
        if not math.isfinite(margin) or not 0 <= margin < 0.5:
            raise ValueError(f"noul_margin must be in [0, 0.5), got {margin!r}")
        if not isinstance(self.allow_null_removals, bool):
            raise ValueError("allow_null_removals must be a bool")


@dataclass(frozen=True)
class Limits:
    """Structural limits applied while compiling a schema.

    The instance is immutable and hashable.

    Attributes:
        max_depth: Deepest nesting of sub-models that is expanded; deeper or
            recursive fields are skipped with a warning.
        any_policy: What to do with ``Any`` and untyped mappings: ``"skip"``
            them (pydantic supplies the default) or ask for ``"text"``.
        estimate_for_bounded_numbers: Emit a Clairic ``estimate`` field for a
            number with both a lower and an upper bound.
        max_instruction_chars: Longest instruction string sent to a model.
    """

    max_depth: int = 8
    any_policy: Literal["skip", "text"] = "skip"
    estimate_for_bounded_numbers: bool = False
    max_instruction_chars: int = 600

    def __post_init__(self) -> None:
        """Validate the limits."""
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
            raise ValueError("max_depth must be an int")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.any_policy not in ("skip", "text"):
            raise ValueError(
                f"any_policy must be 'skip' or 'text', got {self.any_policy!r}"
            )
        if not isinstance(self.estimate_for_bounded_numbers, bool):
            raise ValueError("estimate_for_bounded_numbers must be a bool")
        if isinstance(self.max_instruction_chars, bool) or not isinstance(
            self.max_instruction_chars, int
        ):
            raise ValueError("max_instruction_chars must be an int")
        if self.max_instruction_chars < 20:
            raise ValueError("max_instruction_chars must be at least 20")
