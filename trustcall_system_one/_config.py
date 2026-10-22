"""Configuration dataclasses."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from trustcall_system_one._types import Usage
from trustcall_system_one._wire import CLAIRIC, CLEF, CLEF_FLASH, normalize_model_id

__all__ = [
    "ConcurrencyConfig",
    "CostReport",
    "GateConfig",
    "Limits",
    "PassesConfig",
    "PresenceConfig",
    "Prices",
    "VerifyConfig",
]

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
        max_depth: Longest leaf path that is expanded; a sub-model whose leaves
            would be deeper is skipped with a warning.
        max_recursion: How many times a model may reappear on its own ancestor
            chain (a ``Node`` with ``parent: Optional[Node]`` unrolls this many
            levels below the root); beyond that the field is skipped with one
            warning. ``0`` forbids any reappearance.
        max_groups: Most field groups (the root, sub-models and every union
            member, nested ones included) a schema may compile to. Exceeding it
            raises ``ValueError`` at compile time instead of truncating.
        any_policy: What to do with ``Any`` and untyped mappings: ``"skip"``
            them (pydantic supplies the default) or ask for ``"text"``.
        estimate_for_bounded_numbers: Emit a Clairic ``estimate`` field for a
            number with both a lower and an upper bound.
        max_instruction_chars: Longest instruction string sent to a model.
        clef_state_token_budget: Estimated tokens above which a Clef state is
            narrowed to mention windows. Clef silently truncates at 65,536
            tokens, so the budget leaves headroom for the estimate's error.
    """

    max_depth: int = 8
    max_recursion: int = 2
    max_groups: int = 128
    any_policy: Literal["skip", "text"] = "skip"
    estimate_for_bounded_numbers: bool = False
    max_instruction_chars: int = 600
    clef_state_token_budget: int = 56_000

    def __post_init__(self) -> None:
        """Validate the limits."""
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
            raise ValueError("max_depth must be an int")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        for name, floor in (("max_recursion", 0), ("max_groups", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int")
            if value < floor:
                raise ValueError(f"{name} must be at least {floor}")
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
        if isinstance(self.clef_state_token_budget, bool) or not isinstance(
            self.clef_state_token_budget, int
        ):
            raise ValueError("clef_state_token_budget must be an int")
        if self.clef_state_token_budget < 1_000:
            raise ValueError("clef_state_token_budget must be at least 1000")


def _check_unit(name: str, value: float, *, allow_zero: bool = False) -> None:
    """Require a real number in ``(0, 1]`` (or ``[0, 1]`` with ``allow_zero``)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    low_ok = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not low_ok or value > 1:
        bounds = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {bounds}, got {value!r}")


def _check_flag(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")


@dataclass(frozen=True)
class VerifyConfig:
    """Switches and thresholds of the verification ladder.

    The instance is immutable and hashable. Probabilities must lie in
    ``(0, 1]``.

    Attributes:
        attribution: Check with Clef that each value is stated about the entity.
        attribution_keep: Attribution probability at or above which a value is
            kept.
        attribution_drop: Probability below which a value is dropped; between
            the two the value is re-asked in isolation and checked once more.
        reask_low_confidence: Re-ask answers whose confidence is low.
        low_confidence: Confidence below which an answer is re-asked.
        borderline_null: Double-check answers whose ``p_null`` is in the band.
        null_band: Inclusive ``(low, high)`` range of ``p_null`` that counts as
            borderline.
        null_confirm_yes: Probability that the input states the value at or
            above which a null answer is re-asked.
        null_confirm_no: Probability below which a non-null answer is nulled.
        window_chars: Characters of context kept around each mention when the
            state is narrowed.
        instruction_template: Template for how a field is asked about an
            entity; it may use ``{label}`` and ``{desc}``.
        adjudicate_min: Probability a Clef adjudication needs for one of two
            disagreeing answers to win.
        self_consistency_k: Number of state variants to ask; only 1 is
            supported.
    """

    attribution: bool = True
    attribution_keep: float = 0.6
    attribution_drop: float = 0.3
    reask_low_confidence: bool = True
    low_confidence: float = 0.85
    borderline_null: bool = True
    null_band: tuple[float, float] = (0.25, 0.75)
    null_confirm_yes: float = 0.6
    null_confirm_no: float = 0.4
    window_chars: int = 600
    instruction_template: str = "{label}'s {desc}"
    adjudicate_min: float = 0.6
    self_consistency_k: int = 1

    def __post_init__(self) -> None:
        """Validate every setting."""
        for name in ("attribution", "reask_low_confidence", "borderline_null"):
            _check_flag(name, getattr(self, name))
        for name in (
            "attribution_keep",
            "attribution_drop",
            "low_confidence",
            "null_confirm_yes",
            "null_confirm_no",
            "adjudicate_min",
        ):
            _check_unit(name, getattr(self, name))
        if self.attribution_drop > self.attribution_keep:
            raise ValueError("attribution_drop must not exceed attribution_keep")
        if self.null_confirm_no > self.null_confirm_yes:
            raise ValueError("null_confirm_no must not exceed null_confirm_yes")
        band = self.null_band
        if not isinstance(band, (tuple, list)) or len(band) != 2:
            raise ValueError(f"null_band must be a (low, high) pair, got {band!r}")
        _check_unit("null_band[0]", band[0], allow_zero=True)
        _check_unit("null_band[1]", band[1], allow_zero=True)
        if band[0] > band[1]:
            raise ValueError(f"null_band must have low <= high, got {band!r}")
        object.__setattr__(self, "null_band", (float(band[0]), float(band[1])))
        if isinstance(self.window_chars, bool) or not isinstance(
            self.window_chars, int
        ):
            raise ValueError("window_chars must be an int")
        if self.window_chars < 1:
            raise ValueError("window_chars must be at least 1")
        if not isinstance(self.instruction_template, str):
            raise ValueError("instruction_template must be a string")
        try:
            self.instruction_template.format(label="the Thing", desc="a value")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"instruction_template may only use {{label}} and {{desc}}: {exc!r}"
            ) from exc
        if isinstance(self.self_consistency_k, bool) or self.self_consistency_k != 1:
            raise ValueError(
                "self_consistency_k other than 1 is not supported yet, "
                f"got {self.self_consistency_k!r}"
            )


@dataclass(frozen=True)
class PresenceConfig:
    """Thresholds of the presence gate.

    The instance is immutable and hashable.

    Attributes:
        threshold_absent: Presence probability below which the input is taken
            to contain no entity and nothing is extracted.
        threshold_present: Probability at or above which the entity is surely
            there. Between the two a ``One`` schema still extracts, since it
            has no discovery list to fall back on.
    """

    threshold_absent: float = 0.2
    threshold_present: float = 0.5

    def __post_init__(self) -> None:
        """Validate the thresholds."""
        _check_unit("threshold_absent", self.threshold_absent, allow_zero=True)
        _check_unit("threshold_present", self.threshold_present)
        if self.threshold_absent > self.threshold_present:
            raise ValueError("threshold_absent must not exceed threshold_present")


@dataclass(frozen=True)
class PassesConfig:
    """How Clairic fields are grouped into requests.

    The instance is immutable and hashable.

    Attributes:
        split: ``"entity"`` asks every field of an entity in one request,
            ``"submodel"`` one request per sub-model (the same as ``"entity"``
            for a flat schema) and ``"field"`` one request per field.
    """

    split: Literal["entity", "submodel", "field"] = "submodel"

    def __post_init__(self) -> None:
        """Validate the split."""
        if self.split not in ("entity", "submodel", "field"):
            raise ValueError(
                f"split must be 'entity', 'submodel' or 'field', got {self.split!r}"
            )
