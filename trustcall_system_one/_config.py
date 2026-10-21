"""Configuration dataclasses."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from trustcall_system_one._types import Usage
from trustcall_system_one._wire import CLAIRIC, CLEF, CLEF_FLASH

__all__ = ["ConcurrencyConfig", "Prices"]


def _default_per_model() -> dict[str, int]:
    return {CLAIRIC: 32, CLEF: 32, CLEF_FLASH: 32}


def _default_prices() -> dict[str, float]:
    # Only Clairic's price is published.
    return {CLAIRIC: 0.10}


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Limits on in-flight requests.

    Attributes:
        max_in_flight: Cap on concurrent requests across all models.
        per_model: Starting (and maximum) concurrency per full model id.
        default_per_model: Limit for models missing from ``per_model``.
        adaptive: Halve a model's limit on 429 and restore it on success.
        recover_after: Consecutive successes needed to raise a limit by one.
    """

    max_in_flight: int = 64
    per_model: Mapping[str, int] = field(default_factory=_default_per_model)
    default_per_model: int = 32
    adaptive: bool = True
    recover_after: int = 20

    def limit_for(self, model: str) -> int:
        """Return the configured concurrency cap for a model id."""
        return self.per_model.get(model, self.default_per_model)


@dataclass(frozen=True)
class Prices:
    """Input-token prices in USD per million tokens, keyed by full model id."""

    per_million_input: Mapping[str, float] = field(default_factory=_default_prices)

    def cost(self, usage: Usage) -> float | None:
        """Price a usage total.

        Args:
            usage: Accumulated usage with a per-model breakdown.

        Returns:
            The cost in USD, or ``None`` if any model that consumed input tokens
            has no configured price (or the usage has no per-model breakdown).
        """
        if not usage.by_model:
            return 0.0 if usage.input_tokens == 0 else None
        total = 0.0
        for model, used in usage.by_model.items():
            if used.input_tokens == 0:
                continue
            price = self.per_million_input.get(model)
            if price is None:
                return None
            total += used.input_tokens * price / 1_000_000
        return total
