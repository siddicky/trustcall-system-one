"""Tests for usage accounting, call records and price calculation."""

import dataclasses

import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CLEF_FLASH,
    CallRecord,
    ConcurrencyConfig,
    Prices,
    Usage,
)
from trustcall_system_one._wire import ClairicUsage, ClefUsage

CLAIRIC_USAGE = ClairicUsage(
    state_tokens=61,
    image_tokens=5,
    schema_tokens=472,
    input_tokens=655,
    output_tokens=0,
    passes=3,
)


def test_add_call_updates_totals_and_breakdowns() -> None:
    usage = Usage()
    usage.add_call(CLAIRIC, "extract", CLAIRIC_USAGE)
    usage.add_call(CLEF, "gate", ClefUsage(input_tokens=200, output_tokens=1))
    usage.add_call(CLAIRIC, "reask", ClairicUsage(input_tokens=45, output_tokens=0))

    assert usage.requests == 3
    assert usage.input_tokens == 655 + 200 + 45
    assert usage.output_tokens == 1
    assert usage.state_tokens == 61
    assert usage.image_tokens == 5
    assert usage.schema_tokens == 472
    assert usage.passes == 3

    assert set(usage.by_model) == {CLAIRIC, CLEF}
    assert usage.by_model[CLAIRIC].requests == 2
    assert usage.by_model[CLAIRIC].input_tokens == 700
    assert usage.by_model[CLAIRIC].passes == 3
    assert usage.by_model[CLEF].requests == 1
    assert usage.by_model[CLEF].input_tokens == 200
    assert usage.by_model[CLEF].passes == 0

    assert set(usage.by_kind) == {"extract", "gate", "reask"}
    assert usage.by_kind["extract"].input_tokens == 655
    assert usage.by_kind["reask"].requests == 1
    # Breakdown entries are leaves.
    assert usage.by_model[CLAIRIC].by_model == {}
    assert usage.by_kind["gate"].by_kind == {}


def test_add_call_treats_missing_optionals_as_zero() -> None:
    usage = Usage()
    usage.add_call(CLAIRIC, "extract", ClairicUsage(input_tokens=10, output_tokens=2))
    assert (usage.state_tokens, usage.image_tokens, usage.schema_tokens) == (0, 0, 0)
    assert usage.passes == 0
    assert usage.requests == 1


def test_defaults_are_independent_per_instance() -> None:
    a, b = Usage(), Usage()
    a.add_call(CLAIRIC, "extract", CLAIRIC_USAGE)
    assert b.by_model == {}
    assert b.requests == 0


def test_merge_and_add() -> None:
    a, b = Usage(), Usage()
    a.add_call(CLAIRIC, "extract", CLAIRIC_USAGE)
    b.add_call(CLAIRIC, "gate", ClairicUsage(input_tokens=5, output_tokens=0))
    b.add_call(CLEF, "gate", ClefUsage(input_tokens=7, output_tokens=0))

    total = a + b
    assert total.requests == 3
    assert total.input_tokens == 667
    assert total.by_model[CLAIRIC].input_tokens == 660
    assert total.by_model[CLEF].input_tokens == 7
    assert total.by_kind["gate"].requests == 2
    # Operands untouched, result independent of them.
    assert a.requests == 1
    assert b.requests == 2
    total.by_model[CLAIRIC].input_tokens = 0
    assert a.by_model[CLAIRIC].input_tokens == 655

    a.merge(b)
    assert dataclasses.asdict(a) == dataclasses.asdict(total) | {
        "by_model": dataclasses.asdict(a)["by_model"]
    }
    assert a.requests == 3
    assert a.by_kind["gate"].requests == 2


def test_add_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        Usage() + 1  # type: ignore[operator]


def test_call_record_is_frozen_and_holds_fields() -> None:
    record = CallRecord(
        model=CLAIRIC,
        kind="extract",
        schema="Person",
        entity="e0",
        field="name",
        scope="s",
        n_items=3,
        usage={"input_tokens": 9},
        duration_ms=12.5,
        retries=2,
    )
    assert record.n_items == 3
    assert record.retries == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.retries = 0  # type: ignore[misc]


def test_prices_default_is_clairic_only() -> None:
    assert dict(Prices().per_million_input) == {CLAIRIC: 0.10}


def test_cost_is_numeric_for_clairic_only_usage() -> None:
    usage = Usage()
    usage.add_call(
        CLAIRIC, "extract", ClairicUsage(input_tokens=2_000_000, output_tokens=0)
    )
    assert Prices().cost(usage) == pytest.approx(0.20)


def test_cost_is_none_once_unpriced_clef_usage_is_present() -> None:
    usage = Usage()
    usage.add_call(CLAIRIC, "extract", CLAIRIC_USAGE)
    assert Prices().cost(usage) is not None
    usage.add_call(CLEF, "gate", ClefUsage(input_tokens=10, output_tokens=0))
    assert Prices().cost(usage) is None


def test_cost_with_configured_clef_price() -> None:
    usage = Usage()
    usage.add_call(
        CLAIRIC, "extract", ClairicUsage(input_tokens=1_000_000, output_tokens=0)
    )
    usage.add_call(CLEF, "gate", ClefUsage(input_tokens=500_000, output_tokens=0))
    prices = Prices({CLAIRIC: 0.10, CLEF: 0.20})
    assert prices.cost(usage) == pytest.approx(0.10 + 0.10)
    # A different Clef variant still needs its own price.
    usage.add_call(CLEF_FLASH, "gate", ClefUsage(input_tokens=1, output_tokens=0))
    assert prices.cost(usage) is None
    assert Prices({CLAIRIC: 0.1, CLEF: 0.2, CLEF_FLASH: 0.05}).cost(usage) is not None


def test_cost_ignores_unpriced_models_without_input_tokens() -> None:
    usage = Usage()
    usage.by_model[CLEF] = Usage(requests=1)  # no input tokens
    usage.add_call(CLAIRIC, "extract", CLAIRIC_USAGE)
    assert Prices().cost(usage) == pytest.approx(655 * 0.10 / 1_000_000)


def test_cost_of_empty_usage_is_zero() -> None:
    assert Prices().cost(Usage()) == 0.0


def test_cost_without_breakdown_cannot_be_priced() -> None:
    assert Prices().cost(Usage(input_tokens=5)) is None


def test_concurrency_defaults() -> None:
    config = ConcurrencyConfig()
    assert config.max_in_flight == 64
    assert config.default_per_model == 32
    assert dict(config.per_model) == {CLAIRIC: 32, CLEF: 32, CLEF_FLASH: 32}
    assert config.adaptive is True
    assert config.recover_after == 20
    assert config.limit_for("@cf/unknown") == 32
    assert ConcurrencyConfig(per_model={CLAIRIC: 4}).limit_for(CLAIRIC) == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.adaptive = False  # type: ignore[misc]
