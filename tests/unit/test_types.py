"""Tests for usage accounting, call records and price calculation."""

import dataclasses

import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CLEF_FLASH,
    CallRecord,
    ConcurrencyConfig,
    CostReport,
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
    assert a.requests == 3
    assert a.input_tokens == 667
    assert a.by_model[CLAIRIC].input_tokens == 660
    assert a.by_model[CLEF].input_tokens == 7
    assert a.by_kind["gate"].requests == 2
    assert a.by_kind["extract"].requests == 1
    # merge copies leaf values rather than aliasing b's.
    a.by_model[CLEF].input_tokens = 0
    assert b.by_model[CLEF].input_tokens == 7


def test_add_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        Usage() + 1  # type: ignore[operator]


def test_call_record_error_and_status_default_to_none() -> None:
    record = CallRecord(model=CLAIRIC, kind="extract")
    assert record.error is None
    assert record.status_code is None
    failed = CallRecord(
        model=CLEF, kind="gate", error="HTTP 503", status_code=503, retries=6
    )
    assert (failed.error, failed.status_code, failed.retries) == ("HTTP 503", 503, 6)


def test_call_record_kind_is_the_call_kind_literal() -> None:
    from typing import get_args, get_type_hints

    from trustcall_system_one import CallKind, CallMeta

    assert get_type_hints(CallRecord)["kind"] == CallKind
    assert get_type_hints(CallMeta)["kind"] == CallKind
    assert "extract" in get_args(CallKind)


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


def test_cost_is_complete_for_clairic_only_usage() -> None:
    usage = Usage()
    usage.add_call(
        CLAIRIC, "extract", ClairicUsage(input_tokens=2_000_000, output_tokens=0)
    )
    report = Prices().cost(usage)
    assert report.usd == pytest.approx(0.20)
    assert report.unpriced_models == ()
    assert report.complete is True


def test_cost_sums_priced_models_and_lists_unpriced_ones() -> None:
    usage = Usage()
    usage.add_call(
        CLAIRIC, "extract", ClairicUsage(input_tokens=1_000_000, output_tokens=0)
    )
    assert Prices().cost(usage).complete
    usage.add_call(CLEF, "gate", ClefUsage(input_tokens=10, output_tokens=0))
    report = Prices().cost(usage)
    assert report.usd == pytest.approx(0.10)  # priced models only
    assert report.unpriced_models == (CLEF,)
    assert report.complete is False


def test_cost_with_configured_clef_price() -> None:
    usage = Usage()
    usage.add_call(
        CLAIRIC, "extract", ClairicUsage(input_tokens=1_000_000, output_tokens=0)
    )
    usage.add_call(CLEF, "gate", ClefUsage(input_tokens=500_000, output_tokens=0))
    prices = Prices({CLAIRIC: 0.10, CLEF: 0.20})
    report = prices.cost(usage)
    assert report.usd == pytest.approx(0.10 + 0.10)
    assert report.complete
    # A different Clef variant still needs its own price.
    usage.add_call(CLEF_FLASH, "gate", ClefUsage(input_tokens=1, output_tokens=0))
    partial = prices.cost(usage)
    assert partial.unpriced_models == (CLEF_FLASH,)
    assert not partial.complete
    full = Prices({CLAIRIC: 0.1, CLEF: 0.2, CLEF_FLASH: 0.05}).cost(usage)
    assert full.complete
    assert full.usd == pytest.approx(0.1 + 0.1 + 0.05 / 1_000_000)


def test_bare_model_names_are_normalised_to_full_ids() -> None:
    usage = Usage()
    usage.add_call("clef", "gate", ClefUsage(input_tokens=1_000_000, output_tokens=0))
    usage.add_call(CLEF, "gate", ClefUsage(input_tokens=1_000_000, output_tokens=0))
    assert list(usage.by_model) == [CLEF]
    assert usage.by_model[CLEF].requests == 2
    # A bare price key matches a full-id usage and vice versa.
    prices = Prices({"clef": 0.5})
    assert dict(prices.per_million_input) == {CLEF: 0.5}
    report = prices.cost(usage)
    assert report.complete
    assert report.usd == pytest.approx(1.0)
    manual = Usage()
    manual.by_model["clef"] = Usage(input_tokens=1_000_000, requests=1)
    assert prices.cost(manual).usd == pytest.approx(0.5)
    assert Prices({CLEF: 0.5}).cost(manual).complete


def test_cost_ignores_unpriced_models_without_input_tokens() -> None:
    usage = Usage()
    usage.by_model[CLEF] = Usage(requests=1)  # no input tokens
    usage.add_call(CLAIRIC, "extract", CLAIRIC_USAGE)
    report = Prices().cost(usage)
    assert report.complete
    assert report.usd == pytest.approx(655 * 0.10 / 1_000_000)


def test_cost_of_empty_usage_is_zero_and_complete() -> None:
    assert Prices().cost(Usage()) == CostReport(0.0, (), True)


def test_cost_without_breakdown_is_unattributed() -> None:
    report = Prices().cost(Usage(input_tokens=5))
    assert report.usd == 0.0
    assert report.unpriced_models == ("(unattributed)",)
    assert not report.complete


def test_cost_report_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        CostReport(0.0, (), True).usd = 1.0  # type: ignore[misc]


def test_prices_validation() -> None:
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Prices({CLAIRIC: bad})
    assert Prices({CLAIRIC: 0.0}).cost(Usage()).complete


def test_prices_are_hashable_and_immutable() -> None:
    a, b = Prices({CLAIRIC: 0.1}), Prices({"clairic": 0.1})
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b, Prices({CLAIRIC: 0.2})}) == 2
    with pytest.raises(TypeError):
        a.per_million_input[CLAIRIC] = 1.0  # type: ignore[index]
    source = {CLAIRIC: 0.1}
    prices = Prices(source)
    source[CLAIRIC] = 9.0  # later changes to the caller's dict do not leak in
    assert prices.per_million_input[CLAIRIC] == 0.1


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


def test_concurrency_config_is_hashable_and_immutable() -> None:
    a = ConcurrencyConfig(per_model={"clef": 4})
    b = ConcurrencyConfig(per_model={CLEF: 4})
    assert a == b
    assert hash(a) == hash(b)
    assert a.per_model["clef"] == 4
    assert a.limit_for("clef") == 4
    assert a.limit_for(CLEF) == 4
    assert a.limit_for(CLAIRIC) == 32  # not listed: the default
    assert len({ConcurrencyConfig(), ConcurrencyConfig(), a}) == 2
    with pytest.raises(TypeError):
        a.per_model[CLEF] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_in_flight": 0},
        {"default_per_model": 0},
        {"recover_after": 0},
        {"per_model": {CLAIRIC: 0}},
        {"per_model": {CLAIRIC: 4, CLEF: -1}},
    ],
)
def test_concurrency_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ConcurrencyConfig(**kwargs)  # type: ignore[arg-type]


def test_concurrency_config_accepts_boundary_values() -> None:
    ConcurrencyConfig(
        max_in_flight=1, per_model={CLAIRIC: 1}, default_per_model=1, recover_after=1
    )


def test_models_that_normalise_to_the_same_id_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        ConcurrencyConfig(per_model={"clef": 4, CLEF: 8})
    with pytest.raises(ValueError, match="more than once"):
        Prices({"clairic": 0.1, CLAIRIC: 0.2})
