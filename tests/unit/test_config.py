"""Tests for the gate and limit configuration dataclasses."""

import dataclasses
from typing import Any

import pytest

from trustcall_system_one import GateConfig, Limits


def test_gate_defaults() -> None:
    gates = GateConfig()
    assert gates.required_null_threshold == 0.7
    assert gates.optional_null_threshold == 0.3
    assert gates.min_confidence == 0.5
    assert gates.min_choice_prob == 0.55
    assert gates.min_choice_confidence == 0.5
    assert gates.noul_margin == 0.15
    assert gates.add_threshold == 0.55
    assert gates.replace_threshold == 0.6
    assert gates.remove_threshold == 0.85
    assert gates.allow_null_removals is False


def test_gates_are_frozen_hashable_and_comparable() -> None:
    gates = GateConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        gates.noul_margin = 0.1  # type: ignore[misc]
    assert hash(gates) == hash(GateConfig())
    assert gates == GateConfig()
    assert gates != GateConfig(noul_margin=0.2)
    assert len({gates, GateConfig(), GateConfig(add_threshold=0.9)}) == 2


@pytest.mark.parametrize(
    "name",
    [
        "required_null_threshold",
        "optional_null_threshold",
        "min_confidence",
        "min_choice_prob",
        "min_choice_confidence",
        "add_threshold",
        "replace_threshold",
        "remove_threshold",
    ],
)
@pytest.mark.parametrize(
    "bad", [0, -0.1, 1.01, float("nan"), float("inf"), True, "0.5"]
)
def test_gate_probabilities_must_be_in_the_half_open_unit_interval(
    name: str, bad: Any
) -> None:
    kwargs: dict[str, Any] = {name: bad}
    with pytest.raises(ValueError, match=name):
        GateConfig(**kwargs)


@pytest.mark.parametrize("name", ["min_confidence", "remove_threshold"])
def test_gate_probability_of_one_is_allowed(name: str) -> None:
    kwargs: dict[str, Any] = {name: 1}
    assert getattr(GateConfig(**kwargs), name) == 1


@pytest.mark.parametrize("bad", [-0.01, 0.5, 0.7, float("nan"), True, "0.1"])
def test_noul_margin_must_be_in_zero_to_half(bad: Any) -> None:
    with pytest.raises(ValueError, match="noul_margin"):
        GateConfig(noul_margin=bad)


def test_noul_margin_of_zero_is_allowed() -> None:
    assert GateConfig(noul_margin=0).noul_margin == 0


def test_allow_null_removals_must_be_a_bool() -> None:
    assert GateConfig(allow_null_removals=True).allow_null_removals is True
    with pytest.raises(ValueError, match="allow_null_removals"):
        GateConfig(allow_null_removals=1)  # type: ignore[arg-type]


def test_limit_defaults_and_hashing() -> None:
    limits = Limits()
    assert limits.max_depth == 8
    assert limits.any_policy == "skip"
    assert limits.estimate_for_bounded_numbers is False
    assert limits.max_instruction_chars == 600
    assert hash(limits) == hash(Limits())
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.max_depth = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_depth": 0}, "max_depth"),
        ({"max_depth": True}, "max_depth"),
        ({"max_depth": 1.5}, "max_depth"),
        ({"any_policy": "drop"}, "any_policy"),
        ({"estimate_for_bounded_numbers": 1}, "estimate_for_bounded_numbers"),
        ({"max_instruction_chars": 5}, "max_instruction_chars"),
        ({"max_instruction_chars": True}, "max_instruction_chars"),
    ],
)
def test_limits_are_validated(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Limits(**kwargs)


def test_limits_accept_valid_values() -> None:
    limits = Limits(max_depth=1, any_policy="text", estimate_for_bounded_numbers=True)
    assert (limits.max_depth, limits.any_policy) == (1, "text")
