"""Tests for the gate and limit configuration dataclasses."""

import dataclasses
from typing import Any

import pytest
from pydantic import BaseModel

from trustcall_system_one import (
    GateConfig,
    Limits,
    PassesConfig,
    PresenceConfig,
    RepairConfig,
    VerifyConfig,
)


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
    assert limits.max_recursion == 2
    assert limits.max_groups == 128
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
        ({"max_recursion": -1}, "max_recursion"),
        ({"max_recursion": True}, "max_recursion"),
        ({"max_recursion": 1.5}, "max_recursion"),
        ({"max_groups": 0}, "max_groups"),
        ({"max_groups": True}, "max_groups"),
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
    assert Limits(max_recursion=0, max_groups=1).max_recursion == 0


# --------------------------------------------------------------------------- #
# VerifyConfig, PresenceConfig, PassesConfig, Limits.clef_state_token_budget
# --------------------------------------------------------------------------- #


def test_verify_defaults() -> None:
    verify = VerifyConfig()
    assert verify.attribution is True
    assert (verify.attribution_keep, verify.attribution_drop) == (0.6, 0.3)
    assert verify.reask_low_confidence is True
    assert verify.low_confidence == 0.85
    assert verify.borderline_null is True
    assert verify.null_band == (0.25, 0.75)
    assert (verify.null_confirm_yes, verify.null_confirm_no) == (0.6, 0.4)
    assert verify.window_chars == 600
    assert verify.instruction_template == "{label}'s {desc}"
    assert verify.adjudicate_min == 0.6
    assert verify.self_consistency_k == 1


def test_verify_is_frozen_hashable_and_comparable() -> None:
    verify = VerifyConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        verify.attribution = False  # type: ignore[misc]
    assert hash(verify) == hash(VerifyConfig())
    assert verify == VerifyConfig()
    assert verify != VerifyConfig(window_chars=10)
    assert VerifyConfig(null_band=[0.1, 0.9]) == VerifyConfig(null_band=(0.1, 0.9))  # type: ignore[arg-type]
    assert len({verify, VerifyConfig(), VerifyConfig(low_confidence=0.9)}) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attribution_keep": 0},
        {"attribution_keep": 1.5},
        {"attribution_keep": float("nan")},
        {"attribution_keep": True},
        {"attribution_drop": 0.7},  # above keep
        {"low_confidence": 0},
        {"null_confirm_yes": 0.3},  # below null_confirm_no
        {"adjudicate_min": 2},
        {"null_band": (0.8, 0.2)},
        {"null_band": (0.1,)},
        {"null_band": (-0.1, 0.5)},
        {"window_chars": 0},
        {"window_chars": 1.5},
        {"instruction_template": "{nope}"},
        {"instruction_template": "{}"},
        {"instruction_template": "{label"},
        {"instruction_template": 3},
        {"attribution": 1},
        {"borderline_null": "yes"},
        {"self_consistency_k": 3},
        {"self_consistency_k": 0},
        {"self_consistency_k": True},
    ],
)
def test_verify_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        VerifyConfig(**kwargs)


def test_verify_accepts_boundary_values() -> None:
    VerifyConfig(attribution_keep=1.0, attribution_drop=1.0, null_band=(0.0, 1.0))
    VerifyConfig(instruction_template="{desc}")
    VerifyConfig(instruction_template="{label}: {desc} ({label})")


def test_the_self_consistency_error_says_what_is_supported() -> None:
    with pytest.raises(ValueError, match="not supported"):
        VerifyConfig(self_consistency_k=2)


def test_presence_config() -> None:
    presence = PresenceConfig()
    assert (presence.threshold_absent, presence.threshold_present) == (0.2, 0.5)
    assert hash(presence) == hash(PresenceConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        presence.threshold_absent = 0.1  # type: ignore[misc]
    PresenceConfig(threshold_absent=0.0, threshold_present=0.3)
    PresenceConfig(threshold_absent=0.5, threshold_present=0.5)
    for kwargs in (
        {"threshold_absent": 0.6},
        {"threshold_absent": -0.1},
        {"threshold_present": 0},
        {"threshold_present": 1.1},
        {"threshold_absent": float("nan")},
    ):
        with pytest.raises(ValueError):
            PresenceConfig(**kwargs)


def test_passes_config() -> None:
    assert PassesConfig().split == "submodel"
    for split in ("entity", "submodel", "field"):
        assert PassesConfig(split=split).split == split
    assert hash(PassesConfig()) == hash(PassesConfig())
    with pytest.raises(ValueError, match="split"):
        PassesConfig(split="everything")  # type: ignore[arg-type]


def test_limits_clef_state_token_budget() -> None:
    assert Limits().clef_state_token_budget == 56_000
    assert Limits(clef_state_token_budget=5000).clef_state_token_budget == 5000
    for bad in (999, 0, True, 5000.5):
        with pytest.raises(ValueError, match="clef_state_token_budget"):
            Limits(clef_state_token_budget=bad)  # type: ignore[arg-type]


def test_the_input_only_crosscheck_is_on_by_default_and_must_be_a_bool() -> None:
    assert VerifyConfig().input_only_crosscheck is True
    assert VerifyConfig(input_only_crosscheck=False).input_only_crosscheck is False
    with pytest.raises(ValueError, match="input_only_crosscheck"):
        VerifyConfig(input_only_crosscheck="yes")  # type: ignore[arg-type]


def test_clef_stated_min_defaults_to_0_7_and_is_a_probability() -> None:
    assert VerifyConfig().clef_stated_min == 0.7
    with pytest.raises(ValueError, match="clef_stated_min"):
        VerifyConfig(clef_stated_min=0)
    with pytest.raises(ValueError, match="clef_stated_min"):
        VerifyConfig(clef_stated_min=1.5)


# --------------------------------------------------------------------------- #
# RepairConfig
# --------------------------------------------------------------------------- #


def test_repair_defaults() -> None:
    repair = RepairConfig()
    assert repair.enabled is True
    assert repair.max_attempts == 5
    assert repair.stop_on_repeat is True
    assert repair.allow_pattern_reask is True


def test_repair_is_frozen_hashable_and_comparable() -> None:
    repair = RepairConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        repair.enabled = False  # type: ignore[misc]
    assert hash(repair) == hash(RepairConfig())
    assert repair == RepairConfig()
    assert repair != RepairConfig(max_attempts=2)
    assert len({repair, RepairConfig(), RepairConfig(enabled=False)}) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": -1},
        {"max_attempts": 2.5},
        {"max_attempts": True},
        {"max_attempts": "3"},
        {"enabled": 1},
        {"stop_on_repeat": "yes"},
        {"allow_pattern_reask": None},
    ],
)
def test_repair_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RepairConfig(**kwargs)


def test_repair_is_wired_into_the_extractor() -> None:
    from trustcall_system_one import create_extractor

    class Thing(BaseModel):
        name: str

    assert create_extractor([Thing]).repair == RepairConfig()
    off = RepairConfig(enabled=False)
    extractor = create_extractor([Thing], repair=off)
    assert extractor.repair is off
    assert extractor.with_options(repair=RepairConfig(max_attempts=2)).repair == (
        RepairConfig(max_attempts=2)
    )
