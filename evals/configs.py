"""Extractor configurations: named presets and ``group.field=value`` overrides.

An override names a field of one of the library's config dataclasses
(``verify.attribution_keep=0.8``, ``limits.list_strategy=sentences``,
``gates.replace_threshold=0.7``) or a top-level ``create_extractor`` flag
(``enable_inserts=true``).
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from trustcall_system_one.combos import OPTION_FLAGS, OPTION_GROUPS, extractor_options

__all__ = [
    "GROUPS",
    "PRESETS",
    "REQUEST_CHANGING",
    "REQUEST_LIGHT",
    "build_options",
    "classify",
    "parse_overrides",
    "parse_value",
]

GROUPS: Mapping[str, type] = OPTION_GROUPS
_FLAGS = OPTION_FLAGS

PRESETS: Mapping[str, Mapping[str, Any]] = {
    "default": {},
    "list": {"limits.list_strategy": "list"},
    "sentences": {"limits.list_strategy": "sentences"},
    "auto": {"limits.list_strategy": "auto"},
    "no_verify": {
        "verify.attribution": False,
        "verify.input_only_crosscheck": False,
        "verify.reask_low_confidence": False,
        "verify.borderline_null": False,
    },
    "strict": {
        "verify.attribution_keep": 0.85,
        "verify.attribution_drop": 0.5,
        "gates.add_threshold": 0.7,
        "gates.replace_threshold": 0.75,
    },
    "lenient": {
        "verify.attribution_keep": 0.65,
        "verify.attribution_drop": 0.3,
        "gates.add_threshold": 0.45,
        "gates.replace_threshold": 0.5,
    },
}
"""Named configurations for ``--config``."""

REQUEST_LIGHT = (
    "verify.attribution_keep",
    "verify.attribution_drop",
    "verify.low_confidence",
    "verify.null_confirm_yes",
    "verify.null_confirm_no",
    "verify.adjudicate_min",
    "verify.clef_stated_min",
    "verify.sentence_keep",
    "verify.note_novelty",
    "gates.add_threshold",
    "gates.replace_threshold",
    "gates.remove_threshold",
    "gates.min_confidence",
    "gates.min_choice_prob",
    "gates.min_choice_confidence",
    "gates.noul_margin",
    "gates.allow_null_removals",
    "limits.list_auto_below",
    "match.accept",
    "match.min_confidence",
    "match.accept_compatible",
    "match.compatible_margin",
    "match.confirm_accept",
    "match.new_final",
    "match.delete_threshold",
    "coref.merge_threshold",
    "discovery.record_presence",
    "discovery.qualify_min",
)
"""Knobs that act on answers already received. A sweep over them replays the
first-pass requests from the cassette; only the follow-up requests that depend on
the decision (a re-ask in the middle band, the elements sent to attribution) are
new."""

REQUEST_CHANGING = (
    "limits.list_strategy",
    "limits.sentence_follow",
    "limits.max_sentences",
    "limits.max_instruction_chars",
    "gates.required_null_threshold",
    "gates.optional_null_threshold",
    "verify.instruction_template",
    "verify.window_chars",
    "verify.input_only_crosscheck",
    "verify.attribution",
    "verify.reask_low_confidence",
    "verify.isolate_decisions",
    "verify.relative_edit",
    "verify.borderline_null",
    "passes.split",
    "presence.threshold_absent",
    "presence.threshold_present",
    "discovery.presence_absent",
    "discovery.presence_present",
    "discovery.broaden_on_count_mismatch",
    "repair.enabled",
    "repair.max_attempts",
)
"""Knobs that change the request bodies or which requests are made at all: every
point of a sweep over them needs its requests live (or recorded)."""


def classify(param: str) -> str:
    """``"light"`` (replayable), ``"changes requests"`` or ``"unknown"``."""
    if param in REQUEST_LIGHT:
        return "light"
    if param in REQUEST_CHANGING:
        return "changes requests"
    return "unknown"


def parse_value(text: str) -> Any:
    """Read ``0.7``, ``12``, ``true``, ``none`` or a bare word from the command line."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    if ":" in text:
        try:
            return tuple(float(p) for p in text.split(":"))
        except ValueError:
            pass
    return text.strip()


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """Turn ``["verify.attribution_keep=0.8", ...]`` into a dict.

    Raises:
        ValueError: If an item is not ``key=value``.
    """
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected key=value, got {item!r}")
        out[key.strip()] = parse_value(value)
    return out


def build_options(
    overrides: Mapping[str, Any], base: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``create_extractor`` keyword arguments for a set of overrides.

    The same function as ``trustcall_system_one.combos.extractor_options``, which
    is what a combination's ``options`` go through.

    Args:
        overrides: ``group.field`` and top-level flag overrides.
        base: Keyword arguments the case brings (``enable_inserts`` ...); the
            overrides come on top.

    Returns:
        Keyword arguments for ``create_extractor``.

    Raises:
        ValueError: If a group or field is unknown or a value is invalid.
    """
    return extractor_options(overrides, base)
