"""Configuration dataclasses."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from trustcall_system_one._types import Usage
from trustcall_system_one._wire import CLAIRIC, CLEF, CLEF_FLASH, normalize_model_id

__all__ = [
    "ConcurrencyConfig",
    "CorefConfig",
    "CostReport",
    "DiscoveryConfig",
    "GateConfig",
    "Limits",
    "MatchConfig",
    "PassesConfig",
    "PresenceConfig",
    "Prices",
    "RepairConfig",
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
        usd: Cost, estimated from input tokens, of the models that have a
            configured price.
        unpriced_models: Models that used input tokens but have neither a
            configured price nor a reported cost on every call.
        complete: True when every model with input tokens was priced, by a
            configured price or because all of its calls reported a cost.
        reported_usd: The costs that responses reported (Experiential Labs does),
            summed over the models all of whose calls reported one. It is a
            second, separate figure: where a model has both a configured price
            and reported costs they are two measures of the same spend, not two
            amounts to add.
    """

    usd: float
    unpriced_models: tuple[str, ...]
    complete: bool
    reported_usd: float = 0.0


@dataclass(frozen=True)
class Prices:
    """Input-token prices in USD per million tokens, keyed by model id.

    Keys may be bare names or full ids; they are stored as full ids. The
    instance is immutable and hashable. TypeSafe's Jev has no default price
    ($0.042 per million input tokens): give the version that answers
    (``"jev-1.14.0"``, shown in ``Usage.by_model``) or the alias ``"jev-latest"``,
    which prices any ``jev-*`` version that has no price of its own.
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
            A report whose ``usd`` sums the models with a configured price,
            estimated from input tokens. A model counts as priced too when every
            one of its calls reported a cost; those costs are summed in
            ``reported_usd``, apart from ``usd``. Models that consumed input
            tokens and are priced in neither way are listed in
            ``unpriced_models`` and make ``complete`` false. Input tokens with
            no per-model breakdown are reported under ``"(unattributed)"``.
        """
        total = 0.0
        reported = 0.0
        unpriced: set[str] = set()
        for model, used in usage.by_model.items():
            if used.input_tokens == 0:
                reported += used.reported_cost_usd  # spent on unusable responses
                continue
            all_reported = used.requests > 0 and used.reported_requests == used.requests
            if all_reported:
                reported += used.reported_cost_usd
            price = self.per_million_input.get(model)
            if price is None and model.startswith("jev-"):
                # a price for the alias covers every version the alias points to
                price = self.per_million_input.get("jev-latest")
            if price is not None:
                total += used.input_tokens * price / 1_000_000
            elif not all_reported:
                unpriced.add(normalize_model_id(model))
        if not usage.by_model and usage.input_tokens:
            unpriced.add("(unattributed)")
        return CostReport(
            usd=total,
            unpriced_models=tuple(sorted(unpriced)),
            complete=not unpriced,
            reported_usd=reported,
        )


def _check_probability(name: str, value: float) -> None:
    """Require ``value`` to be a real number in ``(0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


DEFAULT_PLACEHOLDERS: tuple[str, ...] = (
    "tba",
    "tbd",
    "tbc",
    "n/a",
    "n.a.",
    "n/d",
    "-",
    "--",
    "\u2013",
    "\u2014",
    "?",
    "to be advised",
    "to be announced",
    "to be confirmed",
    "to be determined",
    "not available",
    "not applicable",
)
"""Text values that say "no value yet", not a value (compared without case, markup or
a closing full stop). ``GateConfig(placeholders=())`` keeps them as values.
Deliberately short: ``none``, ``unknown`` and ``na`` can be real values (a status, a
country code)."""


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
        placeholders: Text values that are no value (``DEFAULT_PLACEHOLDERS``: TBA,
            TBD, N/A, a dash ...). A plain text field whose answer is one of these
            is null (``"placeholder"`` in the field's warnings), so ``"TBA"`` is
            neither kept as a serial number nor put in a record that is updated;
            matching ignores case, markup and a closing full stop. List elements and
            Clef-typed fields are not touched. Pass ``()`` to keep such text as a
            value.
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
    placeholders: tuple[str, ...] = DEFAULT_PLACEHOLDERS

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
        if not isinstance(self.placeholders, tuple) or not all(
            isinstance(p, str) and p.strip() for p in self.placeholders
        ):
            raise ValueError("placeholders must be a tuple of non-empty strings")


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
        clef_state_token_budget: Estimated tokens (state AND questions,
            ``_render.estimate_request_tokens``, calibrated on recorded
            table-heavy requests) above which a Clef request is split by
            questions or has its state narrowed to mention windows. Clef silently
            truncates at 65,536 tokens and refuses a request over 65,529 (HTTP
            422), so the budget leaves headroom for the estimate's error. A
            transport may advertise a smaller one (``state_token_budget``).
        list_strategy: How a ``list[str]`` field of free text is read unless the
            field says otherwise (``s1_field(list_strategy=...)``). ``"list"``
            asks Clairic's ``list`` type, which is a span picker: confident for
            defined items (names, contacts), hopeless for open-ended ones ("things
            worth remembering"). ``"sentences"`` splits the input into sentences
            in code and asks Clef one yes/no question per candidate sentence;
            the sentences it accepts are the elements, verbatim. ``"auto"`` asks
            Clairic first and falls back to sentences when its answer has a
            confidence below ``list_auto_below``.
        list_auto_below: With ``"auto"``, the list confidence below which the
            sentences are used (a non-empty answer that Clairic itself does not
            trust).
        sentence_follow: With sentences for an entity that is mentioned, how many
            sentences after one that mentions it still count as candidates.
        max_sentences: Most candidate sentences asked about per field; more are
            left out with a warning.
    """

    max_depth: int = 8
    max_recursion: int = 2
    max_groups: int = 128
    any_policy: Literal["skip", "text"] = "skip"
    estimate_for_bounded_numbers: bool = False
    max_instruction_chars: int = 600
    clef_state_token_budget: int = 56_000
    list_strategy: Literal["list", "sentences", "auto"] = "auto"
    list_auto_below: float = 0.5
    sentence_follow: int = 4
    max_sentences: int = 200

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
        if self.list_strategy not in ("list", "sentences", "auto"):
            raise ValueError(
                "list_strategy must be 'list', 'sentences' or 'auto', "
                f"got {self.list_strategy!r}"
            )
        _check_unit("list_auto_below", self.list_auto_below)
        for name, floor in (("sentence_follow", 0), ("max_sentences", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int")
            if value < floor:
                raise ValueError(f"{name} must be at least {floor}")


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
        input_only_crosscheck: Updates only: ask every Clairic field a second
            time with the input alone (no existing record) and reject a changed
            value the input alone does not give (``echo_crosscheck``); two
            different values go to a Clef adjudication.
        clef_stated_min: Updates only: probability a Clef ``noul`` "does the
            input state it?" needs before a Clef-typed answer (a ``bool``, a
            ``Literal`` or an ``Enum``) that would change or add to the record
            becomes an operation. Such answers always resolve, whether or not
            the input says anything about the field.
        sentence_keep: Probability at or above which a candidate sentence is kept
            as a list element when a ``list[str]`` field is read by sentences
            (see ``Limits.list_strategy``).
        note_novelty: Updates only: probability at or above which a sentence read
            as a list element (a note) counts as adding something the record does
            not already hold, once the record's other operations are applied
            (call kind ``novelty``). A sentence below it is dropped as
            ``not_novel``; 0 switches the check off.
        relative_edit: Updates only: probability at or above which a replacement of
            one number by another is refused as ``relative_edit_unsupported``
            because the input states a change relative to the current value ("two
            more", "one fewer", "double it") that the library does not compute
            (call kind ``relative_edit``); 0 switches the check off.
        isolate_decisions: Ask the decisive Clef questions (attribution, the novelty
            check, the record-presence check) one per request. Clef's answer to a
            question depends on the other questions of its request (live: the same
            question gave 0.5597 alone and 0.2800 beside another one), so a decision
            that is worth a request is not asked in a crowd.
        self_consistency_k: Number of state variants to ask; only 1 is
            supported.
    """

    attribution: bool = True
    attribution_keep: float = 0.75
    attribution_drop: float = 0.4
    reask_low_confidence: bool = True
    low_confidence: float = 0.85
    borderline_null: bool = True
    null_band: tuple[float, float] = (0.25, 0.75)
    null_confirm_yes: float = 0.6
    null_confirm_no: float = 0.4
    window_chars: int = 600
    instruction_template: str = "{label}'s {desc}"
    adjudicate_min: float = 0.6
    input_only_crosscheck: bool = True
    clef_stated_min: float = 0.7
    sentence_keep: float = 0.6
    note_novelty: float = 0.5
    relative_edit: float = 0.5
    isolate_decisions: bool = True
    self_consistency_k: int = 1

    def __post_init__(self) -> None:
        """Validate every setting."""
        for name in (
            "attribution",
            "reask_low_confidence",
            "borderline_null",
            "input_only_crosscheck",
            "isolate_decisions",
        ):
            _check_flag(name, getattr(self, name))
        _check_unit("note_novelty", self.note_novelty, allow_zero=True)
        _check_unit("relative_edit", self.relative_edit, allow_zero=True)
        for name in (
            "attribution_keep",
            "attribution_drop",
            "low_confidence",
            "null_confirm_yes",
            "null_confirm_no",
            "adjudicate_min",
            "clef_stated_min",
            "sentence_keep",
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


@dataclass(frozen=True)
class RepairConfig:
    """Settings of the validation repair loop.

    When the assembled record fails pydantic validation, each failing field is
    asked again, alone, with the error in hand (see ``docs/DESIGN.md``). All
    failing fields of one pass go out together and count as ONE attempt. The
    instance is immutable and hashable.

    Attributes:
        enabled: Run the loop. Off, a record that fails validation is dropped to
            what it is without any further request (updates drop the operations
            that fail, inserts become invalid).
        max_attempts: Most repair rounds per record. It is a guard against a
            loop that never ends, not a cost cap: ``ExtractionResult.attempts``
            counts the extraction round plus the repair rounds, so at most
            ``1 + max_attempts``.
        stop_on_repeat: Stop when a round leaves the same set of
            ``(path, error type)`` pairs as an earlier one: asking again would
            not change anything.
        allow_pattern_reask: Allow ONE re-ask for a field that failed its regex
            ``pattern`` (or another format check) when the input holds text that
            could satisfy it. Clairic only quotes the input, so it cannot
            reformat; without this switch such a field is never asked again.
    """

    enabled: bool = True
    max_attempts: int = 5
    stop_on_repeat: bool = True
    allow_pattern_reask: bool = True

    def __post_init__(self) -> None:
        """Validate every setting."""
        for name in ("enabled", "stop_on_repeat", "allow_pattern_reask"):
            _check_flag(name, getattr(self, name))
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise ValueError(f"max_attempts must be an int, got {self.max_attempts!r}")
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be at least 1, got {self.max_attempts}"
            )


def _check_int(
    name: str, value: object, floor: int, ceiling: int | None = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < floor:
        raise ValueError(f"{name} must be at least {floor}, got {value}")
    if ceiling is not None and value > ceiling:
        raise ValueError(f"{name} must be at most {ceiling}, got {value}")


@dataclass(frozen=True)
class DiscoveryConfig:
    """Settings of the discovery step of the collection engine (``Many``).

    The instance is immutable and hashable.

    Attributes:
        presence_absent: Presence probability below which the input is taken to
            mention no entity at all: nothing further is asked.
        presence_present: Probability at or above which the entities are surely
            there. Between the two the run goes on only if discovery found
            mentions.
        count_trust_max: The count of entities is trusted up to this many (1 to
            30); a mismatch between the count and the entities found only warns
            when either number is larger (counting is a weak spot of the model).
            It bounds the work of a count-driven merge: at most ``k * (k - 1) / 2``
            single-question merge checks for ``k`` clusters, so 435 at 30.
        broaden_on_count_mismatch: Act on a mismatch: when the count exceeds the
            entities found, discover again with a broader question (descriptive
            references such as "my cousin"); when it falls short, check whether
            two entities are one. Off, a mismatch only warns.
        qualify_min: Probability at or above which an entity qualifies as the schema's
            kind of thing ("Is `entities.e0.label` a person?"). Below it the entity is
            refused (``not_qualified``). Only a clear non-match is refused: a bare
            first name scored 0.2 to 0.3 when the question named the schema ("a
            Member") instead of its kind, while an animal scored 0.02.
        record_presence: Updates only: probability at or above which Clef's "does the
            input refer to this existing record?" (call kind ``record_presence``,
            one question per record) makes the record's entity count when discovery
            or qualification missed it. A record that no mention fits is asked about
            only if one of its identity words occurs in the input. 0 switches the
            check off.
        record_presence_max: Records asked about at most; above it the check is
            skipped for the collection (one request per record).
        screen_above: With more than this many mentions found, the mentions of a
            collection that has an identity are screened in code (bare figures, one
            or two letters, markup, headings, the names of the item's own fields) and
            qualified one by one before they are clustered, as the items of a list
            without an identity always are. It applies to a nested list and to a
            top-level ``Many`` whose identity is a dotted path (``"a.b"``) only, and
            never takes the candidates below Clairic's count. A table of
            cells is what that many mentions are (live: 1,663 refused maintenance
            checks in one sheet); prose has a handful. 0 switches it off.
    """

    presence_absent: float = 0.2
    presence_present: float = 0.5
    count_trust_max: int = 10
    broaden_on_count_mismatch: bool = True
    qualify_min: float = 0.4
    record_presence: float = 0.5
    record_presence_max: int = 20
    screen_above: int = 12

    def __post_init__(self) -> None:
        """Validate every setting."""
        _check_unit("presence_absent", self.presence_absent, allow_zero=True)
        _check_unit("presence_present", self.presence_present)
        if self.presence_absent > self.presence_present:
            raise ValueError("presence_absent must not exceed presence_present")
        _check_int("count_trust_max", self.count_trust_max, 1, 30)
        _check_flag("broaden_on_count_mismatch", self.broaden_on_count_mismatch)
        _check_unit("qualify_min", self.qualify_min)
        _check_unit("record_presence", self.record_presence, allow_zero=True)
        _check_int("record_presence_max", self.record_presence_max, 0, 500)
        _check_int("screen_above", self.screen_above, 0, 1000)


@dataclass(frozen=True)
class CorefConfig:
    """Settings of mention clustering (coreference) in the collection engine.

    The instance is immutable and hashable.

    Attributes:
        strategy: ``"pairwise"`` asks Clef about every pair of mentions and
            clusters the answers with average linkage; ``"incremental"`` walks the
            mentions in order and assigns each to a cluster so far (or a new one)
            with one Clef choice, which scales linearly in the mentions;
            ``"auto"`` is pairwise up to ``max_pairwise_mentions`` mentions and
            incremental above.
        merge_threshold: Probability (pairwise: the average over a pair of
            clusters) at or above which two mentions or clusters are one entity.
        max_pairwise_mentions: Largest number of mentions ``"auto"`` clusters
            pairwise.
        all_pairs_below: Up to this many mentions EVERY pair is asked about. Above
            it only plausible pairs are (a shared name token, a short form, a
            diminutive such as Bob and Robert, mentions close to each other, a
            description); an unasked pair is "not asked", never a "no".
        context_chars: Characters of the input kept on each side of a mention in
            the state of a coreference question.
    """

    strategy: Literal["pairwise", "incremental", "auto"] = "auto"
    merge_threshold: float = 0.5
    max_pairwise_mentions: int = 60
    context_chars: int = 120
    all_pairs_below: int = 24

    def __post_init__(self) -> None:
        """Validate every setting."""
        if self.strategy not in ("pairwise", "incremental", "auto"):
            raise ValueError(
                "strategy must be 'pairwise', 'incremental' or 'auto', got "
                f"{self.strategy!r}"
            )
        _check_unit("merge_threshold", self.merge_threshold)
        _check_int("max_pairwise_mentions", self.max_pairwise_mentions, 2)
        _check_int("context_chars", self.context_chars, 0)
        _check_int("all_pairs_below", self.all_pairs_below, 0)


@dataclass(frozen=True)
class MatchConfig:
    """Settings of matching entities to existing records (``Many`` with ``existing``).

    The instance is immutable and hashable.

    Attributes:
        accept: Probability an assigned option (a record, or ``__new__``) needs
            for its entity to be updated or inserted; below it the entity is
            ``ambiguous_match``.
        min_confidence: The match answer's confidence needed as well.
        accept_compatible: Probability a record needs when it is Clef's own top
            option AND the only record whose identity fits one of the entity's
            mentions (``identity_compatible``, e.g. "Emma" for "Emma Thompson"). It
            replaces ``accept`` and ``min_confidence`` for that record: Clef's
            choice confidence is not the probability of the chosen option, and a
            name that fits is the strongest evidence of a match. A fit that is
            only a prefix, a typo or an initial of a one-word mention ("Dan" for
            "Daniela Cruz") does not count.
        new_final: Probability of ``__new__`` from which a choice of "none of the
            records" is final. Below it, an entity the choice calls new may still be
            confirmed pairwise (``confirm_accept``) as one record. Clef's choice
            ``confidence`` is calibrated differently by each provider (Cloudflare gave
            0.44 for a ``__new__`` at 0.83, Experiential reports the top probability
            itself), so whether a ``__new__`` is "sure" is decided by its probability
            alone, not by that confidence. **Known risk:** with ``__new__`` between
            0.7 and 0.9 and a pairwise answer above it, a pairwise answer that is
            wrong (a Clef that says yes to a new "Emma" next to the record "Emma
            Thompson") writes the new person's facts onto the record. The live
            matches that need the band (short_form 0.83 then 0.87, Maria 0.84 then
            0.87) have the same shape, so no margin separates them. To opt out, set
            ``new_final=0.7`` (a ``__new__`` from 0.7 is final) or switch the
            confirmation off with ``confirm_accept=0`` (``1.0`` also never confirms).
        confirm_accept: Probability the pairwise confirmation (call kind
            ``match_confirm``) needs to settle a match the choice question did not
            settle: one that picks a record without being sure (``accept`` and
            ``min_confidence``), or picks ``__new__`` below ``new_final``, while
            exactly one existing record is a strong name fit for the entity. The
            question is "is this entity the same kind of thing as that record?"; its
            answer must also exceed the choice's own probability of ``__new__``.
            It never overrides a sure record or a final ``__new__``, is never asked
            when two records fit ("Emma" with Emma Thompson and Emma Watson) or when
            a full name of the entity contradicts the record ("Emma Davis"). 0
            switches it off.
        compatible_margin: By how much that record must beat the runner-up
            option (``__new__`` included) for ``accept_compatible`` to apply; 0.51
            against 0.49 is a coin flip, not a decision.
        render_fields: Non-null scalar fields a record keeps (next to its identity
            fields and the first 3 elements of each list) when the records are
            rendered compactly to fit the Clef state budget.
        collision: What to do when two entities both claim one record with
            probability 0.5 or more: ``"warn"`` keeps the assignment's winner and
            warns (``match_collision``); ``"merge"`` first asks whether the two
            entities are one (the coreference merge check) and merges them if so.
        on_ambiguous: What to do with an ambiguous entity: ``"skip"`` it
            (``SkippedEntity(reason="ambiguous_match")``), ``"insert"`` it as a
            new record, or ``"raise"`` ``AmbiguousMatchError``.
        delete_threshold: Probability of "does the input say this record should be
            removed?" from which a record is deleted (``enable_deletes``).
    """

    accept: float = 0.6
    min_confidence: float = 0.5
    accept_compatible: float = 0.5
    compatible_margin: float = 0.1
    confirm_accept: float = 0.7
    new_final: float = 0.9
    render_fields: int = 4
    collision: Literal["warn", "merge"] = "warn"
    on_ambiguous: Literal["skip", "insert", "raise"] = "skip"
    delete_threshold: float = 0.85

    def __post_init__(self) -> None:
        """Validate every setting."""
        _check_unit("accept", self.accept)
        _check_unit("min_confidence", self.min_confidence)
        _check_unit("accept_compatible", self.accept_compatible)
        _check_unit("compatible_margin", self.compatible_margin, allow_zero=True)
        _check_unit("confirm_accept", self.confirm_accept, allow_zero=True)
        _check_unit("new_final", self.new_final)
        _check_unit("delete_threshold", self.delete_threshold)
        _check_int("render_fields", self.render_fields, 0)
        if self.collision not in ("warn", "merge"):
            raise ValueError(
                f"collision must be 'warn' or 'merge', got {self.collision!r}"
            )
        if self.on_ambiguous not in ("skip", "insert", "raise"):
            raise ValueError(
                "on_ambiguous must be 'skip', 'insert' or 'raise', got "
                f"{self.on_ambiguous!r}"
            )
