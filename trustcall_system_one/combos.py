"""Declarative combinations of providers: which model answers what.

A ``ComboSpec`` says, in a dict, JSON or YAML file, which provider extracts which
field type, which provider decides each call kind, how each provider's probabilities
are calibrated to the engine's thresholds, and which ``create_extractor`` knobs go
with it. ``build_combo_transport`` turns it into a transport stack
(``KindRoutingTransport``, ``FieldTypeRoutingTransport``, ``CalibratedTransport``);
``evals`` runs and searches specs (``python -m evals combo`` and ``search``).

Example::

    spec = ComboSpec.model_validate(
        {
            "name": "jev-dates",
            "extraction": "cloudflare",
            "extraction_by_type": {"date": "typesafe", "datetime": "typesafe"},
            "decision": "cloudflare",
            "by_kind": {"attribution": "decisions", "gate": "typesafe"},
            "calibration": {"decisions": {"attribution": 0.5}},
            "options": {"verify.attribution_keep": 0.7},
        }
    )
    transport = build_combo_transport(spec)
    extractor = create_extractor(
        [Person], transport=transport, **spec.extractor_options()
    )

**Providers** are named explicitly, and naming one is the opt-in to send text to its
company (``companies()`` lists them): ``cloudflare`` (Clairic and Clef),
``experiential-clef`` (Clef on Experiential Labs), ``decisions`` (the Decisions API:
OpenAI behind Experiential Labs, Clef only, text only) and ``typesafe`` (Jev, Clairic
and Clef, text only, no ``forecast``). A key in the environment never selects one.

**Mixing rule.** ``system_one_transport`` refuses to combine its TypeSafe-only
provider with another Clef provider or a fallback because those combinations would
send text somewhere the user did not name. A spec names every provider it uses, so
mixing is allowed and those two rules are not kept. What is kept is about safety and
capability: a provider is only ever given what it serves (a Clef-only provider is
never given a Clairic field or a Clairic call kind; Jev is never given a ``forecast``
field), calibration only moves Clef probabilities and only of providers the spec
uses, and ``only`` is the latch for the TypeSafe-only guarantee: with
``"only": "typesafe"`` (or any provider) naming another provider anywhere in the spec
is refused, so a sweep over candidates can never send text elsewhere.
"""

import dataclasses
import importlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trustcall_system_one._config import (
    ConcurrencyConfig,
    CorefConfig,
    DiscoveryConfig,
    GateConfig,
    Limits,
    MatchConfig,
    PassesConfig,
    PresenceConfig,
    RepairConfig,
    VerifyConfig,
)
from trustcall_system_one._partition import route_keys
from trustcall_system_one._wire import body_model
from trustcall_system_one.transport import (
    CALL_KINDS,
    CLAIRIC_KINDS,
    CLEF_KINDS,
    CalibratedTransport,
    CallMeta,
    CloudflareTransport,
    DecisionsTransport,
    ExperientialTransport,
    FieldTypeRoutingTransport,
    KindRoutingTransport,
    Transport,
    TransportResponse,
    TypeSafeTransport,
    _checked_shifts,
    _Wrapping,
)

__all__ = [
    "CLAIRIC_PROVIDERS",
    "OPTION_FLAGS",
    "OPTION_GROUPS",
    "PROVIDERS",
    "REMOVAL_KINDS",
    "ComboSpec",
    "Provider",
    "build_combo_transport",
    "calibrate_transport",
    "companies",
    "create_provider_transports",
    "extractor_options",
    "label_provider",
    "load_combo",
    "provider_label",
]

Provider = Literal["cloudflare", "experiential-clef", "decisions", "typesafe"]
PROVIDERS: tuple[str, ...] = (
    "cloudflare",
    "experiential-clef",
    "decisions",
    "typesafe",
)
"""The providers a spec may name."""
CLAIRIC_PROVIDERS: frozenset[str] = frozenset({"cloudflare", "typesafe"})
"""The providers that can answer Clairic (typed extraction) requests."""

REMOVAL_KINDS: frozenset[str] = frozenset({"gate", "null_check", "delete"})
"""The call kinds whose Clef answers can authorise a removal: ``gate`` (a removal or
set-null needs ``gates.remove_threshold``), ``null_check`` (``verify.null_confirm_yes``
sets a value to null) and ``delete`` (``match.delete_threshold``). A positive logit
shift on them makes the engine remove more, so a calibration of them is refused by
the search unless asked for twice."""

_LABEL = {
    "cloudflare": "cloudflare",
    "experiential-clef": "experiential",
    "decisions": "decisions",
    "typesafe": "typesafe",
}
"""A provider as ``TransportResponse.provider`` names it."""
_COMPANIES = {
    "cloudflare": ("Cloudflare",),
    "experiential-clef": ("Experiential Labs",),
    "decisions": ("Experiential Labs", "OpenAI"),
    "typesafe": ("TypeSafe",),
}
"""Who sees the text a provider is given."""

OPTION_GROUPS: Mapping[str, type] = {
    "verify": VerifyConfig,
    "gates": GateConfig,
    "limits": Limits,
    "passes": PassesConfig,
    "repair": RepairConfig,
    "presence": PresenceConfig,
    "discovery": DiscoveryConfig,
    "coref": CorefConfig,
    "match": MatchConfig,
}
"""The config dataclasses a ``group.field`` option names."""
OPTION_FLAGS = frozenset(
    {
        "enable_inserts",
        "enable_updates",
        "enable_deletes",
        "enable_item_removals",
        "strict",
        "existing_schema_policy",
    }
)
"""The top-level ``create_extractor`` flags an option may set."""


def companies(providers: Iterable[str]) -> tuple[str, ...]:
    """The companies that see the text sent to these providers.

    ``decisions`` is OpenAI behind Experiential Labs, so it names both.
    """
    return tuple(dict.fromkeys(c for p in providers for c in _COMPANIES[p]))


def extractor_options(
    overrides: Mapping[str, Any], base: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``create_extractor`` keyword arguments for a set of overrides.

    Args:
        overrides: ``group.field`` overrides of a config dataclass
            (``verify.attribution_keep``) and top-level flags (``enable_inserts``).
        base: Keyword arguments to start from (a case's ``enable_inserts``...); the
            overrides come on top.

    Returns:
        Keyword arguments for ``create_extractor``.

    Raises:
        ValueError: If a group or field is unknown or a value is invalid.
    """
    options: dict[str, Any] = dict(base or {})
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        group, dot, name = key.partition(".")
        if not dot:
            if key not in OPTION_FLAGS:
                raise ValueError(
                    f"unknown option {key!r}; flags: {sorted(OPTION_FLAGS)}, or "
                    f"group.field with a group of {sorted(OPTION_GROUPS)}"
                )
            options[key] = value
            continue
        if group not in OPTION_GROUPS:
            raise ValueError(
                f"unknown group {group!r} in {key!r}; groups: {sorted(OPTION_GROUPS)}"
            )
        names = {f.name for f in dataclasses.fields(OPTION_GROUPS[group])}
        if name not in names:
            raise ValueError(f"{group} has no field {name!r}; fields: {sorted(names)}")
        grouped.setdefault(group, {})[name] = value
    for group, values in grouped.items():
        klass = OPTION_GROUPS[group]
        start = options.get(group) or klass()
        options[group] = dataclasses.replace(start, **values)
    return options


def _kind_families(kind: str, model: str | None) -> set[str]:
    """The model families a call kind is asked of (``clairic``, ``clef``)."""
    if model is not None:
        return {"clairic" if model == "clairic" else "clef"}
    if kind in CLAIRIC_KINDS:
        return {"clairic"}
    if kind in CLEF_KINDS:
        return {"clef"}
    return {"clairic", "clef"}  # repair and raw


def _split_kind_key(key: str) -> tuple[str, str | None]:
    """``(kind, model or None)`` of a ``by_kind`` key, validated."""
    kind, at, model = key.partition("@")
    if kind not in CALL_KINDS:
        raise ValueError(
            f"by_kind: unknown call kind {kind!r}; expected one of {sorted(CALL_KINDS)}"
        )
    if not at:
        return kind, None
    try:
        name = body_model(model)
    except ValueError:
        raise ValueError(
            f"by_kind: {key!r}: the model must be clairic, clef or clef-flash"
        ) from None
    if ("clairic" if name == "clairic" else "clef") not in _kind_families(kind, None):
        raise ValueError(f"by_kind: {key!r}: {kind!r} is never asked of {name}")
    return kind, name


class ComboSpec(BaseModel):
    """A combination of providers, as data.

    Attributes:
        name: A label for reports and result directories.
        extraction: The provider of every Clairic request (``cloudflare`` or
            ``typesafe``) whose field type has no entry in ``extraction_by_type``.
        extraction_by_type: Field type (``text``, ``number``, ``date``, ``time``,
            ``datetime``, ``duration``, ``list``, ``estimate``, ``forecast``, or
            ``list:text`` / ``list:number`` / ``list:date``) to the provider that
            extracts it. A Clairic request is split by it and merged back (see
            ``FieldTypeRoutingTransport``).
        decision: The provider of every Clef question without an entry in
            ``by_kind``.
        by_kind: Call kind (any ``CallMeta.kind``; ``kind@clairic`` or ``kind@clef``
            for ``repair`` and ``raw``, which are both) to the provider that answers
            it. A Clairic kind (``extract``, ``crosscheck``, ``reask``, ``discover``,
            ``count``) sends the whole request to that provider, whatever the field
            types.
        calibration: Provider to call kind to logit shift of its Clef probabilities
            (see ``CalibratedTransport``); no entry means no shift.
        options: ``create_extractor`` knobs: ``group.field`` overrides
            (``verify.attribution_keep``, ``gates.add_threshold``,
            ``match.accept``...) and the top-level flags.
        only: When set, no other provider may be named anywhere in the spec (the
            TypeSafe-only guarantee, for any provider).
        labels: Free-form labels, recorded with the results.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    extraction: Provider = "cloudflare"
    extraction_by_type: dict[str, Provider] = Field(default_factory=dict)
    decision: Provider = "cloudflare"
    by_kind: dict[str, Provider] = Field(default_factory=dict)
    calibration: dict[Provider, dict[str, float]] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    only: Provider | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("extraction_by_type")
    @classmethod
    def _check_types(cls, value: dict[str, Provider]) -> dict[str, Provider]:
        known = route_keys()
        for key, provider in value.items():
            if key not in known:
                raise ValueError(
                    f"extraction_by_type: unknown field type {key!r}; expected one "
                    f"of {list(known)}"
                )
            if provider not in CLAIRIC_PROVIDERS:
                raise ValueError(
                    f"extraction_by_type: {provider!r} serves Clef only and cannot "
                    f"extract {key!r} fields"
                )
            if provider == "typesafe" and key == "forecast":
                raise ValueError(
                    "extraction_by_type: typesafe (Jev) cannot serve forecast fields"
                )
        return value

    @field_validator("extraction")
    @classmethod
    def _check_extraction(cls, value: Provider) -> Provider:
        if value not in CLAIRIC_PROVIDERS:
            raise ValueError(
                f"extraction: {value!r} serves Clef only; extraction is "
                "cloudflare or typesafe"
            )
        return value

    @field_validator("by_kind")
    @classmethod
    def _check_kinds(cls, value: dict[str, Provider]) -> dict[str, Provider]:
        normal: dict[str, Provider] = {}
        for key, provider in value.items():
            kind, model = _split_kind_key(key)
            families = _kind_families(kind, model)
            if "clairic" in families and provider not in CLAIRIC_PROVIDERS:
                hint = (
                    f"; write {kind}@clef to route only its Clef questions"
                    if model is None and kind in ("repair", "raw")
                    else ""
                )
                raise ValueError(
                    f"by_kind: {provider!r} serves Clef only but {key!r} is also a "
                    f"Clairic call{hint}"
                )
            normal[kind if model is None else f"{kind}@{model}"] = provider
        return normal

    @field_validator("calibration")
    @classmethod
    def _check_calibration(
        cls, value: dict[Provider, dict[str, float]]
    ) -> dict[Provider, dict[str, float]]:
        for provider, table in value.items():
            _checked_shifts(table, f"calibration of {provider}")
            clairic = sorted(k for k in table if k in CLAIRIC_KINDS)
            if clairic:
                raise ValueError(
                    f"calibration of {provider}: {clairic} are Clairic calls; a "
                    "calibration only moves Clef probabilities"
                )
        return value

    @field_validator("options")
    @classmethod
    def _check_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        extractor_options(value)  # ValueError for an unknown knob or a bad value
        return value

    @model_validator(mode="after")
    def _check_whole(self) -> "ComboSpec":
        named = self.providers()
        for provider in self.calibration:
            if provider not in named:
                raise ValueError(
                    f"calibration names {provider!r}, which the spec never uses "
                    f"(it uses {list(named)})"
                )
        if self.only is not None:
            others = [p for p in named if p != self.only]
            if others:
                raise ValueError(
                    f"only={self.only!r} but the spec also names {others}: text would "
                    "go to another company"
                )
        return self

    # -- what the spec names -------------------------------------------------- #

    def removal_shifts(self) -> list[str]:
        """The positive calibration shifts of kinds that can authorise a removal.

        ``provider.kind+shift`` for each (``gate``, ``null_check`` and ``delete``
        raise the probability that decides to remove or delete): they move the
        engine towards removing data, so a report flags them.
        """
        return [
            f"{provider}.{kind}{shift:+g}"
            for provider, table in sorted(self.calibration.items())
            for kind, shift in sorted(table.items())
            if kind in REMOVAL_KINDS and shift > 0
        ]

    def providers(self) -> tuple[str, ...]:
        """The providers the spec names anywhere, in a stable order."""
        found = [
            self.extraction,
            *self.extraction_by_type.values(),
            self.decision,
            *self.by_kind.values(),
        ]
        return tuple(p for p in PROVIDERS if p in found)

    def companies(self) -> tuple[str, ...]:
        """The companies that will see the text (naming a provider is the opt-in)."""
        return tuple(dict.fromkeys(c for p in self.providers() for c in _COMPANIES[p]))

    def canonical(self) -> "ComboSpec":
        """The same combination with every redundant entry removed.

        Entries that say what the default already says (a ``by_kind`` entry equal to
        what ``decision`` (Clef) and ``extraction`` (Clairic, when no field type is
        routed) already say, an ``extraction_by_type`` entry equal to ``extraction``,
        a shift of 0) are dropped, so two specs that route alike are equal.
        """
        by_type = {
            k: p for k, p in self.extraction_by_type.items() if p != self.extraction
        }

        def redundant(key: str, provider: str) -> bool:
            families = _kind_families(*_split_kind_key(key))
            return ("clef" not in families or provider == self.decision) and (
                "clairic" not in families
                or (provider == self.extraction and not by_type)
            )

        by_kind = {k: p for k, p in self.by_kind.items() if not redundant(k, p)}
        calibration = {
            p: {k: s for k, s in table.items() if s}
            for p, table in self.calibration.items()
        }
        return ComboSpec(
            name=self.name,
            extraction=self.extraction,
            extraction_by_type=by_type,
            decision=self.decision,
            by_kind=by_kind,
            calibration={p: t for p, t in calibration.items() if t},
            options=dict(self.options),
            only=self.only,
            labels=dict(self.labels),
        )

    def key(self) -> str:
        """A canonical string of the combination (routing, calibration, options).

        ``name``, ``only`` and ``labels`` are not part of it: two specs with the same
        key behave the same.
        """
        data = self.canonical().model_dump(exclude={"name", "only", "labels"})
        return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)

    def describe(self) -> str:
        """A one-line description for reports."""
        spec = self.canonical()
        parts = [f"extract {spec.extraction}"]
        parts += [f"{t}={p}" for t, p in sorted(spec.extraction_by_type.items())]
        parts.append(f"decide {spec.decision}")
        parts += [f"{k}={p}" for k, p in sorted(spec.by_kind.items())]
        parts += [
            f"cal {p}.{k}{s:+g}"
            for p, table in sorted(spec.calibration.items())
            for k, s in sorted(table.items())
        ]
        parts += [f"{k}={v}" for k, v in sorted(spec.options.items())]
        return ", ".join(parts)

    def replace(self, **changes: Any) -> "ComboSpec":
        """A validated copy with some fields replaced."""
        return ComboSpec.model_validate({**self.model_dump(), **changes})

    def extractor_options(
        self, base: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The ``create_extractor`` keyword arguments of ``options``."""
        return extractor_options(self.options, base)

    @classmethod
    def from_file(cls, path: str | Path) -> "ComboSpec":
        """Load a spec from a JSON (or, with PyYAML installed, YAML) file."""
        return load_combo(path)


def load_combo(path: str | Path) -> ComboSpec:
    """Read a ``ComboSpec`` from a JSON file, or a YAML one when PyYAML is installed.

    Args:
        path: ``.json``, or ``.yaml`` / ``.yml``.

    Returns:
        The validated spec (its ``name`` defaults to the file's stem).

    Raises:
        ValueError: If the file is not valid JSON/YAML, is not an object, needs PyYAML
            and it is missing, or does not validate (unknown provider, kind, field
            type, option; a provider asked for what it cannot serve; ...).
    """
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    data: Any
    if file.suffix.lower() in (".yaml", ".yml"):
        try:
            yaml = importlib.import_module("yaml")
        except ImportError:
            raise ValueError(
                f"{file.name}: reading YAML needs PyYAML (install pyyaml), or write "
                "the spec as JSON"
            ) from None
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"{file.name}: not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{file.name}: a combination spec is an object")
    data.setdefault("name", file.stem)
    return ComboSpec.model_validate(data)


class _Labelled(_Wrapping):
    """Gives a transport's responses the provider's name when they carry none.

    Real transports name themselves; a scripted or custom one injected into
    ``build_combo_transport`` may not, and a calibration is looked up by the provider
    a response names.
    """

    def __init__(self, inner: Transport, label: str) -> None:
        self._inner = inner
        self._label = label

    @property
    def provider(self) -> str:
        """The provider's name."""
        return self._label

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        """Run the call; name the provider on a response that does not."""
        reply = await self._inner.run(model, payload, meta=meta, timeout=timeout)
        if reply.provider is None:
            return dataclasses.replace(reply, provider=self._label)
        return reply


def _create(provider: str, concurrency: ConcurrencyConfig | None) -> Transport:
    if provider == "cloudflare":
        return CloudflareTransport(concurrency=concurrency)
    if provider == "experiential-clef":
        return ExperientialTransport(concurrency=concurrency)
    if provider == "decisions":
        return DecisionsTransport(concurrency=concurrency)
    return TypeSafeTransport(concurrency=concurrency)


def create_provider_transports(
    spec: ComboSpec, concurrency: ConcurrencyConfig | None = None
) -> dict[str, Transport]:
    """One new transport for each provider the spec names (and for no other).

    For a caller that shares them between several stacks (``providers=`` of
    ``build_combo_transport``); credentials are read when a request is sent.
    """
    return {p: _create(p, concurrency) for p in spec.providers()}


def provider_label(provider: str) -> str:
    """A provider as ``TransportResponse.provider`` (and a recording) names it."""
    return _LABEL[provider]


def label_provider(transport: Transport, provider: str) -> Transport:
    """``transport``, naming ``provider`` on a response that names none."""
    return _Labelled(transport, _LABEL[provider])


def calibrate_transport(spec: ComboSpec, transport: Transport) -> Transport:
    """Wrap a transport in the spec's calibration (``transport`` itself without one).

    The shifts are looked up by the provider a response names, so the wrapper can sit
    outside a recorder or a cassette that stores each provider's raw answers: a
    recording made under one calibration replays under another.
    """
    table = {
        _LABEL[provider]: {k: s for k, s in shifts.items() if s}
        for provider, shifts in spec.calibration.items()
    }
    table = {p: t for p, t in table.items() if t}
    return CalibratedTransport(transport, by_provider=table) if table else transport


def build_combo_transport(
    spec: ComboSpec,
    *,
    providers: Mapping[str, Transport] | None = None,
    concurrency: ConcurrencyConfig | None = None,
    calibrate: bool = True,
    wrap: Callable[[str, Transport], Transport] | None = None,
) -> Transport:
    """Build the transport stack of a spec.

    One transport is created for each provider the spec names (and only those: a
    provider that is not named is never created, so no key of it is ever read). The
    stack is, from the outside in: ``CalibratedTransport`` (when the spec calibrates
    anything), ``KindRoutingTransport`` (``by_kind``, then ``clairic`` and ``clef``
    as the two defaults) and ``FieldTypeRoutingTransport`` for Clairic (when
    ``extraction_by_type`` is not empty). A spec that names one provider returns that
    transport itself.

    A request with images goes to Cloudflare when the routed provider cannot take it
    and the spec names Cloudflare (the same rule as ``RoutingTransport``); otherwise
    the routed provider refuses it locally.

    Args:
        spec: The combination.
        providers: Transports to use instead of creating them (offline runs, tests,
            a shared client): provider name to transport. Every provider the spec
            names must be there.
        concurrency: Concurrency limits for the transports created here.
        calibrate: ``False`` leaves the calibration out (see ``calibrate_transport``,
            to put it outside a recorder).
        wrap: Wraps the transport of each provider (``wrap(provider, transport)``),
            under the routers: a recorder or a cassette that sees, and can tell apart,
            the calls each provider gets (a replay then answers a call only from what
            that provider recorded).

    Returns:
        The transport, for ``create_extractor(..., transport=...)``; close it with
        ``aclose`` or ``close``.

    Raises:
        ValueError: If ``providers`` lacks a provider the spec names or names an
            unknown one.
    """
    spec = spec.canonical()
    named = spec.providers()
    given = dict(providers or {})
    unknown = sorted(set(given) - set(PROVIDERS))
    if unknown:
        raise ValueError(f"unknown provider(s) {unknown}; expected one of {PROVIDERS}")
    missing = [p for p in named if p not in given and providers is not None]
    if missing:
        raise ValueError(f"the spec names {missing}, which were not provided")
    members: dict[str, Transport] = {
        p: _Labelled(given[p], _LABEL[p])
        if providers is not None
        else _create(p, concurrency)
        for p in named
    }
    if wrap is not None:
        members = {p: wrap(p, t) for p, t in members.items()}
    clairic: Transport = members[spec.extraction]
    if spec.extraction_by_type:
        clairic = FieldTypeRoutingTransport(
            {t: members[p] for t, p in spec.extraction_by_type.items()},
            default=clairic,
        )
    stack: Transport
    if (
        len(named) == 1
        and not spec.by_kind
        and not spec.extraction_by_type
        and spec.extraction == spec.decision
    ):
        stack = members[spec.decision]
    else:
        clef = members[spec.decision]
        stack = KindRoutingTransport(
            {
                "clairic": clairic,
                "clef": clef,
                **{k: members[p] for k, p in spec.by_kind.items()},
            },
            default=members.get("cloudflare", clef),
        )
    return calibrate_transport(spec, stack) if calibrate else stack
