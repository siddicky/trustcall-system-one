"""Combination specs: validation, canonical form, files, and the transport stack."""

import asyncio
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CalibratedTransport,
    CallMeta,
    CloudflareTransport,
    DecisionsTransport,
    ExperientialTransport,
    FieldTypeRoutingTransport,
    KindRoutingTransport,
    TransportResponse,
    TypeSafeTransport,
    system_one_transport,
)
from trustcall_system_one import combos as combos_module
from trustcall_system_one.combos import (
    PROVIDERS,
    ComboSpec,
    build_combo_transport,
    calibrate_transport,
    companies,
    extractor_options,
    load_combo,
)
from trustcall_system_one.transport import max_questions, state_token_budget

FULL: dict[str, Any] = {
    "name": "mix",
    "extraction": "cloudflare",
    "extraction_by_type": {"date": "typesafe", "list:date": "typesafe"},
    "decision": "cloudflare",
    "by_kind": {
        "attribution": "decisions",
        "gate": "typesafe",
        "repair@clef": "typesafe",
    },
    "calibration": {"decisions": {"attribution": 0.5}, "typesafe": {"gate": -0.5}},
    "options": {"verify.attribution_keep": 0.7, "gates.add_threshold": 0.5},
    "labels": {"note": "x"},
}


def refused(**spec: Any) -> str:
    with pytest.raises(ValueError) as caught:
        ComboSpec.model_validate(spec)
    return str(caught.value)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_the_default_spec_is_all_cloudflare_and_a_full_one_validates() -> None:
    assert ComboSpec().providers() == ("cloudflare",)
    spec = ComboSpec.model_validate(FULL)
    assert spec.providers() == ("cloudflare", "decisions", "typesafe")
    assert spec.by_kind["repair@clef"] == "typesafe"
    assert spec.companies() == ("Cloudflare", "Experiential Labs", "OpenAI", "TypeSafe")


def test_unknown_providers_kinds_types_and_options_are_refused() -> None:
    assert "typesafe" in refused(decision="gpt")
    assert "unknown call kind 'nope'" in refused(by_kind={"nope": "typesafe"})
    assert "never asked of clairic" in refused(by_kind={"gate@clairic": "typesafe"})
    assert "unknown field type 'color'" in refused(
        extraction_by_type={"color": "typesafe"}
    )
    assert "unknown option" in refused(options={"nope": 1})
    assert "has no field" in refused(options={"verify.nope": 1})
    assert "unknown group" in refused(options={"nope.x": 1})
    assert "unknown call kind" in refused(
        calibration={"typesafe": {"nope": 0.5}}, decision="typesafe"
    )
    with pytest.raises(ValidationError):
        ComboSpec.model_validate({"surprise": 1})  # a typo is an error, not ignored
    assert "model must be" in refused(by_kind={"gate@gpt": "typesafe"})


def test_a_provider_is_only_given_what_it_serves() -> None:
    assert "serves Clef only" in refused(extraction="decisions")
    assert "serves Clef only" in refused(extraction="experiential-clef")
    assert "serves Clef only" in refused(extraction_by_type={"date": "decisions"})
    assert "cannot serve forecast" in refused(
        extraction_by_type={"forecast": "typesafe"}
    )
    assert "Clairic call" in refused(by_kind={"extract": "decisions"})
    assert "Clairic call" in refused(by_kind={"crosscheck": "experiential-clef"})
    ambiguous = refused(by_kind={"repair": "decisions"})
    assert "repair@clef" in ambiguous  # it says how to write what was probably meant
    ComboSpec.model_validate({"by_kind": {"repair@clef": "decisions"}})
    ComboSpec.model_validate({"by_kind": {"repair@clairic": "typesafe"}})
    assert "Clairic call" in refused(by_kind={"repair@clairic": "decisions"})
    ComboSpec.model_validate(
        {"by_kind": {"crosscheck": "typesafe", "extract": "typesafe"}}
    )


def test_calibration_only_moves_clef_probabilities_of_providers_the_spec_uses() -> None:
    assert "never uses" in refused(calibration={"typesafe": {"gate": 1.0}})
    assert "Clairic calls" in refused(
        calibration={"cloudflare": {"extract": 1.0}}, decision="cloudflare"
    )
    assert "within" in refused(calibration={"cloudflare": {"gate": 99}})


def test_mixing_is_allowed_because_each_provider_is_named_but_only_is_a_latch() -> None:
    # system_one_transport(provider="typesafe") refuses another Clef provider and a
    # fallback, because there they would send text somewhere unnamed; a spec names it
    with pytest.raises(ValueError, match="TypeSafe provider answers Clef"):
        system_one_transport(clef="decisions", provider="typesafe")
    mixed = ComboSpec.model_validate(
        {
            "extraction": "typesafe",
            "decision": "typesafe",
            "by_kind": {"gate": "decisions"},
        }
    )
    assert mixed.providers() == ("decisions", "typesafe")
    only = ComboSpec.model_validate(
        {"only": "typesafe", "extraction": "typesafe", "decision": "typesafe"}
    )
    assert only.providers() == ("typesafe",)
    assert "text would go to another company" in refused(
        only="typesafe",
        extraction="typesafe",
        decision="typesafe",
        by_kind={"gate": "decisions"},
    )
    assert "text would go to another company" in refused(only="typesafe")  # cloudflare
    with pytest.raises(ValueError):
        only.replace(extraction_by_type={"date": "cloudflare"})


def test_a_dict_of_ints_and_a_frozen_spec() -> None:
    spec = ComboSpec.model_validate({"calibration": {"cloudflare": {"gate": 1}}})
    assert spec.calibration["cloudflare"]["gate"] == 1.0
    with pytest.raises(ValidationError):
        spec.name = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #


def test_redundant_entries_do_not_change_the_key() -> None:
    plain = ComboSpec()
    noisy = ComboSpec.model_validate(
        {
            "name": "other",
            "labels": {"a": "b"},
            "by_kind": {"gate": "cloudflare", "extract": "cloudflare"},
            "extraction_by_type": {"date": "cloudflare"},
            "calibration": {"cloudflare": {"gate": 0}},
        }
    )
    assert noisy.key() == plain.key()
    assert noisy.canonical().by_kind == {} == noisy.canonical().calibration
    different = ComboSpec.model_validate({"by_kind": {"gate": "typesafe"}})
    assert different.key() != plain.key()
    # a Clairic kind is redundant only when no field type is split away from it
    split = ComboSpec.model_validate(
        {
            "extraction_by_type": {"date": "typesafe"},
            "by_kind": {"extract": "cloudflare"},
        }
    )
    assert split.canonical().by_kind == {"extract": "cloudflare"}
    assert (
        ComboSpec.model_validate({"by_kind": {"repair": "cloudflare"}})
        .canonical()
        .by_kind
        == {}
    )


def test_describe_and_replace() -> None:
    spec = ComboSpec.model_validate(FULL)
    text = spec.describe()
    for part in (
        "extract cloudflare",
        "date=typesafe",
        "attribution=decisions",
        "cal decisions.attribution+0.5",
    ):
        assert part in text
    more = spec.replace(decision="typesafe")
    assert more.decision == "typesafe" and spec.decision == "cloudflare"
    with pytest.raises(ValueError):
        spec.replace(decision="nope")


def test_extractor_options() -> None:
    options = ComboSpec.model_validate(FULL).extractor_options({"enable_inserts": True})
    assert options["verify"].attribution_keep == 0.7
    assert options["gates"].add_threshold == 0.5
    assert options["enable_inserts"] is True
    assert extractor_options({}) == {}
    with pytest.raises(ValueError):
        extractor_options({"verify.attribution_keep": 7})  # the config refuses it


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def test_a_spec_loads_from_json_and_takes_the_files_name(tmp_path: Path) -> None:
    file = tmp_path / "jev-dates.json"
    file.write_text(json.dumps({"extraction_by_type": {"date": "typesafe"}}))
    spec = load_combo(file)
    assert spec.name == "jev-dates" and spec.extraction_by_type == {"date": "typesafe"}
    assert ComboSpec.from_file(file) == spec
    named = tmp_path / "x.json"
    named.write_text(json.dumps({"name": "mine"}))
    assert load_combo(named).name == "mine"


def test_a_bad_file_is_a_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_combo(bad)
    bad.write_text("[1]")
    with pytest.raises(ValueError, match="an object"):
        load_combo(bad)
    bad.write_text(json.dumps({"decision": "gpt"}))
    with pytest.raises(ValueError):
        load_combo(bad)


def test_yaml_is_read_when_pyyaml_is_there_and_explained_when_it_is_not(
    tmp_path: Path,
) -> None:
    file = tmp_path / "spec.yaml"
    file.write_text("name: y\ndecision: typesafe\nextraction: typesafe\n")
    if importlib.util.find_spec("yaml") is None:
        with pytest.raises(ValueError, match="PyYAML"):
            load_combo(file)
    else:  # pragma: no cover - PyYAML is not a dependency
        assert load_combo(file).decision == "typesafe"


# --------------------------------------------------------------------------- #
# The stack
# --------------------------------------------------------------------------- #


class Probe:
    """A scripted transport: answers with who it is, whatever the model."""

    def __init__(
        self,
        name: str,
        *,
        limit: int | None = None,
        serves: bool = True,
        images: bool = False,
    ) -> None:
        self.name = name
        self._images = images
        self.calls: list[tuple[str, str]] = []
        self.closed = 0
        self._serves = serves
        if limit is not None:
            self.max_questions = limit

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        return self._serves and (self._images or not (payload or {}).get("images"))

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.calls.append((meta.kind, model))
        return TransportResponse(body={"served_by": self.name})

    async def aclose(self) -> None:
        self.closed += 1


def stack(spec: Mapping[str, Any]) -> tuple[Any, dict[str, Probe]]:
    probes = {p: Probe(p) for p in PROVIDERS}
    built = build_combo_transport(ComboSpec.model_validate(spec), providers=probes)
    return built, probes


def served(transport: Any, kind: str, model: str = CLEF) -> str:
    reply = asyncio.run(transport.run(model, {}, meta=CallMeta(kind=kind)))  # type: ignore[arg-type]
    return str(reply.body["served_by"])


def test_one_provider_is_the_transport_itself_and_nothing_else_is_created() -> None:
    made: list[str] = []
    real = combos_module._create

    def spy(provider: str, concurrency: Any) -> Any:
        made.append(provider)
        return real(provider, concurrency)

    combos_module._create = spy
    try:
        only_cf = build_combo_transport(ComboSpec())
        only_jev = build_combo_transport(
            ComboSpec.model_validate({"extraction": "typesafe", "decision": "typesafe"})
        )
        mixed = build_combo_transport(ComboSpec.model_validate(FULL))
    finally:
        combos_module._create = real
    assert isinstance(only_cf, CloudflareTransport)
    assert isinstance(only_jev, TypeSafeTransport)
    assert made == ["cloudflare", "typesafe", "cloudflare", "decisions", "typesafe"]
    assert isinstance(mixed, CalibratedTransport)


def test_a_key_in_the_environment_selects_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TYPESAFE_API_KEY", "EXPERIENTIAL_API_KEY"):
        monkeypatch.setenv(name, "x" * 20)
    monkeypatch.setenv("TRUSTCALL_S1_PROVIDER", "typesafe")
    monkeypatch.setenv("TRUSTCALL_S1_CLEF_PROVIDER", "decisions")
    transport = build_combo_transport(ComboSpec())  # the spec is the opt-in, not these
    assert isinstance(transport, CloudflareTransport)


def test_routes_follow_the_spec() -> None:
    transport, _ = stack(
        {
            "extraction": "cloudflare",
            "decision": "experiential-clef",
            "by_kind": {
                "attribution": "decisions",
                "gate": "typesafe",
                "crosscheck": "typesafe",
            },
        }
    )
    assert isinstance(transport, KindRoutingTransport)
    assert served(transport, "attribution") == "decisions"
    assert served(transport, "gate") == "typesafe"
    assert served(transport, "presence") == "experiential-clef"
    assert served(transport, "extract", CLAIRIC) == "cloudflare"
    assert served(transport, "crosscheck", CLAIRIC) == "typesafe"


def test_extraction_by_type_puts_a_field_type_router_on_clairic() -> None:
    transport, probes = stack(
        {"extraction": "cloudflare", "extraction_by_type": {"date": "typesafe"}}
    )
    clairic = transport.transport_for(CLAIRIC, "extract")
    assert isinstance(clairic, FieldTypeRoutingTransport)
    assert served(transport, "gate") == "cloudflare"  # Clef is not split
    assert all(p.calls == [] for n, p in probes.items() if n != "cloudflare")


def test_specs_that_say_what_system_one_transport_says_route_alike() -> None:
    probes = {"cloudflare": Probe("cloudflare"), "experiential-clef": Probe("exp")}
    ours = build_combo_transport(
        ComboSpec.model_validate({"decision": "experiential-clef"}), providers=probes
    )
    theirs = system_one_transport(
        "experiential",
        provider="cloudflare",
        clef_fallback="none",
        cloudflare=probes["cloudflare"],  # type: ignore[arg-type]
        experiential=probes["experiential-clef"],  # type: ignore[arg-type]
    )
    for model in (CLEF, CLAIRIC):
        assert served(ours, "gate", model) == served(theirs, "gate", model)


def test_images_go_to_cloudflare_when_it_is_named_else_the_route_refuses() -> None:
    probes = {p: Probe(p, images=p == "cloudflare") for p in PROVIDERS}
    spec = ComboSpec.model_validate(
        {"decision": "decisions", "by_kind": {"gate": "typesafe"}}
    )
    transport = build_combo_transport(spec, providers=probes)
    # Cloudflare is named (the Clairic default): a pictured Clef request goes there
    pictured = {"images": ["data:image/png;base64,iVBORw0KGgo="]}
    reply = asyncio.run(transport.run(CLEF, pictured, meta=CallMeta(kind="gate")))
    assert (
        reply.body == {"served_by": "cloudflare"} and reply.rerouted_from == "typesafe"
    )
    no_cf = ComboSpec.model_validate(
        {
            "extraction": "typesafe",
            "decision": "decisions",
            "by_kind": {"gate": "typesafe"},
        }
    )
    transport = build_combo_transport(no_cf, providers=probes)
    reply = asyncio.run(transport.run(CLEF, pictured, meta=CallMeta(kind="gate")))
    assert reply.rerouted_from is None  # nothing else was named: the route refuses


def test_the_stack_forwards_limits_and_closes_what_it_made() -> None:
    probes = {
        "cloudflare": Probe("cloudflare", limit=64),
        "decisions": Probe("decisions", limit=32),
    }
    spec = ComboSpec.model_validate({"by_kind": {"attribution": "decisions"}})
    transport: Any = build_combo_transport(spec, providers=probes)
    assert max_questions(transport, CLEF) == 32
    assert state_token_budget(transport, CLEF) is None
    asyncio.run(transport.aclose())
    assert probes["cloudflare"].closed == 1 and probes["decisions"].closed == 1


def test_real_transports_are_the_ones_the_providers_name() -> None:
    spec = ComboSpec.model_validate(
        {
            "decision": "experiential-clef",
            "by_kind": {"attribution": "decisions", "gate": "typesafe"},
            "extraction_by_type": {"date": "typesafe"},
        }
    )
    transport: Any = build_combo_transport(spec)
    members = transport._members()
    kinds = {type(m) for m in members}
    assert {ExperientialTransport, DecisionsTransport, TypeSafeTransport} <= kinds
    assert max_questions(transport, CLEF) == 32  # Experiential's and Decisions' 32
    asyncio.run(transport.aclose())


def test_the_providers_must_be_provided_and_known() -> None:
    spec = ComboSpec.model_validate({"decision": "typesafe"})
    with pytest.raises(ValueError, match="were not provided"):
        build_combo_transport(spec, providers={"cloudflare": Probe("cf")})
    with pytest.raises(ValueError, match="unknown provider"):
        build_combo_transport(spec, providers={"gpt": Probe("x")})


def test_calibration_sits_outside_and_follows_the_provider_that_answered() -> None:
    class Noul(Probe):
        async def run(self, *args: Any, **kwargs: Any) -> TransportResponse:
            return TransportResponse(
                body={
                    "model": "clef",
                    "answers": {"q": {"type": "noul", "noul": 0.5}},
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    probes = {p: Noul(p) for p in ("cloudflare", "decisions")}
    spec = ComboSpec.model_validate(
        {
            "by_kind": {"attribution": "decisions"},
            "calibration": {
                "decisions": {"attribution": 2.0},
                "cloudflare": {"gate": -2.0},
            },
        }
    )
    transport = build_combo_transport(spec, providers=probes)

    def noul(kind: str) -> float:
        reply = asyncio.run(transport.run(CLEF, {}, meta=CallMeta(kind=kind)))  # type: ignore[arg-type]
        return float(reply.body["answers"]["q"]["noul"])

    assert noul("attribution") > 0.8  # decisions, +2
    assert noul("gate") < 0.2  # cloudflare, -2
    assert noul("presence") == 0.5  # no shift for the kind
    raw = build_combo_transport(spec, providers=probes, calibrate=False)
    assert not isinstance(raw, CalibratedTransport)
    outside = calibrate_transport(spec, raw)
    assert isinstance(outside, CalibratedTransport)
    assert calibrate_transport(ComboSpec(), raw) is raw  # nothing to calibrate


def test_companies_names_who_sees_the_text() -> None:
    assert companies(["decisions"]) == ("Experiential Labs", "OpenAI")
    assert companies(["cloudflare", "typesafe"]) == ("Cloudflare", "TypeSafe")
