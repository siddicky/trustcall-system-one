"""``RepairConfig.keep_uncertain`` defaults by who answers attribution and re-ask.

``None`` (auto) is on when the transport that answers the attribution question AND
the one that answers the Clairic repair and re-ask are TypeSafe (Jev), and off
otherwise; ``True`` or ``False`` always wins. Routers answer by (model, call kind).
"""

import json
from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel

from tests.unit.extract_helpers import Script, ca, make_extractor
from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CalibratedTransport,
    CallMeta,
    CloudflareTransport,
    DecisionsTransport,
    ExperientialTransport,
    FallbackTransport,
    FieldTypeRoutingTransport,
    KindRoutingTransport,
    RepairConfig,
    RoutingTransport,
    TransportResponse,
    TypeSafeTransport,
    VerifyConfig,
    create_extractor,
)
from trustcall_system_one._run import Run
from trustcall_system_one.combos import ComboSpec, build_combo_transport
from trustcall_system_one.testing import (
    CassetteTransport,
    FakeTransport,
    RecordingTransport,
)
from trustcall_system_one.transport import Transport, prefers_keep_uncertain

KEY = "xpl_" + "ab12cd34" * 5

__all__: list[str] = []


def _cloudflare() -> CloudflareTransport:
    return CloudflareTransport(account_id="a", api_token="t")


def _jev() -> TypeSafeTransport:
    return TypeSafeTransport(api_key=KEY)


def _keeps(transport: Transport, repair: RepairConfig | None = None) -> bool:
    return Run(transport=transport, repair=repair or RepairConfig()).keeps_uncertain


# --------------------------------------------------------------------------- #
# The config
# --------------------------------------------------------------------------- #


def test_the_default_is_auto() -> None:
    assert RepairConfig().keep_uncertain is None
    assert RepairConfig(keep_uncertain=True).keep_uncertain is True
    assert RepairConfig(keep_uncertain=False).keep_uncertain is False


@pytest.mark.parametrize("bad", ["auto", 1, 0.5, []])
def test_only_a_bool_or_none_is_a_setting(bad: Any) -> None:
    with pytest.raises(ValueError, match="keep_uncertain must be a bool or None"):
        RepairConfig(keep_uncertain=bad)


# --------------------------------------------------------------------------- #
# What a transport says
# --------------------------------------------------------------------------- #


def test_only_typesafe_prefers_it() -> None:
    assert prefers_keep_uncertain(_jev()) is True
    assert prefers_keep_uncertain(_cloudflare()) is False
    assert prefers_keep_uncertain(ExperientialTransport(api_key=KEY)) is False
    assert prefers_keep_uncertain(DecisionsTransport(api_key=KEY)) is False
    assert prefers_keep_uncertain(FakeTransport(lambda *a: None)) is False
    assert prefers_keep_uncertain(object()) is False


def test_a_method_or_a_non_bool_attribute_is_read_defensively() -> None:
    class ByModel:
        def prefers_keep_uncertain(self, model: str, kind: str | None = None) -> bool:
            return model == CLAIRIC and kind != "attribution"

    class Odd:
        prefers_keep_uncertain = "yes"

    assert prefers_keep_uncertain(ByModel(), CLAIRIC) is True
    assert prefers_keep_uncertain(ByModel(), CLEF) is False
    assert prefers_keep_uncertain(ByModel(), CLAIRIC, "attribution") is False
    assert prefers_keep_uncertain(Odd()) is False


# --------------------------------------------------------------------------- #
# The default of a run, and the overrides
# --------------------------------------------------------------------------- #


def test_default_off_with_cloudflare() -> None:
    assert _keeps(_cloudflare()) is False


def test_default_on_with_typesafe() -> None:
    assert _keeps(_jev()) is True


def test_explicit_settings_always_win() -> None:
    assert _keeps(_cloudflare(), RepairConfig(keep_uncertain=True)) is True
    assert _keeps(_jev(), RepairConfig(keep_uncertain=False)) is False
    assert _keeps(_jev(), RepairConfig(keep_uncertain=True)) is True
    assert _keeps(_cloudflare(), RepairConfig(keep_uncertain=False)) is False


# --------------------------------------------------------------------------- #
# Routers and wrappers forward it by (model, kind): ON only where Jev answers the
# attribution AND the repair / re-ask of Clairic
# --------------------------------------------------------------------------- #


def test_a_routing_transport_answers_for_the_model_it_routes() -> None:
    both = RoutingTransport({"clairic": _jev(), "clef": _jev()}, default=_cloudflare())
    assert _keeps(both) is True
    clairic_only = RoutingTransport({"clairic": _jev()}, default=_cloudflare())
    assert _keeps(clairic_only) is False  # the attribution is Cloudflare's
    clef_only = RoutingTransport({"clef": _jev()}, default=_cloudflare())
    assert _keeps(clef_only) is False  # the re-ask is Cloudflare's
    assert _keeps(RoutingTransport({"clef": _cloudflare()}, default=_jev())) is False


def test_a_kind_routing_stack_needs_jev_on_the_attribution_and_the_reask() -> None:
    jev, cf = _jev(), _cloudflare()
    assert _keeps(KindRoutingTransport({}, default=jev)) is True
    # text extraction on Jev, the attribution questions on Cloudflare
    assert _keeps(KindRoutingTransport({"extract": jev}, default=cf)) is False
    assert _keeps(KindRoutingTransport({"clairic": jev}, default=cf)) is False
    # the attribution on Jev, everything Clairic on Cloudflare
    assert _keeps(KindRoutingTransport({"attribution": jev}, default=cf)) is False
    # both on Jev
    assert (
        _keeps(KindRoutingTransport({"clairic": jev, "attribution": jev}, default=cf))
        is True
    )
    # one of the three kinds elsewhere
    assert (
        _keeps(KindRoutingTransport({"reask": cf, "repair@clairic": jev}, default=jev))
        is False
    )
    assert _keeps(KindRoutingTransport({"repair": cf}, default=jev)) is False
    assert _keeps(KindRoutingTransport({"attribution": cf}, default=jev)) is False


def test_a_field_type_stack_answers_for_text_and_the_default_for_clef() -> None:
    jev, cf = _jev(), _cloudflare()
    # Clairic text on Jev but the Clef default is Cloudflare's: no Jev attribution
    assert _keeps(FieldTypeRoutingTransport({"text": jev}, default=cf)) is False
    # Jev is the default, only dates go elsewhere: text and Clef are Jev's
    assert _keeps(FieldTypeRoutingTransport({"date": cf}, default=jev)) is True
    # text on Cloudflare while Jev is the default: the re-ask of a text field is not
    assert _keeps(FieldTypeRoutingTransport({"text": cf}, default=jev)) is False
    helper = FieldTypeRoutingTransport({"text": jev}, default=cf)
    assert prefers_keep_uncertain(helper, CLAIRIC, "reask") is True
    assert prefers_keep_uncertain(helper, CLEF, "attribution") is False


def _combo(**spec: Any) -> Transport:
    return build_combo_transport(
        ComboSpec(**spec), providers={"typesafe": _jev(), "cloudflare": _cloudflare()}
    )


def test_the_four_combos() -> None:
    # (extraction, decision): ON only where Jev answers both
    assert _keeps(_combo(extraction="typesafe", decision="typesafe")) is True
    assert _keeps(_combo(extraction="typesafe", decision="cloudflare")) is False
    assert _keeps(_combo(extraction="cloudflare", decision="typesafe")) is False
    assert _keeps(_combo(extraction="cloudflare", decision="cloudflare")) is False


def test_combos_with_kinds_and_field_types() -> None:
    # Jev reads the text and answers the decisions, Cloudflare reads the dates
    assert (
        _keeps(
            _combo(
                extraction="cloudflare",
                extraction_by_type={"text": "typesafe"},
                decision="typesafe",
            )
        )
        is True
    )
    assert (
        _keeps(
            _combo(
                extraction="typesafe",
                extraction_by_type={"text": "cloudflare"},
                decision="typesafe",
            )
        )
        is False
    )
    # the attribution alone on Jev, over a Jev extraction
    assert (
        _keeps(
            _combo(
                extraction="typesafe",
                decision="cloudflare",
                by_kind={"attribution": "typesafe"},
            )
        )
        is True
    )
    # Jev everywhere but the re-ask
    assert (
        _keeps(
            _combo(
                extraction="typesafe",
                decision="typesafe",
                by_kind={"reask": "cloudflare"},
            )
        )
        is False
    )
    assert (
        _keeps(
            _combo(
                extraction="typesafe",
                decision="typesafe",
                by_kind={"repair@clairic": "cloudflare"},
            )
        )
        is False
    )


def test_a_fallback_answers_for_its_primary() -> None:
    assert _keeps(FallbackTransport(_jev(), _cloudflare())) is True
    assert _keeps(FallbackTransport(_cloudflare(), _jev())) is False


def test_wrappers_forward_it() -> None:
    assert _keeps(CalibratedTransport(_jev(), {"attribution": 1.0})) is True
    assert _keeps(CalibratedTransport(_cloudflare(), {"attribution": 1.0})) is False
    assert _keeps(RecordingTransport(_jev(), "/dev/null")) is True
    assert _keeps(RecordingTransport(_cloudflare(), "/dev/null")) is False
    # a stack of stacks
    nested = KindRoutingTransport(
        {"clairic": CalibratedTransport(FallbackTransport(_jev(), _cloudflare()))},
        default=_jev(),
    )
    assert _keeps(nested) is True


def _line(provider: str, kind: str, model: str, said: bool | None) -> str:
    entry: dict[str, Any] = {
        "key": f"{provider}-{kind}-{model}-{said}",
        "model": f"@cf/cloudflare/{model}",
        "kind": kind,
        "request": {},
        "response": {},
        "provider": provider,
    }
    if said is not None:
        entry["keep_uncertain"] = said
    return json.dumps(entry) + "\n"


def test_a_cassette_replays_the_recorded_decision_per_kind(tmp_path: Any) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        _line("typesafe", "attribution", "clef", True)
        + _line("typesafe", "reask", "clairic", True)
        + _line("typesafe", "repair", "clairic", True)
        # a plain, unlabelled cassette of a Cloudflare extraction under Jev's decisions
        + _line("cloudflare", "extract", "clairic", False)
        + _line("cloudflare", "reask", "clairic", False)
    )
    cassette = CassetteTransport(path)
    assert prefers_keep_uncertain(cassette, CLEF, "attribution") is True
    assert prefers_keep_uncertain(cassette, CLAIRIC, "repair") is True
    # one recorded kind of the family says no: not "all Clairic providers" but by kind
    assert prefers_keep_uncertain(cassette, CLAIRIC, "reask") is False
    assert prefers_keep_uncertain(cassette, CLAIRIC, "extract") is False
    jev_only = CassetteTransport(path, provider="typesafe")
    assert prefers_keep_uncertain(jev_only, CLAIRIC, "reask") is True
    cf_only = CassetteTransport(path, provider="cloudflare")
    assert prefers_keep_uncertain(cf_only, CLAIRIC, "reask") is False
    # the run's decision: Jev answered attribution but the re-ask was mixed
    assert _keeps(cassette) is False
    assert _keeps(jev_only) is True


def test_a_cassette_without_the_recorded_decision_falls_back(tmp_path: Any) -> None:
    empty = CassetteTransport(tmp_path)
    assert prefers_keep_uncertain(empty) is False
    assert prefers_keep_uncertain(CassetteTransport(tmp_path, provider="typesafe"))
    assert not prefers_keep_uncertain(
        CassetteTransport(tmp_path, provider="cloudflare")
    )
    assert (
        prefers_keep_uncertain(CassetteTransport(tmp_path, inner=_jev(), strict=False))
        is True
    )
    assert not prefers_keep_uncertain(
        CassetteTransport(tmp_path, inner=_cloudflare(), strict=False)
    )
    # a recording that predates the field (no decision on its lines): not guessed
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(_line("typesafe", "extract", "clairic", None))
    assert prefers_keep_uncertain(CassetteTransport(legacy)) is False


async def test_a_recorder_writes_the_decision_a_cassette_replays(
    tmp_path: Any,
) -> None:
    from trustcall_system_one import CallMeta as Meta

    class Echo:
        prefers_keep_uncertain = True

        async def run(
            self, model: str, payload: Any, *, meta: Any, timeout: Any = None
        ) -> Any:
            return TransportResponse(body={"ok": True}, provider="typesafe")

    path = tmp_path / "rec.jsonl"
    recorder = RecordingTransport(Echo(), path)  # type: ignore[arg-type]
    await recorder.run(
        "@cf/cloudflare/clef",
        {"model": "clef", "questions": {}},
        meta=Meta(kind="attribution"),
    )
    assert json.loads(path.read_text())["keep_uncertain"] is True
    assert prefers_keep_uncertain(CassetteTransport(path), CLEF, "attribution") is True


# --------------------------------------------------------------------------- #
# End to end: the repair of an insert
# --------------------------------------------------------------------------- #


class Appointment(BaseModel):
    """A clinic appointment reminder."""

    patient_name: str
    clinic: str
    visit_date: date | None = None


REMINDER = "Reminder for Tomas Greer: follow-up visit at Lakeside Family Clinic."


class _PrefersIt:
    """A scripted transport that says it is Jev's (what the engine reads)."""

    prefers_keep_uncertain = True

    def __init__(self, inner: Transport) -> None:
        self._inner = inner

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        return await self._inner.run(model, payload, meta=meta, timeout=timeout)


def _world() -> Script:
    world = Script()
    world.clairic = {
        "patient_name": ca("text", "Tomas Greer", find="Tomas Greer", confidence=0.63),
        "clinic": ca("text", "Lakeside Family Clinic", find="Lakeside Family Clinic"),
        "visit_date": ca("date", None, p_null=0.9),
    }
    world.reask = {"patient_name": ca("text", "Tomas Greer", find="Tomas Greer")}
    world.attribution = {"attr.patient_name": [0.44, 0.46]}  # between drop and keep
    return world


async def test_a_plain_transport_keeps_the_strict_repair_by_default() -> None:
    world = _world()
    extractor = make_extractor([Appointment], world)
    (record,) = (await extractor.ainvoke(REMINDER)).records
    assert not record.valid


async def test_a_transport_that_prefers_it_keeps_the_uncertain_value() -> None:
    world = _world()
    extractor = make_extractor([Appointment], world)
    del extractor
    jev_like = _PrefersIt(world.transport)
    extractor = create_extractor(
        [Appointment], transport=jev_like, verify=VerifyConfig(isolate_decisions=False)
    )
    (record,) = (await extractor.ainvoke(REMINDER)).records
    assert record.valid and record.after is not None
    assert record.after["patient_name"] == "Tomas Greer"
    assert any(
        "attribution_uncertain" in w for w in record.fields["patient_name"].warnings
    )


async def test_the_explicit_setting_beats_the_transport_end_to_end() -> None:
    verify = VerifyConfig(isolate_decisions=False)
    world = _world()
    forced_on = make_extractor(
        [Appointment], world, repair=RepairConfig(keep_uncertain=True)
    )
    (on,) = (await forced_on.ainvoke(REMINDER)).records
    assert on.valid  # a plain transport, but the caller said yes

    world = _world()
    forced_off = create_extractor(
        [Appointment],
        transport=_PrefersIt(world.transport),
        repair=RepairConfig(keep_uncertain=False),
        verify=verify,
    )
    (off,) = (await forced_off.ainvoke(REMINDER)).records
    assert not off.valid  # a Jev-like transport, but the caller said no


async def test_routed_stacks_end_to_end() -> None:
    verify = VerifyConfig(isolate_decisions=False)

    async def valid_on(build: Any) -> bool:
        world = _world()
        extractor = create_extractor(
            [Appointment], transport=build(world.transport), verify=verify
        )
        (record,) = (await extractor.ainvoke(REMINDER)).records
        return record.valid

    # the Jev-like transport answers the attribution and the re-ask
    assert await valid_on(
        lambda base: KindRoutingTransport(
            {"clairic": _PrefersIt(base), "attribution": _PrefersIt(base)},
            default=base,
        )
    )
    assert await valid_on(
        lambda base: FieldTypeRoutingTransport({"date": base}, default=_PrefersIt(base))
    )
    # only the re-ask, or only the attribution: the strict repair stays
    assert not await valid_on(
        lambda base: KindRoutingTransport({"clairic": _PrefersIt(base)}, default=base)
    )
    assert not await valid_on(
        lambda base: KindRoutingTransport(
            {"attribution": _PrefersIt(base)}, default=base
        )
    )
    assert not await valid_on(
        lambda base: FieldTypeRoutingTransport({"text": _PrefersIt(base)}, default=base)
    )
