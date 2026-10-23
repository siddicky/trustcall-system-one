"""``RepairConfig.keep_uncertain`` defaults by the extraction transport.

``None`` (auto) is on when the transport that does the text extraction is TypeSafe
(Jev) and off for every other provider; ``True`` or ``False`` always wins. A router
answers for the transport the TEXT extraction goes to.
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
        def prefers_keep_uncertain(self, model: str) -> bool:
            return model == CLAIRIC

    class Odd:
        prefers_keep_uncertain = "yes"

    assert prefers_keep_uncertain(ByModel(), CLAIRIC) is True
    assert prefers_keep_uncertain(ByModel(), CLEF) is False
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
# Routers and wrappers forward it, for the transport of the TEXT extraction
# --------------------------------------------------------------------------- #


def test_a_routing_transport_answers_for_the_model_it_routes() -> None:
    clairic_on_jev = RoutingTransport({"clairic": _jev()}, default=_cloudflare())
    assert _keeps(clairic_on_jev) is True
    clef_on_jev = RoutingTransport({"clef": _jev()}, default=_cloudflare())
    assert _keeps(clef_on_jev) is False
    jev_default = RoutingTransport({"clef": _cloudflare()}, default=_jev())
    assert _keeps(jev_default) is True


def test_a_kind_routing_stack_with_text_extraction_on_typesafe_turns_on() -> None:
    by_kind = KindRoutingTransport({"extract": _jev()}, default=_cloudflare())
    assert _keeps(by_kind) is True
    by_model = KindRoutingTransport({"clairic": _jev()}, default=_cloudflare())
    assert _keeps(by_model) is True
    everything_but = KindRoutingTransport(
        {"attribution": _cloudflare()}, default=_jev()
    )
    assert _keeps(everything_but) is True


def test_a_kind_routing_stack_with_text_extraction_on_cloudflare_stays_off() -> None:
    # Jev answers the decisions, Cloudflare reads the text
    decisions_only = KindRoutingTransport(
        {"attribution": _jev(), "gate": _jev()}, default=_cloudflare()
    )
    assert _keeps(decisions_only) is False
    extract_on_cf = KindRoutingTransport({"extract": _cloudflare()}, default=_jev())
    assert _keeps(extract_on_cf) is False
    assert _keeps(KindRoutingTransport({}, default=_cloudflare())) is False


def test_a_field_type_stack_whose_text_goes_to_typesafe_turns_it_on() -> None:
    text_on_jev = FieldTypeRoutingTransport({"text": _jev()}, default=_cloudflare())
    assert _keeps(text_on_jev) is True
    rest_on_jev = FieldTypeRoutingTransport({"date": _cloudflare()}, default=_jev())
    assert _keeps(rest_on_jev) is True


def test_a_field_type_stack_whose_text_goes_to_cloudflare_leaves_it_off() -> None:
    dates_on_jev = FieldTypeRoutingTransport(
        {"date": _jev(), "datetime": _jev()}, default=_cloudflare()
    )
    assert _keeps(dates_on_jev) is False
    text_on_cf = FieldTypeRoutingTransport({"text": _cloudflare()}, default=_jev())
    assert _keeps(text_on_cf) is False


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
        {"extract": CalibratedTransport(FallbackTransport(_jev(), _cloudflare()))},
        default=_cloudflare(),
    )
    assert _keeps(nested) is True


def test_a_cassette_settles_it_as_the_live_run_did(tmp_path: Any) -> None:
    empty = CassetteTransport(tmp_path)
    assert prefers_keep_uncertain(empty) is False
    assert prefers_keep_uncertain(CassetteTransport(tmp_path, provider="typesafe"))
    assert not prefers_keep_uncertain(
        CassetteTransport(tmp_path, provider="cloudflare")
    )
    # no label: what the clairic recordings say, then the transport behind a miss
    miss_to_jev = CassetteTransport(tmp_path, inner=_jev(), strict=False)
    assert prefers_keep_uncertain(miss_to_jev) is True
    miss_to_cf = CassetteTransport(tmp_path, inner=_cloudflare(), strict=False)
    assert prefers_keep_uncertain(miss_to_cf) is False


def test_a_cassette_with_recordings_follows_who_answered(tmp_path: Any) -> None:
    def line(provider: str) -> str:
        entry = {
            "key": "k" + provider,
            "model": "@cf/cloudflare/clairic",
            "request": {},
            "response": {},
            "provider": provider,
        }
        return json.dumps(entry) + "\n"

    jev_only = tmp_path / "jev.jsonl"
    jev_only.write_text(line("typesafe"))
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(line("typesafe") + line("cloudflare"))
    assert prefers_keep_uncertain(CassetteTransport(jev_only)) is True
    assert prefers_keep_uncertain(CassetteTransport(mixed)) is False
    # the recordings win over an inner transport that only serves misses
    assert (
        prefers_keep_uncertain(CassetteTransport(jev_only, inner=_cloudflare())) is True
    )


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

    # text extraction goes to the Jev-like transport
    assert await valid_on(
        lambda base: KindRoutingTransport({"extract": _PrefersIt(base)}, default=base)
    )
    assert await valid_on(
        lambda base: FieldTypeRoutingTransport({"text": _PrefersIt(base)}, default=base)
    )
    # only the decisions go to it: the text extraction is elsewhere
    assert not await valid_on(
        lambda base: KindRoutingTransport(
            {"attribution": _PrefersIt(base)}, default=base
        )
    )
    assert not await valid_on(
        lambda base: FieldTypeRoutingTransport({"date": _PrefersIt(base)}, default=base)
    )
