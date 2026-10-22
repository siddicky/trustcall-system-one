"""CalibratedTransport and its logit shifts: monotone, renormalised, a no-op at 0."""

import asyncio
import math
from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CalibratedTransport,
    CallMeta,
    TransportResponse,
    UncertainOutcomeError,
    create_extractor,
)
from trustcall_system_one._calibrate import calibrate_body, shift_probability
from trustcall_system_one._wire import parse_clef
from trustcall_system_one.testing import (
    choice,
    clef_response,
    noul,
    score,
)
from trustcall_system_one.transport import max_questions, state_token_budget

SHIFTS = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]


def answers(body: Mapping[str, Any]) -> Mapping[str, Any]:
    return body["answers"]  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# The shift of one probability
# --------------------------------------------------------------------------- #


def test_shift_zero_is_the_identity_and_the_ends_stay_put() -> None:
    for p in (0.0, 1e-9, 0.2, 0.5, 0.97, 1.0):
        assert shift_probability(p, 0.0) == p
    assert shift_probability(0.0, 2.0) == 0.0
    assert shift_probability(1.0, -2.0) == 1.0


def test_a_logit_shift_of_one_half_matches_the_sigmoid() -> None:
    assert shift_probability(0.5, 1.0) == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert shift_probability(0.5, -1.0) == pytest.approx(1 / (1 + math.exp(1.0)))
    # 0.8 has odds 4; shifted by ln 2 they are 8
    assert shift_probability(0.8, math.log(2)) == pytest.approx(8 / 9)


@given(
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.floats(-10.0, 10.0, allow_nan=False),
)
def test_a_shift_is_monotone_in_the_probability(a: float, b: float, s: float) -> None:
    low, high = sorted((a, b))
    assert shift_probability(low, s) <= shift_probability(high, s)
    assert 0.0 <= shift_probability(low, s) <= 1.0


@given(st.floats(1e-6, 1 - 1e-6), st.floats(-5.0, 5.0), st.floats(-5.0, 5.0))
def test_a_larger_shift_never_lowers_a_probability(
    p: float, s: float, t: float
) -> None:
    low, high = sorted((s, t))
    assert shift_probability(p, low) <= shift_probability(p, high) + 1e-12


# --------------------------------------------------------------------------- #
# Bodies: noul, choice, score
# --------------------------------------------------------------------------- #


def body_of(**by_id: dict[str, Any]) -> dict[str, Any]:
    return clef_response(by_id)


def test_noul_is_shifted_and_monotone_across_answers() -> None:
    body = body_of(a=noul(0.1), b=noul(0.4), c=noul(0.4), d=noul(0.9))
    for shift in SHIFTS:
        out, changed = calibrate_body(body, shift)
        got = [answers(out)[q]["noul"] for q in "abcd"]
        assert got == sorted(got)  # the order of the answers is kept
        assert got[1] == got[2]
        assert changed == (0 if shift == 0 else 4)
    up, _ = calibrate_body(body, 1.0)
    down, _ = calibrate_body(body, -1.0)
    assert answers(up)["b"]["noul"] > 0.4 > answers(down)["b"]["noul"]
    parse_clef(up)  # still a valid Clef response


def test_shift_zero_returns_the_very_same_body() -> None:
    body = body_of(a=noul(0.3), b=choice("x", {"x": 0.6, "y": 0.4}))
    out, changed = calibrate_body(body, 0.0)
    assert out is body and changed == 0
    transport_body, _ = calibrate_body(body, 1e-300)
    assert answers(transport_body)["a"]["noul"] == pytest.approx(0.3)


def test_the_input_body_is_never_modified() -> None:
    body = body_of(a=noul(0.3), b=choice("x", {"x": 0.6, "y": 0.4}))
    before = repr(body)
    calibrate_body(body, 1.5)
    assert repr(body) == before


@pytest.mark.parametrize("shift", SHIFTS)
def test_a_choice_is_renormalised_and_its_pick_follows_the_odds(shift: float) -> None:
    body = body_of(
        q=choice("a", {"a": 0.6, "b": 0.3, "c": 0.1}),
        tie=choice("a", {"a": 0.45, "b": 0.45, "c": 0.1}),
    )
    out, _ = calibrate_body(body, shift)
    for qid in ("q", "tie"):
        answer = answers(out)[qid]
        probs = answer["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in probs.values())
        # `choice` is the most probable option, and `confidence` is its probability
        assert probs[answer["choice"]] == max(probs.values())
        assert answer["confidence"] == pytest.approx(max(probs.values()))
    parse_clef(out)


def test_a_positive_shift_makes_the_leader_surer_and_keeps_the_other_proportions() -> (
    None
):
    body = body_of(q=choice("a", {"a": 0.6, "b": 0.3, "c": 0.1}))
    up = answers(calibrate_body(body, 1.0)[0])["q"]
    down = answers(calibrate_body(body, -1.0)[0])["q"]
    assert up["choice"] == "a" and up["probabilities"]["a"] > 0.6
    assert down["probabilities"]["a"] < 0.6
    # b and c share what is left in the proportion 3:1, as before
    for answer in (up, down):
        p = answer["probabilities"]
        assert p["b"] / p["c"] == pytest.approx(3.0)


def test_a_leader_that_falls_below_the_runner_up_hands_over_the_pick() -> None:
    body = body_of(q=choice("a", {"a": 0.5, "b": 0.45, "c": 0.05}))
    down = answers(calibrate_body(body, -1.0)[0])["q"]
    assert down["choice"] == "b"  # the pick is always the argmax of what it reports
    assert down["probabilities"]["b"] == max(down["probabilities"].values())
    assert down["confidence"] == pytest.approx(down["probabilities"]["b"])


def test_a_confidence_that_was_not_the_top_probability_is_shifted_itself() -> None:
    body = body_of(q=choice("a", {"a": 0.7, "b": 0.3}, confidence=0.44))
    out = answers(calibrate_body(body, 1.0)[0])["q"]
    assert out["confidence"] == pytest.approx(shift_probability(0.44, 1.0))
    assert out["confidence"] != pytest.approx(out["probabilities"]["a"])


def test_probabilities_that_do_not_sum_to_one_are_normalised_first() -> None:
    body = body_of(q=choice("a", {"a": 1.2 / 2, "b": 0.6 / 2}))
    out = answers(calibrate_body(body, 0.5)[0])["q"]
    assert sum(out["probabilities"].values()) == pytest.approx(1.0)


def test_a_score_is_renormalised_and_recomputed_as_the_weighted_level() -> None:
    body = body_of(
        s=score(1.3, {"0": 0.1, "1": 0.5, "2": 0.4}, legend={"0": "no", "2": "yes"})
    )
    for shift in SHIFTS:
        out = answers(calibrate_body(body, shift)[0])["s"]
        probs = out["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0)
        assert out["score"] == pytest.approx(sum(int(k) * v for k, v in probs.items()))
        assert out["legend"] == {"0": "no", "2": "yes"}
        assert out["confidence"] == pytest.approx(max(probs.values()))
    up = answers(calibrate_body(body, 2.0)[0])["s"]["probabilities"]
    assert up["1"] > 0.5  # the leading level is surer
    parse_clef(calibrate_body(body, 2.0)[0])


def test_what_is_certain_hard_or_malformed_is_left_alone() -> None:
    hard = {**noul(0.8), "hard": True}
    body = body_of(
        sure=noul(1.0),
        no=noul(0.0),
        hard=hard,
        point=choice("a", {"a": 1.0, "b": 0.0}),
        junk={"type": "noul", "noul": "high"},
    )
    out, changed = calibrate_body(body, 1.5)
    assert changed == 0 and out is body
    mixed = body_of(h=hard, n=noul(0.5))
    out, changed = calibrate_body(mixed, 1.5)
    assert changed == 1 and answers(out)["h"] == hard


@given(
    st.lists(st.floats(0.01, 1.0), min_size=2, max_size=6),
    st.floats(-6.0, 6.0),
)
def test_any_choice_stays_a_distribution_with_a_consistent_pick(
    weights: list[float], shift: float
) -> None:
    total = sum(weights)
    probs = {f"o{i}": w / total for i, w in enumerate(weights)}
    top = max(probs, key=lambda k: probs[k])
    out = answers(calibrate_body(body_of(q=choice(top, probs)), shift)[0])["q"]
    assert sum(out["probabilities"].values()) == pytest.approx(1.0)
    assert out["probabilities"][out["choice"]] == max(out["probabilities"].values())


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #

PAYLOAD: dict[str, Any] = {"model": "clef", "state": "x", "questions": {}}


class Scripted:
    """A transport that answers with a fixed body and says what it was asked."""

    def __init__(self, body: dict[str, Any], provider: str | None = None) -> None:
        self.body = body
        self.provider = provider
        self.calls: list[CallMeta] = []
        self.closed: list[str] = []
        self.max_questions = 17
        self.state_token_budget = 1234
        self.failure: Exception | None = None

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.calls.append(meta)
        if self.failure is not None:
            raise self.failure
        return TransportResponse(
            body=self.body,
            status_code=200,
            provider=self.provider,
            served_model="m-1",
            warnings=("kept",),
        )

    def sensitive_values(self) -> tuple[str, ...]:
        return ("sekret",)

    def classify_failure(self, exc: Exception) -> str | None:
        return "unavailable"

    async def aclose(self) -> None:
        self.closed.append("aclose")

    def close(self) -> None:
        self.closed.append("close")


def run(transport: Any, kind: str = "attribution", model: str = CLEF) -> Any:
    return asyncio.run(
        transport.run(model, PAYLOAD, meta=CallMeta(kind=kind))  # type: ignore[arg-type]
    )


def test_only_the_listed_kinds_are_shifted_and_the_default_is_no_shift() -> None:
    body = body_of(a=noul(0.3))
    inner = Scripted(body)
    calibrated = CalibratedTransport(inner, {"attribution": 1.0})
    assert answers(run(calibrated, "attribution").body)["a"]["noul"] > 0.3
    assert run(calibrated, "gate").body is body  # no shift for the kind: untouched
    assert run(CalibratedTransport(inner), "attribution").body is body
    assert (
        run(CalibratedTransport(inner, {"attribution": 0}), "attribution").body is body
    )
    assert calibrated.shift_for("attribution") == 1.0
    assert calibrated.shift_for("gate") == 0.0


def test_the_response_says_it_was_calibrated_and_keeps_everything_else() -> None:
    inner = Scripted(body_of(a=noul(0.3)), provider="typesafe")
    reply = run(CalibratedTransport(inner, {"attribution": 0.5}))
    assert reply.warnings == (
        "kept",
        "calibration_applied: attribution logit shift +0.5",
    )
    assert (reply.provider, reply.served_model, reply.status_code) == (
        "typesafe",
        "m-1",
        200,
    )
    untouched = run(
        CalibratedTransport(Scripted(body_of(s=noul(1.0))), {"gate": 1.0}), "gate"
    )
    assert untouched.warnings == ("kept",)  # nothing changed: nothing to say


def test_clairic_answers_are_never_touched() -> None:
    body = {
        "model": "clairic",
        "answers": {"f": {"type": "text", "value": "x", "p_null": 0.3}},
        "truncated": False,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    reply = run(
        CalibratedTransport(Scripted(body), {"extract": 2.0}), "extract", CLAIRIC
    )
    assert reply.body is body


def test_a_shift_can_depend_on_who_answered() -> None:
    body = body_of(a=noul(0.3))
    calibrated = CalibratedTransport(
        Scripted(body, provider="decisions"),
        {"attribution": 1.0},
        by_provider={"decisions": {"attribution": -1.0}, "typesafe": {"gate": 2.0}},
    )
    assert answers(run(calibrated).body)["a"]["noul"] < 0.3  # the provider's shift wins
    other = CalibratedTransport(
        Scripted(body, provider="cloudflare"),
        {"attribution": 1.0},
        by_provider={"decisions": {"attribution": -1.0}},
    )
    assert answers(run(other).body)["a"]["noul"] > 0.3  # then the general one


def test_the_wrapper_forwards_what_the_engine_reads_off_a_transport() -> None:
    inner = Scripted(body_of(), provider="p")
    wrapper = CalibratedTransport(inner, {"gate": 1.0})
    assert max_questions(wrapper, CLEF) == 17
    assert state_token_budget(wrapper, CLEF) == 1234
    assert wrapper.provider == "p"
    assert wrapper.sensitive_values() == ("sekret",)
    assert wrapper.classify_failure(RuntimeError("x")) == "unavailable"
    assert wrapper.serves(CLEF) is True
    asyncio.run(wrapper.aclose())
    wrapper.close()
    assert inner.closed == ["aclose", "close"]


def test_errors_and_partial_usage_pass_through_untouched() -> None:
    inner = Scripted(body_of())
    boom = UncertainOutcomeError("may have been charged")
    boom.partial_usage = {"input_tokens": 3}  # type: ignore[attr-defined]
    inner.failure = boom
    with pytest.raises(UncertainOutcomeError) as caught:
        run(CalibratedTransport(inner, {"attribution": 1.0}))
    assert caught.value is boom
    assert len(inner.calls) == 1  # never sent again


def test_shifts_are_validated() -> None:
    with pytest.raises(ValueError, match="unknown call kind"):
        CalibratedTransport(Scripted({}), {"nope": 1.0})
    with pytest.raises(ValueError, match="within"):
        CalibratedTransport(Scripted({}), {"gate": 99.0})
    with pytest.raises(ValueError, match="within"):
        CalibratedTransport(Scripted({}), {"gate": float("nan")})
    with pytest.raises(ValueError, match="within"):
        CalibratedTransport(Scripted({}), {"gate": True})
    with pytest.raises(ValueError, match="unknown call kind"):
        CalibratedTransport(Scripted({}), by_provider={"decisions": {"nope": 1.0}})


def test_a_calibrated_run_reports_it_and_a_zero_shift_changes_nothing() -> None:
    from evals.cases import CASES

    case = next(c for c in CASES if c.id == "colleague_bleed")
    assert case.world is not None

    async def go(shifts: dict[str, float]) -> Any:
        assert case.world is not None
        transport = CalibratedTransport(case.world().transport(), shifts)
        extractor = create_extractor(case.schemas(), transport=transport)
        return await extractor.ainvoke(case.text, existing=case.existing)

    plain = asyncio.run(go({}))
    zero = asyncio.run(go({"attribution": 0.0}))
    moved = asyncio.run(go({"attribution": 0.5}))
    assert [str(r.patches) for r in plain.records] == [
        str(r.patches) for r in zero.records
    ]
    assert not [w for w in plain.warnings if w.startswith("calibration_applied")]
    assert any(w.startswith("calibration_applied: attribution") for w in moved.warnings)


def test_a_shift_beyond_three_logits_is_refused() -> None:
    CalibratedTransport(Scripted({}), {"attribution": 3.0})
    CalibratedTransport(Scripted({}), {"attribution": -3.0})
    for bad in (3.5, -3.5, 10.0):
        with pytest.raises(ValueError, match=r"within \+-3"):
            CalibratedTransport(Scripted({}), {"attribution": bad})
        with pytest.raises(ValueError, match=r"within \+-3"):
            CalibratedTransport(Scripted({}), by_provider={"typesafe": {"gate": bad}})
    from trustcall_system_one.combos import ComboSpec

    with pytest.raises(ValueError, match="within"):
        ComboSpec.model_validate(
            {"decision": "typesafe", "calibration": {"typesafe": {"gate": 3.5}}}
        )


def test_an_answer_that_names_no_provider_is_not_silently_uncalibrated() -> None:
    """The reviewer's probe P5."""
    body = body_of(q=noul(0.6))
    unlabelled = Scripted(body, provider=None)
    only_by_provider = CalibratedTransport(
        unlabelled, by_provider={"decisions": {"gate": 1.5}}
    )
    reply = run(only_by_provider, "gate")
    assert answers(reply.body)["q"]["noul"] == 0.6  # nothing could be applied
    assert any(
        w.startswith("calibration_skipped: a gate answer") for w in reply.warnings
    )
    # no warning for a kind that has no shift, or when a general shift applies
    assert not [w for w in run(only_by_provider, "presence").warnings if "skipped" in w]
    both = CalibratedTransport(
        unlabelled, {"gate": 0.5}, by_provider={"decisions": {"gate": 1.5}}
    )
    assert answers(run(both, "gate").body)["q"]["noul"] > 0.6
    # a labelled answer is calibrated, and says nothing about skipping
    labelled = CalibratedTransport(
        Scripted(body, provider="decisions"), by_provider={"decisions": {"gate": 1.5}}
    )
    reply = run(labelled, "gate")
    assert answers(reply.body)["q"]["noul"] > 0.6
    assert not [w for w in reply.warnings if "skipped" in w]
