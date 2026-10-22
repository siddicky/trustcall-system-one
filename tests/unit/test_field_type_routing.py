"""FieldTypeRoutingTransport: split a Clairic request by field type, merge exactly."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    BadRequestError,
    CallMeta,
    FieldTypeRoutingTransport,
    KindRoutingTransport,
    ModelResponseError,
    RateLimitError,
    TransportResponse,
    UncertainOutcomeError,
    create_extractor,
)
from trustcall_system_one._errors import is_size_rejection, may_have_run
from trustcall_system_one._wire import ClairicUsage, parse_clairic
from trustcall_system_one.combos import ComboSpec, build_combo_transport
from trustcall_system_one.transport import classify_failure, max_questions

IMAGE = "data:image/png;base64,iVBORw0KGgo="
STATE = "<input>\nAda Lovelace was born on 1815-12-10 and has 2 sons.\n</input>"
FIELDS: dict[str, dict[str, Any]] = {
    "name": {"type": "text"},
    "born": {"type": "date"},
    "sons": {"type": "number", "integer": True},
    "seen": {"type": "datetime"},
    "tags": {"type": "list", "items": "text"},
    "dates": {"type": "list", "items": "date"},
}
PAYLOAD: dict[str, Any] = {
    "model": "clairic",
    "state": STATE,
    "fields": FIELDS,
    "reference_date": "2026-01-02",
    "options": {"return_state_text": True},
}
META = CallMeta(kind="extract", schema="Person")


def answer_for(who: str, name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    kind = spec["type"]
    at = STATE.index("Ada")
    return {
        "type": kind,
        "value": f"{who}:{name}",
        "p_null": 0.01,
        "confidence": 0.9,
        "span": [at, at + 3],
    }


class Provider:
    """A scripted Clairic provider: answers every field it is asked."""

    def __init__(
        self,
        name: str,
        *,
        tokens: int = 10,
        truncated: bool = False,
        state_text: str | None = None,
        served: str | None = None,
        cost: float | None = None,
        serves: bool = True,
        images: bool = False,
        limit: int | None = None,
    ) -> None:
        self.provider = name
        self.tokens = tokens
        self.truncated = truncated
        self.state_text = state_text
        self.served = served
        self.cost = cost
        self._serves = serves
        self._images = images
        if limit is not None:
            self.max_questions = limit
        self.calls: list[Mapping[str, Any]] = []
        self.closed: list[str] = []
        self.gate: Callable[[], Awaitable[None]] | None = None
        self.failure: BaseException | None = None

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
        self.calls.append(payload)
        if "fields" not in payload:  # a Clef request: only who answered matters
            return TransportResponse(body={"served_by": self.provider})
        if self.gate is not None:
            await self.gate()
        if self.failure is not None:
            raise self.failure
        usage: dict[str, Any] = {
            "input_tokens": self.tokens,
            "output_tokens": 1,
            "passes": 1,
        }
        if self.cost is not None:
            usage["cost"] = self.cost
        body: dict[str, Any] = {
            "model": "clairic",
            "answers": {
                n: answer_for(self.provider, n, s) for n, s in payload["fields"].items()
            },
            "truncated": self.truncated,
            "usage": usage,
        }
        text = self.state_text if self.state_text is not None else payload["state"]
        if (payload.get("options") or {}).get("return_state_text"):
            body["state_text"] = text
        return TransportResponse(
            body=body,
            retries=1,
            duration_ms=5.0,
            status_code=200,
            request_id=f"req-{self.provider}",
            provider=self.provider,
            served_model=self.served,
        )

    async def aclose(self) -> None:
        self.closed.append("aclose")

    def close(self) -> None:
        self.closed.append("close")


def run(transport: Any, payload: Mapping[str, Any] = PAYLOAD) -> TransportResponse:
    reply: TransportResponse = asyncio.run(transport.run(CLAIRIC, payload, meta=META))
    return reply


def router(jev: Provider, cf: Provider) -> FieldTypeRoutingTransport:
    return FieldTypeRoutingTransport(
        {"date": jev, "datetime": jev, "number": jev}, default=cf
    )


# --------------------------------------------------------------------------- #
# Partitioning and merging
# --------------------------------------------------------------------------- #


def test_each_type_goes_to_its_provider_and_the_answers_come_back_in_field_order() -> (
    None
):
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    reply = run(router(jev, cf))
    assert list(jev.calls[0]["fields"]) == ["born", "sons", "seen"]
    assert list(cf.calls[0]["fields"]) == ["name", "tags", "dates"]
    answers = reply.body["answers"]
    assert list(answers) == list(FIELDS)  # the order they were asked in
    assert {n: a["value"] for n, a in answers.items()} == {
        "name": "cloudflare:name",
        "born": "typesafe:born",
        "sons": "typesafe:sons",
        "seen": "typesafe:seen",
        "tags": "cloudflare:tags",
        "dates": "cloudflare:dates",
    }
    parse_clairic(reply.body)  # a valid Clairic response


def test_every_partition_gets_the_same_state_and_the_spans_stay_exact() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    reply = run(router(jev, cf))
    for sent in (jev.calls[0], cf.calls[0]):
        assert sent["state"] == STATE
        assert sent["reference_date"] == "2026-01-02"
        assert sent["options"] == {"return_state_text": True}
        assert sent["model"] == "clairic"
    at = STATE.index("Ada")
    assert all(a["span"] == [at, at + 3] for a in reply.body["answers"].values())
    assert reply.body["state_text"] == STATE


def test_images_and_options_reach_every_partition() -> None:
    jev, cf = Provider("typesafe", images=True), Provider("cloudflare", images=True)
    payload = {**PAYLOAD, "images": [IMAGE]}
    run(router(jev, cf), payload)
    assert jev.calls[0]["images"] == [IMAGE] and cf.calls[0]["images"] == [IMAGE]


def test_usage_is_summed_and_the_response_says_who_answered() -> None:
    jev = Provider("typesafe", tokens=30, cost=0.002, served="jev-1")
    cf = Provider("cloudflare", tokens=70, cost=0.001)
    reply = run(router(jev, cf))
    assert reply.body["usage"] == {
        "input_tokens": 100,
        "output_tokens": 2,
        "passes": 2,
        "cost": pytest.approx(0.003),
    }
    assert reply.provider == "cloudflare+typesafe"  # in order of the first field
    assert reply.served_model is None  # two models answered
    assert reply.retries == 2 and reply.request_id == "req-cloudflare"
    partitions = reply.body["routing"]["partitions"]
    assert [(p["provider"], p["served_model"], p["fields"]) for p in partitions] == [
        ("cloudflare", None, 3),
        ("typesafe", "jev-1", 3),
    ]
    assert [p["usage"]["input_tokens"] for p in partitions] == [70, 30]


def test_two_types_on_one_transport_share_a_request_and_list_items_beat_list() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    both = FieldTypeRoutingTransport({"list:date": jev, "list": cf}, default=cf)
    run(both)
    assert list(jev.calls[0]["fields"]) == ["dates"]
    assert list(cf.calls[0]["fields"]) == ["name", "born", "sons", "seen", "tags"]
    assert len(jev.calls) == len(cf.calls) == 1


def test_a_request_for_one_provider_goes_through_unchanged() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    only_dates = {**PAYLOAD, "fields": {"born": FIELDS["born"], "seen": FIELDS["seen"]}}
    reply = run(router(jev, cf), only_dates)
    assert jev.calls == [only_dates] and cf.calls == []
    assert jev.calls[0] is only_dates
    assert reply.provider == "typesafe" and "routing" not in reply.body


def test_a_clef_request_is_never_split() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    clef = {"model": "clef", "state": "x", "questions": {"q": {"type": "noul"}}}
    asyncio.run(router(jev, cf).run(CLEF, clef, meta=CallMeta(kind="gate")))
    assert cf.calls == [clef] and jev.calls == []


def test_truncated_is_true_when_any_partition_was_cut() -> None:
    jev = Provider("typesafe", truncated=True)
    assert run(router(jev, Provider("cloudflare"))).body["truncated"] is True
    assert run(router(Provider("a"), Provider("b"))).body["truncated"] is False


def test_state_text_is_the_prefix_every_partition_read() -> None:
    cut = STATE[:30]
    jev = Provider("typesafe", truncated=True, state_text=cut)
    reply = run(router(jev, Provider("cloudflare")))
    assert reply.body["state_text"] == cut
    assert reply.body["truncated"] is True  # so the engine asks again on chunks


def test_state_text_is_left_out_when_it_was_not_asked_for() -> None:
    plain = {k: v for k, v in PAYLOAD.items() if k != "options"}
    reply = run(router(Provider("typesafe"), Provider("cloudflare")), plain)
    assert "state_text" not in reply.body


def test_partitions_that_disagree_on_the_text_lose_their_spans_with_a_warning() -> None:
    cf = Provider("cloudflare", state_text="something else entirely")
    reply = run(router(Provider("typesafe"), cf))
    assert reply.body["state_text"] == "something else entirely"  # the first one's
    answers = reply.body["answers"]
    assert "span" in answers["name"]  # the first partition keeps its spans
    assert "span" not in answers["born"]  # the other's no longer index this text
    assert any(w.startswith("field_type_routing:") for w in reply.warnings)


def test_a_partition_without_answers_is_a_model_response_error_without_text() -> None:
    class Broken(Provider):
        async def run(self, *args: Any, **kwargs: Any) -> TransportResponse:
            await super().run(*args, **kwargs)
            return TransportResponse(body={"model": "clairic", "answers": None})

    with pytest.raises(ModelResponseError) as caught:
        run(router(Broken("typesafe"), Provider("cloudflare")))
    assert "Ada" not in str(caught.value)


def test_warnings_of_the_partitions_are_kept_once() -> None:
    class Noting(Provider):
        async def run(self, *args: Any, **kwargs: Any) -> TransportResponse:
            import dataclasses

            reply = await super().run(*args, **kwargs)
            return dataclasses.replace(reply, warnings=("narrowed",))

    reply = run(router(Noting("typesafe"), Noting("cloudflare")))
    assert reply.warnings == ("narrowed",)


# --------------------------------------------------------------------------- #
# What cannot be served
# --------------------------------------------------------------------------- #


def test_nothing_is_sent_when_a_partition_cannot_be_served() -> None:
    jev, cf = Provider("typesafe", serves=False), Provider("cloudflare")
    with pytest.raises(BadRequestError) as caught:
        run(router(jev, cf))
    assert jev.calls == [] and cf.calls == []
    assert "typesafe" in str(caught.value) and "Ada" not in str(caught.value)
    assert not may_have_run(caught.value)


def test_images_that_one_partition_cannot_take_go_whole_to_the_default() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare", images=True)
    pictured = {**PAYLOAD, "images": [IMAGE]}
    reply = run(router(jev, cf), pictured)
    assert jev.calls == [] and len(cf.calls) == 1
    assert reply.rerouted_from == "typesafe"
    assert list(cf.calls[0]["fields"]) == list(FIELDS)  # one request, not split


def test_serves_says_whether_every_partition_can_be_served() -> None:
    jev, cf = Provider("typesafe", serves=False), Provider("cloudflare")
    assert router(jev, cf).serves(CLAIRIC, PAYLOAD) is False
    assert router(cf, cf).serves(CLAIRIC, PAYLOAD) is True
    assert router(jev, cf).serves(CLEF) is True  # the default's


def test_keys_are_validated() -> None:
    with pytest.raises(ValueError, match="cannot route field type 'color'"):
        FieldTypeRoutingTransport({"color": Provider("x")}, default=Provider("y"))
    with pytest.raises(ValueError, match="cannot route"):
        FieldTypeRoutingTransport({"list:color": Provider("x")}, default=Provider("y"))


# --------------------------------------------------------------------------- #
# Failures: a part answered and another failed
# --------------------------------------------------------------------------- #


def test_a_failure_after_another_partition_answered_is_uncertain_with_its_usage() -> (
    None
):
    jev, cf = Provider("typesafe", tokens=40, served="jev-1"), Provider("cloudflare")
    release = asyncio.Event()

    async def after_jev() -> None:
        await release.wait()

    async def jev_answered() -> None:
        release.set()

    cf.gate = after_jev  # cloudflare fails only once jev has answered
    jev.gate = jev_answered
    cf.failure = BadRequestError("too large", status_code=413)
    with pytest.raises(BadRequestError) as caught:
        run(router(jev, cf))
    error = caught.value
    assert error.maybe_sent is True
    assert isinstance(error.partial_usage, ClairicUsage)  # type: ignore[attr-defined]
    assert error.partial_usage.input_tokens == 40  # type: ignore[attr-defined]
    assert error.served_model == "jev-1"
    assert may_have_run(error)
    assert not is_size_rejection(error)  # the engine neither contains nor resends it
    assert error.provider == "cloudflare"


def test_a_refusal_after_another_partition_answered_is_classified_uncertain() -> None:
    jev, cf = Provider("typesafe", tokens=40), Provider("cloudflare")
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    async def done() -> None:
        release.set()

    cf.gate, jev.gate = wait, done
    cf.failure = RateLimitError("429")
    with pytest.raises(RateLimitError) as caught:
        run(router(jev, cf))
    assert classify_failure(caught.value) == "uncertain"  # not "unavailable": jev ran


def test_the_failed_partitions_own_partial_usage_is_added_to_the_answered_ones() -> (
    None
):
    jev, cf = Provider("typesafe", tokens=40), Provider("cloudflare")
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    async def done() -> None:
        release.set()

    cf.gate, jev.gate = wait, done
    boom = UncertainOutcomeError("timed out")
    boom.partial_usage = ClairicUsage(input_tokens=5, output_tokens=2, passes=1)  # type: ignore[attr-defined]
    cf.failure = boom
    with pytest.raises(UncertainOutcomeError) as caught:
        run(router(jev, cf))
    usage = caught.value.partial_usage  # type: ignore[attr-defined]
    assert (usage.input_tokens, usage.output_tokens) == (45, 3)


def test_a_failure_while_the_other_partition_is_in_flight_is_uncertain() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    forever = asyncio.Event()

    async def stall() -> None:
        await forever.wait()

    jev.gate = stall
    cf.failure = BadRequestError("refused")
    with pytest.raises(BadRequestError) as caught:
        run(router(jev, cf))
    error = caught.value
    assert error.maybe_sent is True and error.cancelled_siblings == 1  # type: ignore[attr-defined]
    assert "cancelled" in str(error)
    assert getattr(error, "partial_usage", None) is None
    assert not is_size_rejection(error)


def test_a_sibling_that_may_have_run_is_the_error_that_is_raised() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    jev.failure = UncertainOutcomeError("timed out")
    cf.failure = BadRequestError("refused")
    with pytest.raises(Exception) as caught:
        run(router(jev, cf))
    assert may_have_run(caught.value)
    assert isinstance(caught.value, UncertainOutcomeError)


def test_nothing_is_ever_resent() -> None:
    jev, cf = Provider("typesafe"), Provider("cloudflare")
    jev.failure = UncertainOutcomeError("timed out")
    with pytest.raises(UncertainOutcomeError):
        run(router(jev, cf))
    assert len(jev.calls) == 1 and len(cf.calls) <= 1


def test_the_failed_partition_decides_how_the_failure_is_classified() -> None:
    class Sorting(Provider):
        def classify_failure(self, exc: Exception) -> str | None:
            return "mine"

    jev, cf = Sorting("typesafe"), Provider("cloudflare")
    jev.failure = BadRequestError("no")
    transport = router(jev, cf)
    with pytest.raises(BadRequestError) as caught:
        run(transport)
    assert transport.classify_failure(caught.value) == "mine"


# --------------------------------------------------------------------------- #
# Forwarding
# --------------------------------------------------------------------------- #


def test_it_forwards_limits_and_closes_each_transport_once() -> None:
    jev, cf = Provider("typesafe", limit=7), Provider("cloudflare", limit=40)
    transport = router(jev, cf)
    assert max_questions(transport, CLEF) == 40  # Clef is not split: the default's
    asyncio.run(transport.aclose())
    transport.close()
    assert jev.closed == ["aclose", "close"] == cf.closed


def test_it_sits_inside_a_kind_router_and_both_are_transparent() -> None:
    jev, cf = Provider("typesafe", limit=7), Provider("cloudflare", limit=40)
    stack = KindRoutingTransport({"clairic": router(jev, cf)}, default=cf)
    reply = asyncio.run(stack.run(CLAIRIC, PAYLOAD, meta=META))
    assert list(reply.body["answers"]) == list(FIELDS)
    assert max_questions(stack, CLEF) == 40  # a Clairic route does not narrow Clef


# --------------------------------------------------------------------------- #
# The engine on top of it
# --------------------------------------------------------------------------- #


def test_the_engine_files_each_providers_tokens_under_its_own_record() -> None:
    from evals.cases import CASES

    case = next(c for c in CASES if c.id == "invoice_contacts")
    assert case.world is not None
    spec = ComboSpec.model_validate(
        {
            "extraction": "cloudflare",
            "extraction_by_type": {"text": "typesafe"},
            "decision": "cloudflare",
        }
    )

    async def go() -> Any:
        assert case.world is not None
        transport = build_combo_transport(
            spec,
            providers={
                "cloudflare": case.world().transport(),
                "typesafe": case.world().transport(),
            },
        )
        extractor = create_extractor(case.schemas(), transport=transport)
        return await extractor.ainvoke(case.text, existing=case.existing)

    result = asyncio.run(go())
    clairic = [c for c in result.calls if c.model.endswith("clairic")]
    assert {c.provider for c in clairic} == {"cloudflare", "typesafe"}
    split = [c for c in clairic if c.kind == "extract"]
    assert split and all(c.error is None for c in split)
    # the same number of tokens as one request would have used, filed twice
    total = sum(c.usage["input_tokens"] for c in clairic)
    assert result.usage.by_model[CLAIRIC].input_tokens == total


# --------------------------------------------------------------------------- #
# The engine's call log, and a split call that fails
# --------------------------------------------------------------------------- #


def mixed_fields() -> dict[str, Any]:
    from trustcall_system_one._wire import ClairicField

    return {
        "name": ClairicField(type="text", instructions="the name"),
        "born": ClairicField(type="date", instructions="the birth date"),
    }


def test_the_engine_logs_each_partition_under_its_own_provider_and_model() -> None:
    from trustcall_system_one._render import render_clairic
    from trustcall_system_one._run import Run

    jev = Provider("typesafe", tokens=30, served="jev-1.14.0")
    cf = Provider("cloudflare", tokens=70)
    run = Run(transport=router(jev, cf))
    rendered = render_clairic("Ada Lovelace", label="the Person")
    response = asyncio.run(run.call_clairic(mixed_fields(), rendered, meta=META))
    assert set(response.answers) == {"name", "born"}
    first, second = run.calls
    assert (first.provider, first.model, first.usage["input_tokens"]) == (
        "cloudflare",
        CLAIRIC,
        70,
    )
    assert (second.provider, second.model, second.usage["input_tokens"]) == (
        "typesafe",
        "jev-1.14.0",
        30,
    )
    assert (first.n_items, second.n_items) == (1, 1)
    assert run.usage.input_tokens == 100 and run.usage.requests == 2
    assert run.usage.by_model["jev-1.14.0"].input_tokens == 30
    assert run.usage.by_model[CLAIRIC].input_tokens == 70


def test_a_malformed_routing_note_is_ignored_and_the_call_is_logged_as_one() -> None:
    from trustcall_system_one._render import render_clairic
    from trustcall_system_one._run import Run

    class Odd(Provider):
        async def run(self, *args: Any, **kwargs: Any) -> TransportResponse:
            reply = await super().run(*args, **kwargs)
            body = {**reply.body, "routing": {"partitions": [{"usage": "many"}, 3]}}
            return TransportResponse(body=body)

    run = Run(transport=Odd("cloudflare"))
    asyncio.run(
        run.call_clairic(mixed_fields(), render_clairic("Ada", label="x"), meta=META)
    )
    assert len(run.calls) == 1


def test_a_split_call_that_fails_after_a_partition_answered_is_never_resent() -> None:
    from trustcall_system_one._render import render_clairic
    from trustcall_system_one._run import Run

    jev, cf = Provider("typesafe", tokens=30), Provider("cloudflare", tokens=70)
    release = asyncio.Event()

    async def wait() -> None:
        await release.wait()

    async def done() -> None:
        release.set()

    cf.gate, jev.gate = wait, done
    cf.failure = BadRequestError("Request requires 70000 tokens; the maximum is 65529")
    run = Run(transport=router(jev, cf))
    rendered = render_clairic("Ada Lovelace", label="the Person")
    with pytest.raises(BadRequestError) as caught:
        asyncio.run(run.call_clairic(mixed_fields(), rendered, meta=META))
    # a size refusal is contained only when nothing else was run: this one raised
    assert caught.value.maybe_sent is True
    assert len(jev.calls) == 1 and len(cf.calls) == 1
    (failed,) = run.calls
    assert failed.error is not None and failed.error.startswith("BadRequestError")
    assert failed.usage["input_tokens"] == 30  # what the answered partition used
    assert run.usage.input_tokens == 30
