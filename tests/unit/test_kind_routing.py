"""KindRoutingTransport: routing by call kind and/or model, and what it forwards."""

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from trustcall_system_one import (
    CLAIRIC,
    CLEF,
    CLEF_FLASH,
    BadRequestError,
    CallMeta,
    CloudflareTransport,
    DecisionsTransport,
    KindRoutingTransport,
    RateLimitError,
    RoutingTransport,
    ServerError,
    TransportResponse,
    TypeSafeTransport,
    UncertainOutcomeError,
)
from trustcall_system_one._errors import is_size_rejection, may_have_run
from trustcall_system_one.transport import (
    CALL_KINDS,
    CLAIRIC_KINDS,
    CLEF_KINDS,
    ESCALATE_KINDS,
    max_questions,
    state_token_budget,
)

KEY = "xpl_" + "ab12cd34" * 5
IMAGE = "data:image/png;base64,iVBORw0KGgo="
PAYLOAD: dict[str, Any] = {
    "model": "clef",
    "state": "x",
    "questions": {
        "q": {"type": "noul", "instructions": "Is it?"},
    },
}


class Tag:
    """A transport that answers with its name and keeps what it was given."""

    def __init__(
        self,
        name: str,
        *,
        serves: bool | None = None,
        provider: str | None = None,
        limit: int | None = None,
        budget: int | None = None,
    ) -> None:
        self.name = name
        self._serves = serves
        self.provider = provider or name
        if limit is not None:
            self.max_questions = limit
        if budget is not None:
            self.state_token_budget = budget
        self.calls: list[tuple[str, Mapping[str, Any], CallMeta, float | None]] = []
        self.closed: list[str] = []
        self.failure: BaseException | None = None

    async def run(
        self,
        model: str,
        payload: Mapping[str, Any],
        *,
        meta: CallMeta,
        timeout: float | None = None,
    ) -> TransportResponse:
        self.calls.append((model, payload, meta, timeout))
        if self.failure is not None:
            raise self.failure
        return TransportResponse(
            body={"served_by": self.name},
            status_code=200,
            provider=self.provider,
            served_model=f"{self.name}-model",
        )

    def serves(self, model: str, payload: Mapping[str, Any] | None = None) -> bool:
        if self._serves is None:
            return True
        return self._serves and not (payload or {}).get("images")

    def sensitive_values(self) -> tuple[str, ...]:
        return (f"secret-{self.name}",)

    def classify_failure(self, exc: Exception) -> str | None:
        return f"{self.name}-says-uncertain"

    async def aclose(self) -> None:
        self.closed.append("aclose")

    def close(self) -> None:
        self.closed.append("close")


def call(
    transport: Any,
    kind: str = "attribution",
    model: str = CLEF,
    payload: Mapping[str, Any] = PAYLOAD,
) -> TransportResponse:
    reply: TransportResponse = asyncio.run(
        transport.run(model, payload, meta=CallMeta(kind=kind))  # type: ignore[arg-type]
    )
    return reply


def served_by(transport: Any, kind: str = "attribution", model: str = CLEF) -> str:
    return str(call(transport, kind, model).body["served_by"])


def test_the_kinds_split_into_clairic_clef_and_both() -> None:
    both = {"repair", "raw"}
    assert CLAIRIC_KINDS | CLEF_KINDS | ESCALATE_KINDS | both == set(CALL_KINDS)
    assert not CLAIRIC_KINDS & CLEF_KINDS
    assert {"escalate"} == ESCALATE_KINDS and not ESCALATE_KINDS & CLEF_KINDS
    assert "attribution" in CLEF_KINDS and "extract" in CLAIRIC_KINDS


# --------------------------------------------------------------------------- #
# Which route wins
# --------------------------------------------------------------------------- #


def test_a_kind_route_beats_a_model_route_and_a_kind_on_a_model_beats_both() -> None:
    router = KindRoutingTransport(
        {
            "clef": Tag("clef"),
            "attribution": Tag("kind"),
            "repair@clairic": Tag("repair-clairic"),
            "repair": Tag("repair"),
        },
        default=Tag("default"),
    )
    assert served_by(router, "attribution") == "kind"
    assert served_by(router, "gate") == "clef"
    assert served_by(router, "gate", CLEF_FLASH) == "clef"  # the family
    assert served_by(router, "repair", CLEF) == "repair"
    assert served_by(router, "repair", CLAIRIC) == "repair-clairic"
    assert served_by(router, "extract", CLAIRIC) == "default"
    assert served_by(router, "gate", "something-else") == "default"


def test_the_example_of_the_docs_routes_three_kinds_and_a_default() -> None:
    router = KindRoutingTransport(
        {"attribution": Tag("decisions"), "gate": Tag("jev"), "extract": Tag("cf")},
        default=Tag("other"),
    )
    assert served_by(router, "attribution") == "decisions"
    assert served_by(router, "gate") == "jev"
    assert served_by(router, "extract", CLAIRIC) == "cf"
    assert served_by(router, "presence") == "other"


def test_a_flash_route_beats_the_family_and_full_ids_work() -> None:
    router = KindRoutingTransport(
        {
            "clef": Tag("family"),
            "@cf/cloudflare/clef-flash": Tag("flash"),
            "gate@@cf/cloudflare/clef": Tag("gate-clef"),
        },
        default=Tag("default"),
    )
    assert served_by(router, "presence", CLEF) == "family"
    assert served_by(router, "presence", CLEF_FLASH) == "flash"
    assert served_by(router, "gate", CLEF) == "gate-clef"
    # kind@clef covers clef-flash as clef does, and kind@model beats a model key
    assert served_by(router, "gate", CLEF_FLASH) == "gate-clef"


def test_the_call_is_passed_on_unchanged() -> None:
    tag = Tag("t")
    router = KindRoutingTransport({"gate": tag}, default=Tag("d"))
    meta = CallMeta(kind="gate", schema="Person", entity="Ada", field="name")
    asyncio.run(router.run(CLEF, PAYLOAD, meta=meta, timeout=12.5))
    ((model, payload, seen, timeout),) = tag.calls
    assert (model, timeout) == (CLEF, 12.5)
    assert payload is PAYLOAD and seen is meta


def test_keys_are_validated() -> None:
    with pytest.raises(ValueError, match="cannot route 'gpt-4'"):
        KindRoutingTransport({"gpt-4": Tag("x")}, default=Tag("y"))
    with pytest.raises(ValueError, match="not a call kind"):
        KindRoutingTransport({"nope@clef": Tag("x")}, default=Tag("y"))
    with pytest.raises(ValueError, match="cannot route 'gate@gpt'"):
        KindRoutingTransport({"gate@gpt": Tag("x")}, default=Tag("y"))
    with pytest.raises(ValueError, match="more than one route"):
        KindRoutingTransport({"clef": Tag("x"), CLEF: Tag("y")}, default=Tag("z"))
    with pytest.raises(ValueError, match="more than one route"):
        KindRoutingTransport(
            {"gate@clef": Tag("x"), "gate@@cf/cloudflare/clef": Tag("y")},
            default=Tag("z"),
        )


def test_it_is_a_routing_transport_for_models_when_given_model_keys_alone() -> None:
    routes = {"clef": Tag("experiential")}
    kind = KindRoutingTransport(routes, default=Tag("cloudflare"))
    plain = RoutingTransport(routes, default=Tag("cloudflare"))
    for model in (CLEF, CLEF_FLASH, CLAIRIC, "other"):
        assert served_by(kind, "gate", model) == str(
            asyncio.run(plain.run(model, PAYLOAD, meta=CallMeta(kind="gate"))).body[
                "served_by"
            ]
        )


# --------------------------------------------------------------------------- #
# Limits, serving, closing
# --------------------------------------------------------------------------- #


def test_the_limits_are_the_smallest_of_the_transports_that_may_be_asked() -> None:
    decisions = DecisionsTransport(api_key=KEY)
    jev = TypeSafeTransport(api_key=KEY)
    cloudflare = CloudflareTransport(account_id="a", api_token="t")
    router = KindRoutingTransport(
        {"attribution": decisions, "gate": jev}, default=cloudflare
    )
    assert max_questions(router, CLEF) == 32  # Decisions' 32, Cloudflare's 64
    budget = state_token_budget(router, CLEF)
    assert budget is not None and budget == jev.state_token_budget
    # a router without them says what the rest say
    only_cf = KindRoutingTransport({"gate": cloudflare}, default=cloudflare)
    assert max_questions(only_cf, CLEF) == 64
    assert state_token_budget(only_cf, CLEF) is None


def test_a_route_of_the_other_family_does_not_narrow_a_model() -> None:
    narrow = Tag("narrow", limit=4, budget=100)
    router = KindRoutingTransport(
        {"extract": narrow, "clairic": narrow}, default=Tag("wide", limit=64)
    )
    assert max_questions(router, CLEF) == 64  # extract and clairic are never Clef
    assert max_questions(router, CLAIRIC) == 4
    assert state_token_budget(router, CLEF) is None
    clef_side = KindRoutingTransport(
        {"gate": Tag("g", limit=8), "clef-flash": Tag("f", limit=5)},
        default=Tag("wide", limit=64),
    )
    assert max_questions(clef_side, CLEF) == 8
    assert max_questions(clef_side, CLEF_FLASH) == 5
    assert max_questions(clef_side, CLAIRIC) == 64


def test_a_transport_that_cannot_serve_the_model_is_not_counted() -> None:
    clef_only = Tag("clef-only", limit=2, serves=False)
    router = KindRoutingTransport({"gate": clef_only}, default=Tag("d", limit=50))
    assert max_questions(router, CLEF) == 50


def test_serves_with_a_payload_needs_every_candidate_to_serve_it() -> None:
    router = KindRoutingTransport(
        {"attribution": Tag("text-only", serves=True)}, default=Tag("any")
    )
    assert router.serves(CLEF) is True
    assert router.serves(CLEF, PAYLOAD) is True
    with_image = {**PAYLOAD, "images": [IMAGE]}
    assert router.serves(CLEF, with_image) is False  # one candidate would refuse
    nobody = KindRoutingTransport({}, default=Tag("no", serves=False))
    assert nobody.serves(CLEF) is False


def test_it_closes_each_member_once_and_lists_their_secrets() -> None:
    shared, other = Tag("shared"), Tag("other")
    router = KindRoutingTransport({"gate": shared, "clef": shared}, default=other)
    asyncio.run(router.aclose())
    router.close()
    assert shared.closed == ["aclose", "close"] and other.closed == ["aclose", "close"]
    assert set(router.sensitive_values()) == {"secret-shared", "secret-other"}

    async def use() -> None:
        async with KindRoutingTransport({}, default=other):
            pass

    asyncio.run(use())
    assert other.closed[-1] == "aclose"


def test_a_failing_close_does_not_stop_the_others() -> None:
    class Broken(Tag):
        async def aclose(self) -> None:
            raise RuntimeError("no")

    healthy = Tag("ok")
    router = KindRoutingTransport({"gate": Broken("bad")}, default=healthy)
    with pytest.raises(RuntimeError):
        asyncio.run(router.aclose())
    assert healthy.closed == ["aclose"]


# --------------------------------------------------------------------------- #
# Images and what cannot be served
# --------------------------------------------------------------------------- #


def test_only_a_request_with_images_moves_to_the_default() -> None:
    text_only, default = Tag("text-only", serves=True), Tag("default")
    router = KindRoutingTransport({"attribution": text_only}, default=default)
    plain = call(router, "attribution")
    assert plain.body == {"served_by": "text-only"} and plain.rerouted_from is None
    pictured = call(router, "attribution", payload={**PAYLOAD, "images": [IMAGE]})
    assert pictured.body == {"served_by": "default"}
    assert pictured.rerouted_from == "text-only"


def test_images_stay_with_the_route_when_the_default_cannot_take_them_either() -> None:
    first, second = Tag("first", serves=True), Tag("second", serves=True)
    router = KindRoutingTransport({"attribution": first}, default=second)
    reply = call(router, "attribution", payload={**PAYLOAD, "images": [IMAGE]})
    assert reply.body == {"served_by": "first"} and reply.rerouted_from is None
    assert second.calls == []


def test_a_real_decisions_route_hands_images_to_the_default() -> None:
    decisions = DecisionsTransport(api_key=KEY)
    default = Tag("cloudflare")
    router = KindRoutingTransport({"attribution": decisions}, default=default)
    reply = call(router, "attribution", payload={**PAYLOAD, "images": [IMAGE]})
    assert reply.body == {"served_by": "cloudflare"}
    assert reply.rerouted_from == "decisions"


def test_a_clairic_call_routed_to_a_clef_only_provider_is_refused_not_moved() -> None:
    default = Tag("cloudflare")
    router = KindRoutingTransport({"extract": DecisionsTransport(api_key=KEY)}, default)
    payload = {"model": "clairic", "state": "text", "fields": {"f": {"type": "text"}}}
    with pytest.raises(BadRequestError) as caught:
        call(router, "extract", CLAIRIC, payload)
    assert default.calls == []  # the text did not go to the default
    assert caught.value.provider == "decisions"


# --------------------------------------------------------------------------- #
# Failures: never resent, tagged, classified by whoever raised them
# --------------------------------------------------------------------------- #


def test_an_uncertain_failure_is_raised_once_and_not_sent_anywhere_else() -> None:
    primary, default = Tag("primary"), Tag("default")
    primary.failure = UncertainOutcomeError("may have been charged")
    router = KindRoutingTransport({"gate": primary}, default=default)
    with pytest.raises(UncertainOutcomeError) as caught:
        call(router, "gate")
    assert len(primary.calls) == 1 and default.calls == []
    assert caught.value.provider == "primary"
    assert may_have_run(caught.value)


def test_a_partial_usage_and_maybe_sent_pass_through_on_the_same_error() -> None:
    primary = Tag("primary")
    boom = BadRequestError("too large", status_code=413)
    boom.maybe_sent = True
    boom.partial_usage = {"input_tokens": 9}  # type: ignore[attr-defined]
    boom.served_model = "jev-9"
    primary.failure = boom
    router = KindRoutingTransport({"gate": primary}, default=Tag("d"))
    with pytest.raises(BadRequestError) as caught:
        call(router, "gate")
    assert caught.value is boom
    assert (boom.maybe_sent, boom.served_model) == (True, "jev-9")
    assert not is_size_rejection(boom)  # a charged request is never contained
    # a BadRequestError never falls back, whoever raised it
    assert router.classify_failure(boom) is None
    maybe = UncertainOutcomeError("x")  # anything else that may have run is uncertain
    maybe.maybe_sent = True
    assert router.classify_failure(maybe) == "uncertain"


def test_failures_are_classified_by_the_transport_that_raised_them() -> None:
    first, second = Tag("first"), Tag("second")
    first.failure, second.failure = ServerError("a"), RateLimitError("b")
    router = KindRoutingTransport({"gate": first}, default=second)
    with pytest.raises(ServerError) as one:
        call(router, "gate")
    with pytest.raises(RateLimitError) as two:
        call(router, "presence")
    assert router.classify_failure(one.value) == "first-says-uncertain"
    assert router.classify_failure(two.value) == "second-says-uncertain"
    assert router.classify_failure(ServerError("never routed")) == "uncertain"
    assert router.classify_failure(RuntimeError("bug")) is None


def test_the_innermost_transport_is_the_one_that_classifies() -> None:
    leaf = Tag("leaf")
    leaf.failure = UncertainOutcomeError("x")
    inner = KindRoutingTransport({"gate": leaf}, default=Tag("d"))
    outer = KindRoutingTransport({"gate": inner}, default=Tag("e"))
    with pytest.raises(UncertainOutcomeError) as caught:
        call(outer, "gate")
    assert outer.classify_failure(caught.value) == "leaf-says-uncertain"
    assert caught.value.provider == "leaf"


def test_a_cancellation_is_not_swallowed_and_is_marked() -> None:
    primary = Tag("primary")
    primary.failure = asyncio.CancelledError()
    router = KindRoutingTransport({"gate": primary}, default=Tag("d"))
    with pytest.raises(asyncio.CancelledError):
        call(router, "gate")


def test_provenance_of_the_answer_is_the_routed_transports() -> None:
    router = KindRoutingTransport({"gate": Tag("jev", provider="typesafe")}, Tag("d"))
    reply = call(router, "gate")
    assert (reply.provider, reply.served_model) == ("typesafe", "jev-model")


def test_a_5xx_behind_the_router_ends_the_call_after_one_request() -> None:
    import httpx

    from trustcall_system_one import ExperientialTransport
    from trustcall_system_one._run import Run
    from trustcall_system_one._wire import NoulQuestion

    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    experiential = ExperientialTransport(
        api_key=KEY, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    default = Tag("default")
    run = Run(transport=KindRoutingTransport({"attribution": experiential}, default))
    question = NoulQuestion(type="noul", instructions="Does `input` say yes?")
    with pytest.raises(UncertainOutcomeError):
        asyncio.run(
            run.call_clef(
                {"q": question}, {"input": "x"}, meta=CallMeta(kind="attribution")
            )
        )
    assert len(sent) == 1  # never resent, and never handed to the default
    assert default.calls == []
    (logged,) = run.calls
    assert logged.provider == "experiential" and logged.kind == "attribution"
    assert logged.error is not None and logged.error.startswith("UncertainOutcomeError")
