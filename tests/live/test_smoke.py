"""Live smoke tests: the library against the real Clairic and Clef.

Skipped unless ``TRUSTCALL_S1_LIVE=1`` and the Cloudflare credentials are in the
environment (``make smoke``). They assert invariants only: no exception, every
patch applies, what the text does not mention is untouched, no operation reaches
another entity's record. Exact values belong to the evals (``make evals``). With
``TRUSTCALL_S1_RECORD=<dir>`` each scenario is also written to a cassette
(``make record``).
"""

from collections.abc import Callable

import pytest

from evals.smoke import SCENARIOS, Scenario
from tests.live.conftest import SKIP_REASON, live_enabled
from trustcall_system_one import Transport, create_extractor

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not live_enabled(), reason=SKIP_REASON),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario(
    scenario: Scenario, live_transport: Callable[[str], Transport]
) -> None:
    extractor = create_extractor(
        scenario.schemas(),
        transport=live_transport(scenario.name),
        **scenario.options,
    )
    result = extractor.invoke(scenario.text, existing=scenario.existing)
    extractor.close()
    print(
        f"\n[{scenario.name}] {result.usage.requests} requests, "
        f"{result.usage.input_tokens} input tokens, "
        f"{len(result.records)} record(s), warnings={list(result.warnings)}"
    )
    scenario.check(result)
