"""The evals' scripted worlds, answered by a scripted Jev through ``TypeSafeTransport``.

Every case of the eval dataset has a ``FakeWorld`` that knows the ideal Clairic and Clef
answers. Here the world is only the *ground truth*: ``tests/unit/jev_mock.py`` plays
Jev from it, and the library runs on ``TypeSafeTransport`` (so the requests are real
Jev requests, validated by the mock) end to end: ``One`` inserts, updates and ``Many``
scenarios included. What is under test is the translation, the candidate selection and
the engine on top of them, not the model.
"""

import asyncio
from typing import Any

import httpx
import pytest

from evals.cases import CASES
from evals.configs import build_options
from evals.model import Case, CaseResult
from evals.scoring import score
from tests.unit.jev_mock import Bridge, JevBackend
from trustcall_system_one import TypeSafeTransport, create_extractor

KEY = "test-key-for-typesafe"


async def run_case(case: Case) -> tuple[CaseResult, JevBackend]:
    assert case.world is not None
    world = case.world()
    backend = JevBackend(world)
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    transport = TypeSafeTransport(api_key=KEY, client=client)
    options = build_options({}, case.options)
    extractor = create_extractor(
        case.schemas(), transport=Bridge(transport, backend), **options
    )
    result = await extractor.ainvoke(case.text, existing=case.existing)
    return score(case, result), backend


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.world is not None], ids=lambda c: c.id
)
def test_case_runs_on_scripted_jev(case: Case) -> None:
    scored, backend = asyncio.run(run_case(case))
    problems: list[Any] = [*scored.missing, *scored.extra, *scored.violations]
    assert scored.error is None
    assert scored.passed, problems
    assert backend.requests
