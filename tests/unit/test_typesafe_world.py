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


def test_a_stitch_that_joins_lines_runs_end_to_end() -> None:
    """A scripted Jev that joins every line to the next: one block, and it runs."""
    case = next(c for c in CASES if c.id == "readme_people")

    async def go(join: float) -> tuple[CaseResult, list[str]]:
        assert case.world is not None
        backend = JevBackend(case.world())
        backend.join = join  # the stitch answer: above the threshold, lines join
        client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        transport = TypeSafeTransport(api_key=KEY, client=client)
        extractor = create_extractor(
            case.schemas(),
            transport=Bridge(transport, backend),
            **build_options({}, case.options),
        )
        result = await extractor.ainvoke(case.text, existing=case.existing)
        stitches = [r for r in backend.requests if "L001" in r["questions"]]
        assert stitches, "the stitch request was never sent"
        seen = [
            r["state"]["input"]
            for r in backend.requests
            if isinstance(r["state"], dict) and "input" in r["state"]
        ]
        return score(case, result), seen

    _, apart = asyncio.run(go(0.04))
    assert any("B001|" in s for s in apart)  # the lines stay apart: many blocks
    scored, joined = asyncio.run(go(0.95))
    assert scored.error is None
    tagged = [s for s in joined if "B000|" in s]
    assert tagged
    assert not any("B001|" in s for s in joined)  # every line joined into one block
