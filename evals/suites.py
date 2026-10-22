"""The case suites: the dev set (``default``) and the held-out set (``holdout``)."""

import importlib

from evals.cases import CASES
from evals.model import Case

__all__ = ["DEFAULT_SUITE", "SUITES", "load_suite"]

DEFAULT_SUITE = "default"
SUITES: tuple[str, ...] = ("default", "holdout")
"""``default`` is ``evals/cases.py``; ``holdout`` is ``evals/holdout_cases.py``: cases
a search never sees, to check that a combination found on the dev set holds up."""


def load_suite(name: str = DEFAULT_SUITE) -> list[Case]:
    """The cases of a suite.

    Args:
        name: ``default`` (the dev cases) or ``holdout``.

    Returns:
        The cases, in order.

    Raises:
        ValueError: If the suite is unknown, or ``holdout`` is asked for while
            ``evals/holdout_cases.py`` does not exist (or has no ``CASES`` list).
    """
    if name == DEFAULT_SUITE:
        return list(CASES)
    if name != "holdout":
        raise ValueError(f"unknown suite {name!r}; suites: {list(SUITES)}")
    try:
        module = importlib.import_module("evals.holdout_cases")
    except ModuleNotFoundError as exc:
        if exc.name != "evals.holdout_cases":
            raise
        raise ValueError(
            "the holdout suite needs evals/holdout_cases.py, which does not exist "
            "yet; it exports CASES (a list of evals.model.Case)"
        ) from None
    found = getattr(module, "CASES", None) or getattr(module, "HOLDOUT_CASES", None)
    if not found:
        raise ValueError(
            "evals/holdout_cases.py has no cases: it must export CASES (or "
            "HOLDOUT_CASES), a non-empty list of evals.model.Case"
        )
    return list(found)
