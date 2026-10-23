"""The case suites: the dev set and the two held-out sets.

``default`` is the dev set; ``holdout`` and ``holdout2`` are the held-out sets.
"""

import importlib

from evals.cases import CASES
from evals.model import Case

__all__ = ["DEFAULT_SUITE", "SUITES", "load_suite"]

DEFAULT_SUITE = "default"
SUITES: tuple[str, ...] = ("default", "holdout", "holdout2")
"""``default`` is ``evals/cases.py``; ``holdout`` is ``evals/holdout_cases.py`` and
``holdout2`` is ``evals/holdout2_cases.py``: cases a search never sees, to check that
a combination found on the dev set holds up. ``holdout2`` is the clean held-out set
(``holdout`` was used for diagnosis); neither is ever swept, searched or calibrated
on."""

_HELD_OUT: dict[str, tuple[str, str, str]] = {
    # suite: (module, loader function, case list)
    "holdout": ("holdout_cases", "holdout_cases", "HOLDOUT_CASES"),
    "holdout2": ("holdout2_cases", "holdout2_cases", "HOLDOUT2_CASES"),
}


def load_suite(name: str = DEFAULT_SUITE) -> list[Case]:
    """The cases of a suite.

    Args:
        name: ``default`` (the dev cases), ``holdout`` or ``holdout2``.

    Returns:
        The cases, in order.

    Raises:
        ValueError: If the suite is unknown, or a held-out suite is asked for while
            its file (``evals/holdout_cases.py``, ``evals/holdout2_cases.py``) does
            not exist or has no cases. The file exports its case list
            (``HOLDOUT_CASES``, ``HOLDOUT2_CASES``), a loader function
            (``holdout_cases()``, ``holdout2_cases()``) and ``SUITE_NAME``.
    """
    if name == DEFAULT_SUITE:
        return list(CASES)
    if name not in _HELD_OUT:
        raise ValueError(f"unknown suite {name!r}; suites: {list(SUITES)}")
    stem, loader_name, list_name = _HELD_OUT[name]
    module_name = f"evals.{stem}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ValueError(
            f"the {name} suite needs evals/{stem}.py, which does not exist "
            f"yet; it exports {list_name} (a list of evals.model.Case)"
        ) from None
    loader = getattr(module, loader_name, None)
    found = (
        loader()
        if callable(loader)
        else getattr(module, list_name, None) or getattr(module, "CASES", None)
    )
    if not found:
        raise ValueError(
            f"evals/{stem}.py has no cases: it must export {loader_name}() "
            f"(or {list_name}), a non-empty list of evals.model.Case"
        )
    return list(found)
