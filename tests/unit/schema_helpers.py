"""Loaders for the published JSON schemas used by conformance tests."""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"


def load_schema(name: str) -> dict[str, Any]:
    """Load ``docs/schemas/<name>.json``."""
    data: dict[str, Any] = json.loads((SCHEMA_DIR / f"{name}.json").read_text())
    return data


def validator(name: str) -> Draft202012Validator:
    """Return a validator for ``docs/schemas/<name>.json``."""
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)
