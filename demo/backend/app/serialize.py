"""JSON-safe serialisation of the library's result dataclasses."""

import base64
import dataclasses
import math
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from trustcall_system_one import ExtractionResult, PatchOp

__all__ = ["serialize_result", "to_jsonable"]

_OMITTED = frozenset({"responses"})
"""Result fields left out: ``responses`` repeats the ``value`` of each record."""


def to_jsonable(obj: Any) -> Any:
    """Convert any value of a result into something ``json.dumps`` accepts.

    Dataclasses become dicts of their fields, pydantic models are dumped in JSON
    mode, mappings get string keys, tuples and sets become lists, enums become
    their values, dates and times ISO strings, decimals and UUIDs strings, and a
    non-finite float ``None``. Anything else falls back to ``str``.

    Args:
        obj: The value to convert.

    Returns:
        A structure of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool``
        and ``None`` only.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, PatchOp):
        return _patch(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, (Decimal, UUID, PurePath)):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    if type(obj).__name__ == "_Unstated":  # the library's UNSTATED sentinel
        return None
    return str(obj)


def _patch(op: PatchOp) -> dict[str, Any]:
    """Serialise one RFC 6902 operation (``value`` only for add and replace)."""
    out: dict[str, Any] = {"op": op.op, "path": op.path}
    if op.op != "remove":
        out["value"] = to_jsonable(op.value)
    out["repaired_from"] = op.repaired_from
    return out


def serialize_result(result: ExtractionResult, elapsed_ms: float) -> dict[str, Any]:
    """Serialise a whole ``ExtractionResult`` for the API response.

    Args:
        result: The result of ``Extractor.ainvoke``.
        elapsed_ms: Wall-clock time of the extraction in milliseconds.

    Returns:
        Every field of the result (except ``responses``) as JSON-safe values,
        plus ``elapsed_ms``.
    """
    out = {
        f.name: to_jsonable(getattr(result, f.name))
        for f in dataclasses.fields(result)
        if f.name not in _OMITTED
    }
    out["elapsed_ms"] = round(elapsed_ms, 1)
    return out
