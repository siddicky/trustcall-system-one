"""Tests of the JSON-safe serialiser."""

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel
from trustcall_system_one import ExtractionResult, FieldAnswer, PatchOp, ResultRecord

from app.serialize import serialize_result, to_jsonable


class Color(Enum):
    RED = "red"


class Pet(BaseModel):
    name: str
    born: date


@dataclass
class Box:
    items: tuple[int, ...]
    meta: dict[int, float]


def test_scalars_and_containers() -> None:
    value = {
        1: Box((1, 2), {3: math.inf}),
        "c": Color.RED,
        "d": date(2026, 10, 22),
        "t": datetime(2026, 10, 22, 12, 0),
        "n": Decimal("1.5"),
        "s": {"x"},
        "b": b"hi",
        "nan": math.nan,
    }
    out = to_jsonable(value)
    json.dumps(out, allow_nan=False)
    assert out["1"] == {"items": [1, 2], "meta": {"3": None}}
    assert out["c"] == "red" and out["d"] == "2026-10-22"
    assert out["n"] == "1.5" and out["s"] == ["x"] and out["b"] == "aGk="
    assert out["nan"] is None


def test_patch_ops() -> None:
    assert to_jsonable(PatchOp("remove", "/a")) == {
        "op": "remove",
        "path": "/a",
        "repaired_from": None,
    }
    assert to_jsonable(PatchOp("add", "/a", 0))["value"] == 0


def test_result_with_a_model_value() -> None:
    record = ResultRecord(
        schema_name="Pet",
        json_doc_id=None,
        kind="insert",
        value=Pet(name="Rex", born=date(2020, 1, 2)),
        fields={"name": FieldAnswer(("name",), "name", "name", "clairic", "Rex")},
    )
    out = serialize_result(ExtractionResult(records=(record,)), 12.34)
    json.dumps(out, allow_nan=False)
    assert out["records"][0]["value"] == {"name": "Rex", "born": "2020-01-02"}
    assert out["records"][0]["fields"]["name"]["path"] == ["name"]
    assert out["elapsed_ms"] == 12.3
    assert out["records"][0]["delete_p"] is None
