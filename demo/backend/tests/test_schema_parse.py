"""Tests of POST /api/schema/parse."""

import pytest
from fastapi.testclient import TestClient

VALID = '''
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """How urgent."""

    LOW = "low"
    HIGH = "high"


class Address(BaseModel):
    """Where someone lives."""

    city: Optional[str] = Field(default=None, description="City name")
    zip: str | None = None


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str = Field(description="Full name")
    kind: Literal["person", "company"] = "person"
    priority: Priority = Priority.LOW
    age: int | None = Field(default=None, ge=0, le=150)
    address: Address | None = None
    notes: list[str] = Field(default_factory=list)
'''


def parse(client: TestClient, source: str):
    return client.post("/api/schema/parse", json={"source": source})


def test_valid_source_gives_a_schema_per_model_in_source_order(
    client: TestClient,
) -> None:
    response = parse(client, VALID)
    assert response.status_code == 200
    schemas = response.json()["schemas"]
    assert [s["name"] for s in schemas] == ["Address", "Contact"]  # not the enum
    address, contact = (s["json_schema"] for s in schemas)
    assert address["properties"]["city"]["description"] == "City name"
    assert contact["properties"]["name"]["description"] == "Full name"
    assert contact["properties"]["kind"]["enum"] == ["person", "company"]
    assert "Priority" in contact["$defs"] and "Address" in contact["$defs"]
    assert contact["required"] == ["name"]
    assert contact["description"] == "A contact in the CRM."


ATTACKS = {
    "dunder import call": "__import__('os').system('id')",
    "import os": "import os",
    "from os import path": "from os import path",
    "submodule import": "import pydantic.v1",
    "star import": "from pydantic import *",
    "forbidden name": "from pydantic import import_string",
    "open": "x = open('x')",
    "class of tuple": "().__class__",
    "lambda": "from pydantic import BaseModel, Field\n"
    "class A(BaseModel):\n    x: int = Field(default_factory=lambda: 1)",
    "def": "def f():\n    return 1",
    "decorator": "from pydantic import BaseModel\n@dec\nclass A(BaseModel):\n"
    "    x: int",
    "method with decorator": "from pydantic import BaseModel, field_validator\n"
    "class A(BaseModel):\n    @field_validator('x')\n"
    "    def v(cls, x):\n        return x",
    "comprehension": "x = [i for i in range(3)]",
    "dict comprehension": "x = {i: i for i in 'ab'}",
    "global": "global x",
    "while": "while True:\n    pass",
    "for": "for i in 'ab':\n    pass",
    "with": "with open('x') as f:\n    pass",
    "try": "try:\n    pass\nexcept Exception:\n    pass",
    "starred": "x = [*'ab']",
    "string forward reference": "from pydantic import BaseModel\n"
    "class A(BaseModel):\n    x: \"__import__('os').system('id')\"",
    "string in subscript": "from pydantic import BaseModel\n"
    "class A(BaseModel):\n    x: list[\"__import__('os')\"]",
    "string in alias": "X = list[\"__import__('os')\"]",
    "attribute escape": "from pydantic import BaseModel\n"
    "class A(BaseModel):\n    x: int = BaseModel.__subclasses__",
    "private attribute": "import typing\nx = typing._sys",
    "dunder name": "__builtins__ = 1",
    "call of unimported name": "from pydantic import BaseModel\nclass A(BaseModel):\n"
    "    x: int = eval('1')",
    "metaclass": "from pydantic import BaseModel\n"
    "class A(BaseModel, metaclass=type):\n    x: int",
    "arithmetic": "x = 10 ** 10 ** 10",
    "bare expression": "1 + 1",
    "keyword unpacking": "from pydantic import Field\nx = Field(**{'a': 1})",
    "f-string": "x = f'{1}'",
}


@pytest.mark.parametrize("source", ATTACKS.values(), ids=ATTACKS.keys())
def test_attacks_are_rejected_as_unsafe(client: TestClient, source: str) -> None:
    response = parse(client, source)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "unsafe_source"
    assert detail["message"]


def test_the_line_of_the_problem_is_reported(client: TestClient) -> None:
    response = parse(client, "from pydantic import BaseModel\n\nimport os\n")
    assert response.json()["detail"]["line"] == 3


def test_a_source_over_20k_chars_is_unsafe(client: TestClient) -> None:
    response = parse(client, "# " + "a" * 20_000)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "unsafe_source" and detail["line"] is None


def test_syntax_error(client: TestClient) -> None:
    response = parse(client, "class A(:\n    pass\n")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "syntax_error" and detail["line"] == 1


def test_no_models(client: TestClient) -> None:
    response = parse(client, "from pydantic import BaseModel\nX = 1\n")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "no_models"


def test_enum_only_is_no_models(client: TestClient) -> None:
    source = "from enum import Enum\nclass E(Enum):\n    A = 'a'\n"
    assert parse(client, source).json()["detail"]["code"] == "no_models"


def test_a_model_that_fails_to_build_is_invalid_model(client: TestClient) -> None:
    source = "from pydantic import BaseModel\nclass A(BaseModel):\n    x: Missing\n"
    response = parse(client, source)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_model"
    assert detail["line"] is not None


def test_a_name_used_but_not_imported_is_invalid_model(client: TestClient) -> None:
    source = (
        "from pydantic import BaseModel\nclass A(BaseModel):\n    x: Optional[int]\n"
    )
    detail = parse(client, source).json()["detail"]
    assert detail["code"] == "invalid_model" and detail["line"] == 3


def test_each_call_is_independent(client: TestClient) -> None:
    source = "from pydantic import BaseModel\nclass A(BaseModel):\n    x: int\n"
    first = parse(client, source).json()
    second = parse(client, source).json()
    assert first == second
