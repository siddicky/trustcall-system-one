"""Tests of POST /api/schema/parse."""

import inspect
import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app import schemas
from app._sandbox import SandboxCrash, SandboxTimeout, SandboxUnavailable
from app._schema_guard import SourceError, check_source

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

HEADER = "from pydantic import BaseModel, Field\n"
ALL_IMPORTS = (
    "import annotated_types, datetime, decimal, enum, pydantic, typing, uuid\n" + HEADER
)

# The two reproductions of the review: an assignment chain to an already callable name.
ATTACKS["rce: Field = uuid.os.system"] = (
    "import uuid\n"
    "from pydantic import BaseModel\n"
    "Field = uuid.os.system\n"
    'r = Field("id > /tmp/pwned.txt")\n'
    "class M(BaseModel):\n    x: int\n"
)
ATTACKS["secret leak: Any = uuid.os.getenv"] = (
    "import uuid\n"
    "from pydantic import BaseModel, Field\n"
    "from typing import Any\n"
    "Any = uuid.os.getenv\n"
    'tok = Any("CLOUDFLARE_API_TOKEN")\n'
    "class M(BaseModel):\n    x: str = Field(description=tok)\n"
)
ATTACKS["rce in a class body"] = (
    "import uuid\n"
    "from pydantic import BaseModel\n"
    "class M(BaseModel):\n"
    "    Field = uuid.os.system\n"
    '    x: int = Field("id > /tmp/pwned.txt")\n'
)
ATTACKS["call of uuid.os.system"] = (
    HEADER + 'import uuid\nclass M(BaseModel):\n    x: int = uuid.os.system("id")\n'
)

# Attribute chains through the re-exports of allowed modules.
CHAINS = [
    "datetime.sys",
    "datetime.datetime.now",
    "datetime.date.today",
    "decimal.sys",
    "decimal.Decimal.__init__",
    "typing.sys",
    "typing.re",
    "typing.Optional.__call__",
    "enum.sys",
    "enum.os",
    "uuid.os",
    "uuid.os.environ",
    "uuid.os.system",
    "uuid.sys.modules",
    "pydantic.main",
    "pydantic.v1",
    "pydantic.fields.FieldInfo",
    "pydantic.BaseModel.model_construct",
    "pydantic.Field.__globals__",
    "annotated_types.sys",
    "annotated_types.types",
    "annotated_types.Ge.__init__",
    "annotated_types.Ge.__module__",
    "annotated_types.BaseMetadata",
    "annotated_types.Interval",
]
for _chain in CHAINS:
    ATTACKS[f"chain {_chain} as annotation"] = (
        ALL_IMPORTS + f"class M(BaseModel):\n    x: {_chain}\n"
    )
    ATTACKS[f"chain {_chain} called"] = (
        ALL_IMPORTS + f"class M(BaseModel):\n    x: int = {_chain}('id')\n"
    )
ATTACKS["chain on a builtin"] = HEADER + "class M(BaseModel):\n    x: int = int.mro\n"
ATTACKS["attribute on a model class"] = (
    HEADER
    + "class A(BaseModel):\n    x: int\n"
    + "class M(BaseModel):\n    y: int = A.model_fields\n"
)
ATTACKS["attribute on a call result"] = (
    HEADER + "class M(BaseModel):\n    x: int = Field(1).default\n"
)
ATTACKS["submodule imported as module"] = "import pydantic.main\n"
ATTACKS["from-import of a module"] = "from typing import sys\n"
ATTACKS["from-import of a member of another module"] = "from typing import Decimal\n"
ATTACKS["import from the wrong module"] = "from pydantic import date\n"

# Rebinding an imported name, an allowlisted name, a module name or a builtin.
REBINDS = {
    "assign Field": HEADER + "Field = 1\n",
    "assign Field (not imported)": "Field = 1\n",
    "assign BaseModel": HEADER + "BaseModel = 1\n",
    "assign str": HEADER + "str = 'x'\n",
    "assign int": "int = 1\nfrom pydantic import BaseModel\n",
    "assign list": "list = 1\n",
    "assign Any (not imported)": "Any = 1\n",
    "assign a module name": "import uuid\nuuid = 1\n",
    "assign an unimported module name": "datetime = 1\n",
    "assign date (allowlisted)": "date = 1\n",
    "annotated assign Field": HEADER + "Field: int = 1\n",
    "annotated assign str": "str: type = 1\n",
    "chained assign": "a = Field = 1\n",
    "class named Field": HEADER + "class Field(BaseModel):\n    x: int\n",
    "class named BaseModel": HEADER + "class BaseModel:\n    pass\n",
    "class named str": "class str:\n    pass\n",
    "class named Optional": "class Optional:\n    pass\n",
    "class named uuid": "class uuid:\n    pass\n",
    "class defined twice": HEADER
    + "class A(BaseModel):\n    x: int\nclass A:\n    pass\n",
    "class rebinds a constant": "A = 1\nclass A:\n    pass\n",
    "constant defined twice": "A = 1\nA = 2\n",
    "alias onto Field": "from pydantic import BaseModel as Field\n",
    "alias onto str": "from typing import Any as str\n",
    "module alias onto Field": "import uuid as Field\n",
    "module alias onto a module": "import uuid as typing\n",
    "import over a class": "class A:\n    pass\nfrom typing import Any as A\n",
    "datetime class then module": "from datetime import datetime\nimport datetime\n",
    "module then datetime class": "import datetime\nfrom datetime import datetime\n",
    "class body Field": HEADER + "class M(BaseModel):\n    Field = 1\n    x: int\n",
    "class body annotated Field": HEADER + "class M(BaseModel):\n    Field: int = 1\n",
    "class body BaseModel": HEADER + "class M(BaseModel):\n    BaseModel = 1\n",
    "class body str": HEADER + "class M(BaseModel):\n    str = 'x'\n    x: int\n",
    "class body int": HEADER + "class M(BaseModel):\n    int: int = 1\n",
    "class body Any (not imported)": HEADER + "class M(BaseModel):\n    Any = 1\n",
    "class body Optional": HEADER + "class M(BaseModel):\n    Optional: int = 1\n",
    "class body imported date": "from datetime import date\n"
    + HEADER
    + "class M(BaseModel):\n    date: str = 'x'\n",
    "class body imported module": "import uuid\n"
    + HEADER
    + "class M(BaseModel):\n    uuid: str = 'x'\n",
    "class body shadows a model": HEADER
    + "class A(BaseModel):\n    x: int\nclass M(BaseModel):\n    A: int = 1\n",
    "nested class named Field": HEADER
    + "class M(BaseModel):\n    class Field:\n        pass\n",
    "class body import": HEADER + "class M(BaseModel):\n    from typing import Any\n",
    "class body import module": HEADER + "class M(BaseModel):\n    import uuid\n",
    "enum body rebinds": "from enum import Enum, auto\n"
    "class E(Enum):\n    auto = 1\n    Enum = 2\n",
}
ATTACKS.update({f"rebind: {k}": v for k, v in REBINDS.items()})

# Top-level assignments may only bind fresh names to constants and simple literals.
NOT_LITERAL = {
    "a type alias": HEADER + "Alias = list[str]\n",
    "a builtin": "X = int\n",
    "a call": HEADER + "X = Field(description='a')\n",
    "an attribute": "import uuid\nX = uuid.UUID\n",
    "a name": "A = 1\nB = A\n",
    "a binary operation": "X = 1 | 2\n",
    "an annotated call": HEADER + "x: int = Field(1)\n",
    "a list with a name": "X = [int]\n",
}
ATTACKS.update({f"top-level assignment of {k}": v for k, v in NOT_LITERAL.items()})

# Field(default_factory=...) is only allowed with list, dict or set.
FACTORIES = {
    "Decimal": "from decimal import Decimal\n",
    "UUID": "from uuid import UUID\n",
    "uuid.UUID": "import uuid\n",
    "str": "",
    "int": "",
    "bytes": "",
    "tuple": "",
    "Field": "",
    "uuid.os.getcwd": "import uuid\n",
    "BaseModel": "",
}
for _name, _imports in FACTORIES.items():
    ATTACKS[f"default_factory={_name}"] = (
        HEADER
        + _imports
        + f"class M(BaseModel):\n    x: str = Field(default_factory={_name})\n"
    )
ATTACKS["default_factory call"] = (
    HEADER + "class M(BaseModel):\n    x: list = Field(default_factory=list())\n"
)
ATTACKS["default_factory via ConfigDict-like call"] = (
    "from pydantic import BaseModel, ConfigDict\n"
    "class M(BaseModel):\n"
    "    model_config = ConfigDict(alias_generator=str)\n"
    "    x: int\n"
)
ATTACKS["json_schema_extra with a name"] = (
    HEADER + "class M(BaseModel):\n    x: int = Field(json_schema_extra=int)\n"
)

# Enum members: only members of an enum class defined earlier, one level deep.
ATTACKS["enum member chain"] = (
    "from enum import Enum\n"
    "from pydantic import BaseModel\n"
    "class E(Enum):\n    A = 'a'\n"
    "class M(BaseModel):\n    x: E = E.A.value\n"
)
ATTACKS["enum dunder"] = (
    "from enum import Enum\n"
    "from pydantic import BaseModel\n"
    "class E(Enum):\n    A = 'a'\n"
    "class M(BaseModel):\n    x: E = E.__members__\n"
)
ATTACKS["enum unknown member"] = (
    "from enum import Enum\n"
    "from pydantic import BaseModel\n"
    "class E(Enum):\n    A = 'a'\n"
    "class M(BaseModel):\n    x: E = E.B\n"
)
ATTACKS["enum used before its definition"] = (
    "from enum import Enum\n"
    "from pydantic import BaseModel\n"
    "class M(BaseModel):\n    x: str = E.A\n"
    "class E(Enum):\n    A = 'a'\n"
)
ATTACKS["enum referenced inside itself"] = (
    "from enum import Enum\nclass E(Enum):\n    A = 'a'\n    B = E.A\n"
)
ATTACKS["non-enum class member"] = (
    "from pydantic import BaseModel\n"
    "class A(BaseModel):\n    x: int = 1\n"
    "class M(BaseModel):\n    y: int = A.x\n"
)
ATTACKS["slice"] = HEADER + "class M(BaseModel):\n    x: list[int] = [1][0:1]\n"
ATTACKS["subscript of a call"] = (
    HEADER + "class M(BaseModel):\n    x: int = Field(1)[0]\n"
)


@pytest.fixture
def no_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a source reaches the worker process."""

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("an unsafe source was sent to the worker")

    monkeypatch.setattr(schemas, "run_worker", boom)


@pytest.mark.parametrize("source", ATTACKS.values(), ids=ATTACKS.keys())
def test_attacks_are_rejected_at_the_ast_level(source: str) -> None:
    with pytest.raises(SourceError) as info:
        check_source(source)
    assert info.value.code == "unsafe_source"
    assert info.value.message


@pytest.mark.parametrize("source", ATTACKS.values(), ids=ATTACKS.keys())
def test_attacks_are_rejected_as_unsafe(
    client: TestClient, no_worker: None, source: str
) -> None:
    response = parse(client, source)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "unsafe_source"
    assert detail["message"]


def test_the_reproductions_leave_no_trace(
    client: TestClient, tmp_path: pathlib.Path
) -> None:
    target = tmp_path / "pwned.txt"
    source = (
        "import uuid\n"
        "from pydantic import BaseModel\n"
        "Field = uuid.os.system\n"
        f'r = Field("id > {target}")\n'
        "class M(BaseModel):\n    x: int\n"
    )
    assert parse(client, source).json()["detail"]["code"] == "unsafe_source"
    assert not target.exists()


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


# -- What must keep working -------------------------------------------------- #

LEGIT = {
    "nested models, list[Model], Optional, Literal": """
from typing import Literal, Optional
from pydantic import BaseModel, Field

class Item(BaseModel):
    sku: str
    qty: int = Field(default=1, ge=0)

class Order(BaseModel):
    items: list[Item]
    note: Optional[str] = None
    kind: Literal["web", "shop"] = "web"
""",
    "date and datetime as bare names": """
from datetime import date, datetime
from pydantic import BaseModel

class E(BaseModel):
    day: date
    at: datetime | None = None
""",
    "datetime.date and datetime.datetime": """
import datetime
from pydantic import BaseModel

class E(BaseModel):
    day: datetime.date
    at: datetime.datetime | None = None
    gap: datetime.timedelta | None = None
    clock: datetime.time | None = None
""",
    "qualified pydantic, typing, decimal, uuid": """
import decimal, typing, uuid, pydantic

class E(pydantic.BaseModel):
    price: decimal.Decimal = pydantic.Field(description="Price", ge=0)
    id: uuid.UUID
    tags: typing.Optional[typing.List[str]] = None
    kind: typing.Literal["a", "b"] = "a"
""",
    "module alias": """
import datetime as dt
from pydantic import BaseModel

class E(BaseModel):
    day: dt.date
""",
    "enums": """
from enum import Enum, IntEnum, StrEnum, auto
import enum
from pydantic import BaseModel

class Color(str, Enum):
    RED = "red"
    BLUE = "blue"

class Level(IntEnum):
    LOW = 1
    HIGH = 2

class Mode(enum.Enum):
    A = auto()
    B = auto()

class Shade(StrEnum):
    DARK = "dark"

class E(BaseModel):
    color: Color = Color.RED
    level: Level = Level.HIGH
    mode: Mode = Mode.A
    shade: Shade = Shade.DARK
""",
    "constraints": """
from typing import Annotated
from annotated_types import Ge, Le, MinLen
import annotated_types
from pydantic import BaseModel, Field, conint

class E(BaseModel):
    a: Annotated[int, Ge(0), Le(10)]
    b: Annotated[str, annotated_types.MinLen(2)]
    c: Annotated[list[int], MinLen(1)]
    d: int = Field(ge=0, le=9, multiple_of=3)
    e: str = Field(min_length=1, max_length=5, pattern="^a")
    f: conint(ge=0) = 0
""",
    "default factories": """
from pydantic import BaseModel, Field

class E(BaseModel):
    a: list[str] = Field(default_factory=list)
    b: dict[str, int] = Field(default_factory=dict)
    c: set[int] = Field(default_factory=set)
""",
    "config, docstrings, aliases and examples": """
from pydantic import BaseModel, ConfigDict, Field

class E(BaseModel):
    \"\"\"A record.\"\"\"

    model_config = ConfigDict(extra="forbid")
    first: str = Field(alias="first-name", description="He said \\"hi\\"")
    n: float = Field(-1.5, examples=[1, 2])
""",
    "constants and field names that look like types": """
from pydantic import BaseModel, Field

LIMIT = 5

class E(BaseModel):
    date: str | None = None
    time: str = Field(description="free text")
    id: str
    type: str = "x"
    count: int = Field(le=LIMIT)
""",
    "models that inherit": """
from pydantic import BaseModel

class Base(BaseModel):
    id: int

class Child(Base):
    name: str
""",
}


@pytest.mark.parametrize("source", LEGIT.values(), ids=LEGIT.keys())
def test_legitimate_models_parse(client: TestClient, source: str) -> None:
    check_source(source)
    response = parse(client, source)
    assert response.status_code == 200, response.text
    assert response.json()["schemas"]


def test_enum_defaults_and_dates_are_in_the_schema(client: TestClient) -> None:
    source = LEGIT["datetime.date and datetime.datetime"]
    schema = parse(client, source).json()["schemas"][0]["json_schema"]
    assert schema["properties"]["day"]["format"] == "date"
    assert schema["properties"]["at"]["anyOf"][0]["format"] == "date-time"


FIXTURES = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "frontend_pydantic.json").read_text()
)


@pytest.mark.parametrize("source", FIXTURES.values(), ids=FIXTURES.keys())
def test_the_source_the_frontend_generates_is_accepted(
    client: TestClient, source: str
) -> None:
    """``fieldsToPydantic`` in ``demo/frontend/src/lib/schema.ts`` wrote these."""
    check_source(source)
    response = parse(client, source)
    assert response.status_code == 200, response.text
    assert response.json()["schemas"]


def test_the_frontend_invoice_keeps_its_descriptions(client: TestClient) -> None:
    schemas_ = parse(client, FIXTURES["preset_invoice"]).json()["schemas"]
    props = schemas_[0]["json_schema"]["properties"]
    assert props["issue_date"]["description"] == "date the invoice was issued"
    assert props["currency"]["enum"] == ["USD", "EUR", "GBP"]


# -- The worker's answers map to HTTP errors ---------------------------------- #


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (SandboxTimeout("slow"), 400, "timeout"),
        (SandboxCrash("died"), 400, "invalid_model"),
        (SandboxUnavailable("no python"), 503, "sandbox_unavailable"),
    ],
)
def test_sandbox_failures_map_to_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: int,
    code: str,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(schemas, "run_worker", fail)
    response = parse(client, LEGIT["enums"])
    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["code"] == code and detail["message"]


def test_endpoints_run_in_the_thread_pool() -> None:
    """A coroutine would block the event loop while it waits for a worker."""
    assert not inspect.iscoroutinefunction(app_main.schema_parse)
    assert not inspect.iscoroutinefunction(app_main.document)
