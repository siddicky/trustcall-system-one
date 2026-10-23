"""Tests of the process boundary of POST /api/schema/parse.

The allowlist is the first layer. These tests switch it off inside the worker
(``skip_check=True``, a flag only tests pass) to prove that the worker process is
safe on its own: no credentials in its environment, a timeout, resource limits.
"""

import builtins
import pathlib
import time

import pytest

from app import schemas
from app._sandbox import Limits, child_env
from app._schema_guard import SourceError

BASE = "import uuid\nfrom pydantic import BaseModel, Field\n"


def describe(expression: str) -> str:
    """A source whose field description is the value of ``expression``."""
    return (
        f"{BASE}class M(BaseModel):\n"
        f"    x: str = Field(default='', description=str({expression}))\n"
    )


def description_of(source: str, **kwargs: object) -> str:
    result = schemas.run_schema_worker(source, skip_check=True, **kwargs)  # type: ignore[arg-type]
    description = result[0]["json_schema"]["properties"]["x"]["description"]
    assert isinstance(description, str)
    return description


def test_the_check_is_off_only_with_the_test_flag() -> None:
    """Without the flag the worker refuses what the server's check would refuse."""
    source = describe("uuid.os.getenv('PATH')")
    assert description_of(source)  # reachable only with the check off
    with pytest.raises(SourceError) as info:
        schemas.run_schema_worker(source)
    assert info.value.code == "unsafe_source"


def test_the_worker_checks_again_even_if_the_server_did_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(schemas, "check_source", lambda source: None)
    with pytest.raises(SourceError) as info:
        schemas.parse_models("import os\n")
    assert info.value.code == "unsafe_source"
    with pytest.raises(SourceError) as info:
        schemas.parse_models(describe("uuid.os.getenv('CLOUDFLARE_API_TOKEN')"))
    assert info.value.code == "unsafe_source"


def test_the_worker_has_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = {
        "CLOUDFLARE_API_TOKEN": "cf-secret-token",
        "CLOUDFLARE_ACCOUNT_ID": "cf-account",
        "TYPESAFE_API_KEY": "ts-secret-key",
        "TRUSTCALL_S1_PROVIDER": "typesafe",
        "SOME_OTHER_SECRET": "other-secret",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)

    # The payload of the review, with the allowlist bypassed.
    leaked = description_of(describe("uuid.os.getenv('CLOUDFLARE_API_TOKEN')"))
    assert leaked == "None"

    environment = description_of(describe("dict(uuid.os.environ)"))
    assert "PATH" in environment  # the payload does reach os.environ
    assert "PYTHONHASHSEED" in environment
    for name, value in secrets.items():
        assert name not in environment
        assert value not in environment
    assert "CLOUDFLARE" not in environment and "TYPESAFE" not in environment


def test_child_env_is_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "secret")
    monkeypatch.setenv("HOME", "/root")
    env = child_env()
    assert "CLOUDFLARE_API_TOKEN" not in env and "HOME" not in env
    assert set(env) <= {
        "PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "SYSTEMROOT",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
    }
    assert env["PYTHONHASHSEED"] == "0"


def test_the_worker_runs_in_a_temporary_directory() -> None:
    cwd = description_of(describe("uuid.os.getcwd()"))
    assert "demo-sandbox-" in cwd
    assert not pathlib.Path(cwd).exists()  # and it is gone afterwards


def test_a_loop_times_out() -> None:
    source = "while True:\n    pass\n"
    started = time.monotonic()
    with pytest.raises(SourceError) as info:
        schemas.run_schema_worker(
            source, skip_check=True, limits=Limits(timeout=1.0, cpu_seconds=30)
        )
    assert info.value.code == "timeout"
    assert info.value.status == 400
    assert time.monotonic() - started < 4


def test_the_loop_is_also_stopped_by_the_cpu_limit() -> None:
    """The worker kills itself on its own CPU budget, whatever the parent does."""
    started = time.monotonic()
    with pytest.raises(SourceError) as info:
        schemas.run_schema_worker(
            "while True:\n    pass\n",
            skip_check=True,
            limits=Limits(timeout=30.0, cpu_seconds=1),
        )
    assert info.value.code == "timeout"
    assert time.monotonic() - started < 10


def test_memory_is_limited() -> None:
    source = f"{BASE}x = bytes(4_000_000_000)\nclass M(BaseModel):\n    y: int\n"
    with pytest.raises(SourceError) as info:
        schemas.run_schema_worker(source, skip_check=True)
    assert info.value.code == "invalid_model"
    assert "MemoryError" in info.value.message


def test_the_worker_cannot_write_files() -> None:
    source = (
        f"{BASE}fd = uuid.os.open('out.txt', 65)\n"
        "uuid.os.write(fd, b'x')\n"
        "class M(BaseModel):\n    y: int\n"
    )
    with pytest.raises(SourceError) as info:
        schemas.run_schema_worker(source, skip_check=True)
    assert info.value.code == "invalid_model"
    assert "too large" in info.value.message.lower()


def test_a_crash_of_the_worker_is_an_invalid_model() -> None:
    source = f"{BASE}uuid.os._exit(7)\nclass M(BaseModel):\n    y: int\n"
    with pytest.raises(SourceError) as info:
        schemas.run_schema_worker(source, skip_check=True)
    assert info.value.code == "invalid_model"
    assert "sandbox limits" in info.value.message


def test_the_server_process_never_executes_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executing in this process fails; the worker is another process."""

    def no_exec(*args: object, **kwargs: object) -> None:
        raise AssertionError("exec called in the server process")

    monkeypatch.setattr(builtins, "exec", no_exec)
    source = "from pydantic import BaseModel\nclass A(BaseModel):\n    x: int\n"
    assert schemas.parse_models(source)[0]["name"] == "A"
