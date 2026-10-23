"""Run untrusted work in a separate, locked-down Python process.

Two endpoints do work on input from strangers that could harm the server:
``/api/schema/parse`` executes submitted pydantic source and ``/api/document``
parses PDFs. Both do it in a worker process started here:

* ``python -I`` (isolated mode: no ``PYTHON*`` variables, no user site, no
  current directory on ``sys.path``);
* an environment with nothing but ``PATH`` and a few variables Python needs, so no
  ``CLOUDFLARE_*``, ``TYPESAFE_*`` or any other credential;
* a throwaway working directory;
* a wall-clock timeout, enforced by the parent, which kills the worker;
* ``resource`` limits the worker sets on itself before it touches the input: CPU
  time, address space, file size 0, no new processes (best effort, see
  ``apply_limits``), no core dumps.

This module uses the standard library only, because the workers import it.
"""

import contextlib
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ROOT",
    "Limits",
    "SandboxCrash",
    "SandboxError",
    "SandboxTimeout",
    "SandboxUnavailable",
    "apply_limits",
    "child_env",
    "run_worker",
]

ROOT = Path(__file__).resolve().parent.parent
"""The directory that contains the ``app`` package; put on ``sys.path`` of workers."""

MAX_OUTPUT_BYTES = 4 * 1024 * 1024
"""Most a worker may write to stdout."""

_BOOTSTRAP = (
    "import importlib, sys; sys.path.insert(0, sys.argv[1]); "
    "raise SystemExit(importlib.import_module(sys.argv[2]).main(sys.argv[3:]))"
)
# Variables a Python process may need to start; nothing here is a secret.
_PASS_THROUGH = ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "SYSTEMROOT")


class SandboxError(Exception):
    """Base class of the failures of a worker run."""


class SandboxTimeout(SandboxError):
    """The worker ran out of wall-clock or CPU time and was killed."""


class SandboxCrash(SandboxError):
    """The worker died, or wrote output that is not the expected JSON."""


class SandboxUnavailable(SandboxError):
    """The worker process could not be started."""


@dataclass(frozen=True)
class Limits:
    """Resource limits of a worker.

    Attributes:
        timeout: Wall-clock seconds before the parent kills the worker.
        cpu_seconds: ``RLIMIT_CPU`` of the worker.
        memory_bytes: ``RLIMIT_AS`` of the worker.
    """

    timeout: float = 5.0
    cpu_seconds: int = 3
    memory_bytes: int = 512 * 1024 * 1024

    def args(self) -> list[str]:
        """The command-line form the worker reads back with ``worker_main``."""
        return [f"--cpu={self.cpu_seconds}", f"--memory={self.memory_bytes}"]


def child_env() -> dict[str, str]:
    """The environment of a worker: ``PATH`` and the few variables Python needs."""
    env = {k: os.environ[k] for k in _PASS_THROUGH if k in os.environ}
    env.setdefault("PATH", os.defpath)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_worker(
    module: str,
    payload: bytes,
    limits: Limits,
    extra_args: tuple[str, ...] = (),
) -> Any:
    """Run ``module.main(argv)`` in a locked-down subprocess.

    Args:
        module: Dotted name of the worker module (it defines ``main(argv)``).
        payload: Bytes written to the worker's stdin.
        limits: The resource limits.
        extra_args: Further arguments for the worker.

    Returns:
        The JSON the worker wrote to stdout.

    Raises:
        SandboxTimeout: The worker ran too long or used too much CPU time.
        SandboxCrash: The worker died or wrote something that is not JSON.
        SandboxUnavailable: The process could not be started.
    """
    command = [
        sys.executable,
        "-I",
        "-c",
        _BOOTSTRAP,
        str(ROOT),
        module,
        *limits.args(),
        *extra_args,
    ]
    with tempfile.TemporaryDirectory(prefix="demo-sandbox-") as cwd:
        try:
            done = subprocess.run(
                command,
                input=payload,
                capture_output=True,
                timeout=limits.timeout,
                env=child_env(),
                cwd=cwd,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeout(f"no result within {limits.timeout:g} s") from exc
        except OSError as exc:
            raise SandboxUnavailable(f"cannot start the worker: {exc}") from exc
    if done.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
        raise SandboxTimeout("the worker used up its CPU time or was killed")
    if done.returncode != 0:
        raise SandboxCrash(f"the worker exited with status {done.returncode}")
    if len(done.stdout) > MAX_OUTPUT_BYTES:
        raise SandboxCrash("the worker wrote too much")
    try:
        return json.loads(done.stdout)
    except ValueError as exc:
        raise SandboxCrash("the worker wrote invalid JSON") from exc


def apply_limits(argv: list[str]) -> list[str]:
    """Set the resource limits requested in ``argv`` on this process.

    Meant to be the first thing a worker does. Soft and hard limits are both set,
    so the code that runs afterwards cannot raise them (it also cannot import
    ``resource``). ``RLIMIT_NPROC`` is best effort: the kernel does not enforce it
    for root, and it counts every process of the user.

    Args:
        argv: The worker's arguments (``--cpu=N``, ``--memory=N`` and others).

    Returns:
        The arguments that are not limits.
    """
    rest: list[str] = []
    cpu, memory = 3, 512 * 1024 * 1024
    for arg in argv:
        if arg.startswith("--cpu="):
            cpu = int(arg.removeprefix("--cpu="))
        elif arg.startswith("--memory="):
            memory = int(arg.removeprefix("--memory="))
        else:
            rest.append(arg)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    with contextlib.suppress(ValueError, OSError):  # platform dependent
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    return rest
