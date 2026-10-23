"""Turn an uploaded file into plain text for the extraction form."""

import pathlib
from typing import Any, Literal

from fastapi import HTTPException

from app._sandbox import (
    Limits,
    SandboxCrash,
    SandboxTimeout,
    SandboxUnavailable,
    run_worker,
)

__all__ = ["MAX_BYTES", "MAX_CHARS", "MAX_PAGES", "PDF_LIMITS", "read_document"]

MAX_BYTES = 10 * 1024 * 1024
"""Largest accepted upload, in bytes."""

MAX_CHARS = 200_000
"""Largest accepted text, in characters (also the cap of ``/api/extract``)."""

MAX_PAGES = 300
"""Most pages of a PDF that are read."""

PDF_LIMITS = Limits(timeout=20.0, cpu_seconds=15, memory_bytes=1024 * 1024 * 1024)
"""Limits of the PDF worker: 20 s wall clock, 15 s CPU, 1 GB of address space."""

Kind = Literal["markdown", "pdf", "text"]

_KINDS: dict[str, Kind] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
}


def _error(status: int, code: str, message: str) -> HTTPException:
    """Build the ``{"detail": {"code", "message"}}`` error of this API."""
    return HTTPException(status, detail={"code": code, "message": message})


def _pdf_text(data: bytes) -> tuple[str, int]:
    """Extract the text of a PDF in a sandboxed worker process.

    pypdf runs in ``app._pdf_worker`` (isolated interpreter, no credentials, CPU,
    memory and wall-clock limits), so that a crafted file cannot hang or exhaust
    the server. It stops once the text passes ``MAX_CHARS`` and refuses more than
    ``MAX_PAGES`` pages.

    Args:
        data: The bytes of the file.

    Returns:
        The page texts joined with a blank line, and the number of pages.

    Raises:
        HTTPException: 413 ``too_large`` over ``MAX_PAGES`` pages or ``MAX_CHARS``
            characters; 422 ``pdf_parse_error`` if the file cannot be parsed, is
            password protected or takes too long; 422 ``pdf_no_text`` if it has
            no text layer (a scan); 503 ``sandbox_unavailable`` if the worker
            cannot be started.
    """
    try:
        answer = run_worker(
            "app._pdf_worker",
            data,
            PDF_LIMITS,
            (f"--max-pages={MAX_PAGES}", f"--max-chars={MAX_CHARS}"),
        )
    except SandboxTimeout as exc:
        raise _error(
            422,
            "pdf_parse_error",
            f"The PDF took too long to read ({PDF_LIMITS.timeout:g} s limit).",
        ) from exc
    except SandboxCrash as exc:
        raise _error(
            422, "pdf_parse_error", f"The PDF could not be read ({exc})."
        ) from exc
    except SandboxUnavailable as exc:
        raise _error(503, "sandbox_unavailable", str(exc)) from exc
    if not isinstance(answer, dict):
        raise _error(422, "pdf_parse_error", "The PDF could not be read.")
    if answer.get("ok") is True:
        return str(answer["text"]), int(answer["pages"])
    error = answer.get("error")
    if error == "too_many_pages":
        raise _error(413, "too_large", f"The PDF has more than {MAX_PAGES} pages.")
    if error == "too_large":
        raise _error(413, "too_large", f"The text is over {MAX_CHARS:,} characters.")
    if error == "encrypted":
        raise _error(422, "pdf_parse_error", "The PDF is password protected.")
    if error == "no_text":
        raise _error(
            422,
            "pdf_no_text",
            "The PDF has no text layer (it may be a scan); paste the text instead.",
        )
    raise _error(
        422, "pdf_parse_error", f"The PDF could not be read: {answer.get('message')}"
    )


def read_document(filename: str, data: bytes) -> dict[str, Any]:
    """Decode an uploaded document.

    Args:
        filename: The name the client gave the file; its extension decides how
            the file is read.
        data: The bytes of the file.

    Returns:
        ``{"filename", "kind", "text", "pages", "chars"}``; ``pages`` is ``None``
        unless the file is a PDF.

    Raises:
        HTTPException: 415 ``unsupported_type``, 413 ``too_large``, or 422 for a
            PDF that cannot be read or has no text.
    """
    suffix = pathlib.PurePath(filename).suffix.lower()
    kind = _KINDS.get(suffix)
    if kind is None:
        raise _error(
            415,
            "unsupported_type",
            f"Unsupported file type {suffix or '(none)'!r}; "
            "upload a .md, .markdown, .txt or .pdf file.",
        )
    if len(data) > MAX_BYTES:
        raise _error(
            413, "too_large", f"The file is over {MAX_BYTES // (1024 * 1024)} MB."
        )
    pages: int | None = None
    if kind == "pdf":
        text, pages = _pdf_text(data)
    else:
        text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_CHARS:
        raise _error(413, "too_large", f"The text is over {MAX_CHARS:,} characters.")
    return {
        "filename": filename,
        "kind": kind,
        "text": text,
        "pages": pages,
        "chars": len(text),
    }
