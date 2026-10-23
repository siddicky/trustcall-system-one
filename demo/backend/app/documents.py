"""Turn an uploaded file into plain text for the extraction form."""

import io
import pathlib
from typing import Any, Literal

from fastapi import HTTPException
from pypdf import PdfReader

__all__ = ["MAX_BYTES", "MAX_CHARS", "read_document"]

MAX_BYTES = 10 * 1024 * 1024
"""Largest accepted upload, in bytes."""

MAX_CHARS = 200_000
"""Largest accepted text, in characters (also the cap of ``/api/extract``)."""

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
    """Extract the text of a PDF.

    Args:
        data: The bytes of the file.

    Returns:
        The page texts joined with a blank line, and the number of pages.

    Raises:
        HTTPException: 422 ``pdf_parse_error`` if the file cannot be parsed and
            422 ``pdf_no_text`` if it has no text layer (a scan).
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise _error(422, "pdf_parse_error", "The PDF is password protected.")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except HTTPException:
        raise
    except Exception as exc:  # pypdf raises many types on damaged files
        raise _error(
            422, "pdf_parse_error", f"The PDF could not be read: {exc}"
        ) from exc
    text = "\n\n".join(pages).strip()
    if not text:
        raise _error(
            422,
            "pdf_no_text",
            "The PDF has no text layer (it may be a scan); paste the text instead.",
        )
    return text, len(pages)


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
