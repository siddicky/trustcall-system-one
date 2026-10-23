"""Worker process of ``POST /api/document`` for PDFs: read the text of a PDF.

Started by ``app.documents`` through ``app._sandbox.run_worker`` (``python -I``, no
credentials in the environment, resource limits, timeout), because a crafted PDF can
make pypdf use unbounded time and memory (decompression bombs, page trees that loop).
It reads the file from stdin and writes one JSON object to stdout:

* ``{"ok": true, "text": str, "pages": int}``
* ``{"ok": false, "error": "parse" | "encrypted" | "too_many_pages" | "too_large" |
  "no_text", "message": str, "pages": int | None}``

It stops reading pages as soon as the text passes ``--max-chars`` and refuses a file
with more than ``--max-pages`` pages before it extracts anything.
"""

import io
import json
import sys
from typing import Any

from app._sandbox import apply_limits

__all__ = ["main"]


def _read(data: bytes, max_pages: int, max_chars: int) -> dict[str, Any]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            return {"ok": False, "error": "encrypted", "message": "password protected"}
        count = len(reader.pages)
        if count > max_pages:
            return {
                "ok": False,
                "error": "too_many_pages",
                "message": f"{count} pages",
                "pages": count,
            }
        texts: list[str] = []
        total = 0
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            texts.append(text)
            total += len(text) + 2
            if total - 2 > max_chars:
                return {"ok": False, "error": "too_large", "message": "too much text"}
    except Exception as exc:  # pypdf raises many types on damaged files
        return {"ok": False, "error": "parse", "message": str(exc)[:300]}
    joined = "\n\n".join(texts).strip()
    if not joined:
        return {"ok": False, "error": "no_text", "message": "no text layer"}
    return {"ok": True, "text": joined, "pages": count}


def main(argv: list[str]) -> int:
    """Entry point: apply the limits, read the file, write the answer.

    Args:
        argv: Limit arguments (see ``app._sandbox.apply_limits``), ``--max-pages=N``
            and ``--max-chars=N``.

    Returns:
        The exit status.
    """
    rest = apply_limits(argv)
    values = dict(arg.removeprefix("--").split("=", 1) for arg in rest)
    answer = _read(
        sys.stdin.buffer.read(),
        int(values["max-pages"]),
        int(values["max-chars"]),
    )
    sys.stdout.write(json.dumps(answer))
    sys.stdout.flush()
    return 0
