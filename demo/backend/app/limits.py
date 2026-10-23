"""Request body size limits, enforced before a body is parsed or buffered.

Starlette reads a whole multipart upload or JSON body before the endpoint runs, so
a limit checked in the endpoint comes too late. ``BodyLimitMiddleware`` is a pure ASGI
middleware that looks at the ``Content-Length`` header first (413 without reading
a byte) and also counts the bytes it passes on, which covers chunked bodies and
a header that lies: past the limit it raises a 413 ``HTTPException`` from the
``receive`` call the application is reading the body with.
"""

from typing import Any

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.documents import MAX_BYTES

__all__ = [
    "DOCUMENT_LIMIT",
    "EXTRACT_LIMIT",
    "SCHEMA_LIMIT",
    "BodyLimitMiddleware",
    "default_limits",
]

MULTIPART_OVERHEAD = 64 * 1024
"""Allowance for the boundary, part headers and the file name around an upload."""

DOCUMENT_LIMIT = MAX_BYTES + MULTIPART_OVERHEAD
"""Largest ``/api/document`` request body, in bytes."""

SCHEMA_LIMIT = 1024 * 1024
"""Largest ``/api/schema/parse`` request body (1 MB; the source is at most 20,000
characters)."""

EXTRACT_LIMIT = 2 * 1024 * 1024
"""Largest ``/api/extract`` request body (2 MB: 200,000 characters of text, which
JSON can escape to 6 bytes each, plus schemas and existing records)."""


def default_limits() -> dict[str, int]:
    """The body limit of each limited path, in bytes."""
    return {
        "/api/document": DOCUMENT_LIMIT,
        "/api/schema/parse": SCHEMA_LIMIT,
        "/api/extract": EXTRACT_LIMIT,
    }


def _too_large(limit: int) -> dict[str, Any]:
    return {
        "code": "too_large",
        "message": f"The request body is over {limit:,} bytes.",
    }


class BodyLimitMiddleware:
    """Reject request bodies over a per-path limit with 413 ``too_large``."""

    def __init__(self, app: ASGIApp, limits: dict[str, int]) -> None:
        """Create the middleware.

        Args:
            app: The wrapped ASGI application.
            limits: Maximum body size in bytes by exact request path.
        """
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI call."""
        limit = (
            self.limits.get(scope["path"].rstrip("/"))
            if scope["type"] == "http"
            else None
        )
        if limit is None:
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > limit
            except ValueError:
                response = JSONResponse(
                    {
                        "detail": {
                            "code": "bad_request",
                            "message": "Invalid Content-Length header.",
                        }
                    },
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if too_big:
                response = JSONResponse({"detail": _too_large(limit)}, status_code=413)
                await response(scope, receive, send)
                return

        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, detail=_too_large(limit))
            return message

        await self.app(scope, counted_receive, send)
