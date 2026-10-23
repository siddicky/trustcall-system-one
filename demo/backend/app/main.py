"""FastAPI backend of the trustcall-system-one demo UI."""

import os
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from trustcall_system_one import Transport, __version__

from app.documents import MAX_BYTES, read_document
from app.extract import ExtractRequest, detect_provider, get_transport, run_extraction
from app.limits import BodyLimitMiddleware, default_limits
from app.schemas import SourceError, parse_models, to_http

__all__ = ["app", "get_transport"]

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

app = FastAPI(
    title="trustcall-system-one demo",
    version=__version__,
    description="Backend of the demo UI of trustcall-system-one.",
)


def allowed_origins() -> list[str]:
    """Read the CORS origins from ``DEMO_ALLOWED_ORIGINS`` (comma separated)."""
    raw = os.environ.get("DEMO_ALLOWED_ORIGINS") or DEFAULT_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Added first, so that CORS wraps it and a 413 still carries the CORS headers.
app.add_middleware(BodyLimitMiddleware, limits=default_limits())
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class SchemaParseRequest(BaseModel):
    """Body of ``POST /api/schema/parse``.

    Attributes:
        source: Python source that defines pydantic models.
    """

    source: str


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Report that the API is up, the configured provider and the library version."""
    return {"ok": True, "provider": detect_provider(), "version": __version__}


@app.post("/api/document")
def document(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    """Read an uploaded .md, .markdown, .txt or .pdf file into text.

    A plain ``def``, so that it runs in the thread pool: PDF parsing waits on a
    worker process and must not block the event loop.
    """
    data = file.file.read(MAX_BYTES + 1)
    return read_document(file.filename or "", data)


@app.post("/api/schema/parse")
def schema_parse(body: SchemaParseRequest) -> dict[str, Any]:
    """Turn the source of pydantic models into JSON Schemas.

    The source is checked against an allowlist, then built in a sandboxed worker
    process; see ``app.schemas``. This is a plain ``def`` so that it runs in the
    thread pool while it waits for the worker.
    """
    try:
        return {"schemas": parse_models(body.source)}
    except SourceError as exc:
        raise to_http(exc) from exc


@app.post("/api/extract")
async def extract(
    body: ExtractRequest,
    transport: Annotated[Transport | None, Depends(get_transport)],
) -> dict[str, Any]:
    """Extract records from text, or update existing records from it."""
    return await run_extraction(body, transport)
