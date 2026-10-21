# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this is

trustcall-system-one is a reimplementation of the ideas behind
[trustcall](https://github.com/hinthornw/trustcall): extract structured data
from text, and update existing structured data through RFC-6902 JSON Patch
operations. It runs on Cloudflare's System One models: Clairic
(`@cf/cloudflare/clairic`, typed extraction) and Clef (`@cf/cloudflare/clef`,
decisions). The authoritative design is `docs/DESIGN.md`. The authoritative
API schemas for both models are in `docs/schemas/`.

## Rules

- Use `uv` only. Never use pip.
- Run `make lint` and `make test` before every commit.
- Write Google-style docstrings.
- Keep full type hints. mypy runs in strict mode.
- Expose the public API only through `trustcall_system_one/__init__.py`.
- Name internal modules with a leading underscore.
- Tests never touch the network. The unit suite runs with `--disable-socket`.
- Live tests are marked `live` and are gated on the `TRUSTCALL_S1_LIVE`
  environment variable.

## Licensing

This repository is public domain under the Unlicense. The reference
implementation at `/home/user/trustcall` (upstream
github.com/hinthornw/trustcall) is MIT-licensed. Reimplement its behaviour
from the design doc. Never copy its code, and credit it in the README.

## Layout

Planned modules, from the "Package layout" table in `docs/DESIGN.md`:

- `trustcall_system_one/__init__.py`: public exports only.
- `trustcall_system_one/_errors.py`: exceptions.
- `trustcall_system_one/_types.py`: result types, `SchemaInstance`, `Mention` and `Entity`.
- `trustcall_system_one/_config.py`: all config dataclasses.
- `trustcall_system_one/_cardinality.py`: `Many` and `One` wrappers, `Annotated` markers, default resolution.
- `trustcall_system_one/transport.py` (public): `Transport` protocol, `CloudflareTransport`, adaptive per-model limiter.
- `trustcall_system_one/_wire.py`: exact Clairic and Clef request and response pydantic models.
- `trustcall_system_one/_schemas.py`: coerces BaseModel, TypedDict, JSON Schema or callable into a BaseModel.
- `trustcall_system_one/_names.py`: path to wire-name map, with deterministic blake2s shortening.
- `trustcall_system_one/_compile.py`: schema compiler producing a `SchemaPlan`.
- `trustcall_system_one/_flatten.py`: flatten, unflatten and prune_nulls.
- `trustcall_system_one/_coerce.py`: converts a model answer into a Python value.
- `trustcall_system_one/_render.py`: Clairic text state (focus, input and existing blocks, offsets) and Clef structured state.
- `trustcall_system_one/_collection.py`: collection engine.
- `trustcall_system_one/_coref.py`: mention clustering.
- `trustcall_system_one/_match.py`: record matching and assignment.
- `trustcall_system_one/_entity.py`: per-entity passes.
- `trustcall_system_one/_verify.py`: attribution, cross-check, re-ask and borderline-null handling.
- `trustcall_system_one/_diff.py`: ops and apply.
- `trustcall_system_one/_gates.py`: thresholds and the change gate.
- `trustcall_system_one/_repair.py`: repair loop.
- `trustcall_system_one/_run.py`: per-invoke context (usage, call log, semaphores).
- `trustcall_system_one/_extractor.py`: extractor factory and sync bridge.
- `trustcall_system_one/testing.py` (public): `FakeTransport`, `FakeWorld`, recording and cassette transports.
