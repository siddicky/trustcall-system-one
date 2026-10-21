.PHONY: install test cov smoke record evals lint format build

install:
	uv sync --all-extras --group dev

test:
	uv run pytest --disable-socket --allow-unix-socket -m "not live and not eval" tests/unit

cov:
	uv run pytest --disable-socket --allow-unix-socket -m "not live and not eval" tests/unit --cov=trustcall_system_one --cov-report=term-missing

smoke:
	TRUSTCALL_S1_LIVE=1 uv run pytest -m live tests/live

record:
	TRUSTCALL_S1_LIVE=1 TRUSTCALL_S1_RECORD=1 uv run pytest -m live tests/live

evals:
	TRUSTCALL_S1_LIVE=1 uv run pytest -m eval tests/evals

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --select I --fix .
	uv run ruff format .
	uv run ruff check --fix .

build:
	uv build
