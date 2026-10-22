.PHONY: help install test cov smoke record evals lint format build clean

# help: List the available targets
help:
	@awk '/^# help: /{desc = substr($$0, 9); next} /^[a-zA-Z_-]+:/ && desc != "" {printf "  %-10s %s\n", substr($$0, 1, index($$0, ":") - 1), desc; desc = ""}' $(MAKEFILE_LIST)

# help: Install the package with all extras and the dev group
install:
	uv sync --all-extras --group dev

# help: Unit tests with the network disabled
test:
	uv run pytest --disable-socket --allow-unix-socket -m "not live and not eval" tests/unit

# help: Unit tests with a coverage report
cov:
	uv run pytest --disable-socket --allow-unix-socket -m "not live and not eval" tests/unit --cov=trustcall_system_one --cov-report=term-missing

# Live smoke tests: need TRUSTCALL_S1_LIVE=1 (set here) and the Cloudflare
# credentials (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN) in the environment.
# help: Live smoke tests against the Cloudflare API (needs credentials)
smoke:
	TRUSTCALL_S1_LIVE=1 uv run pytest -m live -s tests/live

# The same, writing each scenario's requests and responses to tests/recorded.
# help: Live smoke tests, recording requests and responses to tests/recorded
record:
	TRUSTCALL_S1_LIVE=1 TRUSTCALL_S1_RECORD=tests/recorded uv run pytest -m live -s tests/live

# The eval dataset against the scripted worlds (offline). Live, replayed or swept:
# see docs/LIVE_RUNBOOK.md (`uv run python -m evals run --live`).
# help: Eval dataset against the scripted worlds (offline)
evals:
	uv run python -m evals run

# help: Lock check, ruff lint, ruff format check and mypy (strict)
lint:
	uv lock --check
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

# help: Apply import sorting, ruff formatting and autofixes
format:
	uv run ruff check --select I --fix .
	uv run ruff format .
	uv run ruff check --fix .

# help: Build the sdist and wheel into dist/
build:
	uv build

# help: Remove build, cache and coverage artifacts
clean:
	rm -rf dist build htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache
	find . \( -path ./.venv -o -path ./.claude -o -path ./.git \) -prune -o -name __pycache__ -type d -exec rm -rf {} +
