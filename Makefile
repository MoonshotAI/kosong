.PHONY: format check test

RUFF := $(shell command -v ruff 2> /dev/null || echo "uv run ruff")
PYRIGHT := $(shell command -v pyright 2> /dev/null || echo "uv run pyright")

format:
	$(RUFF) check --fix
	$(RUFF) format

check:
	$(RUFF) check
	$(RUFF) format --check
	$(PYRIGHT)

test:
	uv run pytest --doctest-modules -vv
