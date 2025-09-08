.PHONY: format lint

RUFF := $(shell command -v ruff 2> /dev/null || echo "uv run ruff")

format:
	$(RUFF) check --fix
	$(RUFF) format

lint:
	$(RUFF) check
	$(RUFF) format --check
