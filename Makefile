PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all verify lint test

all: lint test verify

verify:
	$(PYTHON) scripts/verify_series.py

test:
	$(PYTHON) -m pytest -q tests

lint:
	$(RUFF) check .
