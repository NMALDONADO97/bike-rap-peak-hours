.PHONY: all install run test clean help

# Default target
all: run

# Detect Python
PYTHON := $(shell command -v python3 || command -v python)
VENV := .venv
BIN := $(VENV)/bin

## install   : Install dependencies into a virtual environment
install:
	@$(PYTHON) run.py --install-only || (echo "Error: run.py failed to install requirements" && exit 1)

## run       : Run the analysis pipeline
run:
	@$(PYTHON) run.py

## test      : Run smoke tests
test: install
	@$(BIN)/python -m pytest tests/

## clean     : Remove virtual environment and generated outputs
clean:
	rm -rf $(VENV) .venv-win
	rm -rf data/processed outputs reports
	find . -type d -name "__pycache__" -exec rm -rf {} +

## help      : Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^##' Makefile | sed -e 's/## //g' -e 's/ :/:/g' | column -t -s ':'
