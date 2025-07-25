.PHONY: clean clean-pyc clean-build clean-test install develop sync lock all test lint format check coverage report

PYTHON := python
SRC_DIR := src
TEST_DIR := tests
COV_TARGET := src/ask_llm

help:
	@echo "Available targets:"
	@echo "  install        Install the package in editable mode with uv."
	@echo "  develop        Install the package with development dependencies."
	@echo "  sync           Sync dependencies using uv (if uv.lock exists)."
	@echo "  lock           Generate uv.lock file."
	@echo "  clean          Remove temporary files and build artifacts."
	@echo "  clean-pyc      Remove Python file artifacts."
	@echo "  clean-build    Remove build artifacts."
	@echo "  clean-test     Remove test and coverage artifacts."
	@echo "  clean-venv     Remove virtual environments."
	@echo "  test           Run tests with coverage report."
	@echo "  coverage       Alias for 'test'."
	@echo "  report         Run tests and generate an HTML coverage report."
	@echo "  lint           Run Ruff linter."
	@echo "  format         Run Ruff formatter."
	@echo "  check          Run all checks (lint, format --check, type check)."

clean: clean-pyc clean-build clean-test clean-bs

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.so' -exec rm -f {} +
	find . -name '*.c' -type f -name "*.py[cod]" -exec rm -f {} +
	find . -name '*$py.class' -exec rm -f {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf pip-wheel-metadata/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '.eggs' -exec rm -rf {} +
	rm -rf *.egg-info/

clean-test:
	rm -rf .tox/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .hypothesis/
	rm -rf .ruff_cache/
	rm -f coverage.xml
	rm -f *.cover
	rm -rf cython_debug/
	rm -rf .dmypy.json
	rm -rf dmypy.json

clean-bs:
	find . -name '._*' -exec rm -f {} +
clean-venv:
	rm -rf venv/
	rm -rf .venv/
	rm -rf env/
	rm -rf ENV/

all: install

install-deps:
	@echo "Installing additional dependencies for huggingface and llamacpp..."
	@echo "Attempting to uninstall existing llama-cpp-python..."
	@uv pip uninstall llama-cpp-python || true
	@echo "Installing llama-cpp-python with CUDA support..."
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "CUDA compiler found, installing with CUDA support..."; \
		CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 CUDACXX=/usr/local/cuda/bin/nvcc uv pip install llama-cpp-python --no-cache-dir; \
	else \
		echo "CUDA compiler not found, installing CPU-only version..."; \
		uv pip install llama-cpp-python --no-cache-dir; \
	fi
	@echo "Installing huggingface-hub..."
	@uv pip install huggingface-hub

install:
	@echo "Setting up environment in .venv"
	@if [ ! -d .venv ]; then \
		echo "Creating new virtual environment with uv"; \
		uv venv .venv --python 3.12.11; \
	else \
		echo "Virtual environment already exists"; \
	fi
	@echo "Installing/updating package in development mode with uv"
	@uv pip install -e .
	@echo "Setup complete. Activate with: source .venv/bin/activate"
	@make install-deps

develop:
	@echo "Installing package with development dependencies using uv"
	@uv pip install -e ".[dev]" # Assuming a [dev] extra for dev dependencies

sync:
	@echo "Syncing dependencies with uv..."
	@if [ -f uv.lock ]; then \
		uv sync; \
	else \
		echo "No uv.lock file found. Run 'make lock' first or use 'make install'."; \
	fi

lock:
	@echo "Generating uv.lock file..."
	@uv lock

test:
	@echo "Running tests with coverage..."
	uv run pytest --cov=$(COV_TARGET) --cov-report=term-missing $(TEST_DIR)

coverage: test # Alias for running tests with coverage

report:
	@echo "Generating HTML coverage report..."
	uv run pytest --cov=$(COV_TARGET) --cov-report=html $(TEST_DIR)
	@echo "HTML report generated in htmlcov/ directory."

lint:
	@echo "Running Ruff linter..."
	uv run ruff check $(SRC_DIR) $(TEST_DIR)

format:
	@echo "Running Ruff formatter..."
	uv run ruff format $(SRC_DIR) $(TEST_DIR)

check:
	@echo "Running Ruff check and format..."
	uv run ruff check $(SRC_DIR) $(TEST_DIR)
	uv run ruff format $(SRC_DIR) $(TEST_DIR) --check
	@echo "Running MyPy type checker..."
	uv run mypy $(SRC_DIR) 