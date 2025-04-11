.PHONY: clean clean-pyc clean-build clean-test install develop all test lint format check coverage report

# Variables
PYTHON := python
SRC_DIR := src
TEST_DIR := tests
COV_TARGET := src/ask_llm

help:
	@echo "Available targets:"
	@echo "  install        Install the package in editable mode."
	@echo "  develop        Install the package with development dependencies."
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

clean: clean-pyc clean-build clean-test

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

clean-venv:
	rm -rf venv/
	rm -rf .venv/
	rm -rf env/
	rm -rf ENV/

all: install

# Combined install target
install:
	@echo "Setting up environment in ~/.venv/ask-llm"
	@mkdir -p ~/.venv
	@if [ ! -d ~/.venv/ask-llm ]; then \
		echo "Creating new virtual environment"; \
		uv venv ~/.venv/ask-llm --python `which python3`; \
	else \
		echo "Virtual environment already exists"; \
	fi
	@if [ -f ~/.venv/ask-llm/bin/pip ]; then \
		echo "Installing/updating package in development mode"; \
		~/.venv/ask-llm/bin/pip install -e .; \
	elif [ -f ~/.venv/ask-llm/bin/python ]; then \
		echo "Pip not found, installing via ensurepip"; \
		~/.venv/ask-llm/bin/python -m ensurepip; \
		~/.venv/ask-llm/bin/python -m pip install -e .; \
	else \
		echo "ERROR: Virtual environment seems corrupted. Run 'make clean-venv' first."; \
		exit 1; \
	fi
	@echo "Setup complete. Activate with: source ~/.venv/ask-llm/bin/activate"
	# Alternative short install if env already active
	# uv pip install -e .

develop:
	uv pip install -e ".[dev]" # Assuming a [dev] extra for dev dependencies
	# If no [dev] extra, list dev deps explicitly:
	# uv pip install -e . pytest pytest-cov ruff mypy

# Testing and Coverage
test:
	@echo "Running tests with coverage..."
	pytest --cov=$(COV_TARGET) --cov-report=term-missing $(TEST_DIR)

coverage: test # Alias for running tests with coverage

report:
	@echo "Generating HTML coverage report..."
	pytest --cov=$(COV_TARGET) --cov-report=html $(TEST_DIR)
	@echo "HTML report generated in htmlcov/ directory."

# Linting and Formatting
lint:
	@echo "Running Ruff linter..."
	ruff check $(SRC_DIR) $(TEST_DIR)

format:
	@echo "Running Ruff formatter..."
	ruff format $(SRC_DIR) $(TEST_DIR)

check:
	@echo "Running Ruff check and format..."
	ruff check $(SRC_DIR) $(TEST_DIR)
	ruff format $(SRC_DIR) $(TEST_DIR) --check
	@echo "Running MyPy type checker..."
	mypy $(SRC_DIR) 