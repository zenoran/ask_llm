.PHONY: clean clean-pyc clean-build clean-test install develop all test lint format check coverage report

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
	@if [ -f .venv/bin/pip3 ]; then \
		echo "Attempting to uninstall existing llama-cpp-python..."; \
		.venv/bin/pip3 uninstall llama-cpp-python -y || true; \
		echo "Installing llama-cpp-python with CUDA support..."; \
		CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 .venv/bin/pip3 install llama-cpp-python --no-cache-dir; \
		echo "Installing huggingface-hub..."; \
		.venv/bin/pip3 install huggingface-hub; \
	elif [ -f .venv/bin/python ]; then \
		echo "Attempting to uninstall existing llama-cpp-python..."; \
		# .venv/bin/python -m pip3 uninstall llama-cpp-python -y || true; \
		echo "Installing llama-cpp-python with CUDA support..."; \
		CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 .venv/bin/python -m pip3 install llama-cpp-python --no-cache-dir; \
		echo "Installing huggingface-hub..."; \
		.venv/bin/python -m pip3 install huggingface-hub; \
	else \
		echo "ERROR: Virtual environment seems corrupted. Run 'make clean-venv' first."; \
		exit 1; \
	fi

install:
	@echo "Setting up environment in .venv"
	@if [ ! -d .venv ]; then \
		echo "Creating new virtual environment"; \
		uv venv .venv --python `which python3`; \
	else \
		echo "Virtual environment already exists"; \
	fi
	@if [ -f .venv/bin/pip3 ]; then \
		echo "Installing/updating package in development mode"; \
		.venv/bin/pip3 install -e .; \
	elif [ -f .venv/bin/python ]; then \
		echo "Pip3 not found, installing via ensurepip3"; \
		.venv/bin/python -m ensurepip3; \
		.venv/bin/python -m pip3 install -e .; \
	else \
		echo "ERROR: Virtual environment seems corrupted. Run 'make clean-venv' first."; \
		exit 1; \
	fi
	@echo "Setup complete. Activate with: source .venv/bin/activate"
	# Alternative short install if env already active
	# uv pip3 install -e .
	@make install-deps

develop:
	uv pip3 install -e ".[dev]" # Assuming a [dev] extra for dev dependencies

test:
	@echo "Running tests with coverage..."
	pytest --cov=$(COV_TARGET) --cov-report=term-missing $(TEST_DIR)

coverage: test # Alias for running tests with coverage

report:
	@echo "Generating HTML coverage report..."
	pytest --cov=$(COV_TARGET) --cov-report=html $(TEST_DIR)
	@echo "HTML report generated in htmlcov/ directory."

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