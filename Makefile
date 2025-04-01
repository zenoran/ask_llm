.PHONY: clean clean-pyc clean-build clean-test install

help:
	@echo "clean - remove all build, test, and Python artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-test - remove test and coverage artifacts"

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