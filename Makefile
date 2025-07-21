# Makefile for MMS Magnetopause Analysis Toolkit
# Common development tasks

.PHONY: help install install-dev test test-basic lint format clean build docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in current environment"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run tests with pytest"
	@echo "  test-basic   - Run basic tests without pytest"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=mms_mp --cov-report=html --cov-report=term

test-basic:
	python test_basic.py

# Code quality
lint:
	flake8 mms_mp/ --max-line-length=88 --extend-ignore=E203,W503
	pydocstyle mms_mp/ --convention=google

format:
	black mms_mp/ tests/ --line-length=88
	isort mms_mp/ tests/ --profile=black --line-length=88

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build
build: clean
	python -m build

# Documentation
docs:
	@echo "Documentation is in the docs/ directory"
	@echo "For API docs, consider using sphinx-apidoc or similar tools"

# Development setup
setup-dev: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test-basic' to verify installation"
