.PHONY: help install install-dev test test-gpu lint format clean setup check docs run

# Default target
help:
	@echo "Sovereign AI Agent - Development Commands"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install package in development mode with dev dependencies"
	@echo "  setup        Run initial project setup"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests"
	@echo "  test-gpu     Run GPU-specific tests"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black and isort"
	@echo "  check        Run all checks (lint, type check, etc.)"
	@echo ""
	@echo "Operations:"
	@echo "  run          Run Sovereign AI Agent"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  docs         Generate documentation"
	@echo ""
	@echo "System:"
	@echo "  requirements Check system requirements"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Setup
setup:
	python -m sovereign --setup
	@echo "✅ Setup complete! Run 'make requirements' to check system compatibility."

requirements:
	python -m sovereign --check-requirements

# Testing
test:
	pytest tests/ -v

test-gpu:
	pytest tests/ -v -m gpu

test-unit:
	pytest tests/ -v -m unit

test-integration:
	pytest tests/ -v -m integration

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

check: lint test
	@echo "✅ All checks passed!"

# Operations
run:
	python -m sovereign

run-debug:
	python -m sovereign --debug

run-cpu:
	python -m sovereign --no-gpu

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "- README.md: Project overview"
	@echo "- INSTALL.md: Installation guide"
	@echo "- API docs: Coming soon..."

# Docker (future)
docker-build:
	@echo "🐳 Docker support coming in future release"

docker-run:
	@echo "🐳 Docker support coming in future release"

# Release
build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*

# Development helpers
dev-env:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "✅ Virtual environment created. Activate with:"
	@echo "   source venv/bin/activate  # Linux/macOS"
	@echo "   venv\\Scripts\\activate     # Windows"

install-gpu-torch:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

install-cpu-torch:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# System checks
check-gpu:
	python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

check-python:
	python --version
	pip --version

check-deps:
	pip check 