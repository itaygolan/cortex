# Makefile for Cortex autograd framework

# Variables
PYTHON := python3
PIP := pip3
SRC_DIR := cortex
TEST_DIR := tests

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  format       - Run isort and black to format code"
	@echo "  format-check - Check if code formatting is correct"
	@echo "  isort        - Sort imports with isort"
	@echo "  black        - Format code with black"
	@echo "  isort-check  - Check import sorting without making changes"
	@echo "  black-check  - Check code formatting without making changes"
	@echo "  lint         - Run all linting checks"
	@echo "  test         - Run tests"
	@echo "  install      - Install package in development mode"
	@echo "  install-deps - Install development dependencies"
	@echo "  clean        - Clean up build artifacts"

# Formatting targets
.PHONY: format
format: isort black
	@echo "Code formatting complete!"

.PHONY: isort
isort:
	@echo "Sorting imports with isort..."
	isort $(SRC_DIR) $(TEST_DIR) --profile black

.PHONY: black  
black:
	@echo "Formatting code with black..."
	black $(SRC_DIR) $(TEST_DIR)

# Check formatting without making changes
.PHONY: format-check
format-check: isort-check black-check
	@echo "Format checking complete!"

.PHONY: isort-check
isort-check:
	@echo "Checking import sorting..."
	isort $(SRC_DIR) $(TEST_DIR) --check-only --profile black

.PHONY: black-check
black-check:
	@echo "Checking code formatting..."
	black $(SRC_DIR) $(TEST_DIR) --check

# Linting
.PHONY: lint
lint: format-check
	@echo "Running linting checks..."
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8..."; \
		flake8 $(SRC_DIR) $(TEST_DIR); \
	else \
		echo "flake8 not found, skipping..."; \
	fi

# Testing
.PHONY: test
test:
	@echo "Running tests..."
	@if [ -d "$(TEST_DIR)" ]; then \
		$(PYTHON) -m pytest $(TEST_DIR) -v; \
	else \
		echo "No tests directory found"; \
	fi

# Installation
.PHONY: install
install:
	@echo "Installing package in development mode..."
	$(PIP) install -e .

.PHONY: install-deps
install-deps:
	@echo "Installing development dependencies..."
	$(PIP) install black isort flake8 pytest

# Cleanup
.PHONY: clean
clean:
	@echo "Cleaning up build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -delete