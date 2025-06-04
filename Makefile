# Python Project Makefile
# Development automation commands

.PHONY: help install dev-install run lint format test clean logs monitor setup check all

# Default target - show help
help:
	@echo "Available commands:"
	@echo "  make install      - Install project dependencies"
	@echo "  make dev-install  - Install development dependencies"
	@echo "  make run          - Run the agent"
	@echo "  make lint         - Run code linting (ruff check)"
	@echo "  make format       - Format code (ruff format)"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean generated files and caches"
	@echo "  make logs         - Show recent logs"
	@echo "  make monitor      - Monitor agent performance"
	@echo "  make setup        - Initial project setup"
	@echo "  make check        - Run all quality checks (lint + test)"
	@echo "  make all          - Run format, lint, and test"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev-install:
	pip install -r requirements-dev.txt

# Run the agent
run:
	PYTHONPATH=. python3 src/run_agent.py

# Run with specific number of iterations
run-iterations:
	@read -p "Enter number of iterations: " n; \
	PYTHONPATH=. python3 src/run_agent.py --iterations $$n

# Lint code using ruff
lint:
	ruff check src/
	ruff check --fix src/

# Format code using ruff
format:
	ruff format src/

# Run tests
test:
	python3 -m pytest tests/ -v

# Run tests with coverage
test-coverage:
	python3 -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# View recent logs
logs:
	@echo "=== Recent Agent Logs ==="
	tail -n 50 src/logs/agent.log

# View error logs
logs-errors:
	@echo "=== Recent Error Logs ==="
	tail -n 50 src/logs/errors/errors_$$(date +%Y%m%d).log

# View learning logs
logs-learning:
	@echo "=== Recent Learning Logs ==="
	tail -n 20 src/logs/learning/learning_$$(date +%Y%m%d).jsonl | jq '.'

# Monitor performance
monitor:
	@echo "=== Performance Summary ==="
	@if [ -f src/logs/performance/summary.json ]; then \
		cat src/logs/performance/summary.json | jq '.'; \
	else \
		echo "No performance summary available"; \
	fi

# Watch logs in real-time
watch-logs:
	tail -f src/logs/agent.log

# Initial project setup
setup:
	pip install --upgrade pip
	pip install ruff pytest pytest-cov
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Activate virtual environment with: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)"

# Run all quality checks
check: format lint test

# Run everything
all: format lint test

# Type checking with mypy (if available)
typecheck:
	@command -v mypy >/dev/null 2>&1 && mypy src/ || echo "mypy not installed, skipping type check"

# Security check with bandit (if available)  
security:
	@command -v bandit >/dev/null 2>&1 && bandit -r src/ || echo "bandit not installed, skipping security check"

# Generate requirements.txt from pyproject.toml (if using poetry/pip-tools)
requirements:
	@if [ -f pyproject.toml ]; then \
		echo "Generating requirements.txt from pyproject.toml..."; \
		pip-compile pyproject.toml -o requirements.txt; \
	else \
		echo "No pyproject.toml found"; \
	fi

# Docker commands (if using Docker)
docker-build:
	docker build -t python-agent .

docker-run:
	docker run -it --rm -v $$(pwd)/src/logs:/app/src/logs python-agent

# Database commands (if using a database)
db-init:
	python3 -m src.infrastructure.storage --init

db-migrate:
	python3 -m src.infrastructure.storage --migrate

# Development server with auto-reload
dev:
	watchmedo auto-restart --directory=./src --pattern="*.py" --recursive -- PYTHONPATH=. python3 src/run_agent.py

# Generate documentation
docs:
	@command -v sphinx-build >/dev/null 2>&1 && sphinx-build -b html docs/ docs/_build/ || echo "Sphinx not installed"

# Count lines of code
loc:
	@echo "Lines of code by file type:"
	@find src -name "*.py" | xargs wc -l | tail -1

# Show project structure
tree:
	@command -v tree >/dev/null 2>&1 && tree -I '__pycache__|*.pyc|.git|venv' src/ || find src -type f -name "*.py" | sort
