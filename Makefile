.PHONY: install lint format test

install:
	poetry install

lint:
	poetry run ruff check src tests

format:
	poetry run black src tests

test:
	poetry run pytest -q
