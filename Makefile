.PHONY: install lint test api ingest docker-up docker-down

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest

api:
	uvicorn rag_engine.api.main:app --reload

ingest:
	rag-ingest --source data/raw

docker-up:
	docker compose up -d

docker-down:
	docker compose down

