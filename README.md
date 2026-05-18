# RAG Multi-Agent Intelligence Engine

A production-shaped Retrieval Augmented Generation project using LangChain, LangGraph, Qdrant, and local/open-source model providers. This scaffold intentionally excludes Claude and OpenAI dependencies.

## What This Builds

- Document ingestion for PDFs, Markdown, text, CSV, JSON, and web-ready sources.
- Semantic chunking with metadata preservation.
- Embeddings stored in Qdrant for semantic retrieval.
- LangGraph multi-agent workflow for query planning, retrieval, answer synthesis, and self-checking.
- Model routing and token budgeting across local models served by Ollama.
- FastAPI service for ingestion, querying, health checks, and evaluation hooks.
- Docker Compose for local infrastructure and CI-ready Docker build.
- Optional LangSmith tracing for latency, token, and workflow inspection.

## Stack

- Python 3.11+
- LangChain and LangGraph
- Qdrant vector database
- Sentence Transformers embeddings
- Ollama local LLM runtime
- FastAPI
- Docker and Docker Compose
- Pytest

## Project Layout

```text
.
├── src/rag_engine
│   ├── api              # FastAPI entrypoints
│   ├── agents           # LangGraph state and workflow
│   ├── chunking         # Semantic chunking
│   ├── config           # Runtime settings
│   ├── embeddings       # Embedding provider factory
│   ├── evaluation       # Grounding and hallucination checks
│   ├── ingestion        # Loaders and ingestion pipeline
│   ├── observability    # Tracing and metrics helpers
│   ├── retrieval        # Retriever assembly
│   ├── routing          # Model routing and token budgeting
│   ├── tools            # Structured data tools
│   └── vectorstore      # Qdrant integration
├── data
│   ├── raw              # Source documents
│   └── processed        # Generated artifacts
├── docs                 # Architecture notes
├── scripts              # Operational scripts
└── tests
```

## Quick Start

1. Create an environment file:

```bash
cp .env.example .env
```

2. Start local services:

```bash
docker compose up -d qdrant ollama
```

3. Pull local models:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
```

4. Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

5. Ingest documents:

```bash
rag-ingest --source data/raw
```

6. Run the API:

```bash
uvicorn rag_engine.api.main:app --reload
```

7. Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main themes in the ingested documents?"}'
```

## Deployment Notes

This repo includes `vercel.json` for a lightweight FastAPI deployment shape. For heavier RAG workloads, deploy the API as a container service and keep Qdrant as a managed or long-running service. Vercel is best used for thin API routes or a frontend gateway.

## Environment

The default setup uses local providers:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `CHAT_MODEL_FAST=mistral:7b`
- `CHAT_MODEL_REASONING=llama3.1:8b`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `QDRANT_URL=http://localhost:6333`

LangSmith is optional. Enable it only if you want hosted tracing.

