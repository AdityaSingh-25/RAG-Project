# RAG Multi-Agent Intelligence Engine

A production-shaped Retrieval Augmented Generation project using LangChain, LangGraph, Qdrant, and local/open-source model providers. This scaffold intentionally excludes Claude and OpenAI dependencies.

## What This Builds

- Document ingestion for PDFs, Markdown, text, CSV, JSON, and web-ready sources.
- Semantic chunking with metadata preservation.
- Embeddings stored in Qdrant for semantic retrieval.
- LangGraph multi-agent workflow with retrieval, answer synthesis, a self-checking critic, feedback-loop retry, and an explicit "insufficient evidence" exit.
- Hybrid retrieval (BM25 + dense) fused with Reciprocal Rank Fusion.
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

### Option A: Docker Compose (recommended)

1. Create an environment file:

```bash
cp .env.example .env
```

2. Start the full stack (API, Qdrant, Ollama):

```bash
docker compose up -d
```

The API container starts immediately; the Qdrant collection is created
on first ingest, and the LLM connection is established lazily on the
first `/query` call.

3. Pull local models into the Ollama container:

```bash
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull mistral:7b
```

4. Drop documents in `data/raw/` (the directory is bind-mounted into the
   API container), then trigger ingestion:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path":"data/raw"}'
```

5. Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main themes in the ingested documents?"}'
```

The response includes an `iteration` field showing how many feedback-loop
retries the critic triggered (0 when the first answer met the grounding
threshold) and a `status` field that is `"ok"` when the critic was
satisfied or `"insufficient_evidence"` when the retry budget was exhausted
without clearing the grounding threshold — in that case `answer` is a
structured refusal rather than a fabricated response.

6. Evaluate the engine against a JSONL of cases:

```bash
rag-eval --cases data/eval/seed_cases.jsonl --format markdown
```

Customize `data/eval/seed_cases.jsonl` for your own corpus before running.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the case schema and
the feedback-loop semantics.

### Option B: Local Python (no API container)

1. Start infrastructure only:

```bash
docker compose up -d qdrant ollama
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull mistral:7b
```

2. Install the package and run the API on the host:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
rag-ingest --source data/raw
uvicorn rag_engine.api.main:app --reload
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
- `ENABLE_FEEDBACK_LOOP=true` (set to `false` to keep the legacy linear pipeline)
- `GROUNDING_THRESHOLD=0.6` (minimum score to skip the rewrite-and-retry loop)
- `MAX_RETRY_ITERATIONS=1` (hard cap on feedback-loop iterations per query)
- `RETRIEVAL_MODE=hybrid` (`dense` to disable BM25 fusion)
- `BM25_INDEX_PATH=data/processed/bm25_index.pkl` (rebuilt on every ingest)
- `RRF_K=60` (Reciprocal Rank Fusion constant)
- `EVAL_BASELINE_GROUNDING=0.0` (CI eval gate floor; raise to gate on regressions)

LangSmith is optional. Enable it only if you want hosted tracing.

