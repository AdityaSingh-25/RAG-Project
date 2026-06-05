# RAG Multi-Agent Intelligence Engine

A production-shaped Retrieval Augmented Generation project using LangChain, LangGraph, Qdrant and local/open-source model providers. This scaffold intentionally excludes Claude and OpenAI dependencies.

## What This Builds

- Document ingestion for PDFs, Markdown, text, CSV, JSON and web-ready sources.
- Semantic chunking with metadata preservation.
- Embeddings stored in Qdrant for semantic retrieval.
- LangGraph multi-agent workflow with retrieval, answer synthesis, a self-checking critic, feedback-loop retry and an explicit "insufficient evidence" exit.
- Hybrid retrieval (BM25 + dense) fused with Reciprocal Rank Fusion.
- Two-stage retrieval with a neural cross-encoder reranker as the second stage.
- Source confidence scoring (freshness + glob-keyed trust weights + dense/BM25 agreement) multiplied into the reranker score so trusted sources rise in ranking.
- Content-hash deduplication at ingest, so the same chunk appearing in multiple files isn't indexed or BM25-corpused twice.
- Per-claim citation grounding: every sentence in an answer is verified against the chunks it cites and the run is rejected when too many sentences are unsupported.
- SQLite-backed caching for embeddings, cross-encoder scoring and final answers; JSON-structured logging with per-query trace IDs and a `/metrics` endpoint.
- CI eval gate covering a happy-path dataset and an adversarial dataset, with a fixture-replay mode so real engine behavior can be checked offline.
- Model routing and token budgeting across local models served by Ollama.
- FastAPI service for ingestion, querying, health checks and evaluation hooks.
- Docker Compose for local infrastructure and CI-ready Docker build.
- Optional LangSmith tracing for latency, token and workflow inspection.

## Stack

- Python 3.11+
- LangChain and LangGraph
- Qdrant vector database
- Sentence Transformers embeddings
- Ollama local LLM runtime
- FastAPI
- Next.js and React for the optional local dashboard
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
├── web                  # Next.js dashboard for queries, citations, grounding, and metrics
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
on first ingest and the LLM connection is established lazily on the
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
without clearing the grounding threshold - in that case `answer` is a
structured refusal rather than a fabricated response. The response also
carries `grounded_claim_rate` and a `claim_grounding` array, one entry per
sentence in the answer with its parsed citations and per-claim support
score, so downstream UIs can highlight unsupported claims.

6. Start the optional dashboard:

```bash
cd web
cp .env.local.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard proxies `/api/query` and
`/api/metrics` to FastAPI through `RAG_API_URL`, then shows the answer,
citations, per-claim grounding, pipeline trace and runtime counters.

7. Evaluate the engine against a JSONL of cases:

```bash
rag-eval --cases data/eval/seed_cases.jsonl --format markdown
```

Customize `data/eval/seed_cases.jsonl` for your own corpus before running.
The companion `data/eval/adversarial_cases.jsonl` is meant to *fail* —
each case is a question the engine should refuse with
`status: "insufficient_evidence"`. To run the harness offline against
captured outputs, pass `--fixture data/eval/fixtures/seed.json`. Refresh
fixtures from a live stack with `python scripts/eval_capture.py`.

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
- `ENFORCE_PER_CLAIM_CITATIONS=true` (set to `false` to keep claim grounding for visibility only and skip the gate)
- `CLAIM_SUPPORT_THRESHOLD=0.2` (term-overlap floor for a single claim to count as supported)
- `MIN_GROUNDED_CLAIM_RATE=0.5` (fraction of claims that must be grounded for the answer to ship)
- `RETRIEVAL_MODE=hybrid` (`dense` to disable BM25 fusion)
- `BM25_INDEX_PATH=data/processed/bm25_index.pkl` (rebuilt on every ingest)
- `RRF_K=60` (Reciprocal Rank Fusion constant)
- `RETRIEVE_K=20` (first-stage candidate pool, narrowed to `TOP_K` by the reranker)
- `RERANKER_MODE=cross_encoder` (`keyword` for the term-overlap heuristic, `disabled` to skip reranking entirely)
- `CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB, downloaded on first use)
- `ENABLE_INGEST_DEDUP=true` (set to `false` to keep duplicate chunks; the SHA-256 hash is still written to metadata so you can dedupe later)
- `ENABLE_SOURCE_CONFIDENCE=true` (multiplies reranker scores by freshness × trust × agreement)
- `FRESHNESS_HALF_LIFE_DAYS=365` (how fast a doc's freshness signal decays)
- `AGREEMENT_BOOST=1.2` (multiplier when both dense and BM25 ranked a doc)
- `SOURCE_WEIGHTS={}` (JSON dict mapping glob -> trust multiplier; e.g. `{"docs/**": 1.2}`)
- `EVAL_BASELINE_GROUNDING=0.0` (CI eval gate floor for `seed.mean_grounding`; raise to gate on regressions)
- `EVAL_BASELINE_STATUS_MATCH_RATE=0.0` (CI eval gate floor for `status_match_rate` on both datasets; catches adversarial cases that stop being refused)
- `CACHE_ENABLED=true` (set to `false` to bypass all caches)
- `CACHE_PATH=data/processed/cache.sqlite` (single SQLite file holding embedding, reranker and answer namespaces)
- `CACHE_TTL_SECONDS=86400` (default expiry; embeddings can be much longer in practice, answer cache benefits from shorter)
- `ANSWER_CACHE_ENABLED=true` (orthogonal toggle for the question-keyed answer cache; pass `bypass_cache=true` on a `/query` to fetch fresh)
- `LOG_FORMAT=json` (set to `text` for human-readable lines instead)

LangSmith is optional. Enable it only if you want hosted tracing.
