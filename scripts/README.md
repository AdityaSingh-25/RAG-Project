# Scripts

Operational scripts live here. The main ingestion entrypoint is exposed as:

```bash
rag-ingest --source data/raw
```

## Load testing

```bash
# 32 concurrent clients, 5 queries each, against a running API.
python scripts/load_test.py --concurrency 32 --requests 5

# Stream endpoint instead — measures end-to-end SSE wall time.
python scripts/load_test.py --mode stream --concurrency 16 --requests 3

# Hammer with cache misses (e.g. before/after a tuning change).
python scripts/load_test.py --concurrency 16 --requests 10 --bypass-cache
```

Output reports p50/p95/p99 latency, throughput, and success rate.
Exit code is non-zero if any request failed — handy for CI.

The script does not start the API. Bring it up separately with
`uvicorn rag_engine.api.main:app` or `rag-api`. For higher useful
concurrency, also raise Ollama's parallelism:

```bash
OLLAMA_NUM_PARALLEL=8 OLLAMA_MAX_QUEUE=1024 ollama serve
```

