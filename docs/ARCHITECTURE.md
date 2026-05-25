# Architecture

## Runtime Flow

1. Ingestion loads source files and normalizes them into LangChain `Document` objects.
2. Semantic chunking splits documents into retrieval-sized units and preserves metadata.
3. Embeddings are generated with Sentence Transformers.
4. Qdrant stores vectors and metadata for semantic search.
5. The API receives a user question and hands it to a LangGraph workflow.
6. The planner decides whether retrieval, structured lookup, or direct response is needed.
7. The retriever returns ranked context.
8. The answer agent synthesizes a response using a local Ollama model.
9. The critic checks whether the answer is grounded in retrieved context.
10. When grounding is below `grounding_threshold` and the iteration cap has not been reached,
    a query-rewrite node expands the original question with high-signal terms from the
    retrieved documents and routes back to the retriever. Otherwise the run terminates.
11. Observability hooks record latency, token estimates, routing decisions, and grounding scores.

## Feedback Loop and Fallback

The workflow uses a 3-way conditional edge from `critique`:

- `critique → rewrite → retrieve → answer → critique` while the answer
  fails either the overall grounding check (`grounding_score < grounding_threshold`)
  or the per-claim check (`grounded_claim_rate < min_grounded_claim_rate`),
  and `iteration < max_retry_iterations`.
- `critique → fallback` when either check fails and the retry budget is
  exhausted (or the loop is disabled). The fallback node replaces the
  drafted answer with a structured "insufficient evidence" response,
  clears citations, and sets `status = "insufficient_evidence"`. The
  fallback `reason` field distinguishes `no_retrieved_context`,
  `low_per_claim_grounding`, and `low_grounding_after_retry`.
- `critique → finalize → END` when both checks pass; `finalize`
  marks `status = "ok"`.

## Per-Claim Grounding

The critic does two things, in order:

1. **Overall grounding** — the legacy `verify_answer_confidence` heuristic
   computes a single score for the whole answer (citation density, term
   overlap with all retrieved context, uncertainty hedging).
2. **Per-claim grounding** — `verify_claims` splits the answer into
   sentences, parses each sentence's `[n]` citations, and checks whether
   the cited chunks actually contain the sentence's content terms. Each
   claim gets a `support_score` (term overlap with cited chunks) and an
   `is_grounded` flag (`support_score ≥ claim_support_threshold` AND at
   least one cited index is in range). `grounded_claim_rate` is the
   fraction of claims that are grounded.

The answer prompt now requires a `[n]` citation on every factual sentence
and forbids citing chunk numbers outside the provided context. The
per-claim check catches the failure mode the prompt can't prevent: the
model citing a real chunk that doesn't actually support the sentence.

`ENFORCE_PER_CLAIM_CITATIONS=false` skips the claim-rate gate entirely;
the per-claim report is still computed and surfaced for visibility but
won't cause routing changes.

`rewrite` is a deterministic pseudo-relevance-feedback step: it keeps the user's
original question intact and appends novel high-frequency terms mined from the
top retrieved documents. This widens the lexical surface for the next pass
without an additional LLM call. The loop is bounded by `max_retry_iterations`
(default 1) so a stuck query cannot spin forever.

## Hybrid Retrieval

The retriever defaults to `RETRIEVAL_MODE=hybrid`, which fuses two ranked
lists via Reciprocal Rank Fusion:

- **Dense**: vector search over Qdrant (semantic similarity).
- **Sparse**: BM25 over an in-process index built alongside ingestion and
  persisted at `BM25_INDEX_PATH` (default `data/processed/bm25_index.pkl`).

RRF combines the lists with `score(doc) = Σ 1 / (rrf_k + rank_i)` across the
lists in which the doc appears. `rrf_k = 60` matches the Cormack et al.
default; smaller values weight top ranks more aggressively. If the BM25
index is missing (no ingest has run yet), the retriever falls back to dense
only. Switch back to `RETRIEVAL_MODE=dense` to disable BM25 entirely.

## Two-Stage Retrieval and Reranking

Retrieval is split into a cheap first stage and an accurate second stage:

1. **Candidate pool** — the hybrid retriever returns `RETRIEVE_K` documents
   (default 20). This is the recall stage; we want it wide enough that the
   right answer is somewhere in the pool.
2. **Rerank** — `apply_reranker` narrows the pool to `TOP_K` (default 6)
   using whichever reranker is configured by `RERANKER_MODE`:
   - `cross_encoder` (default): neural cross-encoder
     (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default), loaded lazily and
     cached for the life of the process. Reads each `(question, passage)`
     pair jointly, which is strictly more accurate than dot-product of
     independent embeddings but ~3-5x slower — bearable because it runs
     only over the small candidate pool, not the corpus.
   - `keyword`: term-overlap heuristic. No model download.
   - `disabled`: passthrough; truncates to `TOP_K`.

Each reranked document carries `metadata["rerank_score"]`,
`metadata["source_confidence"]`, and `metadata["final_score"]` so
downstream code (and the eval harness) can inspect ranking decisions.

## Source Confidence

When `ENABLE_SOURCE_CONFIDENCE=true` (default), the reranker multiplies
its raw score by a per-document confidence factor built from three
signals:

- **freshness** — half-life decay from `metadata['published_at']` (ISO
  date) or `metadata['mtime']` (POSIX timestamp). A doc one
  `FRESHNESS_HALF_LIFE_DAYS` old scores 0.5; twice as old, 0.25. Missing
  dates default to 1.0 so undated curated docs aren't penalised.
- **trust** — first-match glob against `SOURCE_WEIGHTS` (a JSON dict like
  `{"docs/**": 1.2, "data/raw/notes/**": 0.7}`). Default is 1.0.
- **agreement** — `AGREEMENT_BOOST` (default 1.2) when both the dense
  and BM25 retrievers ranked the doc; 1.0 otherwise. The HybridRetriever
  sets `metadata['agreement_count']` during RRF.

The three signals multiply (`confidence = freshness * trust * agreement`)
so a doc strong on all three compounds. The result becomes
`final_score = rerank_score * confidence`, and `apply_reranker` sorts and
truncates on `final_score`. This lets a confident-but-mid-ranked doc
rescue itself into `TOP_K`, which the previous "score then truncate" flow
couldn't do.

## Ingest Deduplication

`ENABLE_INGEST_DEDUP=true` (default) hashes each chunk's content
(whitespace-and-case-insensitive SHA-256) during `ingest_path` and skips
later copies. `IngestReport.duplicates_removed` is surfaced via the
`/ingest` response and the `rag-ingest` CLI output. The hash is also
written to `metadata['content_hash']` so downstream tooling can detect
duplicates across ingest runs.

## Evaluation Harness

`rag-eval` (or `python -m rag_engine.cli eval_command`) runs a JSONL of cases
through the graph and reports per-case and aggregate metrics:

- **grounding** — from the critic's `verify_answer_confidence`.
- **citation hit rate** — fraction of `must_cite` substrings that appear in
  cited sources.
- **term recall** — fraction of `expected_terms` that appear as whole words
  in the answer.
- **iteration** — how many feedback-loop retries the case triggered.
- **status** — `ok` or `insufficient_evidence` (also surfaced as the
  aggregate `insufficient_evidence_rate`).
- **latency** — wall-clock per case.

The runner is injectable, so the harness can be exercised in unit tests
without Qdrant or Ollama.

## Evaluation in CI

`scripts/eval_ci.py` runs two datasets in CI:

- `data/eval/seed_cases.jsonl` — happy-path questions the engine should
  answer.
- `data/eval/adversarial_cases.jsonl` — questions the engine should
  *refuse* (out-of-corpus facts, false-premise leading questions,
  under-specified queries). Each adversarial case carries
  `expected_status: insufficient_evidence`.

Each dataset is driven by one of two runners:

- **Fixture runner** (default when a fixture exists): replays previously
  captured live outputs from `data/eval/fixtures/{seed,adversarial}.json`.
  Captured with `scripts/eval_capture.py` against a working Qdrant +
  Ollama stack; refreshed in PRs whenever model behavior should change.
  CI gates on real engine behavior without loading models.
- **Stub runner** (fallback when no fixture is present): synthesises an
  output that satisfies the case's `expected_status`. Useful for
  bootstrapping a brand-new corpus.

The gate fails when any of three baselines aren't met:

- `EVAL_BASELINE_GROUNDING` — `seed.mean_grounding` floor.
- `EVAL_BASELINE_STATUS_MATCH_RATE` — applied to both
  `seed.status_match_rate` (catches refused-by-mistake regressions) and
  `adversarial.status_match_rate` (catches answered-by-mistake
  regressions on questions that should be refused).

Refreshing fixtures::

    python scripts/eval_capture.py \\
        --cases data/eval/seed_cases.jsonl \\
        --fixture data/eval/fixtures/seed.json

`rag-eval` also accepts `--fixture <path>` so the same replay path is
available for ad-hoc local runs without infrastructure.

## Caching

A single SQLite file at `CACHE_PATH` backs three namespaces sharing one
TTL:

- **embeddings** — keyed on `(model, text)`. Sentence Transformers calls
  are deterministic, so cache hits skip inference outright. Both `/query`
  and `/ingest` paths benefit.
- **cross_encoder** — keyed on `(model, question, passage)`. The slowest
  hot path after the Phase 5 reranker landed; per-pair caching pays off
  whenever a follow-up query reuses passages from the candidate pool.
- **answers** — keyed on the normalised question text. Bypassed with
  `bypass_cache=true` on the `/query` payload. Only successful answers
  (`status="ok"`) are cached; `insufficient_evidence` exits are not.

Cache hits and misses increment counters of the form
`cache.<namespace>.{hit,miss}`, visible via `/metrics`. Set
`CACHE_ENABLED=false` for full pass-through (useful when measuring the
unprotected hot path).

## Observability

Every `/query` mints a `trace_id` (UUID4 hex). Each graph node emits a
single JSON log line tagged with that trace ID plus stage-specific
fields:

- `graph.retrieve` — `n_candidates`, `n_reranked`, `retrieval_mode`,
  `reranker_mode`, `duration_ms`.
- `graph.answer` — `model`, `answer_chars`, `duration_ms`.
- `graph.critique` — `grounding_score`, `warnings`.
- `graph.rewrite` — `iteration`, `rewritten_chars`.
- `graph.fallback` — `reason`, `iteration`.
- `graph.finalize` — terminal grounding_score and iteration.

The `/metrics` endpoint returns a JSON snapshot of the same counters that
drive the logs:

- **totals** — `api.query.total`, `api.query.status.{ok,insufficient_evidence}`,
  `cache.<namespace>.{hit,miss}`, `graph.rewrite.invocations`,
  `graph.fallback.invocations`, `graph.fallback.reason.<reason>`.
- **samples** — latency observations summarised with count, mean, p50,
  p95, and max (`api.query.latency_ms`, `graph.retrieve.latency_ms`,
  `graph.answer.latency_ms`, `graph.iteration`).

`LOG_FORMAT=text` switches to a human-readable formatter for local
debugging; the JSON output is the production default.

## Agent Responsibilities

- Planner agent: classifies the query and selects tools.
- Retrieval agent: searches Qdrant and prepares citations.
- Answer agent: drafts grounded responses from context.
- Critic agent: flags weak grounding, missing context, and unsupported claims.

## Provider Policy

This project does not use Claude or OpenAI SDKs. Model calls are routed through Ollama by default. Embeddings are generated locally through Sentence Transformers.

## Deployment Shape

For local development, Docker Compose runs the API, Qdrant, and Ollama. For hosted deployment, keep Qdrant and model serving outside Vercel unless your workload is very small. The Vercel config is included for lightweight API deployments and demos.

