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

- `critique → rewrite → retrieve → answer → critique` while
  `grounding_score < grounding_threshold` and `iteration < max_retry_iterations`.
- `critique → fallback` when grounding is below threshold and the retry budget
  is exhausted (or the loop is disabled). The fallback node replaces the
  drafted answer with a structured "insufficient evidence" response,
  clears citations, and sets `status = "insufficient_evidence"`. This is the
  hallucination-prevention exit — we refuse to ship an answer the critic
  doesn't trust.
- `critique → finalize → END` when grounding clears the threshold; `finalize`
  marks `status = "ok"`.

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

Each reranked document carries the raw score in `metadata["rerank_score"]`
so downstream code (and the eval harness) can inspect ranking decisions.

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

`scripts/eval_ci.py` runs the harness over `data/eval/seed_cases.jsonl` with
a deterministic stub runner (CI doesn't have Qdrant/Ollama) and exits
non-zero when `mean_grounding < EVAL_BASELINE_GROUNDING`. Today the gate
catches harness regressions and malformed seed cases. When CI gains access
to real models — or to a recorded fixture corpus — the same script can be
pointed at a real runner to gate on actual answer quality.

## Agent Responsibilities

- Planner agent: classifies the query and selects tools.
- Retrieval agent: searches Qdrant and prepares citations.
- Answer agent: drafts grounded responses from context.
- Critic agent: flags weak grounding, missing context, and unsupported claims.

## Provider Policy

This project does not use Claude or OpenAI SDKs. Model calls are routed through Ollama by default. Embeddings are generated locally through Sentence Transformers.

## Deployment Shape

For local development, Docker Compose runs the API, Qdrant, and Ollama. For hosted deployment, keep Qdrant and model serving outside Vercel unless your workload is very small. The Vercel config is included for lightweight API deployments and demos.

