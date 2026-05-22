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

## Feedback Loop

The workflow uses a conditional edge from `critique`:

- `critique → rewrite → retrieve → answer → critique` while
  `grounding_score < grounding_threshold` and `iteration < max_retry_iterations`.
- `critique → END` otherwise.

`rewrite` is a deterministic pseudo-relevance-feedback step: it keeps the user's
original question intact and appends novel high-frequency terms mined from the
top retrieved documents. This widens the lexical surface for the next pass
without an additional LLM call. The loop is bounded by `max_retry_iterations`
(default 1) so a stuck query cannot spin forever.

## Evaluation Harness

`rag-eval` (or `python -m rag_engine.cli eval_command`) runs a JSONL of cases
through the graph and reports per-case and aggregate metrics:

- **grounding** — from the critic's `verify_answer_confidence`.
- **citation hit rate** — fraction of `must_cite` substrings that appear in
  cited sources.
- **term recall** — fraction of `expected_terms` that appear as whole words
  in the answer.
- **iteration** — how many feedback-loop retries the case triggered.
- **latency** — wall-clock per case.

The runner is injectable, so the harness can be exercised in unit tests
without Qdrant or Ollama.

## Agent Responsibilities

- Planner agent: classifies the query and selects tools.
- Retrieval agent: searches Qdrant and prepares citations.
- Answer agent: drafts grounded responses from context.
- Critic agent: flags weak grounding, missing context, and unsupported claims.

## Provider Policy

This project does not use Claude or OpenAI SDKs. Model calls are routed through Ollama by default. Embeddings are generated locally through Sentence Transformers.

## Deployment Shape

For local development, Docker Compose runs the API, Qdrant, and Ollama. For hosted deployment, keep Qdrant and model serving outside Vercel unless your workload is very small. The Vercel config is included for lightweight API deployments and demos.

