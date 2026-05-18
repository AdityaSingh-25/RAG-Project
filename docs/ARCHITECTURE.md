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
10. Observability hooks record latency, token estimates, routing decisions, and grounding scores.

## Agent Responsibilities

- Planner agent: classifies the query and selects tools.
- Retrieval agent: searches Qdrant and prepares citations.
- Answer agent: drafts grounded responses from context.
- Critic agent: flags weak grounding, missing context, and unsupported claims.

## Provider Policy

This project does not use Claude or OpenAI SDKs. Model calls are routed through Ollama by default. Embeddings are generated locally through Sentence Transformers.

## Deployment Shape

For local development, Docker Compose runs the API, Qdrant, and Ollama. For hosted deployment, keep Qdrant and model serving outside Vercel unless your workload is very small. The Vercel config is included for lightweight API deployments and demos.

