from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "rag-multi-agent-intelligence-engine"
    app_env: str = "local"
    log_level: str = "INFO"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    ollama_base_url: str = "http://localhost:11434"
    chat_model_fast: str = "mistral:7b"
    chat_model_reasoning: str = "llama3.1:8b"
    chat_model_structured: str = "llama3.1:8b"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    token_encoding: str = "cl100k_base"
    chunk_size: int = Field(default=900, ge=200)
    chunk_overlap: int = Field(default=120, ge=0)
    # "recursive" uses character/separator boundaries (cheap, default).
    # "semantic" embeds sentences and breaks where similarity drops — slower,
    # but produces topic-coherent chunks. Costs ~one embedding call per
    # sentence at ingest time.
    chunking_mode: Literal["recursive", "semantic"] = "recursive"
    semantic_breakpoint_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile"
    top_k: int = Field(default=6, ge=1, le=25)
    retrieve_k: int = Field(default=20, ge=1, le=100)
    max_context_tokens: int = Field(default=6000, ge=1000)

    reranker_mode: Literal["keyword", "cross_encoder", "disabled"] = "cross_encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    cache_enabled: bool = True
    cache_path: str = "data/processed/cache.sqlite"
    cache_ttl_seconds: int = Field(default=86_400, ge=0)
    answer_cache_enabled: bool = True

    log_format: Literal["json", "text"] = "json"

    enable_feedback_loop: bool = True
    grounding_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_retry_iterations: int = Field(default=1, ge=0, le=3)

    enforce_per_claim_citations: bool = True
    claim_support_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    min_grounded_claim_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    # When True, the answer node uses ChatOllama.with_structured_output to
    # force a {claims: [{text, citations[]}]} JSON shape via the model's
    # function-call API. Makes per-claim citation parsing exact instead of
    # regex-based. Trade-off: bypasses token streaming.
    structured_answers: bool = False
    # "overlap" is the term-overlap heuristic (no model download).
    # "nli" runs a cross-encoder NLI model over (chunk, claim) pairs and
    # interprets claim_support_threshold as a P(entailment) floor.
    claim_verifier_mode: Literal["overlap", "nli"] = "overlap"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    bm25_index_path: str = "data/processed/bm25_index.pkl"
    rrf_k: int = Field(default=60, ge=1)

    enable_ingest_dedup: bool = True

    enable_source_confidence: bool = True
    freshness_half_life_days: int = Field(default=365, ge=1)
    agreement_boost: float = Field(default=1.2, ge=1.0, le=3.0)
    source_weights: dict[str, float] = Field(default_factory=dict)

    eval_baseline_grounding: float = Field(default=0.0, ge=0.0, le=1.0)
    eval_baseline_status_match_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Backpressure: fail-fast when too many requests are already in flight.
    # /query gets a higher ceiling than /ingest because ingest is IO-heavy
    # and competes for disk + Qdrant write throughput.
    max_concurrent_queries: int = Field(default=4, ge=1, le=64)
    max_concurrent_ingest: int = Field(default=1, ge=1, le=16)
    backpressure_retry_after_seconds: int = Field(default=5, ge=1, le=300)

    langchain_tracing_v2: bool = False
    langchain_project: str = "rag-multi-agent-engine"
    langchain_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

