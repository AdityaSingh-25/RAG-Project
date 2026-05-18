from functools import lru_cache

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
    top_k: int = Field(default=6, ge=1, le=25)
    max_context_tokens: int = Field(default=6000, ge=1000)

    langchain_tracing_v2: bool = False
    langchain_project: str = "rag-multi-agent-engine"
    langchain_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

