from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from rag_engine.cache.embedding_cache import CachingEmbeddings
from rag_engine.cache.store import CacheStore
from rag_engine.config.settings import Settings


@lru_cache(maxsize=4)
def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def build_cached_embeddings(settings: Settings) -> Any:
    """Return embeddings wrapped in the SQLite cache when ``cache_enabled``."""
    inner = build_embeddings(settings.embedding_model)
    if not settings.cache_enabled:
        return inner
    store = CacheStore(Path(settings.cache_path), settings.cache_ttl_seconds)
    return CachingEmbeddings(inner=inner, store=store, model_name=settings.embedding_model)
