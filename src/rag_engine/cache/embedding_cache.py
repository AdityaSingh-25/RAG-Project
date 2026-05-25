"""Cache wrapper for Sentence Transformers embeddings.

Embedding inference is deterministic — same (model, text) always produces
the same vector — so caching is free correctness-wise and a big latency win
for repeated queries (and repeated chunks during ingest).
"""

from __future__ import annotations

import hashlib
from typing import Any

from rag_engine.cache.store import CacheStore
from rag_engine.observability.counters import counters
from rag_engine.observability.logging import get_logger

_NAMESPACE = "embeddings"
_logger = get_logger("rag_engine.cache.embeddings")


def _key(model: str, text: str) -> str:
    digest = hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()
    return digest


class CachingEmbeddings:
    """Drop-in proxy around any LangChain embeddings implementation.

    Implements both ``embed_query`` and ``embed_documents`` so it can stand
    in wherever a HuggingFaceEmbeddings instance was used. Misses fall
    through to the wrapped object; hits skip inference entirely.
    """

    def __init__(self, inner: Any, store: CacheStore, model_name: str) -> None:
        self._inner = inner
        self._store = store
        self._model = model_name

    def embed_query(self, text: str) -> list[float]:
        key = _key(self._model, text)
        cached = self._store.get(_NAMESPACE, key)
        if cached is not None:
            counters().increment("cache.embeddings.hit")
            return list(cached)
        counters().increment("cache.embeddings.miss")
        vector = list(self._inner.embed_query(text))
        self._store.set(_NAMESPACE, key, vector)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for idx, text in enumerate(texts):
            cached = self._store.get(_NAMESPACE, _key(self._model, text))
            if cached is None:
                misses.append((idx, text))
            else:
                counters().increment("cache.embeddings.hit")
                results[idx] = list(cached)

        if misses:
            counters().increment("cache.embeddings.miss", len(misses))
            fresh = self._inner.embed_documents([text for _, text in misses])
            for (idx, text), vector in zip(misses, fresh):
                vector_list = list(vector)
                results[idx] = vector_list
                self._store.set(_NAMESPACE, _key(self._model, text), vector_list)

        return [v if v is not None else [] for v in results]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
