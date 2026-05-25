"""Cache wrapper for cross-encoder scoring.

The cross-encoder ``predict`` call is the slowest hot path in the engine
post-Phase 5. Keying on ``(model, question, passage)`` is safe — the model
is deterministic and the function is pure.
"""

from __future__ import annotations

import hashlib

from rag_engine.cache.store import CacheStore
from rag_engine.observability.counters import counters
from rag_engine.observability.logging import get_logger
from rag_engine.retrieval.cross_encoder_reranker import CrossEncoderLike

_NAMESPACE = "cross_encoder"
_logger = get_logger("rag_engine.cache.cross_encoder")


def _key(model: str, question: str, passage: str) -> str:
    return hashlib.sha256(
        f"{model}\0{question}\0{passage}".encode("utf-8")
    ).hexdigest()


class CachingCrossEncoder:
    """Proxy that caches per-pair scores so a repeated pair never re-scores."""

    def __init__(self, inner: CrossEncoderLike, store: CacheStore, model_name: str) -> None:
        self._inner = inner
        self._store = store
        self._model = model_name

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float | None] = [None] * len(pairs)
        misses: list[tuple[int, tuple[str, str]]] = []
        for idx, (question, passage) in enumerate(pairs):
            cached = self._store.get(_NAMESPACE, _key(self._model, question, passage))
            if cached is None:
                misses.append((idx, (question, passage)))
            else:
                counters().increment("cache.cross_encoder.hit")
                scores[idx] = float(cached)

        if misses:
            counters().increment("cache.cross_encoder.miss", len(misses))
            fresh_pairs = [pair for _, pair in misses]
            fresh_scores = self._inner.predict(fresh_pairs)
            for (idx, (question, passage)), score in zip(misses, fresh_scores):
                value = float(score)
                scores[idx] = value
                self._store.set(_NAMESPACE, _key(self._model, question, passage), value)

        return [s if s is not None else 0.0 for s in scores]
