"""Second-stage neural reranker using a cross-encoder.

A cross-encoder reads the (question, passage) pair jointly and outputs a
single relevance score. This is strictly more accurate than the dot-product
of independent embeddings, but ~3-5x slower, which is why it only runs over
the small candidate pool returned by hybrid retrieval — not the full corpus.

The default model is ``cross-encoder/ms-marco-MiniLM-L-6-v2``: ~80 MB,
trained on MS MARCO, the standard baseline in IR literature. Substitute
via ``CROSS_ENCODER_MODEL`` if you have a domain-tuned model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class CrossEncoderLike(Protocol):
    """Subset of sentence-transformers' CrossEncoder that we depend on."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoderLike:
    """Lazy-load and cache the cross-encoder so model files are read once."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def load_cross_encoder_with_cache(
    model_name: str,
    cache_enabled: bool,
    cache_path: str,
    cache_ttl_seconds: int,
) -> CrossEncoderLike:
    """Return the raw cross-encoder, optionally wrapped in the per-pair cache."""
    inner = _load_cross_encoder(model_name)
    if not cache_enabled:
        return inner
    from rag_engine.cache.reranker_cache import CachingCrossEncoder
    from rag_engine.cache.store import CacheStore

    store = CacheStore(Path(cache_path), cache_ttl_seconds)
    return CachingCrossEncoder(inner=inner, store=store, model_name=model_name)


def rerank_with_cross_encoder(
    question: str,
    documents: list[Document],
    top_k: int,
    model_name: str,
    *,
    encoder: CrossEncoderLike | None = None,
) -> list[Document]:
    """Rerank ``documents`` with a cross-encoder and return the top ``top_k``.

    ``encoder`` is injectable so tests can stub the model without pulling weights.
    Each returned document carries the raw score in ``metadata["rerank_score"]``.
    """
    if not documents:
        return []
    if encoder is None:
        encoder = _load_cross_encoder(model_name)
    pairs = [(question, doc.page_content) for doc in documents]
    scores = encoder.predict(pairs)
    scored = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": float(score)},
        )
        for score, doc in scored[:top_k]
    ]
