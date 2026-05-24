"""Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused with Reciprocal Rank Fusion.

RRF treats each retriever's output as a ranked list and combines scores via
``sum_over_lists(1 / (rrf_k + rank))``. It needs no score calibration between
retrievers, which is why it's the default fusion strategy in production RAG
stacks. ``rrf_k`` defaults to 60, the value reported in the original Cormack
et al. paper; smaller values weight top ranks more aggressively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from langchain_core.documents import Document

from rag_engine.retrieval.bm25_store import BM25Index


class DenseRetrieverLike(Protocol):
    def invoke(self, query: str) -> list[Document]: ...


def _doc_key(doc: Document) -> tuple[str, str]:
    """A stable identity for fusion. Source + content hash avoids depending on a vector id."""
    return (
        str(doc.metadata.get("source", "")),
        doc.page_content[:120],
    )


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[Document]],
    rrf_k: int,
    top_k: int,
) -> list[Document]:
    """Fuse multiple ranked document lists into a single ranking via RRF."""
    scores: dict[tuple[str, str], float] = {}
    representatives: dict[tuple[str, str], Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            representatives.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused: list[Document] = []
    for key, score in ordered[:top_k]:
        doc = representatives[key]
        merged_metadata = {**doc.metadata, "rrf_score": round(score, 6)}
        fused.append(Document(page_content=doc.page_content, metadata=merged_metadata))
    return fused


@dataclass
class HybridRetriever:
    """Combines a dense retriever and a (possibly absent) BM25 index via RRF."""

    dense: DenseRetrieverLike
    bm25_index: BM25Index | None
    top_k: int
    rrf_k: int

    def invoke(self, query: str) -> list[Document]:
        dense_hits = list(self.dense.invoke(query))
        if self.bm25_index is None:
            return dense_hits[: self.top_k]
        sparse_hits = [doc for _, _, doc in self.bm25_index.search(query, top_k=self.top_k)]
        if not sparse_hits:
            return dense_hits[: self.top_k]
        return reciprocal_rank_fusion(
            ranked_lists=[dense_hits, sparse_hits],
            rrf_k=self.rrf_k,
            top_k=self.top_k,
        )
