from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from rag_engine.config.settings import Settings
from rag_engine.retrieval.bm25_store import load_bm25_index
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.vectorstore.qdrant_store import build_vectorstore_for


class Retriever(Protocol):
    def invoke(self, query: str) -> list[Document]: ...


def build_retriever(settings: Settings) -> Retriever:
    """Build a retriever according to ``settings.retrieval_mode``.

    - ``dense``: vector-only retrieval from Qdrant.
    - ``hybrid``: dense + BM25 fused with Reciprocal Rank Fusion. Falls back
      to dense-only when the BM25 index is absent (e.g., no ingest has run yet).

    First-stage retrieval fetches ``settings.retrieve_k`` candidates so the
    downstream reranker has a meaningful pool to choose from before narrowing
    to ``settings.top_k``.
    """
    vectorstore = build_vectorstore_for(settings)
    dense = vectorstore.as_retriever(search_kwargs={"k": settings.retrieve_k})

    if settings.retrieval_mode == "dense":
        return dense

    bm25_index = load_bm25_index(Path(settings.bm25_index_path))
    return HybridRetriever(
        dense=dense,
        bm25_index=bm25_index,
        top_k=settings.retrieve_k,
        rrf_k=settings.rrf_k,
    )
