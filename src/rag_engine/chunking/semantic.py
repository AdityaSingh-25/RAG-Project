"""Chunkers for ingested documents.

Two strategies are supported:

- ``recursive`` (default): :class:`RecursiveCharacterTextSplitter`. Splits at
  paragraph / line / sentence boundaries by a character budget. Cheap, no
  embedding calls — but ignores topic shifts within a long paragraph.

- ``semantic``: :class:`SemanticChunker`. Embeds each sentence, computes the
  similarity gradient between adjacent sentences, and breaks where the
  similarity drops past a configurable percentile. Topic-coherent at the
  cost of one embedding call per sentence at ingest time.

We dispatch via :func:`chunk_documents` keyed on ``settings.chunking_mode``
so callers don't change. ``langchain-experimental`` is imported lazily so
the ``recursive`` deployment never touches it (the package is officially
in maintenance-mode; if it ever breaks we replace ``SemanticChunker`` with
a small inline implementation without touching the dispatcher).
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_engine.config.settings import Settings


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Back-compat entry point. Always uses the recursive splitter."""
    return _chunk_recursive(documents, chunk_size, chunk_overlap)


def chunk_documents_with_settings(
    documents: list[Document],
    settings: Settings,
) -> list[Document]:
    """Settings-aware dispatcher: picks recursive or semantic chunker."""
    if settings.chunking_mode == "semantic":
        return _chunk_semantic(documents, settings)
    return _chunk_recursive(documents, settings.chunk_size, settings.chunk_overlap)


def _chunk_recursive(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def _chunk_semantic(documents: list[Document], settings: Settings) -> list[Document]:
    """Embedding-similarity chunker. Loads embeddings lazily."""
    from langchain_experimental.text_splitter import SemanticChunker

    from rag_engine.embeddings.factory import build_cached_embeddings

    embeddings = build_cached_embeddings(settings)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=settings.semantic_breakpoint_type,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
