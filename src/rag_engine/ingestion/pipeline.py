from pathlib import Path

from rag_engine.chunking.semantic import chunk_documents
from rag_engine.config.settings import Settings
from rag_engine.ingestion.loaders import load_documents
from rag_engine.retrieval.bm25_store import build_bm25_index
from rag_engine.vectorstore.qdrant_store import index_documents


def ingest_path(source: Path, settings: Settings) -> int:
    documents = load_documents(source)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0
    index_documents(chunks, settings)
    if settings.retrieval_mode == "hybrid":
        build_bm25_index(chunks, Path(settings.bm25_index_path))
    return len(chunks)
