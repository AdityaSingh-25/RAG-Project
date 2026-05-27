from dataclasses import dataclass
from pathlib import Path

from rag_engine.chunking.semantic import chunk_documents_with_settings
from rag_engine.config.settings import Settings
from rag_engine.ingestion.dedup import dedupe_chunks
from rag_engine.ingestion.loaders import load_documents
from rag_engine.retrieval.bm25_store import build_bm25_index
from rag_engine.vectorstore.qdrant_store import index_documents


@dataclass
class IngestReport:
    indexed: int
    duplicates_removed: int


def ingest_path(source: Path, settings: Settings) -> IngestReport:
    documents = load_documents(source)
    chunks = chunk_documents_with_settings(documents, settings)
    if not chunks:
        return IngestReport(indexed=0, duplicates_removed=0)

    if settings.enable_ingest_dedup:
        result = dedupe_chunks(chunks)
        chunks = result.unique
        duplicates_removed = result.duplicates_removed
    else:
        duplicates_removed = 0

    if not chunks:
        return IngestReport(indexed=0, duplicates_removed=duplicates_removed)

    index_documents(chunks, settings)
    if settings.retrieval_mode == "hybrid":
        build_bm25_index(chunks, Path(settings.bm25_index_path))
    return IngestReport(indexed=len(chunks), duplicates_removed=duplicates_removed)
