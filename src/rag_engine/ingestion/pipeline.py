from dataclasses import dataclass
from pathlib import Path

from rag_engine.chunking.semantic import chunk_documents_with_settings
from rag_engine.config.settings import Settings
from rag_engine.ingestion.dedup import dedupe_chunks
from rag_engine.ingestion.loaders import load_documents
from rag_engine.retrieval.bm25_store import build_bm25_index
from rag_engine.vectorstore.qdrant_store import existing_content_hashes, index_documents


@dataclass
class IngestReport:
    indexed: int
    # Chunks dropped because the same content_hash appeared earlier in this
    # same batch (in-process dedup, has been here since the original feature).
    duplicates_removed: int
    # Chunks dropped because Qdrant already had a point with the same
    # content_hash from a previous ingest. Zero on first ever ingest.
    cross_run_duplicates_removed: int = 0


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

    cross_run_removed = 0
    if settings.enable_ingest_dedup and chunks:
        candidates = [
            chunk.metadata.get("content_hash")
            for chunk in chunks
            if isinstance(chunk.metadata.get("content_hash"), str)
        ]
        already_indexed = existing_content_hashes(settings, candidates)
        if already_indexed:
            filtered = [
                chunk
                for chunk in chunks
                if chunk.metadata.get("content_hash") not in already_indexed
            ]
            cross_run_removed = len(chunks) - len(filtered)
            chunks = filtered

    if not chunks:
        return IngestReport(
            indexed=0,
            duplicates_removed=duplicates_removed,
            cross_run_duplicates_removed=cross_run_removed,
        )

    index_documents(chunks, settings)
    if settings.retrieval_mode == "hybrid":
        build_bm25_index(chunks, Path(settings.bm25_index_path))
    return IngestReport(
        indexed=len(chunks),
        duplicates_removed=duplicates_removed,
        cross_run_duplicates_removed=cross_run_removed,
    )
