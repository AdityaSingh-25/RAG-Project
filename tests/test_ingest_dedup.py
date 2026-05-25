from langchain_core.documents import Document

from rag_engine.ingestion.dedup import chunk_hash, dedupe_chunks


def _doc(text: str, **metadata: object) -> Document:
    return Document(page_content=text, metadata=dict(metadata))


def test_chunk_hash_ignores_whitespace_and_case_differences() -> None:
    a = chunk_hash("Qdrant is a vector database.")
    b = chunk_hash("  qdrant   is\n a  VECTOR\tdatabase. ")
    assert a == b


def test_dedupe_keeps_first_occurrence_and_counts_removed() -> None:
    chunks = [
        _doc("Qdrant is a vector database.", source="docs/qdrant.md"),
        _doc("Ollama serves local LLMs.", source="docs/ollama.md"),
        _doc("Qdrant is a vector database.", source="data/raw/notes/copy.md"),
        _doc("Ollama serves local LLMs.", source="docs/dupe.md"),
    ]
    result = dedupe_chunks(chunks)
    assert result.duplicates_removed == 2
    assert [d.metadata["source"] for d in result.unique] == [
        "docs/qdrant.md",
        "docs/ollama.md",
    ]


def test_dedupe_attaches_content_hash_to_metadata() -> None:
    chunks = [_doc("hello world", source="a")]
    result = dedupe_chunks(chunks)
    assert result.unique[0].metadata["content_hash"] == chunk_hash("hello world")


def test_dedupe_reuses_existing_hash_when_present() -> None:
    """If a loader already computed a hash, dedup should respect it."""
    chunks = [
        _doc("custom payload", content_hash="precomputed-abc"),
        _doc("different content", content_hash="precomputed-abc"),
    ]
    result = dedupe_chunks(chunks)
    # Both chunks claim the same precomputed hash; the second is dropped.
    assert result.duplicates_removed == 1
    assert result.unique[0].page_content == "custom payload"


def test_dedupe_on_empty_input() -> None:
    result = dedupe_chunks([])
    assert result.unique == []
    assert result.duplicates_removed == 0
