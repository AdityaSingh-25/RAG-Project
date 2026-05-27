"""Tests for the chunker dispatcher.

The recursive path is exercised directly. The semantic path is patched so
we don't need to load a real embedding model — we verify the dispatcher
routes to ``SemanticChunker`` and threads the breakpoint type through.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag_engine.chunking.semantic import (
    chunk_documents,
    chunk_documents_with_settings,
)
from rag_engine.config.settings import Settings


def _doc(text: str, **meta: object) -> Document:
    return Document(page_content=text, metadata=dict(meta))


def test_recursive_chunker_back_compat_helper_assigns_chunk_ids() -> None:
    docs = [_doc("a" * 1500), _doc("b" * 1500)]
    out = chunk_documents(docs, chunk_size=900, chunk_overlap=120)
    assert len(out) >= 2
    assert all("chunk_id" in c.metadata for c in out)
    assert [c.metadata["chunk_id"] for c in out] == list(range(len(out)))


def test_dispatcher_uses_recursive_by_default() -> None:
    docs = [_doc("paragraph one.\n\nparagraph two.\n\nparagraph three.")]
    settings = Settings(chunk_size=200, chunk_overlap=0)
    out = chunk_documents_with_settings(docs, settings)
    assert all("chunk_id" in c.metadata for c in out)


def test_dispatcher_routes_to_semantic_when_configured(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeSemanticChunker:
        def __init__(self, embeddings, breakpoint_threshold_type, **_kw):
            captured["embeddings"] = embeddings
            captured["breakpoint"] = breakpoint_threshold_type

        def split_documents(self, documents):
            # Pretend the semantic split produces two distinct chunks.
            return [
                Document(page_content=documents[0].page_content[:50], metadata={}),
                Document(page_content=documents[0].page_content[50:], metadata={}),
            ]

    monkeypatch.setattr(
        "langchain_experimental.text_splitter.SemanticChunker",
        _FakeSemanticChunker,
    )

    def fake_embeddings(_settings):
        return "fake-embeddings-sentinel"

    monkeypatch.setattr(
        "rag_engine.embeddings.factory.build_cached_embeddings",
        fake_embeddings,
    )

    docs = [_doc("a" * 200)]
    settings = Settings(
        chunking_mode="semantic",
        semantic_breakpoint_type="standard_deviation",
    )
    out = chunk_documents_with_settings(docs, settings)

    assert captured["embeddings"] == "fake-embeddings-sentinel"
    assert captured["breakpoint"] == "standard_deviation"
    assert len(out) == 2
    assert [c.metadata["chunk_id"] for c in out] == [0, 1]
