from pathlib import Path

from langchain_core.documents import Document

from rag_engine.retrieval.bm25_store import build_bm25_index, load_bm25_index, tokenize
from rag_engine.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


class _StubDense:
    def __init__(self, results: list[Document]) -> None:
        self._results = results

    def invoke(self, _query: str) -> list[Document]:
        return list(self._results)


def _doc(text: str, source: str) -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_tokenize_lowercases_and_splits_on_non_alnum() -> None:
    assert tokenize("Hello, World-Class System!") == ["hello", "world-class", "system"]


def test_bm25_round_trip_persists_and_loads(tmp_path: Path) -> None:
    docs = [
        _doc("Qdrant is a vector database for semantic search.", "qdrant.md"),
        _doc("Ollama serves local large language models on your machine.", "ollama.md"),
        _doc("BM25 is a sparse retrieval algorithm based on term frequency.", "bm25.md"),
    ]
    index_path = tmp_path / "bm25.pkl"
    build_bm25_index(docs, index_path)

    loaded = load_bm25_index(index_path)
    assert loaded is not None
    hits = loaded.search("BM25 sparse retrieval", top_k=2)
    assert hits, "BM25 should retrieve at least one match for an exact-keyword query"
    assert hits[0][2].metadata["source"] == "bm25.md"


def test_load_bm25_index_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_bm25_index(tmp_path / "missing.pkl") is None


def test_load_bm25_index_returns_none_on_empty_corpus(tmp_path: Path) -> None:
    index_path = tmp_path / "empty.pkl"
    build_bm25_index([], index_path)
    assert load_bm25_index(index_path) is None


def test_rrf_prefers_documents_ranked_high_in_both_lists() -> None:
    a = _doc("alpha doc", "a")
    b = _doc("beta doc", "b")
    c = _doc("gamma doc", "c")

    dense_ranking = [a, b, c]
    sparse_ranking = [b, a, c]

    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], rrf_k=60, top_k=3)
    sources = [d.metadata["source"] for d in fused]
    # a is rank 1 in dense + rank 2 in sparse; b is rank 2 in dense + rank 1 in sparse.
    # Both should beat c (rank 3 in both lists).
    assert sources.index("c") == 2
    # Every fused doc carries an rrf_score in metadata.
    assert all("rrf_score" in d.metadata for d in fused)


def test_hybrid_retriever_falls_back_to_dense_when_no_bm25(tmp_path: Path) -> None:
    dense_docs = [_doc("the only doc we have", "only.md")]
    retriever = HybridRetriever(
        dense=_StubDense(dense_docs),
        bm25_index=None,
        top_k=5,
        rrf_k=60,
    )
    assert retriever.invoke("anything") == dense_docs


def test_hybrid_retriever_uses_bm25_for_keyword_queries(tmp_path: Path) -> None:
    # BM25 IDF needs a non-trivial corpus; with only two docs a term that
    # appears in exactly one of them scores zero. Use four docs so IDF stays
    # positive — mirrors realistic ingestion.
    docs = [
        _doc("Qdrant is a vector database.", "qdrant.md"),
        _doc("Ollama serves local LLMs.", "ollama.md"),
        _doc("Sentence transformers build dense embeddings.", "embeddings.md"),
        _doc("Reciprocal rank fusion combines ranked lists.", "rrf.md"),
    ]
    index_path = tmp_path / "bm25.pkl"
    build_bm25_index(docs, index_path)
    bm25 = load_bm25_index(index_path)
    assert bm25 is not None

    # Dense returns only ollama; BM25 finds qdrant via the keyword.
    dense_only = [docs[1]]
    retriever = HybridRetriever(
        dense=_StubDense(dense_only),
        bm25_index=bm25,
        top_k=5,
        rrf_k=60,
    )
    result_sources = {d.metadata["source"] for d in retriever.invoke("Qdrant")}
    assert "qdrant.md" in result_sources
    assert "ollama.md" in result_sources
