from langchain_core.documents import Document

from rag_engine.retrieval.reranker import (
    filter_by_score_threshold,
    get_relevance_score,
    rerank_documents,
)


def test_rerank_documents_returns_same_or_fewer() -> None:
    docs = [
        Document(page_content="Qdrant is a vector database.", metadata={"score": 0.9}),
        Document(page_content="Vector databases store embeddings.", metadata={"score": 0.7}),
        Document(page_content="Llama is a language model.", metadata={"score": 0.5}),
    ]
    result = rerank_documents("What is Qdrant?", docs, top_k=2)
    assert len(result) <= 2


def test_get_relevance_score_returns_float() -> None:
    doc = Document(page_content="Qdrant vector database embeddings")
    score = get_relevance_score("What is Qdrant?", doc)
    assert 0.0 <= score <= 1.0


def test_filter_by_score_threshold_removes_low_score_docs() -> None:
    docs = [
        Document(page_content="High quality.", metadata={"score": 0.9}),
        Document(page_content="Low quality.", metadata={"score": 0.1}),
    ]
    result = filter_by_score_threshold(docs, threshold=0.5)
    assert len(result) == 1
    assert result[0].metadata["score"] == 0.9
