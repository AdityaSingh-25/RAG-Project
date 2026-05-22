from langchain_core.documents import Document

from rag_engine.retrieval.query_rewriter import rewrite_query


def test_rewrite_appends_novel_terms_from_documents() -> None:
    docs = [
        Document(page_content="Qdrant stores embeddings for semantic search using HNSW indexes."),
        Document(page_content="HNSW provides approximate nearest neighbor search."),
    ]
    rewritten = rewrite_query("What does Qdrant use for search?", docs)
    assert "what does qdrant use for search?" in rewritten.lower()
    assert "hnsw" in rewritten.lower()


def test_rewrite_returns_question_unchanged_when_no_documents() -> None:
    assert rewrite_query("anything", []) == "anything"


def test_rewrite_skips_terms_already_in_question() -> None:
    docs = [Document(page_content="Qdrant Qdrant Qdrant is a vector database.")]
    rewritten = rewrite_query("Tell me about Qdrant", docs)
    assert rewritten.lower().count("qdrant") == 1


def test_rewrite_caps_expansion_terms() -> None:
    docs = [
        Document(
            page_content=(
                "alpha beta gamma delta epsilon zeta eta theta iota kappa "
                "lambda mu nu xi omicron pi rho sigma tau upsilon"
            )
        )
    ]
    rewritten = rewrite_query("base question", docs, max_expansion_terms=3)
    appended = rewritten.replace("base question", "").split()
    assert len(appended) == 3


def test_rewrite_returns_question_unchanged_when_no_novel_terms() -> None:
    docs = [Document(page_content="apple banana")]
    rewritten = rewrite_query("apple banana", docs)
    assert rewritten == "apple banana"
