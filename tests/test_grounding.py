from langchain_core.documents import Document

from rag_engine.evaluation.hallucination import score_grounding


def test_grounding_scores_supported_answer() -> None:
    docs = [Document(page_content="Qdrant stores document embeddings for semantic search.")]
    score, warnings = score_grounding("Qdrant stores embeddings for semantic search [1].", docs)
    assert score > 0.5
    assert "missing_citations" not in warnings


def test_grounding_warns_without_documents() -> None:
    score, warnings = score_grounding("A claim with no context.", [])
    assert score == 0.0
    assert "no_retrieved_context" in warnings

