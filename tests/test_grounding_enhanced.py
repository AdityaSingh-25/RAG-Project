from langchain_core.documents import Document

from rag_engine.evaluation.hallucination import score_grounding, verify_answer_confidence


def test_verify_answer_confidence_returns_dict_with_required_keys() -> None:
    doc = Document(page_content="Qdrant is a vector database.")
    answer = "Qdrant is a vector database [1]."
    result = verify_answer_confidence(answer, [doc])
    assert "grounding_score" in result
    assert "is_confident" in result
    assert "has_citations" in result
    assert result["has_citations"] is True


def test_score_grounding_detects_missing_citations() -> None:
    doc = Document(page_content="Qdrant is a vector database.")
    answer = "Qdrant is a vector database."
    score, warnings = score_grounding(answer, [doc])
    assert "missing_citations" in warnings


def test_score_grounding_detects_uncertain_language() -> None:
    doc = Document(page_content="Feature X does Y.")
    answer = "Feature X might do Y, possibly [1]."
    score, warnings = score_grounding(answer, [doc])
    assert "high_uncertainty_language" in warnings or len(warnings) > 0
