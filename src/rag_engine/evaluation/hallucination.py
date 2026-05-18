import re

from langchain_core.documents import Document


def score_grounding(answer: str, documents: list[Document]) -> tuple[float, list[str]]:
    """Score answer grounding against retrieved documents."""
    if not answer.strip():
        return 0.0, ["empty_answer"]
    if not documents:
        return 0.0, ["no_retrieved_context"]

    context = " ".join(doc.page_content.lower() for doc in documents)
    answer_terms = _important_terms(answer)
    if not answer_terms:
        return 0.5, ["answer_too_short_for_grounding_check"]

    supported = sum(1 for term in answer_terms if term in context)
    term_score = supported / len(answer_terms)

    citation_score = _check_citations(answer)
    hallucination_risk = _detect_uncertain_language(answer)

    final_score = (0.6 * term_score) + (0.3 * citation_score) + (0.1 * (1.0 - hallucination_risk))
    warnings: list[str] = []

    if term_score < 0.35:
        warnings.append("low_context_overlap")
    if citation_score < 0.5:
        warnings.append("missing_citations")
    if hallucination_risk > 0.5:
        warnings.append("high_uncertainty_language")

    return round(final_score, 3), warnings


def verify_answer_confidence(answer: str, documents: list[Document]) -> dict[str, float | bool]:
    """Return detailed confidence metrics for the answer."""
    score, warnings = score_grounding(answer, documents)
    has_citations = "[" in answer and "]" in answer
    has_uncertainty = _detect_uncertain_language(answer) > 0.5

    return {
        "grounding_score": score,
        "is_confident": score >= 0.6 and has_citations and not has_uncertainty,
        "has_citations": has_citations,
        "uncertainty_level": _detect_uncertain_language(answer),
        "warnings": warnings,
    }


def _important_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "have",
        "your",
        "will",
        "are",
        "for",
        "was",
        "were",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
        if token not in stopwords
    }


def _check_citations(answer: str) -> float:
    """Score based on presence and count of citations."""
    citation_count = len(re.findall(r"\[\d+\]", answer))
    if citation_count == 0:
        return 0.0
    if citation_count >= 3:
        return 1.0
    return min(citation_count / 3.0, 1.0)


def _detect_uncertain_language(answer: str) -> float:
    """Detect hedging and uncertainty language in the answer."""
    uncertain_markers = [
        "might",
        "may",
        "could",
        "possibly",
        "perhaps",
        "unclear",
        "uncertain",
        "unknown",
        "insufficient",
        "i think",
        "probably",
        "seems",
        "appears",
    ]
    lower = answer.lower()
    matches = sum(1 for marker in uncertain_markers if marker in lower)
    return min(matches / 5.0, 1.0)

