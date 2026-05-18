import re
from typing import Any

from langchain_core.documents import Document


def rerank_documents(
    question: str, documents: list[Document], top_k: int = 6
) -> list[Document]:
    """Rerank documents by relevance to the question using keyword overlap and position signals."""
    if not documents:
        return []

    scored = []
    question_terms = _extract_terms(question)

    for doc in documents:
        content_terms = _extract_terms(doc.page_content)
        overlap = len(question_terms & content_terms)
        density = overlap / max(len(question_terms), 1) if question_terms else 0
        existing_score = doc.metadata.get("score", 0)
        combined_score = (0.5 * density) + (0.5 * (existing_score if isinstance(existing_score, (int, float)) else 0))
        scored.append((combined_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def get_relevance_score(question: str, document: Document) -> float:
    """Compute relevance score between 0 and 1 for a single document."""
    question_terms = _extract_terms(question)
    if not question_terms:
        return 0.5

    content_terms = _extract_terms(document.page_content)
    overlap = len(question_terms & content_terms)
    density = overlap / len(question_terms)
    return min(density, 1.0)


def filter_by_score_threshold(documents: list[Document], threshold: float = 0.2) -> list[Document]:
    """Filter out low-scoring documents based on metadata score."""
    return [
        doc
        for doc in documents
        if isinstance(doc.metadata.get("score"), (int, float))
        and doc.metadata.get("score", 0) >= threshold
    ]


def _extract_terms(text: str) -> set[str]:
    """Extract meaningful terms from text for comparison."""
    stopwords = {
        "the",
        "and",
        "or",
        "a",
        "an",
        "in",
        "is",
        "at",
        "be",
        "by",
        "for",
        "of",
        "on",
        "to",
        "with",
        "from",
        "have",
        "has",
        "had",
        "that",
        "this",
        "these",
        "those",
        "as",
        "are",
        "was",
        "were",
        "been",
        "which",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
    }
    terms = set(re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", text.lower()))
    return {term for term in terms if term not in stopwords}
