"""Deterministic query rewriting for the retrieval feedback loop.

Strategy: lightweight pseudo-relevance feedback. Take the original question,
mine the top retrieved documents for content terms the question did not
contain, and append the strongest novel terms. The rewrite stays grounded
in the user's wording while widening the lexical surface area used for the
next retrieval pass.
"""

from __future__ import annotations

import re
from collections import Counter

from langchain_core.documents import Document

_STOPWORDS = {
    "the", "and", "or", "a", "an", "in", "is", "at", "be", "by", "for",
    "of", "on", "to", "with", "from", "have", "has", "had", "that", "this",
    "these", "those", "as", "are", "was", "were", "been", "which", "who",
    "what", "when", "where", "why", "how", "it", "its", "their", "them",
    "they", "we", "our", "you", "your", "i", "me", "my", "but", "not",
    "do", "does", "did", "can", "could", "would", "should", "will", "may",
    "might", "about", "into", "than", "then", "so", "if", "because",
}


def _terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in _STOPWORDS
    ]


def rewrite_query(
    question: str,
    documents: list[Document],
    *,
    max_expansion_terms: int = 4,
    docs_to_mine: int = 3,
) -> str:
    """Return an expanded query string for the next retrieval pass.

    The original question is preserved verbatim; novel high-signal terms
    from the top retrieved documents are appended. If there are no
    documents or no novel terms, the original question is returned
    unchanged so the loop can detect a fixed point and stop.
    """
    if not question.strip():
        return question
    if not documents or max_expansion_terms <= 0:
        return question

    existing = set(_terms(question))
    counter: Counter[str] = Counter()
    for doc in documents[:docs_to_mine]:
        for term in _terms(doc.page_content):
            if term in existing:
                continue
            counter[term] += 1

    if not counter:
        return question

    novel = [term for term, _ in counter.most_common(max_expansion_terms)]
    return f"{question.strip()} {' '.join(novel)}"
