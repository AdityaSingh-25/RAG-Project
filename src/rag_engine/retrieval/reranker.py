import re

from langchain_core.documents import Document

from rag_engine.config.settings import Settings
from rag_engine.retrieval.cross_encoder_reranker import load_cross_encoder_with_cache
from rag_engine.retrieval.source_confidence import score_source_confidence


def _score_documents(
    question: str, documents: list[Document], settings: Settings
) -> list[float]:
    """Compute one reranker score per document. Does NOT truncate.

    Truncation is deferred until after source-confidence multiplication so a
    confident-but-mid-ranked doc can still rescue itself into the top_k.
    """
    mode = settings.reranker_mode
    if mode == "disabled":
        return [
            float(
                doc.metadata.get("rrf_score")
                or doc.metadata.get("score")
                or 0.0
            )
            for doc in documents
        ]
    if mode == "cross_encoder":
        encoder = load_cross_encoder_with_cache(
            model_name=settings.cross_encoder_model,
            cache_enabled=settings.cache_enabled,
            cache_path=settings.cache_path,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        )
        pairs = [(question, doc.page_content) for doc in documents]
        return [float(s) for s in encoder.predict(pairs)]
    return [get_relevance_score(question, doc) for doc in documents]


def apply_reranker(
    question: str, documents: list[Document], settings: Settings
) -> list[Document]:
    """Score, multiply by source confidence, sort, then truncate to ``top_k``.

    Three reranker modes:

    - ``cross_encoder`` (default): neural cross-encoder over the candidate pool.
    - ``keyword``: term-overlap heuristic; no model download required.
    - ``disabled``: passthrough; preserves the retriever's order.

    When ``settings.enable_source_confidence`` is true, each document's
    reranker score is multiplied by a confidence factor (freshness + trust
    + retriever agreement). The final ordering reflects both relevance to
    the query and the engine's trust in the source.
    """
    if not documents:
        return []

    rerank_scores = _score_documents(question, documents, settings)
    enriched: list[tuple[float, Document]] = []
    for doc, rerank_score in zip(documents, rerank_scores):
        confidence = score_source_confidence(doc, settings)
        final = rerank_score * confidence
        enriched.append(
            (
                final,
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "rerank_score": round(rerank_score, 6),
                        "source_confidence": round(confidence, 6),
                        "final_score": round(final, 6),
                    },
                ),
            )
        )
    enriched.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in enriched[: settings.top_k]]


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
