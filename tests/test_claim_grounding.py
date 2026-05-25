from langchain_core.documents import Document

from rag_engine.evaluation.claim_grounding import (
    parse_citations,
    score_claim,
    split_into_sentences,
    verify_claims,
)


def _doc(text: str, **metadata: object) -> Document:
    return Document(page_content=text, metadata=dict(metadata))


def test_split_into_sentences_handles_basic_punctuation() -> None:
    sentences = split_into_sentences(
        "Qdrant stores vectors [1]. Ollama runs models locally [2]!"
    )
    assert sentences == [
        "Qdrant stores vectors [1].",
        "Ollama runs models locally [2]!",
    ]


def test_parse_citations_handles_multiple_and_adjacent_markers() -> None:
    assert parse_citations("Claim with [1][2] and another [3].") == (1, 2, 3)
    assert parse_citations("No citations here.") == ()


def test_score_claim_marks_uncited_sentence_as_ungrounded() -> None:
    docs = [_doc("Qdrant is a vector database.")]
    out = score_claim("Qdrant is a vector database.", docs, support_threshold=0.2)
    assert out.cited_indices == ()
    assert out.is_grounded is False


def test_score_claim_flags_citation_index_out_of_range() -> None:
    docs = [_doc("Qdrant is a vector database.")]
    out = score_claim("Some claim [5].", docs, support_threshold=0.2)
    assert out.cited_indices == (5,)
    assert out.valid_indices == ()
    assert out.is_grounded is False


def test_score_claim_passes_when_cited_chunk_supports_terms() -> None:
    docs = [_doc("Qdrant is a vector database used for semantic search.")]
    out = score_claim(
        "Qdrant is a vector database [1].",
        docs,
        support_threshold=0.2,
    )
    assert out.valid_indices == (1,)
    assert out.is_grounded is True
    assert out.support_score > 0.5


def test_score_claim_fails_when_cited_chunk_does_not_support_terms() -> None:
    docs = [_doc("Ollama runs large language models locally.")]
    out = score_claim(
        "Qdrant supports HNSW indexes [1].",
        docs,
        support_threshold=0.5,
    )
    assert out.valid_indices == (1,)
    assert out.is_grounded is False
    assert out.support_score < 0.5


def test_verify_claims_aggregates_grounded_rate() -> None:
    docs = [
        _doc("Qdrant is a vector database for semantic search."),
        _doc("Ollama serves local LLMs."),
    ]
    answer = (
        "Qdrant is a vector database [1]. Ollama serves local LLMs [2]. "
        "It also makes coffee [1]."
    )
    report = verify_claims(answer, docs, support_threshold=0.2)
    assert len(report.claims) == 3
    grounded = [c.is_grounded for c in report.claims]
    assert grounded[0] is True
    assert grounded[1] is True
    # "makes coffee" is unsupported by chunk 1's content.
    assert grounded[2] is False
    assert report.grounded_claim_rate == round(2 / 3, 3)


def test_verify_claims_handles_empty_answer() -> None:
    report = verify_claims("", [], support_threshold=0.2)
    assert report.claims == []
    assert report.grounded_claim_rate == 0.0


def test_verify_claims_short_pure_citation_sentence_is_grounded() -> None:
    docs = [_doc("Yes — confirmed in the docs.")]
    report = verify_claims("Yes [1].", docs, support_threshold=0.2)
    # A sentence with no content terms but a valid cite should be grounded;
    # it gives the LLM credit for declining to invent content.
    assert report.claims[0].is_grounded is True
