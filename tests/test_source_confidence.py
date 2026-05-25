import datetime

from langchain_core.documents import Document

from rag_engine.config.settings import Settings
from rag_engine.retrieval.hybrid import reciprocal_rank_fusion
from rag_engine.retrieval.reranker import apply_reranker
from rag_engine.retrieval.source_confidence import (
    agreement_score,
    freshness_score,
    score_source_confidence,
    trust_score,
)


def _doc(text: str = "x", **metadata: object) -> Document:
    return Document(page_content=text, metadata=dict(metadata))


def test_freshness_decays_with_age() -> None:
    now = datetime.datetime(2026, 5, 26, tzinfo=datetime.timezone.utc).timestamp()
    # Half-life of 365 days: a doc one half-life old should score ~0.5.
    one_half_life_ago = (
        datetime.datetime(2025, 5, 26, tzinfo=datetime.timezone.utc).timestamp()
    )
    doc = _doc(published_at="2025-05-26T00:00:00+00:00")
    assert abs(freshness_score(doc, half_life_days=365, now=now) - 0.5) < 0.01
    assert one_half_life_ago < now  # sanity


def test_freshness_defaults_to_one_when_no_date_metadata() -> None:
    assert freshness_score(_doc(), half_life_days=365) == 1.0


def test_trust_score_matches_glob_pattern() -> None:
    weights = {"docs/**": 1.2, "data/raw/notes/**": 0.7}
    assert trust_score(_doc(source="docs/architecture.md"), weights) == 1.2
    assert trust_score(_doc(source="data/raw/notes/scratch.md"), weights) == 0.7
    assert trust_score(_doc(source="other/unknown.md"), weights) == 1.0


def test_trust_score_handles_empty_weights_map() -> None:
    assert trust_score(_doc(source="anything.md"), {}) == 1.0


def test_agreement_score_boosts_multi_list_docs() -> None:
    assert agreement_score(_doc(agreement_count=1), boost=1.2) == 1.0
    assert agreement_score(_doc(agreement_count=2), boost=1.2) == 1.2


def test_score_source_confidence_returns_one_when_disabled() -> None:
    settings = Settings(enable_source_confidence=False)
    # Even with a date that would heavily penalise, confidence is neutralised.
    doc = _doc(published_at="2000-01-01T00:00:00+00:00", source="docs/old.md")
    assert score_source_confidence(doc, settings) == 1.0


def test_rrf_attaches_agreement_count_per_doc() -> None:
    a = _doc("alpha", source="a")
    b = _doc("beta", source="b")
    c = _doc("gamma", source="c")
    fused = reciprocal_rank_fusion(
        ranked_lists=[[a, b], [b, c]],
        rrf_k=60,
        top_k=3,
    )
    counts = {d.metadata["source"]: d.metadata["agreement_count"] for d in fused}
    assert counts == {"a": 1, "b": 2, "c": 1}


def test_confidence_rescues_lower_reranked_doc_into_top_k() -> None:
    """A trusted doc with a slightly lower rerank score can outrank a less-trusted one."""
    settings = Settings(
        reranker_mode="disabled",
        top_k=1,
        enable_source_confidence=True,
        agreement_boost=1.0,  # isolate the trust signal
        source_weights={"docs/trusted.md": 2.0},
    )
    docs = [
        _doc("low-trust winner", source="data/raw/random.md", rrf_score=0.6),
        _doc("trusted runner-up", source="docs/trusted.md", rrf_score=0.5),
    ]
    out = apply_reranker("Q", docs, settings)
    # Without confidence, the first doc would win (0.6 > 0.5). With 2x trust,
    # the trusted doc reaches 1.0 final_score and overtakes it.
    assert out[0].metadata["source"] == "docs/trusted.md"
    assert round(out[0].metadata["final_score"], 3) == 1.0
