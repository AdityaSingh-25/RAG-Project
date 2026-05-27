"""Tests for the NLI per-claim verifier.

The real DeBERTa cross-encoder is ~700MB; we stub it so tests don't touch
HuggingFace. The bits worth verifying without the model are:

- ``EntailmentScorer`` picks the right output column based on ``id2label``.
- ``score_claim_nli`` max-pools across cited chunks.
- The dispatcher in ``verify_claims_with_settings`` routes to the NLI scorer
  when ``claim_verifier_mode == "nli"``.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from rag_engine.config.settings import Settings
from rag_engine.evaluation.claim_grounding import verify_claims_with_settings
from rag_engine.evaluation.nli_verifier import (
    EntailmentScorer,
    score_claim_nli,
)


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={})


class _FakeRawNli:
    """Pretends to be sentence-transformers' CrossEncoder for an NLI model."""

    def __init__(self, rows: Sequence[Sequence[float]], id2label: dict[int, str]):
        self._rows = rows
        self.config = type("Cfg", (), {"id2label": id2label})()
        self.calls: list[tuple[list[tuple[str, str]], bool]] = []

    def predict(self, pairs: list[tuple[str, str]], apply_softmax: bool = False):
        # Lock in apply_softmax=True so the adapter never reads raw logits.
        self.calls.append((list(pairs), apply_softmax))
        return [self._rows[i % len(self._rows)] for i in range(len(pairs))]


class _StubEntailmentScorer:
    """Map (premise, hypothesis) → P(entail) by (passage, claim) lookup."""

    def __init__(self, scores: dict[tuple[str, str], float]):
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [self._scores.get(p, 0.0) for p in pairs]


def test_entailment_scorer_picks_column_from_id2label() -> None:
    # Standard ordering: contradiction / entailment / neutral.
    raw = _FakeRawNli(
        rows=[[0.05, 0.90, 0.05], [0.40, 0.10, 0.50]],
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
    )
    scorer = EntailmentScorer(raw)
    out = scorer.predict([("p1", "h1"), ("p2", "h2")])
    assert out == [0.90, 0.10]
    # Adapter must always softmax — raw logits aren't probabilities.
    assert raw.calls[-1][1] is True


def test_entailment_scorer_handles_alternate_label_ordering() -> None:
    raw = _FakeRawNli(
        rows=[[0.10, 0.20, 0.70]],
        # entailment is at index 2 here.
        id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
    )
    scorer = EntailmentScorer(raw)
    assert scorer.predict([("p", "h")]) == [0.70]


def test_entailment_scorer_falls_back_to_index_one_when_labels_missing() -> None:
    raw = _FakeRawNli(rows=[[0.10, 0.80, 0.10]], id2label={})
    scorer = EntailmentScorer(raw)
    assert scorer.predict([("p", "h")]) == [0.80]


def test_entailment_scorer_returns_empty_on_no_pairs() -> None:
    raw = _FakeRawNli(rows=[[1.0, 0.0, 0.0]], id2label={1: "entailment"})
    assert EntailmentScorer(raw).predict([]) == []


def test_score_claim_nli_uses_entailment_probability() -> None:
    docs = [_doc("Qdrant is a vector database for semantic search.")]
    scorer = _StubEntailmentScorer({
        ("Qdrant is a vector database for semantic search.", "Qdrant is a vector database"): 0.92,
    })
    out = score_claim_nli(
        "Qdrant is a vector database [1].",
        docs,
        support_threshold=0.5,
        scorer=scorer,
    )
    assert out.valid_indices == (1,)
    assert out.support_score == 0.92
    assert out.is_grounded is True


def test_score_claim_nli_max_pools_over_cited_chunks() -> None:
    docs = [
        _doc("Premise A"),
        _doc("Premise B"),
        _doc("Premise C"),
    ]
    # Claim cites three chunks; only chunk C entails strongly.
    scorer = _StubEntailmentScorer({
        ("Premise A", "X"): 0.10,
        ("Premise B", "X"): 0.20,
        ("Premise C", "X"): 0.85,
    })
    out = score_claim_nli(
        "X [1][2][3].",
        docs,
        support_threshold=0.5,
        scorer=scorer,
    )
    assert out.support_score == 0.85
    assert out.is_grounded is True
    # All three pairs were sent for scoring.
    assert len(scorer.calls[0]) == 3


def test_score_claim_nli_marks_uncited_sentence_as_ungrounded() -> None:
    docs = [_doc("Premise A")]
    scorer = _StubEntailmentScorer({})
    out = score_claim_nli("Bare claim.", docs, support_threshold=0.5, scorer=scorer)
    assert out.cited_indices == ()
    assert out.is_grounded is False
    # No pairs were sent — we short-circuited.
    assert scorer.calls == []


def test_score_claim_nli_drops_out_of_range_citations() -> None:
    docs = [_doc("Only chunk")]
    scorer = _StubEntailmentScorer({("Only chunk", "Y"): 0.75})
    out = score_claim_nli("Y [1][9].", docs, support_threshold=0.5, scorer=scorer)
    assert out.cited_indices == (1, 9)
    assert out.valid_indices == (1,)
    assert out.is_grounded is True


def test_score_claim_nli_pure_citation_sentence_is_grounded() -> None:
    docs = [_doc("Yes — confirmed.")]
    scorer = _StubEntailmentScorer({})
    out = score_claim_nli("[1].", docs, support_threshold=0.5, scorer=scorer)
    assert out.is_grounded is True
    assert out.support_score == 1.0
    assert scorer.calls == []  # short-circuit, no model call


def test_verify_claims_with_settings_routes_to_nli_when_configured(monkeypatch) -> None:
    docs = [_doc("The cat sat on the mat.")]
    scorer = _StubEntailmentScorer({
        ("The cat sat on the mat.", "Cat on mat"): 0.81,
    })

    def fake_loader(**_kw):
        return scorer

    monkeypatch.setattr(
        "rag_engine.evaluation.claim_grounding.load_nli_with_cache",
        # The import inside the dispatcher is what we need to patch — patch
        # the module attribute the dispatcher reads.
        fake_loader,
        raising=False,
    )
    # The dispatcher imports lazily, so also patch the source module so the
    # lazy import resolves to our fake before the dispatcher closes over it.
    monkeypatch.setattr(
        "rag_engine.evaluation.nli_verifier.load_nli_with_cache",
        fake_loader,
    )

    settings = Settings(
        claim_verifier_mode="nli",
        claim_support_threshold=0.5,
        cache_enabled=False,
    )
    report = verify_claims_with_settings("Cat on mat [1].", docs, settings)

    assert len(report.claims) == 1
    assert report.claims[0].is_grounded is True
    assert report.grounded_claim_rate == 1.0
    assert scorer.calls, "NLI scorer should have been invoked"


def test_verify_claims_with_settings_uses_overlap_by_default() -> None:
    docs = [_doc("Qdrant is a vector database for semantic search.")]
    settings = Settings(claim_support_threshold=0.2)
    # Default mode is "overlap" — no NLI patching required.
    report = verify_claims_with_settings(
        "Qdrant is a vector database [1].", docs, settings
    )
    assert len(report.claims) == 1
    assert report.claims[0].is_grounded is True
