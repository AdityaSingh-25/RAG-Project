"""Natural-language-inference variant of the per-claim verifier.

The default (overlap) scorer in :mod:`rag_engine.evaluation.claim_grounding`
counts how many of a claim's content words appear in the cited chunks. That
catches gross drift but happily passes paraphrased contradictions ("Qdrant
does NOT support HNSW [1]" still has the right words). An NLI cross-encoder
reads the (premise, hypothesis) pair jointly and outputs an entailment
probability — strictly more accurate, with the same shape of inputs.

Design choices that drop out of the architecture:

- We wrap the raw 3-class cross-encoder in :class:`EntailmentScorer` so the
  public `.predict(pairs) -> list[float]` interface matches the reranker's
  ``CrossEncoderLike`` protocol. This lets us reuse :class:`CachingCrossEncoder`
  unchanged — the cache key already includes ``model_name``.
- For a claim citing multiple chunks we score each (chunk, claim) pair and
  keep the **max** entailment probability. If any cited chunk entails the
  claim, we count it as grounded. Averaging would mask one strong support.
- ``support_threshold`` is reinterpreted as the entailment-probability floor.
  This is a different scale than overlap fraction; the default of 0.2 is
  loose for NLI — recommend ~0.5 in deployments. (Left as a knob, not a
  hard-coded different default, so the settings surface stays simple.)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from rag_engine.evaluation.claim_grounding import (
    ClaimGrounding,
    _strip_citations,
    parse_citations,
)

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_FINAL_PUNCT = " .!?,;:"


class EntailmentScorerLike(Protocol):
    """Same shape as ``CrossEncoderLike`` — one scalar per pair.

    For NLI use, the returned scalar is P(entailment) in [0, 1].
    """

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class EntailmentScorer:
    """Adapter: raw 3-class NLI cross-encoder → entailment probability per pair.

    Reads ``id2label`` off the underlying model to find the entailment class
    index — different NLI models use different label orderings.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._entailment_idx = _find_entailment_index(model)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        # ``apply_softmax`` normalises the 3 logits into a distribution; we
        # take the entailment slot.
        scores = self._model.predict(pairs, apply_softmax=True)
        return [float(row[self._entailment_idx]) for row in scores]


def _find_entailment_index(model) -> int:
    """Inspect the model's label map to find which output column is entailment.

    Falls back to index 1 (the most common ordering) if the label map is
    missing or unrecognisable, with a warning logged by the caller.
    """
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not id2label:
        return 1
    for idx, label in id2label.items():
        if str(label).lower().startswith("entail"):
            return int(idx)
    return 1


@lru_cache(maxsize=2)
def _load_nli_model(model_name: str):
    """Lazy-load and cache the NLI cross-encoder so weights are read once."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def load_nli_with_cache(
    model_name: str,
    cache_enabled: bool,
    cache_path: str,
    cache_ttl_seconds: int,
) -> EntailmentScorerLike:
    """Return the entailment scorer, optionally wrapped in the per-pair cache."""
    inner: EntailmentScorerLike = EntailmentScorer(_load_nli_model(model_name))
    if not cache_enabled:
        return inner
    from rag_engine.cache.reranker_cache import CachingCrossEncoder
    from rag_engine.cache.store import CacheStore

    store = CacheStore(Path(cache_path), cache_ttl_seconds)
    # Reuse the cross-encoder cache wrapper — the predict signature is
    # identical and the model name namespaces NLI entries from reranker entries.
    return CachingCrossEncoder(inner=inner, store=store, model_name=f"nli::{model_name}")


def score_claim_nli(
    sentence: str,
    documents: list[Document],
    support_threshold: float,
    scorer: EntailmentScorerLike,
) -> ClaimGrounding:
    """NLI counterpart to :func:`score_claim`.

    Pairs the (stripped) claim text with each cited chunk and keeps the max
    entailment probability across chunks.
    """
    cited = parse_citations(sentence)
    valid_indices = tuple(i for i in cited if 1 <= i <= len(documents))

    # Strip citations, collapse whitespace, drop trailing sentence-final
    # punctuation so the NLI model sees a clean hypothesis.
    stripped = _WHITESPACE_RE.sub(" ", _strip_citations(sentence)).strip()
    claim_text = stripped.rstrip(_SENTENCE_FINAL_PUNCT).strip()
    if not valid_indices:
        return ClaimGrounding(
            sentence=sentence,
            cited_indices=cited,
            valid_indices=(),
            support_score=0.0,
            is_grounded=False,
        )
    if not claim_text:
        # Pure-citation sentence ("Yes [1].") — give credit, same as overlap.
        return ClaimGrounding(
            sentence=sentence,
            cited_indices=cited,
            valid_indices=valid_indices,
            support_score=1.0,
            is_grounded=True,
        )

    pairs = [(documents[i - 1].page_content, claim_text) for i in valid_indices]
    entail_probs = scorer.predict(pairs)
    support = max(entail_probs) if entail_probs else 0.0
    return ClaimGrounding(
        sentence=sentence,
        cited_indices=cited,
        valid_indices=valid_indices,
        support_score=round(float(support), 3),
        is_grounded=support >= support_threshold,
    )
