"""Per-claim grounding verification.

The Phase 3 critic computes a single grounding score for the whole answer.
That's good enough to detect "the model wandered off the corpus" but not
"the third sentence is unsupported." This module fills that gap by:

1. Splitting the answer into sentences (claims).
2. Parsing the ``[n]`` citations on each claim.
3. Scoring each claim against the chunks it cites — term overlap, the same
   cheap heuristic the rest of the engine uses. The structure is what
   matters; an NLI cross-encoder can be swapped in later without changing
   the call sites.

A claim is **grounded** when it has at least one citation pointing at a
real retrieved chunk AND the cited chunk supports it. A "supports" check
here means a non-trivial fraction of the claim's content terms also appear
in the cited chunk.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{3,}")
_STOPWORDS = frozenset(
    {
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
        "into",
        "onto",
        "than",
        "then",
    }
)


@dataclass
class ClaimGrounding:
    """One sentence's worth of grounding evidence."""

    sentence: str
    cited_indices: tuple[int, ...]
    valid_indices: tuple[int, ...]
    support_score: float
    is_grounded: bool


@dataclass
class ClaimReport:
    claims: list[ClaimGrounding] = field(default_factory=list)
    grounded_claim_rate: float = 0.0

    @property
    def ungrounded(self) -> list[ClaimGrounding]:
        return [c for c in self.claims if not c.is_grounded]

    def to_dict(self) -> dict[str, object]:
        return {
            "grounded_claim_rate": self.grounded_claim_rate,
            "claims": [asdict(c) for c in self.claims],
        }


def split_into_sentences(answer: str) -> list[str]:
    """Cheap sentence splitter. Strips empty fragments."""
    parts = _SENTENCE_SPLIT_RE.split(answer.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_citations(sentence: str) -> tuple[int, ...]:
    """Extract ``[n]`` integers in the order they appear."""
    return tuple(int(m.group(1)) for m in _CITATION_RE.finditer(sentence))


def _content_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS
    }


def _strip_citations(sentence: str) -> str:
    return _CITATION_RE.sub("", sentence)


def score_claim(
    sentence: str,
    documents: list[Document],
    support_threshold: float,
) -> ClaimGrounding:
    """Score a single sentence against the documents it cites."""
    cited = parse_citations(sentence)
    valid_indices = tuple(i for i in cited if 1 <= i <= len(documents))

    claim_text = _strip_citations(sentence)
    claim_terms = _content_terms(claim_text)

    if not valid_indices:
        return ClaimGrounding(
            sentence=sentence,
            cited_indices=cited,
            valid_indices=(),
            support_score=0.0,
            is_grounded=False,
        )

    if not claim_terms:
        # A claim with no content terms (e.g., "Yes [1].") gets credit if it
        # at least cites a real chunk — there's nothing to disprove.
        return ClaimGrounding(
            sentence=sentence,
            cited_indices=cited,
            valid_indices=valid_indices,
            support_score=1.0,
            is_grounded=True,
        )

    cited_text = " ".join(documents[i - 1].page_content.lower() for i in valid_indices)
    supported = sum(1 for term in claim_terms if term in cited_text)
    support = supported / len(claim_terms)
    return ClaimGrounding(
        sentence=sentence,
        cited_indices=cited,
        valid_indices=valid_indices,
        support_score=round(support, 3),
        is_grounded=support >= support_threshold,
    )


def verify_claims(
    answer: str,
    documents: list[Document],
    support_threshold: float = 0.2,
) -> ClaimReport:
    """Score every sentence in ``answer`` and report the grounded-claim rate."""
    if not answer.strip():
        return ClaimReport(claims=[], grounded_claim_rate=0.0)

    sentences = split_into_sentences(answer)
    if not sentences:
        return ClaimReport(claims=[], grounded_claim_rate=0.0)

    claims = [score_claim(s, documents, support_threshold) for s in sentences]
    grounded = sum(1 for c in claims if c.is_grounded)
    rate = grounded / len(claims)
    return ClaimReport(claims=claims, grounded_claim_rate=round(rate, 3))


def verify_claims_with_settings(
    answer: str,
    documents: list[Document],
    settings,
) -> ClaimReport:
    """Settings-aware dispatcher that picks the overlap or NLI scorer.

    Imported lazily so ``claim_verifier_mode = "overlap"`` deployments never
    touch the sentence-transformers / NLI model machinery.
    """
    if not answer.strip():
        return ClaimReport(claims=[], grounded_claim_rate=0.0)
    sentences = split_into_sentences(answer)
    if not sentences:
        return ClaimReport(claims=[], grounded_claim_rate=0.0)

    threshold = settings.claim_support_threshold
    if settings.claim_verifier_mode == "nli":
        from rag_engine.evaluation.nli_verifier import (
            load_nli_with_cache,
            score_claim_nli,
        )

        scorer = load_nli_with_cache(
            model_name=settings.nli_model,
            cache_enabled=settings.cache_enabled,
            cache_path=settings.cache_path,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        )
        claims = [score_claim_nli(s, documents, threshold, scorer) for s in sentences]
    else:
        claims = [score_claim(s, documents, threshold) for s in sentences]

    grounded = sum(1 for c in claims if c.is_grounded)
    rate = grounded / len(claims)
    return ClaimReport(claims=claims, grounded_claim_rate=round(rate, 3))
