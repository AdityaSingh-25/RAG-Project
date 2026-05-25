"""Per-document confidence signal mixed into the reranker score.

Three multiplicative components, each in roughly ``[0.5, 1.5]``:

- **freshness** — half-life decay from ``metadata['published_at']`` (ISO
  date) or ``metadata['mtime']`` (POSIX timestamp). Unknown dates default
  to ``1.0`` rather than penalising — we don't want to silently bury
  hand-curated docs with no date.
- **trust** — a glob-keyed weights map (``settings.source_weights``)
  matched against ``metadata['source']``. Useful for promoting curated
  docs (``docs/**`` -> 1.2) or de-emphasising scratch notes
  (``data/raw/notes/**`` -> 0.7).
- **agreement** — 1.0 when a document only appeared in one retrieval
  list; ``settings.agreement_boost`` (default 1.2) when it appeared in
  BOTH dense and BM25 results. The HybridRetriever sets
  ``metadata['agreement_count']`` during RRF.

The final confidence is the product, so any one component pulling
hard (high freshness, high trust, both retrievers agree) compounds with
the others rather than averaging out.
"""

from __future__ import annotations

import datetime
import fnmatch
import math

from langchain_core.documents import Document

from rag_engine.config.settings import Settings


def _to_timestamp(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


def freshness_score(doc: Document, half_life_days: int, now: float | None = None) -> float:
    """Exponential decay anchored on ``published_at`` or ``mtime``.

    A doc as old as ``half_life_days`` scores 0.5; twice as old scores 0.25.
    Missing or unparseable dates score 1.0 (neutral).
    """
    ts = _to_timestamp(doc.metadata.get("published_at")) or _to_timestamp(
        doc.metadata.get("mtime")
    )
    if ts is None:
        return 1.0
    now_ts = now if now is not None else datetime.datetime.now(datetime.timezone.utc).timestamp()
    age_seconds = max(now_ts - ts, 0.0)
    age_days = age_seconds / 86_400.0
    return float(0.5 ** (age_days / max(half_life_days, 1)))


def trust_score(doc: Document, weights: dict[str, float]) -> float:
    """First matching glob in ``weights`` wins. Default 1.0 (neutral)."""
    source = str(doc.metadata.get("source", ""))
    if not source or not weights:
        return 1.0
    for pattern, weight in weights.items():
        if fnmatch.fnmatch(source, pattern):
            return float(weight)
    return 1.0


def agreement_score(doc: Document, boost: float) -> float:
    """Boost docs that appeared in more than one retrieval list."""
    count = int(doc.metadata.get("agreement_count", 1) or 1)
    return float(boost) if count >= 2 else 1.0


def score_source_confidence(doc: Document, settings: Settings) -> float:
    """Multiplicative product of the three component scores."""
    if not settings.enable_source_confidence:
        return 1.0
    f = freshness_score(doc, settings.freshness_half_life_days)
    t = trust_score(doc, settings.source_weights)
    a = agreement_score(doc, settings.agreement_boost)
    confidence = f * t * a
    if math.isnan(confidence) or confidence < 0.0:
        return 0.0
    return confidence
