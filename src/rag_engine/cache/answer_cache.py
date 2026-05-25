"""Answer cache keyed on the question text.

Bounded by TTL. We deliberately key on the *original* question — not the
rewritten one or the retrieved set — so two requests for the same question
hit the cache even when retrieval would have picked different documents.
The trade-off: when the corpus changes, cached answers grow stale until the
TTL expires (or the user passes ``bypass_cache=true`` on the request).
"""

from __future__ import annotations

import hashlib
from typing import Any

from rag_engine.cache.store import CacheStore
from rag_engine.observability.counters import counters

_NAMESPACE = "answers"


def _key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def get(store: CacheStore, question: str) -> dict[str, Any] | None:
    value = store.get(_NAMESPACE, _key(question))
    if value is None:
        counters().increment("cache.answers.miss")
        return None
    counters().increment("cache.answers.hit")
    return value


def put(store: CacheStore, question: str, value: dict[str, Any]) -> None:
    store.set(_NAMESPACE, _key(question), value)
