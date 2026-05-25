from pathlib import Path

from rag_engine.cache import answer_cache
from rag_engine.cache.embedding_cache import CachingEmbeddings
from rag_engine.cache.reranker_cache import CachingCrossEncoder
from rag_engine.cache.store import CacheStore
from rag_engine.observability.counters import counters


def _store(tmp_path: Path) -> CacheStore:
    return CacheStore(path=tmp_path / "cache.sqlite", default_ttl_seconds=3600)


class _StubInner:
    def __init__(self) -> None:
        self.query_calls = 0
        self.doc_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [float(len(text))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls.append(list(texts))
        return [[float(len(t))] for t in texts]


def test_caching_embeddings_serves_hits_without_invoking_inner(tmp_path: Path) -> None:
    inner = _StubInner()
    cache = CachingEmbeddings(inner=inner, store=_store(tmp_path), model_name="m")
    assert cache.embed_query("hello") == [5.0]
    assert cache.embed_query("hello") == [5.0]
    assert inner.query_calls == 1


def test_caching_embeddings_mixed_hits_and_misses_in_batch(tmp_path: Path) -> None:
    inner = _StubInner()
    cache = CachingEmbeddings(inner=inner, store=_store(tmp_path), model_name="m")
    cache.embed_query("alpha")  # primes the cache for "alpha"

    out = cache.embed_documents(["alpha", "beta", "gamma"])

    assert out == [[5.0], [4.0], [5.0]]
    # Only the misses ("beta", "gamma") should go to the underlying model.
    assert inner.doc_calls == [["beta", "gamma"]]


class _StubCrossEncoder:
    def __init__(self) -> None:
        self.batches: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.batches.append(list(pairs))
        return [float(len(passage)) for _, passage in pairs]


def test_caching_cross_encoder_skips_repeated_pairs(tmp_path: Path) -> None:
    inner = _StubCrossEncoder()
    cache = CachingCrossEncoder(inner=inner, store=_store(tmp_path), model_name="m")

    first = cache.predict([("Q", "alpha"), ("Q", "beta")])
    second = cache.predict([("Q", "alpha"), ("Q", "gamma")])

    assert first == [5.0, 4.0]
    assert second == [5.0, 5.0]
    # Second call should only forward "gamma" — "alpha" is cached.
    assert inner.batches == [[("Q", "alpha"), ("Q", "beta")], [("Q", "gamma")]]


def test_answer_cache_round_trip_and_normalizes_question(tmp_path: Path) -> None:
    counters().reset()
    store = _store(tmp_path)
    assert answer_cache.get(store, "What is Qdrant?") is None
    answer_cache.put(store, "What is Qdrant?", {"answer": "vector db", "status": "ok"})
    # Same question with different surrounding whitespace and case should hit.
    assert answer_cache.get(store, "  WHAT IS QDRANT?  ") == {
        "answer": "vector db",
        "status": "ok",
    }
    snapshot = counters().snapshot()
    assert snapshot["totals"]["cache.answers.miss"] == 1
    assert snapshot["totals"]["cache.answers.hit"] == 1
