"""Tests for the API-key auth + token-bucket rate limiter and the API
endpoints that use them.

We don't spin up Qdrant or Ollama — the graph is stubbed for the
integration checks. The point is to confirm the auth gate, 401/429
shapes, and that headers (Retry-After, WWW-Authenticate) survive."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_engine.api import main as api_main
from rag_engine.api.auth import (
    ANONYMOUS_IDENTITY,
    ApiKeyRegistry,
    RateLimitExceeded,
    TokenBucket,
    parse_keys_csv,
)


# ---------- unit tests ----------


def test_parse_keys_csv_handles_whitespace_and_empties():
    assert parse_keys_csv(" a , b ,, c, ") == ["a", "b", "c"]
    assert parse_keys_csv("") == []
    assert parse_keys_csv(",,") == []


def test_token_bucket_starts_full():
    b = TokenBucket(rate_per_minute=60, burst=5)
    for _ in range(5):
        ok, wait = b.try_consume(1)
        assert ok and wait == 0.0


def test_token_bucket_blocks_when_empty_then_refills():
    b = TokenBucket(rate_per_minute=60, burst=2)
    assert b.try_consume(1)[0]
    assert b.try_consume(1)[0]
    ok, wait = b.try_consume(1)
    assert not ok
    # 60/min = 1/s, so we need ~1s for the next token.
    assert 0.5 < wait <= 1.0
    # Force the bucket clock forward and confirm it refills.
    b.last_refill -= 2.0  # pretend 2s have passed
    ok, wait = b.try_consume(1)
    assert ok and wait == 0.0


def test_token_bucket_rejects_bad_config():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_minute=0, burst=10)
    with pytest.raises(ValueError):
        TokenBucket(rate_per_minute=60, burst=0)


def test_registry_disabled_when_no_keys():
    r = ApiKeyRegistry([], rate_per_minute=60, burst=10)
    assert r.enabled is False
    # Even with no key supplied, auth-disabled mode passes through.
    assert r.authenticate(None) == ANONYMOUS_IDENTITY
    assert r.authenticate("anything") == ANONYMOUS_IDENTITY


def test_registry_authenticate_matches_known_keys():
    r = ApiKeyRegistry(["alpha", "beta"], rate_per_minute=60, burst=10)
    ident_a = r.authenticate("alpha")
    ident_b = r.authenticate("beta")
    # Each known key gets a distinct stable identity (its sha256).
    assert ident_a != ident_b
    assert len(ident_a) == 64
    # Repeat calls are stable.
    assert r.authenticate("alpha") == ident_a


def test_registry_rejects_missing_or_wrong_key():
    r = ApiKeyRegistry(["alpha"], rate_per_minute=60, burst=10)
    with pytest.raises(PermissionError):
        r.authenticate(None)
    with pytest.raises(PermissionError):
        r.authenticate("")
    with pytest.raises(PermissionError):
        r.authenticate("not-the-right-key")
    snap = r.snapshot()
    assert snap["unauthenticated_total"] == 3


def test_registry_consume_then_exhaust_then_refill():
    r = ApiKeyRegistry(["a"], rate_per_minute=60, burst=2)
    ident = r.authenticate("a")
    r.consume(ident)
    r.consume(ident)
    with pytest.raises(RateLimitExceeded) as ei:
        r.consume(ident)
    assert ei.value.limit_per_minute == 60
    assert ei.value.wait_seconds > 0
    # Per-identity isolation: a second key has its own bucket.
    r2 = ApiKeyRegistry(["a", "b"], rate_per_minute=60, burst=1)
    ident_a = r2.authenticate("a")
    ident_b = r2.authenticate("b")
    r2.consume(ident_a)
    # b is still full — shouldn't be punished for a's burst.
    r2.consume(ident_b)


def test_registry_snapshot_never_leaks_plaintext():
    r = ApiKeyRegistry(["sekret"], rate_per_minute=60, burst=10)
    snap = r.snapshot()
    assert "sekret" not in str(snap)
    assert snap == {
        "enabled": True,
        "keys_configured": 1,
        "rate_per_minute": 60,
        "burst": 10,
        "accepted_total": 0,
        "rejected_total": 0,
        "unauthenticated_total": 0,
        "distinct_identities": 0,
    }


# ---------- API integration tests ----------


@pytest.fixture
def client():
    return TestClient(api_main.app)


@pytest.fixture(autouse=True)
def _fresh_singletons():
    """Each test gets a fresh registry + a stubbed answer store. The
    counters singleton accumulates across the file — tests that care
    about counts query before/after rather than expecting zero."""
    api_main._get_auth_registry.cache_clear()
    api_main._get_answer_store.cache_clear()
    api_main._get_graph.cache_clear()
    yield
    api_main._get_auth_registry.cache_clear()
    api_main._get_answer_store.cache_clear()
    api_main._get_graph.cache_clear()


def test_health_metrics_livez_unrestricted(client, monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_keys_csv", "test-key-1,test-key-2")
    api_main._get_auth_registry.cache_clear()
    # Hit each public endpoint without any header — must all be 200.
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_metrics_includes_auth_block(client, monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_keys_csv", "one,two")
    api_main._get_auth_registry.cache_clear()
    body = client.get("/metrics").json()
    assert "auth" in body
    assert body["auth"]["enabled"] is True
    assert body["auth"]["keys_configured"] == 2


def test_query_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_keys_csv", "abc123")
    monkeypatch.setattr(api_main.settings, "answer_cache_enabled", False)
    api_main._get_auth_registry.cache_clear()
    api_main._get_answer_store.cache_clear()

    # No header → 401 with WWW-Authenticate, no graph invoked.
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "ApiKey"
    body = resp.json()["detail"]
    assert body["error"] == "unauthenticated"

    # Wrong header → 401.
    resp = client.post(
        "/query",
        json={"question": "anything"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_query_accepts_with_correct_key(client, monkeypatch):
    class _FakeGraph:
        async def ainvoke(self, _state: dict[str, Any]) -> dict[str, Any]:
            return {
                "answer": "hi",
                "citations": [],
                "grounding_score": 1.0,
                "warnings": [],
                "iteration": 0,
                "status": "ok",
                "grounded_claim_rate": 1.0,
                "claim_grounding": [],
                "pipeline_trace": [],
            }

    monkeypatch.setattr(api_main.settings, "api_keys_csv", "abc123")
    monkeypatch.setattr(api_main.settings, "answer_cache_enabled", False)
    api_main._get_auth_registry.cache_clear()
    api_main._get_answer_store.cache_clear()
    monkeypatch.setattr(api_main, "_get_graph", lambda: _FakeGraph())

    resp = client.post(
        "/query",
        json={"question": "anything"},
        headers={"X-API-Key": "abc123"},
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "hi"


def test_query_rate_limited_when_bucket_empty(client, monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_keys_csv", "abc123")
    monkeypatch.setattr(api_main.settings, "rate_limit_per_minute", 60)
    monkeypatch.setattr(api_main.settings, "rate_limit_burst", 1)
    monkeypatch.setattr(api_main.settings, "answer_cache_enabled", False)
    api_main._get_auth_registry.cache_clear()
    api_main._get_answer_store.cache_clear()

    class _FakeGraph:
        async def ainvoke(self, _state: dict[str, Any]) -> dict[str, Any]:
            return {
                "answer": "x",
                "citations": [],
                "grounding_score": 1.0,
                "warnings": [],
                "iteration": 0,
                "status": "ok",
                "grounded_claim_rate": 1.0,
                "claim_grounding": [],
                "pipeline_trace": [],
            }

    monkeypatch.setattr(api_main, "_get_graph", lambda: _FakeGraph())

    headers = {"X-API-Key": "abc123"}
    # First request consumes the only token in the bucket.
    assert (
        client.post("/query", json={"question": "first"}, headers=headers).status_code
        == 200
    )
    # Second request: bucket empty → 429.
    resp = client.post("/query", json={"question": "second"}, headers=headers)
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1
    body = resp.json()["detail"]
    assert body["error"] == "rate_limited"
    assert body["limit_per_minute"] == 60


def test_corpus_endpoints_require_key(client, monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_keys_csv", "abc123")
    api_main._get_auth_registry.cache_clear()
    # No header → 401, *before* Qdrant is touched.
    assert client.get("/corpus/stats").status_code == 401
    assert client.get("/corpus/sources").status_code == 401
    assert client.get("/corpus/source?path=anything").status_code == 401


def test_auth_disabled_still_rate_limits_anonymous(client, monkeypatch):
    """In dev mode (no keys configured) all requests use the same
    anonymous bucket. That stops a buggy local script from hammering
    the loop into the ground."""
    monkeypatch.setattr(api_main.settings, "api_keys_csv", "")
    monkeypatch.setattr(api_main.settings, "rate_limit_per_minute", 60)
    monkeypatch.setattr(api_main.settings, "rate_limit_burst", 1)
    monkeypatch.setattr(api_main.settings, "answer_cache_enabled", False)
    api_main._get_auth_registry.cache_clear()
    api_main._get_answer_store.cache_clear()

    class _FakeGraph:
        async def ainvoke(self, _state: dict[str, Any]) -> dict[str, Any]:
            return {
                "answer": "x",
                "citations": [],
                "grounding_score": 1.0,
                "warnings": [],
                "iteration": 0,
                "status": "ok",
                "grounded_claim_rate": 1.0,
                "claim_grounding": [],
                "pipeline_trace": [],
            }

    monkeypatch.setattr(api_main, "_get_graph", lambda: _FakeGraph())

    # First call (no header, auth disabled) lands.
    assert (
        client.post("/query", json={"question": "first"}).status_code == 200
    )
    # Second call drains the anonymous bucket of size 1 → 429.
    resp = client.post("/query", json={"question": "second"})
    assert resp.status_code == 429
    assert resp.json()["detail"]["error"] == "rate_limited"
