"""Tests for the concurrency limiter and the API endpoints that use it.

These are stub-graph tests — we don't spin up Ollama/Qdrant. The point is
to confirm the limiter math and the 429 wiring, not the pipeline itself.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_engine.api import main as api_main
from rag_engine.observability.backpressure import (
    BackpressureError,
    ConcurrencyLimiter,
)


# ---------- unit tests ----------


@pytest.mark.asyncio
async def test_limiter_accepts_up_to_limit():
    limiter = ConcurrencyLimiter("t", limit=2)
    async with limiter.acquire():
        async with limiter.acquire():
            assert limiter.snapshot()["in_flight"] == 2
    assert limiter.snapshot()["in_flight"] == 0
    assert limiter.snapshot()["accepted_total"] == 2
    assert limiter.snapshot()["rejected_total"] == 0


@pytest.mark.asyncio
async def test_limiter_rejects_past_limit():
    limiter = ConcurrencyLimiter("t", limit=1)
    async with limiter.acquire():
        with pytest.raises(BackpressureError) as ei:
            async with limiter.acquire():
                pass
        assert ei.value.in_flight == 1
        assert ei.value.limit == 1
        assert ei.value.name == "t"
    # After the outer slot is released a new acquire succeeds.
    async with limiter.acquire():
        pass
    snap = limiter.snapshot()
    assert snap["in_flight"] == 0
    assert snap["accepted_total"] == 2
    assert snap["rejected_total"] == 1


@pytest.mark.asyncio
async def test_limiter_releases_on_exception_inside_block():
    """A handler that crashes mid-request must not leak a slot."""
    limiter = ConcurrencyLimiter("t", limit=1)
    with pytest.raises(RuntimeError):
        async with limiter.acquire():
            raise RuntimeError("boom")
    assert limiter.snapshot()["in_flight"] == 0
    # Slot is free again.
    async with limiter.acquire():
        pass


@pytest.mark.asyncio
async def test_manual_acquire_release_pair():
    limiter = ConcurrencyLimiter("t", limit=1)
    await limiter.acquire_slot()
    assert limiter.snapshot()["in_flight"] == 1
    with pytest.raises(BackpressureError):
        await limiter.acquire_slot()
    await limiter.release_slot()
    assert limiter.snapshot()["in_flight"] == 0


def test_limiter_rejects_zero_limit():
    with pytest.raises(ValueError):
        ConcurrencyLimiter("t", limit=0)


# ---------- API integration tests ----------


@pytest.fixture
def client():
    return TestClient(api_main.app)


@pytest.fixture(autouse=True)
def _reset_limiters_and_cache(monkeypatch):
    """Each test gets fresh limiters at the configured limits, and the
    answer cache is disabled so /query always goes through the graph.

    Cache clears run *before* the test only — monkeypatch teardown
    restores any patched factories, and the next test's setup wipes
    the lru caches again."""
    api_main._query_limiter.cache_clear()
    api_main._ingest_limiter.cache_clear()
    api_main._get_answer_store.cache_clear()
    api_main._get_graph.cache_clear()
    monkeypatch.setattr(api_main.settings, "answer_cache_enabled", False)
    yield


def test_livez_does_not_touch_dependencies(client):
    # No Qdrant or Ollama probes — this should pass even when neither
    # exists. If it ever starts touching them, the test will hang or
    # 5xx instead.
    resp = client.get("/livez")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_metrics_includes_backpressure_block(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "backpressure" in body
    assert set(body["backpressure"].keys()) == {"query", "ingest"}
    query_block = body["backpressure"]["query"]
    assert query_block["limit"] == api_main.settings.max_concurrent_queries
    assert query_block["in_flight"] == 0


def test_query_returns_429_when_limiter_full(client, monkeypatch):
    """Pretend the limiter is full by pre-seeding its in_flight counter.
    /query should refuse with 429 + Retry-After before touching the graph."""
    limiter = api_main._query_limiter()
    monkeypatch.setattr(limiter, "limit", 1)
    # Pretend one slot is already taken.
    limiter._in_flight = 1
    try:
        resp = client.post("/query", json={"question": "anything"})
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == str(
            api_main.settings.backpressure_retry_after_seconds
        )
        body = resp.json()["detail"]
        assert body["error"] == "backpressure"
        assert body["kind"] == "query"
        assert body["limit"] == 1
    finally:
        limiter._in_flight = 0


def test_ingest_returns_429_when_limiter_full(client, monkeypatch):
    limiter = api_main._ingest_limiter()
    monkeypatch.setattr(limiter, "limit", 1)
    limiter._in_flight = 1
    try:
        resp = client.post("/ingest", json={"source_path": "data/raw"})
        assert resp.status_code == 429
        body = resp.json()["detail"]
        assert body["kind"] == "ingest"
    finally:
        limiter._in_flight = 0


def test_query_releases_slot_after_success(client, monkeypatch):
    """A clean /query must return the slot, so the next one can run."""

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

    monkeypatch.setattr(api_main, "_get_graph", lambda: _FakeGraph())

    limiter = api_main._query_limiter()
    assert limiter.snapshot()["in_flight"] == 0
    resp = client.post("/query", json={"question": "what is up"})
    assert resp.status_code == 200
    assert limiter.snapshot()["in_flight"] == 0
    assert limiter.snapshot()["accepted_total"] >= 1


def test_query_releases_slot_after_graph_error(client, monkeypatch):
    """Even when the graph blows up, the limiter slot must come back."""

    class _ExplodingGraph:
        async def ainvoke(self, _state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("graph blew up")

    monkeypatch.setattr(api_main, "_get_graph", lambda: _ExplodingGraph())

    limiter = api_main._query_limiter()
    with pytest.raises(RuntimeError):
        # TestClient surfaces the exception rather than translating it.
        client.post("/query", json={"question": "what is up"})
    assert limiter.snapshot()["in_flight"] == 0


def test_query_cache_hit_bypasses_limiter(client, monkeypatch):
    """Cache hits shouldn't be counted against the limiter — they don't
    touch the graph and would otherwise rate-limit the dashboard polling."""
    # Re-enable answer cache and seed it manually via a fake store.
    monkeypatch.setattr(api_main.settings, "answer_cache_enabled", True)

    class _FakeStore:
        pass

    fake_cached = {
        "answer": "cached",
        "citations": [],
        "grounding_score": 1.0,
        "warnings": [],
        "iteration": 0,
        "status": "ok",
        "grounded_claim_rate": 1.0,
        "claim_grounding": [],
        "pipeline_trace": [],
        "total_duration_ms": 0.0,
    }
    monkeypatch.setattr(api_main, "_get_answer_store", lambda: _FakeStore())
    monkeypatch.setattr(
        api_main.answer_cache, "get", lambda _store, _q: fake_cached
    )

    limiter = api_main._query_limiter()
    monkeypatch.setattr(limiter, "limit", 1)
    limiter._in_flight = 1  # limiter is "full"
    try:
        resp = client.post("/query", json={"question": "anything"})
        # Should be a 200 cache hit even though the limiter is full.
        assert resp.status_code == 200
        assert resp.json()["cached"] is True
    finally:
        limiter._in_flight = 0


@pytest.mark.asyncio
async def test_limiter_blocks_third_concurrent_caller_directly():
    """Limiter math under real coroutine contention.

    Verified at the limiter level rather than through the API to avoid
    the brittle interplay of httpx + ASGITransport + asyncio cleanup,
    while still covering the case the API actually depends on: while
    two coroutines are inside their slot, a third gets rejected."""
    limiter = ConcurrencyLimiter("t", limit=2)
    release = asyncio.Event()
    inside = asyncio.Event()
    entered = 0

    async def worker():
        nonlocal entered
        async with limiter.acquire():
            entered += 1
            if entered == 2:
                inside.set()
            await release.wait()

    a = asyncio.create_task(worker())
    b = asyncio.create_task(worker())
    await inside.wait()

    snap = limiter.snapshot()
    assert snap["in_flight"] == 2
    with pytest.raises(BackpressureError):
        async with limiter.acquire():
            pass

    release.set()
    await asyncio.gather(a, b)
    snap = limiter.snapshot()
    assert snap["in_flight"] == 0
    assert snap["accepted_total"] == 2
    assert snap["rejected_total"] == 1
