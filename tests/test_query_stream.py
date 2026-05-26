"""Tests for the SSE /query/stream endpoint.

The graph is stubbed out — we're testing the SSE bridge in api/main.py, not
the real retrieve→answer pipeline (which would need Ollama and Qdrant).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_engine.api import main as api_main


@pytest.fixture
def client():
    return TestClient(api_main.app)


class _FakeGraph:
    """Yields a fixed sequence of (mode, chunk) tuples like CompiledStateGraph.astream."""

    def __init__(self, events: list[tuple[str, Any]]):
        self._events = events

    async def astream(self, _initial_state, stream_mode=None):
        for mode, chunk in self._events:
            yield mode, chunk


class _ExplodingGraph:
    async def astream(self, _initial_state, stream_mode=None):
        if False:  # pragma: no cover — yield to make this an async generator
            yield None
        raise RuntimeError("graph blew up")


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event: str | None = None
        data: dict | None = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None and data is not None:
            out.append((event, data))
    return out


def _final_state(answer: str = "Hello world.") -> dict:
    return {
        "answer": answer,
        "citations": [{"id": 1, "source": "doc-a", "page": 1, "score": 0.9, "content": "..."}],
        "grounding_score": 0.91,
        "warnings": [],
        "iteration": 0,
        "status": "ok",
        "grounded_claim_rate": 1.0,
        "claim_grounding": [
            {"sentence": answer, "cited_indices": [1], "valid_indices": [1], "support_score": 0.9, "is_grounded": True}
        ],
        "pipeline_trace": [
            {"node": "retrieve", "duration_ms": 1.0, "iteration": 0},
            {"node": "answer", "duration_ms": 2.0, "iteration": 0},
            {"node": "critique", "duration_ms": 0.5, "iteration": 0},
            {"node": "finalize", "duration_ms": 0.0, "iteration": 0},
        ],
    }


def test_stream_emits_trace_tokens_and_done(client, monkeypatch):
    events = [
        ("updates", {"retrieve": {"pipeline_trace": [{"node": "retrieve", "duration_ms": 1.0}]}}),
        ("custom", {"answer.token": "Hello"}),
        ("custom", {"answer.token": " world."}),
        ("updates", {"answer": {"pipeline_trace": [{"node": "retrieve"}, {"node": "answer", "duration_ms": 2.0}]}}),
        ("values", _final_state()),
    ]
    monkeypatch.setattr(api_main, "_get_graph", lambda: _FakeGraph(events))
    monkeypatch.setattr(api_main, "_get_answer_store", lambda: None)

    with client.stream("POST", "/query/stream", json={"question": "Hi there?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    parsed = _parse_sse(body)
    kinds = [k for k, _ in parsed]

    assert kinds.count("trace") >= 2
    assert kinds.count("token") == 2
    assert kinds.count("citations") == 1
    assert kinds.count("grounding") == 1
    assert kinds.count("done") == 1
    # Final `done` event closes the stream — anything after would be a bug.
    assert kinds[-1] == "done"

    tokens = [d["delta"] for k, d in parsed if k == "token"]
    assert "".join(tokens) == "Hello world."

    done = next(d for k, d in parsed if k == "done")
    assert done["status"] == "ok"
    assert done["answer"] == "Hello world."
    assert done["cached"] is False
    assert done["trace_id"]


def test_stream_uses_cache_when_available(client, monkeypatch):
    cached_payload = {
        "answer": "from cache",
        "citations": [],
        "grounding_score": 0.95,
        "warnings": [],
        "iteration": 0,
        "status": "ok",
        "grounded_claim_rate": 1.0,
        "claim_grounding": [],
        "pipeline_trace": [],
        "total_duration_ms": 0.1,
    }

    class _FakeStore:
        pass

    monkeypatch.setattr(api_main, "_get_answer_store", lambda: _FakeStore())
    monkeypatch.setattr(api_main.answer_cache, "get", lambda _store, _q: cached_payload)
    # Graph must not be touched on a cache hit.
    monkeypatch.setattr(
        api_main,
        "_get_graph",
        lambda: (_ for _ in ()).throw(AssertionError("graph should not be built on cache hit")),
    )

    with client.stream("POST", "/query/stream", json={"question": "anything"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    parsed = _parse_sse(body)
    assert [k for k, _ in parsed] == ["done"]
    done = parsed[0][1]
    assert done["answer"] == "from cache"
    assert done["cached"] is True
    assert done["trace_id"]


def test_stream_emits_error_event_on_graph_failure(client, monkeypatch):
    monkeypatch.setattr(api_main, "_get_graph", lambda: _ExplodingGraph())
    monkeypatch.setattr(api_main, "_get_answer_store", lambda: None)

    with client.stream("POST", "/query/stream", json={"question": "boom"}) as r:
        assert r.status_code == 200  # body carries the error event
        body = "".join(r.iter_text())

    parsed = _parse_sse(body)
    kinds = [k for k, _ in parsed]
    assert "error" in kinds
    err = next(d for k, d in parsed if k == "error")
    assert "graph blew up" in err["detail"]


def test_stream_writes_to_answer_cache_on_success(client, monkeypatch):
    final = _final_state(answer="cache me")
    events = [
        ("custom", {"answer.token": "cache me"}),
        ("values", final),
    ]
    captured: dict[str, Any] = {}

    class _FakeStore:
        pass

    monkeypatch.setattr(api_main, "_get_graph", lambda: _FakeGraph(events))
    monkeypatch.setattr(api_main, "_get_answer_store", lambda: _FakeStore())
    monkeypatch.setattr(api_main.answer_cache, "get", lambda _store, _q: None)

    def fake_put(_store, question, payload):
        captured["question"] = question
        captured["payload"] = payload

    monkeypatch.setattr(api_main.answer_cache, "put", fake_put)

    with client.stream("POST", "/query/stream", json={"question": "Cache this?"}) as r:
        assert r.status_code == 200
        list(r.iter_text())  # drain

    assert captured["question"] == "Cache this?"
    assert captured["payload"]["answer"] == "cache me"
    assert captured["payload"]["status"] == "ok"
