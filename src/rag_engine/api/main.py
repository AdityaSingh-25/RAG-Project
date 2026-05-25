import logging
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine.agents.graph import build_graph
from rag_engine.cache import answer_cache
from rag_engine.cache.store import CacheStore
from rag_engine.config.settings import get_settings
from rag_engine.ingestion.pipeline import ingest_path
from rag_engine.observability.counters import counters
from rag_engine.observability.logging import configure_logging, get_logger, log_event, now_ms
from rag_engine.observability.tracing import configure_tracing

settings = get_settings()
configure_logging(log_format=settings.log_format, level=settings.log_level)
configure_tracing(settings)

_logger = get_logger("rag_engine.api")

app = FastAPI(title=settings.app_name, version="0.1.0")

_graph = None
_answer_store: CacheStore | None = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(settings)
    return _graph


def _get_answer_store() -> CacheStore | None:
    global _answer_store
    if not (settings.cache_enabled and settings.answer_cache_enabled):
        return None
    if _answer_store is None:
        _answer_store = CacheStore(Path(settings.cache_path), settings.cache_ttl_seconds)
    return _answer_store


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    filters: dict[str, Any] | None = None
    bypass_cache: bool = False


class IngestRequest(BaseModel):
    source_path: str = "data/raw"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return counters().snapshot()


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    trace_id = uuid.uuid4().hex
    counters().increment("api.query.total")
    store = _get_answer_store()
    if store is not None and not request.bypass_cache:
        cached = answer_cache.get(store, request.question)
        if cached is not None:
            log_event(
                _logger,
                "api.query.cache_hit",
                trace_id=trace_id,
                question_chars=len(request.question),
            )
            return {**cached, "trace_id": trace_id, "cached": True}

    started = now_ms()
    try:
        graph = _get_graph()
    except Exception as exc:
        log_event(
            _logger,
            "api.query.backend_unready",
            trace_id=trace_id,
            error=str(exc),
            level=logging.ERROR,
        )
        raise HTTPException(status_code=503, detail=f"Backend not ready: {exc}") from exc

    result = graph.invoke(
        {
            "question": request.question,
            "original_question": request.question,
            "filters": request.filters or {},
            "documents": [],
            "answer": "",
            "citations": [],
            "grounding_score": 0.0,
            "warnings": [],
            "iteration": 0,
            "status": "",
            "trace_id": trace_id,
            "grounded_claim_rate": 1.0,
            "claim_grounding": [],
        }
    )
    duration = now_ms() - started
    counters().observe("api.query.latency_ms", duration)
    counters().increment(f"api.query.status.{result.get('status') or 'ok'}")
    log_event(
        _logger,
        "api.query.complete",
        trace_id=trace_id,
        duration_ms=round(duration, 2),
        status=result.get("status") or "ok",
        iteration=result.get("iteration", 0),
        grounding_score=result.get("grounding_score", 0.0),
    )

    payload = {
        "answer": result["answer"],
        "citations": result["citations"],
        "grounding_score": result["grounding_score"],
        "warnings": result["warnings"],
        "iteration": result.get("iteration", 0),
        "status": result.get("status") or "ok",
        "grounded_claim_rate": result.get("grounded_claim_rate", 1.0),
        "claim_grounding": result.get("claim_grounding", []),
    }
    if store is not None and payload["status"] == "ok":
        answer_cache.put(store, request.question, payload)
    return {**payload, "trace_id": trace_id, "cached": False}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    source = Path(request.source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Source path does not exist: {source}")
    report = ingest_path(source, settings)
    return {
        "ingested_chunks": report.indexed,
        "duplicates_removed": report.duplicates_removed,
    }


def run() -> None:
    uvicorn.run("rag_engine.api.main:app", host="0.0.0.0", port=8000, reload=True)


handler = app
