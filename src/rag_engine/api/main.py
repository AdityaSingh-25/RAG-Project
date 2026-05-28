import json
import logging
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_engine.agents.graph import ANSWER_TOKEN_CHANNEL, build_graph
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Module-level singletons. ``lru_cache`` gives us a thread-safe first-call
# initializer — concurrent first requests no longer race into building two
# graphs (wasteful but not incorrect) or two CacheStore instances pointing
# at the same SQLite file.
@lru_cache(maxsize=1)
def _get_graph():
    return build_graph(settings)


@lru_cache(maxsize=1)
def _get_answer_store() -> CacheStore | None:
    if not (settings.cache_enabled and settings.answer_cache_enabled):
        return None
    return CacheStore(Path(settings.cache_path), settings.cache_ttl_seconds)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    filters: dict[str, Any] | None = None
    bypass_cache: bool = False
    # Optional per-request overrides of the corresponding Settings fields.
    # ``None`` means "use the deployment default" — keeps the API surface
    # unchanged for callers that don't care.
    claim_verifier_mode: Literal["overlap", "nli"] | None = None
    structured_answers: bool | None = None


class IngestRequest(BaseModel):
    source_path: str = "data/raw"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return counters().snapshot()


@app.post("/query")
async def query(request: QueryRequest) -> dict[str, Any]:
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

    result = await graph.ainvoke(_initial_state(request, trace_id))
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
        "pipeline_trace": result.get("pipeline_trace", []),
        "total_duration_ms": round(duration, 2),
    }
    if store is not None and payload["status"] == "ok":
        answer_cache.put(store, request.question, payload)
    return {**payload, "trace_id": trace_id, "cached": False}


def _sse(event: str, data: dict) -> str:
    """Format one SSE frame. Frames are separated by a blank line per spec."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _initial_state(request: QueryRequest, trace_id: str) -> dict[str, Any]:
    return {
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
        "pipeline_trace": [],
        # Per-request overrides — None means "use settings default". The
        # graph nodes resolve these via _effective_settings.
        "override_claim_verifier_mode": request.claim_verifier_mode,
        "override_structured_answers": request.structured_answers,
    }


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """SSE variant of /query.

    Emits these event types:
      - trace      : one per pipeline node completion
      - token      : per-chunk answer text
      - citations  : full citation list, once after the answer is generated
      - grounding  : per-claim grounding + warnings
      - done       : final payload identical to /query's response shape
      - error      : on backend failure
    """
    trace_id = uuid.uuid4().hex
    counters().increment("api.query.total")
    counters().increment("api.query_stream.total")

    store = _get_answer_store()
    if store is not None and not request.bypass_cache:
        cached = answer_cache.get(store, request.question)
        if cached is not None:
            log_event(
                _logger,
                "api.query.cache_hit",
                trace_id=trace_id,
                question_chars=len(request.question),
                streaming=True,
            )
            cached_payload = {**cached, "trace_id": trace_id, "cached": True}

            async def cached_gen():
                yield _sse("done", cached_payload)

            return StreamingResponse(cached_gen(), media_type="text/event-stream")

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

    initial = _initial_state(request, trace_id)

    async def gen():
        started = now_ms()
        final_state: dict | None = None
        try:
            async for mode, chunk in graph.astream(
                initial, stream_mode=["custom", "updates", "values"]
            ):
                if mode == "custom":
                    if isinstance(chunk, dict) and ANSWER_TOKEN_CHANNEL in chunk:
                        yield _sse("token", {"delta": chunk[ANSWER_TOKEN_CHANNEL]})
                elif mode == "updates":
                    for _node, diff in (chunk or {}).items():
                        trace = (diff or {}).get("pipeline_trace") or []
                        if trace:
                            yield _sse("trace", trace[-1])
                elif mode == "values":
                    final_state = chunk
        except Exception as exc:
            log_event(
                _logger,
                "api.query.stream_error",
                trace_id=trace_id,
                error=str(exc),
                level=logging.ERROR,
            )
            yield _sse("error", {"detail": str(exc)})
            return

        if final_state is None:
            yield _sse("error", {"detail": "graph produced no final state"})
            return

        duration = now_ms() - started
        counters().observe("api.query.latency_ms", duration)
        status = final_state.get("status") or "ok"
        counters().increment(f"api.query.status.{status}")

        yield _sse("citations", {"citations": final_state.get("citations", [])})
        yield _sse(
            "grounding",
            {
                "grounding_score": final_state.get("grounding_score", 0.0),
                "grounded_claim_rate": final_state.get("grounded_claim_rate", 1.0),
                "claim_grounding": final_state.get("claim_grounding", []),
                "warnings": final_state.get("warnings", []),
            },
        )

        payload = {
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "grounding_score": final_state.get("grounding_score", 0.0),
            "warnings": final_state.get("warnings", []),
            "iteration": final_state.get("iteration", 0),
            "status": status,
            "grounded_claim_rate": final_state.get("grounded_claim_rate", 1.0),
            "claim_grounding": final_state.get("claim_grounding", []),
            "pipeline_trace": final_state.get("pipeline_trace", []),
            "total_duration_ms": round(duration, 2),
        }
        if store is not None and status == "ok":
            answer_cache.put(store, request.question, payload)

        log_event(
            _logger,
            "api.query.complete",
            trace_id=trace_id,
            duration_ms=round(duration, 2),
            status=status,
            iteration=final_state.get("iteration", 0),
            grounding_score=final_state.get("grounding_score", 0.0),
            streaming=True,
        )
        yield _sse("done", {**payload, "trace_id": trace_id, "cached": False})

    return StreamingResponse(gen(), media_type="text/event-stream")


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


# Mirrors the suffixes accepted by `_load_file` in ingestion/loaders.py. Kept in
# the API layer so we can reject unsupported uploads before writing anything to
# disk — the loader would silently produce zero documents otherwise.
_ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".csv", ".json", ".txt", ".md", ".rst"}


@app.post("/ingest/upload")
async def ingest_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    trace_id = uuid.uuid4().hex
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    for upload in files:
        name = upload.filename or ""
        if not name:
            raise HTTPException(status_code=400, detail="Uploaded file is missing a name")
        suffix = Path(name).suffix.lower()
        if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type {suffix or '(none)'} for {name!r}. "
                    f"Allowed: {sorted(_ALLOWED_UPLOAD_SUFFIXES)}"
                ),
            )

    with TemporaryDirectory(prefix="rag-upload-") as tmp:
        tmp_path = Path(tmp)
        for upload in files:
            # Strip any path components from the client — only keep the basename
            # so a malicious filename like "../etc/passwd" can't escape the
            # temp dir. `Path(...).name` returns only the final component.
            safe_name = Path(upload.filename or "upload").name
            target = tmp_path / safe_name
            with target.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            await upload.close()
        try:
            report = ingest_path(tmp_path, settings)
        except Exception as exc:
            log_event(
                _logger,
                "api.ingest.upload.failed",
                trace_id=trace_id,
                error=str(exc),
                file_count=len(files),
                level=logging.ERROR,
            )
            raise HTTPException(status_code=503, detail=f"Backend not ready: {exc}") from exc

    counters().increment("api.ingest.upload.total")
    counters().increment("api.ingest.upload.files", amount=len(files))
    return {
        "ingested_chunks": report.indexed,
        "duplicates_removed": report.duplicates_removed,
        "files_received": len(files),
    }


def run() -> None:
    uvicorn.run("rag_engine.api.main:app", host="0.0.0.0", port=8000, reload=True)


handler = app
