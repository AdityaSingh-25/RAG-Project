from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine.agents.graph import build_graph
from rag_engine.config.settings import get_settings
from rag_engine.ingestion.pipeline import ingest_path
from rag_engine.observability.tracing import configure_tracing

settings = get_settings()
configure_tracing(settings)

app = FastAPI(title=settings.app_name, version="0.1.0")

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(settings)
    return _graph


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    filters: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    source_path: str = "data/raw"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        graph = _get_graph()
    except Exception as exc:
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
        }
    )
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "grounding_score": result["grounding_score"],
        "warnings": result["warnings"],
        "iteration": result.get("iteration", 0),
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    source = Path(request.source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Source path does not exist: {source}")
    count = ingest_path(source, settings)
    return {"ingested_chunks": count}


def run() -> None:
    uvicorn.run("rag_engine.api.main:app", host="0.0.0.0", port=8000, reload=True)


handler = app
