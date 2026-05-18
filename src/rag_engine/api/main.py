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
graph = build_graph(settings)


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
    result = graph.invoke(
        {
            "question": request.question,
            "filters": request.filters or {},
            "documents": [],
            "answer": "",
            "citations": [],
            "grounding_score": 0.0,
            "warnings": [],
        }
    )
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "grounding_score": result["grounding_score"],
        "warnings": result["warnings"],
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

