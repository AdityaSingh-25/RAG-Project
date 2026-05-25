from typing import Any, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    original_question: str
    filters: dict[str, Any]
    documents: list[Document]
    answer: str
    citations: list[dict[str, Any]]
    grounding_score: float
    warnings: list[str]
    iteration: int
    status: str
    trace_id: str
    grounded_claim_rate: float
    claim_grounding: list[dict[str, Any]]

