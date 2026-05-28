from typing import Any, Literal, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
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
    pipeline_trace: list[dict[str, Any]]
    # Optional per-request overrides resolved by _effective_settings().
    # ``None`` means "use the deployment-level Settings value".
    override_claim_verifier_mode: Literal["overlap", "nli"] | None
    override_structured_answers: bool | None

