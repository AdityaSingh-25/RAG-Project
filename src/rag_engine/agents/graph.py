from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from rag_engine.agents.state import AgentState
from rag_engine.config.settings import Settings
from rag_engine.evaluation.hallucination import verify_answer_confidence
from rag_engine.retrieval.query_rewriter import rewrite_query
from rag_engine.retrieval.retriever import build_retriever
from rag_engine.retrieval.reranker import rerank_documents
from rag_engine.routing.model_router import ModelRouter
from rag_engine.utils.tokenization import count_tokens


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence to answer confidently from the retrieved context."
)


def route_after_critique(state: AgentState, settings: Settings) -> str:
    """Decide what happens after the critic runs.

    Returns the next node name: ``"rewrite"`` to loop, ``"fallback"`` to refuse,
    or ``END`` when the answer is good enough to ship.
    """
    score = state.get("grounding_score", 0.0)
    iteration = state.get("iteration", 0)
    threshold = settings.grounding_threshold

    if score >= threshold:
        return END

    can_retry = (
        settings.enable_feedback_loop
        and iteration < settings.max_retry_iterations
    )
    if can_retry:
        return "rewrite"
    return "fallback"


def build_graph(settings: Settings):
    retriever = build_retriever(settings)
    router = ModelRouter(settings)

    def retrieve(state: AgentState) -> AgentState:
        documents = retriever.invoke(state["question"])
        reranked = rerank_documents(state["question"], documents, top_k=settings.top_k)
        return {**state, "documents": reranked}

    def answer(state: AgentState) -> AgentState:
        sorted_documents = sorted(
            state["documents"],
            key=lambda doc: doc.metadata.get("score", 0),
            reverse=True,
        )
        context = _format_context(sorted_documents, settings)
        llm = router.for_question(state["question"], context)
        response = llm.invoke(_answer_prompt(state["question"], context))
        answer_text = getattr(response, "content", str(response))
        citations = [_citation(doc, index) for index, doc in enumerate(sorted_documents, start=1)]
        return {**state, "answer": answer_text, "citations": citations}

    def critique(state: AgentState) -> AgentState:
        confidence = verify_answer_confidence(state["answer"], state["documents"])
        score = confidence["grounding_score"]
        warnings = confidence["warnings"]
        return {**state, "grounding_score": score, "warnings": warnings}

    def rewrite(state: AgentState) -> AgentState:
        original = state.get("original_question") or state["question"]
        rewritten = rewrite_query(original, state.get("documents", []))
        return {
            **state,
            "question": rewritten,
            "iteration": state.get("iteration", 0) + 1,
        }

    def fallback(state: AgentState) -> AgentState:
        reason = (
            "no_retrieved_context"
            if not state.get("documents")
            else "low_grounding_after_retry"
        )
        warnings = list(state.get("warnings", []))
        if reason not in warnings:
            warnings.append(reason)
        return {
            **state,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "citations": [],
            "status": "insufficient_evidence",
            "warnings": warnings,
        }

    def finalize_ok(state: AgentState) -> AgentState:
        if state.get("status"):
            return state
        return {**state, "status": "ok"}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer)
    workflow.add_node("critique", critique)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("fallback", fallback)
    workflow.add_node("finalize", finalize_ok)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", "critique")
    workflow.add_conditional_edges(
        "critique",
        lambda state: route_after_critique(state, settings),
        {"rewrite": "rewrite", "fallback": "fallback", END: "finalize"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("fallback", END)
    workflow.add_edge("finalize", END)
    return workflow.compile()


def _format_context(documents: list[Document], settings: Settings) -> str:
    parts: list[str] = []
    used_tokens = 0
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk = f"[{index}] source={source}\n{doc.page_content}\n"
        chunk_tokens = count_tokens(chunk, encoding_name=settings.token_encoding)
        if used_tokens + chunk_tokens > settings.max_context_tokens:
            break
        parts.append(chunk)
        used_tokens += chunk_tokens
    return "\n".join(parts)


def _answer_prompt(question: str, context: str) -> str:
    return (
        "You are a grounded RAG assistant. Answer only from the provided context. "
        "If the context is insufficient, say what is missing.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer with concise reasoning and cite sources like [1], [2]."
    )


def _citation(document: Document, index: int) -> dict[str, object]:
    return {
        "id": index,
        "source": document.metadata.get("source", "unknown"),
        "page": document.metadata.get("page"),
        "score": document.metadata.get("score"),
    }
