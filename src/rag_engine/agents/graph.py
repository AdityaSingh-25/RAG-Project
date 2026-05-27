from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from rag_engine.agents.state import AgentState
from rag_engine.config.settings import Settings
from rag_engine.evaluation.claim_grounding import verify_claims_with_settings
from rag_engine.evaluation.hallucination import verify_answer_confidence
from rag_engine.observability.counters import counters
from rag_engine.observability.logging import get_logger, log_event, now_ms
from rag_engine.retrieval.query_rewriter import rewrite_query
from rag_engine.retrieval.reranker import apply_reranker
from rag_engine.retrieval.retriever import build_retriever
from rag_engine.routing.model_router import ModelRouter
from rag_engine.utils.tokenization import count_tokens

_logger = get_logger("rag_engine.graph")

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence to answer confidently from the retrieved context."
)

# Channel name used to emit per-token deltas via langgraph's "custom" stream.
# The SSE bridge in api/main.py looks for this exact key.
ANSWER_TOKEN_CHANNEL = "answer.token"


def _stream_writer():
    """Return langgraph's stream writer, or a no-op outside a streaming context.

    When the graph is driven by `astream(..., stream_mode=["custom", ...])`
    the writer forwards tokens to the SSE bridge; when driven by `invoke()`
    (legacy `/query`, tests), there's no writer in scope and we fall through
    to a no-op so the node behaves identically.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return lambda _payload: None


def _append_trace(state: AgentState, node: str, duration_ms: float, **extras) -> list[dict]:
    """Append a per-node trace entry. Used by the UI's pipeline sidebar."""
    trace = list(state.get("pipeline_trace") or [])
    trace.append(
        {
            "node": node,
            "duration_ms": round(duration_ms, 2),
            "iteration": state.get("iteration", 0),
            **extras,
        }
    )
    return trace


def route_after_critique(state: AgentState, settings: Settings) -> str:
    """Decide what happens after the critic runs.

    Returns the next node name: ``"rewrite"`` to loop, ``"fallback"`` to refuse,
    or ``END`` when the answer is good enough to ship.

    Both the overall grounding score AND the per-claim grounding rate must
    clear their thresholds. The per-claim check is what catches answers
    that mention the right keywords overall but have individual sentences
    drifting away from cited chunks.
    """
    score = state.get("grounding_score", 0.0)
    claim_rate = state.get("grounded_claim_rate", 1.0)
    iteration = state.get("iteration", 0)

    overall_ok = score >= settings.grounding_threshold
    claims_ok = (
        not settings.enforce_per_claim_citations
        or claim_rate >= settings.min_grounded_claim_rate
    )

    if overall_ok and claims_ok:
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
        trace_id = state.get("trace_id", "-")
        started = now_ms()
        documents = retriever.invoke(state["question"])
        reranked = apply_reranker(state["question"], documents, settings)
        duration = now_ms() - started
        counters().observe("graph.retrieve.latency_ms", duration)
        log_event(
            _logger,
            "graph.retrieve",
            trace_id=trace_id,
            duration_ms=round(duration, 2),
            n_candidates=len(documents),
            n_reranked=len(reranked),
            retrieval_mode=settings.retrieval_mode,
            reranker_mode=settings.reranker_mode,
        )
        return {
            **state,
            "documents": reranked,
            "pipeline_trace": _append_trace(
                state,
                "retrieve",
                duration,
                n_candidates=len(documents),
                n_reranked=len(reranked),
            ),
        }

    def answer(state: AgentState) -> AgentState:
        trace_id = state.get("trace_id", "-")
        started = now_ms()
        sorted_documents = sorted(
            state["documents"],
            key=lambda doc: doc.metadata.get("score", 0),
            reverse=True,
        )
        context = _format_context(sorted_documents, settings)
        llm = router.for_question(state["question"], context)
        prompt = _answer_prompt(state["question"], context)

        writer = _stream_writer()
        chunks: list[str] = []
        for chunk in llm.stream(prompt):
            delta = getattr(chunk, "content", "") or ""
            if delta:
                chunks.append(delta)
                writer({ANSWER_TOKEN_CHANNEL: delta})
        answer_text = "".join(chunks)

        citations = [_citation(doc, index) for index, doc in enumerate(sorted_documents, start=1)]
        duration = now_ms() - started
        counters().observe("graph.answer.latency_ms", duration)
        log_event(
            _logger,
            "graph.answer",
            trace_id=trace_id,
            duration_ms=round(duration, 2),
            model=getattr(llm, "model", "unknown"),
            answer_chars=len(answer_text),
        )
        return {
            **state,
            "answer": answer_text,
            "citations": citations,
            "pipeline_trace": _append_trace(
                state,
                "answer",
                duration,
                model=getattr(llm, "model", "unknown"),
                answer_chars=len(answer_text),
            ),
        }

    def critique(state: AgentState) -> AgentState:
        trace_id = state.get("trace_id", "-")
        started = now_ms()
        confidence = verify_answer_confidence(state["answer"], state["documents"])
        score = confidence["grounding_score"]
        warnings = list(confidence["warnings"])

        report = verify_claims_with_settings(
            state["answer"],
            state["documents"],
            settings,
        )
        claim_rate = report.grounded_claim_rate
        if (
            settings.enforce_per_claim_citations
            and report.claims
            and claim_rate < settings.min_grounded_claim_rate
        ):
            warnings.append("low_per_claim_grounding")

        counters().observe("graph.critique.grounded_claim_rate", claim_rate)
        duration = now_ms() - started
        log_event(
            _logger,
            "graph.critique",
            trace_id=trace_id,
            grounding_score=score,
            grounded_claim_rate=claim_rate,
            n_claims=len(report.claims),
            n_ungrounded=len(report.ungrounded),
            warnings=warnings,
        )
        return {
            **state,
            "grounding_score": score,
            "warnings": warnings,
            "grounded_claim_rate": claim_rate,
            "claim_grounding": [
                {
                    "sentence": c.sentence,
                    "cited_indices": list(c.cited_indices),
                    "valid_indices": list(c.valid_indices),
                    "support_score": c.support_score,
                    "is_grounded": c.is_grounded,
                }
                for c in report.claims
            ],
            "pipeline_trace": _append_trace(
                state,
                "critique",
                duration,
                grounding_score=score,
                grounded_claim_rate=claim_rate,
                n_claims=len(report.claims),
            ),
        }

    def rewrite(state: AgentState) -> AgentState:
        trace_id = state.get("trace_id", "-")
        started = now_ms()
        original = state.get("original_question") or state["question"]
        rewritten = rewrite_query(original, state.get("documents", []))
        next_iteration = state.get("iteration", 0) + 1
        duration = now_ms() - started
        counters().increment("graph.rewrite.invocations")
        counters().observe("graph.iteration", next_iteration)
        log_event(
            _logger,
            "graph.rewrite",
            trace_id=trace_id,
            iteration=next_iteration,
            rewritten_chars=len(rewritten),
        )
        return {
            **state,
            "question": rewritten,
            "iteration": next_iteration,
            "pipeline_trace": _append_trace(
                {**state, "iteration": next_iteration},
                "rewrite",
                duration,
                rewritten_chars=len(rewritten),
            ),
        }

    def fallback(state: AgentState) -> AgentState:
        trace_id = state.get("trace_id", "-")
        if not state.get("documents"):
            reason = "no_retrieved_context"
        elif (
            settings.enforce_per_claim_citations
            and state.get("grounded_claim_rate", 1.0) < settings.min_grounded_claim_rate
        ):
            reason = "low_per_claim_grounding"
        else:
            reason = "low_grounding_after_retry"
        warnings = list(state.get("warnings", []))
        if reason not in warnings:
            warnings.append(reason)
        counters().increment("graph.fallback.invocations")
        counters().increment(f"graph.fallback.reason.{reason}")
        log_event(
            _logger,
            "graph.fallback",
            trace_id=trace_id,
            reason=reason,
            iteration=state.get("iteration", 0),
        )
        return {
            **state,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "citations": [],
            "status": "insufficient_evidence",
            "warnings": warnings,
            "claim_grounding": [],
            "pipeline_trace": _append_trace(state, "fallback", 0.0, reason=reason),
        }

    def finalize_ok(state: AgentState) -> AgentState:
        trace_id = state.get("trace_id", "-")
        counters().increment("graph.finalize.ok")
        log_event(
            _logger,
            "graph.finalize",
            trace_id=trace_id,
            grounding_score=state.get("grounding_score", 0.0),
            iteration=state.get("iteration", 0),
        )
        traced = _append_trace(state, "finalize", 0.0)
        if state.get("status"):
            return {**state, "pipeline_trace": traced}
        return {**state, "status": "ok", "pipeline_trace": traced}

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
        "You are a grounded RAG assistant. Use ONLY the provided context — do "
        "not draw on outside knowledge.\n"
        "Rules:\n"
        " - Write in short, self-contained sentences.\n"
        " - Every factual sentence MUST end with one or more citation markers "
        "such as [1] or [2][3] pointing to the chunks that support it.\n"
        " - Only cite chunk numbers that appear in the Context section below.\n"
        " - If the context does not answer the question, say exactly what is "
        "missing instead of guessing.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def _citation(document: Document, index: int) -> dict[str, object]:
    # Truncated content lets the UI render a hover preview without making the
    # response payload unbounded. Keep this small; the full chunk lives in
    # Qdrant if a caller really needs it.
    snippet = document.page_content.strip().replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:237] + "…"
    return {
        "id": index,
        "source": document.metadata.get("source", "unknown"),
        "page": document.metadata.get("page"),
        "score": document.metadata.get("score"),
        "content": snippet,
    }
