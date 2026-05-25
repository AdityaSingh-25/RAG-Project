from langgraph.graph import END

from rag_engine.agents.graph import route_after_critique
from rag_engine.config.settings import Settings


def _base_state(**overrides):
    state = {
        "question": "q",
        "original_question": "q",
        "filters": {},
        "documents": [],
        "answer": "a",
        "citations": [],
        "grounding_score": 0.0,
        "warnings": [],
        "iteration": 0,
        "status": "",
        "grounded_claim_rate": 1.0,
        "claim_grounding": [],
    }
    state.update(overrides)
    return state


def test_route_loops_on_low_grounding() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert route_after_critique(_base_state(grounding_score=0.2), settings) == "rewrite"


def test_route_terminates_when_grounding_high_enough() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert route_after_critique(_base_state(grounding_score=0.7), settings) == END


def test_route_falls_back_after_max_iterations() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert (
        route_after_critique(_base_state(grounding_score=0.1, iteration=1), settings)
        == "fallback"
    )


def test_route_falls_back_when_loop_disabled() -> None:
    settings = Settings(enable_feedback_loop=False, grounding_threshold=0.6, max_retry_iterations=2)
    assert route_after_critique(_base_state(grounding_score=0.0), settings) == "fallback"


def test_route_threshold_boundary_is_inclusive() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert route_after_critique(_base_state(grounding_score=0.6), settings) == END


def test_route_loops_on_low_claim_rate_even_with_good_overall_grounding() -> None:
    settings = Settings(
        enable_feedback_loop=True,
        grounding_threshold=0.6,
        max_retry_iterations=1,
        enforce_per_claim_citations=True,
        min_grounded_claim_rate=0.5,
    )
    state = _base_state(grounding_score=0.9, grounded_claim_rate=0.2)
    assert route_after_critique(state, settings) == "rewrite"


def test_route_ignores_claim_rate_when_enforcement_disabled() -> None:
    settings = Settings(
        enable_feedback_loop=True,
        grounding_threshold=0.6,
        max_retry_iterations=1,
        enforce_per_claim_citations=False,
        min_grounded_claim_rate=0.9,
    )
    state = _base_state(grounding_score=0.9, grounded_claim_rate=0.0)
    assert route_after_critique(state, settings) == END
