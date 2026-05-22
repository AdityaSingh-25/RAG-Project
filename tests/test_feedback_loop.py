from langgraph.graph import END

from rag_engine.agents.graph import should_retry
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
    }
    state.update(overrides)
    return state


def test_should_retry_loops_on_low_grounding() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert should_retry(_base_state(grounding_score=0.2), settings) == "rewrite"


def test_should_retry_stops_when_grounding_high_enough() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert should_retry(_base_state(grounding_score=0.7), settings) == END


def test_should_retry_stops_at_max_iterations() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert should_retry(_base_state(grounding_score=0.1, iteration=1), settings) == END


def test_should_retry_disabled_short_circuits() -> None:
    settings = Settings(enable_feedback_loop=False, grounding_threshold=0.6, max_retry_iterations=2)
    assert should_retry(_base_state(grounding_score=0.0), settings) == END


def test_should_retry_threshold_boundary_is_inclusive() -> None:
    settings = Settings(enable_feedback_loop=True, grounding_threshold=0.6, max_retry_iterations=1)
    assert should_retry(_base_state(grounding_score=0.6), settings) == END
