from rag_engine.config.settings import Settings
from rag_engine.routing.model_router import ModelRouter


def test_router_uses_reasoning_model_for_complex_question() -> None:
    settings = Settings()
    router = ModelRouter(settings)
    assert router._choose_model("Explain the tradeoff in this architecture", "") == (
        settings.chat_model_reasoning
    )


def test_router_uses_fast_model_for_simple_question() -> None:
    settings = Settings()
    router = ModelRouter(settings)
    assert router._choose_model("What is Qdrant?", "short context") == settings.chat_model_fast


def test_router_uses_reasoning_model_for_high_context() -> None:
    settings = Settings()
    router = ModelRouter(settings)
    long_context = "context " * 4000
    assert router._choose_model("What is Qdrant?", long_context) == settings.chat_model_reasoning

