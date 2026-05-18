from rag_engine.utils.tokenization import count_tokens


def test_count_tokens_returns_nonzero_value() -> None:
    assert count_tokens("Hello world") > 0
