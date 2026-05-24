from langchain_core.documents import Document

from rag_engine.config.settings import Settings
from rag_engine.retrieval.cross_encoder_reranker import rerank_with_cross_encoder
from rag_engine.retrieval.reranker import apply_reranker


class _StubEncoder:
    """Predictable cross-encoder substitute keyed on the document text."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores = scores_by_text
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return [self._scores.get(text, 0.0) for _q, text in pairs]


def _doc(text: str, **metadata: object) -> Document:
    return Document(page_content=text, metadata=dict(metadata))


def test_cross_encoder_reorders_by_score_and_truncates() -> None:
    docs = [
        _doc("alpha noise filler", source="a.md"),
        _doc("beta winner highly relevant", source="b.md"),
        _doc("gamma middle ground", source="c.md"),
    ]
    encoder = _StubEncoder(
        {
            "alpha noise filler": 0.1,
            "beta winner highly relevant": 0.9,
            "gamma middle ground": 0.5,
        }
    )
    out = rerank_with_cross_encoder(
        "Q",
        docs,
        top_k=2,
        model_name="ignored",
        encoder=encoder,
    )
    assert [d.metadata["source"] for d in out] == ["b.md", "c.md"]
    assert [round(d.metadata["rerank_score"], 2) for d in out] == [0.9, 0.5]


def test_cross_encoder_preserves_existing_metadata() -> None:
    docs = [_doc("only doc", source="only.md", rrf_score=0.42)]
    encoder = _StubEncoder({"only doc": 0.7})
    out = rerank_with_cross_encoder("Q", docs, top_k=5, model_name="m", encoder=encoder)
    assert out[0].metadata["source"] == "only.md"
    assert out[0].metadata["rrf_score"] == 0.42
    assert out[0].metadata["rerank_score"] == 0.7


def test_cross_encoder_handles_empty_input() -> None:
    out = rerank_with_cross_encoder("Q", [], top_k=5, model_name="m", encoder=_StubEncoder({}))
    assert out == []


def test_apply_reranker_disabled_truncates_only(monkeypatch) -> None:
    settings = Settings(reranker_mode="disabled", top_k=2)
    docs = [_doc("a"), _doc("b"), _doc("c")]

    # If anything tried to call the cross-encoder, this would raise.
    def boom(*_a, **_k):
        raise AssertionError("disabled mode must not invoke the cross-encoder")

    monkeypatch.setattr("rag_engine.retrieval.reranker.rerank_with_cross_encoder", boom)
    out = apply_reranker("Q", docs, settings)
    assert [d.page_content for d in out] == ["a", "b"]


def test_apply_reranker_keyword_mode_uses_legacy_path() -> None:
    settings = Settings(reranker_mode="keyword", top_k=1)
    docs = [
        _doc("alpha", score=0.1),
        _doc("Qdrant vector database", score=0.9),
    ]
    out = apply_reranker("What is Qdrant?", docs, settings)
    assert out[0].page_content == "Qdrant vector database"


def test_apply_reranker_cross_encoder_mode_calls_cross_encoder(monkeypatch) -> None:
    settings = Settings(
        reranker_mode="cross_encoder",
        cross_encoder_model="stub-model",
        top_k=1,
    )
    docs = [_doc("alpha"), _doc("beta")]

    captured: dict[str, object] = {}

    def fake_cross_encoder(question, documents, top_k, model_name, encoder=None):
        captured["question"] = question
        captured["model_name"] = model_name
        captured["top_k"] = top_k
        return documents[:top_k]

    monkeypatch.setattr(
        "rag_engine.retrieval.reranker.rerank_with_cross_encoder",
        fake_cross_encoder,
    )
    out = apply_reranker("Q", docs, settings)
    assert captured == {"question": "Q", "model_name": "stub-model", "top_k": 1}
    assert len(out) == 1
