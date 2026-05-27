"""Tests for the structured-output answer path.

Covers the Pydantic schema + the flatten step that turns it back into the
``[n]``-marker text the rest of the engine expects. The graph-level
``_generate_structured`` is also tested with a stub LLM so we don't need
a real Ollama server.
"""

from __future__ import annotations

from rag_engine.agents.graph import _generate_structured
from rag_engine.agents.structured_answer import (
    Claim,
    StructuredAnswer,
    render_structured_answer,
)


def test_render_attaches_markers_and_period() -> None:
    answer = StructuredAnswer(
        claims=[
            Claim(text="Qdrant stores vectors", citations=[1]),
            Claim(text="Ollama serves local LLMs", citations=[2, 3]),
        ]
    )
    rendered = render_structured_answer(answer)
    assert rendered == "Qdrant stores vectors [1]. Ollama serves local LLMs [2][3]."


def test_render_strips_trailing_punctuation_before_appending_markers() -> None:
    # Model may include a trailing period despite the prompt; render must
    # not produce "fact. [1]." which would break the per-claim verifier.
    answer = StructuredAnswer(
        claims=[Claim(text="A fact.", citations=[1])]
    )
    assert render_structured_answer(answer) == "A fact [1]."


def test_render_drops_empty_claims() -> None:
    answer = StructuredAnswer(
        claims=[
            Claim(text="", citations=[1]),
            Claim(text="   ", citations=[2]),
            Claim(text="Real claim", citations=[3]),
        ]
    )
    assert render_structured_answer(answer) == "Real claim [3]."


def test_render_no_citations_still_adds_period() -> None:
    answer = StructuredAnswer(claims=[Claim(text="Unsupported", citations=[])])
    assert render_structured_answer(answer) == "Unsupported."


def test_render_empty_answer() -> None:
    assert render_structured_answer(StructuredAnswer(claims=[])) == ""


class _StubStructuredLLM:
    def __init__(self, result):
        self._result = result
        self.invoke_calls: list[str] = []

    def invoke(self, prompt: str):
        self.invoke_calls.append(prompt)
        return self._result


class _StubLLM:
    """Pretends to be a ChatOllama with ``.with_structured_output``."""

    def __init__(self, structured_result):
        self._structured = _StubStructuredLLM(structured_result)
        self.with_structured_output_calls: list = []

    def with_structured_output(self, schema):
        self.with_structured_output_calls.append(schema)
        return self._structured


def test_generate_structured_invokes_with_structured_output() -> None:
    expected = StructuredAnswer(
        claims=[Claim(text="Qdrant is fast", citations=[1])]
    )
    llm = _StubLLM(structured_result=expected)
    out = _generate_structured(llm, "the prompt")
    assert out == "Qdrant is fast [1]."
    assert llm.with_structured_output_calls == [StructuredAnswer]
    assert llm._structured.invoke_calls == ["the prompt"]


def test_generate_structured_coerces_dict_results() -> None:
    # Some LangChain backends return a dict instead of a Pydantic instance.
    raw_dict = {"claims": [{"text": "Hello", "citations": [2]}]}
    llm = _StubLLM(structured_result=raw_dict)
    out = _generate_structured(llm, "prompt")
    assert out == "Hello [2]."
