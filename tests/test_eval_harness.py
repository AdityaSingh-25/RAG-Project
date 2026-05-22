import json
from pathlib import Path

from rag_engine.evaluation.harness import (
    EvalCase,
    citation_hit_rate,
    evaluate_dataset,
    format_json,
    format_markdown,
    load_cases,
    term_recall,
)


def test_citation_hit_rate_full_when_no_expectations() -> None:
    assert citation_hit_rate([], []) == 1.0


def test_citation_hit_rate_partial_match() -> None:
    citations = [{"source": "data/raw/docs/qdrant.md"}, {"source": "data/raw/docs/ollama.md"}]
    assert citation_hit_rate(citations, ["qdrant.md", "missing.md"]) == 0.5


def test_term_recall_case_insensitive_word_boundary() -> None:
    assert term_recall("Qdrant powers semantic search.", ["qdrant", "search"]) == 1.0
    # "rant" is a substring of "Qdrant" but not a whole word -> miss.
    assert term_recall("Qdrant powers semantic search.", ["rant"]) == 0.0


def test_term_recall_full_when_no_expectations() -> None:
    assert term_recall("anything", []) == 1.0


def test_load_cases_parses_jsonl_with_comments(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(
        "# header comment\n"
        '{"id": "001", "question": "What is Qdrant?", "expected_terms": ["Qdrant"]}\n'
        "\n"
        '{"id": "002", "question": "What is Ollama?", "must_cite": ["ollama.md"]}\n',
        encoding="utf-8",
    )
    cases = load_cases(fixture)
    assert [c.id for c in cases] == ["001", "002"]
    assert cases[0].expected_terms == ("Qdrant",)
    assert cases[1].must_cite == ("ollama.md",)


def test_evaluate_dataset_aggregates_metrics() -> None:
    cases = [
        EvalCase(id="a", question="Q1", expected_terms=("alpha",)),
        EvalCase(id="b", question="Q2", expected_terms=("beta",), must_cite=("doc-b",)),
    ]

    def runner(question: str) -> dict:
        if question == "Q1":
            return {
                "answer": "alpha is the answer [1].",
                "citations": [{"source": "doc-a"}],
                "grounding_score": 0.8,
                "warnings": [],
                "iteration": 0,
            }
        return {
            "answer": "gamma not beta [1].",
            "citations": [{"source": "doc-b"}],
            "grounding_score": 0.4,
            "warnings": ["low_context_overlap"],
            "iteration": 1,
        }

    report = evaluate_dataset(cases, runner)
    assert report.aggregate["n"] == 2.0
    assert report.aggregate["mean_grounding"] == 0.6
    # Q1: term recall 1.0, Q2: "beta" appears as a word -> 1.0; mean = 1.0
    assert report.aggregate["mean_term_recall"] == 1.0
    # Q1: no must_cite -> 1.0; Q2: doc-b cited -> 1.0; mean = 1.0
    assert report.aggregate["mean_citation_hit_rate"] == 1.0
    assert report.aggregate["mean_iteration"] == 0.5


def test_format_outputs_are_valid() -> None:
    cases = [EvalCase(id="a", question="Q1", expected_terms=("alpha",))]
    runner = lambda q: {
        "answer": "alpha [1].",
        "citations": [],
        "grounding_score": 0.9,
        "warnings": [],
        "iteration": 0,
    }
    report = evaluate_dataset(cases, runner)
    parsed = json.loads(format_json(report))
    assert parsed["aggregate"]["n"] == 1.0
    md = format_markdown(report)
    assert "# RAG Evaluation Report" in md
    assert "| a |" in md
