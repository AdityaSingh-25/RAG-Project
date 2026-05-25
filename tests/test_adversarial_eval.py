"""Tests for the adversarial/expected_status pathway in the eval harness."""

from __future__ import annotations

from pathlib import Path

from rag_engine.evaluation.harness import EvalCase, evaluate_dataset, load_cases


def test_load_cases_parses_expected_status(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(
        '{"id": "a", "question": "Q"}\n'
        '{"id": "b", "question": "Q2", "expected_status": "insufficient_evidence"}\n',
        encoding="utf-8",
    )
    cases = load_cases(fixture)
    assert cases[0].expected_status == "ok"  # default applied
    assert cases[1].expected_status == "insufficient_evidence"


def test_status_match_rate_rewards_correct_refusals() -> None:
    cases = [
        EvalCase(id="ok", question="Q1", expected_status="ok"),
        EvalCase(
            id="adv-refused",
            question="Q2",
            expected_status="insufficient_evidence",
        ),
        EvalCase(
            id="adv-leaked",
            question="Q3",
            expected_status="insufficient_evidence",
        ),
    ]

    def runner(question: str) -> dict:
        if question == "Q1":
            return {"answer": "alpha", "grounding_score": 0.9, "status": "ok"}
        if question == "Q2":
            return {
                "answer": "Insufficient evidence",
                "grounding_score": 0.1,
                "status": "insufficient_evidence",
            }
        # adv-leaked: critic mis-allowed a refusal-worthy question.
        return {"answer": "fabricated [1]", "grounding_score": 0.8, "status": "ok"}

    report = evaluate_dataset(cases, runner)
    # 2 of 3 matched; the leaked adversarial drags the rate down.
    assert report.aggregate["status_match_rate"] == round(2 / 3, 3)
    matches = {r.case_id: r.status_matches_expected for r in report.results}
    assert matches == {"ok": True, "adv-refused": True, "adv-leaked": False}


def test_status_match_default_is_ok() -> None:
    # When a case has no expected_status and the runner returns no status,
    # both default to "ok" so they match.
    cases = [EvalCase(id="x", question="Q")]
    report = evaluate_dataset(
        cases, lambda _q: {"answer": "answer", "grounding_score": 0.9}
    )
    assert report.results[0].status_matches_expected is True
    assert report.aggregate["status_match_rate"] == 1.0
