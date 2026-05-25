"""Tests for the insufficient-evidence path and the CI eval gate script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rag_engine.evaluation.harness import EvalCase, evaluate_dataset, format_json

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_includes_insufficient_evidence_rate() -> None:
    cases = [
        EvalCase(id="ok", question="Q1"),
        EvalCase(id="bad", question="Q2"),
    ]

    def runner(question: str) -> dict:
        if question == "Q1":
            return {"answer": "answer", "grounding_score": 0.9, "status": "ok"}
        return {
            "answer": "Insufficient evidence to answer.",
            "grounding_score": 0.1,
            "status": "insufficient_evidence",
        }

    report = evaluate_dataset(cases, runner)
    assert report.aggregate["insufficient_evidence_rate"] == 0.5
    parsed = json.loads(format_json(report))
    statuses = sorted(r["status"] for r in parsed["results"])
    assert statuses == ["insufficient_evidence", "ok"]


def test_harness_defaults_status_to_ok_when_runner_omits_it() -> None:
    cases = [EvalCase(id="x", question="Q")]
    report = evaluate_dataset(cases, lambda _q: {"answer": "a", "grounding_score": 0.9})
    assert report.results[0].status == "ok"
    assert report.aggregate["insufficient_evidence_rate"] == 0.0


def test_eval_ci_passes_with_default_baseline() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_ci.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "EVAL_BASELINE_GROUNDING": "0.0",
            "EVAL_BASELINE_STATUS_MATCH_RATE": "0.0",
        },
    )
    assert result.returncode == 0, result.stderr
    assert "OK:   seed.mean_grounding" in result.stdout
    assert "OK:   adversarial.status_match_rate" in result.stdout


def test_eval_ci_fails_when_grounding_baseline_is_unachievable() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_ci.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "EVAL_BASELINE_GROUNDING": "1.0",
            "EVAL_BASELINE_STATUS_MATCH_RATE": "0.0",
        },
    )
    # Fixture grounding is ~0.8 on seed, so baseline=1.0 must fail.
    assert result.returncode == 1
    assert "FAIL: seed.mean_grounding" in result.stderr


def test_eval_ci_fails_when_status_match_baseline_is_unachievable() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_ci.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "EVAL_BASELINE_GROUNDING": "0.0",
            "EVAL_BASELINE_STATUS_MATCH_RATE": "1.1",
        },
    )
    # No achievable status_match_rate clears 1.1; both seed + adversarial fail.
    assert result.returncode == 1
    assert "FAIL: seed.status_match_rate" in result.stderr
