"""CI gate that runs the evaluation harness against the seed + adversarial datasets.

CI environments don't have Qdrant or Ollama, so this script runs each dataset
through one of two runners:

1. **Fixture runner** (default for both datasets, since committed fixtures
   exist): replays previously captured live outputs from a JSON file under
   ``data/eval/fixtures/``. Refresh fixtures with ``scripts/eval_capture.py``
   when prompts or corpus change.
2. **Stub runner** (fallback when no fixture is present): synthesises an
   output that satisfies the case's grounding heuristics. Useful for
   bootstrapping a brand-new corpus or sanity-checking harness changes.

The script gates on two floors, both configurable via env vars:

- ``EVAL_BASELINE_GROUNDING`` — mean grounding across the seed dataset.
- ``EVAL_BASELINE_STATUS_MATCH_RATE`` — fraction of cases whose status
  matches ``expected_status`` (catches regressions where adversarial cases
  stop being refused, or where ok cases start being refused).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_engine.evaluation.fixtures import build_fixture_runner  # noqa: E402
from rag_engine.evaluation.harness import (  # noqa: E402
    EvalCase,
    EvalReport,
    evaluate_dataset,
    format_markdown,
    load_cases,
)

SEED_CASES = REPO_ROOT / "data" / "eval" / "seed_cases.jsonl"
ADVERSARIAL_CASES = REPO_ROOT / "data" / "eval" / "adversarial_cases.jsonl"
SEED_FIXTURE = REPO_ROOT / "data" / "eval" / "fixtures" / "seed.json"
ADVERSARIAL_FIXTURE = REPO_ROOT / "data" / "eval" / "fixtures" / "adversarial.json"


def _stub_runner_for(case: EvalCase) -> dict[str, object]:
    """Synthesises an output matching the case's ``expected_status``."""
    if case.expected_status == "insufficient_evidence":
        return {
            "answer": "Insufficient evidence to answer confidently from the retrieved context.",
            "citations": [],
            "grounding_score": 0.0,
            "warnings": ["no_retrieved_context"],
            "iteration": 0,
            "status": "insufficient_evidence",
            "grounded_claim_rate": 0.0,
        }
    expected_terms = " ".join(case.expected_terms)
    citations = [
        {"id": idx, "source": source, "page": None, "score": 1.0}
        for idx, source in enumerate(case.must_cite or ("seed.md",), start=1)
    ]
    citation_markers = " ".join(f"[{c['id']}]" for c in citations)
    answer = (
        f"{expected_terms} {citation_markers}".strip()
        if expected_terms
        else f"answer {citation_markers}"
    )
    return {
        "answer": answer,
        "citations": citations,
        "grounding_score": 0.95,
        "warnings": [],
        "iteration": 0,
        "status": "ok",
        "grounded_claim_rate": 1.0,
    }


def _runner_for(cases: list[EvalCase], fixture_path: Path):
    """Use the fixture if available; otherwise fall back to the synthesis stub."""
    if fixture_path.exists():
        return build_fixture_runner(fixture_path)

    case_by_question = {c.question: c for c in cases}

    def stub(question: str) -> dict[str, object]:
        case = case_by_question.get(question)
        if case is None:
            raise RuntimeError(f"stub runner: unknown question {question!r}")
        return _stub_runner_for(case)

    return stub


def _evaluate(name: str, cases_path: Path, fixture_path: Path) -> EvalReport:
    cases = load_cases(cases_path)
    if not cases:
        raise RuntimeError(f"{name} dataset is empty: {cases_path}")
    runner = _runner_for(cases, fixture_path)
    source = "fixture" if fixture_path.exists() else "stub"
    print(f"==> {name} ({len(cases)} cases, runner={source})")
    report = evaluate_dataset(cases, runner)
    print(format_markdown(report))
    return report


def _check(name: str, value: float, baseline: float) -> bool:
    if value < baseline:
        print(f"FAIL: {name} {value:.3f} < baseline {baseline:.3f}", file=sys.stderr)
        return False
    print(f"OK:   {name} {value:.3f} >= baseline {baseline:.3f}")
    return True


def main() -> int:
    if not SEED_CASES.exists():
        print(f"seed dataset not found: {SEED_CASES}", file=sys.stderr)
        return 1
    if not ADVERSARIAL_CASES.exists():
        print(f"adversarial dataset not found: {ADVERSARIAL_CASES}", file=sys.stderr)
        return 1

    seed_report = _evaluate("seed", SEED_CASES, SEED_FIXTURE)
    adv_report = _evaluate("adversarial", ADVERSARIAL_CASES, ADVERSARIAL_FIXTURE)

    grounding_baseline = float(os.environ.get("EVAL_BASELINE_GROUNDING", "0.0"))
    status_baseline = float(os.environ.get("EVAL_BASELINE_STATUS_MATCH_RATE", "0.0"))

    seed_grounding = seed_report.aggregate["mean_grounding"]
    seed_status = seed_report.aggregate["status_match_rate"]
    adv_status = adv_report.aggregate["status_match_rate"]

    print()
    grounding_ok = _check("seed.mean_grounding", seed_grounding, grounding_baseline)
    seed_status_ok = _check("seed.status_match_rate", seed_status, status_baseline)
    adv_status_ok = _check("adversarial.status_match_rate", adv_status, status_baseline)

    return 0 if grounding_ok and seed_status_ok and adv_status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
