"""CI gate that runs the evaluation harness against the seed dataset.

CI environments don't have Qdrant or Ollama, so we run the harness with a
deterministic stub runner. The stub answers each case by embedding its
expected terms and the must_cite filenames, which keeps the math under our
control while still exercising the full harness pipeline.

The script's job is to:

1. Catch regressions in the harness itself (scoring, JSON shape, formatters).
2. Catch malformed seed cases (loader raises, missing fields).
3. Hold a configurable grounding floor (``EVAL_BASELINE_GROUNDING``) so the
   gate has somewhere to grow once real models are wired into CI.

Run locally with ``python scripts/eval_ci.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_engine.evaluation.harness import (  # noqa: E402
    EvalCase,
    evaluate_dataset,
    format_markdown,
    load_cases,
)

SEED_CASES = REPO_ROOT / "data" / "eval" / "seed_cases.jsonl"


def stub_runner_for(case: EvalCase) -> dict[str, object]:
    """Deterministic answer that satisfies the case's grounding heuristics."""
    expected = " ".join(case.expected_terms)
    citations = [
        {"id": idx, "source": source, "page": None, "score": 1.0}
        for idx, source in enumerate(case.must_cite or ("seed.md",), start=1)
    ]
    citation_markers = " ".join(f"[{c['id']}]" for c in citations)
    answer = f"{expected} {citation_markers}".strip() if expected else f"answer {citation_markers}"
    return {
        "answer": answer,
        "citations": citations,
        "grounding_score": 0.95,
        "warnings": [],
        "iteration": 0,
        "status": "ok",
    }


def main() -> int:
    if not SEED_CASES.exists():
        print(f"seed dataset not found: {SEED_CASES}", file=sys.stderr)
        return 1

    cases = load_cases(SEED_CASES)
    if not cases:
        print(f"seed dataset is empty: {SEED_CASES}", file=sys.stderr)
        return 1

    case_by_id = {c.id: c for c in cases}

    def runner(question: str) -> dict[str, object]:
        match = next((c for c in cases if c.question == question), None)
        if match is None:
            raise RuntimeError(f"runner could not match question: {question!r}")
        return stub_runner_for(case_by_id[match.id])

    report = evaluate_dataset(cases, runner)
    print(format_markdown(report))

    baseline = float(os.environ.get("EVAL_BASELINE_GROUNDING", "0.0"))
    actual = report.aggregate["mean_grounding"]
    if actual < baseline:
        print(
            f"FAIL: mean grounding {actual:.3f} < baseline {baseline:.3f}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: mean grounding {actual:.3f} >= baseline {baseline:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
