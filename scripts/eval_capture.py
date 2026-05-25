"""Capture live engine outputs into a fixture file for CI replay.

Run this against a working Qdrant + Ollama stack to refresh the fixtures
committed under ``data/eval/fixtures/``. The output JSON is human-readable
and intended to be reviewed in PR diffs whenever model behavior shifts.

Usage::

    python scripts/eval_capture.py \\
        --cases data/eval/seed_cases.jsonl \\
        --fixture data/eval/fixtures/seed.json

The fixture is keyed on the case question. ``scripts/eval_ci.py`` and
``rag-eval --fixture`` both consume it.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_engine.agents.graph import build_graph  # noqa: E402
from rag_engine.config.settings import get_settings  # noqa: E402
from rag_engine.evaluation.fixtures import FixtureEntry, save_fixture  # noqa: E402
from rag_engine.evaluation.harness import load_cases  # noqa: E402


def _initial_state(question: str) -> dict[str, object]:
    return {
        "question": question,
        "original_question": question,
        "filters": {},
        "documents": [],
        "answer": "",
        "citations": [],
        "grounding_score": 0.0,
        "warnings": [],
        "iteration": 0,
        "status": "",
        "trace_id": "capture",
        "grounded_claim_rate": 1.0,
        "claim_grounding": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture live engine output into a fixture file.")
    parser.add_argument("--cases", required=True, help="JSONL of EvalCases to capture.")
    parser.add_argument("--fixture", required=True, help="Output fixture path.")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(Path(args.cases))
    if not cases:
        print(f"no cases in {args.cases}", file=sys.stderr)
        return 1

    graph = build_graph(settings)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    entries: list[FixtureEntry] = []
    for case in cases:
        print(f"capturing {case.id}: {case.question[:60]}...", file=sys.stderr)
        result = graph.invoke(_initial_state(case.question))
        entries.append(
            FixtureEntry(
                question=case.question,
                captured_at=now,
                output={
                    "answer": result.get("answer", ""),
                    "citations": result.get("citations", []),
                    "grounding_score": result.get("grounding_score", 0.0),
                    "warnings": list(result.get("warnings", [])),
                    "iteration": result.get("iteration", 0),
                    "status": result.get("status") or "ok",
                    "grounded_claim_rate": result.get("grounded_claim_rate", 1.0),
                },
            )
        )

    save_fixture(Path(args.fixture), entries)
    print(f"wrote {len(entries)} entries to {args.fixture}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
