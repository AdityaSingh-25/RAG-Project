"""Replay captured graph outputs as an EvalRunner.

Phase 4 introduced the CI eval gate but ran it against a deterministic
stub — useful for catching harness regressions, useless for catching
*model* regressions. Fixtures bridge the gap: capture the live graph's
output once with ``scripts/eval_capture.py``, commit the resulting JSON,
and replay it in CI to gate on the recorded behavior.

When the corpus or prompts change in a way that's expected to shift
outputs, refresh the fixture; the diff in PR review is the change set
for the eval suite. The CI gate stays cheap (no models loaded) while
representing real engine behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_engine.evaluation.harness import EvalRunner


@dataclass
class FixtureEntry:
    question: str
    output: dict[str, Any]
    captured_at: str | None = None


def load_fixture(path: Path) -> dict[str, dict[str, Any]]:
    """Load a fixture file and return a ``{question: output}`` mapping.

    Raises ``ValueError`` on duplicate questions so silent overrides can't
    hide capture mistakes.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of entries, got {type(entries).__name__}")

    by_question: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "question" not in entry or "output" not in entry:
            raise ValueError(
                f"{path}[{index}]: each entry must be an object with 'question' and 'output'"
            )
        question = str(entry["question"])
        if question in by_question:
            raise ValueError(f"{path}[{index}]: duplicate question {question!r}")
        by_question[question] = dict(entry["output"])
    return by_question


def save_fixture(path: Path, entries: list[FixtureEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [
            {
                "question": e.question,
                "captured_at": e.captured_at,
                "output": e.output,
            }
            for e in entries
        ]
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def build_fixture_runner(fixture_path: Path) -> EvalRunner:
    """Return a runner that replays the fixture, raising on unknown questions."""
    by_question = load_fixture(fixture_path)

    def runner(question: str) -> dict[str, Any]:
        if question not in by_question:
            raise RuntimeError(
                f"fixture has no recording for {question!r} (from {fixture_path}). "
                "Regenerate the fixture with scripts/eval_capture.py."
            )
        return by_question[question]

    return runner
