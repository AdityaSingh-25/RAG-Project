"""Tests for the FixtureRunner replay path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.evaluation.fixtures import (
    FixtureEntry,
    build_fixture_runner,
    load_fixture,
    save_fixture,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_save_then_load_round_trips_entries(tmp_path: Path) -> None:
    target = tmp_path / "fx.json"
    save_fixture(
        target,
        [
            FixtureEntry(
                question="Q1",
                captured_at="2026-05-26T00:00:00+00:00",
                output={"answer": "A", "grounding_score": 0.8},
            ),
            FixtureEntry(question="Q2", output={"answer": "B"}),
        ],
    )
    loaded = load_fixture(target)
    assert set(loaded.keys()) == {"Q1", "Q2"}
    assert loaded["Q1"]["answer"] == "A"


def test_build_fixture_runner_replays_outputs(tmp_path: Path) -> None:
    target = tmp_path / "fx.json"
    save_fixture(target, [FixtureEntry(question="Q1", output={"answer": "A"})])
    runner = build_fixture_runner(target)
    assert runner("Q1") == {"answer": "A"}


def test_build_fixture_runner_raises_on_unknown_question(tmp_path: Path) -> None:
    target = tmp_path / "fx.json"
    save_fixture(target, [FixtureEntry(question="Q1", output={"answer": "A"})])
    runner = build_fixture_runner(target)
    with pytest.raises(RuntimeError, match="no recording"):
        runner("never seen")


def test_load_fixture_rejects_duplicate_questions(tmp_path: Path) -> None:
    target = tmp_path / "dupe.json"
    _write(
        target,
        {
            "entries": [
                {"question": "Q", "output": {"a": 1}},
                {"question": "Q", "output": {"a": 2}},
            ]
        },
    )
    with pytest.raises(ValueError, match="duplicate question"):
        load_fixture(target)


def test_load_fixture_rejects_malformed_entries(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    _write(target, {"entries": [{"question": "missing output field"}]})
    with pytest.raises(ValueError, match="must be an object with"):
        load_fixture(target)


def test_load_fixture_accepts_bare_list_root(tmp_path: Path) -> None:
    """Older fixtures or hand-edited files may omit the 'entries' wrapper."""
    target = tmp_path / "bare.json"
    _write(target, [{"question": "Q", "output": {"answer": "A"}}])
    loaded = load_fixture(target)
    assert loaded["Q"]["answer"] == "A"
