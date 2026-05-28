"""Tests for the per-request settings overrides plumbed through QueryRequest.

Each request can opt into ``claim_verifier_mode`` / ``structured_answers``
without touching the global ``Settings``. We verify that the override
propagates through to :func:`_effective_settings` and reaches the nodes
that read those fields.
"""

from __future__ import annotations

from rag_engine.agents.graph import _effective_settings
from rag_engine.config.settings import Settings


def test_effective_settings_returns_original_when_no_overrides() -> None:
    settings = Settings(claim_verifier_mode="overlap", structured_answers=False)
    state = {}  # No overrides keyed in state.
    effective = _effective_settings(state, settings)
    assert effective is settings  # Same instance — no copy made.


def test_override_claim_verifier_mode_takes_effect() -> None:
    settings = Settings(claim_verifier_mode="overlap", structured_answers=False)
    state = {"override_claim_verifier_mode": "nli"}
    effective = _effective_settings(state, settings)
    assert effective is not settings
    assert effective.claim_verifier_mode == "nli"
    # Untouched field still reflects deployment default.
    assert effective.structured_answers is False


def test_override_structured_answers_takes_effect() -> None:
    settings = Settings(claim_verifier_mode="overlap", structured_answers=False)
    state = {"override_structured_answers": True}
    effective = _effective_settings(state, settings)
    assert effective.structured_answers is True
    assert effective.claim_verifier_mode == "overlap"


def test_both_overrides_combine() -> None:
    settings = Settings(claim_verifier_mode="overlap", structured_answers=False)
    state = {
        "override_claim_verifier_mode": "nli",
        "override_structured_answers": True,
    }
    effective = _effective_settings(state, settings)
    assert effective.claim_verifier_mode == "nli"
    assert effective.structured_answers is True


def test_explicit_none_does_not_override() -> None:
    """``None`` is the wire-level "no override" sentinel from the API layer.

    A frontend that doesn't send the field at all OR sends it as ``null``
    must inherit the deployment default.
    """
    settings = Settings(claim_verifier_mode="nli", structured_answers=True)
    state = {
        "override_claim_verifier_mode": None,
        "override_structured_answers": None,
    }
    effective = _effective_settings(state, settings)
    assert effective is settings
