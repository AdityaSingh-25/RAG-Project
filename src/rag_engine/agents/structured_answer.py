"""Structured-output schema for the answer node.

Opt-in via ``settings.structured_answers``. When enabled, the answer node
calls ``ChatOllama.with_structured_output(StructuredAnswer)`` which uses
the model's function-call API to force a JSON-shaped response, then we
re-serialise the claims into the same ``[n]``-marker text the rest of the
pipeline (per-claim verifier, frontend renderer, citation chips) already
understands. This makes per-claim citation extraction exact instead of
regex-based, and gives the model schema validation up front.

Trade-off worth knowing: structured-output mode bypasses token streaming
(the model emits the JSON object atomically rather than token-by-token).
``/query/stream`` therefore emits no ``token`` events when this mode is on
— the answer arrives in the ``done`` frame instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One sentence of the answer, with the chunks that support it."""

    text: str = Field(
        description=(
            "A single self-contained factual sentence taken strictly from the "
            "provided context. Do NOT include citation markers like [1] — list "
            "those in the citations field instead."
        ),
    )
    citations: list[int] = Field(
        default_factory=list,
        description=(
            "Chunk numbers from the Context section that support this sentence. "
            "Use only numbers that actually appear in Context."
        ),
    )


class StructuredAnswer(BaseModel):
    """The full answer, broken into per-claim sentences with citations."""

    claims: list[Claim] = Field(
        description=(
            "Ordered list of sentences making up the answer. If the context "
            "doesn't answer the question, return an empty list."
        ),
    )


def render_structured_answer(answer: StructuredAnswer) -> str:
    """Flatten ``StructuredAnswer`` into the [n]-marker text the rest of the
    pipeline expects.

    Example: claims = [Claim(text="Qdrant stores vectors", citations=[1, 2])]
        →  "Qdrant stores vectors [1][2]."
    """
    sentences: list[str] = []
    for claim in answer.claims:
        text = claim.text.strip()
        if not text:
            continue
        # Drop trailing punctuation so we can re-attach the markers + period
        # deterministically. Avoids "fact. [1]." / "fact [1] ." mixed shapes.
        text = text.rstrip(" .!?,;:")
        markers = "".join(f"[{i}]" for i in claim.citations)
        sentences.append(f"{text} {markers}." if markers else f"{text}.")
    return " ".join(sentences)
