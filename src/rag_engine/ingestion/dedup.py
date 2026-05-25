"""Content-hash deduplication for ingested chunks.

Two copies of the same document (or the same chunk in different docs)
would otherwise both be embedded, indexed in Qdrant, and added to the
BM25 corpus — dominating retrieval and inflating both vector storage
and reranker cost. We dedupe at chunk granularity: a SHA-256 over the
normalised chunk text. Source path is intentionally NOT part of the
key so the same paragraph appearing in two locations is recognised
as a duplicate.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from langchain_core.documents import Document

_WHITESPACE_RE = re.compile(r"\s+")


def chunk_hash(text: str) -> str:
    """Stable hash for chunk content; insensitive to whitespace differences."""
    normalised = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@dataclass
class DedupResult:
    unique: list[Document]
    duplicates_removed: int


def dedupe_chunks(chunks: list[Document]) -> DedupResult:
    """Keep the first occurrence of each chunk; drop later duplicates.

    The hash is also written into ``metadata["content_hash"]`` so downstream
    code can detect duplicates introduced by later ingests (without needing
    to rehash).
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        existing = chunk.metadata.get("content_hash")
        digest = existing if isinstance(existing, str) and existing else chunk_hash(chunk.page_content)
        if digest in seen:
            continue
        seen.add(digest)
        # Attach the hash without mutating the input document.
        unique.append(
            Document(
                page_content=chunk.page_content,
                metadata={**chunk.metadata, "content_hash": digest},
            )
        )
    return DedupResult(unique=unique, duplicates_removed=len(chunks) - len(unique))
