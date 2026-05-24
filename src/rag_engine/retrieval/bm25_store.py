"""BM25 sparse index built alongside Qdrant ingestion.

The corpus is kept in-process and persisted as a pickle so it survives
across API restarts. Rebuilding is idempotent: every call to
:func:`build_bm25_index` writes a fresh snapshot from the supplied docs.

We deliberately keep this small and dependency-light. A larger corpus or
incremental updates would warrant a real sparse store (e.g., Qdrant sparse
vectors or Elasticsearch), but for the corpus sizes this project targets
the in-memory index is more than adequate.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric. Shared by index and query."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    bm25: BM25Okapi
    documents: list[Document]

    def search(self, query: str, top_k: int) -> list[tuple[int, float, Document]]:
        """Return up to ``top_k`` (index, score, document) tuples sorted by score desc."""
        tokens = tokenize(query)
        if not tokens or not self.documents:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        results: list[tuple[int, float, Document]] = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                break
            results.append((idx, float(score), self.documents[idx]))
        return results


def build_bm25_index(documents: list[Document], path: Path) -> BM25Index:
    """Build a BM25 index over ``documents`` and persist it to ``path``."""
    tokenized = [tokenize(doc.page_content) for doc in documents]
    if not any(tokenized):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"corpus": [], "documents": []}))
        empty = BM25Okapi([["__placeholder__"]])
        return BM25Index(bm25=empty, documents=[])

    bm25 = BM25Okapi(tokenized)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "corpus": tokenized,
        "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
    }
    path.write_bytes(pickle.dumps(payload))
    return BM25Index(bm25=bm25, documents=list(documents))


def load_bm25_index(path: Path) -> BM25Index | None:
    """Load a previously persisted index. Returns ``None`` if the index is missing or empty."""
    if not path.exists():
        return None
    payload = pickle.loads(path.read_bytes())
    corpus = payload.get("corpus") or []
    raw_docs = payload.get("documents") or []
    if not corpus or not raw_docs:
        return None
    documents = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw_docs]
    return BM25Index(bm25=BM25Okapi(corpus), documents=documents)
