from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from rag_engine.config.settings import Settings
from rag_engine.embeddings.factory import build_cached_embeddings


def build_vectorstore_for(settings: Settings, embeddings: Any | None = None) -> QdrantVectorStore:
    """Build a Qdrant vectorstore using cached embeddings.

    ``embeddings`` is injectable so tests can pass a stub; otherwise the
    function wires in the SQLite-cached HuggingFace embeddings.
    """
    embeddings = embeddings or build_cached_embeddings(settings)
    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(settings.qdrant_collection):
        vector_size = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def index_documents(documents: list[Document], settings: Settings) -> QdrantVectorStore:
    store = build_vectorstore_for(settings)
    store.add_documents(documents)
    return store


# Batch size for the MatchAny scroll query. Chosen to keep the request body
# small enough that Qdrant's request size limits won't bite on huge ingests
# while amortising the per-call overhead.
_HASH_LOOKUP_BATCH = 256


def existing_content_hashes(
    settings: Settings, candidate_hashes: Iterable[str]
) -> set[str]:
    """Subset of ``candidate_hashes`` already present in the collection.

    Skipped if the collection does not exist yet (nothing to be a duplicate
    of). Uses ``MatchAny`` so a single Qdrant scroll call covers a whole
    batch of hashes, then paginates if the result set is larger than one
    page.
    """
    candidates = [h for h in candidate_hashes if h]
    if not candidates:
        return set()
    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(settings.qdrant_collection):
        return set()

    seen: set[str] = set()
    # Filter request bodies grow linearly with len(candidates), and very large
    # `MatchAny` lists hit Qdrant's request-size cap. Chunk the candidate set.
    for chunk_start in range(0, len(candidates), _HASH_LOOKUP_BATCH):
        chunk = candidates[chunk_start : chunk_start + _HASH_LOOKUP_BATCH]
        flt = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.content_hash",
                    match=rest.MatchAny(any=chunk),
                )
            ]
        )
        offset: Any = None
        while True:
            batch, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=flt,
                limit=256,
                with_payload=True,
                offset=offset,
            )
            for point in batch:
                metadata = (point.payload or {}).get("metadata") or {}
                digest = metadata.get("content_hash")
                if isinstance(digest, str):
                    seen.add(digest)
            if offset is None:
                break
    return seen
