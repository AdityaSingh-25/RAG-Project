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
