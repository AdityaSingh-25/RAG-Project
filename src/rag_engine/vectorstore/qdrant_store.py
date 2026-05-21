from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from rag_engine.config.settings import Settings
from rag_engine.embeddings.factory import build_embeddings


@lru_cache(maxsize=4)
def build_vectorstore(
    qdrant_url: str, collection_name: str, embedding_model: str
) -> QdrantVectorStore:
    embeddings = build_embeddings(embedding_model)
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection_name):
        vector_size = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def index_documents(documents: list[Document], settings: Settings) -> QdrantVectorStore:
    store = build_vectorstore(
        qdrant_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )
    store.add_documents(documents)
    return store
