"""Chroma query and collection removal, removed from CLI.

Usage example (query):
    from embedding.builder import EmbeddingBuilder
    builder = EmbeddingBuilder(embedding_model="BAAI/bge-m3")
    results = builder.query_chroma("creation of the world", top_k=5)
    for r in results:
        print(f"Score: {1 - r['distance']:.3f}  {r['metadata'].get('tradition')}  {r['document'][:100]}")
"""

import numpy as np
import chromadb
from typing import Any


def query_chroma(builder, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Was EmbeddingBuilder.query_chroma()."""
    from embedding.chroma_manager import collection_name_for_model

    collection_name = collection_name_for_model(builder._models.model_name)
    try:
        collection = builder.chroma_client.get_collection(name=collection_name)
    except Exception as err:
        raise RuntimeError(f"Collection '{collection_name}' not found in ChromaDB.") from err

    query_embedding = builder._generate_embeddings([query])[0]
    return query_chroma_collection(collection=collection, query_embedding=query_embedding.tolist(), top_k=top_k)


def query_chroma_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    formatted = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0], strict=False):
        formatted.append(
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return formatted


def delete_chroma_collection(model: str) -> None:
    """Was CLI command `mytho embeddings remove --model <model>`."""
    from embedding.chroma_manager import collection_name_for_model, delete_collection
    from model_registry import resolve_embedding_model
    from settings import settings

    model_name = resolve_embedding_model(model)
    collection = collection_name_for_model(model_name)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    deleted = delete_collection(client, collection)
    print(f"Collection '{collection}' {'deleted' if deleted else 'does not exist'}")
