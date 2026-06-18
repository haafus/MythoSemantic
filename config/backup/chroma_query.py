"""Chroma query functionality, removed from CLI.

Usage example:
    from embedding.builder import EmbeddingBuilder
    builder = EmbeddingBuilder(embedding_model="BAAI/bge-m3")
    results = builder.query_chroma("creation of the world", top_k=5)
    for r in results:
        print(f"Score: {1 - r['distance']:.3f}  {r['metadata'].get('tradition')}  {r['document'][:100]}")
"""

import numpy as np
import chromadb
from typing import Any


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
