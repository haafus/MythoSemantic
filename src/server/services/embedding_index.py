import logging

import numpy as np

from corpus.utils import chunk_id
from model_registry import model_name_for_key

logger = logging.getLogger(__name__)


class EmbeddingIndexService:
    def __init__(self):
        self._encoder = None

    def get_point(self, model_key: str, text_id: str, chunk_index: int,
                  top_k: int = 1) -> list[dict]:
        collection = self._get_collection(model_key)
        cid = chunk_id(text_id, chunk_index)
        point = collection.get(ids=[cid], include=["embeddings"])
        if not point["ids"]:
            return []
        return self._query(collection, point["embeddings"][0], top_k)

    def search(self, model_key: str, query: str, top_k: int = 20) -> list[dict]:
        model_name = model_name_for_key(model_key)
        collection = self._get_collection(model_key)
        embedding = self._encode_query(model_name, query)
        return self._query(collection, embedding.tolist(), top_k)

    def _get_collection(self, model_key: str):
        from embeddings import chroma_manager
        return chroma_manager.get_collection(model_name_for_key(model_key))

    def _query(self, collection, embedding, top_k: int) -> list[dict]:
        raw = collection.query(
            query_embeddings=[embedding], n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        return [
            {**meta, "id": meta.pop("text_id"), "text": doc,
             "similarity_score": round(1 - dist, 6)}
            for meta, doc, dist in zip(
                raw["metadatas"][0], raw["documents"][0],
                raw["distances"][0], strict=True,
            )
        ]

    def _encode_query(self, model_name: str, query: str) -> np.ndarray:
        if self._encoder is None:
            from embeddings.model_manager import EmbeddingEncoder
            self._encoder = EmbeddingEncoder()
        self._encoder.load(model_name)
        raw = self._encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(raw[0], dtype=np.float32)


embedding_index_service = EmbeddingIndexService()
