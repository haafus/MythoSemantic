import logging
import threading
from dataclasses import dataclass

import numpy as np

from corpus.utils import chunk_id
from model_registry import model_name_for_key

logger = logging.getLogger(__name__)


@dataclass
class ModelIndex:
    model_name: str
    items: list[dict]
    normalized_matrix: np.ndarray
    id_to_index: dict[str, int]


class EmbeddingIndexService:
    def __init__(self):
        self._index: ModelIndex | None = None
        self._index_lock = threading.Lock()
        self._encoder = None

    def get_index(self, model_name: str) -> ModelIndex:
        with self._index_lock:
            if self._index is not None and self._index.model_name == model_name:
                return self._index

            logger.info(f"Loading index for model: {model_name}")
            self._index = self._load_index(model_name)
            return self._index

    def get_point(self, model_key: str, text_id: str, chunk_index: int,
                  top_k: int = 1) -> list[dict]:
        index = self.get_index(model_name_for_key(model_key))
        item_index = self._resolve_index(index, text_id, chunk_index)
        if item_index is None:
            return []
        query_vector = index.normalized_matrix[item_index]
        similarities = index.normalized_matrix @ query_vector
        return self._top_results(index, similarities, top_k)

    def search(self, model_key: str, query: str, top_k: int = 20) -> list[dict]:
        model_name = model_name_for_key(model_key)
        index = self.get_index(model_name)
        query_embedding = self._encode_query(model_name, query)
        similarities = index.normalized_matrix @ query_embedding
        return self._top_results(index, similarities, top_k)

    def warmup(self, model_key: str) -> None:
        model_name = model_name_for_key(model_key)
        self.get_index(model_name)
        self._encode_query(model_name, "warmup")

    def _load_index(self, model_name: str) -> ModelIndex:
        from embeddings import chroma_manager

        items, embeddings = chroma_manager.get_collection(model_name).load_data()

        id_to_index = {chunk_id(item["id"], item["chunk_index"]): idx for idx, item in enumerate(items)}

        return ModelIndex(
            model_name=model_name,
            items=items,
            normalized_matrix=embeddings,
            id_to_index=id_to_index,
        )

    @staticmethod
    def _resolve_index(index: ModelIndex, text_id: str, chunk_index: int) -> int | None:
        return index.id_to_index.get(chunk_id(text_id, chunk_index))

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

    @staticmethod
    def _top_results(index: ModelIndex, similarities: np.ndarray, limit: int) -> list[dict]:
        limit = min(limit, len(index.items))
        if limit <= 0:
            return []

        candidate_indices = np.argpartition(-similarities, limit - 1)[:limit]
        candidate_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]

        results = []
        for idx in candidate_indices:
            similarity = float(similarities[idx])
            if not np.isfinite(similarity):
                continue
            results.append({**index.items[int(idx)], "similarity_score": similarity})

        return results


embedding_index_service = EmbeddingIndexService()
