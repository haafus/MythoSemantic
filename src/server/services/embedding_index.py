import logging
import threading
from dataclasses import dataclass

import numpy as np

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

    def get_point(self, model_key: str, point_id: str, chunk_index: int | None = None,
                  neighbors: int = 0, offset: int = 0) -> list[dict]:
        index = self.get_index(model_name_for_key(model_key))
        item_index = self._resolve_index(index, point_id, chunk_index)
        query_vector = index.normalized_matrix[item_index]
        similarities = index.normalized_matrix @ query_vector
        return self._top_results(index, similarities, 1 + neighbors, offset)

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
        if not items:
            raise KeyError(f"No embedding data found for {model_name}")

        id_to_index: dict[str, int] = {}
        for idx, item in enumerate(items):
            point_id = item["id"]
            id_to_index.setdefault(point_id, idx)
            id_to_index[self._point_key(point_id, item["chunk_index"])] = idx

        return ModelIndex(
            model_name=model_name,
            items=items,
            normalized_matrix=embeddings,
            id_to_index=id_to_index,
        )

    @staticmethod
    def _point_key(point_id: str, chunk_index: int | None = None) -> str:
        if chunk_index is None:
            return str(point_id)
        return f"{point_id}::{chunk_index}"

    @staticmethod
    def _resolve_index(index: ModelIndex, point_id: str, chunk_index: int | None = None) -> int:
        key = EmbeddingIndexService._point_key(point_id, chunk_index)
        item_index = index.id_to_index.get(key)
        if item_index is None:
            raise KeyError(point_id)
        return item_index

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
    def _top_results(index: ModelIndex, similarities: np.ndarray, limit: int, offset: int = 0) -> list[dict]:
        total_needed = min(offset + limit, len(index.items))
        if total_needed <= 0:
            return []

        candidate_indices = np.argpartition(-similarities, total_needed - 1)[:total_needed]
        candidate_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]
        candidate_indices = candidate_indices[offset:]

        results = []
        for idx in candidate_indices:
            similarity = float(similarities[idx])
            if not np.isfinite(similarity):
                continue
            results.append({**index.items[int(idx)], "similarity_score": similarity})

        return results


embedding_index_service = EmbeddingIndexService()
