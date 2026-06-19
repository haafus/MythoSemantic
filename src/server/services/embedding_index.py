import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from embeddings.model_cache import EmbeddingModelCache
from model_registry import key_to_model

logger = logging.getLogger(__name__)

MAX_PREVIEW_CHARS = 700
MAX_CACHED_INDEXES = 3


@dataclass
class ModelIndex:
    model_name: str
    items: list[dict]
    normalized_matrix: np.ndarray
    id_to_index: dict[str, int]


class EmbeddingIndexService:
    def __init__(self):
        self._indexes: OrderedDict[str, ModelIndex] = OrderedDict()
        self._index_lock = threading.RLock()
        self._model_cache = EmbeddingModelCache()

    def get_index(self, model_key: str) -> ModelIndex:
        model_name = key_to_model(model_key)

        with self._index_lock:
            if model_name in self._indexes:
                self._indexes.move_to_end(model_name)
                return self._indexes[model_name]

            index = self._load_index(model_name)
            self._indexes[model_name] = index

            while len(self._indexes) > MAX_CACHED_INDEXES:
                evicted_name, _ = self._indexes.popitem(last=False)
                logger.info(f"Evicted index cache for model: {evicted_name}")

            return index

    def get_point(self, model_key: str, point_id: str, chunk_index: int | None = None) -> dict:
        index = self.get_index(model_key)
        item_index = index.id_to_index.get(self._point_key(point_id, chunk_index))
        if item_index is None:
            item_index = index.id_to_index.get(str(point_id))
        if item_index is None:
            raise KeyError(point_id)
        item = index.items[item_index]

        return {
            "id": item["text_id"],
            "text": item["text"],
            "tradition": item["tradition"],
            "chunk_index": item["chunk_index"],
            "book_title": item["text_id"],
            "model": index.model_name,
            "metadata": {
                "filename": item["filename"],
                "major_tradition": item["major_tradition"],
                "url": item["url"],
            },
        }

    def get_neighbors(self, model_key: str, point_id: str, n: int = 10, chunk_index: int | None = None) -> list[dict]:
        index = self.get_index(model_key)
        item_index = index.id_to_index.get(self._point_key(point_id, chunk_index))
        if item_index is None:
            item_index = index.id_to_index.get(str(point_id))
        if item_index is None:
            raise KeyError(point_id)

        query_vector = index.normalized_matrix[item_index]
        similarities = index.normalized_matrix @ query_vector
        similarities[item_index] = -np.inf
        return self._top_results(index, similarities, n)

    def search(self, model_key: str, query: str, top_k: int = 20) -> list[dict]:
        model_name = key_to_model(model_key)
        index = self.get_index(model_name)
        query_embedding = self._model_cache.encode(model_name, [query])
        similarities = index.normalized_matrix @ query_embedding
        return self._top_results(index, similarities, top_k)

    def _load_index(self, model_name: str) -> ModelIndex:
        from projections.loader import EmbeddingDataLoader

        items, embeddings = EmbeddingDataLoader().load_data(model_name=model_name)
        if not items:
            raise KeyError(f"No embedding data found for {model_name}")

        normalized_matrix = self._normalize_matrix(embeddings)
        id_to_index: dict[str, int] = {}
        for idx, item in enumerate(items):
            point_id = item["text_id"]
            id_to_index.setdefault(point_id, idx)
            id_to_index[self._point_key(point_id, item["chunk_index"])] = idx

        return ModelIndex(
            model_name=model_name,
            items=items,
            normalized_matrix=normalized_matrix,
            id_to_index=id_to_index,
        )

    @staticmethod
    def _point_key(point_id: str, chunk_index: int | None = None) -> str:
        if chunk_index is None:
            return str(point_id)
        return f"{point_id}::{chunk_index}"

    def warmup(self, model_key: str) -> None:
        self.get_index(model_key)
        self._model_cache.encode(model_key, ["warmup"])

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

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

            item = index.items[int(idx)]
            text = item.get("text", "") or ""
            preview = text[:MAX_PREVIEW_CHARS]
            if len(text) > MAX_PREVIEW_CHARS:
                preview += "..."

            results.append(
                {
                    "id": item["text_id"],
                    "tradition": item["tradition"],
                    "major_tradition": item["major_tradition"],
                    "chunk_index": item["chunk_index"],
                    "similarity_score": similarity,
                    "distance": 1 - similarity,
                    "text": text,
                    "text_preview": preview,
                    "filename": item["filename"],
                    "book_title": item["text_id"],
                }
            )

        return results


embedding_index_service = EmbeddingIndexService()
