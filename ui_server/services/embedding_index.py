from dataclasses import dataclass
import csv
import hashlib
import logging
import re
import threading
from typing import Dict, List, Optional

import numpy as np

from embedding_analyzer.config import get_chroma_path
from ui_server.services.models import get_model_output_dir, key_to_model


MAX_PREVIEW_CHARS = 700
MAX_CHROMA_COLLECTION_NAME = 63
MODEL_COLLECTION_HASH_LEN = 8
logger = logging.getLogger(__name__)


@dataclass
class ModelIndex:
    model_name: str
    items: List[Dict]
    matrix: np.ndarray
    normalized_matrix: np.ndarray
    id_to_index: Dict[str, int]


class EmbeddingIndexService:
    def __init__(self):
        self._indexes: Dict[str, ModelIndex] = {}
        self._point_records: Dict[str, Dict[str, Dict]] = {}
        self._search_models: Dict[str, object] = {}
        self._chroma_client = None
        self._collections: Dict[str, object] = {}
        self._index_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._chroma_lock = threading.RLock()

    def get_index(self, model_key: str) -> ModelIndex:
        model_name = key_to_model(model_key)

        with self._index_lock:
            if model_name in self._indexes:
                return self._indexes[model_name]

            index = self._load_index(model_name)
            self._indexes[model_name] = index
            return index

    def get_point(self, model_key: str, point_id: str, chunk_index: Optional[int] = None) -> Dict:
        model_name = key_to_model(model_key)
        try:
            item = self._get_chroma_point(model_name, point_id, chunk_index)
        except Exception:
            logger.exception("Failed to load point from Chroma; falling back to local records")
            item = None

        if item is None:
            item = self._get_point_record(model_name, point_id, chunk_index)
        if item is None:
            index = self.get_index(model_name)
            item_index = index.id_to_index.get(self._point_key(point_id, chunk_index))
            if item_index is None:
                item_index = index.id_to_index.get(str(point_id))
            if item_index is None:
                raise KeyError(point_id)
            item = index.items[item_index]

        if not item:
            raise KeyError(point_id)

        return {
            "id": str(item.get("id")),
            "text": item.get("text", ""),
            "tradition": item.get("tradition", "Unknown"),
            "chunk_index": item.get("chunk_index", 0),
            "book_title": item.get("filename", "") or item.get("id", ""),
            "model": item.get("model", model_name),
            "metadata": {
                "filename": item.get("filename", ""),
                "major_tradition": item.get("major_tradition", ""),
                "doc_type": item.get("doc_type", ""),
                "language": item.get("language", ""),
                "url": item.get("url", ""),
            },
        }

    def get_neighbors(
        self,
        model_key: str,
        point_id: str,
        n: int = 10,
        chunk_index: Optional[int] = None,
        exclude_same_tradition: bool = False,
    ) -> List[Dict]:
        model_name = key_to_model(model_key)
        try:
            item = self._get_chroma_point(model_name, point_id, chunk_index, include_embedding=True)
            if item and item.get("embedding") is not None:
                query_vector = self._normalize_vector(np.asarray(item["embedding"], dtype=np.float32))
                exclude_tradition = item.get("tradition") if exclude_same_tradition else None
                raw_results = self._query_chroma(
                    model_name,
                    query_vector,
                    n + 8,
                    where=self._different_tradition_where(exclude_tradition),
                )
                return self._format_chroma_results(
                    raw_results,
                    model_name,
                    query_vector=query_vector,
                    limit=n,
                    exclude_key=self._point_key(point_id, chunk_index),
                    exclude_tradition=exclude_tradition,
                )
        except Exception:
            logger.exception("Chroma neighbor search failed; falling back to in-memory index")

        index = self.get_index(model_key)
        item_index = index.id_to_index.get(self._point_key(point_id, chunk_index))
        if item_index is None:
            item_index = index.id_to_index.get(str(point_id))
        if item_index is None:
            raise KeyError(point_id)

        query_vector = index.normalized_matrix[item_index]
        similarities = index.normalized_matrix @ query_vector
        similarities[item_index] = -np.inf
        if exclude_same_tradition:
            exclude_tradition = index.items[item_index].get("tradition")
            if exclude_tradition is not None:
                normalized_exclude = self._normalize_tradition(exclude_tradition)
                for idx, item in enumerate(index.items):
                    if self._normalize_tradition(item.get("tradition")) == normalized_exclude:
                        similarities[idx] = -np.inf
        return self._top_results(index, similarities, n)

    def search(self, model_key: str, query: str, top_k: int = 20) -> List[Dict]:
        model_name = key_to_model(model_key)
        query_embedding = self._embed_query(model_name, query)
        query_embedding = self._normalize_vector(query_embedding)

        try:
            raw_results = self._query_chroma(model_name, query_embedding, top_k)
            results = self._format_chroma_results(
                raw_results,
                model_name,
                query_vector=query_embedding,
                limit=top_k,
            )
            if results:
                return results
        except Exception:
            logger.exception("Chroma semantic search failed; falling back to in-memory index")

        index = self.get_index(model_name)
        similarities = index.normalized_matrix @ query_embedding
        return self._top_results(index, similarities, top_k)

    def warmup(self, model_key: str) -> None:
        model_name = key_to_model(model_key)
        self._get_collection(model_name).count()
        self._embed_query(model_name, "warmup")

    def _get_chroma_client(self):
        with self._chroma_lock:
            if self._chroma_client is None:
                import chromadb

                self._chroma_client = chromadb.PersistentClient(path=get_chroma_path())
            return self._chroma_client

    def _get_collection(self, model_name: str):
        with self._chroma_lock:
            if model_name not in self._collections:
                client = self._get_chroma_client()
                self._collections[model_name] = client.get_collection(
                    name=self._collection_name_for_model(model_name)
                )
            return self._collections[model_name]

    @staticmethod
    def _collection_name_for_model(model_name: str) -> str:
        raw_name = str(model_name or "unknown").strip()
        digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:MODEL_COLLECTION_HASH_LEN]

        safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", raw_name).strip("_-").lower()
        safe_name = re.sub(r"_+", "_", safe_name)
        if not safe_name:
            safe_name = "model"

        suffix = f"_{digest}"
        max_base_len = MAX_CHROMA_COLLECTION_NAME - len(suffix)
        safe_name = safe_name[:max_base_len].strip("_-")
        if len(safe_name) < 3:
            safe_name = f"{safe_name}_model".strip("_-")
            safe_name = safe_name[:max_base_len].strip("_-")
        if len(safe_name) < 3:
            safe_name = "model"

        return f"{safe_name}{suffix}"

    @staticmethod
    def _chunk_index_value(chunk_index: Optional[int]) -> Optional[int]:
        if chunk_index is None or chunk_index == "":
            return None
        try:
            return int(chunk_index)
        except (TypeError, ValueError):
            return None

    def _point_where(self, point_id: str, chunk_index: Optional[int] = None) -> Dict:
        point_filter = {"text_id": str(point_id)}
        chunk_value = self._chunk_index_value(chunk_index)
        if chunk_value is None:
            return point_filter
        return {"$and": [point_filter, {"chunk_index": chunk_value}]}

    def _get_chroma_point(
        self,
        model_name: str,
        point_id: str,
        chunk_index: Optional[int] = None,
        include_embedding: bool = False,
    ) -> Optional[Dict]:
        collection = self._get_collection(model_name)
        include = ["documents", "metadatas"]
        if include_embedding:
            include.append("embeddings")

        result = collection.get(
            where=self._point_where(point_id, chunk_index),
            limit=1,
            include=include,
        )
        ids = result.get("ids") or []
        if not ids:
            return None

        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []
        embeddings = result.get("embeddings")
        item = self._item_from_metadata(
            ids[0],
            metadatas[0] if metadatas else {},
            documents[0] if documents else "",
            model_name,
        )
        if include_embedding and embeddings is not None and len(embeddings) > 0:
            item["embedding"] = embeddings[0]
        return item

    def _query_chroma(
        self,
        model_name: str,
        query_vector: np.ndarray,
        top_k: int,
        where: Optional[Dict] = None,
    ) -> Dict:
        collection = self._get_collection(model_name)
        count = collection.count()
        if count <= 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]], "embeddings": [[]]}

        n_results = min(max(int(top_k), 1), count)
        query_kwargs = {}
        if where:
            query_kwargs["where"] = where

        return collection.query(
            query_embeddings=[query_vector.astype(np.float32).tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"],
            **query_kwargs,
        )

    def _load_index(self, model_name: str) -> ModelIndex:
        from embedding_analyzer.analyzer import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer(model_name=model_name)
        items = analyzer.filter_by_model()
        if not items:
            raise RuntimeError(f"No embedding data found for {model_name}")

        matrix = np.stack([item["embedding"] for item in items]).astype(np.float32)
        normalized_matrix = self._normalize_matrix(matrix)
        id_to_index = {}
        for idx, item in enumerate(items):
            point_id = str(item.get("id"))
            id_to_index.setdefault(point_id, idx)
            id_to_index[self._point_key(point_id, item.get("chunk_index"))] = idx

        return ModelIndex(
            model_name=model_name,
            items=items,
            matrix=matrix,
            normalized_matrix=normalized_matrix,
            id_to_index=id_to_index,
        )

    def _get_point_record(self, model_name: str, point_id: str, chunk_index: Optional[int] = None) -> Optional[Dict]:
        if model_name not in self._point_records:
            self._point_records[model_name] = self._load_point_records(model_name)

        records = self._point_records[model_name]
        item = records.get(self._point_key(point_id, chunk_index))
        if item is not None:
            return item
        return records.get(str(point_id))

    @staticmethod
    def _load_point_records(model_name: str) -> Dict[str, Dict]:
        csv_path = get_model_output_dir(model_name) / "embeddings_data.csv"
        if not csv_path.exists():
            return {}

        records: Dict[str, Dict] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                point_id = str(row.get("id", ""))
                if point_id:
                    record = {
                        "id": point_id,
                        "text": row.get("text", ""),
                        "tradition": row.get("tradition", "Unknown"),
                        "major_tradition": row.get("major_tradition", ""),
                        "chunk_index": int(row.get("chunk_index") or 0),
                        "model": row.get("model", model_name),
                        "filename": row.get("filename", ""),
                        "doc_type": row.get("doc_type", ""),
                        "language": row.get("language", ""),
                        "url": row.get("url", ""),
                    }
                    records.setdefault(point_id, record)
                    records[EmbeddingIndexService._point_key(point_id, record["chunk_index"])] = record

        return records

    @staticmethod
    def _point_key(point_id: str, chunk_index: Optional[int] = None) -> str:
        if chunk_index is None or chunk_index == "":
            return str(point_id)
        return f"{point_id}::{chunk_index}"

    @staticmethod
    def _normalize_tradition(tradition) -> str:
        return str(tradition or "Unknown").strip().casefold()

    @staticmethod
    def _different_tradition_where(tradition) -> Optional[Dict]:
        if tradition is None:
            return None
        return {"tradition": {"$ne": str(tradition)}}

    @staticmethod
    def _item_from_metadata(chroma_id: str, metadata: Optional[Dict], document: str, model_name: str) -> Dict:
        metadata = metadata or {}
        point_id = metadata.get("text_id") or chroma_id
        return {
            "id": str(point_id),
            "tradition": metadata.get("tradition", "Unknown"),
            "major_tradition": metadata.get("major_tradition", ""),
            "chunk_index": int(metadata.get("chunk_index") or 0),
            "model": metadata.get("model", model_name),
            "filename": metadata.get("filename", ""),
            "doc_type": metadata.get("doc_type", ""),
            "language": metadata.get("language", ""),
            "url": metadata.get("url", ""),
            "text": document or "",
        }

    @classmethod
    def _similarity_from_embedding(
        cls,
        embedding,
        query_vector: Optional[np.ndarray],
        distance: Optional[float] = None,
    ) -> float:
        if query_vector is not None and embedding is not None:
            vector = np.asarray(embedding, dtype=np.float32)
            vector = cls._normalize_vector(vector)
            return float(np.dot(vector, query_vector))

        if distance is None:
            return 0.0
        return float(1 - distance)

    def _format_chroma_results(
        self,
        raw_results: Dict,
        model_name: str,
        query_vector: Optional[np.ndarray] = None,
        limit: int = 20,
        exclude_key: Optional[str] = None,
        exclude_tradition: Optional[str] = None,
    ) -> List[Dict]:
        ids = self._first_chroma_batch(raw_results, "ids")
        documents = self._first_chroma_batch(raw_results, "documents")
        metadatas = self._first_chroma_batch(raw_results, "metadatas")
        distances = self._first_chroma_batch(raw_results, "distances")
        embeddings = self._first_chroma_batch(raw_results, "embeddings")

        results = []
        for index, chroma_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            document = documents[index] if index < len(documents) else ""
            distance = distances[index] if index < len(distances) else None
            embedding = embeddings[index] if index < len(embeddings) else None
            item = self._item_from_metadata(chroma_id, metadata, document, model_name)

            point_key = self._point_key(item["id"], item.get("chunk_index"))
            if exclude_key and point_key == exclude_key:
                continue
            if exclude_tradition is not None and (
                self._normalize_tradition(item.get("tradition")) == self._normalize_tradition(exclude_tradition)
            ):
                continue

            text = item.get("text", "") or ""
            preview = text[:MAX_PREVIEW_CHARS]
            if len(text) > MAX_PREVIEW_CHARS:
                preview += "..."

            similarity = self._similarity_from_embedding(embedding, query_vector, distance)
            if not np.isfinite(similarity):
                continue

            results.append(
                {
                    "id": item["id"],
                    "tradition": item.get("tradition", "Unknown"),
                    "major_tradition": item.get("major_tradition", ""),
                    "chunk_index": item.get("chunk_index", 0),
                    "similarity_score": similarity,
                    "distance": 1 - similarity,
                    "text": text,
                    "text_preview": preview,
                    "filename": item.get("filename", ""),
                    "book_title": item.get("filename", "") or item["id"],
                }
            )

            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _first_chroma_batch(raw_results: Dict, key: str):
        value = raw_results.get(key)
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or len(value) == 0:
                return []
            return value[0] if value.ndim > 1 else value
        if len(value) == 0:
            return []
        first = value[0]
        return [] if first is None else first

    def _embed_query(self, model_name: str, query: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        from embeddings_builder.models_repository import MODELS

        with self._model_lock:
            if model_name not in self._search_models:
                model_path = MODELS.get(model_name, {}).get("path", model_name)
                self._search_models[model_name] = SentenceTransformer(model_path)

            model = self._search_models[model_name]
            embedding = model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def _top_results(index: ModelIndex, similarities: np.ndarray, limit: int) -> List[Dict]:
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
                    "id": str(item.get("id")),
                    "tradition": item.get("tradition", "Unknown"),
                    "major_tradition": item.get("major_tradition", ""),
                    "chunk_index": item.get("chunk_index", 0),
                    "similarity_score": similarity,
                    "distance": 1 - similarity,
                    "text": text,
                    "text_preview": preview,
                    "filename": item.get("filename", ""),
                    "book_title": item.get("filename", "") or item.get("id", ""),
                }
            )

        return results


embedding_index_service = EmbeddingIndexService()
