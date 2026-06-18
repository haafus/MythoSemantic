import logging
from typing import Any

import chromadb
import numpy as np

from embeddings.chroma_manager import collection_name_for_model
from settings import settings

logger = logging.getLogger(__name__)


class EmbeddingDataLoader:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.embeddings_dir))

    def _list_collection_names(self) -> list[str]:
        try:
            return sorted(col.name for col in self.client.list_collections())
        except Exception as e:
            logger.warning(f"Failed to list Chroma collections: {e}")
            return []

    def _resolve_collection_names(self, model_name: str | None = None) -> list[str]:
        if model_name:
            return [collection_name_for_model(model_name)]
        return self._list_collection_names()

    def _iter_collections(self, model_name: str | None = None):
        names = self._resolve_collection_names(model_name=model_name)
        if not names:
            raise RuntimeError("Model-based Chroma collections not found")

        for name in names:
            try:
                yield self.client.get_collection(name=name)
            except Exception as e:
                logger.warning(f"Failed to get collection '{name}': {e}")

    def load_data(
        self, model_name: str | None = None, batch_size: int = 5000, max_records: int | None = None
    ) -> list[dict[str, Any]]:
        all_data: list[dict[str, Any]] = []

        for collection in self._iter_collections(model_name=model_name):
            offset = 0

            while True:
                if max_records and len(all_data) >= max_records:
                    logger.info(f"Record limit reached: {max_records}")
                    return all_data[:max_records]

                try:
                    results = collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=["embeddings", "metadatas", "documents"],
                    )
                except Exception:
                    logger.exception("Failed to fetch data from '%s' at offset %d", collection.name, offset)
                    break

                if not results.get("ids"):
                    break

                batch_data = self._process_batch(results)
                all_data.extend(batch_data)

                offset += batch_size

                if len(results["ids"]) < batch_size:
                    break

        return all_data

    def _process_batch(self, results: dict) -> list[dict[str, Any]]:
        batch_data = []
        ids = results.get("ids", [])
        embeddings = results.get("embeddings", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])

        for i, doc_id in enumerate(ids):
            try:
                if i >= len(embeddings) or embeddings[i] is None:
                    continue

                meta = metadatas[i] if i < len(metadatas) else {}
                doc = documents[i] if i < len(documents) else ""

                embedding = np.array(embeddings[i]) if isinstance(embeddings[i], list) else embeddings[i]

                batch_data.append(
                    {
                        "id": meta.get("text_id", doc_id),
                        "tradition": meta.get("tradition", "unknown"),
                        "major_tradition": meta.get("major_tradition", "unknown"),
                        "chunk_index": meta.get("chunk_index", 0),
                        "embedding": embedding,
                        "text": doc,
                        "filename": meta.get("filename", "unknown"),
                        "url": meta.get("url", ""),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to process document {doc_id}: {e}")
                continue

        return batch_data

    def get_available_models(self) -> list[str]:
        models: set[str] = set()

        try:
            if not self._resolve_collection_names():
                return []

            for collection in self._iter_collections():
                model = (collection.metadata or {}).get("model")
                if model:
                    models.add(model)

            return sorted(models)
        except Exception:
            logger.exception("Failed to get available models")
            return []

    def close(self):
        if hasattr(self, "client"):
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
