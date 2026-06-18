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

    def get_available_models(self) -> list[str]:
        return sorted(
            col.metadata["model"] for col in self.client.list_collections()
        )

    def load_data(
        self, model_name: str, batch_size: int = 5000
    ) -> list[dict[str, Any]]:
        collection = self.client.get_collection(name=collection_name_for_model(model_name))
        all_data: list[dict[str, Any]] = []
        offset = 0

        while True:
            results = collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "metadatas", "documents"],
            )

            if not results.get("ids"):
                break

            all_data.extend(self._process_batch(results))
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
                meta = metadatas[i]
                doc = documents[i]
                embedding = np.array(embeddings[i])

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

