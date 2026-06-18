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
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        collection = self.client.get_collection(name=collection_name_for_model(model_name))
        all_records: list[dict[str, Any]] = []
        all_embeddings: list[list[float]] = []
        offset = 0

        while True:
            results = collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "metadatas", "documents"],
            )

            if not results.get("ids"):
                break

            for meta, emb, doc in zip(
                results["metadatas"], results["embeddings"], results["documents"], strict=True
            ):
                all_records.append({**meta, "text": doc})
                all_embeddings.append(emb)

            offset += batch_size

            if len(results["ids"]) < batch_size:
                break

        embeddings = np.array(all_embeddings, dtype=np.float32) if all_embeddings else np.empty((0, 0), dtype=np.float32)
        return all_records, embeddings

