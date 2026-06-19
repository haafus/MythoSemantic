from typing import Any

import chromadb
import numpy as np

from embeddings.chroma_manager import collection_name_for_model
from settings import settings


class EmbeddingDataLoader:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.embeddings_dir))

    def get_available_models(self) -> list[str]:
        return sorted(
            col.metadata["model"] for col in self.client.list_collections()
        )

    def load_data(self, model_name: str) -> tuple[list[dict[str, Any]], np.ndarray]:
        collection = self.client.get_collection(name=collection_name_for_model(model_name))
        results = collection.get(include=["embeddings", "metadatas", "documents"])

        records = [
            {**meta, "text": doc}
            for meta, doc in zip(results["metadatas"], results["documents"], strict=True)
        ]
        embeddings = np.array(results["embeddings"], dtype=np.float32) if records else np.empty((0, 0), dtype=np.float32)
        return records, embeddings
