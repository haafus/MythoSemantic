import hashlib
import logging
import re
from typing import Any

import chromadb
import numpy as np

from settings import settings

logger = logging.getLogger(__name__)

class ChromaStore:
    _MAX_COLLECTION_NAME = 63
    _COLLECTION_HASH_LEN = 8

    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(settings.embeddings_dir))

    @staticmethod
    def _collection_name(model_name: str) -> str:
        raw_name = str(model_name or "unknown").strip()
        digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:ChromaStore._COLLECTION_HASH_LEN]

        safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", raw_name).strip("_-").lower()
        safe_name = re.sub(r"_+", "_", safe_name)
        if not safe_name:
            safe_name = "model"

        suffix = f"_{digest}"
        max_base_len = ChromaStore._MAX_COLLECTION_NAME - len(suffix)
        safe_name = safe_name[:max_base_len].strip("_-")
        if len(safe_name) < 3:
            safe_name = f"{safe_name}_model".strip("_-")
            safe_name = safe_name[:max_base_len].strip("_-")
        if len(safe_name) < 3:
            safe_name = "model"

        return f"{safe_name}{suffix}"

    def get_or_create_collection(self, model_name: str, **kwargs) -> chromadb.Collection:
        return self._client.get_or_create_collection(name=self._collection_name(model_name), **kwargs)

    def get_collection(self, model_name: str) -> chromadb.Collection:
        return self._client.get_collection(name=self._collection_name(model_name))

    def list_collections(self):
        return self._client.list_collections()

    def get_available_models(self) -> list[str]:
        return sorted(
            col.metadata["model"] for col in self._client.list_collections()
        )

    def load_data(self, model_name: str) -> tuple[list[dict[str, Any]], np.ndarray]:
        collection = self._client.get_collection(name=self._collection_name(model_name))
        results = collection.get(include=["embeddings", "metadatas", "documents"])

        records = [
            {**meta, "text": doc}
            for meta, doc in zip(results["metadatas"], results["documents"], strict=True)
        ]
        embeddings = np.array(results["embeddings"], dtype=np.float32) if records else np.empty((0, 0), dtype=np.float32)
        return records, embeddings

    def delete_collection(self, model_name: str) -> bool:
        try:
            self._client.delete_collection(name=self._collection_name(model_name))
            return True
        except Exception as error:
            msg = str(error).lower()
            if "does not exist" in msg or "doesn't exist" in msg or "not found" in msg:
                return False
            if "readonly database" in msg or "read-only database" in msg:
                raise RuntimeError(
                    "Chroma database is read-only. Move chroma_path to a writable directory "
                    "or fix permissions for the Chroma DB files."
                ) from error
            raise
