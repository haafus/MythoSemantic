import hashlib
import logging
import re
from typing import Any

import chromadb
import numpy as np

from settings import settings

logger = logging.getLogger(__name__)

MAX_CHROMA_COLLECTION_NAME = 63
MODEL_COLLECTION_HASH_LEN = 8


def collection_name_for_model(model_name: Any) -> str:
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


def _safe_id_part(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "unknown")).strip("_") or "unknown"


def build_chroma_entries(
    chunks: list[str], info: "CorpusFileInfo", model_name: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    from corpus.corpus_iterator import CorpusFileInfo  # noqa: F811

    text_id_safe = _safe_id_part(info.text_id)
    model_id = _safe_id_part(model_name)

    ids = [f"{text_id_safe}_{model_id}_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "filename": info.filename,
            "tradition": info.tradition,
            "major_tradition": info.major_tradition,
            "chunk_index": i,
            "text_id": info.text_id,
            "url": info.url,
        }
        for i in range(len(chunks))
    ]
    return ids, metadatas


class ChromaStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(settings.embeddings_dir))

    def get_or_create_collection(self, name: str, **kwargs) -> chromadb.Collection:
        return self._client.get_or_create_collection(name=name, **kwargs)

    def get_collection(self, name: str) -> chromadb.Collection:
        return self._client.get_collection(name=name)

    def list_collections(self):
        return self._client.list_collections()

    def get_available_models(self) -> list[str]:
        return sorted(
            col.metadata["model"] for col in self._client.list_collections()
        )

    def load_data(self, model_name: str) -> tuple[list[dict[str, Any]], np.ndarray]:
        collection = self._client.get_collection(name=collection_name_for_model(model_name))
        results = collection.get(include=["embeddings", "metadatas", "documents"])

        records = [
            {**meta, "text": doc}
            for meta, doc in zip(results["metadatas"], results["documents"], strict=True)
        ]
        embeddings = np.array(results["embeddings"], dtype=np.float32) if records else np.empty((0, 0), dtype=np.float32)
        return records, embeddings

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self._client.delete_collection(name=collection_name)
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
