import hashlib
import logging
import re
from pathlib import Path
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


def save_to_chroma_collection(
    collection: chromadb.Collection,
    ids: list[str],
    embeddings: np.ndarray | list[list[float]],
    metadatas: list[dict[str, Any]],
    documents: list[str],
):
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def _is_missing_collection_error(error: Exception) -> bool:
    message = str(error).lower()
    return "does not exist" in message or "doesn't exist" in message or "not found" in message


def _is_readonly_database_error(error: Exception) -> bool:
    message = str(error).lower()
    return "readonly database" in message or "read-only database" in message


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(settings.embeddings_dir))


def get_available_models() -> list[str]:
    if not Path(settings.embeddings_dir).exists():
        return []
    return sorted(
        col.metadata["model"] for col in _client().list_collections()
    )


def load_data(model_name: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    collection = _client().get_collection(name=collection_name_for_model(model_name))
    results = collection.get(include=["embeddings", "metadatas", "documents"])

    records = [
        {**meta, "text": doc}
        for meta, doc in zip(results["metadatas"], results["documents"], strict=True)
    ]
    embeddings = np.array(results["embeddings"], dtype=np.float32) if records else np.empty((0, 0), dtype=np.float32)
    return records, embeddings


def delete_collection(client: chromadb.PersistentClient, collection_name: str) -> bool:
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception as error:
        if _is_missing_collection_error(error):
            return False
        if _is_readonly_database_error(error):
            raise RuntimeError(
                "Chroma database is read-only. Move chroma_path to a writable directory "
                "or fix permissions for the Chroma DB files."
            ) from error
        raise


