import hashlib
import logging
import re
from typing import Any

import chromadb

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


def is_model_collection_name(collection_name: str) -> bool:
    if not collection_name:
        return False
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,53}_[0-9a-f]{8}", collection_name))



def save_to_chroma_collection(
    collection: chromadb.Collection,
    ids: list[str],
    embeddings: list[list[float]],
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


