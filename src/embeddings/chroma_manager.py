from typing import Any

import chromadb
import numpy as np

from model_registry import model_to_key
from settings import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(settings.embeddings_dir))
    return _client


class ChromaCollection:
    def __init__(self, collection: chromadb.Collection):
        self._collection = collection

    @property
    def name(self) -> str:
        return self._collection.name

    @property
    def metadata(self) -> dict:
        return self._collection.metadata or {}

    def count(self) -> int:
        return self._collection.count()

    def existing_ids(self) -> set[str]:
        return set(self._collection.get(include=[])["ids"])

    def get(self, **kwargs) -> dict:
        return self._collection.get(**kwargs)

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray | list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents,
        )

    def modify(self, metadata: dict) -> None:
        self._collection.modify(metadata=metadata)

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def load_data(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        results = self._collection.get(include=["embeddings", "metadatas", "documents"])

        records = [
            {**meta, "text": doc}
            for meta, doc in zip(results["metadatas"], results["documents"], strict=True)
        ]
        embeddings = np.array(results["embeddings"], dtype=np.float32) if records else np.empty((0, 0), dtype=np.float32)
        return records, embeddings


def get_or_create_collection(model_name: str, **kwargs) -> ChromaCollection:
    return ChromaCollection(
        _get_client().get_or_create_collection(name=model_to_key(model_name), **kwargs)
    )


def get_collection(model_name: str) -> ChromaCollection:
    return ChromaCollection(
        _get_client().get_collection(name=model_to_key(model_name))
    )


def list_collections() -> list[ChromaCollection]:
    return [ChromaCollection(col) for col in _get_client().list_collections()]


def get_available_models() -> list[str]:
    return sorted(col.name for col in _get_client().list_collections())


def delete_collection(model_name: str) -> bool:
    try:
        _get_client().delete_collection(name=model_to_key(model_name))
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
