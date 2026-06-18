import json
import logging
from typing import Any

import chromadb
import numpy as np

from embeddings.chroma_manager import collection_name_for_model
from settings import settings

logger = logging.getLogger(__name__)


class EmbeddingDataLoader:
    def __init__(self, auto_migrate: bool = True):
        self.client = chromadb.PersistentClient(path=str(settings.embeddings_dir))
        self._metadata_map: dict[str, str] | None = None

        if auto_migrate:
            self._auto_migrate_all()

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

    def _load_metadata_map(self) -> dict[str, str]:
        if self._metadata_map is not None:
            return self._metadata_map

        metadata_path = settings.corpus_metadata_path
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            self._metadata_map = {}
            return self._metadata_map

        try:
            self._metadata_map = {}
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
                for item in metadata:
                    if "id" in item and "tradition" in item:
                        self._metadata_map[str(item["id"])] = item["tradition"]
                        self._metadata_map[str(item["id"]).replace(" ", "_")] = item["tradition"]
        except Exception:
            logger.exception("Failed to load metadata")
            self._metadata_map = {}

        return self._metadata_map

    def _auto_migrate_all(self) -> None:
        try:
            if not self._resolve_collection_names():
                return

            for collection in self._iter_collections():
                count = collection.count()

                if count == 0:
                    continue

                if not self._needs_migration(collection):
                    continue

                logger.info(
                    f"Records without tradition found in Chroma collection '{collection.name}'. Running migration..."
                )
                metadata_map = self._load_metadata_map()

                if not metadata_map:
                    logger.warning("No data to migrate")
                    return

                migrated = self._migrate_records(collection, metadata_map)
                logger.info(f"Migration complete. Updated {migrated} records.")

        except Exception:
            logger.exception("Auto-migration failed")

    def _needs_migration(self, collection) -> bool:
        try:
            sample = collection.get(limit=min(5, collection.count()), include=["metadatas"])
            if not sample["metadatas"]:
                return False

            return any(
                "tradition" not in meta or meta.get("tradition") == "unknown" for meta in sample["metadatas"] if meta
            )
        except Exception as e:
            logger.warning(f"Failed to check migration need: {e}")
            return False

    def _migrate_records(self, collection, metadata_map: dict[str, str]) -> int:
        batch_size = 1000
        offset = 0
        migrated = 0

        while True:
            try:
                results = collection.get(limit=batch_size, offset=offset, include=["metadatas"])

                if not results["ids"]:
                    break

                updates = self._prepare_updates(results, metadata_map)

                for doc_id, meta in updates:
                    try:
                        collection.update(ids=[doc_id], metadatas=[meta])
                        migrated += 1
                    except Exception as e:
                        logger.warning(f"Failed to update {doc_id}: {e}")

                offset += batch_size

                if migrated > 0 and offset % (batch_size * 5) == 0:
                    logger.info(f"  Migrated {migrated} records...")

                if len(results["ids"]) < batch_size:
                    break

            except Exception:
                logger.exception("Migration batch failed at offset %d", offset)
                break

        return migrated

    def _prepare_updates(self, results: dict, metadata_map: dict[str, str]) -> list[tuple]:
        updates = []
        for doc_id, meta in zip(results["ids"], results["metadatas"], strict=False):
            if not meta:
                continue

            if "tradition" not in meta or meta.get("tradition") == "unknown":
                text_id = meta.get("text_id", doc_id)
                tradition = metadata_map.get(str(text_id), "unknown")
                if tradition != "unknown":
                    meta["tradition"] = tradition
                    updates.append((doc_id, meta))

        return updates

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
