import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Generator
import chromadb
import csv
from .config import get_analyzer_config
from embeddings_builder.chroma_manager import collection_name_for_model, is_model_collection_name
import logging

logger = logging.getLogger(__name__)


class EmbeddingDataLoader:
    def __init__(self, auto_migrate: bool = True):
        self.config = get_analyzer_config()
        self.client = chromadb.PersistentClient(path=self.config.chroma_path)
        self._metadata_map: Optional[Dict[str, str]] = None
        self._collection = None
        self._collection_names_cache: Dict[str, List[str]] = {}

        if auto_migrate:
            self._auto_migrate_all()

    @staticmethod
    def _collection_name(collection) -> str:
        return collection if isinstance(collection, str) else collection.name

    def _list_collection_names(self) -> List[str]:
        try:
            return sorted(self._collection_name(collection) for collection in self.client.list_collections())
        except Exception as e:
            logger.warning(f"Failed to list Chroma collections: {e}")
            return []

    def _resolve_collection_names(self, model_name: Optional[str] = None) -> List[str]:
        cache_key = model_name or "*"
        if cache_key in self._collection_names_cache:
            return self._collection_names_cache[cache_key]

        available = [
            name for name in self._list_collection_names()
            if is_model_collection_name(name)
        ]

        if model_name:
            expected_name = collection_name_for_model(model_name)
            names = [expected_name] if expected_name in available else []
        else:
            names = available

        self._collection_names_cache[cache_key] = names
        return names

    def _iter_collections(self, model_name: Optional[str] = None):
        names = self._resolve_collection_names(model_name=model_name)
        if not names:
            raise RuntimeError("Model-based Chroma collections not found")

        for name in names:
            try:
                yield self.client.get_collection(name=name)
            except Exception as e:
                logger.warning(f"Failed to get collection '{name}': {e}")

    @property
    def collection(self):
        if self._collection is None:
            try:
                
                names = self._resolve_collection_names()
                if not names:
                    raise ValueError("Model-based Chroma collections do not exist")
                self._collection = self.client.get_collection(name=names[0])
            except ValueError:
                
                logger.warning("Model-based Chroma collections do not exist. Data may not have been generated yet.")
                raise RuntimeError("Model-based Chroma collections not found")
            except Exception as e:
                logger.error(f"Failed to get model-based collection: {e}")
                raise
        return self._collection

    def _load_metadata_map(self) -> Dict[str, str]:
        if self._metadata_map is not None:
            return self._metadata_map

        metadata_path = self.config.corpus_metadata_path
        if not os.path.exists(metadata_path):
            catalog_path = os.path.join(self.config.corpus_dir, 'corpus_catalog.csv')
            if os.path.exists(catalog_path):
                metadata_path = catalog_path
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                self._metadata_map = {}
                return self._metadata_map

        try:
            self._metadata_map = {}
            
            if metadata_path.endswith('.json'):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    for item in metadata:
                        if "id" in item and "tradition" in item:
                            self._metadata_map[str(item["id"])] = item["tradition"]
                            self._metadata_map[str(item["id"]).replace(" ", "_")] = item["tradition"]
            else:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        for row in reader:
                            text_id = row.get("id") or row.get("tid")
                            tradition = row.get("tradition")
                            if text_id and tradition:
                                self._metadata_map[str(text_id)] = tradition
                                self._metadata_map[str(text_id).replace(" ", "_")] = tradition
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
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

                logger.info(f"Records without tradition found in Chroma collection '{collection.name}'. Running migration...")
                metadata_map = self._load_metadata_map()

                if not metadata_map:
                    logger.warning("No data to migrate")
                    return

                migrated = self._migrate_records(collection, metadata_map)
                logger.info(f"Migration complete. Updated {migrated} records.")

        except Exception as e:
            logger.error(f"Auto-migration failed: {e}")

    def _needs_migration(self, collection) -> bool:
        try:
            sample = collection.get(limit=min(5, collection.count()), include=["metadatas"])
            if not sample["metadatas"]:
                return False

            return any(
                "tradition" not in meta or meta.get("tradition") == "unknown"
                for meta in sample["metadatas"] if meta
            )
        except Exception as e:
            logger.warning(f"Failed to check migration need: {e}")
            return False

    def _migrate_records(self, collection, metadata_map: Dict[str, str]) -> int:
        batch_size = 1000
        offset = 0
        migrated = 0

        while True:
            try:
                results = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
                )

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

            except Exception as e:
                logger.error(f"Migration batch failed at offset {offset}: {e}")
                break

        return migrated

    def _prepare_updates(self, results: Dict, metadata_map: Dict[str, str]) -> List[tuple]:
        updates = []
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
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
            self,
            model_name: Optional[str] = None,
            batch_size: int = 5000,
            max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        all_data = []
        where_filter = {"model": model_name} if model_name else None

        for collection in self._iter_collections(model_name=model_name):
            offset = 0

            while True:
                if max_records and len(all_data) >= max_records:
                    logger.info(f"Record limit reached: {max_records}")
                    return all_data[:max_records]

                try:
                    results = collection.get(
                        where=where_filter,
                        limit=batch_size,
                        offset=offset,
                        include=["embeddings", "metadatas", "documents"]
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch data from '{collection.name}' at offset {offset}: {e}")
                    break

                if not results.get("ids"):
                    break

                batch_data = self._process_batch(results)
                all_data.extend(batch_data)

                offset += batch_size

                if len(results["ids"]) < batch_size:
                    break

        return all_data

    def _process_batch(self, results: Dict) -> List[Dict[str, Any]]:
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

                embedding = np.asarray(embeddings[i], dtype=np.float32)

                batch_data.append({
                    "id": meta.get("text_id", doc_id),
                    "tradition": meta.get("tradition", "unknown"),
                    "major_tradition": meta.get("major_tradition", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "embedding": embedding,
                    "text": doc,
                    "model": meta.get("model", "unknown"),
                    "filename": meta.get("filename", "unknown"),
                    "chunking": meta.get("chunking", "unknown"),
                    "doc_type": meta.get("doc_type", "unknown"),
                    "color": meta.get("color", "#CCCCCC"),
                    "language": meta.get("language", "unknown"),
                    "url": meta.get("url", ""),
                })
            except Exception as e:
                logger.warning(f"Failed to process document {doc_id}: {e}")
                continue

        return batch_data

    def get_available_models(self) -> List[str]:
        models = set()
        batch_size = 5000

        try:
            if not self._resolve_collection_names():
                return []

            for collection in self._iter_collections():
                offset = 0
                while True:
                    result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
                    metadatas = result.get("metadatas", [])
                    models.update(m.get("model") for m in metadatas if m and "model" in m)

                    ids = result.get("ids", [])
                    if len(ids) < batch_size:
                        break
                    offset += batch_size

            return sorted(list(models))
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def close(self):
        if hasattr(self, 'client'):
            self._collection = None
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
