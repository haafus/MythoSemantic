import logging
import time
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from tqdm import tqdm

from .chroma_manager import (
    collection_name_for_model,
    query_chroma_collection,
)
from .chroma_writer import ChromaWriter
from .chunking import create_chunking_strategies
from corpus.corpus_iterator import iter_corpus_files
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    def __init__(self, embedding_model: str = "BAAI/bge-m3"):
        from settings import settings

        emb = settings.embedding
        self.corpus_dir = Path(settings.corpus_dir)
        self.batch_size = emb.batch_size

        self._models = ModelManager()

        chroma_dir = Path(settings.chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self._chroma = ChromaWriter(emb.chroma_batch_size, emb.queue_maxsize)

        self._chunking_strategies = create_chunking_strategies()
        self.set_chunking_strategy(emb.default_chunking)

        self._models.set_model(embedding_model)

    def set_model(self, model_name: str) -> None:
        self._models.set_model(model_name)

    # --- Chunking ----------------------------------------------------------

    def set_chunking_strategy(self, strategy_name: str) -> None:
        if strategy_name not in self._chunking_strategies:
            available = list(self._chunking_strategies.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")
        self.current_chunking = self._chunking_strategies[strategy_name]

    def _chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [chunk for chunk in self.current_chunking(text) if chunk.strip()]

    # --- Embeddings --------------------------------------------------------

    def _generate_embeddings(self, sentences: list[str]) -> np.ndarray:
        embeddings = self._models.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_embeddings(self, text: str) -> dict[str, Any]:
        chunks = self._chunk_text(text)
        if not chunks:
            return {
                "chunks": [],
                "embeddings": np.array([]),
                "model": self._models.model_name,
                "chunking": self.current_chunking.name,
                "num_chunks": 0,
            }
        embeddings = self._generate_embeddings(chunks)
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "model": self._models.model_name,
            "chunking": self.current_chunking.name,
            "num_chunks": len(chunks),
        }

    # --- Chroma I/O --------------------------------------------------------

    def save_all_corpus_to_chroma(self) -> None:
        collection_name = collection_name_for_model(self._models.model_name)
        t0 = time.monotonic()
        files_info = list(iter_corpus_files(self.corpus_dir))

        if not files_info:
            logger.warning("No files found in corpus/. Check the folder structure.")
            return

        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        existing_ids = set(collection.get(include=[])["ids"])
        if existing_ids:
            logger.info(f"Collection '{collection_name}' has {len(existing_ids)} existing chunks, resuming")

        file_chunks: list[tuple[dict, list[str]]] = []
        total_chunks = 0
        for fi in files_info:
            content = Path(fi["path"]).read_text(encoding="utf-8")
            chunks = self._chunk_text(content)
            if chunks:
                file_chunks.append((fi, chunks))
                total_chunks += len(chunks)

        write_queue, writer_thread = self._chroma.start_background_writer(collection)

        logger.info(f"Saving {len(files_info)} files ({total_chunks} chunks) to collection '{collection_name}'")
        added_total = 0
        skipped_total = 0
        batch_size = self.batch_size

        with tqdm(total=total_chunks, desc="Embedding", unit="chunk") as pbar:
            for file_info, chunks in file_chunks:
                n_chunks = len(chunks)
                chunks_accounted = 0
                try:
                    ids, metadatas = self._chroma.build_entries(
                        chunks, file_info, self._models.model_name, self.current_chunking.name
                    )

                    missing = [
                        (i, chunk) for i, (cid, chunk) in enumerate(zip(ids, chunks, strict=True))
                        if cid not in existing_ids
                    ]

                    n_skipped = n_chunks - len(missing)
                    if n_skipped:
                        skipped_total += n_skipped
                        pbar.update(n_skipped)
                        chunks_accounted += n_skipped

                    if not missing:
                        continue

                    missing_indices = [i for i, _ in missing]
                    missing_chunks = [chunk for _, chunk in missing]
                    missing_ids = [ids[i] for i in missing_indices]
                    missing_metas = [metadatas[i] for i in missing_indices]

                    for b_start in range(0, len(missing_chunks), batch_size):
                        b_end = min(b_start + batch_size, len(missing_chunks))
                        b_chunks = missing_chunks[b_start:b_end]

                        b_embs = self._models.model.encode(
                            b_chunks,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                        )
                        b_embs = np.asarray(b_embs, dtype=np.float32)

                        b_ids = missing_ids[b_start:b_end]
                        b_metas = missing_metas[b_start:b_end]

                        chroma_bs = min(self._chroma.chroma_batch_size, len(b_chunks))
                        for c_start in range(0, len(b_chunks), chroma_bs):
                            c_end = min(c_start + chroma_bs, len(b_chunks))
                            write_queue.put((
                                b_ids[c_start:c_end],
                                b_embs[c_start:c_end].tolist(),
                                b_metas[c_start:c_end],
                                b_chunks[c_start:c_end],
                            ))

                        pbar.update(len(b_chunks))
                        chunks_accounted += len(b_chunks)

                    added_total += len(missing_chunks)

                except Exception:
                    logger.exception("Error processing %s", file_info.get('filename', 'unknown'))
                finally:
                    remaining = n_chunks - chunks_accounted
                    if remaining > 0:
                        pbar.update(remaining)

        logger.info("Generation complete. Waiting for final batches to be written to disk...")
        self._chroma.stop_background_writer(write_queue, writer_thread)

        elapsed = time.monotonic() - t0
        logger.info(f"Total added: {added_total}, skipped: {skipped_total} chunks in collection '{collection_name}' ({elapsed:.1f}s)")

    def query_chroma(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        collection_name = collection_name_for_model(self._models.model_name)
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except Exception as err:
            raise RuntimeError(f"Collection '{collection_name}' not found in ChromaDB.") from err

        query_embedding = self._generate_embeddings([query])[0]
        return query_chroma_collection(collection=collection, query_embedding=query_embedding.tolist(), top_k=top_k)

    # --- Resource management -----------------------------------------------

    def close(self) -> None:
        if hasattr(self, "_models"):
            self._models.close()
