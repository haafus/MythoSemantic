import logging
import shutil
import time
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from tqdm import tqdm

from .chroma_manager import (
    collection_name_for_model,
    ensure_chroma_writable,
    query_chroma_collection,
)
from .chroma_writer import ChromaWriter
from .chunking import create_chunking_strategies
from .corpus_iterator import iter_corpus_files
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


def _get_model_output_dir(base_out_dir: Path, model_name: str) -> Path:
    if not model_name:
        return base_out_dir
    from settings import Settings

    model_dir = base_out_dir / Settings.safe_model_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


class EmbeddingBuilder:
    def __init__(
        self,
        corpus_dir: str | Path,
        out_dir: str | Path,
        chroma_path: str | Path = "outputs/chroma_db",
        chunked_dir: str | Path = "outputs/corpus_chunked",
        embedding_model: str = "BAAI/bge-m3",
        chunking: str = "paragraph",
        batch_size: int | None = None,
        chroma_batch_size: int = 100,
        queue_maxsize: int = 10,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.base_out_dir = Path(out_dir)
        self.chroma_path = ensure_chroma_writable(chroma_path)
        self.chunked_dir = Path(chunked_dir)
        self.chunked_dir.mkdir(parents=True, exist_ok=True)

        self._models = ModelManager(batch_size=batch_size)

        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._chroma = ChromaWriter(self.chroma_client, chroma_batch_size, queue_maxsize)

        self.chunking_strategies = create_chunking_strategies()
        self.set_chunking_strategy(chunking)

        self._models.set_model(embedding_model)
        self._update_output_dir()

    # --- Delegated properties for backward compat --------------------------

    @property
    def model_name(self) -> str | None:
        return self._models.model_name

    @property
    def model(self) -> Any:
        return self._models.model

    @property
    def model_dim(self) -> int:
        return self._models.model_dim

    @property
    def batch_size(self) -> int:
        return self._models.batch_size

    # --- Output dir --------------------------------------------------------

    def _update_output_dir(self) -> None:
        self.out_dir = _get_model_output_dir(self.base_out_dir, self.model_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results directory: {self.out_dir}")

    # --- Model management (delegated) --------------------------------------

    def unload_model(self) -> None:
        self._models.unload_model()

    def set_model(self, model_name: str) -> None:
        self._models.set_model(model_name)
        self._update_output_dir()

    # --- Chunking ----------------------------------------------------------

    def set_chunking_strategy(self, strategy_name: str) -> None:
        if strategy_name not in self.chunking_strategies:
            available = list(self.chunking_strategies.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")
        self.current_chunking = self.chunking_strategies[strategy_name]

    def _chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [chunk for chunk in self.current_chunking(text) if chunk.strip()]

    # --- Embeddings --------------------------------------------------------

    def _generate_embeddings(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.array([])
        embeddings = self._models.model.encode(
            sentences,
            batch_size=self._models.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_embeddings(self, text: str, chunking_strategy: str | None = None, batch_size: int | None = None) -> dict[str, Any]:
        if chunking_strategy:
            self.set_chunking_strategy(chunking_strategy)

        original_batch_size = self._models.batch_size
        if batch_size is not None:
            self._models.batch_size = batch_size

        try:
            chunks = self._chunk_text(text)
            if not chunks:
                return {
                    "chunks": [],
                    "embeddings": np.array([]),
                    "model": self.model_name,
                    "chunking": self.current_chunking.name,
                    "num_chunks": 0,
                    "batch_size_used": self._models.batch_size,
                }
            embeddings = self._generate_embeddings(chunks)
            return {
                "chunks": chunks,
                "embeddings": embeddings,
                "model": self.model_name,
                "chunking": self.current_chunking.name,
                "num_chunks": len(chunks),
                "batch_size_used": self._models.batch_size,
            }
        finally:
            if batch_size is not None:
                self._models.batch_size = original_batch_size

    # --- Chroma I/O --------------------------------------------------------

    def save_all_corpus_to_chroma(self) -> None:
        collection_name = collection_name_for_model(self.model_name)
        t0 = time.monotonic()
        files_info = list(iter_corpus_files(self.corpus_dir))
        total_files = len(files_info)

        if total_files == 0:
            logger.warning("No files found in corpus/. Check the folder structure.")
            return

        traditions_file = self.corpus_dir / "traditions.json"
        if traditions_file.exists():
            try:
                dest_file = self.chunked_dir / "traditions.json"
                self.chunked_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(traditions_file, dest_file)
                logger.info(f"File {traditions_file.name} copied successfully to {self.chunked_dir}")
            except Exception as e:
                logger.warning(f"Failed to copy {traditions_file.name}: {e}")

        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        write_queue, writer_thread = self._chroma.start_background_writer(collection)

        logger.info(f"Saving {total_files} files to collection '{collection_name}'")
        added_total = 0

        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for file_info in files_info:
                try:
                    content = Path(file_info["path"]).read_text(encoding="utf-8")
                    result = self.build_embeddings(content, batch_size=self._models.batch_size)
                    chunks = result.get("chunks", [])
                    embeddings = result.get("embeddings", [])

                    if not chunks:
                        continue

                    ids, metadatas = self._chroma.build_entries(
                        chunks, file_info, self.model_name, self.current_chunking.name
                    )

                    chroma_bs = min(self._chroma.chroma_batch_size, len(chunks))
                    for i in range(0, len(chunks), chroma_bs):
                        end = min(i + chroma_bs, len(chunks))
                        write_queue.put((ids[i:end], embeddings[i:end].tolist(), metadatas[i:end], chunks[i:end]))

                    added_total += len(chunks)

                    rel_path = Path(file_info["path"]).relative_to(self.corpus_dir)
                    out_file_path = self.chunked_dir / rel_path
                    out_file_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(out_file_path, "w", encoding="utf-8") as out_f:
                        for i, chunk in enumerate(chunks, 1):
                            out_f.write(f"=== [ CHUNK {i} | Size: {len(chunk)} chars ] ===\n")
                            out_f.write(chunk)
                            out_f.write("\n\n")

                except Exception:
                    logger.exception("Error processing %s", file_info.get('filename', 'unknown'))
                finally:
                    pbar.update(1)

        logger.info("Generation complete. Waiting for final batches to be written to disk...")
        self._chroma.stop_background_writer(write_queue, writer_thread)

        elapsed = time.monotonic() - t0
        logger.info(f"Total added: {added_total} chunks to collection '{collection_name}' in {elapsed:.1f}s")

    def query_chroma(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        collection_name = collection_name_for_model(self.model_name)
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

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "EmbeddingBuilder":
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()
