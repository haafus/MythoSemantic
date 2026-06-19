import logging
import time
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm

from .chroma_manager import build_chroma_entries, collection_name_for_model, save_to_chroma_collection
from .chunking import create_chunking_strategies
from corpus.corpus_iterator import iter_corpus_files
from .model_manager import EmbeddingEncoder

logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    def __init__(self, embedding_model: str = "BAAI/bge-m3"):
        from settings import settings

        emb = settings.embedding
        self.corpus_dir = Path(settings.corpus_dir)
        self.batch_size = emb.batch_size

        self._encoder = EmbeddingEncoder()

        chroma_dir = Path(settings.embeddings_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

        self._chunking_strategies = create_chunking_strategies()
        self.set_chunking_strategy(emb.default_chunking)

        self._encoder.load(embedding_model)

    def load_model(self, model_name: str) -> None:
        self._encoder.load(model_name)

    def set_chunking_strategy(self, strategy_name: str) -> None:
        if strategy_name not in self._chunking_strategies:
            available = list(self._chunking_strategies.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")
        self.current_chunking = self._chunking_strategies[strategy_name]

    def _chunk_text(self, text: str) -> list[str]:
        if not text:
            return []
        return [chunk for chunk in self.current_chunking(text) if chunk.strip()]

    def save_all_corpus_to_chroma(self) -> None:
        collection_name = collection_name_for_model(self._encoder.model_name)
        t0 = time.monotonic()
        files_info = list(iter_corpus_files(self.corpus_dir))

        if not files_info:
            logger.warning("No files found in corpus/. Check the folder structure.")
            return

        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"model": self._encoder.model_name, "chunking": self.current_chunking.name},
        )

        existing_ids = set(collection.get(include=[])["ids"])
        if existing_ids:
            logger.info(f"Collection '{collection_name}' has {len(existing_ids)} existing chunks, resuming")

        batch_size = self.batch_size
        added_total = 0
        skipped_total = 0
        total_chunks = 0
        encode_seconds = 0.0

        logger.info(f"Embedding {len(files_info)} files to collection '{collection_name}'")

        with tqdm(desc="Embedding", unit="chunk") as pbar:
            for file_info in files_info:
                content = file_info.read()
                chunks = self._chunk_text(content)
                if not chunks:
                    continue
                n_chunks = len(chunks)
                total_chunks += n_chunks
                try:
                    ids, metadatas = build_chroma_entries(
                        chunks, file_info, self._encoder.model_name,
                    )

                    missing = [
                        (i, chunk) for i, (cid, chunk) in enumerate(zip(ids, chunks, strict=True))
                        if cid not in existing_ids
                    ]

                    n_skipped = n_chunks - len(missing)
                    if n_skipped:
                        skipped_total += n_skipped
                        pbar.update(n_skipped)

                    if not missing:
                        continue

                    missing_indices = [i for i, _ in missing]
                    missing_chunks = [chunk for _, chunk in missing]
                    missing_ids = [ids[i] for i in missing_indices]
                    missing_metas = [metadatas[i] for i in missing_indices]

                    for b_start in range(0, len(missing_chunks), batch_size):
                        b_end = min(b_start + batch_size, len(missing_chunks))
                        b_chunks = missing_chunks[b_start:b_end]

                        t_enc = time.monotonic()
                        b_embs = self._encoder.encode(
                            b_chunks,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                        )
                        encode_seconds += time.monotonic() - t_enc
                        b_embs = np.asarray(b_embs, dtype=np.float32)

                        save_to_chroma_collection(
                            collection=collection,
                            ids=missing_ids[b_start:b_end],
                            embeddings=b_embs,
                            metadatas=missing_metas[b_start:b_end],
                            documents=b_chunks,
                        )

                        pbar.update(len(b_chunks))

                    added_total += len(missing_chunks)

                except Exception:
                    logger.exception("Error processing %s", file_info.filename)

        collection.modify(metadata={
            "model": self._encoder.model_name,
            "chunking": self.current_chunking.name,
            "total_chunks": total_chunks,
        })

        elapsed = time.monotonic() - t0
        logger.info(f"Done: {added_total} added, {skipped_total} skipped, {total_chunks} total in '{collection_name}' ({elapsed:.1f}s)")
        if encode_seconds > 0 and added_total > 0:
            speed = added_total / encode_seconds
            logger.info(f"Encode speed: {speed:,.1f} chunks/sec ({added_total} chunks in {encode_seconds:.1f}s)")

    def close(self) -> None:
        if hasattr(self, "_encoder"):
            self._encoder.unload()
