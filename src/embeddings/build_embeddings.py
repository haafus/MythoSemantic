import dataclasses
import logging
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from corpus.iterator import CorpusFileInfo, iter_files
from corpus.utils import chunk_id
from model_registry import active_embedding_models, resolve_embedding_model
from settings import settings

from . import chroma_manager
from .chunking import chunk_text
from .model_manager import EmbeddingEncoder

logger = logging.getLogger(__name__)


def build_embeddings(
    model_name: str | None = None,
    models: list | None = None,
    force: bool = False,
) -> None:
    if model_name:
        models_to_run = [resolve_embedding_model(model_name)]
    else:
        models_to_run = models or active_embedding_models()

    encoder = EmbeddingEncoder()
    encoder.load(models_to_run[0])

    logger.info("Starting embedding generation...")
    logger.info(f"   Source: {settings.corpus_dir}")
    logger.info(f"   Embeddings: {settings.embeddings_dir}")

    try:
        for model in models_to_run:
            if force:
                chroma_manager.delete_collection(model)
            encoder.load(model)
            logger.info(f"   Model: {model}")
            _save_corpus_to_chroma(encoder)
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise
    finally:
        encoder.unload()

    logger.info("All embeddings saved to Chroma.")


def _save_corpus_to_chroma(encoder: EmbeddingEncoder) -> None:
    model_name = encoder.model_name
    emb = settings.embedding
    corpus_dir = settings.corpus_dir
    batch_size = emb.batch_size

    t0 = time.monotonic()
    files_info = list(iter_files(corpus_dir))

    if not files_info:
        logger.warning("No files found in corpus/. Check the folder structure.")
        return

    collection = chroma_manager.get_or_create_collection(
        model_name,
        metadata={"model": model_name, "hnsw:space": "cosine"},
    )

    existing_ids = collection.existing_ids()
    if existing_ids:
        logger.info(f"Collection '{collection.name}' has {len(existing_ids)} existing chunks, resuming")

    added_total = 0
    skipped_total = 0
    total_chunks = 0
    encode_seconds = 0.0

    logger.info(f"Embedding {len(files_info)} files to collection '{collection.name}'")

    with tqdm(desc="Embedding", unit="chunk") as pbar:
        for file_info in files_info:
            content = file_info.read_text()
            chunks = [c for c in chunk_text(content, emb.chunk_size, emb.chunk_overlap) if c.strip()]
            if not chunks:
                continue
            n_chunks = len(chunks)
            total_chunks += n_chunks
            try:
                ids, metadatas = _build_chroma_entries(chunks, file_info)

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
                    b_embs = encoder.encode(
                        b_chunks,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    encode_seconds += time.monotonic() - t_enc
                    b_embs = np.asarray(b_embs, dtype=np.float32)

                    collection.upsert(
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
        "model": model_name,
        "chunk_size": emb.chunk_size,
        "total_chunks": total_chunks,
    })

    elapsed = time.monotonic() - t0
    logger.info(f"Done: {added_total} added, {skipped_total} skipped, {total_chunks} total in '{collection.name}' ({elapsed:.1f}s)")
    if encode_seconds > 0 and added_total > 0:
        speed = added_total / encode_seconds
        logger.info(f"Encode speed: {speed:,.1f} chunks/sec ({added_total} chunks in {encode_seconds:.1f}s)")


def _build_chroma_entries(
    chunks: list[str], info: CorpusFileInfo,
) -> tuple[list[str], list[dict[str, Any]]]:
    ids = [chunk_id(info.text_id, i) for i in range(len(chunks))]
    base = dataclasses.asdict(info)
    metadatas = [{**base, "chunk_index": i} for i in range(len(chunks))]
    return ids, metadatas
