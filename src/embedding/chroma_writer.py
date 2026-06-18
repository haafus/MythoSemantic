import logging
import queue
import re
import threading
from typing import Any

from corpus.corpus_iterator import CorpusFileInfo
from .chroma_manager import save_to_chroma_collection

logger = logging.getLogger(__name__)


def _safe_id_part(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "unknown")).strip("_") or "unknown"


def _safe_meta(val: Any) -> str:
    return "" if val is None else val


class ChromaWriter:
    def __init__(self, chroma_batch_size: int = 100, queue_maxsize: int = 10):
        self.chroma_batch_size = chroma_batch_size
        self.queue_maxsize = queue_maxsize
        self._write_error: Exception | None = None

    def build_entries(
        self, chunks: list[str], info: CorpusFileInfo, model_name: str, chunking_name: str
    ) -> tuple[list[str], list[dict[str, Any]]]:
        text_id_safe = _safe_id_part(info.text_id)
        model_id = _safe_id_part(model_name)

        ids = [f"{text_id_safe}_{model_id}_{i}" for i in range(len(chunks))]

        metadatas = [
            {
                "filename": info.filename,
                "tradition": info.tradition,
                "major_tradition": info.major_tradition,
                "chunk_index": i,
                "model": _safe_meta(model_name),
                "chunking": _safe_meta(chunking_name),
                "text_id": info.text_id,
                "url": info.url,
            }
            for i in range(len(chunks))
        ]
        return ids, metadatas

    def _background_worker(self, collection: Any, write_queue: queue.Queue) -> None:
        while True:
            batch = write_queue.get()
            if batch is None:
                write_queue.task_done()
                break
            ids, embeddings, metadatas, documents = batch
            try:
                save_to_chroma_collection(
                    collection=collection, ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
                )
                logger.debug(f"Background thread saved a batch of {len(ids)} chunks.")
            except Exception as e:
                logger.exception("Background Chroma write error")
                if self._write_error is None:
                    self._write_error = e
            finally:
                write_queue.task_done()

    def start_background_writer(self, collection: Any) -> tuple[queue.Queue, threading.Thread]:
        write_queue: queue.Queue[tuple | None] = queue.Queue(maxsize=self.queue_maxsize)
        thread = threading.Thread(target=self._background_worker, args=(collection, write_queue), daemon=True)
        thread.start()
        return write_queue, thread

    def stop_background_writer(self, write_queue: queue.Queue, thread: threading.Thread) -> None:
        write_queue.put(None)
        thread.join(timeout=300)
        if thread.is_alive():
            logger.error("Background writer thread did not finish within timeout")
        if self._write_error is not None:
            raise RuntimeError(f"Background Chroma write failed: {self._write_error}") from self._write_error
