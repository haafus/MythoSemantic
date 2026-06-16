import json
import logging
import re
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


def _normalize_catalog_id(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip())


def iter_corpus_files(corpus_dir: Path) -> Generator[dict[str, Any], None, None]:
    """Yield metadata dicts for every entry in *corpus_dir*/corpus.json.

    The returned dict intentionally does NOT include file content — callers
    read one file at a time so the whole corpus is never held in memory.
    """
    metadata_file = corpus_dir / "corpus.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"{metadata_file} not found. Run 'mytho corpus build' first."
        )

    with open(metadata_file, encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        tid = item.get("id")
        if not tid:
            continue

        path = item.get("path")
        if not path:
            logger.warning("Skipping entry '%s': no path", tid)
            continue

        txt_file = corpus_dir / path
        if not txt_file.exists():
            logger.warning("Skipping entry '%s': file not found at %s", tid, txt_file)
            continue

        text_id = _normalize_catalog_id(tid)

        yield {
            "filename": txt_file.name,
            "path": str(txt_file),
            "text_id": text_id,
            "catalog_id": tid,
            "major_tradition": item.get("major_tradition", "unknown"),
            "tradition": item.get("tradition", "unknown"),
            "url": item.get("url", ""),
        }
