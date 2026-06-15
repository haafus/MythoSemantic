import json
import logging
import re
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


def _normalize_catalog_id(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip())


def iter_corpus_files(corpus_dir: Path) -> Generator[dict[str, Any], None, None]:
    """Yield metadata dicts for every .txt file in *corpus_dir*.

    The returned dict intentionally does NOT include file content — callers
    read one file at a time so the whole corpus is never held in memory.
    """
    metadata_file = corpus_dir / "corpus.json"
    text_info: dict[str, dict[str, Any]] = {}

    if metadata_file.exists():
        try:
            with open(metadata_file, encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                tid = item.get("id")
                if not tid:
                    continue
                row_info = {
                    "text_id": _normalize_catalog_id(tid),
                    "catalog_id": tid,
                    "color": item.get("color", "#CCCCCC"),
                    "major_tradition": item.get("major_tradition", "unknown"),
                    "tradition": item.get("tradition", "unknown"),
                    "url": item.get("url", ""),
                }
                text_info[str(tid)] = row_info
                text_info[_normalize_catalog_id(tid)] = row_info
        except Exception:
            logger.exception("Error reading %s", metadata_file)
    else:
        logger.warning(f"File {metadata_file} not found.")

    for txt_file in corpus_dir.rglob("*.txt"):
        tid = txt_file.stem

        info = text_info.get(tid, {})

        try:
            rel_parts = txt_file.relative_to(corpus_dir).parts
            major_tradition = info.get("major_tradition") or (rel_parts[0] if len(rel_parts) > 1 else "unknown")
            tradition = info.get("tradition") or (rel_parts[1] if len(rel_parts) > 2 else major_tradition)
        except ValueError:
            major_tradition = "unknown"
            tradition = txt_file.parent.name

        yield {
            "filename": txt_file.name,
            "path": str(txt_file),
            "text_id": info.get("text_id", tid),
            "catalog_id": info.get("catalog_id", tid),
            "major_tradition": major_tradition,
            "tradition": tradition,
            "color": info.get("color", "#CCCCCC"),
            "url": info.get("url", ""),
        }
