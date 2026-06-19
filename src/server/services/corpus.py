import io
import json
import logging
import time
import zipfile
from pathlib import Path

from corpus.utils import corpus_text_path, load_traditions, read_document, sanitize_filename
from settings import settings

logger = logging.getLogger(__name__)

_catalog_cache: dict[str, tuple[float, list[dict]]] = {}
_CATALOG_TTL = 300


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_catalog_documents() -> list[dict]:
    cached = _catalog_cache.get("corpus")
    if cached and time.monotonic() - cached[0] < _CATALOG_TTL:
        return cached[1]

    metadata_path = settings.corpus_dir / "corpus.json"

    metadata_rows = []
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata_rows = json.load(handle)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read metadata %s: %s", metadata_path, e)

    documents = []
    traditions_info = load_traditions(settings.corpus_dir)

    for row in metadata_rows:
        tradition_info = traditions_info.get(row.get("tradition", ""), {})
        documents.append(
            {
                "id": row.get("id", ""),
                "major_tradition": row.get("major_tradition", ""),
                "tradition": row.get("tradition", ""),
                "url": row.get("url", ""),
                "word_count": to_int(row.get("word_count")),
                "sentence_count": to_int(row.get("sentence_count")),
                "char_count": to_int(row.get("char_count")),
                "color": tradition_info.get("color") or "#6b7280",
                "description": row.get("description") or tradition_info.get("description", ""),
            }
        )

    documents.sort(
        key=lambda item: (
            item.get("major_tradition", ""),
            item.get("tradition", ""),
            item.get("id", ""),
        )
    )

    _catalog_cache["corpus"] = (time.monotonic(), documents)
    return documents


def build_corpus_archive() -> io.BytesIO:
    documents = get_catalog_documents()
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for doc in documents:
            file_path = corpus_text_path(
                settings.corpus_dir,
                doc.get("major_tradition", ""),
                doc.get("tradition", ""),
                doc.get("id", ""),
            )

            if not file_path.exists():
                continue

            archive_name = (
                Path(sanitize_filename(doc.get("major_tradition", "Unknown")))
                / sanitize_filename(doc.get("tradition", "Unknown"))
                / file_path.name
            ).as_posix()
            archive.write(file_path, archive_name)

    buf.seek(0)
    return buf
