import io
import json
import logging
import time
import zipfile
from pathlib import Path

from corpus.utils import sanitize_filename
from settings import settings

logger = logging.getLogger(__name__)

_catalog_cache: dict[str, tuple[float, list[dict]]] = {}
_CATALOG_TTL = 300


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default



def source_root() -> Path:
    return settings.corpus_dir


def get_catalog_documents() -> list[dict]:
    cached = _catalog_cache.get("corpus")
    if cached and time.monotonic() - cached[0] < _CATALOG_TTL:
        return cached[1]

    root = source_root()
    metadata_path = root / "corpus.json"

    metadata_rows = []
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata_rows = json.load(handle)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read metadata %s: %s", metadata_path, e)

    documents = []
    traditions_info = get_traditions_info()

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
                "color": row.get("color") or tradition_info.get("color") or "#6b7280",
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



def resolve_document_path(
    doc_id: str, major_tradition: str, tradition: str,
) -> tuple[Path | None, str]:
    corpus_root = source_root().resolve()
    major_path = sanitize_filename(major_tradition)
    tradition_path = sanitize_filename(tradition)
    title_path = sanitize_filename(doc_id)
    file_path = (corpus_root / major_path / tradition_path / f"{title_path}.txt").resolve()

    try:
        file_path.relative_to(corpus_root)
    except ValueError:
        return None, title_path

    return file_path, title_path


def read_document(doc_id: str, major_tradition: str, tradition: str) -> tuple[str, str]:
    file_path, title_path = resolve_document_path(doc_id, major_tradition, tradition)
    if not file_path:
        raise PermissionError("Access denied")
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    return file_path.read_text(encoding="utf-8"), title_path


def build_corpus_archive() -> io.BytesIO:
    documents = get_catalog_documents()
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for doc in documents:
            file_path, title_path = resolve_document_path(
                doc.get("id", ""),
                doc.get("major_tradition", ""),
                doc.get("tradition", ""),
            )

            if not file_path or not file_path.exists():
                continue

            archive_name = (
                Path(sanitize_filename(doc.get("major_tradition", "Unknown")))
                / sanitize_filename(doc.get("tradition", "Unknown"))
                / f"{title_path}.txt"
            ).as_posix()
            archive.write(file_path, archive_name)

    buf.seek(0)
    return buf


def get_traditions_info() -> dict:
    path = settings.corpus_dir / "traditions.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read %s: %s", path, e)
    return {}
