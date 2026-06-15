import io
import json
import logging
import time
import zipfile
from pathlib import Path

from corpus.utils import count_sentences, count_words, sanitize_filename
from settings import settings

logger = logging.getLogger(__name__)

_catalog_cache: dict[str, tuple[float, list[dict]]] = {}
_doc_index_cache: dict[str, tuple[float, dict[tuple[str, str, str], Path]]] = {}
_CATALOG_TTL = 300


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default



def _catalog_sources() -> dict[str, Path]:
    return {
        "corpus": settings.corpus_dir,
        "chunked": settings.corpus_chunked_dir,
    }


def source_root(source: str = "corpus") -> Path:
    return _catalog_sources().get(source, settings.corpus_dir)


def get_catalog_documents(source: str = "corpus") -> list[dict]:
    cached = _catalog_cache.get(source)
    if cached and time.monotonic() - cached[0] < _CATALOG_TTL:
        return cached[1]

    root = source_root(source)
    metadata_path = root / "corpus.json"

    metadata_rows = []
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata_rows = json.load(handle)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read metadata %s: %s", metadata_path, e)

    source_rows = metadata_rows or scan_document_rows(root)
    documents = []
    traditions_info = get_traditions_info(source)

    for row in source_rows:
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
                "source": source,
            }
        )

    documents.sort(
        key=lambda item: (
            item.get("major_tradition", ""),
            item.get("tradition", ""),
            item.get("id", ""),
        )
    )

    _catalog_cache[source] = (time.monotonic(), documents)
    return documents


def scan_document_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows

    for file_path in root.glob("*/*/*/*.txt"):
        try:
            major, tradition, title_dir, _ = file_path.relative_to(root).parts
        except ValueError:
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        title = title_dir.replace("_", " ")
        rows.append(
            {
                "id": title,
                "major_tradition": major.replace("_", " "),
                "tradition": tradition.replace("_", " "),
                "path": str(file_path.relative_to(root)),
                "char_count": len(text),
                "word_count": count_words(text),
                "sentence_count": count_sentences(text),
            }
        )

    return rows


def _document_index(source: str) -> dict[tuple[str, str, str], Path]:
    """TTL-cached index of (major, tradition, title) -> file path, symlink-escape safe."""
    cached = _doc_index_cache.get(source)
    if cached and time.monotonic() - cached[0] < _CATALOG_TTL:
        return cached[1]

    root = source_root(source).resolve()
    index: dict[tuple[str, str, str], Path] = {}
    if root.exists():
        for candidate in root.glob("*/*/*/*.txt"):
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            key = (
                candidate.parents[2].name.replace("_", " ").casefold(),
                candidate.parents[1].name.replace("_", " ").casefold(),
                candidate.stem.replace("_", " ").casefold(),
            )
            index[key] = resolved

    _doc_index_cache[source] = (time.monotonic(), index)
    return index


def resolve_document_path(
    doc_id: str, major_tradition: str, tradition: str, source: str = "corpus"
) -> tuple[Path | None, str]:
    corpus_root = source_root(source).resolve()
    major_path = sanitize_filename(major_tradition)
    tradition_path = sanitize_filename(tradition)
    title_path = sanitize_filename(doc_id)
    file_path = (corpus_root / major_path / tradition_path / title_path / f"{title_path}.txt").resolve()

    try:
        file_path.relative_to(corpus_root)
    except ValueError:
        return None, title_path

    if source == "chunked" and not file_path.exists():
        key = (major_tradition.casefold(), tradition.casefold(), doc_id.casefold())
        file_path = _document_index(source).get(key, file_path)

    return file_path, title_path


def read_document(doc_id: str, major_tradition: str, tradition: str, source: str = "corpus") -> tuple[str, str]:
    file_path, title_path = resolve_document_path(doc_id, major_tradition, tradition, source)
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
                doc.get("source", "corpus"),
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


def get_traditions_info(source: str | None = None) -> dict:
    paths = (
        [source_root(source) / "traditions_info.json"]
        if source
        else [
            settings.corpus_chunked_dir / "traditions_info.json",
            settings.corpus_dir / "traditions_info.json",
        ]
    )
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read %s: %s", path, e)
    return {}
