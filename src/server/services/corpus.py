import io
import json
import logging
import zipfile
from pathlib import Path

from corpus.utils import text_path, read_traditions, read_document, sanitize_filename
from settings import settings

logger = logging.getLogger(__name__)


def get_catalog_documents() -> list[dict]:
    metadata_path = settings.corpus_dir / "corpus.json"

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata_rows = json.load(handle)

    documents = []
    traditions_info = read_traditions(settings.corpus_dir)

    for row in metadata_rows:
        tradition_info = traditions_info.get(row.get("tradition", ""), {})
        documents.append(
            {
                "id": row.get("id", ""),
                "major_tradition": row.get("major_tradition", ""),
                "tradition": row.get("tradition", ""),
                "url": row.get("url", ""),
                "word_count": row.get("word_count", 0),
                "sentence_count": row.get("sentence_count", 0),
                "char_count": row.get("char_count", 0),
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

    return documents


def build_corpus_archive() -> io.BytesIO:
    documents = get_catalog_documents()
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for doc in documents:
            file_path = text_path(
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
