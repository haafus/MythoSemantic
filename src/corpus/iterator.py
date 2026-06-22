import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from .utils import normalize_catalog_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorpusFileInfo:
    filename: str
    text_id: str
    major_tradition: str
    tradition: str
    url: str


def iter_files(corpus_dir: Path) -> Generator[tuple[Path, CorpusFileInfo], None, None]:
    metadata_file = corpus_dir / "corpus.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"{metadata_file} not found. Run 'mytho corpus build' first."
        )

    with open(metadata_file, encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        title = item.get("title")
        if not title:
            continue

        path = item.get("path")
        if not path:
            logger.warning("Skipping entry '%s': no path", title)
            continue

        txt_file = corpus_dir / path
        if not txt_file.exists():
            logger.warning("Skipping entry '%s': file not found at %s", title, txt_file)
            continue

        yield txt_file, CorpusFileInfo(
            filename=txt_file.name,
            text_id=normalize_catalog_id(title),
            major_tradition=item.get("major_tradition", "unknown"),
            tradition=item.get("tradition", "unknown"),
            url=item.get("url", ""),
        )
