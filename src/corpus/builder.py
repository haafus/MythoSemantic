import concurrent.futures
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from settings import settings

from .clean_gutenberg import clean_gutenberg_in_builder
from .downloader import download_file, load_download_list
from .extraction import _decode_bytes, html_to_text, pdf_to_text
from .utils import (
    text_path,
    count_sentences,
    count_words,
    ensure_dir,
    get_tradition_color,
    md5,
    normalize_text,
)

logger = logging.getLogger(__name__)
data_lock = threading.Lock()


def _item_tid(item: dict) -> str:
    return item["title"]


def _finalize_text(text: str, url: str, tid: str) -> dict:
    text = normalize_text(text)
    text = clean_gutenberg_in_builder(text, url, tid)
    data_utf8 = text.encode("utf-8")
    return {
        "text": text,
        "data_utf8": data_utf8,
        "md5": md5(data_utf8),
        "char_count": len(text),
        "word_count": count_words(text),
        "sentence_count": count_sentences(text),
    }


def _build_metadata(item: dict, *, path: str, stats: dict) -> dict:
    return {
        "title": _item_tid(item),
        "major_tradition": item.get("major_tradition", "Unknown"),
        "tradition": item["tradition"],
        "url": item["url"],
        "date_downloaded": datetime.now(timezone.utc).isoformat(),
        "md5": stats["md5"],
        "path": path,
        "char_count": stats["char_count"],
        "word_count": stats["word_count"],
        "sentence_count": stats["sentence_count"],
        "description": item.get("description", ""),
    }



def _extract_text(data: bytes, url: str, tid: str, content_type: str = "") -> str:
    is_pdf = url.lower().endswith(".pdf") or "application/pdf" in content_type or data[:4] == b"%PDF"
    is_html = (
        b"<html" in data[:200].lower() or "text/html" in content_type or b"<!doctype html" in data[:200].lower()
    )

    if is_pdf:
        logger.debug(f"{tid}: PDF detected, extracting text")
        return pdf_to_text(
            data,
            extract_tables=settings.corpus.pdf_extract_tables,
            preserve_layout=settings.corpus.pdf_preserve_layout,
        )
    if is_html:
        logger.debug(f"{tid}: HTML detected, converting to text")
        return html_to_text(
            data,
            include_comments=settings.corpus.html_include_comments,
            include_tables=settings.corpus.html_include_tables,
        )
    return _decode_bytes(data)


def _download_and_process(item: dict) -> dict | None:
    tid = _item_tid(item)
    url = item["url"]

    try:
        data = download_file(url)
        content_type = item.get("content_type", "")
        text = _extract_text(data, url, tid, content_type)

        if not text or not text.strip():
            raise ValueError("Empty content after conversion")

        stats = _finalize_text(text, url, tid)

        filename = text_path(settings.corpus_dir, item.get("major_tradition", "Unknown"), item["tradition"], tid)

        with data_lock:
            ensure_dir(filename.parent)
            filename.write_bytes(stats["data_utf8"])

        rel_path = str(filename.resolve().relative_to(Path(settings.corpus_dir).resolve()))
        meta = _build_metadata(item, path=rel_path, stats=stats)
        logger.info(f"Saved successfully: {filename.name} (words: {stats['word_count']})")
        return meta

    except Exception:
        logger.exception("%s: Processing error", tid)
        return None


def _update_traditions(force: bool) -> None:
    config_path = settings.config_dir / "traditions.json"
    output_path = settings.corpus_dir / "traditions.json"

    static: dict = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                static = json.load(f)
        except Exception:
            logger.exception("Error reading %s", config_path)

    corpus_traditions: set[str] = set()
    corpus_config = settings.config_dir / "corpus.json"
    if corpus_config.exists():
        with open(corpus_config, encoding="utf-8") as f:
            for item in json.load(f):
                if "tradition" in item:
                    corpus_traditions.add(item["tradition"])

    result: dict = {}
    for trad in sorted(corpus_traditions | set(static.keys())):
        info = static.get(trad, {"description": "", "coordinates": []})
        info["color"] = get_tradition_color(trad)
        result[trad] = info

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _load_existing_metadata() -> dict[str, dict]:
    path = settings.corpus_dir / "corpus.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return {row["id"]: row for row in rows if "id" in row}
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to read existing %s: %s", path, e)
        return {}


def build_corpus(force: bool = False, max_texts: int | None = None):
    ensure_dir(settings.corpus_dir)

    download_list = load_download_list()
    if max_texts is not None:
        download_list = download_list[:max_texts]

    _update_traditions(force)

    existing = {} if force else _load_existing_metadata()

    to_download = []
    metadata: list[dict] = []
    corpus_root = Path(settings.corpus_dir).resolve()

    for item in download_list:
        tid = _item_tid(item)
        prev = existing.get(tid)
        if prev and (corpus_root / prev.get("path", "")).exists():
            metadata.append(prev)
            logger.debug(f"{tid}: already in corpus, skipping")
        else:
            to_download.append(item)

    logger.info(f"Corpus: {len(metadata)} cached, {len(to_download)} to download")

    if to_download:
        new_metadata: list[dict] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.corpus.max_workers) as executor:
            futures = {executor.submit(_download_and_process, item): item for item in to_download}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    new_metadata.append(result)

        metadata.extend(new_metadata)
        logger.info(f"Downloaded: {len(new_metadata)}, failed: {len(to_download) - len(new_metadata)}")

    with open(settings.corpus_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Corpus build complete. Total: %d texts", len(metadata))

    if metadata:
        total_words = sum(m.get("word_count", 0) for m in metadata)
        total_sentences = sum(m.get("sentence_count", 0) for m in metadata)
        logger.info(f"  Total words: {total_words}")
        logger.info(f"  Total sentences: {total_sentences}")
        logger.info(f"  Average words per text: {total_words / len(metadata):.1f}")
