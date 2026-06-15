import concurrent.futures
import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from settings import settings

from .clean_gutenberg import clean_gutenberg_in_builder
from .downloader import download_file, load_download_list
from .extraction import _decode_bytes, html_to_text, pdf_to_text
from .utils import (
    corpus_text_path,
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
    return item.get("title", item.get("id", "unknown_id"))


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


def _build_metadata(item: dict, *, path: str, color: str, stats: dict) -> dict:
    return {
        "id": _item_tid(item),
        "major_tradition": item.get("major_tradition", "Unknown"),
        "tradition": item["tradition"],
        "url": item["url"],
        "date_downloaded": datetime.now(timezone.utc).isoformat(),
        "md5": stats["md5"],
        "path": path,
        "char_count": stats["char_count"],
        "word_count": stats["word_count"],
        "sentence_count": stats["sentence_count"],
        "color": color,
        "available": True,
        "description": item.get("description", ""),
    }


def _build_failure_metadata(item: dict, *, color: str, error: str) -> dict:
    description = item.get("description", "")
    return {
        "id": _item_tid(item),
        "major_tradition": item.get("major_tradition", "Unknown"),
        "tradition": item["tradition"],
        "url": item["url"],
        "available": False,
        "word_count": 0,
        "sentence_count": 0,
        "color": color,
        "description": f"{description} [{error}]" if description else error,
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


def process_local_file(filename: Path, item: dict, color: str) -> dict | None:
    tid = _item_tid(item)
    url = item["url"]

    try:
        logger.info(f"{tid}: Processing existing file {filename}")
        data = filename.read_bytes()
        text = _extract_text(data, url, tid)
        stats = _finalize_text(text, url, tid)
        return _build_metadata(item, path=str(filename.resolve()), color=color, stats=stats)
    except Exception:
        logger.exception("%s: Error processing local file %s", tid, filename)
        return None


def process_single_item(item: dict, force: bool, metadata: list[dict]):
    tid = _item_tid(item)
    url = item["url"]
    color = get_tradition_color(item["tradition"])

    if "_local_file" in item and not force:
        filename = Path(item["_local_file"])
        if filename.exists():
            local_meta = process_local_file(filename, item, color)
            with data_lock:
                if local_meta:
                    metadata.append(local_meta)
                else:
                    metadata.append(_build_failure_metadata(item, color=color, error="Local file read error"))
            return
        else:
            logger.warning(f"{tid}: Local file {filename} not found, trying download")

    try:
        data = download_file(url)
        content_type = item.get("content_type", "")
        text = _extract_text(data, url, tid, content_type)

        if not text or not text.strip():
            raise ValueError("Empty content after conversion")

        stats = _finalize_text(text, url, tid)

        filename = corpus_text_path(settings.corpus_dir, item.get("major_tradition", "Unknown"), item["tradition"], tid)

        with data_lock:
            ensure_dir(filename.parent)
            filename.write_bytes(stats["data_utf8"])

        with data_lock:
            metadata.append(
                _build_metadata(item, path=str(filename.resolve()), color=color, stats=stats)
            )

        logger.info(f"Saved successfully: {filename.name} (words: {stats['word_count']}, color: {color})")

    except Exception as e:
        logger.exception("%s: Processing error", tid)
        with data_lock:
            metadata.append(_build_failure_metadata(item, color=color, error=str(e)))


def _update_traditions_info(force: bool) -> None:
    traditions: set[str] = set()
    if Path(settings.corpus_config_file).exists():
        with open(settings.corpus_config_file, encoding="utf-8") as f:
            for item in json.load(f):
                if "tradition" in item:
                    traditions.add(item["tradition"])

    info_file_path = settings.corpus_dir / "traditions.json"
    existing_info: dict = {}

    if info_file_path.exists():
        if force:
            backup_path = settings.corpus_dir / "traditions_backup.json"
            shutil.copy2(info_file_path, backup_path)
            logger.warning(f"force=True: old reference file saved as {backup_path.name}, creating a clean template.")
        else:
            try:
                with open(info_file_path, encoding="utf-8") as f:
                    existing_info = json.load(f)
            except Exception:
                logger.exception("Error reading traditions.json")

    changed = False
    for trad in sorted(traditions):
        color = get_tradition_color(trad)

        if trad not in existing_info:
            existing_info[trad] = {
                "description": "",
                "coordinates": [],
                "color": color,
            }
            changed = True
        elif "color" not in existing_info[trad]:
            existing_info[trad]["color"] = color
            changed = True

    with open(info_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)

    if changed and not force:
        logger.info("traditions.json updated.")


def build_corpus(force: bool = False):
    ensure_dir(settings.corpus_dir)
    metadata: list[dict] = []

    download_list = load_download_list(force)

    _update_traditions_info(force)

    logger.info(f"Starting multithreaded build (items: {len(download_list)})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.corpus.max_workers) as executor:
        futures = [executor.submit(process_single_item, item, force, metadata) for item in download_list]
        for future in concurrent.futures.as_completed(futures):
            exc = future.exception()
            if exc:
                logger.error(f"Unhandled error in worker thread: {exc}")

    with open(settings.corpus_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Corpus build complete.")
    logger.info(f"Total records: {len(metadata)}")

    available = [m for m in metadata if m.get("available", True)]
    logger.info(f"Available: {len(available)}")
    logger.info(f"Unavailable: {len(metadata) - len(available)}")

    if available:
        total_words = sum(m.get("word_count", 0) for m in available)
        total_sentences = sum(m.get("sentence_count", 0) for m in available)
        logger.info("\nOverall statistics for available texts:")
        logger.info(f"  Total words: {total_words}")
        logger.info(f"  Total sentences: {total_sentences}")
        logger.info(f"  Average words per text: {total_words / len(available):.1f}")
