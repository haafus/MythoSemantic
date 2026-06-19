import hashlib
import json
import logging
import random
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

_color_lock = threading.Lock()
_used_colors: set[str] = set()
_tradition_colors: dict[str, str] = {}
logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    name = (name or "").strip().replace("/", "_").replace(" ", "_")
    name = re.sub(r'[\\*?:"<>|]', "_", name)
    name = name.replace("..", "_")
    return name


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def normalize_catalog_id(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip())


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "".join(char for char in text if char in ("\n", "\t") or ord(char) >= 32)
    return text.strip()


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text, re.UNICODE))


def count_sentences(text: str) -> int:
    if not text:
        return 0
    sentences = re.split(r"[.!?…]+[\s\n]+", text)
    return len([s for s in sentences if s.strip()])


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def text_path(corpus_dir: Path, major_tradition: str, tradition: str, tid: str) -> Path:
    major = sanitize_filename(major_tradition)
    trad = sanitize_filename(tradition)
    title = sanitize_filename(tid)
    return corpus_dir / major / trad / f"{title}.txt"


def read_document(corpus_dir: Path, doc_id: str, major_tradition: str, tradition: str) -> tuple[str, str]:
    file_path = text_path(corpus_dir, major_tradition, tradition, doc_id)
    resolved = file_path.resolve()
    if not resolved.is_relative_to(corpus_dir.resolve()):
        raise PermissionError("Access denied")
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    title = sanitize_filename(doc_id)
    return resolved.read_text(encoding="utf-8"), title


def read_traditions(corpus_dir: Path) -> dict:
    path = corpus_dir / "traditions.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read %s: %s", path, e)
    return {}


def get_tradition_color(tradition: str) -> str:
    with _color_lock:
        if tradition in _tradition_colors:
            return _tradition_colors[tradition]

        while True:
            color = f"#{random.randint(0, 0xFFFFFF):06X}"
            if color not in _used_colors:
                _used_colors.add(color)
                _tradition_colors[tradition] = color
                return color
