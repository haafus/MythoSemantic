from __future__ import annotations

import hashlib
import logging
import random
import re
import threading
import unicodedata
from pathlib import Path

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


def corpus_text_path(corpus_dir: Path, major_tradition: str, tradition: str, tid: str) -> Path:
    """Canonical on-disk location of a corpus text: <major>/<tradition>/<title>.txt"""
    major = sanitize_filename(major_tradition)
    trad = sanitize_filename(tradition)
    title = sanitize_filename(tid)
    return corpus_dir / major / trad / f"{title}.txt"


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
