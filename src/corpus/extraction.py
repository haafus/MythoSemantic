"""Heavy text-extraction helpers (PDF, HTML).

Separated from utils.py so that modules needing only lightweight helpers
(sanitize, counts, paths) don't pull in pymupdf / trafilatura / bs4.
"""

from __future__ import annotations

import logging
import re

import pymupdf
import trafilatura
from bs4 import BeautifulSoup

from .utils import normalize_text

logger = logging.getLogger(__name__)


def _decode_bytes(content: bytes) -> str:
    encodings_to_try = ["utf-8", "windows-1251", "iso-8859-1", "cp1251", "gb2312"]
    for enc in encodings_to_try:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def html_to_text(
    html_content: bytes, include_comments: bool = False, include_tables: bool = True, target_language: str | None = None
) -> str:
    if not html_content:
        return ""

    try:
        text = trafilatura.extract(
            html_content,
            include_comments=include_comments,
            include_tables=include_tables,
            target_language=target_language,
            favor_precision=True,
        )
        if text:
            logger.debug("HTML processed with Trafilatura")
            return normalize_text(text)
    except Exception as e:
        logger.warning(f"Trafilatura failed, falling back to BeautifulSoup: {e}")

    try:
        decoded = _decode_bytes(html_content)
        soup = BeautifulSoup(decoded, "html.parser")

        for element in soup(
            ["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "form", "button"]
        ):
            element.decompose()

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(role="main")
            or soup.find("div", class_=re.compile(r"content|main|article", re.I))
        )

        text = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")
        logger.debug("HTML processed with BeautifulSoup")
        return normalize_text(text)

    except Exception:
        logger.exception("Critical HTML extraction error")
        return ""


def pdf_to_text(
    pdf_content: bytes, extract_tables: bool = False, preserve_layout: bool = True, ocr_fallback: bool = False
) -> str:
    if not pdf_content:
        return ""

    text_parts = []
    try:
        doc = pymupdf.open(stream=pdf_content, filetype="pdf")

        if doc.is_encrypted:
            if not doc.authenticate(""):
                logger.warning("Encrypted PDF could not be decrypted")
                doc.close()
                return ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            if preserve_layout:
                blocks = page.get_text("blocks")
                page_text = "\n".join(block[4] for block in blocks if block[4].strip())
            else:
                page_text = page.get_text()

            if extract_tables:
                tables = page.find_tables()
                if tables and tables.tables:
                    table_texts = []
                    for table in tables.tables:
                        table_str = table.to_text()
                        if table_str:
                            table_texts.append(f"[TABLE]\n{table_str}\n[/TABLE]")
                    if table_texts:
                        page_text = (page_text + "\n\n" if page_text else "") + "\n\n".join(table_texts)

            if page_text:
                text_parts.append(page_text)

        doc.close()

        if text_parts:
            logger.debug(f"PDF processed with PyMuPDF ({len(text_parts)} pages)")
            return normalize_text("\n\n".join(text_parts))
        else:
            logger.warning("Failed to extract text from PDF")
            return ""

    except Exception:
        logger.exception("Error processing PDF with PyMuPDF")
        return ""
