import json
import logging
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from settings import settings

from .utils import corpus_text_path

logger = logging.getLogger(__name__)

# Lazily initialized: creating a session / UserAgent at import time would make
# any `import corpus.*` pay for it (UserAgent may even hit the network).
_http_session: requests.Session | None = None
_user_agent = None


def create_retry_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=settings.corpus.retry_total,
        backoff_factor=settings.corpus.retry_backoff_factor,
        status_forcelist=settings.corpus.retry_status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = create_retry_session()
    return _http_session


def _get_user_agent():
    global _user_agent
    if _user_agent is None:
        from fake_useragent import UserAgent

        _user_agent = UserAgent(fallback="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    return _user_agent


def load_download_list(force: bool = False) -> list[dict]:
    if not Path(settings.corpus_config_file).exists():
        logger.error(f"Download list file not found: {settings.corpus_config_file}")
        return []

    with open(settings.corpus_config_file, encoding="utf-8") as f:
        items = json.load(f)

    seen_urls = set()
    filtered = []

    for item in items:
        url = item["url"]
        tid = item.get("title", item.get("id", "unknown_id"))
        tradition = item["tradition"]

        if url in seen_urls:
            logger.warning(f"Duplicate URL: {url}, skipping")
            continue

        filename = corpus_text_path(settings.corpus_dir, item.get("major_tradition", "Unknown"), tradition, tid)

        if filename.exists() and not force:
            logger.info(f"File exists for {tid}, will be processed locally")
            new_item = item.copy()
            new_item["_local_file"] = str(filename)
            seen_urls.add(url)
            filtered.append(new_item)
        else:
            seen_urls.add(url)
            filtered.append(item)

    return filtered


def download_file(url: str) -> bytes:
    logger.info(f"Downloading: {url}")
    headers = {
        "User-Agent": _get_user_agent().random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    }
    response = _get_http_session().get(url, headers=headers, timeout=(settings.corpus.timeout_connect, settings.corpus.timeout_read))
    response.raise_for_status()
    content: bytes = response.content
    return content
