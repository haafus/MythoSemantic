import logging

from settings import settings

LEVEL_COLORS = {
    logging.DEBUG: "\033[90m",     # gray
    logging.INFO: "\033[36m",      # cyan
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str):
        super().__init__(fmt)
        self._formatters = {
            level: logging.Formatter(f"{color}{fmt}{RESET}")
            for level, color in LEVEL_COLORS.items()
        }

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._formatters.get(record.levelno, self._default_formatter)
        return formatter.format(record)

    @property
    def _default_formatter(self) -> logging.Formatter:
        return logging.Formatter(self._fmt)


def setup_logging() -> None:
    log_dir = settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_ColorFormatter(fmt))

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
            stream_handler,
        ],
    )
