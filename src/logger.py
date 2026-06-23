"""
logger.py
---------
Structured logging for the RAG project.

Replaces print() statements with proper logging that:
- Shows timestamps and log levels
- Can be silenced in tests without code changes
- Can be redirected to files without touching business logic
- Differentiates INFO (normal flow) from WARNING and ERROR

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("vectorstore loaded", extra={"chunks": 42})
"""

import logging
import sys
from pathlib import Path

from src.config import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Calling get_logger(__name__) in each module produces loggers like:
        src.vectorstore, src.agent, src.loader
    This makes it easy to trace which module emitted each log line.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — INFO and above to stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — DEBUG and above to logs/rag.log
    LOGS_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOGS_DIR / "rag.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
