"""
PhantomTouch Logging Facility.
Provides formatted, multi-level logging to console and optional log files.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "PhantomTouch", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with standard stream formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%H:%M:%S"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
