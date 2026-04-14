"""Logging configuration used by training and evaluation scripts."""

import logging
import sys


def setup_logger(name: str = "trust_ssl", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a clean console handler.

    Calling this multiple times is safe: the logger is configured only on
    the first invocation.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
