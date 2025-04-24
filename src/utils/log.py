"""Module defining utility to instantiate python logging."""

import logging

from rich.logging import RichHandler


def rich_logger(level="INFO"):
    """Prepare python logging for rich print."""
    logging.basicConfig(
        level=level, handlers=[RichHandler()], format="%(message)s", datefmt="[%X]"
    )
