"""
    Sets up timestamped logging to stdout for pipeline monitoring.
"""

import logging
import sys

def setup_logging():
    """
        Set up a logger that outputs to stdout with INFO level and timestamps.

        Returns:
            logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("robust-train")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger