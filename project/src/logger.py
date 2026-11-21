import logging
import os
from pathlib import Path


# =========================================================
# PATH
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

LOG_DIR = BASE_DIR / "logs"

LOG_FILE = (
    LOG_DIR /
    os.environ.get("RECOMMENDATION_LOG_NAME", "recommendation.log")
)


# =========================================================
# CREATE LOG DIRECTORY
# =========================================================

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOGGER
# =========================================================

logger = logging.getLogger(
    "fashion_recommendation"
)

logger.setLevel(
    logging.INFO
)


# =========================================================
# HANDLERS
# =========================================================

if not logger.handlers:

    console_handler = (
        logging.StreamHandler()
    )

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    # Logging must never prevent the API or local tests from starting.  On
    # Windows, a bind-mounted log can be locked by a running Docker container.
    # Keep console logging available and attach the file handler only when the
    # configured path can be opened.
    try:
        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        )
        file_handler.setFormatter(
            formatter
        )
        logger.addHandler(
            file_handler
        )
    except OSError as exc:
        logger.warning(
            "File logging unavailable at %s; continuing with console logging: %s",
            LOG_FILE,
            exc,
        )


# =========================================================
# HELPERS
# =========================================================

def log_info(message):
    logger.info(message)


def log_warning(message):
    logger.warning(message)


def log_error(message):
    logger.error(message)


def log_exception(message):
    logger.exception(message)
