import logging
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
    "recommendation.log"
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

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    console_handler = (
        logging.StreamHandler()
    )

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(
        formatter
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
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