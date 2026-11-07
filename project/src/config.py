import json
from pathlib import Path
import os


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = BASE_DIR / "data"

IMAGE_DIR = DATA_DIR / "data"

CSV_FILE = DATA_DIR / "data.csv"


# =========================================================
# PROCESSED DATA
# =========================================================

PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_CSV = (
    PROCESSED_DIR /
    "products.csv"
)

# Offline validation produces this project-local artifact with
# ``python -m src.tune_retrieval_weights``. When the file is absent, the
# conservative built-in values below remain available.
RETRIEVAL_WEIGHTS_PATH = Path(
    os.environ.get(
        "RETRIEVAL_WEIGHTS_PATH",
        str(BASE_DIR / "retrieval_weights.json"),
    )
)


def _load_retrieval_weights():
    if not RETRIEVAL_WEIGHTS_PATH.is_file():
        return {}
    try:
        payload = json.loads(RETRIEVAL_WEIGHTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read retrieval weights: {RETRIEVAL_WEIGHTS_PATH}"
        ) from error
    weights = payload.get("weights", payload)
    if not isinstance(weights, dict):
        raise ValueError("retrieval_weights.json must contain a weights object.")
    return weights


RETRIEVAL_WEIGHTS = _load_retrieval_weights()


def _configured_float(name, fallback):
    # Explicit process environment values remain useful for controlled runs.
    # Otherwise the validation-selected artifact is authoritative.
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        raw_value = RETRIEVAL_WEIGHTS.get(name, fallback)
    try:
        return float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error


# =========================================================
# EMBEDDINGS
# =========================================================

# EMBEDDING_DIR = (
#     Path(os.environ.get("CLIP_EMBEDDING_DIR", str(BASE_DIR / "embeddings")))
# )

EMBEDDING_DIR = (
    Path(os.environ.get(
        "CLIP_EMBEDDING_DIR",
        str(BASE_DIR / "embeddings")
    ))
)

EMBEDDINGS_FILE = (
    EMBEDDING_DIR /
    "image_embeddings.npy"
)

TEXT_EMBEDDINGS_FILE = (
    EMBEDDING_DIR /
    "text_embeddings.npy"
)


# =========================================================
# LOGGING
# =========================================================

LOG_DIR = (
    BASE_DIR /
    "logs"
)

LOG_FILE = (
    LOG_DIR /
    "recommendation.log"
)


# =========================================================
# DATASET SETTINGS
# =========================================================

MAX_PRODUCTS = 5000

RANDOM_SEED = 42

# Target floor used by catalog sampling and dataset audits. When the requested
# catalog is too small to reach the floor for every category, the sampler
# distributes the available budget as evenly as possible and reports warnings.
MIN_PRODUCTS_PER_CATEGORY = int(
    os.environ.get("MIN_PRODUCTS_PER_CATEGORY", "25")
)

if MIN_PRODUCTS_PER_CATEGORY < 1:
    raise ValueError("MIN_PRODUCTS_PER_CATEGORY must be at least 1.")

# Mild inverse-frequency sampling for CLIP fine-tuning. Square-root balancing
# avoids allowing one-sample categories to dominate every epoch.
CATEGORY_BALANCE_POWER = float(
    os.environ.get("CATEGORY_BALANCE_POWER", "0.50")
)
CATEGORY_MAX_SAMPLE_WEIGHT = float(
    os.environ.get("CATEGORY_MAX_SAMPLE_WEIGHT", "5.0")
)

if not 0.0 <= CATEGORY_BALANCE_POWER <= 1.0:
    raise ValueError("CATEGORY_BALANCE_POWER must be between 0 and 1.")
if CATEGORY_MAX_SAMPLE_WEIGHT < 1.0:
    raise ValueError("CATEGORY_MAX_SAMPLE_WEIGHT must be at least 1.")


# =========================================================
# CLIP
# =========================================================

# CLIP_MODEL = (
#     "openai/clip-vit-base-patch32"
# )

# # Serving and catalog encoding can use a local fine-tuned checkpoint.
# ACTIVE_CLIP_MODEL = os.environ.get("CLIP_MODEL_PATH", CLIP_MODEL)

CLIP_MODEL = (
    "openai/clip-vit-base-patch32"
)

FINETUNED_CLIP_MODEL = (
    BASE_DIR
    / "runs"
    / "clip_finetune"
    / "best"
)

ACTIVE_CLIP_MODEL = os.environ.get(
    "CLIP_MODEL_PATH",
    str(FINETUNED_CLIP_MODEL)
)

# Vietnamese -> English translation before CLIP text encoding
TRANSLATION_MODEL = (
    "Helsinki-NLP/opus-mt-vi-en"
)


# =========================================================
# EMBEDDING
# =========================================================

EMBEDDING_BATCH_SIZE = 8

TEXT_EMBEDDING_BATCH_SIZE = 64


# =========================================================
# RECOMMENDATION
# =========================================================

TOP_K = 5

# Text search can compare a text query with both catalog images and catalog
# descriptions. This value controls that internal text-retrieval mixture.
TEXT_RETRIEVAL_IMAGE_WEIGHT = _configured_float(
    "TEXT_RETRIEVAL_IMAGE_WEIGHT", 0.50
)

# Default balance for a query that contains both an image and text. A value of
# 0.50 gives the two query modalities equal influence on the final ranking.
COMBINED_IMAGE_WEIGHT = _configured_float("COMBINED_IMAGE_WEIGHT", 0.50)

# Combined search treats explicit text attributes as requested edits to the
# reference image. These weights are applied after raw cosine fusion.
COMBINED_CATEGORY_WEIGHT = _configured_float(
    "COMBINED_CATEGORY_WEIGHT", 0.10
)
COMBINED_ATTRIBUTE_WEIGHT = _configured_float(
    "COMBINED_ATTRIBUTE_WEIGHT", 0.25
)

for setting_name, setting_value in (
    ("TEXT_RETRIEVAL_IMAGE_WEIGHT", TEXT_RETRIEVAL_IMAGE_WEIGHT),
    ("COMBINED_IMAGE_WEIGHT", COMBINED_IMAGE_WEIGHT),
    ("COMBINED_CATEGORY_WEIGHT", COMBINED_CATEGORY_WEIGHT),
    ("COMBINED_ATTRIBUTE_WEIGHT", COMBINED_ATTRIBUTE_WEIGHT),
):
    if not 0.0 <= setting_value <= 1.0:
        raise ValueError(f"{setting_name} must be between 0 and 1.")


# =========================================================
# CATEGORY
# =========================================================

# Số nearest neighbors dùng để
# dự đoán category của ảnh query
CATEGORY_TOP_PER_CLASS = int(
    os.environ.get("CATEGORY_TOP_PER_CLASS", "3")
)

if CATEGORY_TOP_PER_CLASS < 1:
    raise ValueError("CATEGORY_TOP_PER_CLASS must be at least 1.")

# Blend balanced visual-neighbour evidence with zero-shot CLIP category
# prompts. A value of 1.0 gives both raw evidence terms equal scale before the
# confidence-weighted category bonus is applied.
CATEGORY_PROTOTYPE_WEIGHT = _configured_float(
    "CATEGORY_PROTOTYPE_WEIGHT", 1.0
)

if CATEGORY_PROTOTYPE_WEIGHT < 0.0:
    raise ValueError("CATEGORY_PROTOTYPE_WEIGHT cannot be negative.")

# Category relationships are inferred automatically from normalized CLIP text
# prototypes. Two labels are related only when they are mutual nearest
# neighbours, avoiding a manually maintained category list.
CATEGORY_RELATION_NEIGHBORS = int(
    RETRIEVAL_WEIGHTS.get("CATEGORY_RELATION_NEIGHBORS", 5)
)

if CATEGORY_RELATION_NEIGHBORS < 1:
    raise ValueError("CATEGORY_RELATION_NEIGHBORS must be at least 1.")

# Ambiguous related labels receive asymmetric bonuses so the leading category
# remains preferred. An unrelated runner-up receives no bonus, while Top-1
# keeps only a very small signal instead of falling back completely.
RELATED_TOP1_BONUS_FACTOR = _configured_float(
    "RELATED_TOP1_BONUS_FACTOR", 0.50
)
RELATED_TOP2_BONUS_FACTOR = _configured_float(
    "RELATED_TOP2_BONUS_FACTOR", 0.20
)
UNRELATED_TOP1_BONUS_FACTOR = _configured_float(
    "UNRELATED_TOP1_BONUS_FACTOR", 0.10
)

for setting_name, setting_value in (
    ("RELATED_TOP1_BONUS_FACTOR", RELATED_TOP1_BONUS_FACTOR),
    ("RELATED_TOP2_BONUS_FACTOR", RELATED_TOP2_BONUS_FACTOR),
    ("UNRELATED_TOP1_BONUS_FACTOR", UNRELATED_TOP1_BONUS_FACTOR),
):
    if not 0.0 <= setting_value <= 1.0:
        raise ValueError(f"{setting_name} must be between 0 and 1.")

if RELATED_TOP2_BONUS_FACTOR > RELATED_TOP1_BONUS_FACTOR:
    raise ValueError(
        "RELATED_TOP2_BONUS_FACTOR cannot exceed "
        "RELATED_TOP1_BONUS_FACTOR."
    )

# Confidence at or below this floor has no influence on ranking. Between the
# floor and the full threshold, category influence increases linearly.
CATEGORY_CONFIDENCE_FLOOR = float(
    os.environ.get("CATEGORY_CONFIDENCE_FLOOR", "0.50")
)

# Confidence at or above this threshold receives the full configured bonus.
CATEGORY_CONFIDENCE_THRESHOLD = float(
    os.environ.get("CATEGORY_CONFIDENCE_THRESHOLD", "0.60")
)

if not 0.0 <= CATEGORY_CONFIDENCE_FLOOR < CATEGORY_CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError(
        "Category thresholds must satisfy 0 <= CATEGORY_CONFIDENCE_FLOOR "
        "< CATEGORY_CONFIDENCE_THRESHOLD <= 1."
    )


# =========================================================
# SOFT CATEGORY CONSTRAINT
# =========================================================

# Category không loại bỏ sản phẩm.
# Nó chỉ cộng thêm điểm nếu category
# của sản phẩm trùng với category dự đoán.

CATEGORY_BONUS = _configured_float("CATEGORY_BONUS", 0.05)

if CATEGORY_BONUS < 0.0:
    raise ValueError("CATEGORY_BONUS cannot be negative.")

# Brand chỉ là soft constraint
# BRAND_BONUS = 0.05
