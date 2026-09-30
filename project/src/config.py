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
    os.environ.get("MIN_PRODUCTS_PER_CATEGORY", "10")
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
    / "clip_finetune_3epochs_local"
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

# Soft category is only applied when the weighted-neighbor vote is
# sufficiently decisive. Otherwise image retrieval ignores category.
CATEGORY_CONFIDENCE_THRESHOLD = float(
    os.environ.get("CATEGORY_CONFIDENCE_THRESHOLD", "0.60")
)

if not 0.0 <= CATEGORY_CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError(
        "CATEGORY_CONFIDENCE_THRESHOLD must be between 0 and 1."
    )


# =========================================================
# SOFT CATEGORY CONSTRAINT
# =========================================================

# Category không loại bỏ sản phẩm.
# Nó chỉ cộng thêm điểm nếu category
# của sản phẩm trùng với category dự đoán.

CATEGORY_BONUS = 0.05

# Brand chỉ là soft constraint
# BRAND_BONUS = 0.05
