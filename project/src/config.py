from pathlib import Path


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

EMBEDDING_DIR = (
    BASE_DIR /
    "embeddings"
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


# =========================================================
# CLIP
# =========================================================

CLIP_MODEL = (
    "openai/clip-vit-base-patch32"
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
CATEGORY_NEIGHBORS = 50


# =========================================================
# SOFT CATEGORY CONSTRAINT
# =========================================================

# Category không loại bỏ sản phẩm.
# Nó chỉ cộng thêm điểm nếu category
# của sản phẩm trùng với category dự đoán.

CATEGORY_BONUS = 0.05

# Brand chỉ là soft constraint
# BRAND_BONUS = 0.05
