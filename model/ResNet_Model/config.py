import torch
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ============================================================

# DEVICE

# ============================================================

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================

# DATASET

# ============================================================

DATASET_NAME = "ashraq/fashion-product-images-small"

DATASET_SPLIT = "train"

SELECTED_CLASSES = [
"Tshirts",
"Shirts",
"Jeans",
"Trousers",
"Dresses",
"Jackets"
]

# Add one folder per extra class, for example custom_data/Cheongsam/*.jpg.
CUSTOM_DATA_DIR = ROOT / "custom_data"
CUSTOM_CLASSES = []
MIN_CUSTOM_IMAGES_PER_CLASS = 20

MAX_IMAGES = 3000

# ============================================================

# MODEL

# ============================================================

MODEL_NAME = "resnet18"

NUM_CLASSES = len(SELECTED_CLASSES)

MODEL_PATH = Path(os.getenv(
    "RESNET_MODEL_PATH",
    str(ROOT.parent / "Web_Test" / "models" / "resnet_outfit.pth"),
))

BEST_MODEL_PATH = Path(os.getenv("RESNET_BEST_MODEL_PATH", str(ROOT / "best_model.pth")))

# ============================================================

# TRAINING

# ============================================================

BATCH_SIZE = 16

EPOCHS = int(os.getenv("RESNET_EPOCHS", "8"))

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

TEST_SIZE = 0.2

RANDOM_STATE = 42

LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 3
LR_PATIENCE = 2
LR_FACTOR = 0.3
MIN_LEARNING_RATE = 1e-6
GRADIENT_CLIP_NORM = 1.0
CONTRASTIVE_WEIGHT = 0.10
CONTRASTIVE_TEMPERATURE = 0.07

# ============================================================

# IMAGE

# ============================================================

IMAGE_SIZE = 224

RESIZE_SIZE = 256

NORMALIZE_MEAN = [
0.485,
0.456,
0.406
]

NORMALIZE_STD = [
0.229,
0.224,
0.225
]
