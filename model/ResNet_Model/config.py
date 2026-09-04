import torch

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

MAX_IMAGES = 2000

# ============================================================

# MODEL

# ============================================================

MODEL_NAME = "resnet18"

NUM_CLASSES = len(SELECTED_CLASSES)

MODEL_PATH = "resnet_outfit.pth"

BEST_MODEL_PATH = "best_model.pth"

# ============================================================

# TRAINING

# ============================================================

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

TEST_SIZE = 0.2

RANDOM_STATE = 42

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
