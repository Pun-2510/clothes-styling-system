import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

# Model ResNet cũ
RESNET_DIR = Path(
    os.getenv("RESNET_DIR", PROJECT_DIR / "ResNet_Model")
)

# File model đã train
RESNET_MODEL_PATH = RESNET_DIR / os.getenv(
    "RESNET_MODEL_FILE",
    "../Web_Test/models/resnet_outfit.pth"
)
GENDER_MODEL_PATH = RESNET_DIR.parent / "Web_Test" / "models" / "resnet_gender.pth"

# Search
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_IMAGES = max(1, min(int(os.getenv("MAX_IMAGES", "3000")), 3000))
ENABLE_PATTERN_SEARCH = (
    os.getenv("ENABLE_PATTERN_SEARCH", "false").lower() == "true"
)
COLOR_WEIGHT = float(os.getenv("COLOR_WEIGHT", "0.10"))
FILTER_BY_CATEGORY = (
    os.getenv("FILTER_BY_CATEGORY", "true").lower() == "true"
)

RESULTS_DIR = CURRENT_DIR / "results"
