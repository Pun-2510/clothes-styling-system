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
    "resnet_outfit.pth"
)

# Search
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "2000"))
FILTER_BY_CATEGORY = (
    os.getenv("FILTER_BY_CATEGORY", "true").lower() == "true"
)

RESULTS_DIR = CURRENT_DIR / "results"