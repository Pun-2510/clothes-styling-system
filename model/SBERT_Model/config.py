from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent.parent / "project"

MODEL_PATH = str(CURRENT_DIR / "bert_fashion_model")

BASE_MODEL = "bert-base-uncased"

DATASET_NAME = str(PROJECT_DIR / "data" / "processed" / "products.csv")

TEXT_COLUMN = "product_name"

LABEL_COLUMN = "category"

MAX_SAMPLES = 5000

MAX_LENGTH = 64

TOP_K = 10

SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EMBEDDING_PATH = str(CURRENT_DIR / "embeddings" / "fashion_embeddings.npy")

EMBEDDING_DTYPE = "float16"

MIN_SIMILARITY = 0.0

TEST_SIZE = 0.2
NUM_EPOCHS = 3
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42
