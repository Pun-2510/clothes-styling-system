MODEL_PATH = "./bert_fashion_model"

BASE_MODEL = "bert-base-uncased"

DATASET_NAME = "nreimers/fashion-dataset"

TEXT_COLUMN = "productDisplayName"

LABEL_COLUMN = "articleType"

MAX_SAMPLES = 5000

MAX_LENGTH = 64

TOP_K = 10

SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EMBEDDING_PATH = "./embeddings/fashion_embeddings.npy"

EMBEDDING_DTYPE = "float16"

MIN_SIMILARITY = 0.0