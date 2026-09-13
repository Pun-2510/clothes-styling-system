"""Generate catalog text embeddings; accepts --model and --output-dir."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.clip_data import build_product_text  # Backward-compatible public helper.
from src.generate_clip_embeddings import main as generate_embeddings


def main():
    generate_embeddings("text")


if __name__ == "__main__":
    main()
