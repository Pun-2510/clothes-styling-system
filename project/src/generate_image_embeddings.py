"""Generate catalog image embeddings; accepts --model and --output-dir."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generate_clip_embeddings import main as generate_embeddings


def main():
    generate_embeddings("image")


if __name__ == "__main__":
    main()
