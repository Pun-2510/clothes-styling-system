"""Shared CLI for generating catalog embeddings from either CLIP checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.clip_runtime import encode_products, load_clip, model_identity, write_embedding_metadata
from src.config import (ACTIVE_CLIP_MODEL, CLIP_MODEL, EMBEDDING_DIR,
                        EMBEDDING_BATCH_SIZE, PROCESSED_CSV, TEXT_EMBEDDING_BATCH_SIZE)


def main(modality="both"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=ACTIVE_CLIP_MODEL)
    parser.add_argument("--csv", type=Path, default=PROCESSED_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=(
        TEXT_EMBEDDING_BATCH_SIZE if modality == "text" else EMBEDDING_BATCH_SIZE))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.output_dir is None and args.model != CLIP_MODEL and EMBEDDING_DIR.name == "embeddings":
        parser.error("Use --output-dir embeddings/finetuned for a custom checkpoint.")
    output_dir = args.output_dir or EMBEDDING_DIR
    products = pd.read_csv(args.csv)
    device = torch.device(args.device)
    model, processor = load_clip(args.model, device)
    images, texts = encode_products(model, processor, products, device, args.batch_size, modality)
    identity = model_identity(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, features in (("image_embeddings.npy", images), ("text_embeddings.npy", texts)):
        if features is not None:
            path = output_dir / name
            np.save(path, features)
            write_embedding_metadata(path, identity, args.csv, features)
            print(f"Saved {features.shape}: {path}")


if __name__ == "__main__":
    main()
