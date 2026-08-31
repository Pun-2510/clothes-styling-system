"""Generate normalized CLIP embeddings for the catalog product text.

Run from the project root with ``python -m src.generate_text_embeddings``.
The output row order matches ``products.csv`` and the image embeddings.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CLIP_MODEL,
    PROCESSED_CSV,
    TEXT_EMBEDDINGS_FILE,
    EMBEDDING_DIR,
    TEXT_EMBEDDING_BATCH_SIZE,
)


def build_product_text(row):
    """Create a compact CLIP prompt from product metadata."""
    parts = []
    for label, column in (
        ("Product", "product_name"),
        ("Category", "category"),
        ("Description", "description"),
    ):
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            parts.append(f"{label}: {str(value).strip()}")
    return ". ".join(parts) or "Fashion product"


def main():
    print("=" * 60)
    print("GENERATE CLIP TEXT EMBEDDINGS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    products = pd.read_csv(PROCESSED_CSV)
    if products.empty:
        raise RuntimeError("products.csv is empty.")

    processor = AutoProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()
    max_length = model.config.text_config.max_position_embeddings
    all_embeddings = []

    for start in tqdm(
        range(0, len(products), TEXT_EMBEDDING_BATCH_SIZE),
        desc="Generating text embeddings",
    ):
        batch = products.iloc[start : start + TEXT_EMBEDDING_BATCH_SIZE]
        texts = [build_product_text(row) for _, row in batch.iterrows()]
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            features = model.text_projection(text_outputs.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        all_embeddings.append(features.detach().cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(all_embeddings, axis=0)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TEXT_EMBEDDINGS_FILE, embeddings)
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Saved to: {TEXT_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
