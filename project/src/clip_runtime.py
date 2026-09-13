"""CLIP encoding and catalog provenance shared by offline commands and serving."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from src.clip_data import build_product_text, file_sha256, resolve_image_path


def load_clip(source, device):
    processor = AutoProcessor.from_pretrained(str(source))
    model = CLIPModel.from_pretrained(str(source)).to(device)
    model.eval()
    return model, processor


def encode_products(model, processor, products, device, batch_size=16, modality="both"):
    if batch_size < 1 or products.empty:
        raise ValueError("A positive batch size and a nonempty dataset are required.")
    if modality not in {"image", "text", "both"}:
        raise ValueError("modality must be image, text or both.")
    model.eval()
    image_features, text_features = [], []
    with torch.inference_mode():
        for start in tqdm(range(0, len(products), batch_size), desc=f"Encoding {modality}"):
            batch = products.iloc[start:start + batch_size]
            if modality in {"image", "both"}:
                images = []
                for path in batch.image_path:
                    with Image.open(resolve_image_path(path)) as image:
                        images.append(image.convert("RGB"))
                inputs = processor(images=images, return_tensors="pt").to(device)
                output = model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
                features = model.visual_projection(output.pooler_output)
                image_features.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
            if modality in {"text", "both"}:
                texts = [build_product_text(row) for _, row in batch.iterrows()]
                inputs = processor(text=texts, padding=True, truncation=True,
                                   max_length=model.config.text_config.max_position_embeddings,
                                   return_tensors="pt").to(device)
                output = model.text_model(input_ids=inputs["input_ids"],
                                          attention_mask=inputs.get("attention_mask"),
                                          return_dict=True)
                features = model.text_projection(output.pooler_output)
                text_features.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
    return (
        np.concatenate(image_features).astype(np.float32) if image_features else None,
        np.concatenate(text_features).astype(np.float32) if text_features else None,
    )


def model_identity(source):
    path = Path(source)
    if not path.is_dir():
        return {"source": str(source)}
    files = sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin"))
                   | set(path.glob("*.json")) | set(path.glob("*.txt")))
    # A local checkpoint can be mounted at a different absolute path in Docker.
    # File hashes, rather than its host path, are its portable identity.
    return {"files": {item.name: file_sha256(item) for item in files}}


def same_model_identity(expected, actual):
    """Compare local checkpoints by content while retaining Hub-ID checks."""
    expected_files = expected.get("files")
    actual_files = actual.get("files")
    if expected_files is not None or actual_files is not None:
        return expected_files is not None and expected_files == actual_files
    return expected == actual


def write_embedding_metadata(path, identity, csv_path, embeddings):
    metadata = {"model": identity, "products_sha256": file_sha256(csv_path),
                "shape": list(embeddings.shape), "embeddings_sha256": file_sha256(path)}
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_embedding_metadata(path, identity, csv_path, allow_legacy=False):
    metadata_path = Path(str(path) + ".json")
    if not metadata_path.exists():
        if allow_legacy:
            return
        raise ValueError(f"Missing provenance for {path}; regenerate embeddings for this model.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (not same_model_identity(metadata["model"], identity)
            or metadata["products_sha256"] != file_sha256(csv_path)
            or metadata["embeddings_sha256"] != file_sha256(path)):
        raise ValueError(f"Model, catalog or embeddings changed: regenerate {path}.")
