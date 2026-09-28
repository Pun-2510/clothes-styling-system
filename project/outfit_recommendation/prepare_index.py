"""Prepare an independent CLIP/Polyvore index for the Streamlit prototype."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone Polyvore outfit retrieval index."
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get(
            "POLYVORE_DATA_DIR", str(PROJECT_ROOT / "datasets" / "polyvore-outfits")
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "OUTFIT_ARTIFACT_DIR",
            str(PROJECT_ROOT / "outfit_recommendation" / "artifacts"),
        ),
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument(
        "--max-outfits",
        type=int,
        default=1000,
        help="Number of annotated outfits to index; use 0 for the full split.",
    )
    parser.add_argument("--min-items", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model",
        default=os.environ.get("OUTFIT_CLIP_MODEL", "openai/clip-vit-base-patch32"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing index. Unreferenced old image files are harmless.",
    )
    return parser.parse_args()


def annotation_name(split: str) -> str:
    return "valid.json" if split == "validation" else f"{split}.json"


def choose_outfits(
    outfits: list[dict[str, Any]], max_outfits: int, min_items: int, seed: int
) -> list[dict[str, Any]]:
    eligible = [outfit for outfit in outfits if len(outfit.get("items", [])) >= min_items]
    random.Random(seed).shuffle(eligible)
    return eligible if max_outfits == 0 else eligible[:max_outfits]


def item_metadata(item_id: str, metadata: dict[str, Any]) -> dict[str, str]:
    raw = metadata.get(item_id, {})
    return {
        "title": str(raw.get("title") or raw.get("url_name") or f"Item {item_id}"),
        "description": str(raw.get("description") or ""),
        "semantic_category": str(raw.get("semantic_category") or "unknown"),
        "category_id": str(raw.get("category_id") or ""),
    }


def encode_images(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.vision_model(
            pixel_values=inputs["pixel_values"], return_dict=True
        )
        features = model.visual_projection(output.pooler_output)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.max_outfits < 0:
        raise ValueError("--max-outfits cannot be negative.")
    if args.min_items < 2:
        raise ValueError("--min-items must be at least 2.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    dataset_dir = resolve_project_path(args.dataset_dir)
    output_dir = resolve_project_path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists. Use a new --output-dir or pass --overwrite."
        )

    annotation_path = dataset_dir / "disjoint" / annotation_name(args.split)
    parquet_path = dataset_dir / "data" / "disjoint" / f"{args.split}.parquet"
    metadata_path = dataset_dir / "polyvore_item_metadata.json"
    missing = [
        str(path)
        for path in (annotation_path, parquet_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing Polyvore files:\n" + "\n".join(missing))

    raw_outfits = json.loads(annotation_path.read_text(encoding="utf-8"))
    selected_outfits = choose_outfits(
        raw_outfits, args.max_outfits, args.min_items, args.seed
    )
    selected_ids = {
        str(item["item_id"])
        for outfit in selected_outfits
        for item in outfit["items"]
    }
    if not selected_ids:
        raise ValueError("No items were selected from the requested split.")

    print(
        f"Selected {len(selected_outfits)} outfits and {len(selected_ids)} unique items."
    )
    print(f"Loading CLIP model: {args.model}")
    from transformers import AutoProcessor, CLIPModel

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    item_records: list[dict[str, Any]] = []
    feature_batches: list[np.ndarray] = []
    pending: list[tuple[str, bytes, Image.Image]] = []

    def flush_pending() -> None:
        if not pending:
            return
        images = [record[2] for record in pending]
        features = encode_images(model, processor, images, device)
        feature_batches.append(features)
        for (item_id, raw_bytes, image), _feature in zip(pending, features):
            relative_image = Path("images") / f"{item_id}.jpg"
            (output_dir / relative_image).write_bytes(raw_bytes)
            item_records.append(
                {
                    "item_id": item_id,
                    "image_path": relative_image.as_posix(),
                    **item_metadata(item_id, metadata),
                }
            )
            image.close()
        pending.clear()

    found: set[str] = set()
    parquet = pq.ParquetFile(parquet_path)
    with tqdm(total=len(selected_ids), desc="Encoding selected Polyvore items") as progress:
        for batch in parquet.iter_batches(batch_size=512, columns=["item_id", "image"]):
            for row in batch.to_pylist():
                item_id = str(row["item_id"])
                if item_id not in selected_ids or item_id in found:
                    continue
                try:
                    raw_bytes = row["image"]["bytes"]
                    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                except Exception as error:
                    print(f"Skipping unreadable item {item_id}: {error}")
                    continue
                pending.append((item_id, raw_bytes, image))
                found.add(item_id)
                progress.update(1)
                if len(pending) >= args.batch_size:
                    flush_pending()
        flush_pending()

    if not item_records:
        raise RuntimeError("No selected item images were found in the Parquet file.")

    available = {record["item_id"] for record in item_records}
    prepared_outfits = []
    for outfit in selected_outfits:
        item_ids = [
            str(item["item_id"])
            for item in outfit["items"]
            if str(item["item_id"]) in available
        ]
        if len(item_ids) >= args.min_items:
            prepared_outfits.append(
                {"outfit_id": str(outfit["set_id"]), "item_ids": item_ids}
            )

    if not prepared_outfits:
        raise RuntimeError("No complete outfits remain after matching images.")

    embeddings = np.concatenate(feature_batches).astype(np.float32)
    np.save(output_dir / "item_embeddings.npy", embeddings)
    (output_dir / "catalog.json").write_text(
        json.dumps(
            {"items": item_records, "outfits": prepared_outfits},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "model_source": str(args.model),
        "seed": args.seed,
        "requested_outfits": args.max_outfits,
        "minimum_items": args.min_items,
        "outfit_count": len(prepared_outfits),
        "item_count": len(item_records),
        "embedding_shape": list(embeddings.shape),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared {len(prepared_outfits)} outfits / {len(item_records)} items "
        f"in {output_dir}"
    )


if __name__ == "__main__":
    main()
