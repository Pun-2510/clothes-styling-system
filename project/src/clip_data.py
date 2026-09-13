"""Shared product prompts and reproducible, image-disjoint experiment splits."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import BASE_DIR


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_product_text(row):
    """Use the same English metadata prompt for training and catalog encoding."""
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


def resolve_image_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else BASE_DIR / path


def prepare_splits(csv_path, seed=42, val_fraction=0.1, test_fraction=0.1):
    """Deduplicate exact images; keep identical prompts/products in one split.

    Rare categories with fewer than three groups stay in train. Per-category
    rounding means the actual fractions may differ from requested fractions.
    Missing/corrupt images fail explicitly instead of changing evaluation rows.
    """
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to < 1.")
    products = pd.read_csv(csv_path, dtype={"product_id": str})
    required = {"product_id", "image_path", "category", "product_name"}
    if not required.issubset(products.columns) or products.empty:
        raise ValueError(f"Nonempty CSV must contain {sorted(required)}.")
    if products[list(required)].isna().any().any():
        raise ValueError("Product IDs, image paths, categories and names cannot be missing.")
    for column in required:
        if products[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} cannot be blank.")
    products["image_path"] = products.image_path.map(
        lambda value: str(resolve_image_path(value).resolve())
    )
    hashes = []
    for path in products.image_path:
        with Image.open(path) as image:
            image.verify()
        hashes.append(file_sha256(path))
    products["image_sha256"] = hashes
    before = len(products)
    products = products.drop_duplicates("image_sha256").reset_index(drop=True)
    products["product_text"] = products.apply(build_product_text, axis=1)

    # Union groups so multi-image products and repeated text never cross splits.
    parents = list(range(len(products)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for column in ("product_id", "product_text"):
        seen = {}
        for index, value in enumerate(products[column]):
            if value in seen:
                parents[root(index)] = root(seen[value])
            else:
                seen[value] = index
    products["split_group"] = [root(i) for i in range(len(products))]
    groups = products.groupby("split_group", sort=True).category.first()
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, category_groups in groups.groupby(groups, sort=True):
        ids = rng.permutation(category_groups.index.to_numpy())
        n = len(ids)
        n_val = max(1, int(n * val_fraction)) if n >= 3 else 0
        n_test = max(1, int(n * test_fraction)) if n >= 3 else 0
        if n_val + n_test >= n:
            n_test = n - n_val - 1
        for position, group_id in enumerate(ids):
            assignment[group_id] = (
                "validation" if position < n_val else
                "test" if position < n_val + n_test else "train"
            )
    products["split"] = products.split_group.map(assignment)
    counts = products.split.value_counts().to_dict()
    if counts.get("train", 0) < 2 or any(
        counts.get(split, 0) < 2 for split in ("validation", "test")
    ):
        raise ValueError("Need at least two rows in each split; dataset is too small.")
    return products, {"input_rows": before, "duplicate_images_removed": before - len(products),
                      "split_counts": counts, "seed": seed,
                      "val_fraction": val_fraction, "test_fraction": test_fraction}


def load_experiment_data(run_dir, verify_images=True):
    import json

    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    path = run_dir / "dataset.csv"
    if file_sha256(path) != metadata["dataset_sha256"]:
        raise ValueError("Experiment dataset changed; create a new experiment directory.")
    products = pd.read_csv(path, dtype={"product_id": str})
    if verify_images:
        for row in products.itertuples():
            if file_sha256(row.image_path) != row.image_sha256:
                raise ValueError(f"Image changed since split preparation: {row.image_path}")
    return products, metadata
