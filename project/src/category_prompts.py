"""Shared CLIP prompt ensemble for catalog category prototypes."""

from __future__ import annotations

import numpy as np


CATEGORY_PROMPT_TEMPLATES = (
    "a catalog product photo of {category}",
    "a product photo showing {category}",
    "an ecommerce product image in the category {category}",
    "a studio product image classified as {category}",
)


def category_prompt_ensemble(categories):
    """Return category-major prompt groups using the shared templates."""
    return [
        [template.format(category=str(category).strip()) for template in CATEGORY_PROMPT_TEMPLATES]
        for category in categories
    ]


def aggregate_category_prompt_embeddings(values, category_count):
    """Average normalized prompt vectors and renormalize each category vector."""
    embeddings = np.asarray(values, dtype=np.float32)
    prompt_count = len(CATEGORY_PROMPT_TEMPLATES)
    expected_rows = int(category_count) * prompt_count
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            "Category prompt embeddings must have shape "
            f"({expected_rows}, embedding_dimension)."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Category prompt embeddings must contain finite values.")

    row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(row_norms <= 0.0):
        raise ValueError("Category prompt embeddings contain a zero vector.")
    normalized_rows = embeddings / row_norms
    prototypes = normalized_rows.reshape(
        int(category_count), prompt_count, embeddings.shape[1]
    ).mean(axis=1)
    prototype_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if np.any(prototype_norms <= 0.0):
        raise ValueError("Averaged category prototypes contain a zero vector.")
    return (prototypes / prototype_norms).astype(np.float32)
