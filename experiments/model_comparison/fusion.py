"""Late-fusion methods for independently trained retrieval models."""

from __future__ import annotations

import numpy as np


def _normalize_rows(matrix):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite two-dimensional matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embeddings must be nonzero.")
    return values / norms


def clip_score_fusion(
    image_embeddings,
    text_embeddings,
    product_ids,
    image_weight=0.5,
    text_image_weight=0.5,
):
    """Fuse raw CLIP cosine scores and return aligned rankings."""
    return clip_query_gallery_score_fusion(
        image_embeddings,
        text_embeddings,
        image_embeddings,
        text_embeddings,
        product_ids,
        product_ids,
        image_weight=image_weight,
        text_image_weight=text_image_weight,
    )


def clip_query_gallery_score_fusion(
    query_image_embeddings,
    query_text_embeddings,
    gallery_image_embeddings,
    gallery_text_embeddings,
    query_product_ids,
    gallery_product_ids,
    image_weight=0.5,
    text_image_weight=0.5,
):
    """Fuse raw CLIP cosine scores for queries against a larger gallery."""
    if not 0.0 <= image_weight <= 1.0:
        raise ValueError("image_weight must be between 0 and 1.")
    if not 0.0 <= text_image_weight <= 1.0:
        raise ValueError("text_image_weight must be between 0 and 1.")

    query_images = _normalize_rows(query_image_embeddings)
    query_texts = _normalize_rows(query_text_embeddings)
    gallery_images = _normalize_rows(gallery_image_embeddings)
    gallery_texts = _normalize_rows(gallery_text_embeddings)
    query_ids = np.asarray(query_product_ids, dtype=str)
    gallery_ids = np.asarray(gallery_product_ids, dtype=str)
    if query_images.shape != query_texts.shape or len(query_ids) != len(query_images):
        raise ValueError("Query image/text embeddings and IDs must align.")
    if gallery_images.shape != gallery_texts.shape or len(gallery_ids) != len(gallery_images):
        raise ValueError("Gallery image/text embeddings and IDs must align.")
    if query_images.shape[1] != gallery_images.shape[1]:
        raise ValueError("Query and gallery CLIP dimensions must match.")

    image_scores = query_images @ gallery_images.T
    text_to_image_scores = query_texts @ gallery_images.T
    text_to_text_scores = query_texts @ gallery_texts.T

    text_scores = (
        text_image_weight * text_to_image_scores
        + (1.0 - text_image_weight) * text_to_text_scores
    )
    fused_scores = image_weight * image_scores + (1.0 - image_weight) * text_scores

    rankings = []
    for query_index, row_scores in enumerate(fused_scores):
        candidates = np.flatnonzero(gallery_ids != query_ids[query_index])
        rankings.append(candidates[np.argsort(-row_scores[candidates], kind="stable")])
    return rankings


def clip_text_query_gallery_score_fusion(
    query_text_embeddings,
    gallery_image_embeddings,
    gallery_text_embeddings,
    query_product_ids,
    gallery_product_ids,
    text_image_weight=0.5,
):
    """Mirror the website's text-input retrieval over a multimodal catalog.

    The query is text only, while catalog candidates contribute both an image
    representation (text-to-image) and a text representation (text-to-text).
    The configured weight is applied directly to the raw cosine scores.
    """
    if not 0.0 <= text_image_weight <= 1.0:
        raise ValueError("text_image_weight must be between 0 and 1.")

    query_texts = _normalize_rows(query_text_embeddings)
    gallery_images = _normalize_rows(gallery_image_embeddings)
    gallery_texts = _normalize_rows(gallery_text_embeddings)
    query_ids = np.asarray(query_product_ids, dtype=str)
    gallery_ids = np.asarray(gallery_product_ids, dtype=str)
    if len(query_ids) != len(query_texts):
        raise ValueError("Query text embeddings and IDs must align.")
    if len(gallery_ids) != len(gallery_images) or len(gallery_ids) != len(gallery_texts):
        raise ValueError("Gallery image/text embeddings and IDs must align.")
    if (
        query_texts.shape[1] != gallery_images.shape[1]
        or query_texts.shape[1] != gallery_texts.shape[1]
    ):
        raise ValueError("Query and gallery CLIP dimensions must match.")

    text_to_image_scores = query_texts @ gallery_images.T
    text_to_text_scores = query_texts @ gallery_texts.T
    fused_scores = (
        text_image_weight * text_to_image_scores
        + (1.0 - text_image_weight) * text_to_text_scores
    )

    rankings = []
    for query_index, row_scores in enumerate(fused_scores):
        candidates = np.flatnonzero(gallery_ids != query_ids[query_index])
        rankings.append(candidates[np.argsort(-row_scores[candidates], kind="stable")])
    return rankings


def reciprocal_rank_fusion(ranking_groups, catalog_size, rrf_k=60):
    """Fuse aligned ranking lists using Reciprocal Rank Fusion (RRF).

    RRF combines ranks rather than embedding values, so it is suitable for
    SBERT and ResNet18 even though their vectors have different dimensions and
    were trained in unrelated representation spaces.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")
    groups = list(ranking_groups)
    if len(groups) < 2:
        raise ValueError("RRF needs at least two ranking groups.")
    query_count = len(groups[0])
    if any(len(group) != query_count for group in groups):
        raise ValueError("All ranking groups must contain the same queries.")

    fused = []
    for query_index in range(query_count):
        scores = np.zeros(catalog_size, dtype=np.float64)
        present = np.zeros(catalog_size, dtype=bool)
        for group in groups:
            ranking = np.asarray(group[query_index], dtype=int)
            if ranking.ndim != 1 or len(ranking) != len(np.unique(ranking)):
                raise ValueError("Each component ranking must be one-dimensional and unique.")
            if len(ranking) and (ranking.min() < 0 or ranking.max() >= catalog_size):
                raise ValueError("A component ranking contains an invalid catalog index.")
            positions = np.arange(1, len(ranking) + 1, dtype=np.float64)
            scores[ranking] += 1.0 / (rrf_k + positions)
            present[ranking] = True
        candidates = np.flatnonzero(present)
        fused.append(candidates[np.argsort(-scores[candidates], kind="stable")])
    return fused

