"""Shared category-evidence and serving-policy helpers for experiments.

The functions in this module intentionally mirror ``project/src/recommend.py``.
Keeping the policy in one experiment helper prevents the category ablation,
validation tuning, and end-to-end pipeline benchmark from silently evaluating
different ranking rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .compare_modalities import _cache_signature, _cached_embeddings
from .encoders import encode_clip_text


@dataclass(frozen=True)
class CategoryPolicySettings:
    top_per_class: int
    confidence_floor: float
    confidence_threshold: float
    prototype_weight: float = 1.0
    relation_neighbors: int = 5
    related_top1_factor: float = 0.50
    related_top2_factor: float = 0.20
    unrelated_top1_factor: float = 0.10


def validate_policy_settings(settings: CategoryPolicySettings):
    if settings.top_per_class < 1 or settings.relation_neighbors < 1:
        raise ValueError("Category neighbour counts must be positive.")
    if not (
        0.0
        <= settings.confidence_floor
        < settings.confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "Category thresholds must satisfy 0 <= floor < threshold <= 1."
        )
    if settings.prototype_weight < 0.0:
        raise ValueError("Category prototype weight cannot be negative.")
    factors = (
        settings.related_top1_factor,
        settings.related_top2_factor,
        settings.unrelated_top1_factor,
    )
    if any(value < 0.0 or value > 1.0 for value in factors):
        raise ValueError("Category bonus factors must be between 0 and 1.")
    if settings.related_top2_factor > settings.related_top1_factor:
        raise ValueError("The related Top-2 factor cannot exceed the Top-1 factor.")


def load_category_prototypes(
    categories,
    checkpoint,
    checkpoint_identity,
    cache_dir,
    device,
    batch_size,
    use_cache=True,
):
    """Encode the same category prompts used by the recommendation backend."""
    labels = [str(value) for value in categories]
    prompts = [f"a catalog product photo of {category}" for category in labels]
    frame = pd.DataFrame(
        {
            "product_id": [f"category::{value}" for value in labels],
            "image_sha256": [""] * len(labels),
            "evaluation_text": prompts,
        }
    )
    signature = _cache_signature(
        frame,
        {
            "architecture": "CLIP category text prototype",
            "identity": checkpoint_identity,
            "prompt": "a catalog product photo of {category}",
        },
    )
    values, metadata = _cached_embeddings(
        cache_dir,
        "clip_finetuned_category_prototypes",
        signature,
        lambda: encode_clip_text(prompts, checkpoint, device, batch_size),
        use_cache,
    )
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if values.ndim != 2 or len(values) != len(labels) or np.any(norms <= 0.0):
        raise ValueError("Invalid category prototype embeddings.")
    return values / norms, metadata


def build_category_evidence(
    query_vectors,
    visual_scores,
    categories,
    category_indices,
    category_prototypes,
    settings: CategoryPolicySettings,
):
    """Predict categories and choose the deployed confidence route per query."""
    validate_policy_settings(settings)

    # Import from the project only after callers have placed PROJECT on sys.path.
    from src.category_ranking import (
        build_mutual_category_neighbors,
        categories_are_mutual_neighbors,
        category_confidence_strength,
        combine_category_evidence,
        rank_category_candidates,
        rank_category_score_mapping,
    )

    labels = [str(value) for value in categories]
    relations = build_mutual_category_neighbors(
        labels,
        category_prototypes,
        settings.relation_neighbors,
    )
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    scores_matrix = np.asarray(visual_scores, dtype=np.float32)
    if len(query_vectors) != len(scores_matrix):
        raise ValueError("Query embeddings and visual-score rows must align.")

    evidence = []
    for query_vector, visual_row in zip(query_vectors, scores_matrix):
        visual_ranked, visual_confidence = rank_category_candidates(
            visual_row,
            category_indices,
            settings.top_per_class,
        )
        prototype_values = np.asarray(
            category_prototypes @ query_vector,
            dtype=np.float32,
        )
        prototype_scores = dict(zip(labels, prototype_values))
        prototype_ranked, prototype_confidence = rank_category_score_mapping(
            prototype_scores
        )
        ranked, combined_confidence = combine_category_evidence(
            visual_ranked,
            prototype_scores,
            settings.prototype_weight,
        )

        predicted = ranked[0][0]
        if predicted == prototype_ranked[0][0]:
            confidence = prototype_confidence
            confidence_source = "prototype"
        elif predicted == visual_ranked[0][0]:
            confidence = visual_confidence
            confidence_source = "visual_neighbors"
        else:
            confidence = combined_confidence
            confidence_source = "combined"

        strength = category_confidence_strength(
            confidence,
            settings.confidence_floor,
            settings.confidence_threshold,
        )
        leading = [category for category, _ in ranked[:2]]
        related = (
            len(leading) == 2
            and categories_are_mutual_neighbors(
                leading[0], leading[1], relations
            )
        )
        if strength <= 0.0:
            route = "no_category"
            factors = {}
        elif confidence < settings.confidence_threshold and related:
            route = "ambiguous_related_top2"
            factors = {
                leading[0]: settings.related_top1_factor,
                leading[1]: settings.related_top2_factor,
            }
        elif confidence < settings.confidence_threshold:
            route = "ambiguous_unrelated_top1"
            factors = {leading[0]: settings.unrelated_top1_factor}
        else:
            route = "confident_top1"
            factors = {leading[0]: 1.0}

        evidence.append(
            {
                "predicted": predicted,
                "second_category": leading[1] if len(leading) > 1 else None,
                "confidence": float(confidence),
                "confidence_source": confidence_source,
                "strength": float(strength),
                "route": route,
                "related_top2": bool(related),
                "factors": factors,
                "visual_top": visual_ranked[0][0],
                "prototype_top": prototype_ranked[0][0],
            }
        )
    return evidence, relations


def apply_dynamic_category_bonus(
    scores,
    evidence,
    gallery_categories,
    bonus,
):
    """Apply dynamic full-catalog category bonuses without hard filtering."""
    if bonus < 0.0:
        raise ValueError("Category bonus cannot be negative.")
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    labels = np.asarray(gallery_categories, dtype=str)
    if len(adjusted) != len(evidence):
        raise ValueError("Score rows and category evidence must align.")
    for query_index, item in enumerate(evidence):
        for category, factor in item["factors"].items():
            adjusted[query_index] += (
                float(bonus)
                * float(item["strength"])
                * float(factor)
                * (labels == str(category))
            )
    return adjusted


def evidence_columns(evidence):
    """Convert evidence records into CSV-friendly diagnostic columns."""
    return {
        "predicted_category": [item["predicted"] for item in evidence],
        "second_category": [item["second_category"] or "" for item in evidence],
        "confidence": [item["confidence"] for item in evidence],
        "confidence_source": [item["confidence_source"] for item in evidence],
        "category_strength": [item["strength"] for item in evidence],
        "category_route": [item["route"] for item in evidence],
        "related_top2": [item["related_top2"] for item in evidence],
    }
