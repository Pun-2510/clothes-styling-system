"""Pure helpers shared by online and offline category-aware ranking."""

from __future__ import annotations

import numpy as np


def category_confidence_strength(confidence, floor, full_threshold):
    """Map heuristic confidence to a gradual category strength in ``[0, 1]``."""
    floor = float(floor)
    full_threshold = float(full_threshold)
    if not 0.0 <= floor < full_threshold <= 1.0:
        raise ValueError(
            "Category confidence thresholds must satisfy "
            "0 <= floor < full_threshold <= 1."
        )

    confidence = float(confidence)
    if not np.isfinite(confidence) or confidence <= floor:
        return 0.0
    if confidence >= full_threshold:
        return 1.0
    return (confidence - floor) / (full_threshold - floor)


def rank_category_candidates(scores, category_indices, top_per_class):
    """Rank categories from balanced groups of their strongest visual scores."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if int(top_per_class) < 1:
        raise ValueError("top_per_class must be at least 1")

    category_scores = {}
    for category, indices in category_indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            continue

        similarities = np.maximum(values[indices], 0.0) ** 4
        sample_size = min(int(top_per_class), len(similarities))
        strongest = np.partition(
            similarities,
            len(similarities) - sample_size,
        )[-sample_size:]
        category_scores[str(category)] = float(strongest.mean())

    ranked = sorted(
        category_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        raise RuntimeError("Cannot infer a category from an empty catalog.")

    top_score = float(ranked[0][1])
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    confidence = (
        top_score / (top_score + second_score)
        if top_score + second_score > 0.0
        else 0.0
    )
    return ranked, float(confidence)


def apply_confidence_weighted_category_bonus(
    base_scores,
    category_matches,
    confidence,
    bonus,
    floor,
    full_threshold,
):
    """Add a category bonus whose magnitude grows with prediction confidence."""
    scores = np.asarray(base_scores, dtype=np.float32)
    matches = np.asarray(category_matches, dtype=np.float32)
    if scores.shape != matches.shape:
        raise ValueError("base_scores and category_matches must have the same shape")

    strength = category_confidence_strength(
        confidence,
        floor,
        full_threshold,
    )
    effective_bonus = float(bonus) * strength
    return scores + effective_bonus * matches, strength, effective_bonus
