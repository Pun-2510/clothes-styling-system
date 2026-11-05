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


def rank_category_score_mapping(category_scores):
    """Rank an already aggregated category-score mapping.

    Scores are clipped at zero for the confidence calculation because the
    existing confidence heuristic is defined for non-negative evidence.
    """
    ranked = sorted(
        (
            (str(category), float(score))
            for category, score in category_scores.items()
            if np.isfinite(float(score))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        raise RuntimeError("Cannot rank an empty category-score mapping.")

    top_score = max(float(ranked[0][1]), 0.0)
    second_score = (
        max(float(ranked[1][1]), 0.0)
        if len(ranked) > 1
        else 0.0
    )
    confidence = (
        top_score / (top_score + second_score)
        if top_score + second_score > 0.0
        else 0.0
    )
    return ranked, float(confidence)


def combine_category_evidence(
    visual_ranked_categories,
    prototype_scores,
    prototype_weight,
):
    """Combine catalog-neighbour and CLIP category-prototype evidence.

    ``visual_ranked_categories`` contains the balanced neighbour score for
    every category. ``prototype_scores`` maps the same category names to the
    cosine similarity between the query image and a short category prompt.
    The result remains a soft category signal; it does not remove candidates.
    """
    prototype_weight = float(prototype_weight)
    if prototype_weight < 0.0:
        raise ValueError("prototype_weight cannot be negative")

    prototype_scores = {
        str(category): float(score)
        for category, score in prototype_scores.items()
    }
    combined_scores = {
        str(category): (
            float(visual_score)
            + prototype_weight
            * max(prototype_scores.get(str(category), 0.0), 0.0)
        )
        for category, visual_score in visual_ranked_categories
    }
    return rank_category_score_mapping(combined_scores)


def build_mutual_category_neighbors(categories, embeddings, neighbor_count):
    """Build a symmetric relation graph from category prototype embeddings."""
    labels = [str(category) for category in categories]
    features = np.asarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError(
            "embeddings must be a two-dimensional array aligned with categories"
        )
    if int(neighbor_count) < 1:
        raise ValueError("neighbor_count must be at least 1")
    if not labels:
        return {}
    if len(labels) == 1:
        return {labels[0]: ()}

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)

    k = min(int(neighbor_count), len(labels) - 1)
    nearest_indices = np.argsort(-similarities, axis=1)[:, :k]
    nearest_sets = [set(row.tolist()) for row in nearest_indices]

    relations = {}
    for index, category in enumerate(labels):
        mutual_indices = [
            candidate
            for candidate in nearest_indices[index]
            if index in nearest_sets[int(candidate)]
        ]
        mutual_indices.sort(
            key=lambda candidate: float(similarities[index, int(candidate)]),
            reverse=True,
        )
        relations[category] = tuple(labels[int(value)] for value in mutual_indices)
    return relations


def categories_are_mutual_neighbors(first, second, relation_neighbors):
    """Check a relation graph produced by ``build_mutual_category_neighbors``."""
    left = str(first)
    right = str(second)
    return (
        right in relation_neighbors.get(left, ())
        and left in relation_neighbors.get(right, ())
    )


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
