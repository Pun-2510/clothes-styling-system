"""Create compact, query-scoped explanations for recommendation responses."""

from __future__ import annotations

from collections.abc import Iterable

from src.category_ranking import category_confidence_strength
from src.config import (
    CATEGORY_CONFIDENCE_FLOOR,
    CATEGORY_CONFIDENCE_THRESHOLD,
)


def _unique(values: Iterable[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        label = str(value).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        output.append(label)
    return output


def _candidate_list(
    categories: Iterable[str | None],
    first_score: float | None = None,
) -> list[dict]:
    return [
        {
            "category": category,
            "score": first_score if index == 0 else None,
        }
        for index, category in enumerate(_unique(categories))
    ]


def build_local_category_graph(
    recommender,
    *,
    candidates: list[dict] | None,
    result_categories: Iterable[str | None],
    applied_category: str | None,
    query_mode: str,
) -> dict:
    """Build a small graph for this query, never the complete catalog graph."""
    candidates = candidates or []
    candidate_labels = _unique(
        candidate.get("category") for candidate in candidates[:2]
    )
    result_labels = _unique(result_categories)[:2]
    focus_labels = _unique([applied_category, *candidate_labels])
    relations = getattr(recommender, "category_relation_neighbors", {})

    related_labels: list[str] = []
    for category in focus_labels[:2]:
        related_labels.extend(list(relations.get(category, ()))[:2])

    category_labels = _unique(
        [*focus_labels, *related_labels, *result_labels]
    )[:6]
    candidate_scores = {
        str(candidate.get("category")): candidate.get("score")
        for candidate in candidates
        if candidate.get("category") is not None
    }

    nodes = [{
        "id": "query",
        "label": {
            "image": "Image query",
            "text": "Text query",
            "combined": "Image + text",
        }.get(query_mode, "Query"),
        "role": "query",
        "score": None,
    }]
    primary = applied_category or (
        candidate_labels[0] if candidate_labels else None
    )
    for category in category_labels:
        if category == primary:
            role = "predicted"
        elif category in candidate_labels:
            role = "candidate"
        elif category in related_labels:
            role = "related"
        else:
            role = "result"
        score = candidate_scores.get(category)
        nodes.append({
            "id": f"category:{category}",
            "label": category,
            "role": role,
            "score": None if score is None else float(score),
        })

    edges: list[dict] = []
    connected: set[str] = set()
    for rank, category in enumerate(candidate_labels, start=1):
        edges.append({
            "source": "query",
            "target": f"category:{category}",
            "relation": "evidence",
            "label": f"Top-{rank}",
        })
        connected.add(category)

    if applied_category and applied_category not in connected:
        edges.append({
            "source": "query",
            "target": f"category:{applied_category}",
            "relation": "explicit",
            "label": "Explicit",
        })
        connected.add(applied_category)

    for category in result_labels:
        if category in connected or category not in category_labels:
            continue
        edges.append({
            "source": "query",
            "target": f"category:{category}",
            "relation": "result",
            "label": "Top-K",
        })
        connected.add(category)

    category_set = set(category_labels)
    relation_pairs: set[tuple[str, str]] = set()
    for category in category_labels:
        for neighbor in relations.get(category, ()):
            neighbor = str(neighbor)
            if neighbor not in category_set:
                continue
            pair = tuple(sorted((category, neighbor)))
            if pair in relation_pairs:
                continue
            relation_pairs.add(pair)
            edges.append({
                "source": f"category:{category}",
                "target": f"category:{neighbor}",
                "relation": "mutual_neighbor",
                "label": "Related",
            })

    return {"nodes": nodes, "edges": edges}


def build_text_explanation(
    recommender,
    *,
    guidance: dict,
    image_weight: float,
    result_categories: Iterable[str | None],
) -> dict:
    applied_category = guidance.get("applied_category")
    suggestions = guidance.get("suggestions") or []
    candidates = [
        {
            "category": suggestion.get("category"),
            "score": suggestion.get("confidence"),
        }
        for suggestion in suggestions[:5]
        if suggestion.get("category")
    ]
    if applied_category:
        candidates.insert(0, {
            "category": applied_category,
            "score": None,
        })

    has_text_embeddings = bool(
        getattr(recommender, "text_embeddings", None) is not None
    )
    return {
        "query_mode": "text",
        "ranking_policy": (
            "explicit_category_filter" if applied_category else "unrestricted"
        ),
        "formula": (
            "S_text = β·S_TI + (1−β)·S_TT"
            if has_text_embeddings
            else "S_text = S_TI"
        ),
        "applied_category": applied_category,
        "category_source": "text" if applied_category else None,
        "category_route": guidance.get("status", "unrestricted"),
        "category_strength": 1.0 if applied_category else 0.0,
        "weights": {
            "text_to_image": float(image_weight),
            "text_to_text": (
                float(1.0 - image_weight) if has_text_embeddings else 0.0
            ),
        },
        "category_candidates": candidates,
        "graph": build_local_category_graph(
            recommender,
            candidates=candidates,
            result_categories=result_categories,
            applied_category=applied_category,
            query_mode="text",
        ),
    }


def build_image_explanation(
    recommender,
    *,
    recommendation: dict,
    result_categories: Iterable[str | None],
) -> dict:
    predicted = recommendation.get("predicted_category")
    confidence = recommendation.get("category_confidence")
    category_mode = recommendation.get("category_mode", "no_category")
    candidate_labels = recommendation.get("candidate_categories") or []
    candidates = _candidate_list(
        [*candidate_labels, predicted],
        first_score=confidence,
    )

    if category_mode == "hard_category":
        route = "hard_category"
    elif category_mode == "no_category":
        route = "low_confidence_fallback" if predicted else "no_category"
    elif len(candidate_labels) > 1:
        route = "ambiguous_related_top2"
    elif confidence is not None and confidence >= CATEGORY_CONFIDENCE_THRESHOLD:
        route = "confident_top1"
    else:
        route = "ambiguous_unrelated_top1"

    strength = category_confidence_strength(
        confidence,
        CATEGORY_CONFIDENCE_FLOOR,
        CATEGORY_CONFIDENCE_THRESHOLD,
    ) if confidence is not None else 0.0
    return {
        "query_mode": "image",
        "ranking_policy": route,
        "formula": "S_final = S_II + λ_I·g(h)·C(i)",
        "applied_category": (
            predicted if category_mode != "no_category" else None
        ),
        "category_source": "image" if predicted else None,
        "category_route": route,
        "category_strength": float(strength),
        "weights": {},
        "category_candidates": candidates,
        "graph": build_local_category_graph(
            recommender,
            candidates=candidates,
            result_categories=result_categories,
            applied_category=predicted,
            query_mode="image",
        ),
    }


def build_combined_explanation(
    recommender,
    *,
    recommendation: dict,
    result_categories: Iterable[str | None],
) -> dict:
    applied_category = recommendation.get("applied_category")
    category_source = recommendation.get("category_source")
    route = recommendation.get("category_route", "none")
    candidate_labels = recommendation.get("candidate_categories") or []
    candidates = _candidate_list([*candidate_labels, applied_category])
    image_weight = float(recommendation.get("image_weight", 0.0))
    text_weight = float(recommendation.get("text_weight", 0.0))
    return {
        "query_mode": "combined",
        "ranking_policy": route,
        "formula": (
            "S_base = ω_I·S_II + (1−ω_I)·S_T; "
            "S_final = S_base + attribute/category evidence"
        ),
        "applied_category": applied_category,
        "category_source": category_source,
        "category_route": route,
        "category_strength": float(
            recommendation.get("category_strength", 0.0)
        ),
        "weights": {
            "image": image_weight,
            "text": text_weight,
        },
        "category_candidates": candidates,
        "graph": build_local_category_graph(
            recommender,
            candidates=candidates,
            result_categories=result_categories,
            applied_category=applied_category,
            query_mode="combined",
        ),
    }
