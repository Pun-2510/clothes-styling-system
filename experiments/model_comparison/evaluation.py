"""Ranking and category-retrieval metrics used by the modality benchmark."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def rank_by_cosine(embeddings, product_ids):
    """Rank the catalog for every query and exclude the query product.

    Embeddings must already describe the same modality on both query and gallery
    sides. All rows sharing the query product ID are excluded to prevent a
    trivial self-match.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(product_ids, dtype=str)
    if matrix.ndim != 2 or len(matrix) != len(ids) or len(matrix) < 2:
        raise ValueError("Expected a two-dimensional embedding matrix with at least two rows.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain a non-finite value.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embeddings must be nonzero.")
    matrix = matrix / norms
    scores = matrix @ matrix.T
    rankings = []
    for index, row_scores in enumerate(scores):
        candidates = np.flatnonzero(ids != ids[index])
        rankings.append(candidates[np.argsort(-row_scores[candidates], kind="stable")])
    return rankings


def _f1(precision, recall):
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def evaluate_category_rankings(rankings, product_ids, categories, ks=(1, 5, 10)):
    """Calculate query-macro and category-macro Precision, Recall and F1.

    A relevant result has the same ground-truth category as the query. The query
    product itself is excluded. Categories with no other positive item are
    counted as skipped instead of receiving an arbitrary zero score.
    """
    ids = np.asarray(product_ids, dtype=str)
    labels = np.asarray(categories, dtype=str)
    ks = sorted(set(int(k) for k in ks))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("All K values must be positive integers.")
    if len(rankings) != len(ids) or len(labels) != len(ids):
        raise ValueError("Rankings, product IDs and categories must have equal lengths.")

    metric_names = ("precision", "recall", "f1")
    totals = {metric: {k: 0.0 for k in ks} for metric in metric_names}
    category_totals = defaultdict(
        lambda: {metric: {k: 0.0 for k in ks} for metric in metric_names}
    )
    category_queries = defaultdict(int)
    category_evaluated = defaultdict(int)
    evaluated = 0

    for query_index, ranking in enumerate(rankings):
        category = labels[query_index].strip() or "Unknown"
        category_queries[category] += 1
        eligible = ids != ids[query_index]
        relevant = np.flatnonzero(eligible & (labels == labels[query_index]))
        if not labels[query_index].strip() or len(relevant) == 0:
            continue

        ranking = np.asarray(ranking, dtype=int)
        if ranking.ndim != 1:
            raise ValueError("Every ranking must be one-dimensional.")
        if len(ranking) and (ranking.min() < 0 or ranking.max() >= len(ids)):
            raise ValueError("A ranking contains an invalid catalog index.")
        ranking = ranking[eligible[ranking]]
        if len(ranking) != len(np.unique(ranking)):
            raise ValueError("A ranking contains duplicate catalog indices.")

        relevant_set = set(int(value) for value in relevant)
        for k in ks:
            top = ranking[: min(k, len(ranking))]
            hits = sum(int(candidate) in relevant_set for candidate in top)
            precision = hits / len(top) if len(top) else 0.0
            recall = hits / len(relevant)
            f1 = _f1(precision, recall)
            for accumulator in (totals, category_totals[category]):
                accumulator["precision"][k] += precision
                accumulator["recall"][k] += recall
                accumulator["f1"][k] += f1
        evaluated += 1
        category_evaluated[category] += 1

    def averages(accumulator, denominator):
        return {
            f"{metric}@{k}": accumulator[metric][k] / denominator if denominator else None
            for metric in metric_names
            for k in ks
        }

    per_category = {}
    for category in sorted(category_queries):
        count = category_evaluated[category]
        per_category[category] = {
            **averages(category_totals[category], count),
            "query_count": category_queries[category],
            "evaluated_queries": count,
            "skipped_queries": category_queries[category] - count,
        }

    category_macro = {}
    for metric in metric_names:
        for k in ks:
            key = f"{metric}@{k}"
            values = [row[key] for row in per_category.values() if row[key] is not None]
            category_macro[key] = float(np.mean(values)) if values else None
    category_macro.update(
        evaluated_categories=sum(row["evaluated_queries"] > 0 for row in per_category.values()),
        total_categories=len(per_category),
    )
    return {
        **averages(totals, evaluated),
        "evaluated_queries": evaluated,
        "skipped_queries": len(ids) - evaluated,
        "gallery_size": len(ids),
        "category_macro": category_macro,
        "per_category": per_category,
    }

