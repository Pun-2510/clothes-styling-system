"""Ranking and category-retrieval metrics used by the modality benchmark."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _normalized(matrix, label):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite two-dimensional matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError(f"{label} must not contain zero embeddings.")
    return values / norms


def rank_queries_by_cosine(
    query_embeddings,
    gallery_embeddings,
    query_product_ids,
    gallery_product_ids,
    exclude_same_product=True,
):
    """Rank one gallery for each query using cosine similarity.

    Category-retrieval experiments exclude the query product to prevent a
    trivial self-match. Paired cross-modal retrieval keeps it because that
    exact image/text pair is the relevant target.
    """
    queries = _normalized(query_embeddings, "Query embeddings")
    gallery = _normalized(gallery_embeddings, "Gallery embeddings")
    query_ids = np.asarray(query_product_ids, dtype=str)
    gallery_ids = np.asarray(gallery_product_ids, dtype=str)
    if len(queries) != len(query_ids) or len(gallery) != len(gallery_ids):
        raise ValueError("Embedding rows and product IDs must align.")
    if queries.shape[1] != gallery.shape[1] or not len(queries) or len(gallery) < 2:
        raise ValueError("Query and gallery embeddings must share a valid dimension.")

    scores = queries @ gallery.T
    rankings = []
    for query_index, row_scores in enumerate(scores):
        if exclude_same_product:
            candidates = np.flatnonzero(gallery_ids != query_ids[query_index])
        else:
            candidates = np.arange(len(gallery_ids), dtype=np.int64)
        rankings.append(candidates[np.argsort(-row_scores[candidates], kind="stable")])
    return rankings


def rank_by_cosine(embeddings, product_ids):
    """Rank the catalog for every query and exclude the query product.

    Embeddings must already describe the same modality on both query and gallery
    sides. All rows sharing the query product ID are excluded to prevent a
    trivial self-match.
    """
    return rank_queries_by_cosine(embeddings, embeddings, product_ids, product_ids)


def _f1(precision, recall):
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def evaluate_category_rankings(rankings, product_ids, categories, ks=(1, 5, 10)):
    """Calculate query-macro and category-macro Precision, Recall and F1.

    A relevant result has the same ground-truth category as the query. The query
    product itself is excluded. Categories with no other positive item are
    counted as skipped instead of receiving an arbitrary zero score.
    """
    return evaluate_query_gallery_category_rankings(
        rankings,
        product_ids,
        categories,
        product_ids,
        categories,
        ks,
    )


def evaluate_query_gallery_category_rankings(
    rankings,
    query_product_ids,
    query_categories,
    gallery_product_ids,
    gallery_categories,
    ks=(1, 5, 10),
):
    """Evaluate held-out queries against a possibly larger catalog gallery."""
    query_ids = np.asarray(query_product_ids, dtype=str)
    query_labels = np.asarray(query_categories, dtype=str)
    gallery_ids = np.asarray(gallery_product_ids, dtype=str)
    gallery_labels = np.asarray(gallery_categories, dtype=str)
    ks = sorted(set(int(k) for k in ks))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("All K values must be positive integers.")
    if len(rankings) != len(query_ids) or len(query_labels) != len(query_ids):
        raise ValueError("Rankings, query IDs and query categories must have equal lengths.")
    if len(gallery_ids) != len(gallery_labels) or len(gallery_ids) < 2:
        raise ValueError("Gallery IDs and categories must align and contain at least two rows.")

    metric_names = ("precision", "recall", "f1")
    totals = {metric: {k: 0.0 for k in ks} for metric in metric_names}
    category_totals = defaultdict(
        lambda: {metric: {k: 0.0 for k in ks} for metric in metric_names}
    )
    category_queries = defaultdict(int)
    category_evaluated = defaultdict(int)
    evaluated = 0

    for query_index, ranking in enumerate(rankings):
        category = query_labels[query_index].strip() or "Unknown"
        category_queries[category] += 1
        eligible = gallery_ids != query_ids[query_index]
        relevant = np.flatnonzero(
            eligible & (gallery_labels == query_labels[query_index])
        )
        if not query_labels[query_index].strip() or len(relevant) == 0:
            continue

        ranking = np.asarray(ranking, dtype=int)
        if ranking.ndim != 1:
            raise ValueError("Every ranking must be one-dimensional.")
        if len(ranking) and (ranking.min() < 0 or ranking.max() >= len(gallery_ids)):
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
        "skipped_queries": len(query_ids) - evaluated,
        "gallery_size": len(gallery_ids),
        "category_macro": category_macro,
        "per_category": per_category,
    }


def evaluate_query_gallery_identity_rankings(
    rankings,
    query_product_ids,
    query_categories,
    gallery_product_ids,
    ks=(1, 5, 10),
):
    """Evaluate paired cross-modal retrieval using exact product identity.

    Unlike category retrieval, the query product remains eligible because its
    paired representation in the other modality is the target. The shared
    full catalog may contain one or more rows for an identity, although the
    standard 5,000-product protocol requires unique gallery IDs.
    """
    query_ids = np.asarray(query_product_ids, dtype=str)
    query_labels = np.asarray(query_categories, dtype=str)
    gallery_ids = np.asarray(gallery_product_ids, dtype=str)
    ks = sorted(set(int(k) for k in ks))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("All K values must be positive integers.")
    if len(rankings) != len(query_ids) or len(query_labels) != len(query_ids):
        raise ValueError("Rankings, query IDs and query categories must have equal lengths.")
    if not len(gallery_ids):
        raise ValueError("Gallery IDs must contain at least one row.")

    metric_names = ("precision", "recall", "f1")
    totals = {metric: {k: 0.0 for k in ks} for metric in metric_names}
    category_totals = defaultdict(
        lambda: {metric: {k: 0.0 for k in ks} for metric in metric_names}
    )
    category_queries = defaultdict(int)
    category_evaluated = defaultdict(int)
    evaluated = 0

    for query_index, ranking in enumerate(rankings):
        category = query_labels[query_index].strip() or "Unknown"
        category_queries[category] += 1
        relevant = np.flatnonzero(gallery_ids == query_ids[query_index])
        if len(relevant) == 0:
            continue

        ranking = np.asarray(ranking, dtype=int)
        if ranking.ndim != 1:
            raise ValueError("Every ranking must be one-dimensional.")
        if len(ranking) and (ranking.min() < 0 or ranking.max() >= len(gallery_ids)):
            raise ValueError("A ranking contains an invalid catalog index.")
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
        "skipped_queries": len(query_ids) - evaluated,
        "gallery_size": len(gallery_ids),
        "category_macro": category_macro,
        "per_category": per_category,
    }

