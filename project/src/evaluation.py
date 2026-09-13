import numpy as np


def evaluate_retrieval(image_embeddings, text_embeddings, products, ks=(1, 5, 10), batch_size=128):
    """Macro Recall@K on a held-out gallery shared by every query.

    Cross-modal positives are rows with the same product ID. Image-to-image
    positives share the ground-truth category, excluding the query product.
    Queries without an eligible category positive are excluded and counted.
    Category recall is a coarse proxy, not human relevance ground truth.
    """
    ks = sorted(set(ks))
    if not ks or any(not isinstance(k, (int, np.integer)) or k <= 0 for k in ks):
        raise ValueError("Recall cutoffs must be positive integers.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    images = np.asarray(image_embeddings, dtype=np.float32)
    texts = np.asarray(text_embeddings, dtype=np.float32)
    if (images.ndim != 2 or images.shape != texts.shape or len(images) != len(products)
            or len(images) < 2 or images.shape[1] == 0):
        raise ValueError("Expected aligned image/text matrices and at least two products.")
    for matrix in (images, texts):
        if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
            raise ValueError("Embeddings must be finite and nonzero.")
    images = images / np.linalg.norm(images, axis=1, keepdims=True)
    texts = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    ids = products.product_id.astype(str).to_numpy()
    categories = products.category.fillna("").astype(str).to_numpy()
    metrics = {}
    for name, queries, gallery in (
        ("text_to_image", texts, images),
        ("image_to_text", images, texts),
        ("image_to_image_category", images, images),
    ):
        totals = {k: 0.0 for k in ks}
        evaluated = 0
        for start in range(0, len(products), batch_size):
            scores = queries[start:start + batch_size] @ gallery.T
            for local, row_scores in enumerate(scores):
                index = start + local
                if name == "image_to_image_category":
                    eligible = ids != ids[index]
                    relevant = np.flatnonzero(eligible & (categories == categories[index]))
                    if not categories[index].strip():
                        relevant = np.array([], dtype=int)
                else:
                    eligible = np.ones(len(products), dtype=bool)
                    relevant = np.flatnonzero(ids == ids[index])
                if len(relevant) == 0:
                    continue
                candidates = np.flatnonzero(eligible)
                ranking = candidates[np.argsort(-row_scores[candidates], kind="stable")[:max(ks)]]
                for k in ks:
                    totals[k] += recall_at_k(ranking, relevant, k)
                evaluated += 1
        metrics[name] = {
            **{f"recall@{k}": totals[k] / evaluated if evaluated else None for k in ks},
            "evaluated_queries": evaluated,
            "skipped_queries": len(products) - evaluated,
            "gallery_size": len(products),
        }
    return metrics


# =========================================================
# SYSTEM METRICS
# =========================================================

def calculate_system_metrics(
    results,
    predicted_category
):
    """
    Tính các metric có thể sử dụng
    ngay cả khi chưa có ground truth.

    Metrics:

    1. Average Visual Similarity
    2. Category Match Rate
    """

    if results is None or results.empty:

        return {
            "avg_similarity": 0.0,
            "category_match": 0.0,
        }


    # =====================================================
    # Average Visual Similarity
    # =====================================================

    if "visual_similarity" in results.columns:

        avg_similarity = (
            results[
                "visual_similarity"
            ]
            .astype(float)
            .mean()
        )

    else:

        avg_similarity = 0.0


    # =====================================================
    # Category Match
    # =====================================================

    if "category" in results.columns:

        category_match = (
            results[
                "category"
            ]
            .astype(str)
            .eq(
                str(predicted_category)
            )
            .mean()
        )

    else:

        category_match = 0.0


    return {
        "avg_similarity":
            float(avg_similarity),

        "category_match":
            float(category_match),
    }


# =========================================================
# PRECISION@K
# =========================================================

def precision_at_k(
    recommended_indices,
    relevant_indices,
    k
):
    """
    Precision@K.

    Chỉ sử dụng khi đã có ground truth.
    """

    recommended = set(
        recommended_indices[:k]
    )

    relevant = set(
        relevant_indices
    )

    if not recommended:

        return 0.0

    return (
        len(
            recommended &
            relevant
        )
        /
        len(recommended)
    )


# =========================================================
# RECALL@K
# =========================================================

def recall_at_k(
    recommended_indices,
    relevant_indices,
    k
):
    """
    Recall@K.

    Chỉ sử dụng khi đã có ground truth.
    """

    if k <= 0:
        return 0.0

    recommended = set(
        recommended_indices[:k]
    )

    relevant = set(
        relevant_indices
    )

    if not relevant:

        return 0.0

    return (
        len(
            recommended &
            relevant
        )
        /
        len(relevant)
    )


# =========================================================
# NDCG@K
# =========================================================

def ndcg_at_k(
    recommended_indices,
    relevant_indices,
    k
):
    """
    NDCG@K.

    Chỉ sử dụng khi đã có ground truth.
    """

    relevant = set(
        relevant_indices
    )

    dcg = 0.0

    for rank, index in enumerate(
        recommended_indices[:k],
        start=1
    ):

        if index in relevant:

            dcg += (
                1.0
                /
                np.log2(
                    rank + 1
                )
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:

        return 0.0

    idcg = sum(
        1.0
        /
        np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return (
        dcg /
        idcg
    )
