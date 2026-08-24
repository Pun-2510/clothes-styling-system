import numpy as np


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