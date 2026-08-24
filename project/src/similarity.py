import numpy as np


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    query_embedding,
    embeddings
):
    """
    Tính cosine similarity giữa:

        query_embedding
        và
        toàn bộ catalog embeddings
    """

    query = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    catalog = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # =====================================================
    # Check dimensions
    # =====================================================

    if query.ndim != 1:

        raise ValueError(
            "query_embedding phải là "
            "1-dimensional vector."
        )

    if catalog.ndim != 2:

        raise ValueError(
            "embeddings phải là "
            "2-dimensional matrix."
        )

    if (
        query.shape[0]
        !=
        catalog.shape[1]
    ):

        raise ValueError(
            "Dimension của query embedding "
            "không khớp catalog embeddings."
        )

    # =====================================================
    # Normalize query
    # =====================================================

    query_norm = np.linalg.norm(
        query
    )

    if query_norm == 0:

        raise ValueError(
            "Query embedding là zero vector."
        )

    query = (
        query
        /
        query_norm
    )

    # =====================================================
    # Normalize catalog
    # =====================================================

    catalog_norms = np.linalg.norm(
        catalog,
        axis=1,
        keepdims=True
    )

    safe_catalog = np.divide(
        catalog,
        catalog_norms,
        out=np.zeros_like(catalog),
        where=catalog_norms != 0
    )

    # =====================================================
    # Cosine similarity
    # =====================================================

    scores = (
        safe_catalog @ query
    )

    return scores


# =========================================================
# TOP-K
# =========================================================

def top_k_indices(
    scores,
    k=5
):
    """
    Lấy index của Top-K sản phẩm
    có similarity cao nhất.
    """

    scores = np.asarray(
        scores
    )

    if scores.ndim != 1:

        raise ValueError(
            "scores phải là 1-dimensional array."
        )

    if len(scores) == 0:

        return np.array(
            [],
            dtype=np.int64
        )

    k = max(
        1,
        min(
            int(k),
            len(scores)
        )
    )

    # -----------------------------------------------------
    # Candidate
    # -----------------------------------------------------

    candidate = np.argpartition(
        -scores,
        k - 1
    )[:k]

    # -----------------------------------------------------
    # Sort descending
    # -----------------------------------------------------

    candidate = candidate[
        np.argsort(
            scores[candidate]
        )[::-1]
    ]

    return candidate