"""Late-fusion methods for independently trained retrieval models."""

from __future__ import annotations

import numpy as np


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

