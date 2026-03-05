from __future__ import annotations

import numpy as np


def k_reciprocal_rerank(distance_matrix: np.ndarray, k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> np.ndarray:
    """Lightweight k-reciprocal inspired smoothing on pairwise distance matrix."""
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square (N, N)")

    n = distance_matrix.shape[0]
    original = distance_matrix.astype(np.float32)
    rank = np.argsort(original, axis=1)

    V = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        forward_neighbors = rank[i, : k1 + 1]
        reciprocal = []
        for candidate in forward_neighbors:
            backward_neighbors = rank[candidate, : k1 + 1]
            if i in backward_neighbors:
                reciprocal.append(candidate)
        if not reciprocal:
            reciprocal = [i]
        weights = np.exp(-original[i, reciprocal])
        weights = weights / (weights.sum() + 1e-12)
        V[i, reciprocal] = weights

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(n):
            V_qe[i] = V[rank[i, :k2]].mean(axis=0)
        V = V_qe

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(n)]
    jaccard = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        temp_min = np.zeros(n, dtype=np.float32)
        ind_non_zero = np.where(V[i] != 0)[0]
        for j in ind_non_zero:
            related = inv_index[j]
            temp_min[related] += np.minimum(V[i, j], V[related, j])
        jaccard[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    reranked = (1 - lambda_value) * jaccard + lambda_value * original
    reranked = np.maximum(reranked, 0.0)
    return reranked
