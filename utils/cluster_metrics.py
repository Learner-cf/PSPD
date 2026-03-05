from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def evaluate_cluster_quality(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    prev_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    eps = 1e-12
    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    labels = labels.astype(np.int64)

    total = max(1, labels.shape[0])
    noise_mask = labels == -1
    valid_labels = labels[~noise_mask]
    unique_clusters = sorted(set(valid_labels.tolist()))

    num_clusters = len(unique_clusters)
    noise_ratio = float(noise_mask.sum()) / total

    avg_cluster_size = 0.0
    median_cluster_size = 0.0
    intra_vals = []
    centroids = []

    if num_clusters > 0:
        _, counts = np.unique(valid_labels, return_counts=True)
        avg_cluster_size = float(np.mean(counts))
        median_cluster_size = float(np.median(counts))

        for cid in unique_clusters:
            idx = np.where(labels == cid)[0]
            cemb = emb_np[idx]
            centroids.append(cemb.mean(axis=0))
            if cemb.shape[0] > 1:
                sim = np.clip(cemb @ cemb.T, -1.0, 1.0)
                dist = 1.0 - sim
                tri = dist[np.triu_indices(dist.shape[0], k=1)]
                if tri.size > 0:
                    intra_vals.append(float(tri.mean()))

    intra_cluster_distance = float(np.mean(intra_vals)) if intra_vals else 0.0

    inter_cluster_distance = 0.0
    if len(centroids) > 1:
        c = np.stack(centroids, axis=0)
        c = c / (np.linalg.norm(c, axis=1, keepdims=True) + eps)
        sim = np.clip(c @ c.T, -1.0, 1.0)
        dist = 1.0 - sim
        tri = dist[np.triu_indices(dist.shape[0], k=1)]
        inter_cluster_distance = float(tri.mean()) if tri.size > 0 else 0.0

    inter_intra_ratio = inter_cluster_distance / max(intra_cluster_distance, eps)

    if prev_labels is None:
        label_change_ratio = 1.0
    else:
        prev = prev_labels.astype(np.int64)
        if prev.shape[0] != labels.shape[0]:
            raise ValueError("prev_labels and labels must have same length")
        label_change_ratio = float((prev != labels).sum()) / total

    metrics = {
        "num_clusters": float(num_clusters),
        "noise_ratio": noise_ratio,
        "avg_cluster_size": avg_cluster_size,
        "median_cluster_size": median_cluster_size,
        "label_change_ratio": label_change_ratio,
        "intra_cluster_distance": intra_cluster_distance,
        "inter_cluster_distance": inter_cluster_distance,
        "inter_intra_ratio": inter_intra_ratio,
    }

    print(
        "[Stage2][ClusterMetrics] "
        f"num_clusters={int(metrics['num_clusters'])} "
        f"noise_ratio={metrics['noise_ratio']:.4f} "
        f"avg_cluster_size={metrics['avg_cluster_size']:.2f} "
        f"median_cluster_size={metrics['median_cluster_size']:.2f} "
        f"label_change_ratio={metrics['label_change_ratio']:.4f} "
        f"intra_cluster_distance={metrics['intra_cluster_distance']:.6f} "
        f"inter_cluster_distance={metrics['inter_cluster_distance']:.6f} "
        f"inter_intra_ratio={metrics['inter_intra_ratio']:.6f}"
    )

    return metrics
