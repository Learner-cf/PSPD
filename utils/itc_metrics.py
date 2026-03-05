from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def evaluate_itc_quality(image_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, float]:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("image_features and text_features must be 2D")
    if image_features.size(0) != text_features.size(0):
        raise ValueError("batch size mismatch between image and text features")

    img = F.normalize(image_features, p=2, dim=-1)
    txt = F.normalize(text_features, p=2, dim=-1)
    sim = img @ txt.t()

    bsz = sim.size(0)
    targets = torch.arange(bsz, device=sim.device)
    pred = sim.argmax(dim=1)
    alignment_accuracy = float((pred == targets).float().mean().item())

    pos_sim = sim.diag()
    mask = ~torch.eye(bsz, dtype=torch.bool, device=sim.device)
    neg_sim = sim[mask]

    pos_sim_mean = float(pos_sim.mean().item())
    neg_sim_mean = float(neg_sim.mean().item()) if neg_sim.numel() > 0 else 0.0
    sim_gap = pos_sim_mean - neg_sim_mean

    print(
        "[Stage3][ITCMetrics] "
        f"alignment_acc={alignment_accuracy:.4f} "
        f"pos_sim_mean={pos_sim_mean:.6f} "
        f"neg_sim_mean={neg_sim_mean:.6f} "
        f"sim_gap={sim_gap:.6f}"
    )

    return {
        "alignment_accuracy": alignment_accuracy,
        "pos_sim_mean": pos_sim_mean,
        "neg_sim_mean": neg_sim_mean,
        "sim_gap": sim_gap,
    }
