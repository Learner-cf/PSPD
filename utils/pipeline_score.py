from __future__ import annotations

from typing import Dict, Tuple


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_stage1_score(stage1_metrics: Dict[str, float]) -> float:
    var_mean = float(stage1_metrics.get("raw_variance_mean", 0.0))
    pair_cos = float(stage1_metrics.get("pairwise_cos_mean", 0.0))
    util = float(stage1_metrics.get("utilization_ratio", 0.0))
    eig_ratio = float(stage1_metrics.get("eig_ratio", 1.0))

    variance_score = _clip01(var_mean / 1.0)
    dispersion_score = _clip01(1.0 - abs(pair_cos - 0.4) / 0.4)
    eig_stability = _clip01(1.0 - (eig_ratio - 1.0) / 200.0)

    return _clip01(
        0.20 * variance_score
        + 0.40 * _clip01(util)
        + 0.30 * eig_stability
        + 0.10 * dispersion_score
    )


def compute_stage2_score(stage2_metrics: Dict[str, float]) -> float:
    ratio = float(stage2_metrics.get("inter_intra_ratio", 0.0))
    noise = float(stage2_metrics.get("noise_ratio", 1.0))
    change = float(stage2_metrics.get("label_change_ratio", 1.0))
    ratio_norm = _clip01(ratio / 3.0)
    return ratio_norm - noise - change


def compute_stage3_score(stage3_metrics: Dict[str, float]) -> float:
    acc = float(stage3_metrics.get("alignment_accuracy", 0.0))
    sim_gap = float(stage3_metrics.get("sim_gap", 0.0))
    return acc + sim_gap


def compute_pipeline_score(stage1_score: float, stage2_score: float, stage3_score: float) -> float:
    return 0.3 * stage1_score + 0.4 * stage2_score + 0.3 * stage3_score


def build_stage_report(
    stage1_score: float,
    stage2_score: float,
    stage3_score: float,
) -> Tuple[float, str]:
    final_score = compute_pipeline_score(stage1_score, stage2_score, stage3_score)
    msg = (
        "========== STAGE REPORT ==========\n"
        f"Stage 1 Score: {stage1_score:.6f}\n"
        f"Stage 2 Score: {stage2_score:.6f}\n"
        f"Stage 3 Score: {stage3_score:.6f}\n"
        f"Final Pipeline Score: {final_score:.6f}\n"
        "=================================="
    )
    return final_score, msg
