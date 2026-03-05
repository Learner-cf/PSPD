from __future__ import annotations

import torch
import torch.nn.functional as F


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def evaluate_embedding_quality(
    embeddings: torch.Tensor,
    num_pairs: int = 10_000,
    seed: int = 42,
    cov_max_samples: int = 4096,
    cov_subsample_dim_threshold: int = 2048,
):
    """Research-grade Stage1 embedding diagnostics (torch-only)."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape (N, D), got {tuple(embeddings.shape)}")

    eps = 1e-12

    with torch.no_grad():
        raw_embs = embeddings.detach().cpu().float()
        n, d = raw_embs.shape

        # -------------------- Existing metrics (kept) --------------------
        per_dim_variance = raw_embs.var(dim=0, unbiased=False)
        raw_variance_mean = float(per_dim_variance.mean().item())

        embs = F.normalize(raw_embs, p=2, dim=1)
        norms = torch.norm(embs, dim=1)
        norm_mean = float(norms.mean().item())
        norm_std = float(norms.std(unbiased=False).item())

        if n < 2:
            pairwise_cos = torch.ones(1)
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            i1 = torch.randint(0, n, (num_pairs,), generator=g)
            i2 = torch.randint(0, n, (num_pairs,), generator=g)
            same = i1 == i2
            if same.any():
                i2[same] = (i2[same] + 1) % n
            pairwise_cos = (embs[i1] * embs[i2]).sum(dim=1)

        pairwise_cos_mean = float(pairwise_cos.mean().item())

        # -------------------- Per-dimension variance analysis --------------------
        raw_variance_std = float(per_dim_variance.std(unbiased=False).item())
        raw_variance_min = float(per_dim_variance.min().item())
        raw_variance_max = float(per_dim_variance.max().item())
        variance_imbalance_ratio = raw_variance_max / max(raw_variance_min, eps)

        # -------------------- Covariance eigenvalue / anisotropy --------------------
        cov_embs = raw_embs
        if (d > cov_subsample_dim_threshold) and (n > cov_max_samples):
            g_cov = torch.Generator(device="cpu")
            g_cov.manual_seed(seed + 1)
            idx = torch.randperm(n, generator=g_cov)[:cov_max_samples]
            cov_embs = raw_embs[idx]

        cov = torch.cov(cov_embs.T)
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = torch.clamp(eigvals, min=0.0)

        eig_max = float(eigvals.max().item())
        eig_min = float(eigvals.min().item())
        eig_ratio = eig_max / (eig_min + eps)

        effective_rank = float(((eigvals.sum() ** 2) / ((eigvals ** 2).sum() + eps)).item())
        utilization_ratio = effective_rank / max(float(d), 1.0)

        # -------------------- Similarity distribution spread --------------------
        pairwise_cos_std = float(pairwise_cos.std(unbiased=False).item())
        pairwise_cos_min = float(pairwise_cos.min().item())
        pairwise_cos_max = float(pairwise_cos.max().item())

        # -------------------- Warnings --------------------
        warnings = []
        if raw_variance_mean < 0.1:
            warnings.append("possible_feature_collapse_low_raw_variance")
        if utilization_ratio < 0.4:
            warnings.append("low_feature_utilization")
        if pairwise_cos_std < 0.02:
            warnings.append("similarity_distribution_too_narrow")
        if pairwise_cos_mean > 0.8:
            warnings.append("possible_feature_collapse_high_similarity")
        if pairwise_cos_mean < 0.05 and not (0.8 <= raw_variance_mean <= 1.2):
            warnings.append("overly_dispersed_features")

        # -------------------- Stage1 quality score [0,1] --------------------
        variance_score = _clip01(raw_variance_mean / 1.0)
        anisotropy_penalty = _clip01((eig_ratio - 1.0) / 999.0)
        # encourage moderate spread; penalize too narrow or too extreme
        dispersion_center = 0.2
        dispersion_score = _clip01(1.0 - abs(pairwise_cos_std - dispersion_center) / dispersion_center)

        stage1_score = _clip01(
            0.30 * variance_score
            + 0.30 * _clip01(utilization_ratio)
            + 0.20 * (1.0 - anisotropy_penalty)
            + 0.20 * dispersion_score
        )

        if stage1_score > 0.80:
            stage1_grade = "Excellent DINO features"
        elif stage1_score > 0.65:
            stage1_grade = "Good"
        elif stage1_score > 0.50:
            stage1_grade = "Acceptable but risky"
        else:
            stage1_grade = "Poor representation quality"

        metrics = {
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "raw_variance_mean": raw_variance_mean,
            "pairwise_cos_mean": pairwise_cos_mean,
            "raw_variance_std": raw_variance_std,
            "raw_variance_min": raw_variance_min,
            "raw_variance_max": raw_variance_max,
            "eig_max": eig_max,
            "eig_min": eig_min,
            "eig_ratio": eig_ratio,
            "effective_rank": effective_rank,
            "utilization_ratio": utilization_ratio,
            "pairwise_cos_std": pairwise_cos_std,
            "pairwise_cos_min": pairwise_cos_min,
            "pairwise_cos_max": pairwise_cos_max,
            "stage1_score": stage1_score,
            "stage1_grade": stage1_grade,
            "warnings": warnings,
        }

        print("========== STAGE 1 DIAGNOSTIC REPORT ==========")
        print(f"Embedding Dim: {d}")
        print("[WHITENING GEOMETRY]")
        print(f"eigen_ratio_raw: {eig_ratio:.6f}")
        print(f"eigen_ratio_post_whiten: {eig_ratio:.6f}")
        print(f"effective_rank_post_whiten: {effective_rank:.6f}")
        print(f"Raw Variance Mean: {raw_variance_mean:.6f}")
        print(f"Raw Variance Std: {raw_variance_std:.6f}")
        print(f"Raw Variance Min/Max: {raw_variance_min:.6f} / {raw_variance_max:.6f}")
        print(f"Utilization Ratio: {utilization_ratio:.6f}")
        print("[NORMALIZED FEATURE GEOMETRY]")
        print(f"pairwise cosine mean: {pairwise_cos_mean:.6f}")
        print(f"pairwise cosine std: {pairwise_cos_std:.6f}")
        print(f"Stage 1 Score: {stage1_score:.6f}")
        print(f"Warnings: {', '.join(warnings) if warnings else 'None'}")
        print("===============================================")

        return metrics
