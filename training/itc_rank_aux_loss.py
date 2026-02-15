from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RankAuxConfig:
    lambda_rank: float = 0.2
    base_margin_12: float = 0.15
    base_margin_13: float = 0.20
    base_margin_23: float = 0.05
    conf_scale: float = 8.0
    conf_bias: float = 0.05
    eps: float = 1e-6


class ITCWithRankAuxLoss(nn.Module):
    """
    Total loss:
        L = L_itc + lambda_rank * L_rank

    Inputs (all tensors should be float and on same device):
      image_feat:        [B, D]
      caption_feat:      [B, D]
      logit_scale:       scalar (e.g. CLIP exp(logit_scale))

      sim_rk1:           [B, A]  similarity(image, attr_rk1_text)
      sim_rk2:           [B, A]  similarity(image, attr_rk2_text)
      sim_rk3:           [B, A]  similarity(image, attr_rk3_text)

      rerank_score_rk1:  [B, A]  score from clip_rerank json final_rank[0].score
      rerank_score_rk2:  [B, A]  score from clip_rerank json final_rank[1].score
      rerank_score_rk3:  [B, A]  score from clip_rerank json final_rank[2].score

      valid_mask:        [B, A]  1 for valid attr; 0 for unknown/missing
    """

    def __init__(self, cfg: RankAuxConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def itc_loss(self, image_feat: torch.Tensor, caption_feat: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        image_feat = self._l2_normalize(image_feat, self.cfg.eps)
        caption_feat = self._l2_normalize(caption_feat, self.cfg.eps)

        logits_i2t = logit_scale * image_feat @ caption_feat.t()
        logits_t2i = logits_i2t.t()

        target = torch.arange(image_feat.size(0), device=image_feat.device)
        loss_i = F.cross_entropy(logits_i2t, target)
        loss_t = F.cross_entropy(logits_t2i, target)
        return 0.5 * (loss_i + loss_t)

    def confidence_weight(
        self,
        rerank_score_rk1: torch.Tensor,
        rerank_score_rk2: torch.Tensor,
        rerank_score_rk3: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # gap-based confidence: large (rk1-rk2,rk1-rk3) => higher weight
        gap12 = rerank_score_rk1 - rerank_score_rk2
        gap13 = rerank_score_rk1 - rerank_score_rk3
        gap = 0.5 * (gap12 + gap13)

        conf = torch.sigmoid(self.cfg.conf_scale * (gap - self.cfg.conf_bias))
        conf = conf * valid_mask
        return conf

    def rank_loss(
        self,
        sim_rk1: torch.Tensor,
        sim_rk2: torch.Tensor,
        sim_rk3: torch.Tensor,
        rerank_score_rk1: torch.Tensor,
        rerank_score_rk2: torch.Tensor,
        rerank_score_rk3: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        conf = self.confidence_weight(rerank_score_rk1, rerank_score_rk2, rerank_score_rk3, valid_mask)

        # pairwise hinge ranking with confidence weighting
        # encourage: rk1 > rk2, rk1 > rk3, weakly rk2 > rk3
        l12 = F.relu(self.cfg.base_margin_12 - (sim_rk1 - sim_rk2))
        l13 = F.relu(self.cfg.base_margin_13 - (sim_rk1 - sim_rk3))
        l23 = F.relu(self.cfg.base_margin_23 - (sim_rk2 - sim_rk3))

        per_attr = l12 + l13 + 0.5 * l23
        weighted = per_attr * conf

        denom = conf.sum() + self.cfg.eps
        return weighted.sum() / denom

    def forward(
        self,
        image_feat: torch.Tensor,
        caption_feat: torch.Tensor,
        logit_scale: torch.Tensor,
        sim_rk1: torch.Tensor,
        sim_rk2: torch.Tensor,
        sim_rk3: torch.Tensor,
        rerank_score_rk1: torch.Tensor,
        rerank_score_rk2: torch.Tensor,
        rerank_score_rk3: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_itc = self.itc_loss(image_feat, caption_feat, logit_scale)
        loss_rank = self.rank_loss(
            sim_rk1=sim_rk1,
            sim_rk2=sim_rk2,
            sim_rk3=sim_rk3,
            rerank_score_rk1=rerank_score_rk1,
            rerank_score_rk2=rerank_score_rk2,
            rerank_score_rk3=rerank_score_rk3,
            valid_mask=valid_mask,
        )

        total = loss_itc + self.cfg.lambda_rank * loss_rank
        stats = {
            "loss_total": total.detach(),
            "loss_itc": loss_itc.detach(),
            "loss_rank": loss_rank.detach(),
        }
        return total, stats


__all__ = ["RankAuxConfig", "ITCWithRankAuxLoss"]
