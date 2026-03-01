import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class RankAuxConfig:
    eps: float = 1e-6
    max_logit_scale: float = 100.0  


class ITCWithRankAuxLoss(nn.Module):
    def __init__(self, cfg: RankAuxConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(
        self,
        image_feat: torch.Tensor,           # (B, D)
        caption_feat: torch.Tensor,         # (B, D)
        logit_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        eps = self.cfg.eps

        if image_feat.size(0) != caption_feat.size(0):
            raise ValueError(
                f"Expected one caption per image, but got image batch {image_feat.size(0)} and caption batch {caption_feat.size(0)}"
            )

        # -------- L2 normalize --------
        image_feat = self._l2_normalize(image_feat, eps)
        caption_feat = self._l2_normalize(caption_feat, eps)

        # -------- clamp temperature --------
        logit_scale = logit_scale.clamp(max=self.cfg.max_logit_scale)

        # -------- similarity --------
        logits_i2t = logit_scale * image_feat @ caption_feat.t()  # (B, B)
        logits_t2i = logits_i2t.t()                                # (B, B)

        targets = torch.arange(logits_i2t.size(0), device=logits_i2t.device)
        loss_i = nn.functional.cross_entropy(logits_i2t, targets)
        loss_t = nn.functional.cross_entropy(logits_t2i, targets)


        loss_itc = 0.5 * (loss_i + loss_t)
        total_loss = loss_itc

        stats = {
            "loss_total": total_loss.detach(),
            "loss_itc": loss_itc.detach(),
            "loss_rank": image_feat.new_zeros(()),
        }

        return total_loss, stats


__all__ = ["RankAuxConfig", "ITCWithRankAuxLoss"]
