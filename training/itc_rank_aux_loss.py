import torch
import torch.nn as nn
import torch.nn.functional as F
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
        caption_feat: torch.Tensor,         # (T, D)  T = total captions in batch
        image_to_text_mask: torch.Tensor,   # (B, T)  bool
        text_to_image_index: torch.Tensor,  # (T,)
        logit_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        eps = self.cfg.eps

        # -------- L2 normalize --------
        image_feat = self._l2_normalize(image_feat, eps)
        caption_feat = self._l2_normalize(caption_feat, eps)

        # -------- clamp temperature --------
        logit_scale = logit_scale.clamp(max=self.cfg.max_logit_scale)

        # -------- similarity --------
        logits_i2t = logit_scale * image_feat @ caption_feat.t()  # (B, T)
        logits_t2i = logits_i2t.t()                                # (T, B)

        pos_logits_i = logits_i2t.masked_fill(~image_to_text_mask, float("-inf"))

        # log sum exp over positives
        pos_logsumexp_i = torch.logsumexp(pos_logits_i, dim=1)

        # log sum exp over all
        all_logsumexp_i = torch.logsumexp(logits_i2t, dim=1)

        loss_i = -(pos_logsumexp_i - all_logsumexp_i).mean()


        # 构造 text_to_image_mask  (T, B)
        T = logits_t2i.size(0)
        B = logits_t2i.size(1)

        text_to_image_mask = torch.zeros(
            (T, B),
            dtype=torch.bool,
            device=logits_t2i.device
        )

        text_indices = torch.arange(T, device=logits_t2i.device)
        text_to_image_mask[text_indices, text_to_image_index] = True

        pos_logits_t = logits_t2i.masked_fill(~text_to_image_mask, float("-inf"))

        pos_logsumexp_t = torch.logsumexp(pos_logits_t, dim=1)
        all_logsumexp_t = torch.logsumexp(logits_t2i, dim=1)

        loss_t = -(pos_logsumexp_t - all_logsumexp_t).mean()


        loss_itc = 0.5 * (loss_i + loss_t)
        total_loss = loss_itc

        stats = {
            "loss_total": total_loss.detach(),
            "loss_itc": loss_itc.detach(),
            "loss_rank": image_feat.new_zeros(()),
        }

        return total_loss, stats


__all__ = ["RankAuxConfig", "ITCWithRankAuxLoss"]