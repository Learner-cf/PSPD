from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class DinoV2Backbone(nn.Module):
    """DINOv2 visual backbone returning L2-normalized CLS embedding."""

    def __init__(
        self,
        model_path: str = "/home/u2024218474/jupyterlab/PSPD/hf_models/AI-ModelScope/dinov2-giant",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Cannot infer hidden_size from DINO config.")
        self.embed_dim = int(hidden_size)

    def preprocess(self, images, return_tensors: str = "pt"):
        return self.image_processor(images=images, return_tensors=return_tensors)

    def forward(self, pixel_values: torch.Tensor, return_raw: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        model = self.backbone
        outputs = model(pixel_values=pixel_values)

        def _print_stats(name: str, tensor) -> None:
            if tensor is None:
                return
            if not torch.is_tensor(tensor):
                return
            t = tensor.detach().float()
            print(
                f"[FORWARD DIAG] {name}: shape={tuple(t.shape)} "
                f"mean={t.mean().item():.6f} "
                f"std={t.std(unbiased=False).item():.6f}"
            )

        if not hasattr(model, "_forward_debug_printed"):
            print("========== DINO FORWARD OUTPUT DIAGNOSTIC ==========")
            print("Output type:", type(outputs))

            if isinstance(outputs, dict):
                print("Output keys:", outputs.keys())
                for key in outputs.keys():
                    _print_stats(str(key), outputs[key])
            else:
                if hasattr(outputs, "__dict__"):
                    print("Output attributes:", dir(outputs))
                for attr in [
                    "last_hidden_state",
                    "pooler_output",
                    "x_norm_clstoken",
                    "x_norm_patchtokens",
                ]:
                    if hasattr(outputs, attr):
                        _print_stats(attr, getattr(outputs, attr))

            model._forward_debug_printed = True

        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("DINO forward output missing last_hidden_state.")

        selected_embedding = outputs.last_hidden_state[:, 0, :]
        raw_emb = selected_embedding.detach().clone()
        emb = F.normalize(selected_embedding, p=2, dim=-1)

        if not hasattr(model, "_selected_emb_debug_printed"):
            raw_stats = raw_emb.detach().float()
            emb_stats = emb.detach().float()
            print("========== FINAL SELECTED EMBEDDING ==========")
            print("Shape:", tuple(emb_stats.shape))
            print("Mean:", emb_stats.mean().item())
            print("Std:", emb_stats.std(unbiased=False).item())
            print("Min:", emb_stats.min().item())
            print("Max:", emb_stats.max().item())
            print("Raw Embedding Std:", raw_stats.std(unbiased=False).item())
            print("Normalized Embedding Std:", emb_stats.std(unbiased=False).item())
            print("==============================================")
            model._selected_emb_debug_printed = True

        if return_raw:
            return emb, raw_emb
        return emb

    def freeze_first_blocks(self, ratio: float = 0.7) -> None:
        """Freeze first ratio of transformer blocks for Stage 3."""
        for p in self.backbone.parameters():
            p.requires_grad = True

        blocks: Optional[nn.ModuleList] = None
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            blocks = self.backbone.encoder.layer
        elif hasattr(self.backbone, "vit") and hasattr(self.backbone.vit, "encoder") and hasattr(self.backbone.vit.encoder, "layer"):
            blocks = self.backbone.vit.encoder.layer

        if blocks is None:
            return

        n_blocks = len(blocks)
        n_freeze = int(n_blocks * ratio)
        for i in range(n_freeze):
            for p in blocks[i].parameters():
                p.requires_grad = False
