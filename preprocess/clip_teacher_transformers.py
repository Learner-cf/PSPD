from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

from preprocess.schema import LabelSchema

class TransformersCLIPTeacher:
    """
    Offline CLIP teacher using HuggingFace Transformers.
    Loads from a local directory that contains:
      - open_clip_pytorch_model.bin (or safetensors)
      - open_clip_config.json
      - tokenizer files (vocab.json, merges.txt, tokenizer.json, etc.)
    """
    def __init__(self, model_dir: str, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        self.model = CLIPModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(device)
        self.model.eval()

        self._field_texts: Dict[str, List[str]] = {}
        self._field_text_inputs: Dict[str, Dict[str, torch.Tensor]] = {}

    @torch.inference_mode()
    def encode_image(self, pil: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        img_feat = self.model.get_image_features(**inputs)
        img_feat = F.normalize(img_feat, dim=-1)
        return img_feat  # (1, d)

    def build_field_candidates(self, schema: LabelSchema):
        fc = schema.field_candidates()
        self._field_texts = {
            "upper_color": [f"upper color: {v}." for v in fc["upper_color"]],
            "lower_color": [f"lower color: {v}." for v in fc["lower_color"]],
            "upper_type":  [f"upper type: {v}." for v in fc["upper_type"]],
            "lower_type":  [f"lower type: {v}." for v in fc["lower_type"]],
        }
        # pre-tokenize once
        for field, texts in self._field_texts.items():
            ti = self.processor(text=texts, return_tensors="pt", padding=True)
            self._field_text_inputs[field] = {k: v.to(self.device) for k, v in ti.items()}

    @torch.inference_mode()
    def score_field_candidates(
        self, image_feat: torch.Tensor, tau: float = 0.01
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        scores_out: Dict[str, List[float]] = {}
        probs_out: Dict[str, List[float]] = {}
        for field, text_inputs in self._field_text_inputs.items():
            txt_feat = self.model.get_text_features(**text_inputs)
            txt_feat = F.normalize(txt_feat, dim=-1)  # (n, d)
            scores = (image_feat @ txt_feat.T).squeeze(0)  # (n,)
            probs = F.softmax(scores / tau, dim=-1)
            scores_out[field] = scores.detach().float().cpu().tolist()
            probs_out[field] = probs.detach().float().cpu().tolist()
        return scores_out, probs_out

    @torch.inference_mode()
    def score_caption_triplet(self, image_feat: torch.Tensor, captions: List[str]) -> List[float]:
        ti = self.processor(text=captions, return_tensors="pt", padding=True)
        ti = {k: v.to(self.device) for k, v in ti.items()}
        txt_feat = self.model.get_text_features(**ti)
        txt_feat = F.normalize(txt_feat, dim=-1)
        scores = (image_feat @ txt_feat.T).squeeze(0)
        return scores.detach().float().cpu().tolist()