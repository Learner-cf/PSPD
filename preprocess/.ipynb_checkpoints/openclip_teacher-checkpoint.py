from typing import Dict, List, Tuple, Optional
import os
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

from preprocess.schema import LabelSchema

def _load_state_dict_any(path: str) -> Dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

class OpenCLIPTeacher:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        image_size: int,
        device: str = "cuda",
        local_weights_path: Optional[str] = None,
    ):
        self.device = device
        self.image_size = image_size

        # Always build skeleton first (no network)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None, device=device
        )

        if local_weights_path:
            if not os.path.exists(local_weights_path):
                raise FileNotFoundError(f"OpenCLIP local weights not found: {local_weights_path}")
            state = _load_state_dict_any(local_weights_path)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[openclip] loaded local weights: {local_weights_path}")
            if missing:
                print(f"[openclip] missing keys: {len(missing)}")
            if unexpected:
                print(f"[openclip] unexpected keys: {len(unexpected)}")
        else:
            # will download unless cached; keep for completeness
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self.model = model
            self.preprocess = preprocess
            print(f"[openclip] loaded pretrained={pretrained}")

        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self._field_texts: Dict[str, List[str]] = {}
        self._field_text_emb: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def encode_image(self, pil: Image.Image) -> torch.Tensor:
        img = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img)
        feat = F.normalize(feat, dim=-1)
        return feat

    def build_field_candidates(self, schema: LabelSchema):
        fc = schema.field_candidates()
        self._field_texts = {
            "upper_color": [f"upper color: {v}." for v in fc["upper_color"]],
            "lower_color": [f"lower color: {v}." for v in fc["lower_color"]],
            "upper_type":  [f"upper type: {v}." for v in fc["upper_type"]],
            "lower_type":  [f"lower type: {v}." for v in fc["lower_type"]],
        }

        for field, texts in self._field_texts.items():
            with torch.inference_mode():
                tokens = self.tokenizer(texts).to(self.device)
                txt = self.model.encode_text(tokens)
                txt = F.normalize(txt, dim=-1)
            self._field_text_emb[field] = txt

    @torch.inference_mode()
    def score_field_candidates(self, image_feat: torch.Tensor, tau: float = 0.01) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        scores_out: Dict[str, List[float]] = {}
        probs_out: Dict[str, List[float]] = {}
        for field, txt_emb in self._field_text_emb.items():
            scores = (image_feat @ txt_emb.T).squeeze(0)
            probs = F.softmax(scores / tau, dim=-1)
            scores_out[field] = scores.detach().cpu().tolist()
            probs_out[field] = probs.detach().cpu().tolist()
        return scores_out, probs_out

    @torch.inference_mode()
    def score_caption_triplet(self, image_feat: torch.Tensor, captions: List[str]) -> List[float]:
        tokens = self.tokenizer(captions).to(self.device)
        txt = self.model.encode_text(tokens)
        txt = F.normalize(txt, dim=-1)
        scores = (image_feat @ txt.T).squeeze(0)
        return scores.detach().cpu().tolist()