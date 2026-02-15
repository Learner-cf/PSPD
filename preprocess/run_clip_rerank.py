from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

FIELD_PROMPT = {
    "gender": "a photo of a {value} person",
    "upper_type": "a photo of a person wearing a {value} on the upper body",
    "upper_color": "a photo of a person wearing {value} upper clothing",
    "lower_type": "a photo of a person wearing {value} on the lower body",
    "lower_color": "a photo of a person wearing {value} lower clothing",
}

@dataclass
class RankedCandidate:
    value: str
    score: float

def _parse_yaml_dict(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    sanitized = raw.replace("\t", "  ").replace("：", ":")
    cfg = yaml.safe_load(sanitized)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config root must be a mapping(dict), got: {type(cfg)}")
    return cfg

def load_cfg(path: str) -> Dict:
    try:
        return _parse_yaml_dict(path)
    except Exception as e:
        fallback = Path(path).with_name("clip_rerank.yaml")
        if fallback.exists():
            try:
                cfg = _parse_yaml_dict(str(fallback))
                print(
                    f"[warn] failed to parse config '{path}', fallback to '{fallback}'.\n"
                    f"[warn] original parse error: {e}"
                )
                return cfg
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to parse YAML config: {path}. "
            "Please check indentation and ':' usage. "
            f"Original parser error: {e}"
        ) from e

def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_value(value: str) -> str:
    return (value or "").strip().lower().replace("_", "-")

def build_text(field: str, value: str) -> str:
    template = FIELD_PROMPT.get(field, "a photo of a person with {value}")
    return template.format(value=value)

def _flatten_to_floats(x: Any) -> List[float]:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for item in x:
            out.extend(_flatten_to_floats(item))
        return out
    return [float(x)]

class ClipScorer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        backend: str = "auto",
        open_clip_model_name: str = "ViT-B-16",
        open_clip_pretrained: str = "",
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.backend = self._resolve_backend(backend=backend, model_path=model_path)
        self.logit_scale = 1.0

        if self.backend == "open_clip":
            self._init_open_clip(
                model_path=model_path,
                model_name=open_clip_model_name,
                pretrained=open_clip_pretrained,
            )
        elif self.backend == "hf":
            self._init_hf(model_path=model_path)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @staticmethod
    def _resolve_backend(backend: str, model_path: str) -> str:
        b = (backend or "auto").strip().lower()
        if b in {"hf", "open_clip"}:
            return b
        has_open_clip_lib = importlib.util.find_spec("open_clip") is not None
        has_open_clip_weight = os.path.exists(os.path.join(model_path, "open_clip_pytorch_model.bin"))
        if has_open_clip_lib and has_open_clip_weight:
            return "open_clip"
        return "hf"

    def _init_hf(self, model_path: str):
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        if hasattr(self.model, "logit_scale"):
            self.logit_scale = float(self.model.logit_scale.exp().detach().cpu().item())

    def _init_open_clip(self, model_path: str, model_name: str, pretrained: str):
        import open_clip

        pretrained_source = pretrained or os.path.join(model_path, "open_clip_pytorch_model.bin")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained_source,
            device=self.device,
        )
        self.oc_model = model
        self.oc_preprocess = preprocess
        self.oc_tokenizer = open_clip.get_tokenizer(model_name)
        self.oc_model.eval()
        if hasattr(self.oc_model, "logit_scale"):
            self.logit_scale = float(self.oc_model.logit_scale.exp().detach().cpu().item())

    @torch.inference_mode()
    def score_texts(self, images: List[Image.Image], texts_list: List[List[str]], batch_size: int = 64) -> List[List[float]]:
        # images: [B], texts_list: [B][N]
        # Returns: [B][N]
        assert len(images) == len(texts_list)
        if self.backend == "open_clip":
            return self._score_texts_open_clip(images, texts_list, batch_size)
        else:
            return self._score_texts_hf(images, texts_list, batch_size)

    @torch.inference_mode()
    def _score_texts_hf(self, images: List[Image.Image], texts_list: List[List[str]], batch_size: int = 64) -> List[List[float]]:
        # For each image, compute its own text candidates (支持batch多图片)
        results = []
        for img, texts in zip(images, texts_list):
            img = img.convert("RGB")
            out_scores: List[float] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                inputs = self.processor(text=chunk, images=[img], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                text_features = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                sims = (image_features @ text_features.t()) * self.logit_scale
                out_scores.extend(_flatten_to_floats(sims.squeeze(0)))
            results.append(out_scores)
        return results

    @torch.inference_mode()
    def _score_texts_open_clip(self, images: List[Image.Image], texts_list: List[List[str]], batch_size: int = 64) -> List[List[float]]:
        # open_clip 必须支持多图片
        results = []
        for img, texts in zip(images, texts_list):
            image_tensor = self.oc_preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
            image_features = self.oc_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            out_scores: List[float] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                text_tokens = self.oc_tokenizer(chunk).to(self.device)
                text_features = self.oc_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                sims = (image_features @ text_features.t()) * self.logit_scale
                out_scores.extend(_flatten_to_floats(sims.squeeze(0)))
            results.append(out_scores)
        return results

def rank_generated_candidates(
    scorer: ClipScorer,
    images: List[Image.Image],
    fields: List[str],
    values_list: List[List[str]],
    batch_text_size: int
) -> List[List[RankedCandidate]]:
    texts_list = [
        [build_text(field, normalize_value(v)) for v in values]
        for field, values in zip(fields, values_list)
    ]
    scores_list = scorer.score_texts(images, texts_list, batch_size=batch_text_size)
    ranked = []
    for values, scores in zip(values_list, scores_list):
        normalized = [normalize_value(v) for v in values]
        pairs = [RankedCandidate(v, float(s)) for v, s in zip(normalized, scores)]
        pairs.sort(key=lambda x: x.score, reverse=True)
        ranked.append(pairs)
    return ranked

def rank_slot_candidates(
    scorer: ClipScorer,
    images: List[Image.Image],
    fields: List[str],
    slot_words_list: List[List[str]],
    batch_text_size: int
) -> List[List[RankedCandidate]]:
    texts_list = [
        [build_text(field, normalize_value(v)) for v in slot_words]
        for field, slot_words in zip(fields, slot_words_list)
    ]
    scores_list = scorer.score_texts(images, texts_list, batch_size=batch_text_size)
    ranked = []
    for slot_words, scores in zip(slot_words_list, scores_list):
        normalized = [normalize_value(v) for v in slot_words]
        pairs = [RankedCandidate(v, float(s)) for v, s in zip(normalized, scores)]
        pairs.sort(key=lambda x: x.score, reverse=True)
        ranked.append(pairs)
    return ranked

def merge_unique_ranked(
    generated_rank: List[RankedCandidate],
    slot_rank: List[RankedCandidate],
    top_k: int = 3,
) -> List[RankedCandidate]:
    merged = sorted(generated_rank + slot_rank, key=lambda x: x.score, reverse=True)
    unique: List[RankedCandidate] = []
    used = set()
    for candidate in merged:
        value = normalize_value(candidate.value)
        if not value or value in used:
            continue
        used.add(value)
        unique.append(RankedCandidate(value, float(candidate.score)))
        if len(unique) >= top_k:
            break
    while len(unique) < top_k:
        unique.append(RankedCandidate("unknown", -1e9))
    return unique

def process_batch(
    batch_rows: List[Dict],
    scorer: ClipScorer,
    slots: Dict[str, List[str]],
    threshold: Dict[str, float],
    batch_text_size: int,
    top_k: int = 3,
) -> List[Dict]:
    # 支持同时处理一个 batch 的多图片
    images = [Image.open(row["image_path"]).convert("RGB") for row in batch_rows]
    fields_list = [list(row.get("fields", {}).keys()) for row in batch_rows]
    rerank_fields_list = []

    for batch_idx, row in enumerate(batch_rows):
        fields = list(row.get("fields", {}).keys())
        parsed_values_list = [row["fields"][f].get("parsed_values", []) for f in fields]
        slot_words_list = [slots.get(f, []) if isinstance(slots, dict) else [] for f in fields]

        gen_rank = rank_generated_candidates(scorer, [images[batch_idx]] * len(fields), fields, parsed_values_list, batch_text_size)
        slot_rank = rank_slot_candidates(scorer, [images[batch_idx]] * len(fields), fields, slot_words_list, batch_text_size)

        rerank_fields: Dict[str, Dict] = {}
        for idx, field in enumerate(fields):
            final_rank = merge_unique_ranked(gen_rank[idx], slot_rank[idx], top_k=top_k)
            rerank_fields[field] = {
                "generated_rank": [{"value": c.value, "score": c.score} for c in gen_rank[idx]],
                "slot_rank": [{"value": c.value, "score": c.score} for c in slot_rank[idx]],
                "final_rank": [{"value": c.value, "score": c.score} for c in final_rank],
                "top1_replaced": (
                    bool(gen_rank[idx] and final_rank and gen_rank[idx][0].value != final_rank[0].value)
                ),
                "threshold_used": float(threshold.get(field, threshold.get("default", 0.0))),
                "enforce_unique_top3": True,
            }
        row["clip_rerank"] = rerank_fields
    return batch_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["cuhk", "icfg", "rstp"])
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--image_batch_size", type=int, default=1)
    ap.add_argument("--batch_text_size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    rr_cfg = cfg.get("clip_rerank", {})
    model_path = rr_cfg.get("model_name_or_path", "openai/clip-vit-base-patch16")
    device = cfg.get("device", "cuda")
    threshold = rr_cfg.get("threshold", {"default": 20.0})
    backend = rr_cfg.get("backend", "auto")
    open_clip_model_name = rr_cfg.get("open_clip_model_name", "ViT-B-16")
    open_clip_pretrained = rr_cfg.get("open_clip_pretrained", "")

    slots_path = rr_cfg.get("slots_path", "configs/attr_slots.yaml")
    with open(slots_path, "r", encoding="utf-8") as f:
        slots = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "outputs")
    input_path = args.input or os.path.join(out_dir, f"{args.dataset}_{args.split}_qwen_fields3_raw.jsonl")
    output_path = args.output or os.path.join(out_dir, f"{args.dataset}_{args.split}_qwen_fields3_clip_rerank.jsonl")

    rows = load_jsonl(input_path)
    if args.max_images and args.max_images > 0:
        rows = rows[:args.max_images]
    scorer = ClipScorer(
        model_path=model_path,
        device=device,
        backend=backend,
        open_clip_model_name=open_clip_model_name,
        open_clip_pretrained=open_clip_pretrained,
    )

    out_rows: List[Dict] = []
    batch_size = max(1, args.image_batch_size)
    for i in tqdm(range(0, len(rows), batch_size), desc=f"clip-rerank {args.dataset}:{args.split}"):
        batch_rows = rows[i:i + batch_size]
        batch_results = process_batch(batch_rows, scorer, slots, threshold, args.batch_text_size, top_k=3)
        out_rows.extend(batch_results)

    save_jsonl(output_path, out_rows)
    print(f"saved: {output_path}")

if __name__ == "__main__":
    main()