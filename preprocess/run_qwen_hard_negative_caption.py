import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor


def load_neighbor_groups(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"neighbor_groups file must be list, got {type(data)}")
    return data


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(image_path: str, image_root: str) -> str:
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return str(p)
    if image_root:
        candidate = Path(image_root) / image_path
        if candidate.exists():
            return str(candidate.resolve())
        candidate2 = Path(image_root) / p.name
        if candidate2.exists():
            return str(candidate2.resolve())
    if p.exists():
        return str(p.resolve())
    return ""


def parse_generation_cfg(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--generation_cfg must be valid json string: {exc}")
    if not isinstance(cfg, dict):
        raise ValueError("--generation_cfg must decode to a json object")
    return cfg


def build_prompt() -> str:
    return (
        "You are given one target person image and several visually similar persons.\n\n"

        "Your task is to generate a highly discriminative description of the target.\n\n"

        "Internal reasoning steps (do NOT output these steps):\n"
        "1. Identify attributes shared across most similar persons.\n"
        "2. Identify attributes unique to the target.\n"
        "3. Rank unique attributes by distinctiveness.\n"
        "4. Build the description emphasizing top-ranked distinctive features.\n\n"

        "Description Guidelines:\n"
        "- Emphasize distinctive clothing colors or unusual combinations.\n"
        "- Highlight specific garment categories.\n"
        "- Mention logos, patterns, stripes, or prints if present.\n"
        "- Include accessories or hairstyle if distinctive.\n"
        "- Use precise language.\n\n"

        "Strict Constraints:\n"
        "- Do NOT mention the similar persons.\n"
        "- Do NOT describe common traits.\n"
        "- Do NOT hallucinate.\n"
        "- Output only the final description (1–2 sentences).\n"
    )

class HardNegativeCaptionGenerator:
    def __init__(self, model_name: str, generation_cfg: Dict[str, Any], device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        self.generation_cfg = dict(generation_cfg)
        self.generation_cfg.setdefault("max_new_tokens", 96)
        self.generation_cfg.setdefault("do_sample", True)
        self.generation_cfg.setdefault("temperature", 0.7)
        self.generation_cfg.setdefault("top_p", 0.9)
        self.generation_cfg.setdefault("repetition_penalty", 1.05)
        self.generation_cfg.pop("num_beams", None)

    @torch.inference_mode()
    def generate(self, target_img: Image.Image, neighbor_imgs: List[Image.Image]) -> str:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": build_prompt()},
            {"type": "text", "text": "Target image:"},
            {"type": "image"},
            {"type": "text", "text": "Similar persons:"},
        ]
        for idx in range(len(neighbor_imgs)):
            content.append({"type": "image"})
            content.append({"type": "text", "text": f"(neighbor {idx + 1})"})

        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        all_images = [target_img] + neighbor_imgs
        inputs = self.processor(text=chat_text, images=all_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, **self.generation_cfg)
        gen_ids = out[0][input_len:]
        caption = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        caption = " ".join(caption.split())
        return caption


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate hard-negative discriminative captions from neighbor groups.")
    ap.add_argument("--neighbor_groups_path", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="")
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--generation_cfg", type=str, default="{}", help="JSON string for generation config")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    generation_cfg = parse_generation_cfg(args.generation_cfg)

    groups = load_neighbor_groups(args.neighbor_groups_path)
    generator = HardNegativeCaptionGenerator(
        model_name=args.model_name,
        generation_cfg=generation_cfg,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    for group in tqdm(groups, desc="hard-negative-caption"):
        raw_target = str(group.get("image_path", "")).strip()
        if not raw_target:
            continue

        target_path = resolve_image_path(raw_target, args.image_root)
        if not target_path:
            continue

        raw_neighbors = group.get("neighbors", [])
        if not isinstance(raw_neighbors, list):
            raw_neighbors = []

        neighbor_paths: List[str] = []
        for n in raw_neighbors[: max(0, args.top_k)]:
            npath = resolve_image_path(str(n), args.image_root)
            if npath and npath != target_path:
                neighbor_paths.append(npath)

        if len(neighbor_paths) == 0:
            continue

        target_img = Image.open(target_path).convert("RGB")
        neighbor_imgs = [Image.open(p).convert("RGB") for p in neighbor_paths]
        caption = generator.generate(target_img=target_img, neighbor_imgs=neighbor_imgs)
        if not caption:
            continue

        rows.append(
            {
                "image_path": target_path,
                "caption_type": "hard_negative",
                "caption": caption,
            }
        )

    save_jsonl(args.output_jsonl, rows)
    print(f"saved {len(rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
