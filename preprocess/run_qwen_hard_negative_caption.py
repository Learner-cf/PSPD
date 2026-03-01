import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
import yaml
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

    maybe_path = Path(text)
    if maybe_path.exists() and maybe_path.is_file() and maybe_path.suffix.lower() in {".yaml", ".yml"}:
        with open(maybe_path, "r", encoding="utf-8") as f:
            cfg_obj = yaml.safe_load(f) or {}
        if not isinstance(cfg_obj, dict):
            raise ValueError("YAML config root must be a mapping/dict")

        hn_cfg = cfg_obj.get("hard_negative_caption", {})
        if isinstance(hn_cfg, dict):
            gen = hn_cfg.get("generation", {})
            if isinstance(gen, dict):
                return gen

        # fallback to caption_rewrite.generation for convenience
        cr_cfg = cfg_obj.get("caption_rewrite", {})
        if isinstance(cr_cfg, dict):
            gen = cr_cfg.get("generation", {})
            if isinstance(gen, dict):
                return gen

        raise ValueError(
            "YAML config missing generation settings. Expected `hard_negative_caption.generation` "
            "(or fallback `caption_rewrite.generation`)."
        )

    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--generation_cfg must be valid json string or yaml path: {exc}")
    if not isinstance(cfg, dict):
        raise ValueError("--generation_cfg must decode to a json object")
    return cfg




def ensure_min_image_size(img: Image.Image, min_side: int = 28) -> Image.Image:
    w, h = img.size
    if w >= min_side and h >= min_side:
        return img

    scale = max(min_side / max(1, w), min_side / max(1, h))
    new_w = max(min_side, int(round(w * scale)))
    new_h = max(min_side, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def build_prompt() -> str:
    return (
        "You are given ONE target person image and a set of visually similar person images.\n\n"

        "Your goal is to generate a HIGHLY DISCRIMINATIVE caption for the target person.\n"
        "The description must maximize distinguishability in a retrieval setting.\n\n"

        "Important context:\n"
        "- Similar persons may share clothing color or style.\n"
        "- Some similar persons may not be the same identity.\n"
        "- Focus on features that make the target uniquely identifiable.\n\n"

        "Internal reasoning procedure (DO NOT output these steps):\n"
        "1. Identify attributes shared by many similar persons (common traits).\n"
        "2. Suppress common traits.\n"
        "3. Identify fine-grained unique attributes of the target:\n"
        "   - Specific color shades (dark navy, light gray, faded black)\n"
        "   - Multi-color combinations (black jacket with white stripe)\n"
        "   - Logos, patterns, graphics, prints\n"
        "   - Fabric type (denim, leather, puffer, knit)\n"
        "   - Layering structure (hoodie under jacket)\n"
        "   - Accessories (backpack style, hat type, glasses)\n"
        "   - Sleeve length, pant type, footwear category\n"
        "4. Rank unique attributes by distinctiveness.\n"
        "5. Compose a concise but information-dense description emphasizing top distinctive attributes.\n\n"

        "Description requirements:\n"
        "- Describe ONLY visible facts.\n"
        "- Use precise attribute words.\n"
        "- Prefer attribute combinations over isolated attributes.\n"
        "- Avoid vague phrases (e.g., 'casual clothes', 'normal shirt').\n"
        "- Avoid generic colors unless modified (use 'dark brown' instead of 'brown').\n"
        "- Avoid describing pose unless it helps discrimination.\n\n"

        "Hard constraints:\n"
        "- Do NOT mention other persons.\n"
        "- Do NOT describe common attributes shared by many similar persons.\n"
        "- Do NOT hallucinate unseen details.\n"
        "- Output ONLY the final caption.\n"
        "- Produce exactly 1 or 2 sentences.\n"

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

        all_images = [ensure_min_image_size(target_img)] + [ensure_min_image_size(x) for x in neighbor_imgs]
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
    ap.add_argument("--neighbor_groups_path", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/dualdistance3/hn350/neighbor_groups.json")
    ap.add_argument("--image_root", type=str, default="/home/u2024218474/jupyterlab/PSPD/dataset/CUHK-PEDES")
    ap.add_argument("--output_jsonl", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/dualdistance3/hn350/hard_negative_caption.jsonl ")
    ap.add_argument("--model_name", type=str, default="/home/u2024218474/jupyterlab/PSPD/hf_models/Qwen2-VL-7B-Instruct")
    ap.add_argument("--generation_cfg", type=str, default="/home/u2024218474/jupyterlab/PSPD/configs/default.yaml", help="JSON string or YAML path (e.g. configs/default.yaml) for generation config")
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--neighbors_key", type=str, default="out_cluster_neighbors",
                    help="Neighbor list key in neighbor_groups.json (default: out_cluster_neighbors)")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Number of samples processed per chunk for caption generation")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    generation_cfg = parse_generation_cfg(args.generation_cfg)

    # allow convenience defaults from configs/default.yaml when passed via --generation_cfg
    if Path(str(args.generation_cfg)).suffix.lower() in {".yaml", ".yml"} and Path(str(args.generation_cfg)).exists():
        with open(args.generation_cfg, "r", encoding="utf-8") as f:
            cfg_obj = yaml.safe_load(f) or {}
        hn_cfg = cfg_obj.get("hard_negative_caption", {}) if isinstance(cfg_obj, dict) else {}
        if isinstance(hn_cfg, dict):
            if "neighbors_key" in hn_cfg:
                args.neighbors_key = str(hn_cfg["neighbors_key"])
            if "top_k" in hn_cfg:
                args.top_k = int(hn_cfg["top_k"])
            if "batch_size" in hn_cfg:
                args.batch_size = int(hn_cfg["batch_size"])

    groups = load_neighbor_groups(args.neighbor_groups_path)
    generator = HardNegativeCaptionGenerator(
        model_name=args.model_name,
        generation_cfg=generation_cfg,
        device=args.device,
    )

    prepared: List[Dict[str, Any]] = []
    for group in groups:
        raw_target = str(group.get("image_path", "")).strip()
        if not raw_target:
            continue

        target_path = resolve_image_path(raw_target, args.image_root)
        if not target_path:
            continue

        raw_neighbors = group.get(args.neighbors_key, [])
        if not isinstance(raw_neighbors, list):
            raw_neighbors = []

        neighbor_paths: List[str] = []
        for n in raw_neighbors[: max(0, args.top_k)]:
            npath = resolve_image_path(str(n), args.image_root)
            if npath and npath != target_path:
                neighbor_paths.append(npath)

        if len(neighbor_paths) == 0:
            continue

        prepared.append({"target_path": target_path, "neighbor_paths": neighbor_paths})

    rows: List[Dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    for st in tqdm(range(0, len(prepared), batch_size), desc="hard-negative-caption"):
        batch = prepared[st: st + batch_size]
        for item in batch:
            target_img = Image.open(item["target_path"]).convert("RGB")
            neighbor_imgs = [Image.open(p).convert("RGB") for p in item["neighbor_paths"]]
            caption = generator.generate(target_img=target_img, neighbor_imgs=neighbor_imgs)
            if not caption:
                continue

            rows.append(
                {
                    "image_path": item["target_path"],
                    "caption_type": "hard_negative",
                    "caption": caption,
                }
            )

    save_jsonl(args.output_jsonl, rows)
    print(f"saved {len(rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()

