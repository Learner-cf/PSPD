import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoModelForVision2Seq, AutoProcessor


STYLE_ORDER = ["global", "attribute", "detail", "concise", "template"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


ATTRIBUTE_SCHEMA = {
    # Identity cues
    "gender": "male/female/unknown",
    "body_build": "slim/average/stocky/unknown",
    "age_group": "young/adult/elderly/unknown",

    # Upper body
    "upper_color": "precise color (e.g., dark red, navy blue, light gray)",
    "upper_type": "t-shirt/shirt/jacket/coat/hoodie/sweater/vest/unknown",
    "upper_texture": "solid/striped/plaid/logo/graphic/printed/unknown",
    "upper_length": "cropped/regular/long/unknown",

    # Lower body
    "lower_color": "precise color",
    "lower_type": "pants/jeans/shorts/skirt/dress/unknown",
    "lower_texture": "solid/striped/plaid/unknown",

    # Shoes
    "shoe_color": "precise color",
    "shoe_type": "sneakers/boots/leather shoes/sandals/unknown",

    # Accessories
    "bag_type": "backpack/handbag/shoulder bag/tote/no bag/unknown",
    "bag_position": "left/right/back/unknown",
    "hat_type": "cap/beanie/hat/no hat/unknown",
    "glasses": "yes/no/unknown",
    "mask": "yes/no/unknown",

    # Fine details
    "hair_length": "short/medium/long/unknown",
    "sleeve_length": "short/long/sleeveless/unknown",
    "visible_logo": "yes/no/unknown",
    "dominant_color_contrast": "high contrast/low contrast/unknown"
}

def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_images(image_dir: str, recursive: bool) -> List[str]:
    root = Path(image_dir)
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted([str(p.resolve()) for p in files])


def build_attribute_prompt() -> str:
    fields = "\n".join([f'- "{k}": {v}' for k, v in ATTRIBUTE_SCHEMA.items()])

    return (
        "You are an expert in person re-identification.\n"
        "Your task is to extract discriminative attributes of the SAME person.\n"
        "Only describe what is clearly visible.\n"
        "Avoid generic words like 'casual', 'normal', 'regular'.\n"
        "Be precise with colors (e.g., 'dark navy blue' instead of 'blue').\n"
        "If uncertain, use 'unknown'.\n\n"
        "Return ONLY valid JSON.\n"
        "Fill these fields:\n"
        f"{fields}\n"
    )


def build_style_prompt(style: str, attrs: Dict[str, str]) -> str:
    attr_lines = "\n".join([f"- {k}: {v}" for k, v in attrs.items()])

    style_rules = {
        "global": [
            "Write a global identity description for person ReID.",
            "Focus on overall silhouette, dominant colors, and main garments.",
            "Avoid generic adjectives like 'casual' or 'normal'.",
            "Make the sentence discriminative."
        ],
        "attribute": [
            "Write an attribute-focused caption.",
            "Explicitly mention upper and lower colors and garment types.",
            "Include accessories and position if visible.",
            "Prioritize rare or unusual traits."
        ],
        "detail": [
            "Write a fine-grained detail caption.",
            "Mention texture, logos, contrast patterns, material hints.",
            "Highlight subtle but distinctive elements.",
            "Do NOT repeat the global description wording."
        ],
        "concise": [
            "Write ONE short but highly discriminative sentence.",
            "Keep only the strongest identity clues.",
            "Remove any redundant information."
        ],
        "template": [
            "Use this structured format:",
            "A [gender] with [hair_length] hair wearing "
            "[upper_color] [upper_type] and "
            "[lower_color] [lower_type], "
            "carrying a [bag_type] on the [bag_position], "
            "with [shoe_color] [shoe_type].",
            "Replace uncertain fields with 'unknown'."
        ],
    }

    rules = "\n".join([f"- {r}" for r in style_rules[style]])

    return (
        "You are generating a caption for unsupervised person ReID.\n"
        "All captions across styles MUST describe the SAME person.\n"
        "Do NOT hallucinate unseen attributes.\n"
        "Avoid repeating the same phrasing across styles.\n\n"
        "Known attributes:\n"
        f"{attr_lines}\n\n"
        f"Target style: {style}\n"
        f"Rules:\n{rules}\n"
        "Output only the caption text."
    )

class QwenCaptionRewriter:
    def __init__(self, model_name: str, device: str, gen_cfg: Dict[str, Any], min_image_side: int = 56):
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
        self.min_image_side = int(min_image_side)

        self.gen_cfg = dict(gen_cfg)
        self.gen_cfg.setdefault("max_new_tokens", 96)
        self.gen_cfg.setdefault("do_sample", True)
        self.gen_cfg.setdefault("temperature", 0.7)
        self.gen_cfg.setdefault("top_p", 0.9)
        self.gen_cfg.setdefault("repetition_penalty", 1.05)
        self.gen_cfg.pop("num_beams", None)

    def _ensure_image_ok(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        min_side = min(w, h)
        if min_side < self.min_image_side:
            scale = self.min_image_side / float(min_side)
            new_w = max(self.min_image_side, int(round(w * scale)))
            new_h = max(self.min_image_side, int(round(h * scale)))
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        return image

    @torch.inference_mode()
    def ask_image_question(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = self._ensure_image_ok(image)
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, **self.gen_cfg)
        gen_ids = out[0][input_len:]
        ans = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return ans

    def extract_attributes(self, image: Image.Image) -> Dict[str, str]:
        prompt = build_attribute_prompt()
        raw = self.ask_image_question(image, prompt)
        left = raw.find("{")
        right = raw.rfind("}")
        candidate = raw[left : right + 1] if left >= 0 and right > left else "{}"
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = {}

        attrs: Dict[str, str] = {}
        for key in ATTRIBUTE_SCHEMA.keys():
            value = str(data.get(key, "unknown")).strip().lower()
            attrs[key] = value if value else "unknown"
        return attrs

    def generate_multi_captions(self, image: Image.Image, attrs: Dict[str, str]) -> List[str]:
        captions: List[str] = []
        for style in STYLE_ORDER:
            prompt = build_style_prompt(style=style, attrs=attrs)
            cap = self.ask_image_question(image, prompt).strip()
            if "\n" in cap:
                cap = cap.splitlines()[0].strip()
            cap = cap.strip(" :")
            if cap:
                captions.append(cap)

        uniq: List[str] = []
        seen = set()
        for c in captions:
            if c in seen:
                continue
            seen.add(c)
            uniq.append(c)
        while len(uniq) < 5:
            uniq.append(uniq[-1] if uniq else "a person")
        return uniq[:5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--image_dir", type=str, required=True, default="/root/autodl-tmp/PSPD/dataset/CUHK-PEDES",help="Directory containing input person images")
    ap.add_argument("--output", type=str, default="outputs/cuhk_caption_multi.jsonl")
    ap.add_argument("--recursive", type=int, default=1, choices=[0, 1])
    ap.add_argument("--max_images", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")

    cr_cfg = cfg.get("caption_rewrite", {})
    model_name = cr_cfg.get("hf_model_name", cfg.get("extractor", {}).get("hf_model_name"))
    if not model_name:
        raise ValueError("No caption_rewrite.hf_model_name or extractor.hf_model_name in config")

    image_paths = collect_images(args.image_dir, recursive=bool(args.recursive))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    rewriter = QwenCaptionRewriter(
        model_name=model_name,
        device=device,
        gen_cfg=cr_cfg.get("generation", {}),
        min_image_side=int(cr_cfg.get("min_image_side", 56)),
    )

    out_rows: List[Dict] = []
    for image_path in tqdm(image_paths, desc="qwen-caption-rewrite"):
        image = Image.open(image_path).convert("RGB")
        attrs = rewriter.extract_attributes(image)
        captions = rewriter.generate_multi_captions(image, attrs)
        out_rows.append(
            {
                "image_path": image_path,
                "captions": captions,
                "caption": captions[0],
                "caption_rewrite": {
                    "styles": STYLE_ORDER,
                    "auto_attributes": attrs,
                },
            }
        )

    save_jsonl(args.output, out_rows)
    print(f"saved: {args.output} ({len(out_rows)} samples)")


if __name__ == "__main__":
    main()
