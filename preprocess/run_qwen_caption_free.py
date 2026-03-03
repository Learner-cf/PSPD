import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import random
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor



IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: str, recursive: bool) -> List[str]:
    root = Path(image_dir)
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted([str(p.resolve()) for p in files])


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


import random

def build_caption_prompt_free() -> str:
    style_pool = [
        "Write a concise but discriminative description of the person.",
        "Provide a compact description emphasizing identity-related visual attributes.",
        "Describe the visible clothing and accessories in a natural single sentence.",
    ]

    style_instruction = random.choice(style_pool)

    lines = [
        "You are generating a caption for image-text contrastive learning in person re-identification.",
        "",
        "Goal:",
        "The description should help distinguish this person from others.",
        "",
        "Requirements:",
        "1. Describe ONLY visible clothing, colors, and accessories.",
        "2. Use precise and concrete color words.",
        "3. Mention both upper-body and lower-body clothing if visible.",
        "4. Include distinctive attributes such as patterns, stripes, logos, or texture if visible.",
        "5. Mention bags, backpacks, hats, or shoes if visible.",
        "6. Do NOT describe actions, pose, background, or environment.",
        "7. Use one natural and concise sentence.",
        "8. Avoid repetitive sentence patterns and keep expression varied.",
        "",
        "Additional instruction:",
        style_instruction,
        "",
        "Write the caption now:",
    ]

    return "\n".join(lines)

class QwenCaptionGenerator:
    def __init__(
        self,
        model_name: str,
        device: str,
        gen_cfg: Dict[str, Any],
        min_image_side: int = 56,
    ):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        use_multi_gpu = device.startswith("cuda") and torch.cuda.device_count() > 1
        if use_multi_gpu:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            )
            self.input_device = next(self.model.parameters()).device
            print(f"[info] multi-gpu enabled for caption free: {torch.cuda.device_count()} GPUs, input_device={self.input_device}")
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)
            self.input_device = torch.device(device)
        self.model.eval()

        self.min_image_side = int(min_image_side)

        self.gen_cfg = dict(gen_cfg)
        self.gen_cfg.setdefault("max_new_tokens", 64)
        self.gen_cfg.pop("temperature", None)
        self.gen_cfg.pop("top_p", None)
        self.gen_cfg.setdefault("repetition_penalty", 1.1)
        self.gen_cfg["num_beams"] = 1

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
    def generate_one(self, image: Image.Image) -> str:
        prompt = build_caption_prompt_free()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image = self._ensure_image_ok(image)

        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        out = self.model.generate(**inputs, **self.gen_cfg)
        gen_ids = out[0][input_len:]

        ans = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if "\n" in ans:
            ans = ans.splitlines()[0].strip()

        ans = ans.replace("caption:", "").strip(" :")
        return ans

    @torch.inference_mode()
    def generate_batch(self, images: List[Image.Image]) -> List[str]:

        prompts = [build_caption_prompt_free() for _ in images]

        messages_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }] for prompt in prompts
        ]

        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]

        images = [self._ensure_image_ok(img) for img in images]

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        input_lens = inputs["input_ids"].shape[1]

        outputs = self.model.generate(**inputs, **self.gen_cfg)

        captions = []
        for gen_ids in outputs:
            decoded = self.processor.tokenizer.decode(
                gen_ids[input_lens:], skip_special_tokens=True
            ).strip()

            if "\n" in decoded:
                decoded = decoded.splitlines()[0].strip()

            decoded = decoded.replace("caption:", "").strip(" :")
            captions.append(decoded)

        return captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--image_dir", type=str, default="/home/u2024218474/jupyterlab/PSPD/dataset/CUHK-PEDES/imgs")
    ap.add_argument("--output", type=str, default="outputs/initial_caption.jsonl")
    ap.add_argument("--recursive", type=int, default=1, choices=[0, 1])
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")

    cr_cfg = cfg.get("caption_rewrite", {})
    model_name = cr_cfg.get("hf_model_name", cfg.get("extractor", {}).get("hf_model_name"))

    if not model_name:
        raise ValueError("No model name found in config.")

    image_paths = collect_images(args.image_dir, recursive=bool(args.recursive))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    rows = [{"image_path": p} for p in image_paths]

    generator = QwenCaptionGenerator(
        model_name=model_name,
        device=device,
        gen_cfg=cr_cfg.get("generation", {}),
        min_image_side=int(cr_cfg.get("min_image_side", 56)),
    )

    out_rows: List[Dict] = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(rows), batch_size), desc="qwen-caption-free"):
        batch_rows = rows[i:i+batch_size]
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch_rows]

        if batch_size == 1:
            caption = generator.generate_one(images[0])
            out = dict(batch_rows[0])
            out["caption_rewrite"] = {
                "constraints_rk1": None,
                "caption": caption,
            }
            out_rows.append(out)
        else:
            captions = generator.generate_batch(images)
            for row, caption in zip(batch_rows, captions):
                out = dict(row)
                out["caption_rewrite"] = {
                    "constraints_rk1": None,
                    "caption": caption,
                }
                out_rows.append(out)

    save_jsonl(args.output, out_rows)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
