import argparse
import json
import os
from typing import Any, Dict, List
import random
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_caption_prompt_free() -> str:
    style_pool = [
        "Write the description in a standard person re-identification dataset style.",
        "Provide a slightly detailed and natural description.",
        "Describe the person's appearance clearly and objectively.",
        "Focus on clothing and visible attributes in a natural way.",
    ]

    style_instruction = random.choice(style_pool)

    lines = [
        "You are generating a caption for a person re-identification dataset.",
        "",
        "Requirements:",
        "1. Describe only what is visible in the image.",
        "2. Do NOT hallucinate unseen accessories.",
        "3. Focus mainly on clothing and visible attributes.",
        "4. The caption should contain ONE or TWO natural sentences.",
        "5. The first sentence should describe the main clothing.",
        "6. The second sentence (optional) may add additional visible details such as sleeves, hairstyle, bag, or pose.",
        "7. Keep the style objective and similar to real person re-identification datasets.",
        "8. Do NOT output JSON, field names, or explanations.",
        "",
        "Additional style instruction:",
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
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        self.min_image_side = int(min_image_side)

        self.gen_cfg = dict(gen_cfg)
        self.gen_cfg.setdefault("max_new_tokens", 48)
        self.gen_cfg.setdefault("do_sample", True)
        self.gen_cfg.setdefault("temperature", 0.9)
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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")

    cr_cfg = cfg.get("caption_rewrite", {})
    model_name = cr_cfg.get("hf_model_name", cfg.get("extractor", {}).get("hf_model_name"))

    if not model_name:
        raise ValueError("No model name found in config.")

    rows = load_jsonl(args.input)

    if args.max_images > 0:
        rows = rows[: args.max_images]

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