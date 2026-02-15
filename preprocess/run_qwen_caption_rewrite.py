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

def get_rk1_value(row: Dict, field: str) -> str:
    rr = row.get("clip_rerank", {})
    fobj = rr.get(field, {})
    rank = fobj.get("final_rank", [])
    if rank and isinstance(rank[0], dict):
        v = str(rank[0].get("value", "")).strip().lower()
        return v
    return "unknown"

def build_caption_prompt(attrs: Dict[str, str]) -> str:
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
        "Confirmed attributes:",
    ]

    for k, v in attrs.items():
        lines.append(f"- {k}: {v}")

    lines.extend([
        "",
        "Requirements:",
        "1. The description MUST be fully consistent with the confirmed attributes.",
        "2. Do NOT contradict any attribute.",
        "3. Do NOT add clothing items that are not visible.",
        "4. Avoid hallucinating accessories unless clearly visible.",
        "5. Focus mainly on clothing and appearance.",
        "6. The caption should contain ONE or TWO natural sentences.",
        "7. The first sentence should describe the main clothing.",
        "8. The second sentence (optional) may add additional visible details such as sleeves, hairstyle, bag, or pose.",
        "9. Keep the style objective and similar to real person re-identification datasets.",
        "10. Do NOT output JSON, field names, or explanations.",
        "",
        "Additional style instruction:",
        style_instruction,
        "",
        "Write the caption now:",
    ])

    return "\n".join(lines)


class QwenCaptionRewriter:
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
    def generate_one(self, image: Image.Image, attrs: Dict[str, str]) -> str:
        prompt = build_caption_prompt(attrs)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    def generate_batch(self, images: List[Image.Image], attrs_list: List[Dict[str, str]]) -> List[str]:
        # 批量生成描述
        prompts = [build_caption_prompt(attrs) for attrs in attrs_list]
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
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        images = [self._ensure_image_ok(img) for img in images]

        # 支持 batch 的 preprocess
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_lens = inputs["input_ids"].shape[1]

        # 调用 generate 并处理 batch
        # batch_size = len(texts)
        outputs = self.model.generate(**inputs, **self.gen_cfg)
        captions = []
        for i, gen_ids in enumerate(outputs):
            # gen_ids: [seq_len]，取出新生成的部分
            decoded = self.processor.tokenizer.decode(gen_ids[input_lens:], skip_special_tokens=True).strip()
            if "\n" in decoded:
                decoded = decoded.splitlines()[0].strip()
            decoded = decoded.replace("caption:", "").strip(" :")
            captions.append(decoded)
        return captions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--input", type=str, default="outputs/cuhk_train_qwen_fields3_clip_rerank.jsonl")
    ap.add_argument("--output", type=str, default="outputs/cuhk_train_qwen_caption_v1.jsonl")
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=1, help="Number of samples for one batch inference")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")

    cr_cfg = cfg.get("caption_rewrite", {})
    model_name = cr_cfg.get("hf_model_name", cfg.get("extractor", {}).get("hf_model_name"))
    if not model_name:
        raise ValueError("No caption_rewrite.hf_model_name or extractor.hf_model_name in config")

    rows = load_jsonl(args.input)
    if args.max_images and args.max_images > 0:
        rows = rows[: args.max_images]

    rewriter = QwenCaptionRewriter(
        model_name=model_name,
        device=device,
        gen_cfg=cr_cfg.get("generation", {}),
        min_image_side=int(cr_cfg.get("min_image_side", 56)),
    )

    out_rows: List[Dict] = []
    batch_size = args.batch_size
    for i in tqdm(range(0, len(rows), batch_size), desc="qwen-caption-rewrite"):
        batch_rows = rows[i:i+batch_size]
        attrs_list = []
        images = []
        for row in batch_rows:
            attrs = {
                "gender": get_rk1_value(row, "gender"),
                "upper_type": get_rk1_value(row, "upper_type"),
                "upper_color": get_rk1_value(row, "upper_color"),
                "lower_type": get_rk1_value(row, "lower_type"),
                "lower_color": get_rk1_value(row, "lower_color"),
            }
            attrs_list.append(attrs)
            img = Image.open(row["image_path"]).convert("RGB")
            images.append(img)
        if batch_size == 1:
            # 向后兼容: 如果用户设置为1, 用原来的流程
            caption = rewriter.generate_one(images[0], attrs_list[0])
            out = dict(batch_rows[0])
            out["caption_rewrite"] = {
                "constraints_rk1": attrs_list[0],
                "caption": caption,
            }
            out_rows.append(out)
        else:
            captions = rewriter.generate_batch(images, attrs_list)
            for row, attrs, caption in zip(batch_rows, attrs_list, captions):
                out = dict(row)
                out["caption_rewrite"] = {
                    "constraints_rk1": attrs,
                    "caption": caption,
                }
                out_rows.append(out)

    save_jsonl(args.output, out_rows)
    print(f"saved: {args.output}")

if __name__ == "__main__":
    main()