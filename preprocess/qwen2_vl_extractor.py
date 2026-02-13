import random
from typing import Any, Dict, List

import torch
from PIL import Image

from preprocess.field_prompts import blip_prompt

class Qwen2VLHFExtractor:
    """
    Qwen2-VL extractor:
      - prompt/采样都���制field: value格式，无unknown
      - 采样3次、尽量去重
      - 小图自动resize
      - decode仅取新生成tokens，若无field则自动补field
    """
    def __init__(
        self,
        model_name: str,
        num_samples: int,
        gen_cfg: Dict[str, Any],
        device: str = "cuda",
        seed: int = 42,
        max_words: int = 3,
        max_tries_per_field: int = 12,
        min_image_side: int = 64,
    ):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.num_samples = int(num_samples)
        self.device = device
        self.max_words = int(max_words)

        self.max_tries_per_field = int(max_tries_per_field)
        self.min_image_side = int(min_image_side)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # generation config
        self.gen_cfg = dict(gen_cfg)
        self.gen_cfg.setdefault("max_new_tokens", 24)
        self.gen_cfg.setdefault("do_sample", True)
        self.gen_cfg.setdefault("temperature", 0.8)
        self.gen_cfg.setdefault("top_p", 0.9)
        self.gen_cfg.setdefault("repetition_penalty", 1.05)
        self.gen_cfg.pop("num_beams", None)

        self.base_seed = int(seed)
        random.seed(self.base_seed)

        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError(
                "Processor has no apply_chat_template(); incompatible transformers/processor for Qwen2-VL."
            )

    def _ensure_image_ok(self, pil_image: Image.Image) -> Image.Image:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        w, h = pil_image.size
        min_side = min(w, h)
        if min_side < self.min_image_side:
            scale = self.min_image_side / float(min_side)
            new_w = max(self.min_image_side, int(round(w * scale)))
            new_h = max(self.min_image_side, int(round(h * scale)))
            pil_image = pil_image.resize((new_w, new_h), resample=Image.BICUBIC)
        return pil_image

    def _build_messages(self, field: str) -> List[Dict[str, Any]]:
        prompt = blip_prompt(field, max_words=self.max_words)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _set_random_seed(self) -> int:
        s = random.randint(0, 2**31 - 1)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        return s

    @torch.inference_mode()
    def _gen_one(self, pil_image: Image.Image, field: str) -> str:
        pil_image = self._ensure_image_ok(pil_image)
        self._set_random_seed()
        messages = self._build_messages(field)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, **self.gen_cfg)  # (1, seq_len)
        gen_ids = out[0][input_len:]
        ans = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if "\n" in ans:
            ans = ans.splitlines()[0].strip()
        # 如不是"field: value"格式自动补前缀
        if not ans.lower().startswith(f"{field.lower()}:"):
            ans = f"{field}: {ans}"
        # 彻底禁止unknown等
        ban = {"unknown", "unsure", "uncertain", "n/a", "none", "null"}
        v = ans.split(":")[-1].strip().lower()
        if v in ban:
            return ""
        return ans

    @torch.inference_mode()
    def extract_field_n(self, pil_image: Image.Image, field: str) -> List[str]:
        """
        生成 num_samples 个候选，优先用唯一值，多次尝试去重。
        """
        results: List[str] = []
        seen = set()
        max_tries = max(self.max_tries_per_field, self.num_samples)
        for _ in range(max_tries):
            if len(results) >= self.num_samples:
                break
            s = self._gen_one(pil_image, field)
            key = s.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(s)
        # 不足补齐
        while len(results) < self.num_samples:
            results.append(results[-1] if results else f"{field}: ")
        return results