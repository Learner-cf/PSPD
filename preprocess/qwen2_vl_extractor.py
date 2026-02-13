import random
from typing import Any, Dict, List

import torch
from PIL import Image

from preprocess.field_prompts import blip_prompt


class Qwen2VLHFExtractor:
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

        self.default_num_samples = int(num_samples)
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

        # base sampling cfg for non-gender fields (FORCE sampling on)
        self.gen_cfg = dict(gen_cfg)
        self.gen_cfg["max_new_tokens"] = int(self.gen_cfg.get("max_new_tokens", 24))
        self.gen_cfg["do_sample"] = True
        self.gen_cfg["temperature"] = float(self.gen_cfg.get("temperature", 0.9))
        self.gen_cfg["top_p"] = float(self.gen_cfg.get("top_p", 0.9))
        self.gen_cfg["repetition_penalty"] = float(self.gen_cfg.get("repetition_penalty", 1.05))

        # make sure sampling isn't accidentally combined with beam search
        self.gen_cfg.pop("num_beams", None)
        self.gen_cfg.pop("num_return_sequences", None)

        self.base_seed = int(seed)
        random.seed(self.base_seed)

        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError("Processor has no apply_chat_template(); incompatible transformers/processor.")

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
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

    def _set_random_seed(self):
        s = random.randint(0, 2**31 - 1)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _clean_gen_cfg(self, gen_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove incompatible parameters to prevent transformers warnings.
        - If do_sample is False, remove sampling-only params (temperature/top_p/top_k/typical_p etc.)
        - Force num_beams=1 for greedy decoding unless user explicitly sets otherwise.
        """
        cfg = dict(gen_cfg)

        do_sample = cfg.get("do_sample", False)
        if not do_sample:
            # remove sampling-related params that trigger warnings when do_sample=False
            for k in (
                "temperature",
                "top_p",
                "top_k",
                "typical_p",
                "min_p",
                "epsilon_cutoff",
                "eta_cutoff",
            ):
                cfg.pop(k, None)

            # also ensure we don't accidentally use multi-beam / multi-seq in deterministic mode
            cfg["num_beams"] = 1
            cfg.pop("num_return_sequences", None)

        return cfg

    @torch.inference_mode()
    def _gen_one(self, pil_image: Image.Image, field: str, gen_cfg: Dict[str, Any]) -> str:
        pil_image = self._ensure_image_ok(pil_image)

        # Only matters if do_sample=True, but harmless otherwise
        self._set_random_seed()

        messages = self._build_messages(field)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=text, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_cfg = self._clean_gen_cfg(gen_cfg)

        out = self.model.generate(**inputs, **gen_cfg)
        gen_ids = out[0][input_len:]
        ans = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if "\n" in ans:
            ans = ans.splitlines()[0].strip()
        return ans

    def _field_num_samples(self, field: str) -> int:
        if field == "gender":
            return 1
        return self.default_num_samples

    def _field_gen_cfg(self, field: str) -> Dict[str, Any]:
        if field == "gender":
            # deterministic for gender (no sampling params here)
            return {
                "max_new_tokens": 8,
                "do_sample": False,
                "num_beams": 1,
            }
        return dict(self.gen_cfg)

    @torch.inference_mode()
    def extract_field_n(self, pil_image: Image.Image, field: str) -> List[str]:
        target = self._field_num_samples(field)
        gen_cfg = self._field_gen_cfg(field)

        results: List[str] = []
        seen = set()

        if field == "gender":
            s = self._gen_one(pil_image, field, gen_cfg)
            return [s]

        max_tries = max(self.max_tries_per_field, target)
        for _ in range(max_tries):
            if len(results) >= target:
                break
            s = self._gen_one(pil_image, field, gen_cfg)
            key = s.strip().lower()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            results.append(s)

        while len(results) < target:
            results.append(results[-1] if results else f"{field}:")
        return results