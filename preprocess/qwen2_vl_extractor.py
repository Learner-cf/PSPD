import random
from typing import Any, Dict, List

import torch
from PIL import Image

from preprocess.field_prompts import blip_prompt


class Qwen2VLHFExtractor:
    """
    Qwen2-VL extractor:
      - prompt/采样都限制field: value格式，无unknown
      - 支持严格采样N次（每次1个结果）
      - 支持把采样请求按batch送入模型以提高吞吐
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
        max_tries_per_field: int = 3,
        min_image_side: int = 64,
        enforce_exact_n_samples: bool = True,
        sample_batch_size: int = 8,
    ):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.num_samples = int(num_samples)
        self.device = device
        self.max_words = int(max_words)

        self.max_tries_per_field = int(max_tries_per_field)
        self.min_image_side = int(min_image_side)
        self.enforce_exact_n_samples = bool(enforce_exact_n_samples)
        self.sample_batch_size = max(1, int(sample_batch_size))

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # generation config (偏多样化)
        self.gen_cfg = dict(gen_cfg)
        self.gen_cfg.setdefault("max_new_tokens", 12)
        self.gen_cfg.setdefault("do_sample", True)
        self.gen_cfg.setdefault("temperature", 1.1)
        self.gen_cfg.setdefault("top_p", 0.95)
        self.gen_cfg.setdefault("top_k", 50)
        self.gen_cfg.setdefault("repetition_penalty", 1.0)
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

    def _postprocess_answer(self, ans: str, field: str) -> str:
        ans = (ans or "").strip()
        if "\n" in ans:
            ans = ans.splitlines()[0].strip()
        if not ans.lower().startswith(f"{field.lower()}:"):
            ans = f"{field}: {ans}"

        ban = {"unknown", "unsure", "uncertain", "n/a", "none", "null"}
        v = ans.split(":")[-1].strip().lower()
        if v in ban:
            return ""
        return ans

    @torch.inference_mode()
    def _gen_batch(self, pil_image: Image.Image, field: str, n: int) -> List[str]:
        pil_image = self._ensure_image_ok(pil_image)
        self._set_random_seed()

        messages = self._build_messages(field)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        texts = [text] * n
        images = [pil_image] * n
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_lens = inputs["attention_mask"].sum(dim=-1).tolist()
        out = self.model.generate(**inputs, **self.gen_cfg)  # (n, seq_len)

        outputs: List[str] = []
        for i in range(out.shape[0]):
            start = int(input_lens[i])
            gen_ids = out[i][start:]
            ans = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            outputs.append(self._postprocess_answer(ans, field))
        return outputs

    @torch.inference_mode()
    def _gen_one(self, pil_image: Image.Image, field: str) -> str:
        outs = self._gen_batch(pil_image, field, 1)
        return outs[0] if outs else ""

    @torch.inference_mode()
    def extract_field_n(self, pil_image: Image.Image, field: str) -> List[str]:
        """
        生成 num_samples 个候选。
        默认严格执行“采样N次，每次只生成1次，不额外重试”，
        但会把这些单次采样按sample_batch_size打包，以提升吞吐。
        """
        results: List[str] = []

        if self.enforce_exact_n_samples:
            remain = self.num_samples
            while remain > 0:
                cur = min(remain, self.sample_batch_size)
                batch_out = self._gen_batch(pil_image, field, cur)
                for s in batch_out:
                    results.append(s if s else f"{field}: ")
                remain -= cur
            return results

        # 兼容旧逻辑：尝试去重，允许额外重试。
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

        while len(results) < self.num_samples:
            results.append(results[-1] if results else f"{field}: ")
        return results
