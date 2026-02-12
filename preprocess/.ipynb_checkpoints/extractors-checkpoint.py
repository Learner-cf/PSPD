import random
from typing import Dict, List, Any

import torch
from PIL import Image
from preprocess.prompts import system_prompt

class DummyExtractor:
    def __init__(self, num_samples: int = 3, seed: int = 42):
        self.num_samples = num_samples
        random.seed(seed)

    def extract_n(self, pil_image: Image.Image) -> List[Dict[str, str]]:
        outs = []
        for _ in range(self.num_samples):
            outs.append({"desc": "a person wearing black jacket and black pants"})
        return outs

class InstructBLIPHFExtractor:
    """
    InstructBLIP-FLAN-T5-XL often ignores strict JSON instructions.
    So we let it generate a short description text, then parse slots from text.
    """
    def __init__(self, model_name: str, num_samples: int, gen_cfg: Dict[str, Any], device: str = "cuda"):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

        self.num_samples = num_samples
        self.device = device
        self.processor = InstructBlipProcessor.from_pretrained(model_name)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

        self.gen_cfg = dict(gen_cfg)
        self.gen_cfg["max_new_tokens"] = int(self.gen_cfg.get("max_new_tokens", 64))
        # keep text short to improve parsing stability
        self.sys = system_prompt().strip()
        self.prompt = (
            self.sys
            + "\nDescribe ONLY the clothing of the main person in one short sentence. "
              "Do not mention background, location, brands, or actions."
        )

    @torch.inference_mode()
    def extract_once(self, pil_image: Image.Image) -> Dict[str, str]:
        inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, **self.gen_cfg)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return {"desc": (text or "").strip()}

    def extract_n(self, pil_image: Image.Image) -> List[Dict[str, str]]:
        return [self.extract_once(pil_image) for _ in range(self.num_samples)]

def build_extractor(cfg):
    typ = cfg["extractor"]["type"]
    n = int(cfg["extractor"]["num_samples"])
    if typ == "dummy":
        return DummyExtractor(num_samples=n, seed=int(cfg.get("seed", 42)))
    if typ == "instructblip_hf":
        gen_cfg = dict(cfg["extractor"]["generation"])
        return InstructBLIPHFExtractor(
            model_name=cfg["extractor"]["hf_model_name"],
            num_samples=n,
            gen_cfg=gen_cfg,
            device=cfg.get("device", "cuda"),
        )
    raise ValueError(f"Unknown extractor.type={typ}")