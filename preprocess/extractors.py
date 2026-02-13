from typing import Dict
from preprocess.qwen2_vl_extractor import Qwen2VLHFExtractor

def build_extractor(cfg: Dict):
    typ = cfg["extractor"]["type"]
    if typ != "qwen2_vl_hf":
        raise ValueError(f"Only qwen2_vl_hf supported in this simplified build_extractor, got {typ}")
    return Qwen2VLHFExtractor(
        model_name=cfg["extractor"]["hf_model_name"],
        num_samples=int(cfg["extractor"]["num_samples"]),
        gen_cfg=dict(cfg["extractor"]["generation"]),
        device=cfg.get("device", "cuda"),
        seed=int(cfg.get("seed", 42)),
        max_words=int(cfg.get("extractor", {}).get("max_words", 3)),
    )