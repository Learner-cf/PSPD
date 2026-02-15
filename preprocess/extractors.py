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
        max_tries_per_field=int(cfg.get("extractor", {}).get("max_tries_per_field", 3)),
        enforce_exact_n_samples=bool(cfg.get("extractor", {}).get("enforce_exact_n_samples", True)),
        sample_batch_size=int(cfg.get("extractor", {}).get("sample_batch_size", cfg["extractor"].get("num_samples", 3))),
    )
