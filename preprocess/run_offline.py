import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
import yaml

from data_mvp.datasets import load_cuhk, load_icfg, load_rstp, ImageOnlyDataset
from preprocess.schema import build_schema
from preprocess.extractors import build_extractor
from preprocess.slots_normalize import normalize_slots
from preprocess.caption_compose import compose_caption
from preprocess.clip_teacher_transformers import TransformersCLIPTeacher
from preprocess.io_jsonl import JsonlWriter, load_processed_set
from preprocess.rank_triplet import rank_triplet
from preprocess.calibration import calibrate_threshold
from preprocess.merge_clean import merge_slots
from preprocess.slot_parse_from_text import parse_slots_from_text

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_samples(cfg: Dict, dataset: str):
    dcfg = cfg["datasets"][dataset]
    image_root = dcfg["image_root"]
    ann_file = dcfg["ann_file"]
    if dataset == "cuhk":
        return load_cuhk(ann_file, image_root)
    if dataset == "icfg":
        return load_icfg(ann_file, image_root)
    if dataset == "rstp":
        return load_rstp(ann_file, image_root)
    raise ValueError(dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["cuhk","icfg","rstp"])
    ap.add_argument("--split", type=str, required=True, choices=["train","val","test"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_images", type=int, default=-1, help="for quick test, e.g., 50")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = cfg.get("device", "cuda")

    schema = build_schema(cfg)
    extractor = build_extractor(cfg)

    teacher_cfg = cfg["teacher"]
    backend = teacher_cfg.get("backend", "transformers_clip")
    if backend != "transformers_clip":
        raise ValueError(f"Only transformers_clip backend supported offline now, got {backend}")

    teacher = TransformersCLIPTeacher(
        model_dir=teacher_cfg["model_dir"],
        device=device,
    )
    teacher.build_field_candidates(schema)

    samples = build_samples(cfg, args.dataset)
    ds = ImageOnlyDataset(samples, split=args.split)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_{args.split}_slots.jsonl")

    seen = set()
    if args.resume:
        seen = load_processed_set(out_path, key="image_path")
        print(f"[resume] loaded {len(seen)} processed samples from {out_path}")

    # ---- calibration ----
    calib_cfg = cfg.get("calibration", {"enabled": False})
    delta_field = {f: 1.0 for f in schema.field_candidates().keys()}

    if calib_cfg.get("enabled", False):
        ncal = int(calib_cfg.get("num_images", 1500))
        target = float(calib_cfg.get("target_pass_rate", 0.60))
        tau_teacher = float(teacher_cfg.get("tau_teacher", 0.01))

        margins_by_field = {f: [] for f in schema.field_candidates().keys()}
        count = 0

        from preprocess.merge_clean import top1_top2_margin

        for i in tqdm(range(len(ds)), desc="calibration"):
            img, path, _pid = ds[i]
            if path in seen:
                continue
            img_feat = teacher.encode_image(img)
            _scores, probs = teacher.score_field_candidates(img_feat, tau=tau_teacher)
            for f, p in probs.items():
                margins_by_field[f].append(top1_top2_margin(p))
            count += 1
            if count >= ncal:
                break

        for f, margins in margins_by_field.items():
            delta_field[f] = calibrate_threshold(margins, target_pass_rate=target)

        print("[calibration] delta_field:")
        for k, v in delta_field.items():
            print(f"  {k}: {v:.4f}")

    conf_cover = {
        "upper_color": 0.70,
        "lower_color": 0.70,
        "upper_type": 0.70,
        "lower_type": 0.70,
    }

    tau_teacher = float(teacher_cfg.get("tau_teacher", 0.01))
    rank_cfg = cfg.get("ranking", {"enabled": True})
    rank_enabled_flag = bool(rank_cfg.get("enabled", True))
    delta_global = float(rank_cfg.get("delta_global", 0.03))

    writer = JsonlWriter(out_path)
    try:
        total = len(ds)
        if args.max_images and args.max_images > 0:
            total = min(total, args.max_images)

        for i in tqdm(range(total), desc=f"preprocess {args.dataset}:{args.split}"):
            img, path, pid = ds[i]
            if path in seen:
                continue
                
            # 1) extract 3 descriptions (not JSON), then parse to slot dicts
            try:
                desc_raw = extractor.extract_n(img)  # list of {"desc": "..."}
            except Exception as e:
                print(f"[extract_fail] {path}: {repr(e)}")
                continue

            desc_list = [d.get("desc", "") if isinstance(d, dict) else str(d) for d in desc_raw]
            slots_raw = [parse_slots_from_text(desc, schema) for desc in desc_list]

            # 2) normalize
            slots_norm = [normalize_slots(s, schema) for s in slots_raw]

            # 3) teacher field probs
            img_feat = teacher.encode_image(img)
            field_scores, field_probs = teacher.score_field_candidates(img_feat, tau=tau_teacher)

            # 4) merge+clean
            slots_clean = merge_slots(
                slots_norm=slots_norm,
                teacher_field_probs=field_probs,
                schema=schema,
                delta_field=delta_field,
                conf_cover=conf_cover,
            )

            # 5) compose captions
            caption_candidates = [compose_caption(s) for s in slots_norm]
            caption_clean = compose_caption(slots_clean)

            # 6) caption-level triplet ranking (optional)
            teacher_caption_scores = teacher.score_caption_triplet(img_feat, caption_candidates)
            order, enabled, weights = rank_triplet(teacher_caption_scores, delta_global=delta_global)
            if not rank_enabled_flag:
                enabled = False

            record = {
                "image_path": path,
                "pid": int(pid),

                "slots_raw": slots_raw,
                "slots_norm": slots_norm,

                "teacher_field_scores": field_scores,
                "teacher_field_probs": field_probs,
                "desc_candidates": desc_list,
                "slots_clean": slots_clean,

                "caption_candidates": caption_candidates,
                "teacher_caption_scores": teacher_caption_scores,
                "rank_order": order,
                "rank_enabled": bool(enabled),
                "rank_weights": weights,

                "caption_clean": caption_clean,
            }
            writer.write(record)

    finally:
        writer.close()
        print(f"saved: {out_path}")

if __name__ == "__main__":
    main()