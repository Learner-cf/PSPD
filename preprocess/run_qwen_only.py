import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
import yaml

from data_mvp.datasets import load_cuhk, load_icfg, load_rstp, ImageOnlyDataset
from preprocess.extractors import build_extractor
from preprocess.field_prompts import FIELDS
from preprocess.field_line import parse_field_line
from preprocess.io_jsonl import JsonlWriter, load_processed_set


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
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
    ap.add_argument("--dataset", type=str, required=True, choices=["cuhk", "icfg", "rstp"])
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_images", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    extractor = build_extractor(cfg)

    samples = build_samples(cfg, args.dataset)
    ds = ImageOnlyDataset(samples, split=args.split)

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_{args.split}_qwen_fields3_raw.jsonl")

    seen = set()
    if args.resume:
        seen = load_processed_set(out_path, key="image_path")
        print(f"[resume] loaded {len(seen)} processed samples from {out_path}")

    writer = JsonlWriter(out_path)
    try:
        total = len(ds)
        if args.max_images and args.max_images > 0:
            total = min(total, args.max_images)

        for i in tqdm(range(total), desc=f"qwen-only {args.dataset}:{args.split}"):
            img, path, pid = ds[i]
            if path in seen:
                continue

            fields_out = {}
            for field in FIELDS:
                lines = extractor.extract_field_n(img, field)
                parsed_values = []
                for ln in lines:
                    _, v = parse_field_line(ln, expected_field=field)
                    parsed_values.append(v)

                fields_out[field] = {
                    "raw_lines": lines,
                    "parsed_values": parsed_values,
                }

            writer.write({
                "image_path": path,
                "pid": int(pid),
                "fields": fields_out,
            })

    finally:
        writer.close()
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()