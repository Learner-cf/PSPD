import argparse
import os
from typing import Dict, List

import torch

from training.run_train_itc_rank import (
    RetrievalEvalDataset,
    evaluate_retrieval,
    load_cfg,
    load_clip_components,
    resolve_clip_backend,
)


def collect_eval_image_roots(cfg: Dict, eval_image_root: str) -> List[str]:
    eval_image_roots: List[str] = []
    if eval_image_root:
        eval_image_roots.append(eval_image_root)

    datasets_cfg = cfg.get("datasets", {})
    if isinstance(datasets_cfg, dict):
        for _, ds_cfg in datasets_cfg.items():
            if isinstance(ds_cfg, dict):
                root = ds_cfg.get("image_root", "")
                if root:
                    eval_image_roots.append(root)
                    parent = os.path.dirname(root.rstrip("/"))
                    if parent:
                        eval_image_roots.append(parent)

    dedup_roots: List[str] = []
    seen_roots = set()
    for r in eval_image_roots:
        rr = os.path.abspath(r)
        if rr in seen_roots:
            continue
        seen_roots.add(rr)
        dedup_roots.append(rr)
    return dedup_roots


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CLIP checkpoint without training.")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--clip_model_path", type=str, default="/root/autodl-tmp/PSPD/hf_models/openclip")
    ap.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "hf", "open_clip"])
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--eval_image_root", type=str, default="/root/autodl-tmp/PSPD/dataset/CUHK-PEDES ")
    ap.add_argument("--show_progress", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    if not os.path.exists(args.test_jsonl):
        raise FileNotFoundError(f"test_jsonl not found: {args.test_jsonl}")

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    roots = collect_eval_image_roots(cfg=cfg, eval_image_root=args.eval_image_root)

    backend = resolve_clip_backend(args.clip_backend, args.clip_model_path)
    model, processor, oc_preprocess, oc_tokenizer = load_clip_components(
        model_path=args.clip_model_path,
        backend=backend,
        device=device,
    )
    print(f"[info] clip backend: {backend}")

    eval_dataset = RetrievalEvalDataset(args.test_jsonl, image_roots=roots)
    if len(eval_dataset.image_paths) == 0:
        raise RuntimeError(
            "No valid evaluation images found. Please check --test_jsonl paths or set --eval_image_root. "
            "Current roots: " + str(roots)
        )

    stats = evaluate_retrieval(
        model=model,
        backend=backend,
        processor=processor,
        oc_preprocess=oc_preprocess,
        oc_tokenizer=oc_tokenizer,
        eval_dataset=eval_dataset,
        device=device,
        batch_size=args.eval_batch_size,
        show_progress=bool(args.show_progress),
        epoch=0,
    )

    print(
        f"[eval] rank1={stats['rank1']:.4f} "
        f"rank5={stats['rank5']:.4f} "
        f"rank10={stats['rank10']:.4f} "
        f"mAP={stats['mAP']:.4f}"
    )


if __name__ == "__main__":
    main()
