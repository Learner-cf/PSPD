"""DINOv2 staged pipeline with independent stage execution and quantitative metrics."""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
import yaml

from data_mvp.datasets import Sample, load_cuhk, load_icfg, load_rstp
from models.dino_backbone import DinoV2Backbone
from trainer.unsupervised_trainer import TrainerConfig, UnsupervisedReIDTrainer
from utils.embedding_diagnostics import evaluate_embedding_quality
from utils.pipeline_score import (
    build_stage_report,
    compute_stage1_score,
    compute_stage2_score,
    compute_stage3_score,
)


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_samples(cfg: Dict, dataset: str, split: str) -> List[Sample]:
    dcfg = cfg["datasets"][dataset]
    image_root = dcfg["image_root"]
    ann_file = dcfg["ann_file"]

    if dataset == "cuhk":
        samples = load_cuhk(ann_file, image_root)
    elif dataset == "icfg":
        samples = load_icfg(ann_file, image_root)
    elif dataset == "rstp":
        samples = load_rstp(ann_file, image_root)
    else:
        raise ValueError(dataset)

    return [s for s in samples if s.split == split]


def build_trainer(args, cfg, device: str, samples: List[Sample]) -> UnsupervisedReIDTrainer:
    backbone = DinoV2Backbone(model_path=args.dino_model_path)
    tcfg = TrainerConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        recluster_every=5,
        lr=3e-4,
        weight_decay=1e-4,
        triplet_margin=0.3,
        itc_temperature=0.05,
        itc_weight=0.2,
    )
    return UnsupervisedReIDTrainer(backbone=backbone, samples=samples, device=device, cfg=tcfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="PSPD DINO staged pipeline")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--dataset", type=str, required=True, choices=["cuhk", "icfg", "rstp"])
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--artifacts_dir", type=str, default="outputs/pipeline")
    ap.add_argument("--dino_model_path", type=str, default="/home/u2024218474/jupyterlab/PSPD/hf_models/AI-ModelScope/dinov2-giant")
    ap.add_argument("--stage2_ckpt", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--stage2_epochs", type=int, default=40)
    ap.add_argument("--stage3_epochs", type=int, default=20)
    args = ap.parse_args()

    # Deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.artifacts_dir, exist_ok=True)
    features_path = os.path.join(args.artifacts_dir, f"{args.dataset}_{args.split}_stage1_features.pt")
    labels_path = os.path.join(args.artifacts_dir, f"{args.dataset}_{args.split}_stage2_pseudo_labels.npy")
    stage2_backbone_path = args.stage2_ckpt or os.path.join(args.artifacts_dir, f"{args.dataset}_{args.split}_stage2_backbone.pt")
    stage3_backbone_path = os.path.join(args.artifacts_dir, f"{args.dataset}_{args.split}_stage3_backbone.pt")

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    samples = build_samples(cfg, args.dataset, args.split)
    if len(samples) == 0:
        raise RuntimeError(f"No samples found for dataset={args.dataset} split={args.split}")

    trainer = build_trainer(args, cfg, device, samples)

    stage1_metrics: Dict[str, float] = {}
    stage2_metrics: Dict[str, float] = {}
    stage3_metrics: Dict[str, float] = {}

    if args.stage in {"1", "all"}:
        print("[Run] Stage 1: feature extraction")
        with torch.no_grad():
            features, whitened_raw = trainer.extract_features(return_raw=True)
        torch.save(features, features_path)
        print(f"[Stage1] saved features: {features_path}")
        print("Stage1 diagnostics are computed on WHITENED features.")
        stage1_metrics = evaluate_embedding_quality(whitened_raw)

    if args.stage in {"2", "all"}:
        print("[Run] Stage 2: DBSCAN + CE/Triplet training")
        if os.path.exists(features_path):
            features = torch.load(features_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Stage1 output not found: {features_path}. Please run --stage 1 first.")

        if not stage1_metrics:
            print("[Stage2] Reusing Stage1 diagnostics from current run if available.")

        initial_pseudo_labels = trainer.generate_pseudo_labels(features)
        stage2_out = trainer.train_stage2(initial_pseudo_labels)
        pseudo_labels = stage2_out["pseudo_labels"]
        stage2_metrics = stage2_out.get("last_cluster_metrics", {})

        np.save(labels_path, pseudo_labels)
        torch.save(trainer.backbone.state_dict(), stage2_backbone_path)
        print(f"[Stage2] saved pseudo labels: {labels_path}")
        print(f"[Stage2] saved backbone ckpt: {stage2_backbone_path}")

    if args.stage in {"3", "all"}:
        print("[Run] Stage 3: ITC fine-tuning")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Stage2 pseudo labels not found: {labels_path}. Please run --stage 2 first.")
        if not os.path.exists(stage2_backbone_path):
            raise FileNotFoundError(f"Stage2 backbone ckpt not found: {stage2_backbone_path}. Please run --stage 2 first.")

        trainer.backbone.load_state_dict(torch.load(stage2_backbone_path, map_location=device), strict=True)
        pseudo_labels = np.load(labels_path)
        stage3_metrics = trainer.train_stage3_itc(pseudo_labels)
        torch.save(trainer.backbone.state_dict(), stage3_backbone_path)
        print(f"[Stage3] saved backbone ckpt: {stage3_backbone_path}")

    # Global dashboard output
    s1 = compute_stage1_score(stage1_metrics) if stage1_metrics else 0.0
    s2 = compute_stage2_score(stage2_metrics) if stage2_metrics else 0.0
    s3 = compute_stage3_score(stage3_metrics) if stage3_metrics else 0.0
    _, report = build_stage_report(s1, s2, s3)
    print(report)


if __name__ == "__main__":
    main()
