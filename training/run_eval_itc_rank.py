import argparse
from typing import Dict, List

import torch
import yaml

from data_mvp.datasets import Sample, load_cuhk, load_icfg, load_rstp
from models.dino_backbone import DinoV2Backbone


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
    else:
        samples = load_rstp(ann_file, image_root)
    return [s for s in samples if s.split == split]


@torch.no_grad()
def evaluate(backbone: DinoV2Backbone, samples: List[Sample], device: str, batch_size: int = 128) -> Dict[str, float]:
    from torch.utils.data import DataLoader
    from PIL import Image

    def collate_fn(batch_samples: List[Sample]):
        images = [Image.open(s.image_path).convert("RGB") for s in batch_samples]
        x = backbone.preprocess(images=images, return_tensors="pt")["pixel_values"]
        pids = torch.tensor([s.pid for s in batch_samples], dtype=torch.long)
        return x, pids

    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    feats, pids = [], []
    backbone.eval()
    for x, y in loader:
        x = x.to(device)
        feats.append(backbone(x).cpu())
        pids.append(y)

    feat = torch.cat(feats, dim=0)
    pid = torch.cat(pids, dim=0)
    sim = feat @ feat.t()
    sim.fill_diagonal_(-1e9)
    idx = sim.argmax(dim=1)
    pred = pid[idx]
    rank1 = (pred == pid).float().mean().item()
    return {"rank1": rank1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--dataset", type=str, required=True, choices=["cuhk", "icfg", "rstp"])
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--dino_model_path", type=str, default="/home/u2024218474/jupyterlab/PSPD/hf_models/AI-ModelScope/dinov2-giant")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    samples = build_samples(cfg, args.dataset, args.split)
    backbone = DinoV2Backbone(model_path=args.dino_model_path).to(device)
    stats = evaluate(backbone, samples, device)
    print(f"[eval] rank1={stats['rank1']:.4f}")


if __name__ == "__main__":
    main()
