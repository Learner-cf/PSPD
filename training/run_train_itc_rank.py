import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from training.itc_rank_aux_loss import ITCWithRankAuxLoss, RankAuxConfig

FIELDS = ["gender", "upper_type", "upper_color", "lower_type", "lower_color"]


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config must be dict, got: {type(cfg)}")
    return cfg


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_value(v: str) -> str:
    return (v or "").strip().lower()


def field_prompt(field: str, value: str) -> str:
    if field == "gender":
        return f"a photo of a {value} person"
    if field == "upper_type":
        return f"a photo of a person wearing a {value} on the upper body"
    if field == "upper_color":
        return f"a photo of a person wearing {value} upper clothing"
    if field == "lower_type":
        return f"a photo of a person wearing {value} on the lower body"
    if field == "lower_color":
        return f"a photo of a person wearing {value} lower clothing"
    return f"a photo of a person with {value}"


class ITCRankDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = load_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _get_rank_item(row: Dict, field: str, rank_idx: int) -> Tuple[str, float]:
        rr = row.get("clip_rerank", {})
        fobj = rr.get(field, {})
        final_rank = fobj.get("final_rank", [])
        if rank_idx < len(final_rank) and isinstance(final_rank[rank_idx], dict):
            value = normalize_value(str(final_rank[rank_idx].get("value", "unknown")))
            score = float(final_rank[rank_idx].get("score", 0.0))
            return value if value else "unknown", score
        return "unknown", 0.0

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        caption_obj = row.get("caption_rewrite", {})
        caption = str(caption_obj.get("caption", "")).strip()
        if not caption:
            caption = "a person"

        rk1_vals: List[str] = []
        rk2_vals: List[str] = []
        rk3_vals: List[str] = []
        s1: List[float] = []
        s2: List[float] = []
        s3: List[float] = []
        valid: List[float] = []

        for field in FIELDS:
            v1, sc1 = self._get_rank_item(row, field, 0)
            v2, sc2 = self._get_rank_item(row, field, 1)
            v3, sc3 = self._get_rank_item(row, field, 2)

            rk1_vals.append(field_prompt(field, v1))
            rk2_vals.append(field_prompt(field, v2))
            rk3_vals.append(field_prompt(field, v3))

            s1.append(sc1)
            s2.append(sc2)
            s3.append(sc3)

            is_valid = float(v1 != "unknown" and v2 != "unknown" and v3 != "unknown")
            valid.append(is_valid)

        return {
            "image": image,
            "caption": caption,
            "rk1_texts": rk1_vals,
            "rk2_texts": rk2_vals,
            "rk3_texts": rk3_vals,
            "score_rk1": s1,
            "score_rk2": s2,
            "score_rk3": s3,
            "valid_mask": valid,
        }


def build_collate(processor: CLIPProcessor):
    def _collate(batch: List[Dict]) -> Dict:
        images = [x["image"] for x in batch]
        captions = [x["caption"] for x in batch]

        main_inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True)

        def flatten_rank_texts(key: str) -> List[str]:
            flat: List[str] = []
            for sample in batch:
                flat.extend(sample[key])
            return flat

        rk1_texts = flatten_rank_texts("rk1_texts")
        rk2_texts = flatten_rank_texts("rk2_texts")
        rk3_texts = flatten_rank_texts("rk3_texts")

        tk1 = processor.tokenizer(rk1_texts, return_tensors="pt", padding=True, truncation=True)
        tk2 = processor.tokenizer(rk2_texts, return_tensors="pt", padding=True, truncation=True)
        tk3 = processor.tokenizer(rk3_texts, return_tensors="pt", padding=True, truncation=True)

        score_rk1 = torch.tensor([x["score_rk1"] for x in batch], dtype=torch.float32)
        score_rk2 = torch.tensor([x["score_rk2"] for x in batch], dtype=torch.float32)
        score_rk3 = torch.tensor([x["score_rk3"] for x in batch], dtype=torch.float32)
        valid_mask = torch.tensor([x["valid_mask"] for x in batch], dtype=torch.float32)

        return {
            "main_inputs": main_inputs,
            "rk1_tokens": tk1,
            "rk2_tokens": tk2,
            "rk3_tokens": tk3,
            "score_rk1": score_rk1,
            "score_rk2": score_rk2,
            "score_rk3": score_rk3,
            "valid_mask": valid_mask,
            "batch_size": len(batch),
        }

    return _collate


def to_device(d: Dict, device: str) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: CLIPModel,
    loss_fn: ITCWithRankAuxLoss,
    loader: DataLoader,
    optimizer: AdamW,
    device: str,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()

    meter = {"loss_total": 0.0, "loss_itc": 0.0, "loss_rank": 0.0, "n": 0}

    for batch in loader:
        main_inputs = to_device(batch["main_inputs"], device)
        rk1_tokens = to_device(batch["rk1_tokens"], device)
        rk2_tokens = to_device(batch["rk2_tokens"], device)
        rk3_tokens = to_device(batch["rk3_tokens"], device)

        score_rk1 = batch["score_rk1"].to(device)
        score_rk2 = batch["score_rk2"].to(device)
        score_rk3 = batch["score_rk3"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        image_feat = model.get_image_features(pixel_values=main_inputs["pixel_values"])
        caption_feat = model.get_text_features(
            input_ids=main_inputs["input_ids"],
            attention_mask=main_inputs.get("attention_mask"),
        )

        B = batch["batch_size"]
        A = len(FIELDS)

        txt1 = model.get_text_features(input_ids=rk1_tokens["input_ids"], attention_mask=rk1_tokens.get("attention_mask"))
        txt2 = model.get_text_features(input_ids=rk2_tokens["input_ids"], attention_mask=rk2_tokens.get("attention_mask"))
        txt3 = model.get_text_features(input_ids=rk3_tokens["input_ids"], attention_mask=rk3_tokens.get("attention_mask"))

        img_norm = image_feat / (image_feat.norm(dim=-1, keepdim=True) + 1e-6)
        txt1 = txt1.view(B, A, -1)
        txt2 = txt2.view(B, A, -1)
        txt3 = txt3.view(B, A, -1)

        txt1 = txt1 / (txt1.norm(dim=-1, keepdim=True) + 1e-6)
        txt2 = txt2 / (txt2.norm(dim=-1, keepdim=True) + 1e-6)
        txt3 = txt3 / (txt3.norm(dim=-1, keepdim=True) + 1e-6)

        sim_rk1 = (img_norm.unsqueeze(1) * txt1).sum(dim=-1)
        sim_rk2 = (img_norm.unsqueeze(1) * txt2).sum(dim=-1)
        sim_rk3 = (img_norm.unsqueeze(1) * txt3).sum(dim=-1)

        logit_scale = model.logit_scale.exp()

        total_loss, stats = loss_fn(
            image_feat=image_feat,
            caption_feat=caption_feat,
            logit_scale=logit_scale,
            sim_rk1=sim_rk1,
            sim_rk2=sim_rk2,
            sim_rk3=sim_rk3,
            rerank_score_rk1=score_rk1,
            rerank_score_rk2=score_rk2,
            rerank_score_rk3=score_rk3,
            valid_mask=valid_mask,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        meter["loss_total"] += float(stats["loss_total"].item()) * B
        meter["loss_itc"] += float(stats["loss_itc"].item()) * B
        meter["loss_rank"] += float(stats["loss_rank"].item()) * B
        meter["n"] += B

    n = max(1, meter["n"])
    return {
        "loss_total": meter["loss_total"] / n,
        "loss_itc": meter["loss_itc"] / n,
        "loss_rank": meter["loss_rank"] / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--train_jsonl", type=str, default="outputs/cuhk_train_qwen_caption_v1.jsonl")
    ap.add_argument("--clip_model_path", type=str, default="/root/autodl-tmp/PSPD/hf_models/openclip")
    ap.add_argument("--save_dir", type=str, default="outputs/checkpoints_itc_rank")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_rank", type=float, default=0.2)
    ap.add_argument("--margin12", type=float, default=0.15)
    ap.add_argument("--margin13", type=float, default=0.20)
    ap.add_argument("--margin23", type=float, default=0.05)
    ap.add_argument("--conf_scale", type=float, default=8.0)
    ap.add_argument("--conf_bias", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    model = CLIPModel.from_pretrained(args.clip_model_path).to(device)

    dataset = ITCRankDataset(args.train_jsonl)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=build_collate(processor),
    )

    rank_cfg = RankAuxConfig(
        lambda_rank=args.lambda_rank,
        base_margin_12=args.margin12,
        base_margin_13=args.margin13,
        base_margin_23=args.margin23,
        conf_scale=args.conf_scale,
        conf_bias=args.conf_bias,
    )
    loss_fn = ITCWithRankAuxLoss(rank_cfg)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    meta = {
        "train_jsonl": args.train_jsonl,
        "clip_model_path": args.clip_model_path,
        "rank_cfg": asdict(rank_cfg),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
    }
    with open(Path(args.save_dir) / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": stats,
            "rank_cfg": asdict(rank_cfg),
        }
        torch.save(ckpt, Path(args.save_dir) / f"epoch_{epoch:03d}.pt")

        if stats["loss_total"] < best:
            best = stats["loss_total"]
            torch.save(ckpt, Path(args.save_dir) / "best.pt")

        print(
            f"[epoch {epoch:03d}] "
            f"loss_total={stats['loss_total']:.6f} "
            f"loss_itc={stats['loss_itc']:.6f} "
            f"loss_rank={stats['loss_rank']:.6f}"
        )


if __name__ == "__main__":
    main()
