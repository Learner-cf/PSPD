import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm

from training.itc_rank_aux_loss import ITCWithRankAuxLoss, RankAuxConfig

FIELDS = ["gender", "upper_type", "upper_color", "lower_type", "lower_color"]


def resolve_clip_backend(backend: str, model_path: str) -> str:
    b = (backend or "auto").strip().lower()
    if b in {"hf", "open_clip"}:
        return b
    has_open_clip = os.path.exists(os.path.join(model_path, "open_clip_pytorch_model.bin"))
    if has_open_clip:
        return "open_clip"
    return "hf"


def load_clip_components(model_path: str, backend: str, device: str):
    if backend == "hf":
        processor = CLIPProcessor.from_pretrained(model_path)
        model = CLIPModel.from_pretrained(model_path).to(device)
        return model, processor, None, None

    if backend == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained=os.path.join(model_path, "open_clip_pytorch_model.bin"),
            device=device,
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        return model, None, preprocess, tokenizer

    raise ValueError(f"Unsupported clip backend: {backend}")


def encode_image_features(model: Any, backend: str, pixel_values: torch.Tensor) -> torch.Tensor:
    if backend == "hf":
        return model.get_image_features(pixel_values=pixel_values)
    return model.encode_image(pixel_values)


def encode_text_features(model: Any, backend: str, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
    if backend == "hf":
        return model.get_text_features(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
        )
    return model.encode_text(tokens["input_ids"])


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




def resolve_image_path(path: str, image_roots: List[str]) -> str:
    candidates: List[str] = []

    roots: List[str] = []
    for r in image_roots:
        if not r:
            continue
        rr = os.path.abspath(r)
        if rr not in roots:
            roots.append(rr)

    if path:
        candidates.append(path)

    for root in roots:
        if path and not os.path.isabs(path):
            candidates.append(os.path.join(root, path))
        if path:
            base = os.path.basename(path)
            candidates.append(os.path.join(root, base))
            candidates.append(os.path.join(root, "imgs", base))
            candidates.append(os.path.join(root, "train_query", base))
            candidates.append(os.path.join(root, "query", base))

    augmented: List[str] = []
    for c in candidates:
        augmented.append(c)
        if "/train_query/" in c:
            augmented.append(c.replace('/train_query/', '/imgs/'))
        if "/query/" in c:
            augmented.append(c.replace('/query/', '/imgs/'))
        if "/images/" in c:
            augmented.append(c.replace('/images/', '/imgs/'))

    seen = set()
    for c in augmented:
        if not c:
            continue
        cc = os.path.abspath(c)
        if cc in seen:
            continue
        seen.add(cc)
        if os.path.exists(cc):
            return cc

    return ""


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


class RetrievalEvalDataset(Dataset):
    def __init__(self, jsonl_path: str, image_roots: List[str]):
        self.rows = load_jsonl(jsonl_path)
        self.image_roots = image_roots
        self.image_paths: List[str] = []
        self.image_pids: List[str] = []
        self.texts: List[str] = []
        self.text_pids: List[str] = []
        self._build()

    @staticmethod
    def _row_pid(row: Dict) -> str:
        for key in ("pid", "person_id", "id"):
            if key in row:
                return str(row[key])
        raise KeyError("Each eval row must provide one of keys: pid/person_id/id")

    @staticmethod
    def _row_texts(row: Dict) -> List[str]:
        out: List[str] = []

        if isinstance(row.get("captions"), list):
            out.extend([str(x).strip() for x in row["captions"] if str(x).strip()])

        if isinstance(row.get("caption"), str) and row["caption"].strip():
            out.append(row["caption"].strip())

        cap_obj = row.get("caption_rewrite", {})
        if isinstance(cap_obj, dict):
            cap = str(cap_obj.get("caption", "")).strip()
            if cap:
                out.append(cap)

        if not out:
            out = ["a person"]
        return out

    def _build(self) -> None:
        img_idx_by_path: Dict[str, int] = {}
        for row in self.rows:
            pid = self._row_pid(row)
            raw_image_path = row["image_path"]
            image_path = resolve_image_path(raw_image_path, self.image_roots)
            if not image_path:
                continue

            if image_path not in img_idx_by_path:
                img_idx_by_path[image_path] = len(self.image_paths)
                self.image_paths.append(image_path)
                self.image_pids.append(pid)

            for txt in self._row_texts(row):
                self.texts.append(txt)
                self.text_pids.append(pid)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {"image": image, "pid": self.image_pids[idx], "index": idx}


def build_collate(
    backend: str,
    processor: CLIPProcessor = None,
    oc_preprocess=None,
    oc_tokenizer=None,
):
    def _tokenize(texts: List[str]) -> Dict[str, torch.Tensor]:
        if backend == "hf":
            return processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return {"input_ids": oc_tokenizer(texts)}

    def _collate(batch: List[Dict]) -> Dict:
        images = [x["image"] for x in batch]
        captions = [x["caption"] for x in batch]

        if backend == "hf":
            main_inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True)
        else:
            main_inputs = {
                "pixel_values": torch.stack([oc_preprocess(img.convert("RGB")) for img in images], dim=0),
                "input_ids": oc_tokenizer(captions),
            }

        def flatten_rank_texts(key: str) -> List[str]:
            flat: List[str] = []
            for sample in batch:
                flat.extend(sample[key])
            return flat

        rk1_texts = flatten_rank_texts("rk1_texts")
        rk2_texts = flatten_rank_texts("rk2_texts")
        rk3_texts = flatten_rank_texts("rk3_texts")

        tk1 = _tokenize(rk1_texts)
        tk2 = _tokenize(rk2_texts)
        tk3 = _tokenize(rk3_texts)

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


@torch.no_grad()
def evaluate_retrieval(
    model: Any,
    backend: str,
    processor: CLIPProcessor,
    oc_preprocess,
    oc_tokenizer,
    eval_dataset: RetrievalEvalDataset,
    device: str,
    batch_size: int,
    show_progress: bool,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    image_feats: List[torch.Tensor] = []
    image_steps = range(0, len(eval_dataset.image_paths), batch_size)
    if show_progress:
        image_steps = tqdm(image_steps, desc=f"eval-img e{epoch}", leave=False)
    for st in image_steps:
        images = [Image.open(p).convert("RGB") for p in eval_dataset.image_paths[st: st + batch_size]]
        if backend == "hf":
            img_inputs = processor(images=images, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(device)
        else:
            pixel_values = torch.stack([oc_preprocess(img.convert("RGB")) for img in images], dim=0).to(device)
        feat = encode_image_features(model, backend, pixel_values)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        image_feats.append(feat.cpu())

    text_feats: List[torch.Tensor] = []
    text_steps = range(0, len(eval_dataset.texts), batch_size)
    if show_progress:
        text_steps = tqdm(text_steps, desc=f"eval-txt e{epoch}", leave=False)
    for st in text_steps:
        texts = eval_dataset.texts[st: st + batch_size]
        if backend == "hf":
            txt_inputs = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        else:
            txt_inputs = {"input_ids": oc_tokenizer(texts)}
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
        feat = encode_text_features(model, backend, txt_inputs)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        text_feats.append(feat.cpu())

    image_feat = torch.cat(image_feats, dim=0)
    text_feat = torch.cat(text_feats, dim=0)

    sim = text_feat @ image_feat.t()  # [N_text, N_img]
    txt_pids = eval_dataset.text_pids
    img_pids = eval_dataset.image_pids

    n_query = sim.size(0)
    cmc_hits = torch.zeros(10, dtype=torch.float64)
    ap_sum = 0.0

    for i in range(n_query):
        scores = sim[i]
        order = torch.argsort(scores, descending=True)
        rel = torch.tensor([1 if img_pids[j] == txt_pids[i] else 0 for j in order.tolist()], dtype=torch.float32)

        pos = torch.nonzero(rel, as_tuple=False).squeeze(-1)
        if pos.numel() == 0:
            continue

        first_pos = int(pos[0].item())
        if first_pos < 1:
            cmc_hits[0] += 1
        if first_pos < 5:
            cmc_hits[4] += 1
        if first_pos < 10:
            cmc_hits[9] += 1

        cumsum_rel = torch.cumsum(rel, dim=0)
        ranks = torch.arange(1, rel.numel() + 1, dtype=torch.float32)
        precision = cumsum_rel / ranks
        ap = float((precision * rel).sum().item() / max(1.0, rel.sum().item()))
        ap_sum += ap

    rank1 = float(cmc_hits[0].item() / max(1, n_query))
    rank5 = float(cmc_hits[4].item() / max(1, n_query))
    rank10 = float(cmc_hits[9].item() / max(1, n_query))
    mAP = float(ap_sum / max(1, n_query))

    model.train()
    return {"rank1": rank1, "rank5": rank5, "rank10": rank10, "mAP": mAP}


def train_one_epoch(
    model: Any,
    backend: str,
    loss_fn: ITCWithRankAuxLoss,
    loader: DataLoader,
    optimizer: AdamW,
    device: str,
    grad_clip: float,
    show_progress: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    meter = {"loss_total": 0.0, "loss_itc": 0.0, "loss_rank": 0.0, "n": 0}

    batch_iter = loader
    if show_progress:
        batch_iter = tqdm(loader, desc=f"train e{epoch}", leave=False)

    for batch in batch_iter:
        main_inputs = to_device(batch["main_inputs"], device)
        rk1_tokens = to_device(batch["rk1_tokens"], device)
        rk2_tokens = to_device(batch["rk2_tokens"], device)
        rk3_tokens = to_device(batch["rk3_tokens"], device)

        score_rk1 = batch["score_rk1"].to(device)
        score_rk2 = batch["score_rk2"].to(device)
        score_rk3 = batch["score_rk3"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        image_feat = encode_image_features(model, backend, main_inputs["pixel_values"])
        caption_feat = encode_text_features(model, backend, main_inputs)

        B = batch["batch_size"]
        A = len(FIELDS)

        txt1 = encode_text_features(model, backend, rk1_tokens)
        txt2 = encode_text_features(model, backend, rk2_tokens)
        txt3 = encode_text_features(model, backend, rk3_tokens)

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
    ap.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "hf", "open_clip"])
    ap.add_argument("--save_dir", type=str, default="outputs/checkpoints_itc_rank")
    ap.add_argument("--test_jsonl", type=str, default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
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
    ap.add_argument("--use_itc_loss", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use_rank_loss", type=int, default=0, choices=[0, 1])
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--eval_image_root", type=str, default="")
    ap.add_argument("--show_progress", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    if not args.test_jsonl:
        raise ValueError(
            "--test_jsonl is required for per-epoch retrieval evaluation. "
            "If you only have raw dataset captions, first generate JSONL with "
            "`python data/generate_text_jsonl.py ...`."
        )
    if not os.path.exists(args.test_jsonl):
        raise FileNotFoundError(f"test_jsonl not found: {args.test_jsonl}")

    use_itc_loss = bool(args.use_itc_loss)
    use_rank_loss = bool(args.use_rank_loss)
    if (not use_itc_loss) and (not use_rank_loss):
        raise ValueError("At least one loss must be enabled. Use --use_itc_loss 1 or --use_rank_loss 1.")

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    eval_image_roots: List[str] = []
    if args.eval_image_root:
        eval_image_roots.append(args.eval_image_root)

    datasets_cfg = cfg.get("datasets", {})
    if isinstance(datasets_cfg, dict):
        for _, ds_cfg in datasets_cfg.items():
            if isinstance(ds_cfg, dict):
                root = ds_cfg.get("image_root", "")
                if root:
                    eval_image_roots.append(root)
                    parent = os.path.dirname(root.rstrip('/'))
                    if parent:
                        eval_image_roots.append(parent)

    # dedup roots while keeping order
    dedup_roots: List[str] = []
    seen_roots = set()
    for r in eval_image_roots:
        rr = os.path.abspath(r)
        if rr in seen_roots:
            continue
        seen_roots.add(rr)
        dedup_roots.append(rr)
    eval_image_roots = dedup_roots

    backend = resolve_clip_backend(args.clip_backend, args.clip_model_path)
    model, processor, oc_preprocess, oc_tokenizer = load_clip_components(
        model_path=args.clip_model_path,
        backend=backend,
        device=device,
    )
    print(f"[info] clip backend: {backend}")

    dataset = ITCRankDataset(args.train_jsonl)
    eval_dataset = RetrievalEvalDataset(args.test_jsonl, image_roots=eval_image_roots)
    if len(eval_dataset.image_paths) == 0:
        raise RuntimeError(
            "No valid evaluation images found. Please check --test_jsonl paths or set --eval_image_root. "
            "Current roots: " + str(eval_image_roots)
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=build_collate(backend=backend, processor=processor, oc_preprocess=oc_preprocess, oc_tokenizer=oc_tokenizer),
    )

    rank_cfg = RankAuxConfig(
        use_itc=use_itc_loss,
        use_rank=use_rank_loss,
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
        "clip_backend": backend,
        "test_jsonl": args.test_jsonl,
        "eval_image_root": args.eval_image_root,
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
            backend=backend,
            loss_fn=loss_fn,
            loader=loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            show_progress=bool(args.show_progress),
            epoch=epoch,
        )
        eval_stats = evaluate_retrieval(
            model=model,
            backend=backend,
            processor=processor,
            oc_preprocess=oc_preprocess,
            oc_tokenizer=oc_tokenizer,
            eval_dataset=eval_dataset,
            device=device,
            batch_size=args.eval_batch_size,
            show_progress=bool(args.show_progress),
            epoch=epoch,
        )
        stats.update(eval_stats)

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
            f"loss_rank={stats['loss_rank']:.6f} "
            f"rank1={stats['rank1']:.4f} "
            f"rank5={stats['rank5']:.4f} "
            f"rank10={stats['rank10']:.4f} "
            f"mAP={stats['mAP']:.4f}"
        )


if __name__ == "__main__":
    main()
