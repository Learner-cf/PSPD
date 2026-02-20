import argparse
import json
import os
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm

from training.itc_rank_aux_loss import ITCWithRankAuxLoss, RankAuxConfig

# =============== 添加日志初始化函数 ===============
def setup_logger(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(save_dir, "train.log"), "a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# =============== 原有代码从这继续 ===============
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

def set_text_trainable(
    model: Any,
    backend: str,
    train_encoder: bool,
    train_proj: bool,
) -> None:
    if backend == "hf":
        if hasattr(model, "text_model"):
            for p in model.text_model.parameters():
                p.requires_grad = train_encoder
        if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
            model.text_projection.requires_grad = train_proj
        return

    if backend == "open_clip":
        for attr in ("transformer", "token_embedding", "ln_final"):
            module = getattr(model, attr, None)
            if module is not None and hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad = train_encoder

        if hasattr(model, "positional_embedding") and isinstance(model.positional_embedding, torch.nn.Parameter):
            model.positional_embedding.requires_grad = train_encoder
        if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
            model.text_projection.requires_grad = train_proj
        return

    raise ValueError(f"Unsupported clip backend: {backend}")

def count_trainable_params(model: Any) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total

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
    def __init__(self, jsonl_path: str, image_roots: List[str]):
        raw_rows = load_jsonl(jsonl_path)
        self.rows: List[Dict] = []
        for row in raw_rows:
            if "image_path" not in row:
                continue
            resolved = resolve_image_path(str(row["image_path"]), image_roots)
            if not resolved:
                continue
            nr = dict(row)
            nr["image_path"] = resolved
            self.rows.append(nr)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _extract_captions(row: Dict) -> List[str]:
        captions: List[str] = []

        cap_obj = row.get("caption_rewrite", {})
        if isinstance(cap_obj, dict):
            if isinstance(cap_obj.get("captions"), list):
                captions.extend([str(x).strip() for x in cap_obj["captions"] if str(x).strip()])
            cap = str(cap_obj.get("caption", "")).strip()
            if cap:
                captions.append(cap)

        if isinstance(row.get("captions"), list):
            captions.extend([str(x).strip() for x in row["captions"] if str(x).strip()])

        if isinstance(row.get("caption"), str) and row["caption"].strip():
            captions.append(row["caption"].strip())

        uniq_caps: List[str] = []
        seen = set()
        for caption in captions:
            if caption in seen:
                continue
            seen.add(caption)
            uniq_caps.append(caption)

        if not uniq_caps:
            uniq_caps = ["a person"]
        return uniq_caps

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        captions = self._extract_captions(row)
        pseudo_label = row.get("pseudo_label", row.get("pid", row.get("person_id", row.get("id", -1))))

        return {
            "image": image,
            "image_path": row["image_path"],
            "captions": captions,
            "pseudo_label": str(pseudo_label),
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
        caption_groups = [x["captions"] for x in batch]
        flat_captions = [cap for caps in caption_groups for cap in caps]
        text_to_image_index = torch.tensor(
            [img_idx for img_idx, caps in enumerate(caption_groups) for _ in caps],
            dtype=torch.long,
        )

        image_to_text_mask = torch.zeros(len(batch), len(flat_captions), dtype=torch.bool)
        cursor = 0
        for i, caps in enumerate(caption_groups):
            image_to_text_mask[i, cursor: cursor + len(caps)] = True
            cursor += len(caps)

        if backend == "hf":
            image_inputs = processor(images=images, return_tensors="pt")
            text_inputs = processor.tokenizer(flat_captions, return_tensors="pt", padding=True, truncation=True)
        else:
            image_inputs = {
                "pixel_values": torch.stack([oc_preprocess(img.convert("RGB")) for img in images], dim=0),
            }
            text_inputs = {"input_ids": oc_tokenizer(flat_captions)}

        return {
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
            "captions": flat_captions,
            "image_paths": [x["image_path"] for x in batch],
            "pseudo_labels": [x["pseudo_label"] for x in batch],
            "text_to_image_index": text_to_image_index,
            "image_to_text_mask": image_to_text_mask,
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

@torch.no_grad()
def extract_all_image_embeddings(model: Any, backend: str, dataloader: DataLoader, device: str):
    model.eval()

    all_image_features: List[torch.Tensor] = []
    all_text_features: List[torch.Tensor] = []
    all_paths: List[str] = []
    all_labels: List[str] = []
    has_any_label = False

    for batch in dataloader:
        image_inputs = to_device(batch["image_inputs"], device)
        text_inputs = to_device(batch["text_inputs"], device)

        image_feat = encode_image_features(model, backend, image_inputs["pixel_values"])
        text_feat = encode_text_features(model, backend, text_inputs)

        image_feat = image_feat / (image_feat.norm(dim=-1, keepdim=True) + 1e-6)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)

        all_image_features.append(image_feat.cpu())
        all_text_features.append(text_feat.cpu())
        all_paths.extend(batch.get("image_paths", []))

        labels = batch.get("pseudo_labels")
        if labels is None:
            all_labels.extend(["-1"] * image_feat.size(0))
        else:
            labels_str = [str(x) for x in labels]
            all_labels.extend(labels_str)
            if any(x not in {"", "-1", "none", "None"} for x in labels_str):
                has_any_label = True

    model.train()

    image_features = torch.cat(all_image_features, dim=0) if all_image_features else torch.empty((0, 0))
    text_features = torch.cat(all_text_features, dim=0) if all_text_features else torch.empty((0, 0))
    labels_out = all_labels if has_any_label else None

    return {
        "image_features": image_features,
        "text_features": text_features,
        "paths": all_paths,
        "labels": labels_out,
    }


def visualize_image_tsne(features: torch.Tensor, labels: Optional[List[str]], save_path: str) -> None:
    if features.numel() == 0:
        return

    max_samples = min(2000, features.size(0))
    feats = features[:max_samples]
    sub_labels = labels[:max_samples] if labels is not None else None

    reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats.numpy())

    plt.figure(figsize=(8, 8))
    if sub_labels is not None:
        uniq = {k: idx for idx, k in enumerate(sorted(set(sub_labels)))}
        cvals = [uniq[k] for k in sub_labels]
        plt.scatter(reduced[:, 0], reduced[:, 1], c=cvals, cmap="tab20", s=10, alpha=0.75)
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], c="blue", s=10, alpha=0.75)

    plt.title("Image Embedding t-SNE")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_text_tsne(features: torch.Tensor, labels: Optional[List[str]], save_path: str) -> None:
    if features.numel() == 0:
        return

    max_samples = min(2000, features.size(0))
    feats = features[:max_samples]
    sub_labels = labels[:max_samples] if labels is not None else None

    reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats.numpy())

    plt.figure(figsize=(8, 8))
    if sub_labels is not None:
        uniq = {k: idx for idx, k in enumerate(sorted(set(sub_labels)))}
        cvals = [uniq[k] for k in sub_labels]
        plt.scatter(reduced[:, 0], reduced[:, 1], c=cvals, cmap="tab20", s=10, alpha=0.75)
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], c="red", s=10, alpha=0.75)

    plt.title("Text Embedding t-SNE")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_joint_tsne(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: Optional[List[str]],
    save_path: str,
) -> None:
    if image_features.numel() == 0 or text_features.numel() == 0:
        return

    n = min(image_features.size(0), text_features.size(0), 2000)
    img = image_features[:n]
    txt = text_features[:n]
    sub_labels = labels[:n] if labels is not None else None

    combined = torch.cat([img, txt], dim=0)
    reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(combined.numpy())

    img_reduced = reduced[:n]
    txt_reduced = reduced[n:]

    plt.figure(figsize=(8, 8))
    if sub_labels is not None:
        uniq = {k: idx for idx, k in enumerate(sorted(set(sub_labels)))}
        cvals = [uniq[k] for k in sub_labels]
        plt.scatter(img_reduced[:, 0], img_reduced[:, 1], c=cvals, cmap="tab20", s=10, alpha=0.75, marker="o", label="image")
        plt.scatter(txt_reduced[:, 0], txt_reduced[:, 1], c=cvals, cmap="tab20", s=18, alpha=0.75, marker="x", label="text")
    else:
        plt.scatter(img_reduced[:, 0], img_reduced[:, 1], c="blue", s=10, alpha=0.75, marker="o", label="image")
        plt.scatter(txt_reduced[:, 0], txt_reduced[:, 1], c="red", s=18, alpha=0.75, marker="x", label="text")

    plt.title("Joint Image-Text Embedding t-SNE")
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_dbscan(features: torch.Tensor):
    if features.numel() == 0:
        return []
    feats = features.numpy()
    cluster = DBSCAN(eps=0.5, min_samples=4, metric="cosine")
    return cluster.fit_predict(feats)


def log_cluster_statistics(cluster_labels, total_samples: int) -> None:
    if total_samples == 0:
        print("---------------------------------")
        print("Total samples: 0")
        print("Clusters found: 0")
        print("Noise samples: 0")
        print("Average cluster size: 0.00")
        print("---------------------------------")
        return

    valid_labels = [int(x) for x in cluster_labels if int(x) != -1]
    noise_samples = int(sum(1 for x in cluster_labels if int(x) == -1))

    if valid_labels:
        count_by_cluster: Dict[int, int] = {}
        for lb in valid_labels:
            count_by_cluster[lb] = count_by_cluster.get(lb, 0) + 1
        sizes = list(count_by_cluster.values())
        num_clusters = len(count_by_cluster)
        avg_cluster = float(sum(sizes)) / max(1, len(sizes))
        max_cluster = max(sizes)
        min_cluster = min(sizes)
    else:
        num_clusters = 0
        avg_cluster = 0.0
        max_cluster = 0
        min_cluster = 0

    print("---------------------------------")
    print(f"Total samples: {total_samples}")
    print(f"Clusters found: {num_clusters}")
    print(f"Noise samples: {noise_samples}")
    print(f"Average cluster size: {avg_cluster:.2f}")
    print("---------------------------------")
    print(f"Max cluster size: {max_cluster}")
    print(f"Min cluster size: {min_cluster}")


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
        image_inputs = to_device(batch["image_inputs"], device)
        text_inputs = to_device(batch["text_inputs"], device)

        image_feat = encode_image_features(model, backend, image_inputs["pixel_values"])
        caption_feat = encode_text_features(model, backend, text_inputs)

        logit_scale = model.logit_scale.exp()

        total_loss, stats = loss_fn(
            image_feat=image_feat,
            caption_feat=caption_feat,
            image_to_text_mask=batch["image_to_text_mask"].to(device),
            text_to_image_index=batch["text_to_image_index"].to(device),
            logit_scale=logit_scale,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bsz = batch["batch_size"]
        meter["loss_total"] += float(stats["loss_total"].item()) * bsz
        meter["loss_itc"] += float(stats["loss_itc"].item()) * bsz
        meter["loss_rank"] += float(stats["loss_rank"].item()) * bsz
        meter["n"] += bsz

    n = max(1, meter["n"])
    return {
        "loss_total": meter["loss_total"] / n,
        "loss_itc": meter["loss_itc"] / n,
        "loss_rank": meter["loss_rank"] / n,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--train_jsonl", type=str, default="outputs/cuhk_train_qwen_caption_v2.jsonl")
    ap.add_argument("--clip_model_path", type=str, default="/root/autodl-tmp/PSPD/hf_models/openclip")
    ap.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "hf", "open_clip"])
    ap.add_argument("--save_dir", type=str, default="outputs/checkpoints_itc_rank2")
    ap.add_argument("--test_jsonl", type=str, default="outputs/cuhk_test_caption.jsonl")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=2e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--train_image_root", type=str, default="/root/autodl-tmp/PSPD/dataset/CUHK-PEDES")
    ap.add_argument("--eval_image_root", type=str, default="/root/autodl-tmp/PSPD/dataset/CUHK-PEDES")
    ap.add_argument("--show_progress", type=int, default=1, choices=[0, 1])
    ap.add_argument("--freeze_text_encoder", type=int, default=1, choices=[0, 1])
    ap.add_argument("--freeze_text_proj", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    # ============== 初始化日志 ==============
    logger = setup_logger(args.save_dir)
    logger.info("========== Training Start ==========")

    if not args.test_jsonl:
        raise ValueError(
            "--test_jsonl is required for per-epoch retrieval evaluation. "
            "If you only have raw dataset captions, first generate JSONL with "
            "`python data/generate_text_jsonl.py ...`."
        )
    if not os.path.exists(args.test_jsonl):
        raise FileNotFoundError(f"test_jsonl not found: {args.test_jsonl}")

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    train_image_roots: List[str] = []
    if args.train_image_root:
        train_image_roots.append(args.train_image_root)

    eval_image_roots: List[str] = []
    if args.eval_image_root:
        eval_image_roots.append(args.eval_image_root)

    datasets_cfg = cfg.get("datasets", {})
    if isinstance(datasets_cfg, dict):
        for _, ds_cfg in datasets_cfg.items():
            if isinstance(ds_cfg, dict):
                root = ds_cfg.get("image_root", "")
                if root:
                    train_image_roots.append(root)
                    eval_image_roots.append(root)
                    parent = os.path.dirname(root.rstrip('/'))
                    if parent:
                        train_image_roots.append(parent)
                        eval_image_roots.append(parent)

    # dedup roots while keeping order
    dedup_train_roots: List[str] = []
    seen_train_roots = set()
    for r in train_image_roots:
        rr = os.path.abspath(r)
        if rr in seen_train_roots:
            continue
        seen_train_roots.add(rr)
        dedup_train_roots.append(rr)
    train_image_roots = dedup_train_roots

    dedup_eval_roots: List[str] = []
    seen_eval_roots = set()
    for r in eval_image_roots:
        rr = os.path.abspath(r)
        if rr in seen_eval_roots:
            continue
        seen_eval_roots.add(rr)
        dedup_eval_roots.append(rr)
    eval_image_roots = dedup_eval_roots

    backend = resolve_clip_backend(args.clip_backend, args.clip_model_path)
    model, processor, oc_preprocess, oc_tokenizer = load_clip_components(
        model_path=args.clip_model_path,
        backend=backend,
        device=device,
    )
    logger.info(f"clip backend: {backend}")

    freeze_text_encoder = bool(args.freeze_text_encoder)
    freeze_text_proj = bool(args.freeze_text_proj)

    if freeze_text_encoder or freeze_text_proj:
        set_text_trainable(
            model=model,
            backend=backend,
            train_encoder=not freeze_text_encoder,
            train_proj=not freeze_text_proj,
        )
        logger.info(
            f"text freeze settings: freeze_encoder={freeze_text_encoder}, freeze_proj={freeze_text_proj}"
        )

    trainable_params, total_params = count_trainable_params(model)
    logger.info(f"trainable params: {trainable_params}/{total_params}")

    dataset = ITCRankDataset(args.train_jsonl, image_roots=train_image_roots)
    if len(dataset) == 0:
        raise RuntimeError(
            "No valid training images found. Please check --train_jsonl paths or set --train_image_root. "
            "Current roots: " + str(train_image_roots)
        )
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
    extract_loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=build_collate(backend=backend, processor=processor, oc_preprocess=oc_preprocess, oc_tokenizer=oc_tokenizer),
    )

    rank_cfg = RankAuxConfig(
    )
    loss_fn = ITCWithRankAuxLoss(rank_cfg)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if len(optim_params) == 0:
        raise RuntimeError("No trainable parameters found. Please check freeze flags or model.")
    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    meta = {
        "train_jsonl": args.train_jsonl,
        "clip_model_path": args.clip_model_path,
        "clip_backend": backend,
        "test_jsonl": args.test_jsonl,
        "train_image_root": args.train_image_root,
        "eval_image_root": args.eval_image_root,
        "rank_cfg": asdict(rank_cfg),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "freeze_text_encoder": freeze_text_encoder,
        "freeze_text_proj": freeze_text_proj,
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

        log_str = (
            f"[epoch {epoch:03d}] "
            f"best_loss={best:.6f} "
            f"loss_total={stats['loss_total']:.6f} "
            f"loss_itc={stats['loss_itc']:.6f} "
            f"loss_rank={stats['loss_rank']:.6f} "
            f"rank1={stats['rank1']:.4f} "
            f"rank5={stats['rank5']:.4f} "
            f"rank10={stats['rank10']:.4f} "
            f"mAP={stats['mAP']:.4f}"
        )

        print(log_str)
        logger.info(log_str)

        if epoch % 5 == 0:
            try:
                extracted = extract_all_image_embeddings(model=model, backend=backend, dataloader=extract_loader, device=device)
                emb_save_path = os.path.join(args.save_dir, f"epoch_{epoch}_image_embeddings.pt")
                torch.save(
                    {
                        "image_features": extracted["image_features"].cpu(),
                        "text_features": extracted["text_features"].cpu(),
                        "paths": extracted["paths"],
                        "labels": extracted["labels"],
                    },
                    emb_save_path,
                )
                logger.info(f"saved embeddings: {emb_save_path}")

                image_tsne_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}_image_tsne.png")
                text_tsne_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}_text_tsne.png")
                joint_tsne_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}_joint_tsne.png")
                visualize_image_tsne(features=extracted["image_features"], labels=extracted["labels"], save_path=image_tsne_path)
                visualize_text_tsne(features=extracted["text_features"], labels=extracted["labels"], save_path=text_tsne_path)
                visualize_joint_tsne(
                    image_features=extracted["image_features"],
                    text_features=extracted["text_features"],
                    labels=extracted["labels"],
                    save_path=joint_tsne_path,
                )
                logger.info(f"saved image/text/joint tsne: {image_tsne_path}, {text_tsne_path}, {joint_tsne_path}")

                cluster_labels = run_dbscan(extracted["image_features"])
                cluster_tensor = torch.tensor(cluster_labels, dtype=torch.long)
                cluster_save_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}_cluster_results.pt")
                torch.save(
                    {
                        "features": extracted["image_features"].cpu(),
                        "paths": extracted["paths"],
                        "original_labels": extracted["labels"],
                        "cluster_labels": cluster_tensor.cpu(),
                    },
                    cluster_save_path,
                )

                cluster_map = [
                    {"image_path": path, "cluster_id": int(cid)}
                    for path, cid in zip(extracted["paths"], cluster_labels)
                ]
                cluster_map_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}_cluster_map.json")
                with open(cluster_map_path, "w", encoding="utf-8") as f:
                    json.dump(cluster_map, f, ensure_ascii=False, indent=2)

                logger.info(f"saved cluster results: {cluster_save_path}")
                logger.info(f"saved cluster map: {cluster_map_path}")
                log_cluster_statistics(cluster_labels=cluster_labels, total_samples=int(extracted["image_features"].size(0)))
            except Exception as exc:
                logger.warning(f"embedding extraction/clustering failed at epoch {epoch}: {exc}")
    logger.info("========== Training Finished ==========")

if __name__ == "__main__":
    main()
