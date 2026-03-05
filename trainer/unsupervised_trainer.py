from __future__ import annotations

import random
import copy
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from data_mvp.datasets import Sample
from models.dino_backbone import DinoV2Backbone
from models.reid_head import ReidProjectionHead
from utils.cluster_metrics import evaluate_cluster_quality
from utils.itc_metrics import evaluate_itc_quality
from utils.pipeline_score import compute_stage2_score
import os


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2)
        loss_vals = []
        for i in range(embeddings.size(0)):
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]
            pos_mask[i] = False
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            loss_vals.append(F.relu(hardest_pos - hardest_neg + self.margin))
        if not loss_vals:
            return embeddings.new_zeros(())
        return torch.stack(loss_vals).mean()


class PseudoLabelDataset(Dataset):
    def __init__(self, samples: List[Sample], pseudo_labels: np.ndarray, processor) -> None:
        self.items: List[Tuple[Sample, int]] = []
        for s, lb in zip(samples, pseudo_labels.tolist()):
            if lb == -1:
                continue
            self.items.append((s, int(lb)))
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        sample, label = self.items[idx]
        image = self.processor(images=sample_image(sample.image_path), return_tensors="pt")["pixel_values"][0]
        return image, label


class CaptionDataset(Dataset):
    def __init__(self, samples: List[Sample], pseudo_labels: np.ndarray, processor, vocab_size: int = 50000, max_len: int = 24) -> None:
        self.items: List[Tuple[Sample, int, str]] = []
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.processor = processor
        for s, lb in zip(samples, pseudo_labels.tolist()):
            if lb == -1:
                continue
            if not s.captions:
                continue
            self.items.append((s, int(lb), random.choice(s.captions)))

    def __len__(self):
        return len(self.items)

    def _tokenize(self, text: str) -> torch.Tensor:
        words = text.lower().strip().split()[: self.max_len]
        if not words:
            words = ["[empty]"]
        ids = [abs(hash(w)) % self.vocab_size for w in words]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        s, label, caption = self.items[idx]
        image = self.processor(images=sample_image(s.image_path), return_tensors="pt")["pixel_values"][0]
        token_ids = self._tokenize(caption)
        return image, label, token_ids


def sample_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def collate_pseudo(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return images, labels


def collate_caption(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    seqs = [b[2] for b in batch]
    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros((len(seqs), max_len), dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
        mask[i, : s.size(0)] = 1.0
    return images, labels, padded, mask


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 50000, dim: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return F.normalize(x, p=2, dim=-1)


@dataclass
class TrainerConfig:
    batch_size: int = 64
    num_workers: int = 4
    stage2_epochs: int = 40
    stage3_epochs: int = 20
    recluster_every: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    triplet_margin: float = 0.3
    itc_temperature: float = 0.05
    itc_weight: float = 0.2


class UnsupervisedReIDTrainer:
    def __init__(self, backbone: DinoV2Backbone, samples: List[Sample], device: str, cfg: Optional[TrainerConfig] = None) -> None:
        self.backbone = backbone.to(device)
        self.samples = samples
        self.device = device
        self.cfg = cfg or TrainerConfig()
        self.pca_mean: Optional[torch.Tensor] = None
        self.pca_eigvecs: Optional[torch.Tensor] = None
        self.pca_eigvals: Optional[torch.Tensor] = None
        self.pca_k: Optional[int] = None

    @torch.no_grad()
    def fit_pca_whitening(self, raw_features: torch.Tensor, eps: float = 1e-6, max_samples: int = 4096) -> None:
        """Fit PCA whitening statistics and store them on trainer."""
        x = raw_features.detach().cpu().float()
        n = x.size(0)

        if n > max_samples:
            idx = torch.randperm(n)[:max_samples]
            x_est = x[idx]
        else:
            x_est = x

        mean = x_est.mean(dim=0, keepdim=True)
        x_est_center = x_est - mean

        cov = (x_est_center.T @ x_est_center) / max(1, x_est_center.size(0) - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=1e-6)

        shrink_alpha = 0.1
        eigvals = (1 - shrink_alpha) * eigvals + shrink_alpha * eigvals.mean()

        cumulative = torch.cumsum(eigvals, dim=0) / eigvals.sum()
        k = int((cumulative < 0.85).sum().item()) + 1
        eigvals = eigvals[-k:]
        eigvecs = eigvecs[:, -k:]

        eig_max = float(eigvals.max().item())
        eig_min = float(eigvals.min().item())
        eigen_ratio = eig_max / max(eig_min, eps)

        self.pca_mean = mean
        self.pca_eigvecs = eigvecs
        self.pca_eigvals = eigvals
        self.pca_k = int(k)

        print("[WHITENING GEOMETRY] eigen_ratio_raw:", eigen_ratio)

    @torch.no_grad()
    def apply_pca_whitening(self, raw_features: torch.Tensor) -> torch.Tensor:

        stage1_feature_path = (
            "/home/u2024218474/jupyterlab/PSPD/"
            "outputs/pipeline/cuhk_train_stage1_features.pt"
        )

        need_fit = (
                not hasattr(self, "pca_mean")
                or self.pca_mean is None
                or self.pca_eigvecs is None
                or self.pca_eigvals is None
        )

        if need_fit:

            print("[PCA] Whitening parameters missing. Fitting from Stage1 features...")

            if not os.path.exists(stage1_feature_path):
                raise RuntimeError(
                    f"Stage1 feature file not found at: {stage1_feature_path}"
                )

            stage1_features = torch.load(stage1_feature_path)

            if isinstance(stage1_features, dict):
                stage1_features = stage1_features["features"]

            stage1_features = stage1_features.float()

            mean = stage1_features.mean(dim=0, keepdim=True)
            centered = stage1_features - mean

            cov = centered.t() @ centered / (centered.size(0) - 1)

            eigvals, eigvecs = torch.linalg.eigh(cov)

            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            k = min(484, eigvals.size(0))

            self.pca_mean = mean
            self.pca_eigvals = eigvals[:k]
            self.pca_eigvecs = eigvecs[:, :k]
            self.pca_k = k

            print(f"[PCA] Whitening fitted. Retained dim: {k}")

        raw_features = raw_features.float()

        device = raw_features.device

        mean = self.pca_mean.to(device)
        eigvecs = self.pca_eigvecs.to(device)
        eigvals = self.pca_eigvals.to(device)

        centered = raw_features - mean
        projected = centered @ eigvecs
        whitened = projected / torch.sqrt(eigvals + 1e-6)

        return whitened

    @torch.no_grad()
    def extract_features(self, return_raw: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        if not hasattr(self.backbone, "_eval_mode_printed"):
            print("Model in eval mode:", not self.backbone.training)
            self.backbone._eval_mode_printed = True

        raw_embs: List[torch.Tensor] = []
        loader = DataLoader(
            self.samples,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_samples,
        )
        for pixel_values, _ in tqdm(loader, desc="stage1-extract", leave=False):
            pixel_values = pixel_values.to(self.device)
            _, raw_feat = self.backbone(pixel_values, return_raw=True)
            raw_embs.append(raw_feat.cpu())

        raw_features = torch.cat(raw_embs, dim=0)
        self.fit_pca_whitening(raw_features)
        whitened_raw = self.apply_pca_whitening(raw_features)
        norm_features = F.normalize(whitened_raw, p=2, dim=1)

        if return_raw:
            return norm_features, whitened_raw
        return norm_features

    def generate_pseudo_labels(self, embeddings: torch.Tensor) -> np.ndarray:


        if not hasattr(self, "pca_mean"):
            raise RuntimeError("PCA whitening parameters not found. "
                               "Stage1 must fit PCA before Stage2.")


        raw = embeddings.detach().cpu().float()

        whitened = self.apply_pca_whitening(raw)

        features = F.normalize(whitened, p=2, dim=1)

        print("Clustering dim:", features.shape[1])


        similarity = features @ features.t()
        distance = (1.0 - similarity).clamp_min(0.0)

        n = distance.size(0)
        if n < 2:
            raise RuntimeError("DBSCAN requires at least 2 samples.")

        tri_i, tri_j = torch.triu_indices(n, n, offset=1)
        pairwise = distance[tri_i, tri_j]

        if pairwise.numel() == 0:
            raise RuntimeError("Pairwise distance estimation failed.")


        sample_size = min(10000, pairwise.numel())
        if pairwise.numel() > sample_size:
            sample_idx = torch.randperm(pairwise.numel())[:sample_size]
            sampled = pairwise[sample_idx]
        else:
            sampled = pairwise

        median_distance = float(sampled.median().item())


        eps = median_distance * 0.7

        # reasonable safety bounds
        eps = float(max(0.30, min(1.20, eps)))


        dist_np = distance.numpy().astype(np.float32)

        final_labels = None
        final_num_clusters = 0
        final_noise_ratio = 1.0
        final_avg_cluster_size = 0.0

        for attempt in range(5):

            clusterer = DBSCAN(
                eps=eps,
                min_samples=4,
                metric="precomputed",
            )

            labels = clusterer.fit_predict(dist_np)

            valid = labels[labels != -1]
            final_num_clusters = len(set(valid.tolist())) if valid.size > 0 else 0
            final_noise_ratio = float((labels == -1).sum()) / max(1, len(labels))

            if final_num_clusters > 0:
                _, counts = np.unique(valid, return_counts=True)
                final_avg_cluster_size = float(np.mean(counts))
            else:
                final_avg_cluster_size = 0.0

            final_labels = labels

            print(
                f"[DBSCAN attempt {attempt}] "
                f"eps={eps:.4f} "
                f"clusters={final_num_clusters} "
                f"noise_ratio={final_noise_ratio:.4f}"
            )

            # ---------------------------------------------------
            # Adaptive eps tuning
            # ---------------------------------------------------
            if final_num_clusters < 300 and attempt < 4:
                eps *= 0.9
                continue

            if final_num_clusters > 2000 and attempt < 4:
                eps *= 1.1
                continue

            break

        if final_labels is None:
            raise RuntimeError("DBSCAN failed to produce pseudo labels.")


        log_line = (
            f"[DBSCAN STABLE] "
            f"median_distance={median_distance:.6f} "
            f"final_eps={eps:.6f} "
            f"num_clusters={final_num_clusters} "
            f"noise_ratio={final_noise_ratio:.6f} "
            f"avg_cluster_size={final_avg_cluster_size:.4f}"
        )

        print(log_line)

        Path("outputs").mkdir(parents=True, exist_ok=True)
        with open("outputs/stage2_training.log", "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
            f.flush()

        return final_labels


    def _collate_samples(self, batch_samples: List[Sample]):
        images = [sample_image(s.image_path) for s in batch_samples]
        x = self.backbone.preprocess(images=images, return_tensors="pt")["pixel_values"]
        return x, batch_samples

    def _build_classifier(self, pseudo_labels: np.ndarray) -> nn.Module:
        valid = sorted(set([int(x) for x in pseudo_labels.tolist() if x != -1]))
        n_classes = len(valid)
        if n_classes <= 0:
            raise RuntimeError("No valid clusters produced by DBSCAN.")
        return nn.Linear(self.backbone.embed_dim, n_classes).to(self.device)

    @torch.no_grad()
    def _extract_features_with_backbone(self, model: nn.Module) -> torch.Tensor:
        was_training = model.training
        model.eval()

        raw_embs: List[torch.Tensor] = []
        loader = DataLoader(
            self.samples,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_samples,
        )
        for pixel_values, _ in tqdm(loader, desc="stage2-cluster-extract", leave=False):
            pixel_values = pixel_values.to(self.device)
            _, raw_feat = model(pixel_values, return_raw=True)
            raw_embs.append(raw_feat.cpu())

        raw_features = torch.cat(raw_embs, dim=0)

        if was_training:
            model.train()
        return raw_features

    def _set_backbone_trainability(self, epoch: int, warmup_epochs: int = 5) -> None:
        if epoch <= warmup_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = False
            return

        # Gradual unfreeze: lower frozen ratio as epochs progress.
        unfreeze_progress = min(1.0, (epoch - warmup_epochs) / max(1, self.cfg.stage2_epochs - warmup_epochs))
        freeze_ratio = max(0.0, 0.9 - 0.9 * unfreeze_progress)
        self.backbone.freeze_first_blocks(ratio=freeze_ratio)

    def _remap_labels(self, labels: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
        return torch.tensor([mapping[int(lb)] for lb in labels.tolist()], dtype=torch.long, device=labels.device)

    def train_stage2(self, initial_pseudo_labels: np.ndarray) -> Dict[str, object]:
        max_epoch = self.cfg.stage2_epochs
        min_epoch = 15
        ema_momentum = 0.999

        triplet_loss_fn = TripletLoss(margin=self.cfg.triplet_margin)
        pseudo_labels = initial_pseudo_labels.copy()
        prev_pseudo_labels = initial_pseudo_labels.copy()
        teacher_backbone: DinoV2Backbone = copy.deepcopy(self.backbone).to(self.device)
        for p in teacher_backbone.parameters():
            p.requires_grad = False

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "stage2_training.log"
        stage2_logger = logging.getLogger("stage2_training")
        stage2_logger.setLevel(logging.INFO)
        stage2_logger.propagate = False
        for h in list(stage2_logger.handlers):
            stage2_logger.removeHandler(h)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        stage2_logger.addHandler(fh)

        def _sync_ema_teacher(momentum: float = ema_momentum) -> None:
            with torch.no_grad():
                for t_p, s_p in zip(teacher_backbone.parameters(), self.backbone.parameters()):
                    t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)

        classifier = self._build_classifier(pseudo_labels)
        proj = ReidProjectionHead(in_dim=self.backbone.embed_dim, out_dim=self.backbone.embed_dim).to(self.device)
        optimizer = AdamW(
            [
                {
                    "params": list(self.backbone.parameters()),
                    "lr": self.cfg.lr * 0.1,
                    "group_name": "backbone",
                },
                {
                    "params": list(classifier.parameters()),
                    "lr": self.cfg.lr,
                    "group_name": "classifier",
                },
                {
                    "params": list(proj.parameters()),
                    "lr": self.cfg.lr,
                    "group_name": "projection",
                },
            ],
            weight_decay=self.cfg.weight_decay,
        )

        last_cluster_metrics: Dict[str, float] = {
            "inter_intra_ratio": 0.0,
            "noise_ratio": 1.0,
            "label_change_ratio": 1.0,
            "num_clusters": 0.0,
        }
        prev_recluster_metrics: Optional[Dict[str, float]] = None
        stability_counter = 0
        ce_epoch, tri_epoch, itc_epoch = 0.0, 0.0, 0.0

        ckpt_files: List[Path] = []

        def _save_recluster_checkpoint(current_epoch: int, cluster_stats: Dict[str, float]) -> None:
            nonlocal ckpt_files
            ckpt_path = output_dir / f"stage2_recluster_epoch_{current_epoch:03d}.pt"
            torch.save(
                {
                    "model_state_dict": self.backbone.state_dict(),
                    "teacher_state_dict": teacher_backbone.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "projection_state_dict": proj.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "epoch": current_epoch,
                    "cluster_stats": cluster_stats,
                },
                ckpt_path,
            )
            ckpt_files.append(ckpt_path)
            while len(ckpt_files) > 3:
                old = ckpt_files.pop(0)
                if old.exists():
                    old.unlink()

        for epoch in range(1, max_epoch + 1):
            prev_student_state = copy.deepcopy(self.backbone.state_dict())
            prev_teacher_state = copy.deepcopy(teacher_backbone.state_dict())
            prev_optimizer_state = copy.deepcopy(optimizer.state_dict())
            prev_classifier_state = copy.deepcopy(classifier.state_dict())
            prev_proj_state = copy.deepcopy(proj.state_dict())

            self._set_backbone_trainability(epoch=epoch, warmup_epochs=5)
            valid_labels = sorted(set([int(x) for x in pseudo_labels.tolist() if x != -1]))
            label_map = {lb: i for i, lb in enumerate(valid_labels)}

            dataset = PseudoLabelDataset(self.samples, pseudo_labels, self.backbone.preprocess)
            if len(dataset) == 0:
                raise RuntimeError("All samples are marked as noise. Try a larger eps.")
            loader = DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                collate_fn=collate_pseudo,
            )

            self.backbone.train()
            classifier.train()
            proj.train()
            ce_meter = 0.0
            tri_meter = 0.0
            itc_meter = 0.0
            n = 0
            for images, raw_labels in tqdm(loader, desc=f"stage2-e{epoch}", leave=False):
                images = images.to(self.device)
                raw_labels = raw_labels.to(self.device)
                labels = self._remap_labels(raw_labels, label_map)

                feat = self.backbone(images)
                logits = classifier(feat)

                ce_loss = F.cross_entropy(logits, labels)
                tri_loss = triplet_loss_fn(feat, labels)
                z_feat = F.normalize(feat, p=2, dim=-1)
                z_proj = F.normalize(proj(feat), p=2, dim=-1)
                sim = z_proj @ z_feat.detach().t()
                targets = torch.arange(sim.size(0), device=sim.device)
                itc_loss = 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets))

                loss = ce_loss + tri_loss + self.cfg.itc_weight * itc_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                _sync_ema_teacher()

                bsz = images.size(0)
                ce_meter += float(ce_loss.item()) * bsz
                tri_meter += float(tri_loss.item()) * bsz
                itc_meter += float(itc_loss.item()) * bsz
                n += bsz

            ce_epoch = ce_meter / max(1, n)
            tri_epoch = tri_meter / max(1, n)
            itc_epoch = itc_meter / max(1, n)

            with torch.no_grad():
                raw_teacher_features = self._extract_features_with_backbone(teacher_backbone)
                teacher_features = self.apply_pca_whitening(raw_teacher_features)
                feature_variance = float(teacher_features.var(dim=0, unbiased=False).mean().item())

            if feature_variance < 1e-4:
                print("[COLLAPSE DETECTED] reverting to previous checkpoint")
                self.backbone.load_state_dict(prev_student_state)
                teacher_backbone.load_state_dict(prev_teacher_state)
                optimizer.load_state_dict(prev_optimizer_state)
                classifier.load_state_dict(prev_classifier_state)
                proj.load_state_dict(prev_proj_state)
                for pg in optimizer.param_groups:
                    if pg.get("group_name") == "backbone":
                        pg["lr"] *= 0.5
                continue

            log_payload = {
                "epoch": epoch,
                "ce_loss": ce_epoch,
                "triplet_loss": tri_epoch,
                "itc_loss": itc_epoch,
                "num_clusters": float(last_cluster_metrics.get("num_clusters", 0.0)),
                "noise_ratio": float(last_cluster_metrics.get("noise_ratio", 1.0)),
                "label_change_ratio": float(last_cluster_metrics.get("label_change_ratio", 1.0)),
                "inter_intra_ratio": float(last_cluster_metrics.get("inter_intra_ratio", 0.0)),
                "feature_variance": feature_variance,
                "backbone_lr": float(next(pg["lr"] for pg in optimizer.param_groups if pg.get("group_name") == "backbone")),
            }
            log_line = (
                "[Stage2][Epoch {epoch:03d}] ce={ce_loss:.4f} triplet={triplet_loss:.4f} itc={itc_loss:.4f} "
                "num_clusters={num_clusters:.0f} noise_ratio={noise_ratio:.4f} "
                "label_change_ratio={label_change_ratio:.4f} inter_intra_ratio={inter_intra_ratio:.6f} "
                "feature_variance={feature_variance:.8f} backbone_lr={backbone_lr:.8f}"
            ).format(**log_payload)
            print(log_line)
            stage2_logger.info(log_line)
            fh.flush()

            if ce_epoch < 0.01 and tri_epoch < 0.01:
                print("[Stage2][Warning] Potential overfitting to noisy pseudo labels")

            if epoch % 5 == 0:
                teacher_backbone.eval()
                embeddings = self._extract_features_with_backbone(teacher_backbone)
                new_pseudo_labels = self.generate_pseudo_labels(embeddings)

                metrics = evaluate_cluster_quality(
                    embeddings=embeddings,
                    labels=new_pseudo_labels,
                    prev_labels=prev_pseudo_labels,
                )

                prev_pseudo_labels = pseudo_labels.copy()
                pseudo_labels = new_pseudo_labels
                last_cluster_metrics = metrics

                stage2_score = compute_stage2_score(last_cluster_metrics)
                print(
                    "[Stage2][Recluster] "
                    f"num_clusters={int(metrics['num_clusters'])} "
                    f"noise_ratio={metrics['noise_ratio']:.4f} "
                    f"label_change_ratio={metrics['label_change_ratio']:.4f} "
                    f"inter_intra_ratio={metrics['inter_intra_ratio']:.6f} "
                    f"Stage2_score={stage2_score:.6f}"
                )

                if int(metrics["num_clusters"]) < 5:
                    print("[Stage2 Warning] Cluster collapse detected.")
                    break

                if prev_recluster_metrics is not None:
                    delta_inter_intra = abs(metrics["inter_intra_ratio"] - prev_recluster_metrics["inter_intra_ratio"])
                    delta_noise = abs(metrics["noise_ratio"] - prev_recluster_metrics["noise_ratio"])
                    stable = (
                        metrics["label_change_ratio"] < 0.02
                        and delta_inter_intra < 0.01
                        and delta_noise < 0.01
                    )
                    stability_counter = stability_counter + 1 if stable else 0
                else:
                    stability_counter = 0

                prev_recluster_metrics = metrics

                _save_recluster_checkpoint(epoch, metrics)

                classifier = self._build_classifier(pseudo_labels)
                proj = ReidProjectionHead(in_dim=self.backbone.embed_dim, out_dim=self.backbone.embed_dim).to(self.device)
                optimizer = AdamW(
                    [
                        {
                            "params": list(self.backbone.parameters()),
                            "lr": float(next(pg["lr"] for pg in optimizer.param_groups if pg.get("group_name") == "backbone")),
                            "group_name": "backbone",
                        },
                        {
                            "params": list(classifier.parameters()),
                            "lr": self.cfg.lr,
                            "group_name": "classifier",
                        },
                        {
                            "params": list(proj.parameters()),
                            "lr": self.cfg.lr,
                            "group_name": "projection",
                        },
                    ],
                    weight_decay=self.cfg.weight_decay,
                )

                if epoch >= min_epoch and stability_counter >= 3:
                    print("[Stage2] Cluster-quality convergence detected. Early stopping triggered.")
                    break

        for h in list(stage2_logger.handlers):
            h.flush()
            h.close()
            stage2_logger.removeHandler(h)

        return {
            "pseudo_labels": pseudo_labels,
            "last_cluster_metrics": last_cluster_metrics,
            "final_ce_loss": float(ce_epoch),
            "final_triplet_loss": float(tri_epoch),
            "final_itc_loss": float(itc_epoch),
        }

    def train_stage3_itc(self, pseudo_labels: np.ndarray) -> Dict[str, float]:
        print("[Stage3] ITC fine-tuning")
        self.backbone.freeze_first_blocks(ratio=0.7)

        valid_labels = sorted(set([int(x) for x in pseudo_labels.tolist() if x != -1]))
        label_map = {lb: i for i, lb in enumerate(valid_labels)}

        classifier = self._build_classifier(pseudo_labels)
        proj = ReidProjectionHead(in_dim=self.backbone.embed_dim, out_dim=512).to(self.device)
        text_encoder = SimpleTextEncoder(vocab_size=50000, dim=512).to(self.device)

        dataset = CaptionDataset(self.samples, pseudo_labels, self.backbone.preprocess)
        if len(dataset) == 0:
            print("[Stage3] no captions available; skip ITC fine-tuning")
            return {"alignment_accuracy": 0.0, "pos_sim_mean": 0.0, "neg_sim_mean": 0.0, "sim_gap": 0.0, "itc_loss": 0.0}
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_caption,
        )

        triplet_loss_fn = TripletLoss(margin=self.cfg.triplet_margin)
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        params += list(classifier.parameters()) + list(proj.parameters()) + list(text_encoder.parameters())
        optimizer = AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        prev_sim_gap = None
        prev_itc_loss = None
        stagnation_counter = 0

        last_itc_metrics: Dict[str, float] = {
            "alignment_accuracy": 0.0,
            "pos_sim_mean": 0.0,
            "neg_sim_mean": 0.0,
            "sim_gap": 0.0,
            "itc_loss": 0.0,
        }

        for epoch in range(1, self.cfg.stage3_epochs + 1):
            self.backbone.train()
            ce_meter = 0.0
            tri_meter = 0.0
            itc_meter = 0.0
            n = 0
            epoch_img_proj: List[torch.Tensor] = []
            epoch_txt_proj: List[torch.Tensor] = []

            for images, raw_labels, token_ids, mask in tqdm(loader, desc=f"stage3-e{epoch}", leave=False):
                images = images.to(self.device)
                raw_labels = raw_labels.to(self.device)
                labels = self._remap_labels(raw_labels, label_map)
                token_ids = token_ids.to(self.device)
                mask = mask.to(self.device)

                feat = self.backbone(images)
                logits = classifier(feat)

                ce_loss = F.cross_entropy(logits, labels)
                tri_loss = triplet_loss_fn(feat, labels)

                img_proj = F.normalize(proj(feat), p=2, dim=-1)
                txt_proj = text_encoder(token_ids, mask)
                sim = (img_proj @ txt_proj.t()) / self.cfg.itc_temperature
                targets = torch.arange(sim.size(0), device=sim.device)
                itc_loss = 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets))

                loss = ce_loss + tri_loss + self.cfg.itc_weight * itc_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()

                epoch_img_proj.append(img_proj.detach().cpu())
                epoch_txt_proj.append(txt_proj.detach().cpu())

                bsz = images.size(0)
                ce_meter += float(ce_loss.item()) * bsz
                tri_meter += float(tri_loss.item()) * bsz
                itc_meter += float(itc_loss.item()) * bsz
                n += bsz

            print(
                f"[Stage3][Epoch {epoch:03d}] ce={ce_meter/max(1,n):.4f} "
                f"triplet={tri_meter/max(1,n):.4f} itc={itc_meter/max(1,n):.4f}"
            )
            epoch_itc_loss = itc_meter / max(1, n)
            all_img_proj = torch.cat(epoch_img_proj, dim=0)
            all_txt_proj = torch.cat(epoch_txt_proj, dim=0)
            itc_metrics = evaluate_itc_quality(all_img_proj, all_txt_proj)
            sim_gap = itc_metrics["sim_gap"]

            if prev_sim_gap is not None and prev_itc_loss is not None:
                sim_not_increase = sim_gap <= prev_sim_gap
                small_itc_change = abs(epoch_itc_loss - prev_itc_loss) < 1e-3
                if sim_not_increase and small_itc_change:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

            if prev_sim_gap is not None and sim_gap <= prev_sim_gap:
                print("[Stage3][Warning] sim_gap stagnates or decreases")

            prev_sim_gap = sim_gap
            prev_itc_loss = epoch_itc_loss
            last_itc_metrics = {**itc_metrics, "itc_loss": float(epoch_itc_loss)}

            if stagnation_counter >= 3:
                print("[Stage3] ITC convergence detected.")
                break

        return last_itc_metrics
