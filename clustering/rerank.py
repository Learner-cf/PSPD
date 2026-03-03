import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch


K1 = 20
LAMBDA = 0.3
EPS = 1e-12


def _as_feature_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for key in ("features", "image_features", "feat", "feats", "embeddings"):
            v = obj.get(key)
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, np.ndarray):
                return torch.from_numpy(v)
        # one-level nested fallback
        for _, v in obj.items():
            if isinstance(v, dict):
                try:
                    nested = _as_feature_tensor(v)
                    if isinstance(nested, torch.Tensor):
                        return nested
                except ValueError:
                    continue
        keys = list(obj.keys())
        if {"epoch", "model", "optimizer"}.issubset(set(keys)):
            raise ValueError(
                "Input looks like a training checkpoint (contains epoch/model/optimizer), not feature embeddings. "
                "Please pass an embeddings file such as `epoch_*_image_embeddings.pt` (key: image_features) "
                "or `features.pt` (key: features). "
                f"Current keys: {keys}"
            )
        raise ValueError(
            "features file dict does not contain tensor keys. "
            "Expected one of keys: features, image_features, feat, feats, embeddings. "
            f"available keys: {keys}"
        )
    raise ValueError(
        "features file must be Tensor/ndarray or dict containing one of keys: "
        "features, image_features, feat, feats, embeddings"
    )


def _safe_torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_image_paths(obj: Any) -> Optional[List[str]]:
    if isinstance(obj, dict):
        for key in ("image_paths", "paths"):
            v = obj.get(key)
            if isinstance(v, list):
                out = [str(x) for x in v]
                if out:
                    return out
    return None


def _find_nearby_embedding_file(parent: Path, epoch: Optional[int]) -> Optional[Path]:
    candidates: List[Path] = []
    if isinstance(epoch, int):
        candidates.extend([
            parent / f"epoch_{epoch}_image_embeddings.pt",
            parent / f"epoch_{epoch:03d}_image_embeddings.pt",
        ])

    candidates.extend([
        parent / "epoch_3_image_embeddings.pt",
        parent / "features.pt",
    ])

    for c in candidates:
        if c.exists():
            return c

    globbed = sorted(parent.glob("epoch_*_image_embeddings.pt"))
    if globbed:
        return globbed[-1]
    return None


def _resolve_embeddings_from_checkpoint_path(features_path: str, loaded_obj: Any) -> Any:
    if not isinstance(loaded_obj, dict):
        return loaded_obj
    keys = set(loaded_obj.keys())
    if not {"epoch", "model", "optimizer"}.issubset(keys):
        return loaded_obj

    parent = Path(features_path).resolve().parent
    epoch = loaded_obj.get("epoch", None)
    candidate = _find_nearby_embedding_file(parent=parent, epoch=epoch if isinstance(epoch, int) else None)
    if candidate is not None:
        print(f"[rerank] detected training checkpoint input; auto-loading embeddings from: {candidate}")
        return _safe_torch_load(str(candidate))

    raise ValueError(
        "Input is a training checkpoint and no nearby embeddings file was found. "
        "Expected one of: epoch_*_image_embeddings.pt or features.pt in the same directory. "
        "Tip: run training for at least 5 epochs (it saves epoch_5_image_embeddings.pt), "
        "or first run clustering/generate_hard_negative_groups.py to export features.pt. "
        f"checkpoint_dir={parent}"
    )


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, EPS)
    return (x / denom).astype(np.float32, copy=False)


def _compute_original_distance_and_rank(
    features: np.ndarray,
    original_mem: np.memmap,
    k1: int,
    block_size: int = 1024,
) -> np.ndarray:
    n = features.shape[0]
    initial_rank = np.empty((n, k1 + 1), dtype=np.int32)

    for st in range(0, n, block_size):
        ed = min(st + block_size, n)
        sim = np.matmul(features[st:ed], features.T, dtype=np.float32)
        dist = 1.0 - sim
        original_mem[st:ed] = dist.astype(np.float32, copy=False)

        part = np.argpartition(-sim, kth=k1, axis=1)[:, : k1 + 1]
        part_sim = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        topk = np.take_along_axis(part, order, axis=1)
        initial_rank[st:ed] = topk.astype(np.int32, copy=False)

    original_mem.flush()
    return initial_rank


def _k_reciprocal_neighbors(initial_rank: np.ndarray, index: int, k: int) -> np.ndarray:
    forward = initial_rank[index, : k + 1]
    backward = initial_rank[forward, : k + 1]
    reciprocal_mask = np.any(backward == index, axis=1)
    return forward[reciprocal_mask]


def _build_reciprocal_graph(
    original_mem: np.memmap,
    initial_rank: np.ndarray,
    k1: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    n = initial_rank.shape[0]

    row_indices: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(n)]
    row_values: List[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(n)]

    inv_rows: List[List[int]] = [[] for _ in range(n)]
    inv_vals: List[List[float]] = [[] for _ in range(n)]

    half_k = max(1, k1 // 2)

    for i in range(n):
        reciprocal = _k_reciprocal_neighbors(initial_rank, i, k1)
        expanded = reciprocal.copy()

        for cand in reciprocal:
            cand_recip = _k_reciprocal_neighbors(initial_rank, int(cand), half_k)
            if cand_recip.size == 0:
                continue
            overlap = np.intersect1d(cand_recip, reciprocal, assume_unique=False).size
            if overlap > (2.0 / 3.0) * cand_recip.size:
                expanded = np.concatenate((expanded, cand_recip), axis=0)

        expanded = np.unique(expanded.astype(np.int32, copy=False))
        if expanded.size == 0:
            expanded = np.array([i], dtype=np.int32)

        d = np.asarray(original_mem[i, expanded], dtype=np.float32)
        w = np.exp(-d, dtype=np.float32)
        w_sum = float(np.sum(w))
        if w_sum <= EPS:
            w = np.full_like(w, 1.0 / max(1, w.size), dtype=np.float32)
        else:
            w = (w / w_sum).astype(np.float32, copy=False)

        row_indices[i] = expanded
        row_values[i] = w

        for c, val in zip(expanded.tolist(), w.tolist()):
            inv_rows[c].append(i)
            inv_vals[c].append(float(val))

    inv_rows_np: List[np.ndarray] = [np.asarray(v, dtype=np.int32) for v in inv_rows]
    inv_vals_np: List[np.ndarray] = [np.asarray(v, dtype=np.float32) for v in inv_vals]

    return row_indices, row_values, inv_rows_np, inv_vals_np


def _compute_jaccard_distance(
    row_indices: List[np.ndarray],
    row_values: List[np.ndarray],
    inv_rows: List[np.ndarray],
    inv_vals: List[np.ndarray],
    jaccard_mem: np.memmap,
) -> None:
    n = len(row_indices)

    for i in range(n):
        temp_min = np.zeros(n, dtype=np.float32)
        idx_i = row_indices[i]
        val_i = row_values[i]

        for c, v in zip(idx_i.tolist(), val_i.tolist()):
            related_rows = inv_rows[c]
            related_vals = inv_vals[c]
            if related_rows.size == 0:
                continue
            min_vals = np.minimum(np.float32(v), related_vals)
            temp_min[related_rows] += min_vals

        denom = 2.0 - temp_min
        denom = np.maximum(denom, EPS)
        jacc_row = 1.0 - (temp_min / denom)
        jaccard_mem[i] = jacc_row.astype(np.float32, copy=False)

    jaccard_mem.flush()


def _fuse_distances(
    original_mem: np.memmap,
    jaccard_mem: np.memmap,
    final_mem: np.memmap,
    lambda_value: float,
    block_size: int = 1024,
) -> None:
    n = original_mem.shape[0]
    for st in range(0, n, block_size):
        ed = min(st + block_size, n)
        final_mem[st:ed] = (
            lambda_value * original_mem[st:ed] + (1.0 - lambda_value) * jaccard_mem[st:ed]
        ).astype(np.float32, copy=False)
    final_mem.flush()


def rerank(features: torch.Tensor, output_dir: str, image_paths: Optional[List[str]] = None) -> np.ndarray:
    os.makedirs(output_dir, exist_ok=True)

    feats = features.detach().cpu().numpy().astype(np.float32, copy=False)
    feats = _l2_normalize(feats)
    n = feats.shape[0]

    original_path = os.path.join(output_dir, "original_distance.npy")
    jaccard_path = os.path.join(output_dir, "jaccard_distance.npy")
    final_path = os.path.join(output_dir, "final_distance.npy")
    rerank_paths_path = os.path.join(output_dir, "rerank_paths.json")

    original_mem = np.lib.format.open_memmap(original_path, mode="w+", dtype=np.float32, shape=(n, n))
    initial_rank = _compute_original_distance_and_rank(feats, original_mem, k1=K1)

    row_indices, row_values, inv_rows, inv_vals = _build_reciprocal_graph(
        original_mem=original_mem,
        initial_rank=initial_rank,
        k1=K1,
    )

    jaccard_mem = np.lib.format.open_memmap(jaccard_path, mode="w+", dtype=np.float32, shape=(n, n))
    _compute_jaccard_distance(
        row_indices=row_indices,
        row_values=row_values,
        inv_rows=inv_rows,
        inv_vals=inv_vals,
        jaccard_mem=jaccard_mem,
    )

    final_mem = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.float32, shape=(n, n))
    _fuse_distances(
        original_mem=original_mem,
        jaccard_mem=jaccard_mem,
        final_mem=final_mem,
        lambda_value=LAMBDA,
    )

    orig_mean = float(np.mean(original_mem))
    jacc_mean = float(np.mean(jaccard_mem))

    print(f"N: {n}")
    print(f"k1: {K1}")
    print(f"lambda: {LAMBDA}")
    print(f"mean(original_dist): {orig_mean:.6f}")
    print(f"mean(jaccard_dist): {jacc_mean:.6f}")

    if image_paths is not None and len(image_paths) == n:
        with open(rerank_paths_path, "w", encoding="utf-8") as f:
            json.dump([str(p) for p in image_paths], f, ensure_ascii=False, indent=2)
        print(f"saved rerank paths: {rerank_paths_path}")

    return final_mem


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="k-reciprocal re-ranking for unsupervised ReID clustering")
    ap.add_argument("--features", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/warmup_ckpt/epoch_3_image_embeddings.pt", help="Path to features.pt")
    ap.add_argument("--output_dir", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/dualdistance3")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    loaded = _safe_torch_load(args.features)
    loaded = _resolve_embeddings_from_checkpoint_path(args.features, loaded)
    feat = _as_feature_tensor(loaded)
    paths = _extract_image_paths(loaded)
    rerank(feat, args.output_dir, image_paths=paths)


if __name__ == "__main__":
    main()
