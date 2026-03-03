import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


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
        return model, processor, None

    if backend == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained=os.path.join(model_path, "open_clip_pytorch_model.bin"),
            device=device,
        )
        return model, None, preprocess

    raise ValueError(f"Unsupported clip backend: {backend}")


def encode_image_features(model: Any, backend: str, pixel_values: torch.Tensor) -> torch.Tensor:
    if backend == "hf":
        return model.get_image_features(pixel_values=pixel_values)
    return model.encode_image(pixel_values)


def collect_image_paths(image_root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    image_paths = [
        str(p.resolve())
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    image_paths.sort()
    if not image_paths:
        raise RuntimeError(f"No images found in image_root: {image_root}")
    return image_paths




def collect_image_paths_from_jsonl(image_root: str, subset_jsonl: str, key: str = "image_path") -> List[str]:
    if not os.path.exists(subset_jsonl):
        raise FileNotFoundError(f"subset_jsonl not found: {subset_jsonl}")

    out: List[str] = []
    seen = set()

    with open(subset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = str(row.get(key, "")).strip()
            if not raw:
                continue

            candidates = []
            p = Path(raw)
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append(Path(image_root) / raw)
                candidates.append(Path(image_root) / p.name)

            resolved = ""
            for c in candidates:
                if c.exists():
                    resolved = str(c.resolve())
                    break

            if not resolved:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)

    if not out:
        raise RuntimeError(f"No valid image paths found from subset_jsonl={subset_jsonl}")
    return out

def infer_model_path_from_checkpoint(checkpoint: str, clip_model_path: str) -> str:
    if clip_model_path:
        return clip_model_path

    ckpt_dir = Path(checkpoint).resolve().parent
    meta_path = ckpt_dir / "train_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        inferred = meta.get("clip_model_path", "")
        if inferred:
            return inferred

    raise ValueError(
        "Cannot infer clip model path. Please pass --clip_model_path, "
        "or ensure train_meta.json (with clip_model_path) is next to checkpoint."
    )


def load_model_for_inference(
    checkpoint: str,
    clip_model_path: str,
    clip_backend: str,
    device: str,
):
    model_path = infer_model_path_from_checkpoint(checkpoint=checkpoint, clip_model_path=clip_model_path)
    backend = resolve_clip_backend(clip_backend, model_path)
    model, processor, oc_preprocess = load_clip_components(model_path=model_path, backend=backend, device=device)

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, backend, processor, oc_preprocess


@torch.no_grad()
def extract_image_features(
    model: Any,
    backend: str,
    processor: CLIPProcessor,
    oc_preprocess,
    image_paths: List[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for st in tqdm(range(0, len(image_paths), batch_size), desc="extract-image-features"):
        batch_paths = image_paths[st: st + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        if backend == "hf":
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
        else:
            pixel_values = torch.stack([oc_preprocess(img) for img in images], dim=0).to(device)

        batch_feat = encode_image_features(model=model, backend=backend, pixel_values=pixel_values)
        batch_feat = batch_feat / (batch_feat.norm(dim=-1, keepdim=True) + 1e-6)
        feats.append(batch_feat.cpu())

    return torch.cat(feats, dim=0)


def build_faiss_neighbors(features: torch.Tensor, top_k_neighbors: int) -> np.ndarray:
    feats = features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(feats)

    index = faiss.IndexFlatIP(feats.shape[1])
    index.add(feats)

    search_k = min(top_k_neighbors + 1, feats.shape[0])
    _, indices = index.search(feats, search_k)
    return indices


def build_neighbors_from_distance(distance: np.ndarray, top_k_neighbors: int) -> np.ndarray:
    n = distance.shape[0]
    if distance.shape[1] != n:
        raise ValueError(f"distance matrix must be square, got {distance.shape}")

    search_k = min(top_k_neighbors + 1, n)
    order = np.argpartition(distance, kth=search_k - 1, axis=1)[:, :search_k]
    order_vals = np.take_along_axis(distance, order, axis=1)
    reord = np.argsort(order_vals, axis=1)
    sorted_idx = np.take_along_axis(order, reord, axis=1)
    return sorted_idx


def run_dbscan(features: torch.Tensor, eps: float, min_samples: int) -> np.ndarray:
    feats = features.cpu().numpy().astype(np.float32)
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = cluster.fit_predict(feats)
    return labels



def sanitize_precomputed_distance(distance: np.ndarray) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float32)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"precomputed distance must be square, got {d.shape}")

    if not np.isfinite(d).all():
        d = np.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32, copy=False)

    # enforce symmetric precomputed distances
    d = 0.5 * (d + d.T)

    # DBSCAN(metric='precomputed') requires non-negative distances
    neg_min = float(d.min())
    if neg_min < 0.0:
        print(f"[warn] precomputed distance contains negatives (min={neg_min:.6f}); clipping to 0.")
        d = np.maximum(d, 0.0, dtype=np.float32)

    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32, copy=False)

def run_dbscan_precomputed(distance: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    safe_dist = sanitize_precomputed_distance(distance)
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    return cluster.fit_predict(safe_dist)


def build_neighbor_groups(
    image_paths: List[str],
    knn_indices: np.ndarray,
    top_k_neighbors: int,
    cluster_labels: Optional[np.ndarray] = None,
    in_cluster_k: int = 0,
    out_cluster_k: int = 0,
) -> List[Dict]:
    groups: List[Dict] = []
    n = len(image_paths)

    use_cluster_split = cluster_labels is not None and (in_cluster_k > 0 or out_cluster_k > 0)

    for i in range(n):
        candidates = [j for j in knn_indices[i].tolist() if 0 <= j < n and j != i]

        if not use_cluster_split:
            neighbors = [image_paths[j] for j in candidates[:top_k_neighbors]]
            groups.append({"image_path": image_paths[i], "neighbors": neighbors})
            continue

        in_neighbors: List[str] = []
        out_neighbors: List[str] = []
        cid_i = int(cluster_labels[i])

        for j in candidates:
            same_cluster = cid_i >= 0 and int(cluster_labels[j]) == cid_i
            if same_cluster and len(in_neighbors) < in_cluster_k:
                in_neighbors.append(image_paths[j])
            elif (not same_cluster) and len(out_neighbors) < out_cluster_k:
                out_neighbors.append(image_paths[j])

            if len(in_neighbors) >= in_cluster_k and len(out_neighbors) >= out_cluster_k:
                break

        neighbors = in_neighbors + out_neighbors
        if len(neighbors) < top_k_neighbors:
            used = set(neighbors)
            for j in candidates:
                p = image_paths[j]
                if p in used:
                    continue
                neighbors.append(p)
                used.add(p)
                if len(neighbors) >= top_k_neighbors:
                    break

        groups.append(
            {
                "image_path": image_paths[i],
                "neighbors": neighbors[:top_k_neighbors],
                "cluster_id": cid_i,
                "in_cluster_neighbors": in_neighbors,
                "out_cluster_neighbors": out_neighbors,
            }
        )
    return groups


def build_cluster_statistics(cluster_labels: np.ndarray) -> Tuple[int, int, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for cid in cluster_labels.tolist():
        k = str(int(cid))
        counts[k] = counts.get(k, 0) + 1

    num_clusters = len([k for k in counts.keys() if k != "-1"])
    noise_count = counts.get("-1", 0)
    return num_clusters, noise_count, counts



def load_rerank_paths(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"rerank paths json must be list, got {type(data)}")
    out: List[str] = []
    for x in data:
        xx = str(x).strip()
        if xx:
            out.append(os.path.abspath(xx))
    return out


def align_to_rerank_subset(
    image_paths: List[str],
    features: torch.Tensor,
    rerank_dist: np.ndarray,
    rerank_paths: List[str],
) -> Tuple[List[str], torch.Tensor, np.ndarray]:
    if len(rerank_paths) != rerank_dist.shape[0]:
        raise ValueError(
            f"rerank_paths length mismatch: len(paths)={len(rerank_paths)} vs distance={rerank_dist.shape}"
        )

    idx_by_path: Dict[str, int] = {os.path.abspath(p): i for i, p in enumerate(rerank_paths)}
    keep_current: List[int] = []
    keep_rerank: List[int] = []

    for i, p in enumerate(image_paths):
        ap = os.path.abspath(p)
        if ap in idx_by_path:
            keep_current.append(i)
            keep_rerank.append(idx_by_path[ap])

    if not keep_current:
        raise ValueError("No overlap between current image_root images and rerank paths")

    new_paths = [image_paths[i] for i in keep_current]
    new_features = features[keep_current]
    new_dist = rerank_dist[np.ix_(keep_rerank, keep_rerank)].astype(np.float32, copy=False)

    print(
        f"[info] aligned to rerank subset: {len(new_paths)}/{len(image_paths)} images kept "
        f"(distance size={rerank_dist.shape[0]})"
    )
    return new_paths, new_features, new_dist

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate hard-negative neighbor groups from image embeddings.")
    ap.add_argument("--checkpoint", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/warmup_ckpt/epoch_003.pt")
    ap.add_argument("--image_root", type=str, default="/home/u2024218474/jupyterlab/PSPD/dataset/CUHK-PEDES")
    ap.add_argument("--subset_jsonl", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/cuhk_train_caption.jsonl", help="Optional jsonl to restrict clustering/DBSCAN to a subset (e.g., training set)")
    ap.add_argument("--subset_key", type=str, default="image_path", help="Key in subset_jsonl rows that stores image path")
    ap.add_argument("--batch_size", type=int, default=216)
    ap.add_argument("--top_k_neighbors", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.50)
    ap.add_argument("--min_samples", type=int, default=3)
    ap.add_argument("--output_dir", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/dualdistance3/hn350")

    ap.add_argument("--clip_model_path", type=str, default="/home/u2024218474/jupyterlab/PSPD/hf_models/openclip", help="Optional. If empty, infer from checkpoint dir/train_meta.json")
    ap.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "hf", "open_clip"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--rerank_distance", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/dualdistance3/final_distance.npy", help="Optional path to final_distance.npy for precomputed DBSCAN clustering")
    ap.add_argument("--dbscan_metric", type=str, default="precomputed", choices=["auto", "cosine", "precomputed"])
    ap.add_argument("--in_cluster_k", type=int, default=3, help="Preferred in-cluster neighbors per sample")
    ap.add_argument("--out_cluster_k", type=int, default=2, help="Preferred out-cluster neighbors per sample")
    ap.add_argument("--rerank_paths", type=str, default="/home/u2024218474/jupyterlab/PSPD/outputs/dualdistance3/rerank_paths.json", help="Optional json list of image paths used when computing rerank distance")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, backend, processor, oc_preprocess = load_model_for_inference(
        checkpoint=args.checkpoint,
        clip_model_path=args.clip_model_path,
        clip_backend=args.clip_backend,
        device=args.device,
    )

    if args.subset_jsonl:
        image_paths = collect_image_paths_from_jsonl(args.image_root, args.subset_jsonl, key=args.subset_key)
        print(f"[info] using subset_jsonl: {args.subset_jsonl} (images={len(image_paths)})")
    else:
        image_paths = collect_image_paths(args.image_root)
    features = extract_image_features(
        model=model,
        backend=backend,
        processor=processor,
        oc_preprocess=oc_preprocess,
        image_paths=image_paths,
        batch_size=args.batch_size,
        device=args.device,
    )

    use_precomputed = args.dbscan_metric == "precomputed" or (args.dbscan_metric == "auto" and bool(args.rerank_distance))

    rerank_dist = None
    if use_precomputed:
        if not args.rerank_distance:
            raise ValueError("--rerank_distance is required when using precomputed DBSCAN")
        rerank_dist = sanitize_precomputed_distance(np.load(args.rerank_distance).astype(np.float32, copy=False))
        if rerank_dist.shape != (len(image_paths), len(image_paths)):
            rerank_paths_path = args.rerank_paths
            if not rerank_paths_path:
                auto_paths = os.path.join(os.path.dirname(args.rerank_distance), "rerank_paths.json")
                if os.path.exists(auto_paths):
                    rerank_paths_path = auto_paths
            if not rerank_paths_path or not os.path.exists(rerank_paths_path):
                raise ValueError(
                    "rerank distance shape mismatch and no rerank paths file was provided. "
                    f"expected {(len(image_paths), len(image_paths))}, got {rerank_dist.shape}. "
                    "Pass --rerank_paths <json list of paths> (or place rerank_paths.json next to final_distance.npy)."
                )
            rerank_paths = load_rerank_paths(rerank_paths_path)
            image_paths, features, rerank_dist = align_to_rerank_subset(
                image_paths=image_paths,
                features=features,
                rerank_dist=rerank_dist,
                rerank_paths=rerank_paths,
            )
            rerank_dist = sanitize_precomputed_distance(rerank_dist)

    # STEP 1: features.pt (after optional rerank alignment)
    features_path = os.path.join(args.output_dir, "features.pt")
    torch.save({"image_paths": image_paths, "features": features.cpu()}, features_path)

    # STEP 2: neighbor mining
    # NOTE: neighbor mining always uses original feature-space similarity (cosine/IP),
    # while rerank distance is reserved for clustering when enabled.
    knn_indices = build_faiss_neighbors(features=features, top_k_neighbors=args.top_k_neighbors)

    # STEP 3: cluster info
    if rerank_dist is not None:
        cluster_labels = run_dbscan_precomputed(distance=rerank_dist, eps=args.eps, min_samples=args.min_samples)
    else:
        cluster_labels = run_dbscan(features=features, eps=args.eps, min_samples=args.min_samples)

    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
    num_clusters, noise_count, cluster_sizes = build_cluster_statistics(cluster_labels)

    neighbor_groups = build_neighbor_groups(
        image_paths=image_paths,
        knn_indices=knn_indices,
        top_k_neighbors=args.top_k_neighbors,
        cluster_labels=cluster_labels,
        in_cluster_k=max(0, args.in_cluster_k),
        out_cluster_k=max(0, args.out_cluster_k),
    )
    neighbor_groups_path = os.path.join(args.output_dir, "neighbor_groups.json")
    save_json(neighbor_groups_path, neighbor_groups)

    cluster_map = {
        "meta": {
            "num_samples": len(image_paths),
            "num_clusters": int(num_clusters),
            "noise_count": int(noise_count),
            "dbscan_metric": "precomputed" if rerank_dist is not None else "cosine",
            "neighbor_distance_metric": "cosine",
            "eps": float(args.eps),
            "min_samples": int(args.min_samples),
        },
        "samples": [
            {
                "image_path": path,
                "cluster_id": int(cid),
                "num_clusters": int(num_clusters),
            }
            for path, cid in zip(image_paths, cluster_labels.tolist())
        ],
    }
    cluster_map_path = os.path.join(args.output_dir, "cluster_map.json")
    save_json(cluster_map_path, cluster_map)

    cluster_sizes_path = os.path.join(args.output_dir, "cluster_sizes.json")
    save_json(
        cluster_sizes_path,
        {
            "num_clusters": int(num_clusters),
            "noise_count": int(noise_count),
            "cluster_sizes": cluster_sizes,
        },
    )

    print(f"[done] features: {features_path}")
    print(f"[done] neighbors: {neighbor_groups_path}")
    print(f"[done] clusters: {cluster_map_path}")
    print(f"[done] cluster sizes: {cluster_sizes_path}")


if __name__ == "__main__":
    main()
