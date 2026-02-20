import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def build_neighbor_groups(image_paths: List[str], knn_indices: np.ndarray, top_k_neighbors: int) -> List[Dict]:
    groups: List[Dict] = []
    n = len(image_paths)

    for i in range(n):
        neighbors: List[str] = []
        for j in knn_indices[i].tolist():
            if j == i:
                continue
            if 0 <= j < n:
                neighbors.append(image_paths[j])
            if len(neighbors) >= top_k_neighbors:
                break

        groups.append({
            "image_path": image_paths[i],
            "neighbors": neighbors,
        })
    return groups


def run_dbscan(features: torch.Tensor, eps: float, min_samples: int) -> np.ndarray:
    feats = features.cpu().numpy().astype(np.float32)
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = cluster.fit_predict(feats)
    return labels


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate hard-negative neighbor groups from image embeddings.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--top_k_neighbors", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--min_samples", type=int, default=4)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--clip_model_path", type=str, default="", help="Optional. If empty, infer from checkpoint dir/train_meta.json")
    ap.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "hf", "open_clip"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    # STEP 1: features.pt
    features_path = os.path.join(args.output_dir, "features.pt")
    torch.save({"image_paths": image_paths, "features": features.cpu()}, features_path)

    # STEP 2: hard neighbors via FAISS
    knn_indices = build_faiss_neighbors(features=features, top_k_neighbors=args.top_k_neighbors)
    neighbor_groups = build_neighbor_groups(
        image_paths=image_paths,
        knn_indices=knn_indices,
        top_k_neighbors=args.top_k_neighbors,
    )
    neighbor_groups_path = os.path.join(args.output_dir, "neighbor_groups.json")
    save_json(neighbor_groups_path, neighbor_groups)

    # STEP 3: optional cluster info
    cluster_labels = run_dbscan(features=features, eps=args.eps, min_samples=args.min_samples)
    cluster_map = [
        {"image_path": path, "cluster_id": int(cid)}
        for path, cid in zip(image_paths, cluster_labels.tolist())
    ]
    cluster_map_path = os.path.join(args.output_dir, "cluster_map.json")
    save_json(cluster_map_path, cluster_map)

    print(f"[done] features: {features_path}")
    print(f"[done] neighbors: {neighbor_groups_path}")
    print(f"[done] clusters: {cluster_map_path}")


if __name__ == "__main__":
    main()
