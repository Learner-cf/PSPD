import argparse
import json
import os
from typing import Dict, Iterable, List


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_split(split: str) -> str:
    return (split or "").strip().lower()


def _ensure_list_str(x) -> List[str]:
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def _join_image_path(image_root: str, file_path: str) -> str:
    if os.path.isabs(file_path) or not image_root:
        return file_path
    return os.path.join(image_root, file_path)


def iter_rows(dataset: str, raw_json: str, image_root: str, use_first_caption_for_train: bool) -> Iterable[Dict]:
    dataset = dataset.lower().strip()
    data = _load_json(raw_json)

    if dataset == "icfg" and isinstance(data, dict):
        merged: List[Dict] = []
        for key in ("train", "val", "test"):
            if isinstance(data.get(key), list):
                for row in data[key]:
                    x = dict(row)
                    x["split"] = key if key != "val" else "test"
                    merged.append(x)
        data = merged

    if not isinstance(data, list):
        raise ValueError(f"Unsupported annotation format for dataset={dataset}: expected list, got {type(data)}")

    for row in data:
        split = _normalize_split(str(row.get("split", "")))
        pid = row.get("id", row.get("pid", row.get("person_id", None)))
        if pid is None:
            continue

        file_path = row.get("file_path") or row.get("img_path") or row.get("image_path")
        if not file_path:
            continue

        caps = _ensure_list_str(row.get("captions", []))

        if use_first_caption_for_train and split == "train" and len(caps) > 0:
            caps = [caps[0]]

        if not caps:
            caps = ["a person"]

        yield {
            "image_path": _join_image_path(image_root=image_root, file_path=str(file_path)),
            "pid": int(pid) if str(pid).isdigit() else str(pid),
            "split": split,
            "captions": caps,
            "caption": caps[0],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate retrieval text JSONL from dataset raw annotations.")
    ap.add_argument("--dataset", type=str, required=True, choices=["cuhk", "icfg", "rstp"])
    ap.add_argument("--raw_json", type=str, required=True, help="Path to raw annotation json")
    ap.add_argument("--image_root", type=str, default="", help="Optional image root for joining file_path")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--use_first_caption_for_train", action="store_true", help="Match existing data split behavior")
    args = ap.parse_args()

    rows = list(iter_rows(args.dataset, args.raw_json, args.image_root, args.use_first_caption_for_train))
    target_split = _normalize_split(args.split)
    if target_split != "all":
        rows = [r for r in rows if r.get("split") == target_split]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
