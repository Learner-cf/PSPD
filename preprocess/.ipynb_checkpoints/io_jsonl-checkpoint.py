import json
import os
from typing import Dict, Any

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_jsonable(x: Any) -> Any:
    """
    Convert numpy / torch scalars and other non-JSONable objects
    into plain Python types recursively.
    """
    # basic
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # numpy scalars
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass

    # torch scalars
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return x.item()
            return x.detach().cpu().tolist()
    except Exception:
        pass

    # dict
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # fallback: stringify (last resort)
    return str(x)

class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self.f = open(path, "a", encoding="utf-8")

    def write(self, obj: Dict):
        obj = to_jsonable(obj)
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

def load_processed_set(jsonl_path: str, key: str = "image_path") -> set:
    if not os.path.exists(jsonl_path):
        return set()
    seen = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                seen.add(obj.get(key))
            except Exception:
                continue
    return seen