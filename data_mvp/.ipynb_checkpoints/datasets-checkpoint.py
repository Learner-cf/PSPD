import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from PIL import Image
from torch.utils.data import Dataset

@dataclass
class Sample:
    image_path: str
    pid: int
    split: str
    # optionally include original human captions if present
    captions: Optional[List[str]] = None

def load_cuhk(ann_file: str, image_root: str) -> List[Sample]:
    with open(ann_file, "r") as f:
        cap_list = json.load(f)
    samples = []
    for info in cap_list:
        fp = info["file_path"]
        pid = int(info["id"])
        split = info["split"]
        captions = info.get("captions", None)
        samples.append(Sample(image_path=os.path.join(image_root, fp), pid=pid, split=split, captions=captions))
    return samples

def load_icfg(ann_file: str, image_root: str) -> List[Sample]:
    with open(ann_file, "r") as f:
        cap_list = json.load(f)
    samples = []
    for info in cap_list:
        fp = info["file_path"]
        pid = int(info["id"])
        split = info.get("split", "train")
        captions = None
        # your snippet: caps = ann['captions'][0]
        if "captions" in info:
            captions = info["captions"]
        samples.append(Sample(image_path=os.path.join(image_root, fp), pid=pid, split=split, captions=captions))
    return samples

def load_rstp(ann_file: str, image_root: str) -> List[Sample]:
    with open(ann_file, "r") as f:
        cap_list = json.load(f)
    samples = []
    for info in cap_list:
        fp = info["img_path"]
        pid = int(info["id"])
        split = info["split"]
        captions = info.get("captions", None)
        samples.append(Sample(image_path=os.path.join(image_root, fp), pid=pid, split=split, captions=captions))
    return samples

class ImageOnlyDataset(Dataset):
    def __init__(self, samples: List[Sample], split: str):
        self.samples = [s for s in samples if s.split == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        return img, s.image_path, s.pid