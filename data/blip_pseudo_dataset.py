import json
from PIL import Image
from torch.utils.data import Dataset

from data.utils import pre_caption  # repo usually has pre_caption; if import fails tell me the correct path


class CUHK_BLIP_Pseudo_Train(Dataset):
    def __init__(self, transform, json_path, max_words=72, prompt=""):
        with open(json_path, "r", encoding="utf-8") as f:
            self.ann = json.load(f)
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt

        # map pid -> contiguous label
        self.pid2label = {}
        next_id = 0
        for a in self.ann:
            pid = a["pid"]
            if pid not in self.pid2label:
                self.pid2label[pid] = next_id
                next_id += 1

        self.num_classes = next_id

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        a = self.ann[idx]
        img_path = a["image_path"]
        pid = self.pid2label[a["pid"]]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        cap = self.prompt + pre_caption(a["caption"], self.max_words)

        # 4th item is a dummy placeholder; train.py will set it to None in itc_only mode
        gpt_dummy = ""
        return img, cap, pid, gpt_dummy