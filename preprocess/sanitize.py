from __future__ import annotations

import re

BANNED_SUBSTRINGS = [
    "several people",
    "a group of",
    "group of",
    "crowd",
    "street",
    "sidewalk",
    "road",
    "background",
    "building",
    "archway",
    "letters",
    "sign",
    "traffic",
    "filmed",
    "photo of",
]

#动作/场景词：对 type 字段应当直接拒绝
ACTION_TOKENS = {
    "walking", "walk", "walks",
    "standing", "stand", "stands",
    "holding", "hold", "holds",
    "wearing", "wear", "wears",
    "carrying", "carry", "carries",
    "running", "run", "runs",
    "sitting", "sit", "sits",
    "looking", "look", "looks",
    "image", "photo", "picture",
}

BANNED_VERB_TOKENS = {
    "is", "are", "was", "were",
    *ACTION_TOKENS,
}

UPPER_TYPE_FORBIDDEN_IN_LOWER = {
    "hoodie", "jacket", "coat", "blazer", "shirt", "tshirt", "t-shirt",
    "sweater", "sweatshirt", "dress", "vest",
}
LOWER_TYPE_FORBIDDEN_IN_UPPER = {
    "jeans", "pants", "trousers", "shorts", "skirt", "leggings",
}

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sanitize_value(field: str, value: str, max_words: int = 3, max_chars: int = 30) -> str:
    t = _norm(value)
    if not t:
        return "unknown"
    if len(t) > 160:
        return "unknown"

    for b in BANNED_SUBSTRINGS:
        if b in t:
            return "unknown"

    words = t.split()
    if not words:
        return "unknown"

    # 对 type 字段：如果出现动作词/图像词，直接拒绝
    if field in {"upper_type", "lower_type"}:
        if any(w in ACTION_TOKENS for w in words):
            return "unknown"

    verb_hits = sum(1 for w in words if w in BANNED_VERB_TOKENS)
    if verb_hits >= 1 and len(words) >= 5:
        return "unknown"

    words = words[: int(max_words)]
    t2 = " ".join(words).strip()
    if not t2 or t2 == "unknown":
        return "unknown"
    if len(t2) > int(max_chars):
        return "unknown"

    if field == "gender":
        if any(x in t2 for x in ["male", "man", "boy"]):
            return "male"
        if any(x in t2 for x in ["female", "woman", "girl"]):
            return "female"
        return "unknown"

    if field == "lower_type":
        if any(x in t2 for x in UPPER_TYPE_FORBIDDEN_IN_LOWER):
            return "unknown"
    if field == "upper_type":
        if any(x in t2 for x in LOWER_TYPE_FORBIDDEN_IN_UPPER):
            return "unknown"

    if field in {"upper_color", "lower_color"}:
        clothing_tail = {"jeans", "pants", "trousers", "shorts", "skirt", "hoodie", "jacket", "coat", "shirt", "dress"}
        w = t2.split()
        if w and w[-1] in clothing_tail:
            w = w[:-1]
        if not w:
            return "unknown"
        w = w[: min(2, len(w))]
        t2 = " ".join(w).strip()
        if not t2:
            return "unknown"

        # 对 color 字段：如果输出成 "image/photo" 也拒绝
        if t2 in {"image", "photo", "picture"}:
            return "unknown"

    return t2