import re
from typing import Dict
from preprocess.schema import LabelSchema

COLOR_SYNONYMS = {
    "black": ["black", "dark"],
    "white": ["white", "ivory", "off-white"],
    "gray": ["gray", "grey", "silver"],
    "red": ["red", "maroon", "burgundy", "crimson"],
    "blue": ["blue", "navy", "denim"],
    "green": ["green", "olive"],
    "brown": ["brown", "tan", "beige", "khaki"],
    "yellow": ["yellow", "gold", "orange"],
}

UPPER_TYPE_SYNONYMS = {
    "tshirt": ["tshirt", "t-shirt", "tee", "t shirt"],
    "shirt": ["shirt", "button-up", "button up", "blouse"],
    "hoodie_sweater": ["hoodie", "sweater", "sweatshirt", "pullover", "jumper", "knit"],
    "jacket_coat": ["jacket", "coat", "blazer", "suit", "windbreaker", "overcoat"],
    "dress": ["dress", "gown"],
    "other": ["vest", "tank", "tank top", "top"],
}

LOWER_TYPE_SYNONYMS = {
    "jeans": ["jeans", "denim"],
    "pants": ["pants", "trousers", "slacks"],
    "shorts": ["shorts"],
    "skirt": ["skirt"],
    "other": ["leggings"],
}

def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)

def _match_from_synonyms(text: str, syn_map: Dict[str, list[str]]) -> str:
    for label, syns in syn_map.items():
        if _contains_any(text, syns):
            return label
    return "unknown"

def _match_color(text: str) -> str:
    # pick the first hit; conservative
    for color, syns in COLOR_SYNONYMS.items():
        if _contains_any(text, syns):
            return color
    return "unknown"

def parse_slots_from_text(desc: str, schema: LabelSchema) -> Dict[str, str]:
    """
    Very simple parser for InstructBLIP natural language descriptions.
    Conservative: if not sure -> unknown.
    """
    t = (desc or "").strip().lower()
    # normalize punctuation
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t)

    slots = {
        "gender": "unknown",
        "upper_type": "unknown",
        "upper_color": "unknown",
        "lower_type": "unknown",
        "lower_color": "unknown",
    }

    # gender
    if "woman" in t or "female" in t or "girl" in t:
        slots["gender"] = "female"
    if "man" in t or "male" in t or "boy" in t:
        # if both appear (multi-person), keep unknown
        if slots["gender"] == "female":
            slots["gender"] = "unknown"
        else:
            slots["gender"] = "male"

    # dress check first (it affects lower)
    ut = _match_from_synonyms(t, UPPER_TYPE_SYNONYMS)
    if ut == "dress":
        slots["upper_type"] = "dress"
        slots["upper_color"] = _match_color(t)
        slots["lower_type"] = "unknown"
        slots["lower_color"] = "unknown"
        return slots

    slots["upper_type"] = ut
    slots["lower_type"] = _match_from_synonyms(t, LOWER_TYPE_SYNONYMS)

    # colors: try to infer upper/lower by local context
    # simple heuristics: look for "shirt/jacket/hoodie/sweater/dress" nearby; and "pants/jeans/shorts/skirt" nearby.
    def find_color_near(anchors: list[str]) -> str:
        for color, syns in COLOR_SYNONYMS.items():
            for s in syns:
                for a in anchors:
                    if f"{s} {a}" in t or f"{a} {s}" in t:
                        return color
        return "unknown"

    upper_anchors = ["shirt", "jacket", "coat", "hoodie", "sweater", "tshirt", "t-shirt", "tee", "blazer", "suit", "top"]
    lower_anchors = ["pants", "trousers", "jeans", "shorts", "skirt"]

    slots["upper_color"] = find_color_near(upper_anchors)
    slots["lower_color"] = find_color_near(lower_anchors)

    # fallback: if still unknown, allow a global color only if exactly one color appears in text
    if slots["upper_color"] == "unknown" or slots["lower_color"] == "unknown":
        present = []
        for c, syns in COLOR_SYNONYMS.items():
            if _contains_any(t, syns):
                present.append(c)
        present = list(dict.fromkeys(present))
        if len(present) == 1:
            if slots["upper_color"] == "unknown":
                slots["upper_color"] = present[0]
            if slots["lower_color"] == "unknown" and slots["lower_type"] != "unknown":
                slots["lower_color"] = present[0]

    # validate against schema (keep only allowed)
    if slots["upper_type"] not in schema.upper_types:
        slots["upper_type"] = "unknown"
    if slots["lower_type"] not in schema.lower_types:
        slots["lower_type"] = "unknown"
    if slots["upper_color"] not in schema.colors:
        slots["upper_color"] = "unknown"
    if slots["lower_color"] not in schema.colors:
        slots["lower_color"] = "unknown"
    if slots["gender"] not in schema.genders:
        slots["gender"] = "unknown"

    return slots