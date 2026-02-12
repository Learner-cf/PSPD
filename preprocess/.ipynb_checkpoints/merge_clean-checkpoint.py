from typing import Dict, List
import numpy as np
from preprocess.schema import LabelSchema

SLOT_KEYS = ["gender", "upper_type", "upper_color", "lower_type", "lower_color"]

def majority_vote(vals: List[str]) -> str:
    # returns value with count>=2 else ""
    for v in set(vals):
        if vals.count(v) >= 2:
            return v
    return ""

def top1_label(probs: List[float], labels: List[str]) -> str:
    idx = int(np.argmax(np.array(probs, dtype=np.float32)))
    return labels[idx]

def top1_top2_margin(probs: List[float]) -> float:
    p = np.array(probs, dtype=np.float32)
    if p.size < 2:
        return 0.0
    # get top1 and top2
    idx = int(np.argmax(p))
    top1 = float(p[idx])
    p2 = p.copy()
    p2[idx] = -1.0
    top2 = float(np.max(p2))
    return top1 - top2

def merge_slots(
    slots_norm: List[Dict[str, str]],
    teacher_field_probs: Dict[str, List[float]],
    schema: LabelSchema,
    delta_field: Dict[str, float],
    conf_cover: Dict[str, float] | None = None,
) -> Dict[str, str]:
    """
    Merge 3 normalized slot candidates into one cleaned slot dict.

    Strategy:
    - For each field in {upper_color, lower_color, upper_type, lower_type}:
      * if teacher margin < delta_field[field] => unknown
      * else if majority vote agrees with teacher top1 => keep
      * else if teacher top1 prob >= conf_cover[field] => cover with teacher top1
      * else unknown
    - gender: keep only if majority vote (and not unknown), else unknown
    - dress rule applied at the end.
    """
    if conf_cover is None:
        conf_cover = {
            "upper_color": 0.70,
            "lower_color": 0.70,
            "upper_type": 0.70,
            "lower_type": 0.70,
        }

    out = {}

    # gender (no teacher cover, conservative)
    g_vals = [s.get("gender", "unknown") for s in slots_norm]
    g_vote = majority_vote(g_vals)
    out["gender"] = g_vote if (g_vote and g_vote != "unknown") else "unknown"

    # teacher-driven fields
    field_to_labels = {
        "upper_color": schema.colors,
        "lower_color": schema.colors,
        "upper_type": schema.upper_types,
        "lower_type": schema.lower_types,
    }

    for f, labels in field_to_labels.items():
        probs = teacher_field_probs.get(f, None)
        if probs is None:
            out[f] = "unknown"
            continue

        margin = top1_top2_margin(probs)
        if margin < float(delta_field.get(f, 1.0)):
            out[f] = "unknown"
            continue

        t1 = top1_label(probs, labels)
        conf = float(np.max(np.array(probs, dtype=np.float32)))

        vals = [s.get(f, "unknown") for s in slots_norm]
        v_vote = majority_vote(vals)

        if v_vote and v_vote != "unknown" and v_vote == t1:
            out[f] = v_vote
        elif conf >= float(conf_cover.get(f, 0.70)):
            out[f] = t1
        else:
            out[f] = "unknown"

    # dress rule
    if out.get("upper_type") == "dress":
        out["lower_type"] = "unknown"
        out["lower_color"] = "unknown"

    return out