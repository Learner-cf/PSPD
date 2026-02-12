from typing import Dict, List, Tuple
import numpy as np

def prob_margin(probs: List[float]) -> float:
    p = np.array(probs, dtype=np.float32)
    if p.size < 2:
        return 0.0
    top2 = np.partition(p, -2)[-2:]
    top1 = float(np.max(top2))
    top2v = float(np.min(top2))
    return top1 - top2v

def calibrate_threshold(margins: List[float], target_pass_rate: float) -> float:
    """
    Choose delta so that approximately target_pass_rate samples pass: margin >= delta.
    """
    if len(margins) == 0:
        return 1.0
    margins = np.array(margins, dtype=np.float32)
    # delta is the (1-target) quantile
    q = np.quantile(margins, 1.0 - target_pass_rate)
    return float(q)

def compute_field_margins(probs_list: List[List[float]]) -> List[float]:
    return [prob_margin(p) for p in probs_list]