from typing import List, Dict, Tuple
import numpy as np

def rank_triplet(scores: List[float], delta_global: float = 0.03) -> Tuple[List[int], bool, Dict[str, float]]:
    """
    Given 3 teacher scores, return:
    - rank_order: indices sorted by score desc
    - enabled: whether to apply rank loss (top1-top3 >= delta_global)
    - weights: normalized w12,w23,w13 based on score gaps
    """
    assert len(scores) == 3
    s = np.array(scores, dtype=np.float32)
    order = list(np.argsort(-s))  # desc
    a, b, c = order[0], order[1], order[2]
    m = float(s[a] - s[c])
    enabled = m >= float(delta_global)

    w12 = max(0.0, float(s[a] - s[b]))
    w23 = max(0.0, float(s[b] - s[c]))
    w13 = max(0.0, float(s[a] - s[c]))
    denom = w12 + w23 + w13 + 1e-8
    weights = {"w12": w12 / denom, "w23": w23 / denom, "w13": w13 / denom}
    return order, enabled, weights