from typing import Dict
from preprocess.schema import LabelSchema

SLOT_KEYS = ["gender", "upper_type", "upper_color", "lower_type", "lower_color"]

def normalize_slots(slots: Dict, schema: LabelSchema) -> Dict[str, str]:
    out = {}
    for k in SLOT_KEYS:
        v = slots.get(k, "unknown")
        if v is None:
            v = "unknown"
        if not isinstance(v, str):
            v = str(v)
        v = v.strip().lower()

        # validate values
        if k == "gender":
            if v not in schema.genders:
                v = "unknown"
        elif k in ["upper_color", "lower_color"]:
            if v not in schema.colors and v != "unknown":
                v = "unknown"
        elif k == "upper_type":
            if v not in schema.upper_types and v != "unknown":
                v = "unknown"
        elif k == "lower_type":
            if v not in schema.lower_types and v != "unknown":
                v = "unknown"
        out[k] = v

    # enforce dress rule
    if out["upper_type"] == "dress":
        out["lower_type"] = "unknown"
        out["lower_color"] = "unknown"

    return out