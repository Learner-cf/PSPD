from typing import Dict

def compose_caption(slots: Dict[str, str]) -> str:
    """
    Simple default single-sentence composer.
    Replace this later with your own template library.
    """
    g = slots.get("gender", "unknown")
    ut = slots.get("upper_type", "unknown")
    uc = slots.get("upper_color", "unknown")
    lt = slots.get("lower_type", "unknown")
    lc = slots.get("lower_color", "unknown")

    # dress special-case
    if ut == "dress":
        if uc != "unknown":
            return f"a person wearing a {uc} dress"
        return "a person wearing a dress"

    parts = ["a person"]
    # optionally gender
    if g in ["male", "female"]:
        parts = [f"a {('man' if g=='male' else 'woman')}"]

    # upper phrase
    if uc != "unknown" and ut != "unknown":
        upper = f"wearing a {uc} {ut}"
    elif ut != "unknown":
        upper = f"wearing a {ut}"
    elif uc != "unknown":
        upper = f"wearing a {uc} top"
    else:
        upper = "wearing upper clothing"

    # lower phrase
    if lc != "unknown" and lt != "unknown":
        lower = f"and {lc} {lt}"
    elif lt != "unknown":
        lower = f"and {lt}"
    elif lc != "unknown":
        lower = f"and {lc} lower clothing"
    else:
        lower = ""

    cap = " ".join([parts[0], upper, lower]).strip()
    # normalize spacing
    cap = " ".join(cap.split())
    return cap