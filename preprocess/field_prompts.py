from __future__ import annotations

FIELDS = ["gender", "upper_type", "upper_color", "lower_type", "lower_color"]

_COMMON = (
    "You are a strict information extraction system.\n"
    "Focus on the MAIN person.\n"
    "Output EXACTLY one line. No extra text.\n"
    "Do NOT write a full sentence.\n"
    "Do NOT mention actions or scene.\n"
)

_BAN = (
    "NEVER output any of these tokens as the value: unknown, unsure, uncertain, n/a, none, null.\n"
    "You MUST output a best guess even if not confident.\n"
)

def blip_prompt(field: str, max_words: int = 3) -> str:
    mw = int(max_words)

    if field == "gender":
        return (
            _COMMON
            + "Task: classify gender.\n"
            + "Output format MUST be: gender: male OR gender: female\n"
            + "Output EXACTLY one line.\n"
            + "Do NOT output other words.\n"
            + "Now output:\n"
            + "gender:"
        )

    extra = ""
    if field == "lower_type":
        extra = "Lower type means BOTTOM garment type (pants/jeans/shorts/skirt). Not shoes/footwear.\n"

    return (
        _COMMON
        + f"Task: extract attribute '{field}'.\n"
        + f"Format MUST be: {field}: <value>\n"
        + f"<value> MUST be at most {mw} words.\n"
        + _BAN
        + extra
        + "Now output:\n"
        + f"{field}:"
    )