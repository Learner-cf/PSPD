from __future__ import annotations
from typing import Tuple

def parse_field_line(line: str, expected_field: str) -> Tuple[str, str]:
    """
    Parse a line like 'upper_type: hoodie' -> ('upper_type','hoodie').
    If no ':' found, treat whole line as value (still return expected_field).
    """
    s = (line or "").strip()
    if not s:
        return expected_field, "unknown"
    if ":" not in s:
        return expected_field, s.strip().lower()

    k, v = s.split(":", 1)
    v = (v or "").strip().lower()
    if not v:
        v = "unknown"
    return expected_field, v