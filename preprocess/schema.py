from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class LabelSchema:
    colors: List[str]
    upper_types: List[str]
    lower_types: List[str]
    genders: List[str]

    def field_candidates(self) -> Dict[str, List[str]]:
        return {
            "upper_color": self.colors,
            "lower_color": self.colors,
            "upper_type": self.upper_types,
            "lower_type": self.lower_types,
        }

def build_schema(cfg) -> LabelSchema:
    labels = cfg["labels"]
    return LabelSchema(
        colors=list(labels["colors"]),
        upper_types=list(labels["upper_types"]),
        lower_types=list(labels["lower_types"]),
        genders=list(labels["genders"]),
    )