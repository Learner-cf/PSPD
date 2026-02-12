def system_prompt() -> str:
    return (
        "You are a careful vision assistant for person clothing attribute extraction.\n"
        "Return ONLY a valid JSON object (no markdown, no code block, no explanations).\n"
        "All keys must appear exactly as specified.\n"
        "All values must be lowercase strings.\n"
        "If an attribute is not clearly visible, set it to \"unknown\".\n"
        "Do not mention background or scene.\n"
    )

def user_prompt() -> str:
    # Small closed sets to improve stability and reduce normalization work.
    return (
        "Extract clothing attributes for the person in the image.\n\n"
        "Output a single JSON object with exactly these keys:\n"
        "{\n"
        "  \"gender\": \"...\",\n"
        "  \"upper_type\": \"...\",\n"
        "  \"upper_color\": \"...\",\n"
        "  \"lower_type\": \"...\",\n"
        "  \"lower_color\": \"...\"\n"
        "}\n\n"
        "Constraints:\n"
        "- gender must be one of: \"male\", \"female\", \"unknown\"\n"
        "- upper_type must be one of: \"tshirt\", \"shirt\", \"hoodie_sweater\", \"jacket_coat\", \"dress\", \"other\", \"unknown\"\n"
        "- upper_color must be one of: \"black\", \"white\", \"gray\", \"red\", \"blue\", \"green\", \"brown\", \"yellow\", \"unknown\"\n"
        "- lower_type must be one of: \"pants\", \"jeans\", \"shorts\", \"skirt\", \"other\", \"unknown\"\n"
        "- lower_color must be one of: \"black\", \"white\", \"gray\", \"red\", \"blue\", \"green\", \"brown\", \"yellow\", \"unknown\"\n\n"
        "Rules:\n"
        "- If upper_type is \"dress\", set lower_type and lower_color to \"unknown\".\n"
        "- Choose the dominant clothing color (ignore small logos/patterns).\n"
        "- If multiple colors exist and none is dominant, use \"unknown\".\n"
        "Return JSON only.\n"
    )