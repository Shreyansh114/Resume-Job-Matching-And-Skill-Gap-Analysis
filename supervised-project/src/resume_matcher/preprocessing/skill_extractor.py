"""Ontology-based skill extraction with canonical names and aliases."""

import json
import re
from pathlib import Path

from resume_matcher.config import TAXONOMY_DIR
from resume_matcher.preprocessing.text_cleaner import clean_text


def load_skill_taxonomy(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load a JSON mapping of canonical skill names to aliases."""
    taxonomy_path = Path(path) if path else TAXONOMY_DIR / "technical_skills.json"
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Skill taxonomy does not exist: {taxonomy_path}")
    with taxonomy_path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(aliases, list) for key, aliases in value.items()
    ):
        raise ValueError("Skill taxonomy must map skill names to lists of aliases.")
    return {key: [str(alias) for alias in aliases] for key, aliases in value.items()}


def extract_skills(text: str, taxonomy: dict[str, list[str]]) -> set[str]:
    """Return canonical skills found using case-insensitive, token-boundary matching."""
    normalized = clean_text(text)
    found: set[str] = set()
    for canonical, aliases in taxonomy.items():
        candidates = [canonical, *aliases]
        for candidate in candidates:
            term = clean_text(candidate)
            if term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
                found.add(canonical)
                break
    return found
