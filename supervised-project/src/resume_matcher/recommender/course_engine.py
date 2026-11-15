"""Map missing skills to curated learning resources."""

import json
from pathlib import Path

from resume_matcher.config import TAXONOMY_DIR


def load_learning_resources(path: str | Path | None = None) -> dict[str, list[dict[str, str]]]:
    """Load skill-to-resource recommendations."""
    resource_path = Path(path) if path else TAXONOMY_DIR / "learning_resources.json"
    if not resource_path.is_file():
        raise FileNotFoundError(f"Learning resource catalog does not exist: {resource_path}")
    with resource_path.open(encoding="utf-8") as file:
        resources = json.load(file)
    if not isinstance(resources, dict):
        raise ValueError("Learning resource catalog must be a JSON object.")
    return resources


def recommend_resources(
    missing_skills: list[str], resources: dict[str, list[dict[str, str]]]
) -> dict[str, list[dict[str, str]]]:
    """Return only catalog entries mapped to the requested skills."""
    return {skill: resources.get(skill, []) for skill in missing_skills}
