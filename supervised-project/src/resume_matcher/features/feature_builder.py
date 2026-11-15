"""Build reproducible candidate-job pair features."""

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from resume_matcher.features.embeddings import pair_text_similarity
from resume_matcher.features.similarity_metrics import (
    dice_similarity,
    jaccard_similarity,
    overlap_coefficient,
)
from resume_matcher.preprocessing.skill_extractor import extract_skills

FEATURE_COLUMNS = (
    "text_similarity",
    "skill_jaccard",
    "skill_overlap",
    "skill_dice",
    "required_skill_count",
    "matched_skill_count",
    "missing_skill_count",
    "experience_gap",
    "experience_available",
    "education_match",
    "education_available",
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def build_pair_features(
    pair: Mapping[str, Any],
    taxonomy: dict[str, list[str]],
    *,
    embedding_backend: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, float]:
    """Build the same numeric feature schema used by training and interactive inference."""
    resume_text = _text(pair.get("resume_text"))
    job_text = _text(pair.get("job_description"))
    resume_skills = extract_skills(resume_text, taxonomy)
    job_skills = extract_skills(job_text, taxonomy)
    matched = resume_skills & job_skills
    resume_years = _number(pair.get("resume_years_experience"))
    job_years = _number(pair.get("job_years_experience"))
    experience_available = resume_years is not None and job_years is not None
    resume_education = _text(pair.get("resume_education")).strip().casefold()
    job_education = _text(pair.get("job_education")).strip().casefold()
    education_available = bool(resume_education and job_education)

    return {
        "text_similarity": pair_text_similarity(
            resume_text,
            job_text,
            backend=embedding_backend,
            model_name=embedding_model,
        ),
        "skill_jaccard": jaccard_similarity(resume_skills, job_skills),
        "skill_overlap": overlap_coefficient(resume_skills, job_skills),
        "skill_dice": dice_similarity(resume_skills, job_skills),
        "required_skill_count": float(len(job_skills)),
        "matched_skill_count": float(len(matched)),
        "missing_skill_count": float(len(job_skills - resume_skills)),
        "experience_gap": (min(resume_years - job_years, 20.0) if experience_available else 0.0),
        "experience_available": float(experience_available),
        "education_match": float(education_available and resume_education == job_education),
        "education_available": float(education_available),
    }


def build_feature_matrix(
    pairs: list[Mapping[str, Any]],
    taxonomy: dict[str, list[str]],
    *,
    embedding_backend: str | None = None,
    embedding_model: str | None = None,
):
    """Convert labeled or unlabeled pair records into a feature DataFrame."""
    import pandas as pd

    rows = [
        build_pair_features(
            pair,
            taxonomy,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        for pair in pairs
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
