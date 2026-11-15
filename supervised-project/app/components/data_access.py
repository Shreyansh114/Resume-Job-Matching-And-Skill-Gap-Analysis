"""Read-only access to local dashboard inputs and generated artifacts."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from resume_matcher.config import ARTIFACTS_DIR, DATA_DIR, RAW_DATA_DIR, TAXONOMY_DIR
from resume_matcher.data.loader import load_table
from resume_matcher.preprocessing.skill_extractor import load_skill_taxonomy

PROJECT_ROOT = DATA_DIR.parent
PAIR_CANDIDATES = (
    DATA_DIR / "processed" / "cleaned_pairs.parquet",
    DATA_DIR / "processed" / "feature_matrix.parquet",
    RAW_DATA_DIR / "matches.csv",
)


def _existing_pair_path() -> Path | None:
    return next((path for path in PAIR_CANDIDATES if path.is_file()), None)


@st.cache_data(show_spinner=False)
def _load_table_cached(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    result = load_table(path)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Dashboard data loading expected a complete table.")
    return result


def load_pair_data() -> tuple[pd.DataFrame | None, Path | None]:
    path = _existing_pair_path()
    if path is None:
        return None, None
    return _load_table_cached(str(path), path.stat().st_mtime), path


def load_metrics() -> dict[str, Any] | None:
    path = ARTIFACTS_DIR / "metrics_report.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return metrics


def load_local_model() -> dict[str, Any] | None:
    path = ARTIFACTS_DIR / "matching_model.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)


def load_taxonomy() -> dict[str, list[str]]:
    return load_skill_taxonomy(TAXONOMY_DIR / "technical_skills.json")


def data_inventory() -> list[dict[str, str]]:
    tracked = [
        ("Raw resumes", RAW_DATA_DIR / "cv.csv"),
        ("Raw job postings", RAW_DATA_DIR / "job.csv"),
        ("Labeled pairs", RAW_DATA_DIR / "matches.csv"),
        ("Processed pairs", DATA_DIR / "processed" / "cleaned_pairs.parquet"),
        ("Feature matrix", DATA_DIR / "processed" / "feature_matrix.parquet"),
        ("Trained model", ARTIFACTS_DIR / "matching_model.joblib"),
        ("Evaluation report", ARTIFACTS_DIR / "metrics_report.json"),
    ]
    return [
        {
            "Asset": label,
            "Status": "Available" if path.is_file() else "Not added yet",
            "Size": f"{path.stat().st_size / (1024 * 1024):.1f} MB" if path.is_file() else "—",
            "Location": str(path.relative_to(PROJECT_ROOT)),
        }
        for label, path in tracked
    ]
