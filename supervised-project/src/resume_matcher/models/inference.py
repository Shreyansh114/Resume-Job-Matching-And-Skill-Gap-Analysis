"""Load a trained artifact and score an individual resume-job pair."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from resume_matcher.config import ARTIFACTS_DIR
from resume_matcher.features.feature_builder import build_pair_features
from resume_matcher.preprocessing.skill_extractor import load_skill_taxonomy


def load_model_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """Load a trusted local artifact. Never load joblib files from untrusted sources."""
    artifact_path = Path(path) if path else ARTIFACTS_DIR / "matching_model.joblib"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Model artifact has an unexpected format: {artifact_path}")
    return artifact


def predict_pair(
    pair: dict[str, Any],
    artifact: dict[str, Any],
    taxonomy: dict[str, list[str]] | None = None,
) -> float:
    """Return the model probability for the positive (match) class."""
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Model artifact is missing its feature column schema.")
    features = build_pair_features(
        pair,
        taxonomy or load_skill_taxonomy(),
        embedding_backend=artifact.get("embedding_backend", "tfidf"),
        embedding_model=artifact.get("embedding_model"),
    )
    row = pd.DataFrame(
        [[features.get(column, 0.0) for column in feature_columns]], columns=feature_columns
    )
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Trained model does not contain a positive class.")
    return float(model.predict_proba(row)[0, classes.index(1)])
