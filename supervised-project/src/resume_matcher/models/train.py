"""Train, evaluate, and persist a supervised resume-job matching model."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from resume_matcher.config import ARTIFACTS_DIR, settings
from resume_matcher.data.validator import validate_pair_frame
from resume_matcher.features.feature_builder import FEATURE_COLUMNS, build_feature_matrix
from resume_matcher.models.evaluate import binary_metrics
from resume_matcher.models.explainability import forest_feature_importance
from resume_matcher.preprocessing.skill_extractor import load_skill_taxonomy


def _uses_resume_groups(frame: pd.DataFrame) -> bool:
    return "resume_id" in frame.columns and frame["resume_id"].dropna().nunique() >= 2


def _labels(frame: pd.DataFrame) -> pd.Series:
    if "label" in frame.columns:
        labels = pd.to_numeric(frame["label"], errors="coerce")
    elif "match_score" in frame.columns:
        scores = pd.to_numeric(frame["match_score"], errors="coerce")
        if scores.isna().any():
            raise ValueError("'match_score' must be numeric and contain no missing values.")
        labels = (scores >= settings.label_threshold).astype("float")
    else:
        raise ValueError("Training data must include 'label' or 'match_score'.")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Labels must be complete and binary after applying the score threshold.")
    return labels.astype(int)


def _split_indices(frame: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if labels.nunique() < 2:
        raise ValueError("Training requires both positive and negative labeled pairs.")
    if len(frame) < 10:
        raise ValueError("At least 10 labeled pairs are required for a train/test split.")
    if _uses_resume_groups(frame):
        groups = (
            frame["resume_id"]
            .astype("string")
            .fillna(pd.Series([f"missing-{i}" for i in range(len(frame))], index=frame.index))
        )
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=settings.test_size, random_state=settings.random_state
        )
        train_idx, test_idx = next(splitter.split(frame, labels, groups))
        if labels.iloc[train_idx].nunique() < 2:
            raise ValueError(
                "The resume-grouped training split has only one class. Add more diverse labeled "
                "resumes or review the labels."
            )
        return train_idx, test_idx
    classes, counts = np.unique(labels, return_counts=True)
    stratify = labels if len(classes) == 2 and counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)),
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=stratify,
    )
    return train_idx, test_idx


def train_model(
    frame: pd.DataFrame,
    *,
    artifact_dir: str | Path = ARTIFACTS_DIR,
    taxonomy: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Fit a random forest on labeled pairs and write model and evaluation artifacts."""
    report = validate_pair_frame(frame, require_labels=True)
    if not report.is_valid:
        raise ValueError("Invalid training data: " + "; ".join(report.errors))
    labels = _labels(frame)
    split = _split_indices(frame, labels)
    train_idx, test_idx = split
    skill_taxonomy = taxonomy if taxonomy is not None else load_skill_taxonomy()
    features = build_feature_matrix(
        frame.to_dict(orient="records"),
        skill_taxonomy,
        embedding_backend=settings.embedding_backend,
        embedding_model=settings.embedding_model,
    ).fillna(0.0)

    model = RandomForestClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        min_samples_leaf=settings.min_samples_leaf,
        max_features=settings.max_features,
        class_weight="balanced",
        random_state=settings.random_state,
        n_jobs=-1,
    )
    model.fit(features.iloc[train_idx], labels.iloc[train_idx])
    predictions = model.predict(features.iloc[test_idx])
    probabilities = model.predict_proba(features.iloc[test_idx])[:, list(model.classes_).index(1)]
    metrics = binary_metrics(labels.iloc[test_idx].to_numpy(), predictions, probabilities)
    metrics.update(
        {
            "training_rows": int(len(train_idx)),
            "feature_columns": list(FEATURE_COLUMNS),
            "split_strategy": "resume_id_grouped"
            if _uses_resume_groups(frame)
            else "stratified_random",
            "label_threshold": settings.label_threshold,
            "embedding_backend": settings.embedding_backend,
            "feature_importance": forest_feature_importance(model, list(FEATURE_COLUMNS)),
        }
    )

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(FEATURE_COLUMNS),
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "label_threshold": settings.label_threshold,
            "match_threshold": settings.match_threshold,
        },
        output / "matching_model.joblib",
    )
    with (output / "metrics_report.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics
