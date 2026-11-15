"""Schema and data-quality validation for matching datasets."""

from dataclasses import dataclass, field

import pandas as pd

PAIR_REQUIRED_COLUMNS = ("resume_text", "job_description")
PAIR_OPTIONAL_COLUMNS = (
    "label",
    "match_score",
    "resume_id",
    "job_id",
    "resume_years_experience",
    "job_years_experience",
    "resume_education",
    "job_education",
)


@dataclass(frozen=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_pair_frame(frame: pd.DataFrame, require_labels: bool = False) -> ValidationReport:
    """Validate candidate-job pair fields without mutating the input frame."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = [column for column in PAIR_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return ValidationReport(errors=errors, warnings=warnings, rows=len(frame))

    if frame.empty:
        errors.append("Dataset contains no rows.")
    if frame[list(PAIR_REQUIRED_COLUMNS)].isna().all(axis=1).any():
        warnings.append("Some rows have both text fields missing.")
    for column in PAIR_REQUIRED_COLUMNS:
        if not frame[column].map(lambda value: isinstance(value, str)).any():
            errors.append(f"Column '{column}' must contain text.")
        elif frame[column].map(lambda value: not isinstance(value, str) or not value.strip()).any():
            warnings.append(f"Column '{column}' has missing or blank text values.")

    label_columns = [column for column in ("label", "match_score") if column in frame.columns]
    if require_labels and not label_columns:
        errors.append("Supervised training requires a 'label' or 'match_score' column.")
    if len(label_columns) == 2:
        warnings.append("Both 'label' and 'match_score' exist; training uses 'label'.")
    if "label" in frame.columns:
        labels = frame["label"].dropna()
        if not labels.empty and not labels.isin([0, 1, False, True]).all():
            errors.append("'label' must contain binary values 0 or 1.")
    if "match_score" in frame.columns:
        scores = pd.to_numeric(frame["match_score"], errors="coerce")
        if scores.notna().sum() != frame["match_score"].notna().sum():
            errors.append("'match_score' must be numeric.")
        elif scores.isna().any():
            errors.append("'match_score' cannot contain missing values.")
        elif not scores.dropna().between(0, 1).all():
            errors.append("'match_score' values must be between 0 and 1.")
    for column in ("resume_years_experience", "job_years_experience"):
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if (numeric.dropna() < 0).any():
                errors.append(f"'{column}' cannot contain negative values.")

    return ValidationReport(errors=errors, warnings=warnings, rows=len(frame))
