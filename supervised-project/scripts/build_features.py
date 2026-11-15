"""Normalize labeled pairs and write a reusable feature matrix."""

import argparse
from pathlib import Path

from resume_matcher.config import PROCESSED_DATA_DIR, settings
from resume_matcher.data.loader import load_table
from resume_matcher.data.validator import validate_pair_frame
from resume_matcher.features.feature_builder import build_feature_matrix
from resume_matcher.preprocessing.skill_extractor import load_skill_taxonomy
from resume_matcher.preprocessing.text_cleaner import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normalized labeled pair CSV or Parquet.")
    parser.add_argument("--output-dir", default=str(PROCESSED_DATA_DIR))
    args = parser.parse_args()
    frame = load_table(args.input)
    if not hasattr(frame, "columns"):
        raise TypeError("Expected a complete DataFrame; chunked loading is not supported here.")
    report = validate_pair_frame(frame, require_labels=True)
    if not report.is_valid:
        raise ValueError("Invalid pair data: " + "; ".join(report.errors))
    if "match_score" in frame.columns and "label" not in frame.columns:
        frame["label"] = (frame["match_score"].astype(float) >= settings.label_threshold).astype(
            int
        )
    frame["resume_text"] = frame["resume_text"].map(clean_text)
    frame["job_description"] = frame["job_description"].map(clean_text)
    taxonomy = load_skill_taxonomy()
    features = build_feature_matrix(
        frame.to_dict(orient="records"),
        taxonomy,
        embedding_backend=settings.embedding_backend,
        embedding_model=settings.embedding_model,
    )
    metadata_columns = [
        column for column in ("resume_id", "job_id", "label", "match_score") if column in frame
    ]
    feature_matrix = frame[metadata_columns].reset_index(drop=True).join(features)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "cleaned_pairs.parquet", index=False)
    feature_matrix.to_parquet(output_dir / "feature_matrix.parquet", index=False)
    print(f"Processed {len(frame):,} pairs.")
    print(f"Cleaned pairs: {output_dir / 'cleaned_pairs.parquet'}")
    print(f"Feature matrix: {output_dir / 'feature_matrix.parquet'}")


if __name__ == "__main__":
    main()
