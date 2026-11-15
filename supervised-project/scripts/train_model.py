"""Train and evaluate the supervised resume-job matching model."""

import argparse

from resume_matcher.config import ARTIFACTS_DIR
from resume_matcher.data.loader import load_table
from resume_matcher.models.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled pair CSV or Parquet.")
    parser.add_argument("--output-dir", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()
    frame = load_table(args.input)
    if not hasattr(frame, "columns"):
        raise TypeError("Expected a complete DataFrame; chunked loading is not supported here.")
    metrics = train_model(frame, artifact_dir=args.output_dir)
    print(f"Trained model written to: {args.output_dir}")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "evaluation_rows"):
        print(f"{name}: {metrics[name]}")


if __name__ == "__main__":
    main()
