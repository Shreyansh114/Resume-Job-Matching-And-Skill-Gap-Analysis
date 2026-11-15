"""Validate a labeled pair CSV before training."""

import argparse
import sys

from resume_matcher.data.loader import load_table
from resume_matcher.data.validator import validate_pair_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the CSV or Parquet pair table.")
    args = parser.parse_args()
    frame = load_table(args.input)
    if not hasattr(frame, "columns"):
        raise TypeError("Expected a complete DataFrame; chunked loading is not supported here.")
    report = validate_pair_frame(frame, require_labels=True)
    print(f"Rows checked: {report.rows}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if not report.is_valid:
        return 1
    print("Dataset schema is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
