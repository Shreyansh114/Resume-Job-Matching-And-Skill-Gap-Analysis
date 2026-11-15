"""Tabular data loading helpers."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


def load_table(
    path: str | Path, chunksize: int | None = None
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load a CSV or Parquet file, optionally yielding CSV chunks."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        if chunksize is not None and chunksize < 1:
            raise ValueError("chunksize must be a positive integer.")
        return pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        if chunksize is not None:
            raise ValueError("Chunked loading is supported for CSV files only.")
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported dataset extension '{suffix}'. Use CSV or Parquet.")
