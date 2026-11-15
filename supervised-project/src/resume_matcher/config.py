"""Central project paths and model settings."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TAXONOMY_DIR = DATA_DIR / "taxonomy"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class Settings:
    """Validated defaults for feature generation and supervised training."""

    label_threshold: float = 0.5
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    embedding_backend: str = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    match_threshold: float = 0.7
    max_text_chars: int = 200_000

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from documented environment variables."""
        settings = cls(
            label_threshold=float(os.getenv("RESUME_MATCHER_LABEL_THRESHOLD", "0.5")),
            test_size=float(os.getenv("RESUME_MATCHER_TEST_SIZE", "0.2")),
            random_state=int(os.getenv("RESUME_MATCHER_RANDOM_STATE", "42")),
            n_estimators=int(os.getenv("RESUME_MATCHER_N_ESTIMATORS", "300")),
            max_depth=int(os.getenv("RESUME_MATCHER_MAX_DEPTH", "8")),
            min_samples_leaf=int(os.getenv("RESUME_MATCHER_MIN_SAMPLES_LEAF", "2")),
            max_features=os.getenv("RESUME_MATCHER_MAX_FEATURES", "sqrt"),
            embedding_backend=os.getenv("RESUME_MATCHER_EMBEDDING_BACKEND", "tfidf").lower(),
            embedding_model=os.getenv(
                "RESUME_MATCHER_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            match_threshold=float(os.getenv("RESUME_MATCHER_MATCH_THRESHOLD", "0.7")),
            max_text_chars=int(os.getenv("RESUME_MATCHER_MAX_TEXT_CHARS", "200000")),
        )
        if not 0 < settings.test_size < 1:
            raise ValueError("RESUME_MATCHER_TEST_SIZE must be between 0 and 1.")
        if not 0 <= settings.label_threshold <= 1:
            raise ValueError("RESUME_MATCHER_LABEL_THRESHOLD must be between 0 and 1.")
        if not 0 <= settings.match_threshold <= 1:
            raise ValueError("RESUME_MATCHER_MATCH_THRESHOLD must be between 0 and 1.")
        if settings.embedding_backend not in {"tfidf", "sentence-transformers"}:
            raise ValueError(
                "RESUME_MATCHER_EMBEDDING_BACKEND must be 'tfidf' or 'sentence-transformers'."
            )
        if settings.n_estimators < 1 or settings.min_samples_leaf < 1:
            raise ValueError("Random forest estimator and leaf counts must be positive.")
        if settings.max_depth < 1:
            raise ValueError("RESUME_MATCHER_MAX_DEPTH must be positive.")
        if settings.random_state < 0:
            raise ValueError("RESUME_MATCHER_RANDOM_STATE must be non-negative.")
        if settings.max_text_chars < 1:
            raise ValueError("RESUME_MATCHER_MAX_TEXT_CHARS must be positive.")
        return settings


settings = Settings.from_env()
