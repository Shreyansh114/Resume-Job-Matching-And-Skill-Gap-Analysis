# Configuration reference

The package reads configuration from environment variables at startup. Defaults below are research-friendly starting values, not validated production settings. Invalid thresholds, unsupported embedding backends, and non-positive model dimensions fail with explicit errors.

| Environment variable | Default | Purpose |
|---|---:|---|
| `RESUME_MATCHER_LABEL_THRESHOLD` | `0.5` | Converts a `match_score` into a positive/negative training label. |
| `RESUME_MATCHER_TEST_SIZE` | `0.2` | Fraction of the dataset reserved for holdout evaluation. |
| `RESUME_MATCHER_RANDOM_STATE` | `42` | Reproducible model and split random seed. |
| `RESUME_MATCHER_N_ESTIMATORS` | `300` | Number of trees in the random forest. |
| `RESUME_MATCHER_MAX_DEPTH` | `8` | Maximum tree depth. |
| `RESUME_MATCHER_MIN_SAMPLES_LEAF` | `2` | Minimum observations per leaf. |
| `RESUME_MATCHER_MAX_FEATURES` | `sqrt` | Number of features considered at each split (`sqrt`, `log2`, or a supported scikit-learn value). |
| `RESUME_MATCHER_EMBEDDING_BACKEND` | `tfidf` | `tfidf` or `sentence-transformers`. |
| `RESUME_MATCHER_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model identifier when the Sentence Transformers backend is enabled. |
| `RESUME_MATCHER_MATCH_THRESHOLD` | `0.7` | UI reference threshold; it does not change model training or the reported probability. |
| `RESUME_MATCHER_MAX_TEXT_CHARS` | `200000` | Maximum extracted resume text length retained by the document parser. |

`Settings.from_env()` is the single source of defaults and basic validation in `src/resume_matcher/config.py`. For a study, record the effective configuration alongside the dataset version, label definition, taxonomy version, and split strategy.

## Feature inputs

The pair feature schema is fixed in `features/feature_builder.py`. It combines pairwise text similarity, canonical skill overlap, optional years-of-experience signals, and optional education-match signals. Missing optional fields are encoded with availability indicators. Avoid adding non-job-related or sensitive personal attributes.
