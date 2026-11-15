# Multimodal Resume–Job Matching and Skill Gap Analysis

A research-ready, extensible Python project for matching resumes to job descriptions, evaluating supervised matching models, and identifying actionable skill gaps.

> **Data and model status:** This repository starts without private datasets or a trained model. The dashboard reports missing inputs honestly. The interactive match page offers a clearly labeled TF-IDF/skill-overlap baseline until a trained model is available; it is not presented as a trained prediction.

## Features

- CSV and Parquet ingestion with schema and data-quality checks.
- PDF, DOCX, and plain-text resume extraction.
- Configurable skill taxonomy with aliases and resource recommendations.
- Pair-level feature engineering for text similarity, skill overlap, experience, and education.
- Supervised random-forest training, evaluation, and saved model artifacts.
- Streamlit resume-fit dashboard with a transparent ATS-style demo rubric, skill/keyword gaps, document checks, and model/data evaluation.
- Unit tests for preprocessing, features, validation, and inference.

The ATS-style scan is an original, explainable heuristic demo. It is not a real ATS parser,
Jobscan, a trained model, or a prediction of interview or job performance. Resume text is
processed in memory by the local app; exported scan summaries exclude the resume and job text.

## Quick start

```powershell
cd supervised-project
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,documents]"
streamlit run app\app.py
```

For PDF/DOCX parsing, install the `documents` extra. For Sentence Transformers embeddings, install the optional `semantic` extra and set `RESUME_MATCHER_EMBEDDING_BACKEND=sentence-transformers`. The first use may download the configured model.

## Data contracts

Keep original datasets in `data/raw/`; do not commit personal or confidential resumes. The trainer expects a labeled pair dataset (default: `data/raw/matches.csv`) with:

| Column | Required | Meaning |
|---|---:|---|
| `resume_text` | yes | Resume text or extracted content |
| `job_description` | yes | Job description text |
| `label` or `match_score` | yes | Binary label, or numeric score in `[0, 1]` |
| `resume_id`, `job_id` | no | Pair/group identifiers |
| `resume_years_experience`, `job_years_experience` | no | Numeric experience signals |
| `resume_education`, `job_education` | no | Education signals |

`label` is used as-is after validating it is binary. If only `match_score` is present, scores at or above `RESUME_MATCHER_LABEL_THRESHOLD` become positive labels. Do not mix scores with different definitions in one training run.

Separate resume/job source files may be used for ingestion and exploration; training operates on labeled candidate-job pairs. The application does not assume that `cv.csv`, `job.csv`, and `matches.csv` share a particular schema—inspect and map the actual columns before training.

## Train and evaluate

```powershell
python scripts\validate_data.py --input data\raw\matches.csv
python scripts\build_features.py --input data\raw\matches.csv
python scripts\train_model.py --input data\raw\matches.csv
```

The trainer uses a stratified holdout split where possible and writes a model plus metrics to `artifacts/`. If `resume_id` exists, it uses that field as the split group to reduce candidate leakage. For rigorous research, split by both candidate and job or use a temporal/external test set; do not report the holdout score as evidence of generalization without leakage checks.

The optional tuning helper in `resume_matcher.models.tune` supports group-aware cross-validation. Model reports also include random-forest feature importance; importance is descriptive, not a causal explanation.

Metrics include accuracy, precision, recall, F1, ROC-AUC (when defined), confusion matrix, and the evaluation split size. Choose decision thresholds using validation data and the project's costs of false matches versus missed matches; no target accuracy is guaranteed.

## Project layout

```text
supervised-project/
├── .streamlit/config.toml
├── app/
│   ├── app.py
│   ├── components/
│   └── pages/
├── artifacts/                 # Generated locally; excluded from version control
├── data/
│   ├── raw/                    # Original datasets; excluded from version control
│   ├── processed/              # Generated feature tables; excluded from version control
│   └── taxonomy/
├── docs/
├── notebooks/                  # Exploratory work; production logic stays in src/
├── scripts/
├── src/resume_matcher/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── preprocessing/
│   └── recommender/
├── tests/
├── pyproject.toml
└── README.md
```

See [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for field guidance, [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all parameters, and [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for evaluation and responsible-use notes.

## Responsible use and privacy

Resume matching is decision support, not an automated hiring decision. Validate data provenance, consent, retention, access controls, label quality, subgroup performance, and job-relatedness before use. Remove direct identifiers where feasible, protect uploaded files, and do not use sensitive traits or proxy features. A model trained on biased historical decisions can reproduce those biases. Human review is required.

## Development

```powershell
pytest
ruff check .
```
