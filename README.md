# 🧠 Multimodal Resume–Job Matching & Skill Gap Analysis

<p align="center">
  **An Intelligent, Research-Ready Resume Matching and Skill Gap Analysis System**
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-success" />
  <img src="https://img.shields.io/badge/License-To%20Be%20Defined-lightgrey" />
</p>

---

## 📌 Overview

**Multimodal Resume–Job Matching and Skill Gap Analysis** is a research-oriented Python project designed to evaluate how well a candidate's resume aligns with a given job description.

The system combines Natural Language Processing (NLP), feature engineering, and supervised machine learning to analyze candidate profiles, identify relevant skills, evaluate job compatibility, and recommend resources to bridge skill gaps.

It also provides an interactive Streamlit dashboard for resume analysis, model evaluation, and ATS-style resume scanning.

The project is designed to be modular, extensible, explainable, and suitable for further academic research and experimentation.

> **Current status:** This repository does not include private datasets or a pretrained matching model. Until a supervised model is trained, the interactive matching page uses a clearly labeled TF-IDF and skill-overlap baseline. It does not present baseline scores as trained-model predictions.

## ✨ Key Features

| Feature | Description |
|---|---|
| 📂 Data Ingestion | Supports CSV and Parquet datasets with schema validation and data-quality checks. |
| 📄 Resume Extraction | Extracts text from PDF, DOCX, and TXT documents. |
| 🧠 NLP Processing | Processes resume and job-description text for matching. |
| 🎯 Skill Matching | Identifies matching skills, missing skills, and keyword gaps. |
| ⚙️ Feature Engineering | Generates text similarity, skill overlap, experience, and education features. |
| 🤖 Supervised Learning | Supports Random Forest model training and evaluation. |
| 📊 Model Evaluation | Reports accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix. |
| 📈 Interactive Dashboard | Provides resume-fit analysis and model/data evaluation through Streamlit. |
| 💡 Skill Recommendations | Suggests learning resources for identified skill gaps. |
| 🧪 Unit Testing | Includes tests for preprocessing, features, validation, and inference. |

## 🏗️ System Architecture

The project follows a modular machine-learning pipeline:

```text
             RESUME + JOB DESCRIPTION
                       │
                       ▼
               DATA INGESTION
             (CSV / Parquet / PDF)
                       │
                       ▼
                TEXT EXTRACTION
               (PDF / DOCX / TXT)
                       │
                       ▼
               TEXT PREPROCESSING
                       │
                       ▼
               FEATURE ENGINEERING
          ┌────────────┼────────────┐
          │            │            │
       TF-IDF      Skill Match   Experience
       Similarity    Features    & Education
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
              SUPERVISED MODEL
                 (Random Forest)
                       │
                       ▼
               MODEL EVALUATION
                       │
                       ▼
               MATCHING ANALYSIS
                       │
              ┌────────┴────────┐
              ▼                 ▼
         JOB MATCHING       SKILL GAP
                              ANALYSIS
              │                 │
              └────────┬────────┘
                       ▼
               STREAMLIT DASHBOARD
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF, Text Processing |
| Data Processing | Pandas, NumPy |
| Document Processing | PyMuPDF / PDF and DOCX extraction libraries |
| Visualization | Streamlit |
| Data Formats | CSV, Parquet |
| Testing | Pytest |
| Code Quality | Ruff |
| Optional Semantic Search | Sentence Transformers |

## 📁 Project Structure

```text
supervised-project/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── app.py
│   ├── components/
│   └── pages/
│
├── artifacts/
│   └── # Generated model artifacts (not committed)
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── taxonomy/
│
├── docs/
│   ├── DATA_DICTIONARY.md
│   ├── CONFIGURATION.md
│   └── MODEL_CARD.md
│
├── notebooks/
│
├── scripts/
│   ├── validate_data.py
│   ├── build_features.py
│   └── train_model.py
│
├── src/
│   └── resume_matcher/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── preprocessing/
│       └── recommender/
│
├── tests/
│
├── pyproject.toml
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd supervised-project
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,documents]"
```

### 4. Launch the Application

```bash
streamlit run app/app.py
```

The application will open in your browser.

**Note:** The `documents` extra enables PDF and DOCX extraction. For optional Sentence Transformers embeddings, install the semantic extra:

```bash
pip install -e ".[semantic]"
```

Set the embedding backend:

```powershell
$env:RESUME_MATCHER_EMBEDDING_BACKEND="sentence-transformers"
```

The first use may download the configured embedding model.

## 📊 Dataset Requirements

The project supports CSV and Parquet ingestion.

Training requires a labeled candidate-job pair dataset.

**Default dataset location:**

```text
data/raw/matches.csv
```

### Required Columns

| Column | Required | Description |
|---|---|---|
| `resume_text` | Yes | Resume text or extracted content |
| `job_description` | Yes | Job description text |
| `label` | One of these | Binary matching label |
| `match_score` | One of these | Numeric matching score between 0 and 1 |
| `resume_id` | Optional | Candidate identifier |
| `job_id` | Optional | Job identifier |
| `resume_years_experience` | Optional | Candidate experience |
| `job_years_experience` | Optional | Required experience |
| `resume_education` | Optional | Candidate education |
| `job_education` | Optional | Required education |

**Important Dataset Guidelines:**

- `label` is used directly after binary validation.
- If only `match_score` is available, scores at or above `RESUME_MATCHER_LABEL_THRESHOLD` become positive labels.
- Do not mix matching scores generated using different definitions in one training run.
- Keep original datasets in `data/raw/`.
- Never commit private or confidential resumes.
- Inspect actual dataset schemas before mapping columns.

Separate resume and job-description files can be used for ingestion and exploration. Model training operates on labeled candidate-job pairs.

## ⚙️ Model Training & Evaluation

Run the following commands in sequence:

**Step 1: Validate Dataset**

```bash
python scripts/validate_data.py --input data/raw/matches.csv
```

**Step 2: Build Features**

```bash
python scripts/build_features.py --input data/raw/matches.csv
```

**Step 3: Train the Model**

```bash
python scripts/train_model.py --input data/raw/matches.csv
```

The training pipeline:

1. Validates the input dataset.
2. Processes resume and job-description text.
3. Generates pair-level features.
4. Splits data into training and testing sets.
5. Trains a Random Forest classifier.
6. Evaluates model performance.
7. Saves model artifacts and evaluation metrics in `artifacts/`.

If `resume_id` exists, it is used as the split group to reduce candidate leakage.

For rigorous research, consider splitting by both candidate and job, or using temporal and external test sets. A standard holdout score alone does not establish generalization.

## 📈 Evaluation Metrics

The system evaluates supervised matching models using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Reliability of positive match predictions |
| Recall | Ability to identify relevant matches |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish positive and negative matches |
| Confusion Matrix | Breakdown of classification outcomes |

Additional evaluation capabilities include:

- Evaluation split size.
- Random Forest feature importance.
- Group-aware cross-validation through the optional tuning helper.

**Research consideration:** Feature importance is descriptive, not causal. Decision thresholds should be selected using validation data and the relative costs of false matches and missed matches. No target accuracy or generalization performance is guaranteed.

## 🖥️ Interactive Dashboard

The Streamlit application provides an interface for exploring resume-job compatibility.

Core capabilities include:

- Resume and job-description analysis.
- ATS-style resume scanning using an explainable demo rubric.
- Skill and keyword gap identification.
- Document quality checks.
- Matching analysis using the available baseline.
- Model and dataset evaluation.
- Learning resource recommendations.

The ATS-style scan is an original heuristic demonstration. It is not a real ATS parser, Jobscan, a trained matching model, or a prediction of interview or job performance.

Resume text is processed in memory by the local application. Exported scan summaries exclude the original resume and job-description text.

## 🔬 Research & Extensibility

The project is designed to support further research and experimentation.

Potential research extensions include:

- Semantic resume-job similarity using Sentence Transformers.
- Advanced NLP-based skill extraction.
- Alternative supervised classifiers.
- Hyperparameter optimization.
- Group-aware cross-validation.
- Candidate and job leakage analysis.
- Temporal and external validation.
- Fairness and subgroup performance evaluation.
- Explainable machine-learning techniques.

## 🧪 Testing & Code Quality

Run the complete test suite:

```bash
pytest
```

Check code quality using Ruff:

```bash
ruff check .
```

These checks help maintain consistent code quality and verify core project functionality.

## 🔐 Responsible Use & Privacy

This project is intended for **decision support, not automated hiring decisions**.

Before using the system in a real recruitment environment:

- Validate dataset provenance and consent.
- Establish appropriate data retention and access controls.
- Remove direct identifiers wherever feasible.
- Avoid sensitive attributes and proxy features.
- Evaluate label quality and subgroup performance.
- Assess potential historical bias in training data.
- Require human review for all consequential decisions.

A model trained on biased historical hiring decisions may reproduce or amplify those biases.

## 📚 Documentation

Additional technical documentation is available in the `docs/` directory.

| Document | Description |
|---|---|
| `DATA_DICTIONARY.md` | Dataset fields, schemas, and data requirements |
| `CONFIGURATION.md` | Configuration parameters and environment settings |
| `MODEL_CARD.md` | Model evaluation, limitations, and responsible-use notes |

## 🎯 Project Goals

The long-term objective is to develop a transparent and extensible resume-job matching framework that combines machine learning, NLP, and actionable skill-gap recommendations.

The system aims to make resume evaluation more understandable by showing not only how a candidate aligns with a job description, but also which skills are missing and how those gaps can be addressed.

---

<p align="center">

**Built with Python, Machine Learning & NLP**

*Making Resume Matching More Explainable and Skill Development More Actionable.*

</p>
