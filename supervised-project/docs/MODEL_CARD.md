# Model card (template)

## Intended use

Research and decision-support evaluation of resume/job relevance and possible skill gaps. Not intended for autonomous screening, ranking people for consequential decisions, or making employment decisions.

## Model and inputs

The baseline supervised model is a random-forest binary classifier on pair-level text similarity, skill overlap, and optional structured features. TF-IDF is the default text backend; Sentence Transformers can be enabled explicitly. The exact feature schema and backend are stored with each local model artifact.

The interactive dashboard also includes a separate, transparent ATS-style demo rubric. It summarizes taxonomy skill coverage, representative job-description keywords, target-title token overlap, and basic text/section checks. The demo rubric is not a real ATS parser, is not trained, and must not be interpreted as a hiring score or prediction. When a saved supervised model exists, its positive-class output is labeled separately.

## Training and evaluation

Training requires a labeled resume/job pair table. The holdout report includes accuracy, precision, recall, F1, ROC-AUC when defined, confusion matrix, sample counts, threshold, split strategy, and random-forest feature importance. Impurity-based feature importance is descriptive, not causal. A random or single-identifier grouped split does not establish fairness or generalization. Prefer group-safe, temporal, and external evaluation; check both resume and job overlap.

## Limitations and risks

- Labels may encode historical discrimination, inconsistent recruiter preferences, or noisy judgments.
- Text similarity and keyword taxonomies miss context, transferable skills, and equivalent experience.
- Resume parsing can omit content, especially from scanned PDFs, tables, or unusual layouts.
- This starter taxonomy and resource list are incomplete and require local domain review.
- Aggregate metrics can conceal poor performance for groups or job families.

## Human oversight

Use job-related criteria, validate subgroup performance and data provenance, provide review/appeal paths, and require a qualified human to make any employment decision. Do not treat a score as a probability of job performance.
