"""Inspect training inputs and saved model evaluation."""

import pandas as pd
import streamlit as st
from components.data_access import data_inventory, load_metrics, load_pair_data
from components.ui import page_header, section_title

page_header(
    "Evaluation and data quality",
    "Inspect what the model learned from.",
    "Review available local datasets and saved holdout metrics. Nothing is fabricated "
    "when training outputs are absent.",
)

inventory = data_inventory()
pair_frame, pair_path = load_pair_data()
metrics = load_metrics()

section_title("Training inputs")
st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")
if pair_frame is None:
    st.info(
        "No labeled pair table is available. Add a dataset and run the validation "
        "and training scripts."
    )
else:
    st.caption(
        f"Loaded `{pair_path.name}` with {len(pair_frame):,} rows and "
        f"{len(pair_frame.columns)} columns."
    )
    required = {"resume_text", "job_description"}
    label_column = (
        "label"
        if "label" in pair_frame.columns
        else ("match_score" if "match_score" in pair_frame.columns else None)
    )
    if not required.issubset(pair_frame.columns) or label_column is None:
        st.error(
            "The loaded table does not meet the labeled-pair schema. See docs/DATA_DICTIONARY.md."
        )
    else:
        data_columns = st.columns(3)
        data_columns[0].metric("Rows", f"{len(pair_frame):,}")
        data_columns[1].metric("Columns", str(len(pair_frame.columns)))
        data_columns[2].metric("Missing resume text", f"{pair_frame['resume_text'].isna().sum():,}")
        st.subheader("Label quality checks")
        st.write(f"Target field: `{label_column}`")
        st.write(f"Missing target values: {pair_frame[label_column].isna().sum():,}")
        if label_column == "label":
            st.bar_chart(pair_frame["label"].value_counts(dropna=False).sort_index())
        else:
            scores = pd.to_numeric(pair_frame[label_column], errors="coerce").dropna()
            if not scores.empty:
                st.bar_chart(scores.value_counts(bins=10).sort_index())

st.write("")
section_title("Saved holdout evaluation")
if metrics is None:
    st.info(
        "No evaluation report exists yet. Run `python scripts\\train_model.py "
        "--input data\\raw\\matches.csv` after validating and mapping your labeled pairs."
    )
else:
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    columns = st.columns(len(metric_names))
    for column, name in zip(columns, metric_names, strict=True):
        value = metrics.get(name)
        column.metric(
            name.replace("_", " ").title(),
            f"{value:.3f}" if isinstance(value, (int, float)) else "N/A",
        )
    st.caption(
        f"Split: {metrics.get('split_strategy', 'not recorded')} · "
        f"Training rows: {metrics.get('training_rows', 'not recorded')} · "
        f"Evaluation rows: {metrics.get('evaluation_rows', 'not recorded')} · "
        f"Embedding backend: {metrics.get('embedding_backend', 'not recorded')}"
    )
    matrix = metrics.get("confusion_matrix")
    if (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        st.subheader("Confusion matrix")
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual: no match", "Actual: match"],
                columns=["Predicted: no match", "Predicted: match"],
            ),
            width="stretch",
        )
    importance = metrics.get("feature_importance")
    if isinstance(importance, list) and importance:
        importance_frame = pd.DataFrame(importance)
        if {"feature", "importance"}.issubset(importance_frame.columns):
            st.subheader("Random-forest feature importance")
            st.bar_chart(
                importance_frame.set_index("feature")["importance"].sort_values(),
                horizontal=True,
            )
            st.caption("Impurity-based importance is descriptive, not causal.")
    st.warning(
        "A holdout score is not a fairness or generalization guarantee. Audit both "
        "resume and job overlap, "
        "subgroup performance, label quality, and decision thresholds before any real-world use."
    )

with st.expander("Training configuration"):
    st.markdown(
        "- Positive label threshold for `match_score`: "
        "`RESUME_MATCHER_LABEL_THRESHOLD` (default 0.5)\n"
        "- Holdout fraction: `RESUME_MATCHER_TEST_SIZE` (default 0.2)\n"
        "- Random seed: `RESUME_MATCHER_RANDOM_STATE` (default 42)\n"
        "- Random forest estimator count, maximum depth, minimum leaf size, and feature "
        "sampling are environment-configurable.\n"
        "- Text backend: TF-IDF by default; optional Sentence Transformers embeddings."
    )
