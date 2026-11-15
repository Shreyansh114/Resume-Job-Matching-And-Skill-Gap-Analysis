"""Project status and data inventory."""

import pandas as pd
import streamlit as st
from components.data_access import data_inventory, load_metrics, load_pair_data
from components.ui import page_header, section_title

page_header(
    "Talent intelligence workspace",
    "Resume–job matching, with the evidence in view.",
    "Explore project readiness, compare a resume with a role, and inspect model evaluation "
    "without hiding missing inputs.",
)

inventory = data_inventory()
available = sum(item["Status"] == "Available" for item in inventory)
pair_frame, pair_path = load_pair_data()
metrics = load_metrics()

metric_columns = st.columns(4)
metric_columns[0].metric("Project assets", f"{available}/{len(inventory)}", "available locally")
metric_columns[1].metric("Labeled pairs", f"{len(pair_frame):,}" if pair_frame is not None else "—")
metric_columns[2].metric(
    "Trained model",
    "Ready"
    if any(item["Asset"] == "Trained model" and item["Status"] == "Available" for item in inventory)
    else "Not trained",
)
metric_columns[3].metric(
    "Holdout F1",
    f"{metrics['f1']:.3f}" if metrics and isinstance(metrics.get("f1"), (int, float)) else "—",
    "from saved evaluation report" if metrics else "train and evaluate to populate",
)

st.write("")
section_title(
    "Workspace readiness",
    "Files are read from your local project; this dashboard does not upload data to a service.",
)
st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")

if pair_frame is None:
    st.info(
        "No labeled pair dataset found yet. Add your de-identified pair table as "
        "`data/raw/matches.csv` or `data/processed/cleaned_pairs.parquet`. "
        "Review the data contract in `docs/DATA_DICTIONARY.md` before training."
    )
else:
    st.caption(f"Pair data source: `{pair_path.relative_to(pair_path.parents[2])}`")
    if "label" in pair_frame.columns:
        label_counts = pair_frame["label"].value_counts(dropna=False).sort_index()
        st.subheader("Observed label distribution")
        st.bar_chart(label_counts.rename(index={0: "Not matched", 1: "Matched"}))
    elif "match_score" in pair_frame.columns:
        st.subheader("Observed match-score distribution")
        st.bar_chart(pair_frame["match_score"].dropna().value_counts(bins=10).sort_index())

st.write("")
section_title("Analysis workflow")
workflow = st.columns(3)
workflow[0].markdown(
    "**01 · Prepare data**\n\nValidate pair labels and map source columns to the documented schema."
)
workflow[1].markdown(
    "**02 · Train and evaluate**\n\nUse a leakage-aware split, review errors, "
    "and save reproducible local artifacts."
)
workflow[2].markdown(
    "**03 · Review skill gaps**\n\nUse matching scores as decision support; a human "
    "must review any employment decision."
)
