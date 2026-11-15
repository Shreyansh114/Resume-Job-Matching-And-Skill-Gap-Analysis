"""Compact chart helpers for dashboard pages."""

import pandas as pd
import streamlit as st


def render_skill_chart(matched: list[str], missing: list[str]) -> None:
    rows = [
        {"Skill": skill, "Status": status, "Count": 1}
        for status, skills in (("Matched", matched), ("Gap", missing))
        for skill in skills
    ]
    if rows:
        chart = pd.DataFrame(rows).pivot_table(
            index="Skill", columns="Status", values="Count", aggfunc="sum", fill_value=0
        )
        st.bar_chart(chart, horizontal=True, height=max(180, 34 * len(rows)))
