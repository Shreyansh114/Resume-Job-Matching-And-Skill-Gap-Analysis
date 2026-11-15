"""Streamlit entry point for the resume-job matching dashboard."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="TalentSignal | Resume & Job Intelligence",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { border-right: 1px solid #E5EAF3; }
      .eyebrow { color: #64748B; font-size: .75rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; }
      .hero-title { color: #172033; font-size: 2rem; font-weight: 750;
        letter-spacing: -.04em; margin: .2rem 0 .4rem; }
      .hero-copy { color: #667085; font-size: 1rem; margin-bottom: 1.4rem; }
      .status-note { background: #EEF2FF; border-left: 4px solid #4F46E5;
        border-radius: 8px; padding: 1rem 1.1rem; color: #27335A; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #E7EBF3;
        border-radius: 12px; padding: 1rem 1.1rem; }
      div[data-testid="stMetricLabel"] { color: #667085; }
      .section-heading { color: #172033; font-size: 1.1rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## TalentSignal")
    st.caption("Resume · role · skills")
    st.divider()
    st.caption("Supervised matching and skill-gap analysis")

pages = [
    st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page(
        "pages/match_analysis.py", title="ATS resume scan", icon=":material/compare_arrows:"
    ),
    st.Page("pages/model_data.py", title="Model & data", icon=":material/monitoring:"),
]
selected_page = st.navigation(pages, position="sidebar")
selected_page.run()
