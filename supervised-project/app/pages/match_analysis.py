"""Interactive ATS-style resume scan and skill-gap report."""

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from components.data_access import load_local_model, load_taxonomy
from components.ui import page_header

from resume_matcher.models.inference import predict_pair
from resume_matcher.preprocessing.resume_parser import extract_resume_text
from resume_matcher.recommender.ats_analyzer import analyze_ats_match, scan_report_summary
from resume_matcher.recommender.course_engine import load_learning_resources, recommend_resources
from resume_matcher.recommender.gap_analyzer import analyze_skill_gap

page_header(
    "Resume fit scan",
    "See how your resume lines up with a role.",
    "A transparent demo report checks skills, job keywords, target-title overlap, "
    "and basic document structure.",
)

st.info(
    "This is an original, rule-based ATS-style demo—not Jobscan, a real ATS parser, or a "
    "trained prediction. Your resume text is processed for this session and is not saved by "
    "the dashboard. Use the report as a review aid, not a hiring decision.",
    title="What this scan does",
)

sample_resume = """Alex Morgan
alex.morgan@example.com | (555) 555-0101

SUMMARY
Data analyst with 3 years of experience turning product data into clear recommendations.

EXPERIENCE
- Built Python and SQL reporting workflows with pandas for product and operations teams.
- Created Tableau dashboards and presented trends to cross-functional stakeholders.

SKILLS
Python, SQL, pandas, Tableau, data analysis, communication

EDUCATION
Bachelor of Science in Information Systems
"""

sample_job = """We are hiring a Product Data Analyst to help product teams make
evidence-based decisions.

Responsibilities: build dashboards, analyze experiments, explain insights to stakeholders, and
collaborate with engineering and product teams.

Requirements: Python, SQL, machine learning, data visualization, statistics, communication,
project management, and 2+ years of experience.
"""

with st.form("ats_scan_form", border=True):
    resume_column, job_column = st.columns(2)
    with resume_column:
        st.subheader("Your resume")
        resume_file = st.file_uploader(
            "Upload a resume",
            type=["txt", "pdf", "docx"],
            help="Optional. The file is parsed in memory for this scan and is not stored.",
        )
        resume_input = st.text_area(
            "Or paste resume text",
            value=sample_resume,
            height=330,
            help="The sample is provided so you can try the demo immediately.",
        )
        resume_years = st.number_input(
            "Years of relevant experience (optional)",
            min_value=0.0,
            max_value=60.0,
            value=3.0,
        )
        resume_education = st.text_input(
            "Highest relevant education (optional)",
            placeholder="Bachelor's degree",
        )
    with job_column:
        st.subheader("Target job")
        job_title = st.text_input("Job title", value="Product Data Analyst")
        job_input = st.text_area(
            "Job description",
            value=sample_job,
            height=330,
        )
        job_years = st.number_input(
            "Required experience (optional)",
            min_value=0.0,
            max_value=60.0,
            value=2.0,
        )
        job_education = st.text_input(
            "Required education (optional)",
            placeholder="Bachelor's degree",
        )
    submitted = st.form_submit_button(
        "Run resume scan", type="primary", width="stretch"
    )

if submitted:
    resume_text = resume_input
    upload_extension = ""
    if resume_file is not None:
        try:
            resume_text = extract_resume_text(resume_file.name, resume_file.getvalue())
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc), title="Resume could not be read")
            st.stop()
        upload_extension = Path(resume_file.name).suffix.lower()

    if not resume_text.strip() or not job_input.strip():
        st.warning("Add both a resume and a job description before scanning.")
        st.stop()

    taxonomy = load_taxonomy()
    soft_taxonomy_path = (
        Path(__file__).resolve().parents[2] / "data" / "taxonomy" / "soft_skills.json"
    )
    with soft_taxonomy_path.open(encoding="utf-8") as file:
        soft_taxonomy = json.load(file)
    report = analyze_ats_match(
        resume_text,
        job_input,
        taxonomy,
        soft_taxonomy,
        job_title=job_title,
    )

    st.divider()
    st.subheader("Your scan report")
    score_column, assessment_column = st.columns([1, 2])
    with score_column:
        st.metric(
            "Resume-to-job alignment",
            f"{report['overall_score']}/100",
            border=True,
        )
        st.progress(
            int(report["overall_score"]),
            text=str(report["assessment"]),
        )
    with assessment_column:
        st.markdown(f"### {report['assessment']}")
        st.write(
            "This demo score combines the visible checks below. It is not a probability, "
            "ATS ranking, or prediction of interview or job performance."
        )
        st.caption(
            "Demo rubric: hard skills 40%, soft skills 15%, job keywords 25%, "
            "target-title overlap 10% when supplied, and basic document checks 10%. "
            "Unavailable skill categories are omitted and the remaining weights are normalized."
        )

    component_rows = [
        {"Check": name, "Coverage": score}
        for name, score in report["components"].items()
        if score is not None
    ]
    component_frame = pd.DataFrame(component_rows)
    if not component_frame.empty:
        st.bar_chart(
            component_frame,
            x="Check",
            y="Coverage",
            horizontal=True,
            alt="Coverage across skills, keywords, title alignment, and basic document checks",
        )

    hard_skills = report["hard_skills"]
    soft_skills = report["soft_skills"]
    keyword_data = report["keywords"]
    hard_column, soft_column, keyword_column = st.columns(3)
    with hard_column:
        with st.container(border=True):
            st.subheader("Technical skills")
            st.metric("Matched", f"{len(hard_skills['matched'])}/{len(hard_skills['required'])}")
            st.write("**Found**")
            st.write(", ".join(hard_skills["matched"]) or "No overlap detected")
            st.write("**Not found**")
            st.write(", ".join(hard_skills["missing"]) or "No identified gaps")
    with soft_column:
        with st.container(border=True):
            st.subheader("People skills")
            st.metric("Matched", f"{len(soft_skills['matched'])}/{len(soft_skills['required'])}")
            st.write("**Found**")
            st.write(", ".join(soft_skills["matched"]) or "No overlap detected")
            st.write("**Not found**")
            st.write(", ".join(soft_skills["missing"]) or "No identified gaps")
    with keyword_column:
        with st.container(border=True):
            st.subheader("Job keywords")
            st.metric(
                "Matched",
                f"{len(keyword_data['matched'])}/{len(keyword_data['required'])}",
            )
            st.write("**Found**")
            st.write(", ".join(keyword_data["matched"]) or "No overlap detected")
            st.write("**Review**")
            st.write(", ".join(keyword_data["missing"]) or "No keyword gaps detected")

    doc_checks = report["document_checks"]
    checks_frame = pd.DataFrame(
        [
            {"Check": name, "Status": "Found" if found else "Review"}
            for name, found in doc_checks["checks"].items()
        ]
    )
    checks_frame = pd.concat(
        [
            checks_frame,
            pd.DataFrame(
            [
                {"Check": f"Section: {section}", "Status": "Found"}
                for section in doc_checks["sections_found"]
            ]
            ),
        ],
        ignore_index=True,
    )
    st.subheader("Basic document checks")
    st.caption(
        "Text-only checks provide hints and cannot determine whether a file will parse correctly "
        "in a particular employer's ATS."
    )
    st.dataframe(
        checks_frame,
        hide_index=True,
        width="stretch",
        alt="Basic resume structure and contact-detail checks",
    )

    st.subheader("Suggested edits")
    for suggestion in report["suggestions"]:
        st.markdown(f"- {suggestion}")

    gaps = analyze_skill_gap(resume_text, job_input, taxonomy)
    resources = recommend_resources(
        gaps["missing_skills"],
        load_learning_resources(),
    )
    with st.expander("Learning resources for potential technical skill gaps"):
        for skill, entries in resources.items():
            if entries:
                st.markdown(f"**{skill}**")
                for entry in entries:
                    st.markdown(f"- [{entry['title']} — {entry['provider']}]({entry['url']})")
            else:
                st.markdown(f"**{skill}:** no curated resource mapped yet.")
        if not resources:
            st.caption("No taxonomy-based technical skill gaps were identified.")

    pair = {
        "resume_text": resume_text,
        "job_description": job_input,
        "resume_years_experience": resume_years,
        "job_years_experience": job_years,
        "resume_education": resume_education,
        "job_education": job_education,
    }
    artifact_path = Path(__file__).resolve().parents[2] / "artifacts" / "matching_model.joblib"
    if artifact_path.is_file():
        artifact = load_local_model()
        if artifact is None:
            st.error("A model artifact was present but could not be loaded.")
            st.stop()
        model_score = predict_pair(pair, artifact, taxonomy)
        st.subheader("Optional supervised-model output")
        st.metric("Positive-class model score", f"{model_score:.1%}")
        st.caption(
            "Separate from the demo ATS-style rubric above. This output is meaningful only if "
            "the model was trained and validated on appropriate, representative labeled data."
        )
    else:
        st.info(
            "No supervised model is trained yet. This report uses only the visible rule-based "
            "demo rubric; it does not pretend a model has been fitted.",
            title="Demo mode",
        )

    slug = re.sub(r"[^a-z0-9]+", "-", job_title.lower()).strip("-") or "resume-scan"
    export = scan_report_summary(report, job_title)
    export["uploaded_file_type"] = upload_extension or "pasted text"
    st.download_button(
        "Download scan summary",
        data=json.dumps(export, indent=2),
        file_name=f"{slug}-scan-summary.json",
        mime="application/json",
        width="stretch",
    )
    st.caption(
        "The downloadable summary excludes resume and job-description text. Skill and keyword "
        "detection uses a starter taxonomy and may miss equivalent wording or context."
    )
