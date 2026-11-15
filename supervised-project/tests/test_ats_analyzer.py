import pytest

from resume_matcher.recommender.ats_analyzer import analyze_ats_match, scan_report_summary


@pytest.fixture
def taxonomies():
    return (
        {
            "Python": ["python3"],
            "SQL": ["structured query language"],
            "Machine Learning": ["ml"],
            "Communication": ["communication skills"],
        },
        {
            "Communication": ["communication skills"],
            "Collaboration": ["teamwork", "collaboration", "collaborated", "collaborate"],
        },
    )


def test_ats_report_finds_skills_keywords_and_document_sections(taxonomies):
    technical, soft = taxonomies
    report = analyze_ats_match(
        """Jamie Doe
jamie@example.com | (555) 555-0101
SUMMARY
Product data analyst skilled in Python, SQL, and communication.
EXPERIENCE
- Built Python and SQL reporting dashboards for product teams.
- Presented findings and collaborated with cross-functional partners.
EDUCATION
Bachelor of Science
""",
        "Product Data Analyst: Python, SQL, machine learning, communication, and teamwork.",
        technical,
        soft,
        job_title="Product Data Analyst",
    )

    assert 0 <= report["overall_score"] <= 100
    assert report["hard_skills"]["matched"] == ["Python", "SQL"]
    assert report["hard_skills"]["missing"] == ["Machine Learning"]
    assert report["soft_skills"]["matched"] == ["Collaboration", "Communication"]
    assert report["document_checks"]["sections_found"] == [
        "Summary",
        "Experience",
        "Education",
    ]
    assert any("Machine Learning" in suggestion for suggestion in report["suggestions"])


def test_ats_report_handles_empty_text_without_crashing(taxonomies):
    technical, soft = taxonomies
    report = analyze_ats_match("", "", technical, soft)
    assert report["overall_score"] == 0
    assert report["keywords"]["required"] == []
    assert report["hard_skills"]["missing"] == []
    assert report["assessment"] == "More tailoring recommended"


def test_report_export_excludes_resume_contents(taxonomies):
    technical, soft = taxonomies
    report = analyze_ats_match(
        "Unique private resume phrase with Python",
        "Python SQL data analyst",
        technical,
        soft,
        job_title="Data Analyst",
    )
    export = scan_report_summary(report, "Data Analyst")
    assert "resume_text" not in export
    assert "Unique private resume phrase" not in str(export)


def test_keyword_limit_must_be_positive(taxonomies):
    technical, soft = taxonomies
    with pytest.raises(ValueError, match="max_keywords"):
        analyze_ats_match("Python", "Python developer", technical, soft, max_keywords=0)
