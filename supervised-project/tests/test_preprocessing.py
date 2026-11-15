import pandas as pd

from resume_matcher.preprocessing.skill_extractor import extract_skills
from resume_matcher.preprocessing.text_cleaner import clean_text


def test_clean_text_redacts_contact_details_and_normalizes():
    assert clean_text("  PYTHON  me@example.com https://example.com ") == "python"
    assert clean_text(pd.NA) == ""


def test_skill_extraction_matches_aliases_without_partial_words():
    taxonomy = {"Python": ["python3"], "SQL": ["structured query language"]}
    assert extract_skills("Built Python3 tools and used SQL databases.", taxonomy) == {
        "Python",
        "SQL",
    }
    assert extract_skills("Experienced in pythonista tools.", taxonomy) == set()
