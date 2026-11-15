from resume_matcher.features.feature_builder import FEATURE_COLUMNS, build_pair_features
from resume_matcher.features.similarity_metrics import dice_similarity, jaccard_similarity
from resume_matcher.recommender.gap_analyzer import analyze_skill_gap


def test_similarity_metrics_handle_empty_sets():
    assert jaccard_similarity(set(), set()) == 0
    assert dice_similarity({"python"}, {"python", "sql"}) == 2 / 3


def test_pair_features_have_stable_schema():
    taxonomy = {"Python": [], "SQL": []}
    features = build_pair_features(
        {
            "resume_text": "Python developer",
            "job_description": "Python and SQL developer",
            "resume_years_experience": 3,
            "job_years_experience": 5,
            "resume_education": "bachelor",
            "job_education": "bachelor",
        },
        taxonomy,
        embedding_backend="tfidf",
    )
    assert tuple(features) == FEATURE_COLUMNS
    assert features["matched_skill_count"] == 1
    assert features["missing_skill_count"] == 1
    assert features["education_match"] == 1


def test_gap_analysis_identifies_missing_required_skills():
    result = analyze_skill_gap(
        "Python developer",
        "Python and SQL developer",
        {"Python": [], "SQL": []},
    )
    assert result["matched_skills"] == ["Python"]
    assert result["missing_skills"] == ["SQL"]
