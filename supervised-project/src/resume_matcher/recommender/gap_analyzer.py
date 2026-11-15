"""Identify required skills that are not found in a resume."""

from resume_matcher.preprocessing.skill_extractor import extract_skills


def analyze_skill_gap(
    resume_text: str,
    job_text: str,
    taxonomy: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return canonical matched and missing skills and resume-only skills."""
    resume_skills = extract_skills(resume_text, taxonomy)
    required_skills = extract_skills(job_text, taxonomy)
    return {
        "matched_skills": sorted(resume_skills & required_skills),
        "missing_skills": sorted(required_skills - resume_skills),
        "additional_resume_skills": sorted(resume_skills - required_skills),
    }
