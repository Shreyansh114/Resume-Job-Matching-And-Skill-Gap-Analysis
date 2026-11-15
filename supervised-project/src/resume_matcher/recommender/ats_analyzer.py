"""Transparent heuristic analysis for an ATS-style resume matching demo."""

import re
from collections.abc import Mapping

from sklearn.feature_extraction.text import TfidfVectorizer

from resume_matcher.preprocessing.skill_extractor import extract_skills
from resume_matcher.preprocessing.text_cleaner import clean_text

_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_BULLET = re.compile(r"(?m)^\s*(?:[•●▪◦*+-]|\d+[.)])\s+\S")
_HEADINGS = {
    "Summary": r"(?im)^\s*(?:professional\s+)?summary\s*:?\s*$",
    "Experience": (
        r"(?im)^\s*(?:(?:work\s+)?experience|employment\s+history)\s*:?\s*$"
    ),
    "Education": r"(?im)^\s*education(?:al\s+background)?\s*:?\s*$",
    "Skills": r"(?im)^\s*(?:technical\s+)?skills\s*:?\s*$",
    "Projects": r"(?im)^\s*projects?\s*:?\s*$",
    "Certifications": r"(?im)^\s*certifications?\s*:?\s*$",
}


def _keyword_matches(
    resume_text: str,
    job_text: str,
    *,
    limit: int,
) -> tuple[list[str], list[str]]:
    """Find representative job-description terms and check whether they appear in the resume."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.-]{1,}\b",
            max_features=20_000,
        )
        vector = vectorizer.fit_transform([job_text])
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            return [], []
        raise

    terms = vectorizer.get_feature_names_out()
    weights = vector.toarray()[0]
    ranked = sorted(
        zip(terms, weights, strict=True),
        key=lambda item: (-float(item[1]), item[0].count(" "), item[0]),
    )
    resume = clean_text(resume_text)
    selected = [term for term, _ in ranked[:limit]]
    matched = [term for term in selected if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", resume)]
    return selected, matched


def analyze_ats_match(
    resume_text: str,
    job_description: str,
    technical_taxonomy: dict[str, list[str]],
    soft_skill_taxonomy: dict[str, list[str]],
    *,
    job_title: str = "",
    max_keywords: int = 12,
) -> dict[str, object]:
    """Create an interpretable demo report from skills, keywords, title, and text checks.

    The weighted rubric is a project heuristic, not a real ATS ranking model or a
    validated prediction of interview or job outcomes.
    """
    if max_keywords < 1:
        raise ValueError("max_keywords must be a positive integer.")

    resume = str(resume_text or "")
    job = str(job_description or "")
    normalized_resume = clean_text(resume)
    normalized_job = clean_text(job)
    soft_skill_names = set(soft_skill_taxonomy)

    required_hard = extract_skills(job, technical_taxonomy) - soft_skill_names
    resume_hard = extract_skills(resume, technical_taxonomy) - soft_skill_names
    required_soft = extract_skills(job, soft_skill_taxonomy)
    resume_soft = extract_skills(resume, soft_skill_taxonomy)
    matched_hard = sorted(required_hard & resume_hard)
    missing_hard = sorted(required_hard - resume_hard)
    matched_soft = sorted(required_soft & resume_soft)
    missing_soft = sorted(required_soft - resume_soft)

    keywords, matched_keywords = _keyword_matches(
        normalized_resume, normalized_job, limit=max_keywords
    )
    keyword_set = set(keywords)
    matched_keyword_set = set(matched_keywords)

    title = clean_text(job_title)
    title_terms = set(title.split())
    resume_terms = set(normalized_resume.split())
    title_score = len(title_terms & resume_terms) / len(title_terms) if title_terms else None

    headings = [name for name, pattern in _HEADINGS.items() if re.search(pattern, resume)]
    contact_checks = {
        "Email address": bool(_EMAIL.search(resume)),
        "Phone number": bool(_PHONE.search(resume)),
    }
    bullet_count = len(_BULLET.findall(resume))
    document_checks = {
        "Contact details detected": any(contact_checks.values()),
        "Three or more standard sections": len(headings) >= 3,
        "Two or more bullet points": bullet_count >= 2,
    }

    components: dict[str, float | None] = {
        "Hard skills": len(matched_hard) / len(required_hard) if required_hard else None,
        "Soft skills": len(matched_soft) / len(required_soft) if required_soft else None,
        "Keywords": (
            len(matched_keyword_set) / len(keyword_set) if keyword_set else None
        ),
        "Target title": title_score,
        "Basic document checks": sum(document_checks.values()) / len(document_checks),
    }
    weights = {
        "Hard skills": 0.4,
        "Soft skills": 0.15,
        "Keywords": 0.25,
        "Target title": 0.1,
        "Basic document checks": 0.1,
    }
    active_weight = sum(weights[name] for name, score in components.items() if score is not None)
    score = (
        100
        * sum(
            weights[name] * component_score
            for name, component_score in components.items()
            if component_score is not None
        )
        / active_weight
        if active_weight
        else 0.0
    )

    if score >= 75:
        assessment = "Strong keyword alignment"
    elif score >= 50:
        assessment = "Moderate keyword alignment"
    else:
        assessment = "More tailoring recommended"

    suggestions: list[str] = []
    if missing_hard:
        suggestions.append(
            "If accurate, add evidence for these required technical skills: "
            + ", ".join(missing_hard)
            + "."
        )
    if missing_soft:
        suggestions.append(
            "Where supported by your experience, show examples of: " + ", ".join(missing_soft) + "."
        )
    missing_keywords = sorted(keyword_set - matched_keyword_set)
    if missing_keywords:
        suggestions.append(
            "Review these job-description terms and naturally include only those that reflect "
            "your experience: "
            + ", ".join(missing_keywords[:6])
            + "."
        )
    if len(headings) < 3:
        suggestions.append(
            "Consider clearly labeled sections such as Summary, Experience, Education, and Skills."
        )
    if bullet_count < 2:
        suggestions.append(
            "Use concise bullets to make responsibilities and outcomes easier to scan."
        )
    if not any(contact_checks.values()):
        suggestions.append("Add a professional email address or phone number if appropriate.")
    if not suggestions:
        suggestions.append(
            "The main demo checks are covered. Review every claim and tailor examples to "
            "the role truthfully."
        )

    return {
        "overall_score": round(score),
        "assessment": assessment,
        "components": components,
        "weights": weights,
        "hard_skills": {
            "required": sorted(required_hard),
            "matched": matched_hard,
            "missing": missing_hard,
        },
        "soft_skills": {
            "required": sorted(required_soft),
            "matched": matched_soft,
            "missing": missing_soft,
        },
        "keywords": {
            "required": keywords,
            "matched": matched_keywords,
            "missing": sorted(keyword_set - matched_keyword_set),
        },
        "document_checks": {
            "contact_checks": contact_checks,
            "sections_found": headings,
            "bullet_count": bullet_count,
            "checks": document_checks,
        },
        "suggestions": suggestions,
    }


def scan_report_summary(report: Mapping[str, object], job_title: str) -> dict[str, object]:
    """Prepare a privacy-conscious export with results but without resume contents."""
    return {
        "report_type": "ATS-style resume match demo",
        "job_title": job_title,
        "overall_score": report["overall_score"],
        "assessment": report["assessment"],
        "components": report["components"],
        "hard_skills": report["hard_skills"],
        "soft_skills": report["soft_skills"],
        "keywords": report["keywords"],
        "document_checks": report["document_checks"],
        "suggestions": report["suggestions"],
        "limitations": (
            "Transparent keyword and document heuristics only; not a real ATS system, "
            "trained model output, hiring recommendation, or prediction of job performance."
        ),
    }
