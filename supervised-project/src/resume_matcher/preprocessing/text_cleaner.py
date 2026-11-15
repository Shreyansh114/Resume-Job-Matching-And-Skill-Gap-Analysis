"""Conservative text normalization for resumes and job descriptions."""

import re
import unicodedata

import pandas as pd

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str | None, *, remove_contacts: bool = True) -> str:
    """Normalize Unicode and whitespace, optionally redact direct contact details."""
    if pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = _URL.sub(" ", normalized)
    if remove_contacts:
        normalized = _EMAIL.sub(" ", normalized)
        normalized = _PHONE.sub(" ", normalized)
    normalized = re.sub(r"[^\w+#./-]", " ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()
