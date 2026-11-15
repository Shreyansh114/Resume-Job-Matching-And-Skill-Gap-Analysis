"""Extract text from supported resume documents."""

from pathlib import Path

from resume_matcher.config import settings


def extract_resume_text(file_name: str, content: bytes) -> str:
    """Extract text from TXT, PDF, or DOCX bytes; raise actionable errors on failure."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise ValueError("Unsupported resume format. Upload a TXT, PDF, or DOCX file.")
    if len(content) > settings.max_text_chars * 10:
        raise ValueError("The uploaded document is too large to process.")

    if suffix == ".txt":
        text = content.decode("utf-8-sig", errors="replace")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support is optional. Install with: pip install -e '.[documents]'"
            ) from exc
        import io

        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
    else:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "DOCX support is optional. Install with: pip install -e '.[documents]'"
            ) from exc
        import io

        document = Document(io.BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    text = text[: settings.max_text_chars]
    if not text.strip():
        raise ValueError("No readable text was found in the uploaded resume.")
    return text
