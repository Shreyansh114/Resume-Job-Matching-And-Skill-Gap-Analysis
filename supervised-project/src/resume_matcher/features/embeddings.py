"""Text similarity backends for resume-job pairs."""

from functools import lru_cache

from resume_matcher.config import settings
from resume_matcher.features.similarity_metrics import cosine_similarity
from resume_matcher.preprocessing.text_cleaner import clean_text


@lru_cache(maxsize=2)
def _sentence_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence Transformers is optional. Install with: pip install -e '.[semantic]'"
        ) from exc
    return SentenceTransformer(model_name)


def pair_text_similarity(
    resume_text: str,
    job_text: str,
    *,
    backend: str | None = None,
    model_name: str | None = None,
) -> float:
    """Compute pair similarity with TF-IDF by default or optional Sentence Transformers."""
    selected_backend = backend or settings.embedding_backend
    resume = clean_text(resume_text)
    job = clean_text(job_text)
    if not resume or not job:
        return 0.0

    if selected_backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)
        try:
            matrix = vectorizer.fit_transform([resume, job])
        except ValueError as exc:
            if "empty vocabulary" in str(exc):
                return 0.0
            raise
        return float(matrix[0].multiply(matrix[1]).sum())
    if selected_backend == "sentence-transformers":
        model = _sentence_model(model_name or settings.embedding_model)
        embeddings = model.encode([resume, job], normalize_embeddings=True)
        return cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
    raise ValueError(f"Unsupported embedding backend: {selected_backend}")
