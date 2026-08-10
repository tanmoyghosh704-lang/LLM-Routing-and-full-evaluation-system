"""Sentence-transformer embedding extraction for the router. CPU-only."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: list[str]):
    """Embed a list of query strings into an (n, 384) numpy array."""
    model = _get_model()
    return model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
