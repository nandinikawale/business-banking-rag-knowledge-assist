from collections.abc import Sequence

from sklearn.feature_extraction.text import HashingVectorizer


class LocalHashingEmbeddings:
    """Deterministic, offline text embeddings suitable for a reproducible demo."""

    def __init__(self, dimensions: int = 384) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype("float32").tolist()
