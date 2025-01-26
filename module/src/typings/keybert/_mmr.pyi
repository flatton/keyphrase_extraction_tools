from typing import Any

from numpy.typing import NDArray

def mmr(
    doc_embedding: NDArray[Any],
    word_embeddings: NDArray[Any],
    words: list[str],
    top_n: int = 5,
    diversity: float = 0.8,
) -> list[tuple[str, float]]: ...
