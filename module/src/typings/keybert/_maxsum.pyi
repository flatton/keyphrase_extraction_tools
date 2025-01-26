from typing import Any

from numpy.typing import NDArray

def max_sum_distance(
    doc_embedding: NDArray[Any],
    word_embeddings: NDArray[Any],
    words: list[str],
    top_n: int,
    nr_candidates: int,
) -> list[tuple[str, float]]: ...
