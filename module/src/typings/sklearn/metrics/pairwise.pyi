from typing import Any, Optional

from numpy.typing import NDArray

def cosine_distances(
    X: NDArray[Any],
    Y: Optional[NDArray[Any]] = ...,
) -> NDArray[Any]: ...
def cosine_similarity(
    X: NDArray[Any],
    Y: Optional[NDArray[Any]] = ...,
    dense_output: bool = ...,
) -> NDArray[Any]: ...
