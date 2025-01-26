from typing import Callable, Iterable, Mapping, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

class CountVectorizer:
    def __init__(
        self,
        *,
        input: str = "content",
        encoding: str = "utf-8",
        decode_error: str = "strict",
        strip_accents: Union[str, Callable[[str], str]] | None = None,
        lowercase: bool = True,
        preprocessor: Callable[[str], str] | None = None,
        tokenizer: Callable[[str], Iterable[str]] | None = None,
        stop_words: Union[str, list[str]] | None = None,
        token_pattern: str | None = r"(?u)\b\w\w+\b",
        ngram_range: tuple[int, int] = (1, 1),
        analyzer: Union[str, Callable[[str], Iterable[str]]] = "word",
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        max_features: int | None = None,
        vocabulary: Union[Mapping[str, int], Iterable[str]] | None = None,
        binary: bool = False,
        dtype: type = np.int64,
    ) -> None: ...
    def fit(
        self,
        raw_documents: Iterable[Union[str, bytes]],
        y: None = None,
    ) -> "CountVectorizer": ...
    def transform(
        self,
        raw_documents: Iterable[Union[str, bytes]],
    ) -> csr_matrix: ...
    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> NDArray[np.str_]: ...
