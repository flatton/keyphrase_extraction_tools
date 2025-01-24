from typing import Any, Literal

class LoadFile:
    def load_document(
        self,
        input: str,
        language: str | None = None,
        stoplist: list[str] | None = None,
        normalization: Literal["stemming", "none"] | None = None,
        spacy_model: Any | None = None,
    ) -> None: ...
    def candidate_filtering(
        self,
        minimum_length: int = 3,
        minimum_word_size: int = 2,
        valid_punctuation_marks: str = "-",
        maximum_word_number: int = 5,
        only_alphanum: bool = True,
        pos_blacklist: list[str] | None = None,
    ) -> None: ...
    def candidate_selection(self, **kwargs: Any) -> None: ...
    def candidate_weighting(self, **kwargs: Any) -> None: ...
    def get_n_best(
        self, top_n: int, redundancy_removal: bool = False, stemming: bool = False
    ) -> list[tuple[str, float]]: ...
