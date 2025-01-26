from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, computed_field, confloat, model_validator


class EmbeddingPrompts(BaseModel):
    passage: str
    query: str


class EmbeddingModel(BaseModel):
    name: str
    prompts: EmbeddingPrompts | None = None
    device: str = Field(default="cpu", examples=["cpu", "mps", "cuda", "npu"])
    trust_remote_code: bool = False
    batchsize: int = 32
    show_progress_bar: bool = False


class SentenceEmbeddingBasedExtractionConfig(BaseModel):
    use_rrf_sorting: bool = True
    diversity_mode: Literal["normal", "use_maxsum", "use_mmr"] = "normal"

    max_filtered_phrases: int = 10
    max_filtered_sentences: int = 10

    threshold: Annotated[float, confloat(ge=0.0, le=1.0, strict=False)] | None = None
    nr_candidates: int = 20
    diversity: Annotated[float, confloat(ge=0.0, le=1.0)] = 0.7

    minimum_characters: int = 10
    filter_sentences: bool = True

    grammar_phrasing: bool = True
    grammar: str = """
    NBAR:
        {<NOUN|PROPN|ADJ>*<NOUN|PROPN>}

    NP:
        {<NBAR>}
        {<NBAR><ADP><NBAR>}
    """
    pos_filter: set[
        Literal[
            "NOUN",
            "PROPN",
            "VERB",
            "ADJ",
            "ADV",
            "INTJ",
            "PRON",
            "NUM",
            "AUX",
            "CONJ",
            "SCONJ",
            "DET",
            "ADP",
            "PART",
            "PUNCT",
            "SYM",
            "X",
        ]
    ] = {"NOUN", "PROPN", "ADJ", "NUM"}
    ngram_range: tuple[int, int] | None = None

    use_masked_distance: bool = False
    add_source_text: bool = False

    rrf_k: int = 60

    @model_validator(mode="before")
    @classmethod
    def check_pos_filter(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "pos_filter" in data and data["pos_filter"] is None:
            data["pos_filter"] = {
                "NOUN",
                "PROPN",
                "VERB",
                "ADJ",
                "ADV",
                "INTJ",
                "PRON",
                "NUM",
                "AUX",
                "CONJ",
                "SCONJ",
                "DET",
                "ADP",
                "PART",
                "PUNCT",
                "SYM",
                "X",
            }
        return data

    @computed_field
    @property
    def use_maxsum(self) -> bool:
        return self.diversity_mode == "use_maxsum"

    @computed_field
    @property
    def use_mmr(self) -> bool:
        return self.diversity_mode == "use_mmr"

    @model_validator(mode="after")
    def validate_nr_candidates(self) -> Self:
        if self.diversity_mode == "use_maxsum":
            if (
                self.nr_candidates < self.max_filtered_phrases
                or self.nr_candidates < self.max_filtered_sentences
            ):
                raise ValueError(
                    f"`nr_candidates` ({self.nr_candidates}) must be greater "
                    f"than or equal to both "
                    f"`max_filtered_phrases` ({self.max_filtered_phrases}) and "
                    f"`max_filtered_sentences` ({self.max_filtered_sentences})."
                )
            else:
                return self
        else:
            return self

    @model_validator(mode="after")
    def validate_prompting_mode(self) -> Self:
        if self.use_masked_distance and self.add_source_text:
            raise ValueError(
                f"Either `use_masked_distance` (={self.use_masked_distance=})"
                f"or `add_source_text` (={self.add_source_text=}) must be False"
            )
        else:
            return self

    @model_validator(mode="after")
    def validate_phrasing_mode(self) -> Self:
        if self.grammar_phrasing and self.ngram_range:
            raise ValueError(
                "Both `grammar_phrasing` and `ngram_range` cannot be enabled "
                f"at the same time.\n"
                f"Received: {self.grammar_phrasing=}, {self.ngram_range=}."
            )
        elif self.grammar_phrasing:
            self.ngram_range = None
            return self
        elif self.ngram_range:
            self.grammar_phrasing = False
            if self.ngram_range[0] > self.ngram_range[1]:
                raise ValueError(
                    f"'ngram_range' must have the first value less than or "
                    f"equal to the second value.\nReceived: {self.ngram_range=}"
                    f"."
                )
            return self
        else:
            self.ngram_range = (1, 1)
            return self
