from logging import Logger

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from spacy.language import Language

from ..base_extractor import BaseExtractor
from ..io_data import Inputs, Keyphrase, Outputs
from .data import EmbeddingModel, SentenceEmbeddingBasedExtractionConfig
from .model import JapanesePhraseRankingModel


class SentenceEmbeddingBasedExtractor(BaseExtractor):
    """
    A keyphrase extractor using sentence embeddings for ranking phrases.

    This class leverages sentence embeddings and Japanese language processing to extract
    meaningful keyphrases from input text. It supports custom embedding models and
    configurations.

    Attributes:
        text_processor (Language): A spaCy language processor for Japanese text.
        extraction_config (SentenceEmbeddingBasedExtractionConfig): Configuration for
            keyphrase extraction, including distance measures and sorting.
        kw_model (JapanesePhraseRankingModel): A model for extracting and ranking
            keyphrases based on embeddings.
    """

    def __init__(
        self,
        model_config: EmbeddingModel,
        extraction_config: SentenceEmbeddingBasedExtractionConfig | None = None,
        max_characters: int | None = None,
        stop_words: set[str] | None = None,
        count_vectorizer: CountVectorizer | None = None,
        flat_output: bool = True,
        use_order: bool = False,
        rrf_k: int = 60,
        logger: Logger | None = None,
    ):
        """
        Initializes the SentenceEmbeddingBasedExtractor with an embedding model and
        optional configurations.

        Args:
            model_config (EmbeddingModel): Configuration for the embedding model,
                including model name, batch size, and device.
            extraction_config (SentenceEmbeddingBasedExtractionConfig | None): Optional
                configuration for extraction, such as masked distance or RRF sorting.
            max_characters (int | None): Maximum characters per chunk of text.
            stop_words (set[str] | None): Stop words to filter, or None for defaults.
            count_vectorizer (CountVectorizer | None): Optional CountVectorizer for
                phrase extraction.
            flat_output (bool): Whether to flatten the output structure.
            use_order (bool): Whether to consider order during ranking.
            rrf_k (int): Parameter for Reciprocal Rank Fusion (RRF) scoring.
            logger (Logger | None): Logger instance for logging or None for no logging.
        """
        super().__init__(
            stop_words, max_characters, flat_output, use_order, rrf_k, logger
        )
        # Initialize an embedding model
        model = SentenceTransformer(
            model_name_or_path=model_config.name,
            prompts=model_config.prompts.model_dump() if model_config.prompts else None,
            **model_config.model_dump(include={"device", "trust_remote_code"}),
        )
        if self.logger:
            self.logger.debug(f"Embedding model: {model_config.name}")
            self.logger.debug(
                f"Embedding prompt: {model_config.prompts.model_dump() if model_config.prompts else None}"
            )

        # Initialize an extractor
        self.text_processor: Language = spacy.load("ja_ginza")
        self.extraction_config = (
            extraction_config
            if extraction_config
            else SentenceEmbeddingBasedExtractionConfig()
        )
        self.kw_model = JapanesePhraseRankingModel(
            model=model,
            text_processor=self.text_processor,
            batchsize=model_config.batchsize,
            use_prompt=True if model_config.prompts else False,
            stop_words=self.stop_words,
            show_progress_bar=model_config.show_progress_bar,
            config=extraction_config
            if extraction_config
            else SentenceEmbeddingBasedExtractionConfig(),
            count_vectorizer=count_vectorizer,
            logger=self.logger,
        )
        if self.logger:
            self.logger.debug(
                f"extraction_config: {self.extraction_config.model_dump_json(indent=4)}"
            )

    def get_keyphrase(
        self, input_text: str | list[str] | Inputs, top_n_phrases: int = 10
    ) -> Outputs:
        """
        Extracts keyphrases from the input text using sentence embeddings for ranking.

        Args:
            input_text (str | list[str] | Inputs): The input text(s) or preprocessed data.
            top_n_phrases (int): The maximum number of keyphrases to return.

        Returns:
            Outputs: Extracted keyphrases with their corresponding scores.
        """
        verify_input: Inputs = self._verify_input(input_text=input_text)
        docs: list[str] = verify_input.docs

        if self.logger:
            self.logger.info("Run keyphrase extraction.")
            self.logger.debug(f"Inputs: {docs}")
        results_list: list[list[tuple[str, float]]] = self.kw_model.extract_keyphrases(
            docs=docs
        )
        if self.logger:
            self.logger.info("Completed keyphrase extraction.")
            self.logger.debug(f"Result: {results_list}")

        if self.logger:
            self.logger.info("Validate output")
        outputs: Outputs = Outputs(keyphrases=[])
        for results in results_list:
            _keyphrases = [Keyphrase(phrase=t[0], score=t[1]) for t in results]
            top_n_keyphrases = _keyphrases[:top_n_phrases]
            outputs.keyphrases.append(top_n_keyphrases)

        if self.flat_output and len(outputs.keyphrases) > 1:
            if self.logger:
                self.logger.info("Flatten outputs")
            outputs = self._flatten_outputs(
                keyphrases_list=outputs.keyphrases,
                use_order=self.use_order,
                descending=(
                    (not self.extraction_config.use_masked_distance)
                    or self.extraction_config.use_rrf_sorting
                ),
                rrf_k=self.rrf_k,
            )
        return outputs
