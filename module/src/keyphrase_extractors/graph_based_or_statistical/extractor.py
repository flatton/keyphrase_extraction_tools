from logging import Logger
from typing import Any

from pke.base import LoadFile

from ..base_extractor import BaseExtractor
from ..io_data import Inputs, Keyphrase, Outputs
from ..utils import to_original_expression


class ClassicalExtractor(BaseExtractor):
    """
    An extractor class for graph-based or statistical keyphrase extraction.

    This class extends BaseExtractor to use an external keyphrase extraction
    library (pke). It supports candidate selection, weighting, and filtering
    with customizable parameters.

    Attributes:
        extractor (LoadFile): An instance of the keyphrase extractor from the pke library.
        args_candidate_selection (dict[str, Any]): Parameters for candidate selection.
        args_candidate_weighting (dict[str, Any]): Parameters for candidate weighting.
        stop_words (list[str]): A list of stop words to exclude during processing.
    """

    def __init__(
        self,
        extractor: LoadFile,
        args_candidate_selection: dict[str, Any],
        args_candidate_weighting: dict[str, Any],
        stop_words: set[str] | None = None,
        max_characters: int | None = None,
        flat_output: bool = True,
        use_order: bool = False,
        rrf_k: int = 60,
        logger: Logger | None = None,
    ):
        """
        Initializes the ClassicalExtractor with configuration for candidate
        selection and weighting.

        Args:
            extractor (LoadFile): An extractor instance from the pke library.
            args_candidate_selection (dict[str, Any]): Parameters for candidate selection.
            args_candidate_weighting (dict[str, Any]): Parameters for candidate weighting.
            stop_words (set[str] | None): Stop words to filter, or None for default.
            max_characters (int | None): Maximum characters per chunk of text.
            flat_output (bool): Whether to flatten output structure.
            use_order (bool): Whether to consider keyphrase order during ranking.
            rrf_k (int): Parameter for Reciprocal Rank Fusion (RRF) scoring.
            logger (Logger | None): Logger instance or None for no logging.
        """
        super().__init__(
            stop_words, max_characters, flat_output, use_order, rrf_k, logger
        )

        self.extractor = extractor
        self.args_candidate_selection = args_candidate_selection
        self.args_candidate_weighting = args_candidate_weighting
        self.stop_words = list(self.stop_words)

        if self.logger:
            self.logger.debug(f"Model: {type(self.extractor).__name__}")

    def get_keyphrase(
        self, input_text: str | list[str] | Inputs, top_n_phrases: int = 10
    ) -> Outputs:
        """
        Extracts keyphrases from the input text using the configured extractor.

        Args:
            input_text (str | list[str] | Inputs): The input text(s) or preprocessed data.
            top_n_phrases (int): The maximum number of keyphrases to return.

        Returns:
            Outputs: Extracted keyphrases with their corresponding scores.
        """
        verify_input: Inputs = self._verify_input(input_text=input_text)
        docs: list[str] = verify_input.docs

        outputs: Outputs = Outputs(keyphrases=[])
        if self.logger:
            self.logger.info("Run keyphrase extraction.")
        for i in range(len(docs)):
            doc: str = docs[i]
            if self.logger:
                self.logger.debug(f"Extract keyphrases from: {doc}")

            self.extractor.load_document(
                input=doc,
                language="ja",
                stoplist=self.stop_words,
                normalization=None,
            )
            if self.logger:
                self.logger.debug("Candidate filtering")
            self.extractor.candidate_filtering(pos_blacklist=self.stop_words)
            if self.logger:
                self.logger.debug("Candidate Selection")
            self.extractor.candidate_selection(**self.args_candidate_selection)
            if self.logger:
                self.logger.debug("Candidate weighting")
            self.extractor.candidate_weighting(**self.args_candidate_weighting)

            # get top-k keyphrases
            results: list[tuple[str, float]] = self.extractor.get_n_best(top_n_phrases)

            _phrases = [
                to_original_expression(original_text=doc, phrase=t[0]) for t in results
            ]
            _scores = [t[1] for t in results]

            outputs.keyphrases.append(
                [
                    Keyphrase(phrase=_phrase, score=_score)
                    for _phrase, _score in zip(_phrases, _scores, strict=True)
                ]
            )
            if self.logger:
                self.logger.debug(f"Result: {outputs.keyphrases[-1]}")

        if self.flat_output and len(outputs.keyphrases) > 1:
            if self.logger:
                self.logger.info("Flatten outputs")
            outputs = self._flatten_outputs(
                keyphrases_list=outputs.keyphrases,
                use_order=self.use_order,
                descending=True,
                rrf_k=self.rrf_k,
            )
        return outputs
