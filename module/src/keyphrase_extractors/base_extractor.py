import os
import re
from logging import Logger
from pathlib import Path

from .io_data import Inputs, Keyphrase, Outputs
from .utils import TextPreprocessor


PARENT_DIRPATH = Path(os.path.abspath(__file__)).parent


class BaseExtractor:
    """
    A base class for extracting keyphrases from input text with various preprocessing
    and scoring methods.

    Attributes:
        stop_words (set[str]): A set of stop words to filter out, loaded from a file or
                               provided directly.
        max_characters (int): Maximum number of characters per chunk when splitting text.
        preprocessor (TextPreprocessor): An instance of TextPreprocessor for text
                                         normalization.
        use_order (bool): Whether to consider the order of keyphrases when ranking.
        rrf_k (int): Parameter for Reciprocal Rank Fusion (RRF) scoring.
        flat_output (bool): Whether to flatten the output keyphrase structure.
        logger (Logger | None): Optional logger instance for logging operations.
    """

    def __init__(
        self,
        stop_words: set[str] | Path | None = None,
        max_characters: int = 30,
        flat_output: bool = True,
        use_order: bool = False,
        rrf_k: int = 60,
        logger: Logger | None = None,
    ):
        """
        Initializes the BaseExtractor with optional parameters for text processing
        and logging.

        Args:
            stop_words (set[str] | Path | None): Stop words to filter, provided as a set
                                                 or file path.
            max_characters (int): Maximum number of characters per chunk of text.
            flat_output (bool): Whether to flatten the output keyphrase structure.
            use_order (bool): Whether to consider the order of keyphrases when ranking.
            rrf_k (int): Parameter for Reciprocal Rank Fusion (RRF) scoring.
            logger (Logger | None): Logger instance for logging or None for no logging.
        """
        # Japanese Stopwors
        if stop_words:
            if isinstance(stop_words, Path):
                self.stop_words = self._get_stopword_list(
                    stop_words_filepath=stop_words
                )
            else:
                self.stop_words = stop_words
        else:
            self.stop_words = self._get_stopword_list()
        self.max_characters = max_characters
        self.preprocessor = TextPreprocessor()
        self.use_order = use_order
        self.rrf_k = rrf_k
        self.flat_output = flat_output
        self.logger = logger

    def _get_stopword_list(
        self, stop_words_filepath: Path = PARENT_DIRPATH / "stop_words.txt"
    ) -> set[str]:
        """
        Loads a list of stop words from a file.

        Args:
            stop_words_filepath (Path): Path to the stop words file.

        Returns:
            set[str]: A set of stop words loaded from the file.
        """
        if self.logger:
            self.logger.info("Load stop-words")
        with stop_words_filepath.open(encoding="utf-8") as file:
            stop_words = {line.strip() for line in file}
        stop_words.remove("")
        return stop_words

    def _chunk(self, text: str, max_characters: int) -> list[str]:
        """
        Splits text into chunks that fit within the maximum character count.

        Args:
            text (str): The input text to be split.
            max_characters (int): Maximum number of characters per chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        if self.logger:
            self.logger.info("Splits text into chunks")
        sentence_split_marks: str = r"\. |\? |! |。|！|？|\n"
        phrase_split_marks: str = r", |、 |\s"

        chunked_texts: list[str] = []

        while len(text) > max_characters:
            split_position = None
            for match in re.finditer(sentence_split_marks, text[:max_characters]):
                split_position = match

            if split_position:
                end = split_position.end()
            else:
                space_position = None
                for match in re.finditer(phrase_split_marks, text[:max_characters]):
                    space_position = match

                if space_position:
                    end = space_position.end()
                else:
                    end = self.max_characters - 1

            chunked_texts.append(text[: end + 1])
            text = text[end + 1 :].lstrip()

        chunked_texts.append(text)

        return chunked_texts

    def _verify_input(self, input_text: str | list[str] | Inputs) -> Inputs:
        """
        Verifies and preprocesses the input text, ensuring it conforms to the expected
        format.

        Args:
            input_text (str | list[str] | Inputs): The input text to verify.

        Returns:
            Inputs: Verified and preprocessed input data.
        """
        if self.logger:
            self.logger.info("Verify the input")
        verified_input: Inputs
        if isinstance(input_text, str):
            if self.max_characters:
                input_text = self._chunk(
                    text=input_text,
                    max_characters=self.max_characters,
                )
                verified_input = Inputs(docs=input_text)
            else:
                verified_input = Inputs(docs=[input_text])
        elif isinstance(input_text, list):
            verified_input = Inputs(docs=input_text)
        else:
            verified_input = input_text
        verified_input.docs = [
            self.preprocessor.run(text=_text) for _text in verified_input.docs
        ]
        return verified_input

    def _score_sorting(
        self, keyphrases_list: list[list[Keyphrase]], descending: bool
    ) -> list[Keyphrase]:
        """
        Sorts keyphrases by their scores.

        Args:
            keyphrases_list (list[list[Keyphrase]]): A list of keyphrase groups to sort.
            descending (bool): Whether to sort in descending order.

        Returns:
            list[Keyphrase]: A sorted list of keyphrases.
        """
        if self.logger:
            self.logger.info("Sort all keyphrases by their scores")
        flatten_outputs: dict[str, float] = {}
        for group in keyphrases_list:
            for _keyphrase in group:
                if _keyphrase.phrase in flatten_outputs:
                    current_score = flatten_outputs[_keyphrase.phrase]
                    if descending:
                        flatten_outputs[_keyphrase.phrase] = max(
                            current_score, _keyphrase.score
                        )
                    else:
                        flatten_outputs[_keyphrase.phrase] = min(
                            current_score, _keyphrase.score
                        )
                else:
                    flatten_outputs[_keyphrase.phrase] = _keyphrase.score

        return sorted(
            (
                Keyphrase(phrase=phrase, score=score)
                for phrase, score in flatten_outputs.items()
            ),
            key=lambda x: x.score,
            reverse=descending,
        )

    def _reciprocal_rank_fusion(
        self, keyphrases_list: list[list[Keyphrase]], rrf_k: int
    ) -> list[Keyphrase]:
        """
        Ranks keyphrases using Reciprocal Rank Fusion (RRF)
        based on their importance scores and their source document orders.


        Args:
            keyphrases_list (list[list[Keyphrase]]): A list of keyphrase groups to rank.
            rrf_k (int): The RRF parameter controlling the scoring weight.

        Returns:
            list[Keyphrase]: A ranked list of keyphrases.
        """
        if self.logger:
            self.logger.info("Sort all keyphrases by their scores and their order")
        flatten_outputs: dict[str, float] = {}
        for i, group in enumerate(keyphrases_list, start=1):
            for j, _keyphrase in enumerate(group, start=1):
                rrf_score = 1.0 / (i + rrf_k) + 1.0 / (j + rrf_k)

                if _keyphrase.phrase in flatten_outputs:
                    current_score = flatten_outputs[_keyphrase.phrase]
                    flatten_outputs[_keyphrase.phrase] = max(current_score, rrf_score)
                else:
                    flatten_outputs[_keyphrase.phrase] = rrf_score

        return sorted(
            (
                Keyphrase(phrase=phrase, score=score)
                for phrase, score in flatten_outputs.items()
            ),
            key=lambda x: x.score,
            reverse=True,
        )

    def _flatten_outputs(
        self,
        keyphrases_list: list[list[Keyphrase]],
        use_order: bool = False,
        descending: bool = True,
        rrf_k: int = 60,
    ) -> Outputs:
        """
        Flattens keyphrase outputs using scoring or order-based methods.

        Args:
            keyphrases_list (list[list[Keyphrase]]): A list of keyphrase groups.
            use_order (bool): Whether to consider order when ranking keyphrases.
            descending (bool): Whether to sort in descending order.
            rrf_k (int): The RRF parameter for scoring.

        Returns:
            Outputs: Flattened keyphrase outputs.
        """
        if self.logger:
            self.logger.info("Flatten the output keyphrases")
        if use_order:
            flatten = self._reciprocal_rank_fusion(
                keyphrases_list=keyphrases_list, rrf_k=rrf_k
            )
        else:
            flatten = self._score_sorting(
                keyphrases_list=keyphrases_list, descending=descending
            )
        return Outputs(keyphrases=[flatten])

    def get_keyphrase(
        self, input_text: str | list[str] | Inputs, top_n_phrases: int = 10
    ) -> Outputs:
        """
        Extracts keyphrases from the input text.

        Args:
            input_text (str | list[str] | Inputs): The input text or preprocessed data.
            top_n_phrases (int): The maximum number of keyphrases to extract.

        Returns:
            Outputs: Extracted keyphrase outputs.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        # input_text: Inputs = self._verify_input(input_text=input_text)
        # docs: list[str] = input_text.docs

        # Implement the process of keyphrase extraction here.
        raise NotImplementedError("This method is not implemented.")
