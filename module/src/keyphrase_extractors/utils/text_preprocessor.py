import re
import unicodedata
from logging import Logger

import neologdn


class TextPreprocessor:
    """
    A class for preprocessing text with optional strong normalization. Handles tasks
    such as whitespace cleanup, Unicode normalization, and strong normalization
    (if enabled).

    Attributes:
        strongly_normalize (bool): If True, applies additional strong normalization
                                   by converting text to lowercase and removing spaces.
        logger (Logger | None): Optional logger instance for logging processing steps.
    """

    def __init__(
        self,
        strongly_normalize: bool = False,
        logger: Logger | None = None,
    ):
        """
        Initializes the TextPreprocessor with optional strong normalization and logging.

        Args:
            strongly_normalize (bool): Whether to apply strong normalization.
            logger (Logger | None): Logger instance for logging or None for no logging.
        """
        self.strongly_normalize = strongly_normalize
        self.logger = logger

    def run(self, text: str) -> str:
        """
        Runs the text preprocessing pipeline.

        This method applies normalization and optionally strong normalization to the
        given text. Logs the process if a logger is provided.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        if self.logger:
            self.logger.info("Text preprocess")

        text = self._normalize(text)
        if self.strongly_normalize:
            text = self._strongly_normalize(text=text)
        return text

    def _normalize(self, text: str) -> str:
        """
        Normalizes the text by cleaning up whitespace and applying Unicode normalization.

        Replaces multiple spaces or newlines with single ones, converts full-width
        spaces to half-width, and applies neologdn normalization.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        text = re.sub(r"[\u3000\t]", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n+", "\n", text)
        text = unicodedata.normalize("NFKC", text)
        text = neologdn.normalize(text)
        return text

    def _strongly_normalize(self, text: str) -> str:
        """
        Applies strong normalization by converting text to lowercase and removing spaces.

        Args:
            text (str): The input text to strongly normalize.

        Returns:
            str: The strongly normalized text.
        """
        text = text.lower()
        text = re.sub(r"\s", "", text)
        return text
