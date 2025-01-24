from logging import Logger
from typing import Any

from pke.base import LoadFile

from ..base_extractor import BaseExtractor
from ..io_data import Inputs, Keyphrase, Outputs
from ..utils import to_original_expression


class ClassicalExtractor(BaseExtractor):
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
        super().__init__(
            stop_words, max_characters, flat_output, use_order, rrf_k, logger
        )

        self.extractor = extractor
        self.args_candidate_selection = args_candidate_selection
        self.args_candidate_weighting = args_candidate_weighting
        self.stop_words = list(self.stop_words)

    def get_keyphrase(
        self, input_text: str | list[str] | Inputs, top_n_phrases: int = 10
    ) -> Outputs:
        verify_input: Inputs = self._verify_input(input_text=input_text)
        docs: list[str] = verify_input.docs

        outputs: Outputs = Outputs(keyphrases=[])
        for i in range(len(docs)):
            doc: str = docs[i]

            self.extractor.load_document(
                input=doc,
                language="ja",
                stoplist=self.stop_words,
                normalization=None,
            )
            self.extractor.candidate_filtering(pos_blacklist=self.stop_words)
            self.extractor.candidate_selection(**self.args_candidate_selection)
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

        if self.flat_output and len(outputs.keyphrases) > 1:
            outputs = self._flatten_outputs(
                keyphrases_list=outputs.keyphrases,
                use_order=self.use_order,
                descending=True,
                rrf_k=self.rrf_k,
            )
        return outputs
