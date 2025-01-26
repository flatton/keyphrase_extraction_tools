import json
from collections.abc import Generator
from pathlib import Path

from ..utils.text_preprocessor import TextPreprocessor
from .data import EvaluationSample


class Dataloader:
    def __init__(self, dataset_json_path: Path, label_json_path: Path):
        with dataset_json_path.open(encoding="utf-8") as f:
            self.dataset = json.load(f)
        with label_json_path.open(encoding="utf-8") as f:
            self.label = json.load(f)

        self.preprocessor = TextPreprocessor(strongly_normalize=True)

    def _preprocess(self, keyphrase_list: list[list[str]]) -> list[list[str]]:
        return [
            list({self.preprocessor.run(text=phrase) for phrase in keyphrases})
            for keyphrases in keyphrase_list
        ]

    def __iter__(self) -> Generator[EvaluationSample, None, None]:
        common_keys = set(self.dataset.keys()) & set(self.label.keys())
        for key in common_keys:
            label_dict = {item["sample_id"]: item for item in self.label[key]}

            for text_item in self.dataset[key]:
                sample_id = text_item["sample_id"]
                if sample_id in label_dict:
                    text = text_item["text"]
                    label_item = label_dict[sample_id]
                    main_topic = [
                        _phrase.strip()
                        for _phrase in label_item["main_topic"]
                        if _phrase.strip()
                    ]
                    angle = [
                        _phrase.strip()
                        for _phrase in label_item["angle"]
                        if _phrase.strip()
                    ]
                    essential_terms = [
                        [_phrase.strip() for _phrase in terms if _phrase.strip()]
                        for terms in label_item["essential_terms"]
                    ]

                    keyphrase_list: list[list[str]] = []
                    if main_topic:
                        keyphrase_list.append(main_topic)
                    if angle:
                        keyphrase_list.append(angle)
                    if essential_terms:
                        keyphrase_list += [terms for terms in essential_terms if terms]

                    keyphrase_list = self._preprocess(keyphrase_list=keyphrase_list)
                    yield EvaluationSample(
                        dataset_name=key,
                        id=sample_id,
                        text=text,
                        keyphrase_list=keyphrase_list,
                    )
