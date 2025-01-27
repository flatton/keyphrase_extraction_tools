import json
import time
from logging import Logger
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..base_extractor import BaseExtractor
from ..io_data import Outputs
from .data import Score, Stats
from .dataloader import Dataloader
from .evaluator import Evaluator


class EvaluationPipeline:
    """
    A pipeline for evaluating keyphrase extraction models.

    This class handles dataset loading, prediction generation, and evaluation of
    keyphrase extraction models, logging results and metrics in JSON format.

    Attributes:
        dataloader (Dataloader): Handles dataset and label loading.
        evaluator (Evaluator): Computes evaluation metrics for predictions.
        top_n_phrases (int): Maximum number of keyphrases to predict.
        k_list (list[int]): List of @k values for evaluation metrics.
        output_dirpath (Path): Directory path for saving results and evaluation outputs.
    """

    def __init__(
        self,
        dataset_json_path: Path,
        label_json_path: Path,
        k_list: list[int],
        output_dirpath: Path,
        logger: Logger | None = None,
    ):
        """
        Initializes the EvaluationPipeline with dataset paths, evaluation parameters,
        and output directory.

        Args:
            dataset_json_path (Path): Path to the dataset JSON file.
            label_json_path (Path): Path to the label JSON file.
            k_list (list[int]): List of @k values for evaluation metrics.
            output_dirpath (Path): Path to the directory for saving evaluation results.
        """
        self.logger = logger
        self.dataloader = Dataloader(
            dataset_json_path=dataset_json_path, label_json_path=label_json_path
        )
        self.evaluator = Evaluator()
        self.top_n_phrases = max(k_list)
        self.k_list = k_list
        self.output_dirpath = output_dirpath

    def run(self, extractor: BaseExtractor, output_dirname: str = "model_name") -> None:
        """
        Runs the evaluation pipeline for a given keyphrase extraction model.

        This method generates predictions, computes metrics, and saves the results.

        Args:
            extractor (BaseExtractor): The keyphrase extraction model to evaluate.
            output_dirname (str): Directory name for saving the evaluation results.
                                  Defaults to "model_name".

        Returns:
            None: Results and metrics are saved as JSON files in the output directory.
        """
        _datasets: set[str] = set()
        sample_ids: dict[str, list[str | int]] = {}
        preds: dict[str, list[list[str]]] = {}
        labels: dict[str, list[list[list[str]]]] = {}
        process_time: dict[str, list[float]] = {}
        for eval_sample in tqdm(self.dataloader, desc="Evaluating..."):
            if eval_sample.dataset_name not in _datasets:
                _datasets.add(eval_sample.dataset_name)
                sample_ids[eval_sample.dataset_name] = []
                preds[eval_sample.dataset_name] = []
                labels[eval_sample.dataset_name] = []
                process_time[eval_sample.dataset_name] = []

            if self.logger:
                self.logger.info(f"Ipunt: {eval_sample.text}")
            start = time.perf_counter()
            _preds: Outputs = extractor.get_keyphrase(
                input_text=eval_sample.text, top_n_phrases=self.top_n_phrases
            )
            end = time.perf_counter()
            if self.logger:
                self.logger.info(f"Output: {_preds}")

            process_time[eval_sample.dataset_name].append((end - start) / 60)
            sample_ids[eval_sample.dataset_name].append(eval_sample.id)
            preds[eval_sample.dataset_name].append(
                [_keyphrase.phrase for _keyphrase in _preds.keyphrases[0]]
            )
            labels[eval_sample.dataset_name].append(eval_sample.keyphrase_list)

        # Backup outputs
        filepath = self.output_dirpath / output_dirname / "pred_keyphrases.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as file:
            json.dump(
                {
                    _dataset_name: {
                        "outputs": [
                            {"sample_id": id, "keyphrases": _keyphrases, "time": _time}
                            for id, _keyphrases, _time in zip(
                                sample_ids[_dataset_name],
                                preds[_dataset_name],
                                process_time[_dataset_name],
                                strict=True,
                            )
                        ]
                    }
                    for _dataset_name in sample_ids
                },
                file,
                ensure_ascii=False,
                indent=4,
            )

        # evaluate
        if self.logger:
            self.logger.info("Evaluation")
        results: dict[str, dict[str, list[Score]]] = {}
        stats: dict[str, dict[str, dict[str, Stats]]] = {}
        for _dataset_name in preds:
            results[_dataset_name], stats[_dataset_name] = self.evaluator.evaluate(
                pred_keyphrases_list=preds[_dataset_name],
                true_keyphrases_list=labels[_dataset_name],
                k_list=self.k_list,
            )

        process_time_stats = {
            _dataset_name: Stats(
                mean=np.mean(process_time[_dataset_name]),
                std=np.std(process_time[_dataset_name]),
                max=np.min(process_time[_dataset_name]),
                min=np.max(process_time[_dataset_name]),
            )
            for _dataset_name in process_time
        }

        # save results
        filepath = self.output_dirpath / output_dirname / "evaluation_all_results.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as file:
            json.dump(
                {
                    _dataset_name: {
                        k: [data.model_dump() for data in v]
                        for k, v in _results.items()
                    }
                    for _dataset_name, _results in results.items()
                },
                file,
                ensure_ascii=False,
                indent=4,
            )

        filepath = self.output_dirpath / output_dirname / "evaluation_summary.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as file:
            json.dump(
                {
                    _dataset_name: {
                        k: {s: v[s].model_dump() for s in v} for k, v in _stats.items()
                    }
                    for _dataset_name, _stats in stats.items()
                },
                file,
                ensure_ascii=False,
                indent=4,
            )

        filepath = self.output_dirpath / output_dirname / "process_time.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as file:
            json.dump(
                {
                    _dataset_name: _stats.model_dump()
                    for _dataset_name, _stats in process_time_stats.items()
                },
                file,
                ensure_ascii=False,
                indent=4,
            )
