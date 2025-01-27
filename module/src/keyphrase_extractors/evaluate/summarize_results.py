import json
from pathlib import Path

import pandas as pd


def get_evaluation_summary(
    evaluation_result_dirpath: Path,
    dataset_names: list[str],
    k_values: list[str],
    output_csv_filepath: Path = Path("./summary_for_paper.csv"),
) -> None:
    """
    Summarizes evaluation results from JSON files and saves them as a CSV file.

    This function processes evaluation summaries and processing time data for multiple
    datasets and configurations, formats key metrics, and saves the results in a
    CSV file.

    Args:
        evaluation_result_dirpath (Path): Path to the directory containing evaluation
            result subdirectories. Each subdirectory should include
            `evaluation_summary.json` and `process_time.json`.
        dataset_names (list[str]): Names of the datasets to include in the summary.
        k_values (list[str]): List of @k values to extract metrics for.
        output_csv_filepath (Path): Path to save the output CSV file. Defaults to
            `./summary_for_paper.csv`.

    Raises:
        ValueError: If `evaluation_result_dirpath` is not a valid directory.

    Returns:
        None: The summary is saved as a CSV file at the specified location.
    """
    if not evaluation_result_dirpath.is_dir():
        ValueError(f"{evaluation_result_dirpath=} is not directory path.")

    columns: list[str] = [
        "dataset",
        "@k",
        "Approach",
        "Precision",
        "Recall",
        "HitRate",
        "LCS_T",
        "LCS_P",
        "ProcessTime",
    ]

    csv_data: list[dict[str, str]] = []

    for contidion_dirpath in evaluation_result_dirpath.iterdir():
        if contidion_dirpath.is_dir():
            evaluation_summary_filepath = contidion_dirpath / "evaluation_summary.json"
            process_time_filepath = contidion_dirpath / "process_time.json"
            if (
                evaluation_summary_filepath.is_file()
                and process_time_filepath.is_file()
            ):
                with evaluation_summary_filepath.open(encoding="utf-8") as f:
                    eval_summary = json.load(f)
                with process_time_filepath.open(encoding="utf-8") as f:
                    process_time = json.load(f)

                for dataset_name in dataset_names:
                    if dataset_name in eval_summary and dataset_name in process_time:
                        for k in k_values:
                            if k in eval_summary[dataset_name]:
                                precision_mean_std = "{:.2f} ± {:.2f}".format(
                                    eval_summary[dataset_name][k]["precision"]["mean"],
                                    eval_summary[dataset_name][k]["precision"]["std"],
                                )
                                recall_mean_std = "{:.2f} ± {:.2f}".format(
                                    eval_summary[dataset_name][k]["recall"]["mean"],
                                    eval_summary[dataset_name][k]["recall"]["std"],
                                )
                                hitrate_mean_std = "{:.2f} ± {:.2f}".format(
                                    eval_summary[dataset_name][k]["hitrate"]["mean"],
                                    eval_summary[dataset_name][k]["hitrate"]["std"],
                                )
                                lcs_t_mean_std = "{:.2f} ± {:.2f}".format(
                                    eval_summary[dataset_name][k]["lcs_by_truthset"][
                                        "mean"
                                    ],
                                    eval_summary[dataset_name][k]["lcs_by_truthset"][
                                        "std"
                                    ],
                                )
                                lcs_p_mean_std = "{:.2f} ± {:.2f}".format(
                                    eval_summary[dataset_name][k]["lcs_by_pred"][
                                        "mean"
                                    ],
                                    eval_summary[dataset_name][k]["lcs_by_pred"]["std"],
                                )
                                process_time_mean_std = "{:.2f} ± {:.2f}".format(
                                    process_time[dataset_name]["mean"],
                                    process_time[dataset_name]["std"],
                                )

                                csv_data.append(
                                    {
                                        "Approach": contidion_dirpath.name,
                                        "dataset": dataset_name,
                                        "@k": k,
                                        "Precision": precision_mean_std,
                                        "Recall": recall_mean_std,
                                        "HitRate": hitrate_mean_std,
                                        "LCS_T": lcs_t_mean_std,
                                        "LCS_P": lcs_p_mean_std,
                                        "ProcessTime": process_time_mean_std,
                                    }
                                )

    df = pd.DataFrame(csv_data, columns=columns)
    df = df.sort_values(by=["dataset", "@k", "Approach"])  # type: ignore
    df.to_csv(output_csv_filepath, index=False)
