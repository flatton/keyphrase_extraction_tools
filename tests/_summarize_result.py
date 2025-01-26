import re
from pathlib import Path

import pandas as pd
from keyphrase_extractors.evaluate.summarize_results import get_evaluation_summary


# from pathlib import Path
get_evaluation_summary(
    evaluation_result_dirpath=Path("../output/evaluation"),
    output_csv_filepath=Path("../output/evaluation/summary_for_paper.csv"),
    dataset_names=[
        "length_200",
        "length_2000",
        "length_20000",
    ],
    k_values=["@5", "@10", "@25"],
)
summary_df = pd.read_csv("../output/evaluation/summary_for_paper.csv", dtype=str)


# Function to convert dataframe into latex-table
def parse_mean_std(value_str: str):
    """
    '0.34 ± 0.18' などの文字列から (mean, std) = (0.34, 0.18) を返す。
    ± 記号がない場合も簡易処理で対応。
    """
    pattern = r"([\d.]+)\s*±\s*([\d.]+)"
    match = re.search(pattern, value_str)
    if match:
        mean_val = float(match.group(1))
        std_val = float(match.group(2))
        return mean_val, std_val
    else:
        # "0.34" のように ± が無い場合を簡易処理
        try:
            val = float(value_str)
            return val, 0.0
        except ValueError:
            return 0.0, 0.0


def format_mean_std_as_mathbf(
    mean_val: float, std_val: float, bold: bool = False
) -> str:
    """
    平均値と標準偏差を数値から生成し、"mean \pm std" を LaTeX で数式モード表記する。
    bold=True なら \mathbf{...} で太字にする。
    """
    # 小数点以下の桁数は用途にあわせて調整してください
    mean_str = f"{mean_val:.2f}"
    std_str = f"{std_val:.2f}"
    # "0.34 \pm 0.18"
    val_str = f"{mean_str} \\pm {std_str}"
    if bold:
        return rf"\mathbf{{{val_str}}}"
    else:
        return val_str


def create_latex_table(
    df: pd.DataFrame,
    approach_dict: dict[str, str],
    scores: list[str],
    dataset_names: list[str],
    k_values: list[str],
    table_caption: str = "Title",
    table_label: str = "tab:something",
) -> str:
    """
    df: 入力となる CSV を読み込んだ Pandas DataFrame
        必須の列: ['dataset', '@k', 'Approach'] + scores
        各スコア列は "0.34 ± 0.18" のような文字列が入っている想定。

    approach_dict:
        DataFrame 上の 'Approach' 値(キー) と、LaTeX表示用の名前(値)の対応を持つ辞書。
        例: {
             "embedding_001_1gram": "1-gram",
             "embedding_002_4gram": "4-gram"
            }

    scores: テーブルに残したいスコア名のリスト
             例: ['Precision', 'Recall', 'LCS_P', 'ProcessTime'] など
    dataset_names: テーブルに残したい dataset 名のリスト
             例: ['length_200', 'length_2000', 'length_20000']
    k_values: テーブルに残したい @k のリスト
             例: ['@5', '@10', '@25']

    table_caption: LaTeX の \caption{} に入れる文字列
    table_label: LaTeX の \label{} に入れる文字列

    返り値: 生成した LaTeX テーブルの文字列
    """

    # ----------------------
    # 1) 必要部分のフィルタリング
    # ----------------------
    # approach_dict.keys() だけ残す
    df_filtered = df[
        (df["Approach"].isin(approach_dict.keys()))
        & (df["dataset"].isin(dataset_names))
        & (df["@k"].isin(k_values))
    ].copy()

    # ----------------------
    # 2) スコアが「大きい方が良い」か「小さい方が良い」かを設定
    #    ここでは例として "ProcessTime" は小さい方が良い、それ以外は大きい方が良いとする
    # ----------------------
    score_direction = {}
    for sc in scores:
        score_direction[sc] = "up"  # デフォルトは上が良い
    # 特定スコアは "down" にする
    if "ProcessTime" in scores:
        score_direction["ProcessTime"] = "down"

    # ----------------------
    # 3) (dataset, @k, score)ごとにベストなアプローチを特定
    # ----------------------
    best_approach_map = {}  # {(ds, k_val, sc): set_of_best_approaches}

    for ds in dataset_names:
        for k_val in k_values:
            df_kds = df_filtered[
                (df_filtered["dataset"] == ds) & (df_filtered["@k"] == k_val)
            ]
            if df_kds.empty:
                continue

            for sc in scores:
                approach_to_mean = {}
                # 各アプローチの (mean, std) を取得
                for _, row in df_kds.iterrows():
                    val_str = row[sc]  # "0.34 ± 0.18" 等
                    mean_val, _ = parse_mean_std(val_str)
                    approach_to_mean[row["Approach"]] = mean_val

                if not approach_to_mean:
                    continue

                if score_direction[sc] == "up":
                    best_val = max(approach_to_mean.values())
                else:  # "down"
                    best_val = min(approach_to_mean.values())

                # ベストに等しいアプローチを全て取得 (同値があれば全て太字)
                best_set = {
                    a for a, m in approach_to_mean.items() if abs(m - best_val) < 1e-12
                }
                best_approach_map[(ds, k_val, sc)] = best_set

    # ----------------------
    # 4) テーブルの列見出しに矢印を付与
    # ----------------------
    def arrow_str(direction: str) -> str:
        return r"($\uparrow$)" if direction == "up" else r"($\downarrow$)"

    score_header_map = {}
    for sc in scores:
        score_header_map[sc] = f"{sc} {arrow_str(score_direction[sc])}"

    # ----------------------
    # 5) dataset の表示名置き換え (例: "length_200" -> "200文字")
    # ----------------------
    dataset_title_map = {
        "length_200": "200文字",
        "length_2000": "2,000文字",
        "length_20000": "20,000文字",
        # 必要に応じて追加
    }

    # ----------------------
    # 6) LaTeX tabular のカラム定義
    # ----------------------
    column_def = "l|l|" + "|".join(["c" * len(scores) for _ in dataset_names])

    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(f"    \\caption{{{table_caption}}}")
    latex_lines.append(f"    \\label{{{table_label}}}")
    latex_lines.append(f"    \\begin{{tabular}}{{{column_def}}}")
    latex_lines.append(r"    \hline")

    # ----------------------
    # 7) ヘッダ上段: dataset 見出し
    # ----------------------
    header_line_top = "         &  "
    for ds in dataset_names:
        ds_title = dataset_title_map.get(ds, ds)  # 無ければそのまま
        header_line_top += f" & \\multicolumn{{{len(scores)}}}{{c}}{{{ds_title}}}"
    latex_lines.append(header_line_top + r" \\")

    # ----------------------
    # 8) ヘッダ2段目: スコア名
    # ----------------------
    header_line_second = r"        @k & Approach"
    for ds in dataset_names:
        for sc in scores:
            header_line_second += f" & {score_header_map[sc]}"
    latex_lines.append(header_line_second + r" \\")
    latex_lines.append(r"    \hline\hline")

    # ----------------------
    # 9) テーブル本体
    #    approach_dict をそのまま列挙して表示順を固定
    # ----------------------
    #  注意: Python 3.7+ では dict も挿入順を保持しますが、
    #        場合によっては collections.OrderedDict を使って順序を明示しても OK です。
    for k_idx, k_val in enumerate(k_values):
        # (k_val の中で) すべての approach_key についてループ
        # approach_key が df に全く含まれなくても行は作るが、値は空欄になる場合がある
        row_span = len(approach_dict)

        approach_items = list(approach_dict.items())  # [(key, latex_display_name), ...]

        for i, (approach_key, approach_disp_name) in enumerate(approach_items):
            if i == 0:
                latex_line = rf"    \multirow{{{row_span}}}{{*}}{{{k_val}}}"
            else:
                latex_line = "     "

            # Approach の表示名を latex に出す
            latex_line += f" & {approach_disp_name}"

            # 各 dataset
            df_k = df_filtered[df_filtered["@k"] == k_val]
            for ds in dataset_names:
                df_row = df_k[
                    (df_k["dataset"] == ds) & (df_k["Approach"] == approach_key)
                ]
                if df_row.empty:
                    # 対応するデータが無い
                    for sc in scores:
                        latex_line += " & --"
                    continue

                row_data = df_row.iloc[0]
                # スコアを出力
                for sc in scores:
                    val_str_original = row_data[sc]  # "0.34 ± 0.18" 等
                    mean_val, std_val = parse_mean_std(val_str_original)

                    # ベスト判定
                    best_set = best_approach_map.get((ds, k_val, sc), set())
                    is_best = approach_key in best_set

                    formatted_str = format_mean_std_as_mathbf(
                        mean_val, std_val, bold=is_best
                    )
                    # 数式モード
                    latex_line += f" & ${{{formatted_str}}}$"

            latex_line += r" \\"
            latex_lines.append(latex_line)

        # ブロック終わり
        if k_idx < len(k_values) - 1:
            latex_lines.append(r"    \hline\hline")
        else:
            latex_lines.append(r"    \hline")

    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"\end{table*}")

    return "\n".join(latex_lines)


# Table 1.
print(
    create_latex_table(
        df=summary_df,
        scores=["Precision", "Recall"],
        approach_dict={
            "embedding_003_4gram_thres": "n-gram (1, 4)",
            "embedding_012_grammar_thres": "grammar",
            "embedding_025_grammar_thres_masked-distance": " + Mask",
            "embedding_027_grammar_thres_prompting": " + Prompt",
            "embedding_023_grammar_thres_filter-sentence": " + Filter",
            "embedding_029_grammar_thres_filter-sentence_min100": " + Filter (min=100)",
        },
        dataset_names=["length_200", "length_2000", "length_20000"],
        k_values=["@5", "@10", "@25"],
    )
)

# Table 2.
print(
    create_latex_table(
        df=summary_df,
        scores=["Precision", "Recall", "ProcessTime"],
        approach_dict={
            "TextRank": "TextRank",
            "SingleRank": "SingleRank",
            "TopicRank": "TopicRank",
            "MultipartiteRank": "MultipartiteRank",
            "TfIdf": "Tf-Idf",
            "KPMiner": "KP-Miner",
            "YAKE": "YAKE!",
            "GPT4o-2024-11-20": "GPT4o",
            "embedding_034_grammar_thres_filter-sentence_small_min100": "Ruri-small",
            "embedding_029_grammar_thres_filter-sentence_min100": "Ruri-base",
            "embedding_044_grammar_thres_filter-sentence_large_min100": "Ruri-large",
        },
        dataset_names=["length_200", "length_20000"],
        k_values=["@10"],
    )
)
